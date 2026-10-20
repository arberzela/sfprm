"""Spherical Fixed-Point Reasoner (SFPRM)."""

from dataclasses import dataclass
from typing import Dict

import math
import torch
import torch.nn.functional as F
from torch import nn

from ..common import accumulate_stats, fp_state_stats, halting_stats
from ..layers import RotaryEmbedding, CastedEmbedding, CastedLinear
from ..sparse_embedding import CastedSparseEmbedding
from .sfprm_config import SFPRMConfig
from .spherical import (
    SphericalFixedPointOptimizer,
    l2_normalize,
    spectral_logabs_penalty,
)
from .spherical_block import SphericalFixedPointTransformer

IGNORE_LABEL_ID = -100


@dataclass
class SFPRMInnerCarry:
    z_L_state: dict
    dropout_mask: torch.Tensor


@dataclass
class SFPRMCarry:
    inner_carry: SFPRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class SphericalFixedPointReasoningModel_Inner(nn.Module):
    def __init__(self, config: SFPRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O. On the sphere the embedding is normalized per token rather than scaled.
        embed_init_std = 1.0 / math.sqrt(self.config.hidden_size)
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size,
                                            init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Learned logit scale s_z (nGPT).
        self.s_z = nn.Parameter(torch.tensor(float(self.config.sz_init)))
        self.s_z._no_weight_decay = True

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        )
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size,
                                             init_std=embed_init_std, cast_to=self.forward_dtype)

        # The tied spherical block and the damped-geodesic solver.
        self.L_level = SphericalFixedPointTransformer(self.config, self.config.L_layers)
        self.L_optimizer = SphericalFixedPointOptimizer(self.config)

        # Q-head special init: start biased toward "not halt" (kept for metric compatibility).
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

        # Project all sphere-facing weights onto the unit sphere at init (nGPT).
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self):
        """nGPT weight normalization: unit-normalize the sphere-facing weight matrices.

        Token embedding and lm_head rows are normalized along the hidden dim (each token / output
        vector is unit). Block linears are normalized along their input dim (each output neuron reads
        a unit-norm combination). The eigen LRs, s_z, s_qk and the (detached, metric-only) q-head are
        left untouched. Because every sub-layer output is renormalized inside the retraction, this
        mainly keeps weight magnitudes bounded during training rather than altering the forward.
        """
        if not getattr(self.config, "normalize_weights", True):
            return
        eps = self.config.eps

        def _unit(w, dim):
            w.div_(w.norm(dim=dim, keepdim=True).clamp_min(eps))

        _unit(self.embed_tokens.embedding_weight.data, dim=-1)
        if getattr(self.config, "normalize_lm_head", True):
            _unit(self.lm_head.weight.data, dim=-1)
        if self.config.pos_encodings == "learned":
            _unit(self.embed_pos.embedding_weight.data, dim=-1)
        for blk in self.L_level.layers:
            for lin in (blk.self_attn.qkv_proj, blk.self_attn.o_proj,
                        blk.mlp.gate_up_proj, blk.mlp.down_proj):
                if hasattr(lin, "weight"):
                    _unit(lin.weight.data, dim=1)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            embedding = embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)

        # Project onto the sphere per token (nGPT input on S^{d-1}).
        return l2_normalize(embedding, self.config.eps)

    def empty_carry(self, batch_size: int) -> SFPRMInnerCarry:
        return SFPRMInnerCarry(z_L_state=None, dropout_mask=None)

    def reset_carry(self, reset_flag: torch.Tensor, batch: torch.Tensor,
                    carry: SFPRMInnerCarry) -> SFPRMInnerCarry:
        shape = (batch.shape[0], batch.shape[1] + self.puzzle_emb_len, self.config.hidden_size)
        device = batch.device
        dtype = self.forward_dtype

        if self.training and self.config.variational_dropout > 0.0:
            keep = 1.0 - self.config.variational_dropout
            dropout_mask = torch.empty(*shape, device=device, dtype=dtype).bernoulli_(p=keep).div_(keep)
            if carry.dropout_mask is not None:
                dropout_mask = torch.where(reset_flag.view(-1, 1, 1), dropout_mask, carry.dropout_mask)
        else:
            dropout_mask = torch.ones(*shape, device=device, dtype=dtype)

        return SFPRMInnerCarry(
            z_L_state=self.L_optimizer.reset(reset_flag, shape, dtype, device, carry.z_L_state),
            dropout_mask=dropout_mask,
        )

    @torch._dynamo.disable
    def _spectral_reg(self, state: Dict[str, torch.Tensor], input_embeddings: torch.Tensor,
                      dropout_mask: torch.Tensor, seq_info: Dict[str, any]):
        """Returns ``(penalty, stats)``; ``stats`` holds the raw mu_2(J) diagnostics (empty if off)."""
        if not self.training or self.config.n_spectral_samples == 0:
            return state["y"].new_zeros(()), {}
        stats = {}
        f = lambda z: dropout_mask * self.L_level(z, input_embeddings, **seq_info)
        penalty = spectral_logabs_penalty(
            f, state["y"], n_samples=self.config.n_spectral_samples,
            eps=self.config.spectral_eps, delta=self.config.spectral_delta,
            xi=self.config.spectral_xi, stats=stats)
        return penalty, stats

    def _z_step(self, state: Dict[str, torch.Tensor], input_embeddings: torch.Tensor,
                dropout_mask: torch.Tensor, seq_info: Dict[str, any],
                stats: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        z_new = dropout_mask * self.L_level(state["y"], input_embeddings, **seq_info)
        return self.L_optimizer.step(state, z_new, stats=stats)

    @torch.no_grad()
    def _geometry_stats(self, state: Dict[str, torch.Tensor],
                        input_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """End-of-segment geometry of the iterate on ``(S^{d-1})^L``.

        ``token_concentration = ||mean_t z_t||`` is 1 when every token collapses to the same point on
        the sphere and ~0 when they are spread out: the cheapest collapse detector for a normalized
        state. ``inject_*`` describes the *other* ``gslerp`` call site, the on-sphere input
        re-injection in ``SphericalFixedPointTransformer``: how far the iterate sits from the input
        anchor, and how often that interpolation falls back to ``nlerp``.
        """
        y = state["y"].detach().float()
        x_hat = l2_normalize(input_embeddings.detach().float(), self.config.eps)
        dot = (y * x_hat).sum(dim=-1).clamp(-1.0, 1.0)
        return {
            "sph/token_concentration": y.mean(dim=1).norm(dim=-1).mean(),
            "sph/inject_angle": torch.acos(dot).mean(),
            "sph/inject_nlerp_frac": (dot.abs() > (1.0 - self.config.slerp_eps)).float().mean(),
            "fp/max_iter_sampled": state["max_iter"].float().mean(),
        }

    def forward(self, carry: SFPRMInnerCarry, batch: Dict[str, torch.Tensor],
                force_grad: bool, n_steps: int):
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos, sin = self.rotary_emb()
            s = input_embeddings.shape[1]
            cos_sin = (cos[:s], sin[:s])
        seq_info = dict(cos_sin=cos_sin, puzzle_emb_len=self.puzzle_emb_len)

        z_state, dropout_mask = carry.z_L_state, carry.dropout_mask
        # Solver telemetry: per-iteration stats averaged over the segment's inner iterations.
        telemetry = None
        with torch.set_grad_enabled(force_grad):
            for _ in range(n_steps):
                step_stats = {}
                z_state = self._z_step(z_state, input_embeddings, dropout_mask, seq_info,
                                       stats=step_stats)
                telemetry = accumulate_stats(telemetry, step_stats, n_steps)
        telemetry.update(fp_state_stats(z_state))
        telemetry.update(self._geometry_stats(z_state, input_embeddings))

        # Readout: deep supervision reads the last iterate at training; at inference, the
        # best-residual iterate (#12) unless eval_readout_best=False (then last, matching train).
        use_best = (not self.training) and getattr(self.config, "eval_readout_best", True)
        readout = self.L_optimizer.get_output(z_state, best=use_best)
        logits = (self.s_z * self.lm_head(readout))[:, self.puzzle_emb_len:]

        # Q-head detached from the trunk (metric-only; halting is fixed-point based).
        q_logits = self.q_head(z_state["y"][:, 0].detach()).to(torch.float32)

        spectral_loss, spectral_stats = self._spectral_reg(z_state, input_embeddings,
                                                           dropout_mask, seq_info)
        telemetry.update(spectral_stats)

        new_carry = SFPRMInnerCarry(z_L_state=self.L_optimizer.detach_state(z_state),
                                    dropout_mask=carry.dropout_mask)
        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1], spectral_loss, telemetry)


class SphericalFixedPointReasoningModel_ACTV1(nn.Module):
    """Single-state SFPRM wrapper (fixed-point halting)."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = SFPRMConfig(**config_dict)
        self.inner = SphericalFixedPointReasoningModel_Inner(self.config)
        self.max_iter = self.config.max_iter

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def normalize_weights(self):
        """nGPT weight normalization; called by the train loop after each optimizer step."""
        self.inner.normalize_weights()

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> SFPRMCarry:
        batch_size = batch["inputs"].shape[0]
        return SFPRMCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),       # default halted -> reset on 1st pass
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def set_num_iters(self):
        """Resample the per-batch segment cap (#7). Mirrors FPRM's set_num_iters."""
        if self.training:
            if self.config.max_iter_dist == "gamma":
                from scipy.stats import gamma            # lazy: only gamma/expon need scipy
                sampled = gamma.rvs(a=self.config.gamma_alpha, scale=self.config.gamma_scale)
            elif self.config.max_iter_dist == "expon":
                from scipy.stats import expon
                sampled = expon.rvs(scale=self.config.expon_scale)
            elif self.config.max_iter_dist == "det":
                sampled = self.config.max_iter
            else:
                raise ValueError(f"Unknown max_iter_dist {self.config.max_iter_dist!r}")
            self.max_iter = max(0, int(sampled)) if self.config.max_iter_dist == "det" else max(1, int(sampled))
        else:
            self.max_iter = self.config.max_iter_eval if self.config.max_iter_eval is not None else self.config.max_iter

    def forward(self, carry: SFPRMCarry, batch: Dict[str, torch.Tensor]):
        new_inner_carry = self.inner.reset_carry(carry.halted, batch["inputs"], carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        n_steps = self.config.n_backwards_L if self.training else 1
        new_inner_carry, logits, (q_halt_logits, q_continue_logits, spectral_loss, telemetry) = self.inner(
            new_inner_carry, new_current_data, force_grad=self.training, n_steps=n_steps)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "telemetry": telemetry,
        }
        if self.training and self.config.n_spectral_samples > 0:
            outputs["jacobian_loss"] = spectral_loss   # reuse loss head's jacobian_reg_lambda weight

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.max_iter           # per-sample segment cap (#7)
            halted = is_last_step

            residues = new_inner_carry.z_L_state["residues"]    # chordal, per sample (#11)
            eta = new_inner_carry.z_L_state["stepsize"].view(-1)

            if not self.training:
                # Inference: halt the whole sequence on the worst token (matches FPRM).
                halted = halted | (residues.max() < self.config.fp_thresh) | (eta.max() < self.config.eta_min)
            elif self.max_iter > 1:
                # Training: per-sample fixed-point halting (#10).
                halted = halted | (residues < self.config.fp_thresh) | (eta < self.config.eta_min)

            # Which termination criterion fired this segment (see models/common.py).
            telemetry.update(halting_stats(
                residues=residues, eta=eta, is_last_step=is_last_step, halted=halted,
                fp_thresh=self.config.fp_thresh, eta_min=self.config.eta_min,
                max_iter=self.max_iter))

        return SFPRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
