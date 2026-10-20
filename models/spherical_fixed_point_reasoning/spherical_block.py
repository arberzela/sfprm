"""The weight-tied spherical block f_theta for SFPRM (paper Sec. 3.1).

One application of ``SphericalFixedPointTransformer`` re-injects the input on the sphere and then
applies ``L_layers`` nGPT-style sub-blocks, each a pair of eigen-LR retractions (attention, then
MLP). The whole block is reused across loop iterations (tied), so its eigen learning rates are
shared across iterations; the ``L_layers`` sub-blocks within one application are distinct (each has
its own ``alpha_A``, ``alpha_M``), as in nGPT.
"""

import math

import torch
import torch.nn as nn

from ..layers import Attention, SwiGLU, rms_norm
from .spherical import gslerp, l2_normalize, retract


class SphericalReasoningBlock(nn.Module):
    """One nGPT sub-block: attention retraction followed by MLP retraction (states stay on the sphere)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        weight_dropout = getattr(config, "weight_dropout", 0.0)

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=config.causal,
            softmax_temp=config.softmax_temp,
            weight_dropout=weight_dropout,
            qk_norm=getattr(config, "qk_norm", True),
            s_qk_init=getattr(config, "s_qk_init", 1.0),
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
            weight_dropout=weight_dropout,
        )

        # Per-dimension eigen learning rates (shared across loop iterations via weight tying).
        a0 = float(getattr(config, "eigen_lr_init", 0.5))
        self.alpha_A = nn.Parameter(a0 * torch.ones(config.hidden_size))
        self.alpha_M = nn.Parameter(a0 * torch.ones(config.hidden_size))
        # Eigen LRs are step-size-like; keep them out of weight decay (as nGPT / FPRM do for alphas).
        self.alpha_A._no_weight_decay = True
        self.alpha_M._no_weight_decay = True

        self.eps = float(getattr(config, "eps", 1e-8))
        self.norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))

    def forward(self, z: torch.Tensor, cos_sin) -> torch.Tensor:
        attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=z)
        z = retract(z, attn_out, self.alpha_A.to(z.dtype), self.eps)
        mlp_out = self.mlp(z)
        z = retract(z, mlp_out, self.alpha_M.to(z.dtype), self.eps)
        return z


class SphericalFixedPointTransformer(nn.Module):
    """The tied spherical block f_theta(z; x): on-sphere input injection + L_layers sub-blocks."""

    def __init__(self, config, n_layers: int) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SphericalReasoningBlock(config) for _ in range(n_layers)])
        self.slerp_eps = float(getattr(config, "slerp_eps", 1e-6))
        self.eps = float(getattr(config, "eps", 1e-8))

        # Input-injection fraction for the on-sphere gslerp. Either a fixed constant (original
        # behaviour) or a learned parameter, mirroring FPRM's residual-mix alphas: store a LOGIT
        # and squash with sigmoid so the value stays in (0, 1) without a constraint/clamp, and
        # exclude it from weight decay (it is a mixing rate, not a weight).
        #
        # Scalar, not per-dim: gslerp moves along a great circle and takes eta broadcastable to
        # (..., 1). A per-dimension coefficient would apply a different rotation angle per
        # coordinate, which is not a geodesic step and would not return a unit vector -- breaking
        # the invariant every downstream op (and Prop. 1) relies on. This is the one place the
        # FPRM analogy has to stop.
        lam0 = float(getattr(config, "lambda_inject", 0.1))
        self.lambda_inject_learnable = bool(getattr(config, "lambda_inject_learnable", False))
        if self.lambda_inject_learnable:
            if not (0.0 < lam0 < 1.0):
                raise ValueError(
                    f"lambda_inject must be in (0, 1) when lambda_inject_learnable=True "
                    f"(it is the sigmoid init); got {lam0}"
                )
            self.lambda_inject_param = nn.Parameter(torch.tensor(math.log(lam0 / (1.0 - lam0))))
            self.lambda_inject_param._no_weight_decay = True
        else:
            self.lambda_inject = lam0

        # Optional depthwise conv (as in FPRM, "necessary for FP stability") that mixes adjacent
        # positions each iteration to help propagate the sequential composition.
        #
        # conv_type mirrors models/transformer.py so an SFPRM run can be given the SAME conv as
        # its FPRM counterpart:
        #   conv1d  depthwise CAUSAL 1D over the whole sequence (SFPRM's original behaviour, and
        #           still the default when use_conv is set without a conv_type)
        #   conv2d  depthwise NON-causal 2D over the sqrt(grid_len) x sqrt(grid_len) grid, with
        #           the puzzle-embedding prefix split off and passed through untouched -- the
        #           setting FPRM uses on Sudoku (conv_type=conv2d, k=3, no bias)
        # Both are followed by l2_normalize in forward(): the conv is a plain linear map and does
        # not preserve unit norm, but every downstream op assumes the state is on the sphere.
        self.use_conv = bool(getattr(config, "use_conv", False))
        self.conv_type = str(getattr(config, "conv_type", "conv1d")) if self.use_conv else "none"
        if self.use_conv:
            k = int(getattr(config, "conv_kernel_size", 4))
            bias = bool(getattr(config, "conv_bias", False))
            if self.conv_type == "conv1d":
                self.conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=k,
                                      padding=k - 1, groups=config.hidden_size, bias=bias)
            elif self.conv_type == "conv2d":
                if k % 2 == 0:
                    raise ValueError(f"conv_kernel_size must be odd for conv2d (got {k})")
                self.conv = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=k,
                                      padding=k // 2, groups=config.hidden_size, bias=bias)
            else:
                raise ValueError(f"Unknown conv_type {self.conv_type!r}; expected 'conv1d' or 'conv2d'")

    def _apply_conv(self, z: torch.Tensor, puzzle_emb_len: int) -> torch.Tensor:
        if self.conv_type == "conv1d":
            L = z.shape[1]
            return self.conv(z.transpose(1, 2)).transpose(1, 2)[:, :L]  # causal (left-padded)
        # conv2d: the grid is spatial, so the puzzle-embedding prefix is NOT part of it and is
        # carried through unchanged -- exactly as models/transformer.py does for FPRM.
        B, T, H = z.shape
        puzzle_part, grid_part = z[:, :puzzle_emb_len], z[:, puzzle_emb_len:]
        grid_len = T - puzzle_emb_len
        hw = int(math.sqrt(grid_len))
        assert hw * hw == grid_len, f"grid_len {grid_len} is not a perfect square"
        grid_part = grid_part.reshape(B, hw, hw, H).permute(0, 3, 1, 2).contiguous()
        grid_part = self.conv(grid_part).permute(0, 2, 3, 1).reshape(B, grid_len, H).contiguous()
        return torch.cat([puzzle_part, grid_part], dim=1).contiguous()

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor,
                cos_sin=None, puzzle_emb_len: int = 0, **kwargs) -> torch.Tensor:
        x_hat = l2_normalize(input_injection, self.eps)
        # on-sphere input re-injection (lambda>0 makes an x-independent attractor non-fixed)
        if self.lambda_inject_learnable:
            lam = torch.sigmoid(self.lambda_inject_param).to(hidden_states.dtype)
        else:
            lam = self.lambda_inject
        z = gslerp(hidden_states, x_hat, lam, self.slerp_eps)
        if self.use_conv:
            z = l2_normalize(self._apply_conv(z, puzzle_emb_len), self.eps)
        for layer in self.layers:
            z = layer(z, cos_sin)
        return z
