"""Spherical fixed-point solver for SFPRM (Spherical Fixed-Point Reasoner).

This is the on-sphere analogue of ``FixedPointOptimizer`` (``model_utils.py``). It carries the
per-sample solver state for a weight-tied block whose token states live on the unit sphere
``S^{d-1}`` and advances them with a damped *geodesic* step. It implements the six pieces needed to
make the looped spherical iteration actually run:

  (#9)  guarded SLERP step + chordal residual    -> :func:`gslerp`, :func:`chordal_residual`
  (#8)  all control state under ``no_grad``       -> only the ``gslerp`` step in :meth:`step` carries grad
  (#10) per-sample halting                        -> :meth:`halted` returns a per-sample mask
  (#11) worst-token (``max_t``) residual          -> :func:`chordal_residual`
  (#7)  hard iteration cap + sampled depth        -> per-sample ``max_iter`` sampled in :meth:`reset`
  (#12) best-iterate readout                      -> ``y_best`` tracked; :meth:`get_output`
  (#13) train/eval step-size-decay split          -> decay selected by ``self.training``

The module deliberately keeps FPRM's interface (``reset`` / ``step`` / ``cont`` / ``detach_state``)
so it can drop into a model in place of ``FixedPointOptimizer`` with minimal changes. Config is read
with ``getattr(..., default)`` so it works both with ``FPRMConfig`` and a plain namespace (tests).
"""

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import fp_step_stats


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project each token vector onto the unit sphere along the last dim."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def chordal_residual(z: torch.Tensor, z_tilde: torch.Tensor):
    """Smooth fixed-point residual on the sphere (#9, #11).

    Per token, ``rho_t = 1 - <z_t, z~_t> = 0.5 * ||z_t - z~_t||^2 in [0, 2]`` (assuming unit norm).
    Unlike the geodesic angle ``arccos(<z, z~>)`` this has a *constant* derivative and no
    cancellation near convergence. The per-sample residual is the **worst token** (``max_t``), the
    spherical analogue of FPRM's worst-token relative-inf residual.

    Returns ``(r, rho)`` with ``r`` shape ``(B,)`` and ``rho`` shape ``(B, L)``.
    """
    rho = 1.0 - (z * z_tilde).sum(dim=-1)        # (B, L)
    r = rho.max(dim=-1)[0]                        # (B,)
    return r, rho


def gslerp(z: torch.Tensor, q: torch.Tensor, eta, eps: float = 1e-6) -> torch.Tensor:
    """Guarded spherical-linear interpolation (#9), token-wise.

    Moves ``z`` a fraction ``eta`` along the great circle toward ``q``. The plain SLERP formula
    divides by ``sin(Omega)`` and is ``0/0`` as ``Omega -> 0`` (the convergence/halting regime) and
    undefined at ``Omega -> pi`` (antipodes). We fall back to normalized linear interpolation
    (``nlerp``) whenever ``|<z, q>| > 1 - eps``; ``nlerp`` agrees with SLERP to ``O(Omega^2)`` and is
    well-conditioned in both value and gradient.

    Both ``z`` and ``q`` must be unit-norm along the last dim; ``eta`` is a scalar or a tensor
    broadcastable to ``(..., 1)`` and is treated as a constant (no grad flows into it).
    """
    dot = (z * q).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)   # (..., 1)
    guard = dot.abs() > (1.0 - eps)

    # nlerp branch: finite everywhere (used where guarded). At a (near-)antipodal midpoint the
    # interpolant underflows to ~0 and its direction is undefined; fall back to z (a valid unit
    # point on the non-unique great circle) so the step always returns a unit vector (Prop. 1).
    lin = (1.0 - eta) * z + eta * q
    lin = torch.where(lin.norm(dim=-1, keepdim=True) < eps, z, lin)
    nl = l2_normalize(lin, eps)

    # slerp branch: feed acos a *neutral* value where guarded so its unbounded derivative at +/-1
    # is never multiplied by a zero `where`-gradient (the classic torch.where NaN-grad trap).
    dot_safe = torch.where(guard, torch.zeros_like(dot), dot)
    omega = torch.acos(dot_safe)
    sin_o = torch.sin(omega)
    sin_o = torch.where(guard, torch.ones_like(sin_o), sin_o)  # avoid div-by-zero in unused branch
    coef_z = torch.sin((1.0 - eta) * omega) / sin_o
    coef_q = torch.sin(eta * omega) / sin_o
    sl = coef_z * z + coef_q * q

    return torch.where(guard, nl, sl)


@torch.no_grad()
def gslerp_stats(rho: torch.Tensor, eta: torch.Tensor, slerp_eps: float) -> Dict[str, torch.Tensor]:
    """Geometry of one ``gslerp`` step, derived from the chordal residual ``rho = 1 - <z, q>``.

    ``gslerp`` takes its ``nlerp`` branch exactly where ``|<z, q>| > 1 - slerp_eps``, i.e. where
    ``rho < slerp_eps`` (already converged, the common case near a fixed point) or
    ``rho > 2 - slerp_eps`` (antipodal). ``nlerp_frac`` is the fraction of token slots taking that
    branch, so it says *when* the guard actually fires rather than assuming it never does.
    ``angle`` is the geodesic angle ``Omega = arccos(<z, q>)`` the block wants to travel and
    ``step_angle = eta * Omega`` is what the damped step actually travels.
    """
    dot = (1.0 - rho.float()).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    guard = dot.abs() > (1.0 - slerp_eps)
    eta = eta.reshape(-1, 1).float()
    return {
        "sph/nlerp_frac":   guard.float().mean(),
        "sph/nlerp_frac_converged": (dot > (1.0 - slerp_eps)).float().mean(),
        "sph/nlerp_frac_antipodal": (dot < -(1.0 - slerp_eps)).float().mean(),
        "sph/angle":        omega.mean(),
        "sph/angle_max":    omega.max(),
        "sph/step_angle":   (eta * omega).mean(),
    }


def retract(z: torch.Tensor, sublayer_out: torch.Tensor, alpha, eps: float = 1e-12) -> torch.Tensor:
    """nGPT eigen-LR retraction: move ``z`` toward ``Norm(sublayer_out)`` by the per-dim eigen
    learning rate ``alpha``, then renormalize. ``z`` is unit-norm per token; ``alpha`` broadcasts
    over (batch, seq, hidden)."""
    target = l2_normalize(sublayer_out, eps)
    return l2_normalize(z + alpha * (target - z), eps)


# ---------------------------------------------------------------------------
# Stability regularizer (paper Sec. 3.5)
# ---------------------------------------------------------------------------
def spectral_logabs_penalty(f: Callable[[torch.Tensor], torch.Tensor], z: torch.Tensor,
                            n_samples: int = 1, eps: float = 1e-3, delta: float = 0.05,
                            xi: float = 0.0,
                            stats: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """Tangent-space logarithmic-norm regularizer that backs Prop. 2's ``Re(lambda_i) < 1``.

    Penalizes the symmetric Rayleigh quotient ``rho(v) = v^T J v`` of the tangent differential
    ``J = P_z f'(z) P_z`` exceeding a margin ``1 - delta`` (so it drives ``mu_2(J) = lambda_max((J +
    J^T)/2) < 1 - delta``, hence ``Re(lambda_i) < 1`` by Coppel's inequality). Unlike a Frobenius
    penalty it leaves the skew/rotational part free. ``J v`` is a central-difference JVP through the
    block ``f`` along a random *tangent* direction; the estimate is differentiable w.r.t. ``f``'s
    parameters. The optional ``xi`` term adds the Frobenius penalty to cap rotation magnitude so the
    convergent step size ``eta_0`` stays usable.

    ``stats``, when given, is filled with the *raw* Rayleigh quotient statistics behind the hinge.
    They are the direct empirical read on Prop. 2: ``mu2_mean``/``mu2_max`` estimate the tangent-space
    logarithmic norm ``mu_2(J)`` (values >= 1 mean the iteration is locally expansive and the loop is
    not a contraction), ``active_frac`` is the fraction of token slots the hinge is actually pushing
    on, and ``jvp_norm = ||J v||`` is the local gain along a random unit tangent direction.
    """
    z = z.detach()
    total = z.new_zeros(())
    mu2_sum = z.new_zeros(())
    mu2_max = z.new_full((), -float("inf"))
    active_sum = z.new_zeros(())
    jvp_sum = z.new_zeros(())
    for _ in range(n_samples):
        v = torch.randn_like(z)
        v = v - (v * z).sum(dim=-1, keepdim=True) * z        # project onto T_z (v ⟂ z per token)
        v = l2_normalize(v, 1e-12).detach()
        zp = l2_normalize(z + eps * v, 1e-12)
        zm = l2_normalize(z - eps * v, 1e-12)
        Jv = (f(zp) - f(zm)) / (2.0 * eps)                   # central-difference JVP
        Jv = Jv - (Jv * z).sum(dim=-1, keepdim=True) * z     # project output to T_z
        rho = (v * Jv).sum(dim=-1)                            # = v^T J_sym v (real v)
        total = total + F.relu(rho - (1.0 - delta)).pow(2).mean() / n_samples
        if xi > 0.0:
            total = total + xi * Jv.pow(2).sum(dim=-1).mean() / n_samples
        if stats is not None:
            with torch.no_grad():
                r = rho.detach().float()
                mu2_sum = mu2_sum + r.mean() / n_samples
                mu2_max = torch.maximum(mu2_max, r.max())
                active_sum = active_sum + (r > (1.0 - delta)).float().mean() / n_samples
                jvp_sum = jvp_sum + Jv.detach().float().norm(dim=-1).mean() / n_samples
    if stats is not None:
        stats["sph/mu2_mean"] = mu2_sum
        stats["sph/mu2_max"] = mu2_max
        stats["sph/mu2_active_frac"] = active_sum
        stats["sph/jvp_norm"] = jvp_sum
    return total


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class SphericalFixedPointOptimizer(nn.Module):
    """Per-sample damped geodesic fixed-point solver on ``(S^{d-1})^L``."""

    def __init__(self, config) -> None:
        super().__init__()
        g = lambda name, default: getattr(config, name, default)

        self.stepsize = float(g("stepsize", 1.0))                       # initial eta
        self.stepsize_decay_train = float(g("stepsize_decay_train", 0.9))   # (#13)
        self.stepsize_decay_eval = float(g("stepsize_decay_eval", 0.99))    # (#13)
        self.decay_patience = int(g("decay_patience", 5))
        self.eps = float(g("eps", 1e-8))
        self.outlier_quantile = float(g("outlier_quantile", 0.25))
        self.fp_thresh = float(g("fp_thresh", 0.1))                     # chordal tolerance tau_c
        self.eta_min = float(g("eta_min", 1e-3))                        # step-size floor (#7 termination)
        self.slerp_eps = float(g("slerp_eps", 1e-6))                    # gslerp guard threshold (#9)
        self.additive_noise_std = float(g("additive_noise_std", 0.0))

        # Sampled-depth iteration cap (#7).
        self.max_iter = int(g("max_iter", 16))
        self.max_iter_eval = g("max_iter_eval", None)
        self.max_iter_dist = str(g("max_iter_dist", "gamma"))
        self.gamma_alpha = float(g("gamma_alpha", 4.0))
        self.gamma_scale = float(g("gamma_scale", 50.0 / 3.0))
        self.expon_scale = float(g("expon_scale", 50.0 / math.log(2.0)))



    # -- helpers -----------------------------------------------------------
    def _decay(self) -> float:
        return self.stepsize_decay_train if self.training else self.stepsize_decay_eval

    def _sample_max_iter(self, batch_size: int, device) -> torch.Tensor:
        """Per-sample iteration budget (#7). Training: sampled; eval: deterministic cap."""
        if not self.training:
            cap = self.max_iter_eval if self.max_iter_eval is not None else self.max_iter
            return torch.full((batch_size,), int(cap), dtype=torch.int32, device=device)

        if self.max_iter_dist == "det":
            s = torch.full((batch_size,), float(self.max_iter), device=device)
        elif self.max_iter_dist == "gamma":
            d = torch.distributions.Gamma(self.gamma_alpha, 1.0 / self.gamma_scale)
            s = d.sample((batch_size,)).to(device)
        elif self.max_iter_dist == "expon":
            d = torch.distributions.Exponential(1.0 / self.expon_scale)
            s = d.sample((batch_size,)).to(device)
        else:
            raise ValueError(f"Unknown max_iter_dist: {self.max_iter_dist!r}")
        return s.clamp_min(1.0).to(torch.int32)

    def detach_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.detach()
        return state

    def get_output(self, state: Dict[str, torch.Tensor], best: bool = False) -> torch.Tensor:
        """Read out the best (lowest-residual) iterate (#12) or the last one."""
        return state["y_best"] if best else state["y"]

    # -- lifecycle ---------------------------------------------------------
    def reset(self, reset_flag: torch.Tensor, shape, dtype, device,
              state: Optional[Dict[str, torch.Tensor]], init_y: Optional[torch.Tensor] = None,
              reset_metadata: bool = False) -> Dict[str, torch.Tensor]:
        """(Re)initialize per-sample solver state for halted samples (vectorized ``where``)."""
        batch_size, seq_len, hidden_size = shape
        rf1 = reset_flag.view(-1)
        rf3 = rf1.view(-1, 1, 1)

        if init_y is not None:
            y = l2_normalize(init_y.to(device=device, dtype=dtype), self.eps).contiguous()
        else:
            y = l2_normalize(torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device),
                             self.eps).contiguous()

        residues = torch.full((batch_size,), float("inf"), device=device)
        stepsize = self.stepsize * torch.ones(batch_size, 1, 1, dtype=dtype, device=device)
        patience = float(self.decay_patience) * torch.ones(batch_size, device=device)
        iter_idx = torch.zeros(batch_size, dtype=torch.int32, device=device)
        best_residues = torch.full((batch_size,), float("inf"), device=device)
        y_best = y.detach().clone()       # readout snapshot; never part of the grad graph
        max_iter = self._sample_max_iter(batch_size, device)

        if state is None:
            return dict(y=y, y_best=y_best, residues=residues, stepsize=stepsize,
                        patience=patience, iter_idx=iter_idx, best_residues=best_residues,
                        max_iter=max_iter)

        def pick1(new, old):  # per-sample fields
            return new if reset_metadata else torch.where(rf1, new, old)

        return dict(
            y=torch.where(rf3, y, state["y"]),
            y_best=torch.where(rf3, y_best, state["y_best"]),
            residues=pick1(residues, state["residues"]),
            stepsize=stepsize if reset_metadata else torch.where(rf3, stepsize, state["stepsize"]),
            patience=pick1(patience, state["patience"]),
            iter_idx=pick1(iter_idx, state["iter_idx"]),
            best_residues=pick1(best_residues, state["best_residues"]),
            max_iter=pick1(max_iter, state["max_iter"]),
        )

    # -- iteration ---------------------------------------------------------
    def step(self, state: Dict[str, torch.Tensor], z_new: torch.Tensor,
             stats: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """One damped geodesic update.

        ``z_new = f_theta(z; x)`` is the (unit-norm) block output. The residual and every control
        update (patience, step-size decay, best-iterate) are computed under ``no_grad`` on detached
        operands (#8); only the final ``gslerp`` step carries gradient.

        ``stats``, when given, is filled in-place with this iteration's solver + geometry telemetry
        (residual, contraction, step size, decay events, geodesic angle, gslerp guard fraction).
        """
        z_old = state["y"]
        if z_new.dtype != z_old.dtype:
            z_new = z_new.to(z_old.dtype)

        z_new = l2_normalize(z_new, self.eps)

        # ---- control state: NON-differentiable (#8) ----
        with torch.no_grad():
            r, rho = chordal_residual(z_old.detach(), z_new.detach())  # worst-token per sample (#11)
            prev_r = state["residues"]
            improved = r < state["best_residues"]

            # best-iterate tracking (#12): the iterate z_old has residual r.
            state["y_best"] = torch.where(improved.view(-1, 1, 1), z_old.detach(), state["y_best"])
            state["best_residues"] = torch.where(improved, r, state["best_residues"])
            state["residues"] = r

            # patience: reset on improvement, else count down.
            state["patience"] = torch.where(improved,
                                            float(self.decay_patience) * torch.ones_like(state["patience"]),
                                            state["patience"] - 1.0)

            # decay eta when patience expires and we are still above tolerance (#13 train/eval split).
            adapt = (state["patience"] <= 0) & (state["residues"] >= self.fp_thresh)
            state["patience"] = torch.where(adapt,
                                            float(self.decay_patience) * torch.ones_like(state["patience"]),
                                            state["patience"])
            factor = torch.where(adapt,
                                 self._decay() * torch.ones_like(state["patience"]),
                                 torch.ones_like(state["patience"]))
            state["stepsize"] = state["stepsize"] * factor.view(-1, 1, 1).to(state["stepsize"].dtype)
            state["iter_idx"] = state["iter_idx"] + 1

            if stats is not None:
                stats.update(fp_step_stats(r, prev_r, improved, adapt,
                                           state["stepsize"], self.fp_thresh))
                stats.update(gslerp_stats(rho, state["stepsize"], self.slerp_eps))

        # ---- the only differentiable operation ----
        eta = state["stepsize"]                           # detached constant (built under no_grad)
        z = gslerp(z_old, z_new, eta, self.slerp_eps)
        if self.additive_noise_std > 0.0:
            z = l2_normalize(z + self.additive_noise_std * torch.randn_like(z), self.eps)
        state["y"] = z
        return state

    # -- halting -----------------------------------------------------------
    def halted(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample halting mask (#10): converged, step-size floored, or hit the sampled cap (#7)."""
        eta = state["stepsize"].reshape(-1)
        return ((state["residues"] < self.fp_thresh)
                | (eta < self.eta_min)
                | (state["iter_idx"] >= state["max_iter"]))

    def cont(self, state: Dict[str, torch.Tensor]) -> bool:
        """Batch-level continue predicate with outlier-robust aggregation (train only)."""
        if int(state["iter_idx"].max().item()) == 0:
            return self.max_iter > 0
        q = 1.0 - self.outlier_quantile if self.training else 1.0
        res_q = torch.quantile(state["residues"].float(), q)
        iter_q = torch.quantile(state["iter_idx"].float(), q)
        cap_q = torch.quantile(state["max_iter"].float(), q)
        eta_q = torch.quantile(state["stepsize"].float().reshape(-1), q)
        return bool((res_q >= self.fp_thresh) & (iter_q < cap_q) & (eta_q > self.eta_min))
