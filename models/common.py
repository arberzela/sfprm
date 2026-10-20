import math
from typing import Dict, Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Solver telemetry (shared by FPRM's FixedPointOptimizer and SFPRM's spherical solver)
#
# Every helper returns a dict of 0-dim tensors holding *batch* statistics for a single inner
# iteration. The model accumulates them over the segment with :func:`accumulate_stats` and hands
# them to the loss head, which rescales them so the train loop's `sum / count` reduction recovers
# the plain mean (see ``ACTLossHead.forward``). Everything runs under ``no_grad`` on detached
# operands, so instrumentation can never leak into the graph.
# ---------------------------------------------------------------------------
def accumulate_stats(acc: Optional[Dict[str, torch.Tensor]], new: Dict[str, torch.Tensor], n: int):
    """Running mean of a dict of scalar tensors over ``n`` contributions (keys must be static)."""
    if acc is None:
        return {k: v / n for k, v in new.items()}
    return {k: acc[k] + v / n for k, v in new.items()}


@torch.no_grad()
def fp_step_stats(r: torch.Tensor, prev_r: torch.Tensor, improved: torch.Tensor,
                  adapt: torch.Tensor, eta: torch.Tensor, fp_thresh: float) -> Dict[str, torch.Tensor]:
    """Per-iteration fixed-point telemetry: residual, contraction factor, step-size, decay events.

    ``r`` / ``prev_r`` are the per-sample residuals after / before this iteration, ``improved`` the
    best-residual improvement mask, ``adapt`` the mask of samples whose step size was decayed
    (patience expired), and ``eta`` the current per-sample step size. ``prev_r`` is ``inf`` on the
    first iteration after a reset, so the contraction factor averages only over samples where it is
    finite.
    """
    r = r.float()
    prev_r = prev_r.float()
    eta = eta.reshape(-1).float()

    finite = torch.isfinite(prev_r) & (prev_r > 0)
    safe_prev = torch.where(finite, prev_r, torch.ones_like(prev_r))
    contraction = torch.where(finite, r / safe_prev, torch.zeros_like(r)).sum() / finite.sum().clamp_min(1)

    return {
        "fp/residual":        r.mean(),
        "fp/residual_max":    r.max(),
        "fp/log10_residual":  torch.log10(r.clamp_min(1e-12)).mean(),
        "fp/contraction":     contraction,          # r_t / r_{t-1}; < 1 means the iteration converges
        "fp/improved_frac":   improved.float().mean(),
        "fp/decay_frac":      adapt.float().mean(),  # fraction whose patience expired -> eta decayed
        "fp/converged_frac":  (r < fp_thresh).float().mean(),
        "fp/eta":             eta.mean(),
        "fp/eta_batch_min":   eta.min(),
        "fp/log10_eta":       torch.log10(eta.clamp_min(1e-12)).mean(),
    }


@torch.no_grad()
def fp_state_stats(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """End-of-segment solver-state telemetry: patience left, iterations spent, best residual."""
    best = state["best_residues"].float()
    return {
        "fp/patience":         state["patience"].float().mean(),
        "fp/iters_in_segment": state["iter_idx"].float().mean(),
        "fp/best_residual":    torch.where(torch.isfinite(best), best, torch.zeros_like(best)).mean(),
    }


@torch.no_grad()
def halting_stats(residues: torch.Tensor, eta: torch.Tensor, is_last_step: torch.Tensor,
                  halted: torch.Tensor, fp_thresh: float, eta_min: float,
                  max_iter: int) -> Dict[str, torch.Tensor]:
    """Which termination criterion fires, per segment: tolerance / step-size floor / iteration cap."""
    eta = eta.reshape(-1).float()
    ref = residues.float()
    return {
        "halt/frac_residual": (ref < fp_thresh).float().mean(),
        "halt/frac_eta_floor": (eta < eta_min).float().mean(),
        "halt/frac_iter_cap": is_last_step.float().mean(),
        "halt/frac_halted":   halted.float().mean(),
        "halt/max_iter":      ref.new_tensor(float(max_iter)),   # sampled per-batch segment cap
    }


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor
