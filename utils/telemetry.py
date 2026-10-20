"""Training-loop telemetry: parameter and gradient statistics.

This complements the per-iteration solver telemetry the models emit through the loss head (see
``models/common.py``): that covers what the fixed-point iteration *does* (residual, step size,
patience, gslerp branch, tangent log-norm), this covers the parameters that *control* it and how
hard the optimizer is pushing on them.

The interesting ones are the learned step-size-like scalars, which are otherwise invisible:

* ``param/eigen_lr/*``  - SFPRM's per-dimension nGPT eigen learning rates ``alpha_A`` / ``alpha_M``.
  They set how far each retraction moves the token state toward the sub-layer output, so they are
  the on-sphere analogue of a residual gate. Shrinking eigen LRs mean the block is turning itself
  into (close to) the identity, which trivially makes the loop converge but stops it computing.
* ``param/s_z``, ``param/s_qk/*`` - nGPT's learned logit and QK scales.
* ``param/lambda_inject`` - SFPRM's gslerp input-injection fraction, when
  ``arch.lambda_inject_learnable`` is set (logged post-sigmoid, i.e. as used). How much of the
  input is mixed back in each iteration: larger values pin the iterate closer to the input and
  keep an x-independent attractor from being a fixed point; near-zero means the loop is drifting
  free of its input. Absent when lambda is a fixed constant.
* ``param/alpha_{1,2}`` - FPRM's residual-scale gates (logged post-sigmoid, i.e. as used).

Every function here syncs the GPU (``.item()``), so the train loop calls them on rank 0 on a
stride rather than every step.
"""

from typing import Dict

import torch
from torch import nn


def _clean(name: str) -> str:
    """Strip the torch.compile / loss-head / inner-module wrapper prefixes from a param name."""
    for prefix in ("_orig_mod.", "model.", "inner."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def _layer_tag(name: str) -> str:
    """``L_level.layers.3.alpha_A`` -> ``l3``; falls back to the bare name when there is no index."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return f"l{parts[i + 1]}"
    return "l"


@torch.no_grad()
def grad_norm(model: nn.Module) -> float:
    """Global L2 norm of the gradients, as seen by the optimizer (post-allreduce, pre-step)."""
    total = None
    for p in model.parameters():
        if p.grad is None:
            continue
        sq = p.grad.detach().float().pow(2).sum()
        total = sq if total is None else total + sq
    if total is None:
        return 0.0
    return float(total.sqrt().item())


@torch.no_grad()
def param_metrics(model: nn.Module) -> Dict[str, float]:
    """Learned step-size-like scalars, plus the global parameter norm.

    Returns a flat ``{wandb_key: float}`` dict; which keys appear depends on the architecture
    (SFPRM has eigen LRs and nGPT scales, FPRM has residual-scale gates).
    """
    out: Dict[str, float] = {}
    total_sq = None

    for raw_name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        sq = p.detach().float().pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq

        name = _clean(raw_name)
        v = p.detach().float()

        # SFPRM: per-dimension eigen learning rates (one pair per sub-block, tied across iterations).
        if name.endswith("alpha_A") or name.endswith("alpha_M"):
            head = "A" if name.endswith("alpha_A") else "M"
            tag = f"param/eigen_lr/{_layer_tag(name)}_{head}"
            out[f"{tag}_mean"] = float(v.mean().item())
            out[f"{tag}_min"] = float(v.min().item())
            out[f"{tag}_max"] = float(v.max().item())
            out[f"{tag}_absmean"] = float(v.abs().mean().item())

        # SFPRM: nGPT learned scales.
        elif name.endswith("s_qk"):
            out[f"param/s_qk/{_layer_tag(name)}_mean"] = float(v.mean().item())
        elif name.endswith("s_z"):
            out["param/s_z"] = float(v.mean().item())

        # SFPRM: learned gslerp input-injection fraction, logged as used (post-sigmoid), the
        # direct analogue of FPRM's alpha gates below. Scalar, so one value (no min/max).
        # Only present when arch.lambda_inject_learnable is set; otherwise lambda is a plain
        # Python float and never reaches this loop.
        elif name.endswith("lambda_inject_param"):
            out["param/lambda_inject"] = float(torch.sigmoid(v).item())

        # FPRM: residual-scale gates, logged as used (post-sigmoid) for the parametrized variants.
        elif name.endswith("alpha_1_param") or name.endswith("alpha_2_param"):
            which = "alpha_1" if name.endswith("alpha_1_param") else "alpha_2"
            gate = torch.sigmoid(v) if v.numel() > 1 else v
            out[f"param/{which}_mean"] = float(gate.mean().item())
            out[f"param/{which}_min"] = float(gate.min().item())
            out[f"param/{which}_max"] = float(gate.max().item())

    if total_sq is not None:
        out["param/global_norm"] = float(total_sq.sqrt().item())
    return out
