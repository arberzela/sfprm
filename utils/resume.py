"""Train-state resume bundle.

Saves model + optimizer state + step counter + dataloader epoch + EMA shadow +
wandb run id alongside the regular `step_<N>` model file as
`step_<N>_train_state.pt`. On restart, restoring the bundle continues training
without restarting LR warmup, biasing Adam moments, or starting a new wandb run.

RNG state is intentionally not restored — the dataset shuffle is keyed by
`seed + _iters` (which we do save), and `_iters` covers the only data-side
determinism that matters across restarts.
"""
import glob
import os
from typing import Any, Optional, Tuple

import torch


_BUNDLE_SUFFIX = "_train_state.pt"


def _resolve_bundle_path(resume_from: str) -> str:
    """Accepts a bundle file, the model file (sibling), or a directory."""
    if os.path.isdir(resume_from):
        cands = glob.glob(os.path.join(resume_from, f"step_*{_BUNDLE_SUFFIX}"))
        if not cands:
            raise FileNotFoundError(f"No resume bundle in {resume_from}")
        return max(cands, key=lambda p: int(os.path.basename(p).split("_")[1]))
    if resume_from.endswith(_BUNDLE_SUFFIX):
        return resume_from
    sibling = resume_from + _BUNDLE_SUFFIX
    if os.path.isfile(sibling):
        return sibling
    raise FileNotFoundError(f"Cannot resolve resume bundle from {resume_from}")


def save_resume_bundle(
    *,
    checkpoint_path: Optional[str],
    train_state: Any,
    ema_helper: Optional[Any],
    dataset_iters: int,
    iter_id: int,
    wandb_run_id: Optional[str],
) -> None:
    """`dataset_iters` must be the value the worker's `PuzzleDataset._iters`
    has reached. The main-process dataset copy never observes the worker's
    increments (DataLoader runs `_iter_train` in a forked worker), so the
    caller computes this deterministically from iter_id and the set count."""
    if checkpoint_path is None:
        return
    bundle = {
        "model": train_state.model.state_dict(),
        "step": train_state.step,
        "total_steps": train_state.total_steps,
        "iter_id": iter_id,
        "optimizers": [opt.state_dict() for opt in train_state.optimizers],
        "ema": ema_helper.state_dict() if ema_helper is not None else None,
        "dataset_iters": dataset_iters,
        "wandb_id": wandb_run_id,
    }
    os.makedirs(checkpoint_path, exist_ok=True)
    path = os.path.join(checkpoint_path, f"step_{train_state.step}{_BUNDLE_SUFFIX}")
    torch.save(bundle, path)


def restore_train_state(
    *,
    resume_from: str,
    train_state: Any,
    ema_helper: Optional[Any],
    dataset: Any,
) -> Tuple[int, Optional[str]]:
    """Restore in place. Returns (start_iter_id, wandb_id_to_resume)."""
    # Load onto the CPU, NOT the GPU. The training state (model, optimizers, EMA shadow) is
    # already allocated on the device by this point, so mapping the bundle straight to cuda
    # holds a second full copy of all of it at once. On ARC that is fatal: 876k puzzle
    # identifiers give a 449M-parameter embedding, so the bundle is several GB and the resume
    # OOMs on a 95 GiB card while the cold start fits comfortably. Each consumer below moves
    # only what it needs: Module.load_state_dict copies CPU->device tensor by tensor, and
    # Optimizer.load_state_dict casts state onto each param's device.
    bundle = torch.load(_resolve_bundle_path(resume_from), map_location="cpu")
    # Old checkpoints predate the weight_dropout feature, which added a `.mask`
    # buffer to every CastedLinear (initialised to ones, so a no-op when
    # dropout=0). Tolerate those missing keys; reject any other mismatch.
    missing, unexpected = train_state.model.load_state_dict(bundle["model"], strict=False)
    bad_missing = [k for k in missing if not k.endswith(".mask")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"resume state_dict mismatch: missing={bad_missing} unexpected={list(unexpected)}"
        )
    for opt, sd in zip(train_state.optimizers, bundle["optimizers"]):
        opt.load_state_dict(sd)
    train_state.step = bundle["step"]
    if hasattr(dataset, "_iters"):
        dataset._iters = bundle["dataset_iters"]
    if ema_helper is not None and bundle.get("ema") is not None:
        ema_helper.load_state_dict(bundle["ema"])
        # The shadow came off disk on the CPU (see map_location above), but EMAHelper.update()
        # does `shadow = (1-mu)*param + mu*shadow` against CUDA params and EMAHelper.ema() copies
        # it back into them, so it has to live on the params' device or the first update after a
        # resume raises a device mismatch. Moving it here keeps the CPU-load saving (the shadow
        # alone is far smaller than the full bundle) without changing EMA semantics.
        device = next(train_state.model.parameters()).device
        ema_helper.shadow = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in ema_helper.shadow.items()
        }
    if bundle["total_steps"] != train_state.total_steps:
        print(
            f"[resume] total_steps changed: saved={bundle['total_steps']} "
            f"new={train_state.total_steps}. Using new value; LR schedule "
            f"is rescaled over the new horizon."
        )
    return bundle["iter_id"] + 1, bundle.get("wandb_id")
