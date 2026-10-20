"""Mini-ARC evaluator: the ARC voting evaluator on the 5x5 canvas.

Pair it with data/miniarc-aug-100, built by dataset/miniarc_to_kaggle.py + dataset/build_miniarc_dataset.py.
"""

from typing import Dict, Optional

import numpy as np
import torch
from numba import njit

from dataset.build_arc_dataset import inverse_aug, grid_hash
from evaluators.arc import ARC


@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside (5x5 canvas)."""
    grid = grid.reshape(5, 5)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape

    num_c = nc
    for num_r in range(1, nr + 1):
        # Scan for maximum c
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break

        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


class MiniARC(ARC):
    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        # Identical to ARC.update_batch except the local 5x5 _crop.
        outputs = {}
        q_values = None

        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        q_values = v.to(torch.float64).sigmoid().cpu()
                    else:
                        outputs[k] = v.cpu()

        assert q_values is not None

        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        outputs = {k: v[mask] for k, v in outputs.items()}

        for identifier, input, pred, q in zip(outputs["puzzle_identifiers"].numpy(), outputs["inputs"].numpy(), outputs["preds"].numpy(), q_values.numpy()):
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)

            input_hash = grid_hash(_inverse_fn(_crop(input)))

            pred = _inverse_fn(_crop(pred))
            assert np.all((pred >= 0) & (pred <= 9)), f"Puzzle {name}'s prediction out of 0-9 range."  # Sanity check

            pred_hash = grid_hash(pred)

            self._local_hmap[pred_hash] = pred

            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, [])
            self._local_preds[orig_name][input_hash].append((pred_hash, float(q)))

    def result(self, save_path: Optional[str], rank: int, world_size: int, group=None) -> Optional[Dict[str, float]]:
        r = super().result(save_path, rank, world_size, group=group)
        if r is None:
            return r
        return {k.replace("eval/ARC/", "eval/MiniARC/"): v for k, v in r.items()}


class MiniARCSingle(MiniARC):
    """Per-checkpoint Mini-ARC vote: no pooling across in-run evals, logged as eval/MiniARC_single/*.

    The ARC/MiniARC evaluator defaults to aggregated_voting=True and pretrain.py builds it once per run,
    so the in-run eval/MiniARC/pass@K at the k-th eval votes over the predictions of evals 1..k. This
    class always votes over the current checkpoint only (its 100 augmentations).
    """

    def __init__(self, data_path: str, eval_metadata, **kwargs):
        kwargs["aggregated_voting"] = False
        super().__init__(data_path, eval_metadata, **kwargs)

    def result(self, save_path: Optional[str], rank: int, world_size: int, group=None) -> Optional[Dict[str, float]]:
        r = super().result(save_path, rank, world_size, group=group)
        if r is None:
            return r
        return {k.replace("eval/MiniARC/", "eval/MiniARC_single/"): v for k, v in r.items()}
