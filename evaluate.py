"""Evaluate a checkpoint on its test split with the training loop's own evaluation (pretrain.evaluate).

    python evaluate.py <checkpoint dir> [--batch B] [--max-examples N | --sample N] [key=value ...]
    torchrun --standalone --nproc-per-node=<n_gpus> evaluate.py <checkpoint dir> [...]

    # e.g. the SFPRM Maze checkpoint at its per-GPU training batch, then Mini-ARC at a longer rollout cap
    python evaluate.py checkpoints/sfprm-maze-step78120 --batch 96
    python evaluate.py checkpoints/sfprm-miniarc-step68620 arch.max_iter_eval=200

Model and data. The checkpoint directory's all_config.yaml (or --config <name> from config/, for
checkpoints trained elsewhere) builds the model and the test loader as pretrain.py does, with the
Hydra overrides applied (key=value changes a key, ++key=value adds or changes one). The step_<N>
weights (the latest, or --step N) are loaded strictly: a missing, unexpected or mis-shaped tensor is
an error. torch.compile is on unless DISABLE_COMPILE is set; weights saved either way load.
--renormalize-weights then projects the nGPT weight matrices of SFPRM back onto the unit sphere
(normalize_weights(), as the training loop does after every optimizer step): the stored EMA weights
average unit-norm rows, so they lie slightly inside the sphere. It has no effect on other models.

Evaluation. pretrain.evaluate() runs the model's halting loop on every batch and reduces the metrics
over GPUs. ARC and Mini-ARC configs also run their pass@K voting evaluator, fresh, so the vote is over
this checkpoint alone. Under torchrun each GPU takes a contiguous block of --batch examples per step,
so the batches are the same as on one GPU with that batch. FPRM and SFPRM halt a whole batch on its
worst example, so pass the per-GPU batch the run was trained with. The torch seed (--seed, plus the
rank) is set right before the evaluation: FPRM/SFPRM draw their initial iterate at random.

Subsets (test splits with one example per puzzle, i.e. Sudoku and Maze; the voting evaluators need
every augmentation): --max-examples N takes the first N examples, --sample N a seeded random N.
--shard I --num-shards K evaluates the I-th contiguous slice of the selection, so a long evaluation
can be split into independent jobs; --merge then combines their --out JSON files.
"""
import argparse
import contextlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.abspath(__file__))


def log(msg: str):
    print(f"[evaluate] {msg}", flush=True)


def load_config(ckpt_dir: str, config_name, overrides):
    if config_name:
        config_dir, name = os.path.join(REPO, "config"), config_name
    else:
        config_dir, name = ckpt_dir, "all_config"
        if not os.path.isfile(os.path.join(ckpt_dir, "all_config.yaml")):
            sys.exit(f"[evaluate] no all_config.yaml in {ckpt_dir}; pass --config <name> from config/")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return OmegaConf.to_container(compose(config_name=name, overrides=list(overrides)), resolve=True)


def find_weights(ckpt_dir: str, step):
    steps = sorted(int(m.group(1)) for f in os.listdir(ckpt_dir) if (m := re.fullmatch(r"step_(\d+)", f)))
    if not steps:
        sys.exit(f"[evaluate] no step_<N> weights in {ckpt_dir}; download them first (README.md, Checkpoints)")
    if step is None:
        step = steps[-1]
    elif step not in steps:
        sys.exit(f"[evaluate] no step_{step} in {ckpt_dir}; available: {steps}")
    return step, os.path.join(ckpt_dir, f"step_{step}")


def load_weights(model, path: str):
    sd = torch.load(path, map_location="cuda")
    prefix = "_orig_mod." if hasattr(model, "_orig_mod") else ""
    sd = {prefix + k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)  # still raises on a shape mismatch
    # Checkpoints saved before weight dropout existed lack the CastedLinear `.mask` buffers (ones, unused at eval).
    missing = [k for k in missing if not k.endswith(".mask")]
    if missing or unexpected:
        raise RuntimeError(f"{path} does not match the model built from the config: "
                           f"missing={missing[:8]} unexpected={list(unexpected)[:8]}")


def select_examples(data_path: str, args):
    """(indices of the test examples to evaluate, or None for the whole split; size of the split)."""
    with open(os.path.join(data_path, "test", "dataset.json")) as f:
        sets = json.load(f)["sets"]
    n_total = sum(np.load(os.path.join(data_path, "test", f"{s}__inputs.npy"), mmap_mode="r").shape[0] for s in sets)
    if args.sample is not None:
        idx = np.sort(np.random.default_rng(args.sample_seed).choice(n_total, min(args.sample, n_total), replace=False))
    else:
        idx = np.arange(n_total if args.max_examples is None else min(args.max_examples, n_total))
    if args.num_shards > 1:
        per = math.ceil(len(idx) / args.num_shards)
        idx = idx[args.shard * per:(args.shard + 1) * per]
    if len(idx) == n_total:
        return None, n_total
    if sets != ["all"]:
        sys.exit(f"[evaluate] subsets need a single-set test split, got sets {sets}")
    return idx, n_total


def write_subset(data_path: str, idx: np.ndarray, out_dir: str):
    """A copy of the test split restricted to idx. Needs one example per puzzle and one puzzle per group."""
    src, dst = os.path.join(data_path, "test"), os.path.join(out_dir, "test")
    os.makedirs(dst)
    load = lambda field, mode=None: np.load(os.path.join(src, f"all__{field}.npy"), mmap_mode=mode)  # noqa: E731
    puzzle_indices, group_indices = load("puzzle_indices"), load("group_indices")
    n_total = load("inputs", "r").shape[0]
    if not len(puzzle_indices) - 1 == len(group_indices) - 1 == n_total:
        sys.exit("[evaluate] --max-examples/--sample/--num-shards need one example per puzzle (Sudoku, Maze)")
    for field in ("inputs", "labels", "puzzle_identifiers"):
        np.save(os.path.join(dst, f"all__{field}.npy"), load(field, "r")[idx])
    for field, ref in (("puzzle_indices", puzzle_indices), ("group_indices", group_indices)):
        np.save(os.path.join(dst, f"all__{field}.npy"), np.arange(len(idx) + 1, dtype=ref.dtype))
    with open(os.path.join(src, "dataset.json")) as f:
        meta = json.load(f)
    meta.update(total_groups=len(idx), total_puzzles=len(idx), mean_puzzle_examples=1.0)
    with open(os.path.join(dst, "dataset.json"), "w") as f:
        json.dump(meta, f)


def init_distributed():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, None
    # Ranks only meet at the final reduce, which can be hours apart on Sudoku: allow a long wait.
    timeout = timedelta(hours=float(os.environ.get("EVAL_TIMEOUT_HOURS", 48)))
    dist.init_process_group(backend="nccl", timeout=timeout)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return dist.get_rank(), dist.get_world_size(), dist.new_group(backend="gloo", timeout=timeout)


def run(args):
    from create_model import create_model
    from pretrain import TrainState, create_dataloader, create_evaluators, evaluate
    from pretrain_config import PretrainConfig
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    from utils.flop_profiler import FlopProfiler

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    os.chdir(REPO)  # data_paths in the configs are repo-relative
    rank, world_size, cpu_group = init_distributed()

    config = PretrainConfig(**load_config(ckpt_dir, args.config, args.overrides))
    step, weights = find_weights(ckpt_dir, args.step)
    batch = args.batch or config.global_batch_size // world_size
    config.global_batch_size = batch * world_size
    config.load_checkpoint = config.resume_from = config.checkpoint_path = None  # load here; write nothing
    config.eval_max_examples, config.eval_save_outputs = None, []
    # For a single checkpoint a fresh MiniARC vote is already the per-checkpoint score MiniARCSingle logs.
    config.evaluators = [e for e in config.evaluators if not e.name.endswith("@MiniARCSingle")]
    data_paths = config.data_paths_test or config.data_paths
    if len(data_paths) != 1:
        sys.exit(f"[evaluate] expected one test data path, got {data_paths}")
    data_path = data_paths[0]
    idx, n_total = select_examples(data_path, args)
    if idx is not None and config.evaluators:
        sys.exit("[evaluate] the voting evaluators need the whole test split: drop --max-examples/--sample/--num-shards")

    tmp = None
    try:
        if idx is not None:
            tmp = tempfile.mkdtemp(prefix="sfprm_eval_")
            write_subset(data_path, idx, tmp)
            config.data_paths_test = [tmp]
        train_metadata = PuzzleDataset(PuzzleDatasetConfig(
            seed=config.seed, dataset_paths=config.data_paths, global_batch_size=config.global_batch_size,
            test_set_mode=False, epochs_per_iter=1, rank=rank, num_replicas=world_size), split="train").metadata
        eval_loader, eval_metadata = create_dataloader(
            config, "test", rank=rank, world_size=world_size, test_set_mode=True, epochs_per_iter=1,
            global_batch_size=config.global_batch_size)
        with contextlib.redirect_stdout(io.StringIO()):  # create_model prints the whole module
            model, _, _ = create_model(config, train_metadata, rank=rank, world_size=world_size)
        load_weights(model, weights)
        model.eval()
        if args.renormalize_weights:
            model.normalize_weights()
        evaluators = create_evaluators(config, eval_metadata)

        n = n_total if idx is None else len(idx)
        if rank == 0:
            log(f"{os.path.relpath(weights)} [{config.arch.name}]: {n}/{n_total} test examples of {data_path}, "
                f"{world_size} GPU(s) x batch {batch}")
        state = TrainState(model=model, optimizers=[], optimizer_lrs=[], carry=None, step=step, total_steps=step)
        torch.manual_seed(args.seed + rank)
        t0 = time.time()
        metrics = evaluate(config, state, eval_loader, eval_metadata, evaluators, rank=rank, world_size=world_size,
                           cpu_group=cpu_group, profiler=FlopProfiler(enabled=False))
        wall = time.time() - t0
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    if rank == 0:
        set_metrics = [v for v in metrics.values() if isinstance(v, dict)]
        flat = {k: float(v) for k, v in (set_metrics[0] if len(set_metrics) == 1 else {}).items()}
        flat.update({k: float(v) for k, v in metrics.items() if not isinstance(v, dict)})
        result = {
            "checkpoint": os.path.relpath(weights), "config": args.config or "all_config.yaml",
            "overrides": list(args.overrides), "data": data_path, "n_examples": n, "n_test": n_total,
            "max_examples": args.max_examples, "sample": args.sample, "sample_seed": args.sample_seed,
            "shard": args.shard, "num_shards": args.num_shards, "batch_per_gpu": batch, "world_size": world_size,
            "seed": args.seed, "renormalize_weights": args.renormalize_weights,
            "exact_accuracy": flat.get("sequence_accuracy"),
            "token_accuracy": flat.get("accuracy"), "metrics": flat, "wall_seconds": wall,
        }
        print_result(result)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=1)
            log(f"wrote {args.out}")
    if world_size > 1:
        dist.destroy_process_group()


def print_result(r):
    m = r["metrics"]
    print(f"\n{r['checkpoint']}: {r['n_examples']} examples, {r['world_size']} GPU(s) x batch "
          f"{r['batch_per_gpu']}, seed {r['seed']}, {r['wall_seconds'] / 60:.1f} min")
    print(f"  exact accuracy   {100 * r['exact_accuracy']:.2f}%")
    print(f"  token accuracy   {100 * r['token_accuracy']:.2f}%")
    for k in sorted((k for k in m if "/pass@" in k), key=lambda k: (k.split("/pass@")[0], int(k.split("@")[-1]))):
        print(f"  {k.split('/', 1)[1]:<16} {100 * m[k]:.2f}%")
    if "eval/inference_steps_mean" in m:
        print(f"  segments/batch   mean {m['eval/inference_steps_mean']:.1f}, max {m['eval/inference_steps_max']:.0f}")


def merge(paths):
    """Combine the --out JSON files of the shards of one evaluation."""
    rs = [json.load(open(p)) for p in paths]
    same = ("checkpoint", "config", "overrides", "data", "max_examples", "sample", "sample_seed", "num_shards",
            "batch_per_gpu", "seed", "renormalize_weights")
    for k in same:
        if len({json.dumps(r.get(k)) for r in rs}) > 1:
            sys.exit(f"[evaluate] the files differ in {k}: {[r.get(k) for r in rs]}")
    shards = sorted(r["shard"] for r in rs)
    if shards != list(range(rs[0]["num_shards"])):
        sys.exit(f"[evaluate] expected shards 0..{rs[0]['num_shards'] - 1} once each, got {shards}")
    n = sum(r["n_examples"] for r in rs)
    out = {k: rs[0].get(k) for k in same}
    out.update(n_examples=n, n_test=rs[0]["n_test"],
               exact_accuracy=sum(r["exact_accuracy"] * r["n_examples"] for r in rs) / n,
               token_accuracy=sum(r["token_accuracy"] * r["n_examples"] for r in rs) / n)
    print(f"{out['checkpoint']}: {len(rs)} shards, {n} examples")
    print(f"  exact accuracy   {100 * out['exact_accuracy']:.2f}%")
    print(f"  token accuracy   {100 * out['token_accuracy']:.2f}%")
    print(json.dumps(out))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt_dir", nargs="?", help="checkpoint directory with step_<N> (and all_config.yaml)")
    p.add_argument("overrides", nargs="*", help="Hydra overrides of the config, e.g. arch.max_iter_eval=200")
    p.add_argument("--config", help="build from config/<name>.yaml instead of the directory's all_config.yaml")
    p.add_argument("--step", type=int, help="evaluate step_<N> (default: the latest)")
    p.add_argument("--batch", type=int, help="per-GPU batch (default: the config's global_batch_size / GPUs)")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--max-examples", type=int, help="evaluate the first N test examples")
    sel.add_argument("--sample", type=int, help="evaluate N test examples drawn at random (seeded)")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0, help="evaluate the I-th contiguous slice of the selection")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="torch seed of the evaluation (plus the GPU rank)")
    p.add_argument("--renormalize-weights", action="store_true",
                   help="project SFPRM's nGPT weights onto the unit sphere after loading (normalize_weights())")
    p.add_argument("--out", help="write the result to this JSON file")
    p.add_argument("--merge", nargs="+", metavar="JSON", help="combine the --out files of the shards of one run")
    args = p.parse_args()
    if args.merge:
        return args
    if args.ckpt_dir is None:
        p.error("give a checkpoint directory (or --merge)")
    if not 0 <= args.shard < args.num_shards:
        p.error(f"--shard must be in [0, {args.num_shards})")
    for o in args.overrides:
        if "=" not in o and not o.startswith("~"):
            p.error(f"'{o}' is not a Hydra override (key=value, ++key=value or ~key)")
    return args


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # progress lines reach a redirected log as they happen
    a = parse_args()
    merge(a.merge) if a.merge else run(a)
