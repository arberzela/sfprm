import pydantic

from typing import Optional, List

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig

class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Optimizer for dense parameters
    optimizer: str = "adamw"
    # Muon (used when optimizer == "muon"): 2D matrices use Muon, everything else AdamW.
    muon_lr: float = 0.02
    muon_momentum: float = 0.95

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    wandb_entity: Optional[str] = None  # W&B team/user; falls back to the account default
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None  # weights-only load (transfer / eval)
    resume_from: Optional[str] = None  # full resume: bundle path or its directory
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    checkpoint_every_n_steps: Optional[int] = None  # if set, also save a checkpoint every N training steps
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # number of eval iters (each = eval_interval epochs) to skip before starting eval
    eval_save_outputs: List[str] = []
    eval_max_examples: Optional[int] = None  # cap examples per set at eval time (deterministic prefix); None = full test set

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

    profile_flops: bool = False  # one-shot per-segment FLOPs + accumulator (utils.flop_profiler)

    # Parameter / gradient telemetry stride (utils.telemetry). Each call syncs the GPU, so it runs
    # every N train steps rather than every step; 0 disables it. The per-iteration solver telemetry
    # (residual, step size, patience, gslerp branch, ...) is free and always logged.
    telemetry_every: int = 20
