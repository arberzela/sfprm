import math
from typing import Optional

from models.config import ReasoningModelConfig


class SFPRMConfig(ReasoningModelConfig):
    """Config for the Spherical Fixed-Point Reasoner (SFPRM).

    Mirrors FPRMConfig but for the on-sphere variant: the Euclidean residual-scaling /
    Jacobian-norm fields are replaced by the spherical block (eigen learning rates, on-sphere
    input injection, logit scale) and the tangent-space log-norm regularizer.
    """

    # Base-class halting fields are unused (SFPRM halts via fixed points); default them.
    halt_max_steps: int = 1
    halt_exploration_prob: float = 0.0

    # ---- fixed-point iteration / segment cap ----
    max_iter: int
    max_iter_eval: Optional[int] = None
    max_iter_dist: str = "det"            # 'det' | 'gamma' | 'expon' (per-batch resample at train)
    gamma_alpha: float = 4.0
    gamma_scale: float = 50.0 / 3.0
    expon_scale: float = 50.0 / math.log(2.0)

    # Variational dropout on the block output (multiplicative mask reused across iterations).
    variational_dropout: float = 0.0

    # ---- spherical fixed-point solver (see models/spherical_fixed_point_reasoning/spherical.py) ----
    fp_thresh: float = 0.01               # chordal halting tolerance (1 - cos(angle))
    eta_min: float = 1.0e-3               # angular step-size floor (termination fallback)
    slerp_eps: float = 1.0e-6             # gSlerp small-angle/antipodal guard threshold
    outlier_quantile: float = 0.25
    stepsize: float = 1.0                 # initial angular step eta
    stepsize_decay_train: float = 0.9
    stepsize_decay_eval: float = 0.99
    decay_patience: int = 5
    eps: float = 1.0e-8
    init_std: float = 1.0
    additive_noise_std: float = 0.0
    fixed_init: bool = False

    # ---- spherical block (nGPT-style) ----
    eigen_lr_init: float = 0.5            # init for the per-dim eigen learning rates alpha_A/alpha_M
    lambda_inject: float = 0.1            # on-sphere input re-injection fraction (>0: anti-collapse)
    lambda_inject_learnable: bool = False
    sz_init: float = 1.0                  # learned logit scale s_z
    qk_norm: bool = True                  # nGPT QK-normalization in attention
    s_qk_init: float = 1.0                # init for the learned QK-norm scale
    normalize_weights: bool = True        # nGPT weight normalization (post-step + at init)
    normalize_lm_head: bool = True        # also unit-normalize lm_head rows (False -> unbounded logits)
    eval_readout_best: bool = True        # eval reads best-residual iterate (False -> last, matching train)
    use_conv: bool = False
    conv_type: str = "conv1d"             # 'conv1d' (causal, whole sequence) | 'conv2d' (grid)
    conv_kernel_size: int = 4             # must be ODD for conv2d
    conv_bias: bool = False

    # ---- tangent-space log-norm regularizer (paper Sec. 3.5); weight = loss.jacobian_reg_lambda ----
    n_spectral_samples: int = 1
    spectral_eps: float = 1.0e-3
    spectral_delta: float = 0.05
    spectral_xi: float = 0.0
