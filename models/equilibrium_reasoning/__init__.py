"""EqR (Equilibrium Reasoners, locuslab/EqR) ported to this repo's training stack.

Use it as arch name `equilibrium_reasoning.eqr@EqRModel` (config/arch/eqr.yaml).
"""

from models.equilibrium_reasoning.eqr import EqRConfig, EqRModel, apply_runtime_overrides

__all__ = ["EqRConfig", "EqRModel", "apply_runtime_overrides"]
