"""RMM reference implementation.

Heavy dependencies are imported lazily so metadata and configuration tools can
be inspected without importing PyTorch or Transformers.
"""

from .config import RMMConfig

__all__ = [
    "RMMConfig",
    "InputAwareLinear",
    "install_projection_pruning",
    "install_rmm_patch",
    "projection_pruning",
    "rmm_attention_forward",
    "rmm_patch",
]


def __getattr__(name):
    if name == "rmm_attention_forward":
        from .attention import rmm_attention_forward

        return rmm_attention_forward
    if name in {"install_rmm_patch", "rmm_patch"}:
        from .patch import install_rmm_patch, rmm_patch

        return {"install_rmm_patch": install_rmm_patch, "rmm_patch": rmm_patch}[name]
    if name in {
        "InputAwareLinear",
        "install_projection_pruning",
        "projection_pruning",
    }:
        from .linear import (
            InputAwareLinear,
            install_projection_pruning,
            projection_pruning,
        )

        return {
            "InputAwareLinear": InputAwareLinear,
            "install_projection_pruning": install_projection_pruning,
            "projection_pruning": projection_pruning,
        }[name]
    raise AttributeError(name)
