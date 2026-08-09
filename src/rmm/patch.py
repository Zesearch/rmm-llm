"""Adapters for using RMM with Hugging Face eager attention."""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Iterable, Optional

from .attention import rmm_attention_forward
from .config import RMMConfig


MODEL_MODULES = {
    "llama": "transformers.models.llama.modeling_llama",
    "qwen2": "transformers.models.qwen2.modeling_qwen2",
    "qwen3": "transformers.models.qwen3.modeling_qwen3",
    "mistral": "transformers.models.mistral.modeling_mistral",
    "gemma": "transformers.models.gemma.modeling_gemma",
    "gemma2": "transformers.models.gemma2.modeling_gemma2",
}


@dataclass
class PatchHandle:
    originals: Dict[object, object]

    def remove(self):
        for module, original in self.originals.items():
            module.eager_attention_forward = original
        self.originals.clear()


def _make_forward(original, config):
    def forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        if config.is_baseline:
            return original(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout=dropout,
                **kwargs,
            )
        return rmm_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            config=config,
            **kwargs,
        )

    forward.__name__ = "rmm_eager_attention_forward"
    return forward


def install_rmm_patch(
    config: RMMConfig,
    model_types: Iterable[str] = ("llama",),
) -> PatchHandle:
    """Patch supported Transformers model families and return a removal handle."""

    config.validate()
    originals = {}
    for model_type in model_types:
        if model_type not in MODEL_MODULES:
            supported = ", ".join(sorted(MODEL_MODULES))
            raise ValueError(
                f"Unsupported model_type={model_type!r}. Supported values: {supported}"
            )
        module = import_module(MODEL_MODULES[model_type])
        original = getattr(module, "eager_attention_forward", None)
        if original is None:
            raise RuntimeError(
                f"{MODEL_MODULES[model_type]} has no eager_attention_forward. "
                "Use a compatible Transformers release or add an adapter."
            )
        originals[module] = original
        module.eager_attention_forward = _make_forward(original, config)
    return PatchHandle(originals)


@contextmanager
def rmm_patch(
    config: Optional[RMMConfig] = None,
    model_types: Iterable[str] = ("llama",),
):
    """Temporarily install RMM eager-attention patches."""

    handle = install_rmm_patch(config or RMMConfig(), model_types)
    try:
        yield handle
    finally:
        handle.remove()

