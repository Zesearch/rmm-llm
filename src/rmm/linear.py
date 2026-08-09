"""Dense QKV/MLP projection pruning used by the RMM paper."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECTION_TARGETS = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


class InputAwareLinear(nn.Module):
    """Apply a dense Linear layer using input-dependent selected columns.

    This reference module expects a floating-point ``nn.Linear``. It is kept
    separate from the quantization path because indexing arbitrary quantized
    weight containers is backend-specific.
    """

    def __init__(self, base_linear: nn.Linear, keep_ratio: float = 1.0):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("InputAwareLinear currently supports nn.Linear only")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.base_linear = base_linear
        self.keep_ratio = float(keep_ratio)

    @property
    def in_features(self):
        return self.base_linear.in_features

    @property
    def out_features(self):
        return self.base_linear.out_features

    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias

    def forward(self, inputs):
        if self.keep_ratio >= 1.0:
            return self.base_linear(inputs)
        if inputs.ndim < 2:
            raise ValueError("expected inputs with shape [..., tokens, features]")

        squeezed = inputs.ndim == 2
        if squeezed:
            inputs = inputs.unsqueeze(0)
        batch = inputs.shape[0]
        flattened = inputs.reshape(batch, -1, inputs.shape[-1])
        kept = max(1, int(flattened.shape[-1] * self.keep_ratio))

        with torch.no_grad():
            scores = flattened.norm(dim=1)
            indices = torch.topk(scores, kept, dim=-1, sorted=False).indices

        outputs = []
        for batch_index in range(batch):
            selected_inputs = flattened[batch_index].index_select(
                -1, indices[batch_index]
            )
            selected_weight = self.base_linear.weight.index_select(
                1, indices[batch_index]
            )
            outputs.append(
                F.linear(selected_inputs, selected_weight, self.base_linear.bias)
            )
        output = torch.stack(outputs, dim=0)
        output = output.reshape(*inputs.shape[:-1], self.out_features)
        return output.squeeze(0) if squeezed else output


@dataclass
class ProjectionPatchHandle:
    """Handle for inspecting and restoring temporarily wrapped projections."""

    replacements: List[Tuple[nn.Module, str, nn.Linear, str]]

    @property
    def names(self):
        return [qualified_name for _, _, _, qualified_name in self.replacements]

    def remove(self):
        for parent, child_name, original, _ in reversed(self.replacements):
            setattr(parent, child_name, original)
        self.replacements.clear()


def _projection_suffixes(targets: Iterable[str]):
    if isinstance(targets, str):
        targets = (targets,)

    suffixes = []
    for target in targets:
        if target not in PROJECTION_TARGETS:
            choices = ", ".join(sorted(PROJECTION_TARGETS))
            raise ValueError(f"Unknown projection target {target!r}; choose from {choices}")
        suffixes.extend(PROJECTION_TARGETS[target])
    return tuple(suffixes)


def install_projection_pruning(model, targets=("qkv", "mlp"), keep_ratio=1.0):
    """Wrap selected dense QKV/MLP projections and return a removal handle.

    Selection is based on the standard Hugging Face projection names. Only
    floating-point ``nn.Linear`` modules are accepted; quantized projection
    containers are intentionally outside this dense implementation.
    """

    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")

    targets = (targets,) if isinstance(targets, str) else tuple(targets)
    suffixes = _projection_suffixes(targets)
    candidates = []
    for full_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name not in suffixes:
                continue
            qualified_name = f"{full_name}.{child_name}" if full_name else child_name
            if isinstance(child, InputAwareLinear):
                raise RuntimeError(f"Projection {qualified_name} is already RMM-wrapped")
            if not isinstance(child, nn.Linear):
                raise TypeError(
                    f"Projection {qualified_name} is {type(child).__name__}, not nn.Linear"
                )
            candidates.append((module, child_name, child, qualified_name))

    if not candidates:
        requested = ", ".join(targets)
        raise RuntimeError(f"No dense projections found for target(s): {requested}")

    replacements = []
    for module, child_name, child, qualified_name in candidates:
        setattr(module, child_name, InputAwareLinear(child, keep_ratio))
        replacements.append((module, child_name, child, qualified_name))
    return ProjectionPatchHandle(replacements)


@contextmanager
def projection_pruning(model, targets=("qkv", "mlp"), keep_ratio=1.0):
    """Temporarily enable dense QKV/MLP projection pruning on ``model``."""

    handle = install_projection_pruning(model, targets, keep_ratio)
    try:
        yield handle
    finally:
        handle.remove()


def replace_dense_linears(model, module_suffixes, keep_ratio):
    """Legacy low-level replacement helper; prefer ``projection_pruning``."""

    replaced = []
    suffixes = (
        (module_suffixes,)
        if isinstance(module_suffixes, str)
        else tuple(module_suffixes)
    )
    for full_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            qualified_name = f"{full_name}.{child_name}" if full_name else child_name
            if qualified_name.endswith(suffixes) and isinstance(child, nn.Linear):
                setattr(module, child_name, InputAwareLinear(child, keep_ratio))
                replaced.append(qualified_name)
    return replaced
