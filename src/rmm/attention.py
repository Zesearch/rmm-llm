"""Pure PyTorch reference implementation of RMM attention pruning.

This module prioritizes a transparent implementation of the paper semantics.
It is not a fused kernel and does not claim wall-clock speedup.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .config import RMMConfig


def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    """Expand grouped key/value heads to the number of query heads."""

    if num_repeats == 1:
        return hidden_states
    batch, kv_heads, sequence_length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, num_repeats, sequence_length, head_dim
    )
    return hidden_states.reshape(
        batch, kv_heads * num_repeats, sequence_length, head_dim
    )


def rmm_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    *,
    config: Optional[RMMConfig] = None,
    **_kwargs,
):
    """Compute RMM-pruned eager attention.

    The implementation follows the original experiment code:

    1. Compute the input-dependent L2 norm of each Q feature over the query
       sequence, independently for every batch item and attention head.
    2. Use the selected feature indices for both Q and K when computing scores.
    3. Compute masked softmax attention, rank key/token columns by their L2 norm
       over the query sequence, and retain the same columns from V.
    4. Do not renormalize after token selection.

    The incoming Transformers scaling factor is intentionally preserved after
    feature pruning to match the paper experiments.
    """

    config = config or RMMConfig()
    config.validate()

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    batch_size, num_heads, query_length, head_dim = query.shape
    _, _, key_length, value_dim = value_states.shape
    if key_states.shape[:3] != value_states.shape[:3]:
        raise ValueError("key and value must have matching batch/head/token dimensions")

    kept_dimensions = max(
        1, math.ceil(head_dim * config.dimension_keep_ratio)
    )
    kept_tokens = max(1, math.ceil(key_length * config.token_keep_ratio))

    query_flat = query.reshape(batch_size * num_heads, query_length, head_dim)
    key_flat = key_states.reshape(batch_size * num_heads, key_length, head_dim)

    with torch.no_grad():
        dimension_scores = query_flat.norm(dim=1)
        dimension_indices = torch.topk(
            dimension_scores,
            kept_dimensions,
            dim=-1,
            sorted=False,
        ).indices

    query_indices = dimension_indices.unsqueeze(1).expand(
        -1, query_length, -1
    )
    key_indices = dimension_indices.unsqueeze(1).expand(-1, key_length, -1)
    query_pruned = torch.gather(query_flat, 2, query_indices).reshape(
        batch_size, num_heads, query_length, kept_dimensions
    )
    key_pruned = torch.gather(key_flat, 2, key_indices).reshape(
        batch_size, num_heads, key_length, kept_dimensions
    )

    attention_weights = torch.matmul(
        query_pruned, key_pruned.transpose(2, 3)
    ) * scaling

    if attention_mask is not None:
        attention_weights = attention_weights + attention_mask[..., :key_length]

    attention_weights = nn.functional.softmax(
        attention_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    attention_weights = nn.functional.dropout(
        attention_weights, p=dropout, training=module.training
    )

    attention_flat = attention_weights.reshape(
        batch_size * num_heads, query_length, key_length
    )
    value_flat = value_states.reshape(
        batch_size * num_heads, key_length, value_dim
    )

    with torch.no_grad():
        token_scores = attention_flat.norm(dim=1)
        token_indices = torch.topk(
            token_scores, kept_tokens, dim=-1, sorted=False
        ).indices

    attention_indices = token_indices.unsqueeze(1).expand(
        -1, query_length, -1
    )
    value_indices = token_indices.unsqueeze(-1).expand(-1, -1, value_dim)
    attention_pruned = torch.gather(attention_flat, 2, attention_indices)
    value_pruned = torch.gather(value_flat, 1, value_indices)

    attention_pruned = attention_pruned.reshape(
        batch_size, num_heads, query_length, kept_tokens
    )
    value_pruned = value_pruned.reshape(
        batch_size, num_heads, kept_tokens, value_dim
    )
    output = torch.matmul(attention_pruned, value_pruned)
    output = output.transpose(1, 2).contiguous()
    return output, attention_weights

