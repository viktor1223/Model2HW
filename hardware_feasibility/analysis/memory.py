"""Memory analysis: weights, KV cache, activation buffers, total working set."""

from __future__ import annotations

from dataclasses import dataclass

from ..models.architecture_rules import ModelSpec

BYTES_PER_GB = 1 << 30


@dataclass
class MemoryProfile:
    """All memory estimates in bytes."""

    weight_bytes: int
    kv_cache_bytes: int
    kv_cache_per_token_bytes: int
    activation_buffer_bytes: int
    total_bytes: int

    # Convenience GB values
    @property
    def weight_gb(self) -> float:
        return self.weight_bytes / BYTES_PER_GB

    @property
    def kv_cache_gb(self) -> float:
        return self.kv_cache_bytes / BYTES_PER_GB

    @property
    def kv_cache_per_token_bytes_val(self) -> int:
        return self.kv_cache_per_token_bytes

    @property
    def activation_buffer_gb(self) -> float:
        return self.activation_buffer_bytes / BYTES_PER_GB

    @property
    def total_gb(self) -> float:
        return self.total_bytes / BYTES_PER_GB


def estimate_weight_memory(spec: ModelSpec) -> int:
    """Total weight memory in bytes = params * bytes_per_weight."""
    return int(spec.params * spec.bytes_per_weight)


def estimate_kv_cache_per_token(spec: ModelSpec) -> int:
    """KV cache bytes for a single token across all layers.

    Formula: 2 * num_layers * num_kv_heads * head_dim * bytes_per_kv
    The factor of 2 accounts for both keys and values.
    """
    return int(
        2
        * spec.num_layers
        * spec.num_kv_heads
        * spec.head_dim
        * spec.bytes_per_kv
    )


def estimate_kv_cache(spec: ModelSpec) -> int:
    """Total KV cache at the target context length for the given batch size.

    Formula: 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * bytes_per_kv
    """
    return int(
        2
        * spec.num_layers
        * spec.batch_size
        * spec.context_length
        * spec.num_kv_heads
        * spec.head_dim
        * spec.bytes_per_kv
    )


def estimate_activation_buffer(spec: ModelSpec) -> int:
    """Conservative estimate for activation / temporary buffers.

    During inference the main transient tensors are:
      - hidden states: batch_size * seq_len * hidden_size * bytes (per layer)
      - attention scores: batch_size * num_heads * seq_len * seq_len * bytes
        (only for prefill; decode is seq_len=1)
      - MLP intermediates: batch_size * seq_len * intermediate_size * bytes

    We estimate peak during prefill since that is the high-water mark,
    but only for a single layer (pipeline execution reuses buffers).
    """
    bpw = spec.weight_precision.bytes_per_element  # use weight precision for activations

    # Hidden state buffer (one layer's input/output)
    hidden_buf = spec.batch_size * spec.prefill_length * spec.hidden_size * bpw

    # Attention score matrix during prefill (one head group at a time is optimistic,
    # but full heads is conservative)
    attn_scores = (
        spec.batch_size
        * spec.num_attention_heads
        * spec.prefill_length
        * spec.prefill_length
        * bpw
    )

    # MLP intermediate
    mlp_buf = spec.batch_size * spec.prefill_length * spec.intermediate_size * bpw

    # We take the max of attention-dominated and MLP-dominated phases,
    # plus a hidden state buffer that is always present.
    peak = hidden_buf + max(attn_scores, mlp_buf)

    return int(peak)


def analyze_memory(spec: ModelSpec) -> MemoryProfile:
    """Run the full memory analysis and return a MemoryProfile."""
    w = estimate_weight_memory(spec)
    kv = estimate_kv_cache(spec)
    kv_per_tok = estimate_kv_cache_per_token(spec)
    act = estimate_activation_buffer(spec)
    total = w + kv + act

    return MemoryProfile(
        weight_bytes=w,
        kv_cache_bytes=kv,
        kv_cache_per_token_bytes=kv_per_tok,
        activation_buffer_bytes=act,
        total_bytes=total,
    )
