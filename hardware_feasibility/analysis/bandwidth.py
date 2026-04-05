"""Bandwidth analysis: memory bandwidth required for target throughput."""

from __future__ import annotations

from dataclasses import dataclass

from ..models.architecture_rules import ModelSpec
from .memory import estimate_weight_memory, estimate_kv_cache, BYTES_PER_GB


@dataclass
class BandwidthProfile:
    """Bandwidth estimates for autoregressive decode."""

    # Bytes that must be streamed from memory per output token
    weight_bytes_per_token: int
    kv_read_bytes_per_token: int
    total_bytes_per_token: int

    # Required bandwidth to hit target tok/s
    required_bandwidth_bytes_per_sec: float
    target_tokens_per_sec: float

    @property
    def required_bandwidth_gbps(self) -> float:
        return self.required_bandwidth_bytes_per_sec / BYTES_PER_GB

    @property
    def weight_bytes_per_token_gb(self) -> float:
        return self.weight_bytes_per_token / BYTES_PER_GB

    @property
    def total_bytes_per_token_gb(self) -> float:
        return self.total_bytes_per_token / BYTES_PER_GB


def analyze_bandwidth(spec: ModelSpec) -> BandwidthProfile:
    """Estimate memory bandwidth requirements for autoregressive decode.

    During decode, each output token requires reading:
      1. All model weights (assuming no weight caching / reuse across tokens,
         which is the worst-case for memory-bandwidth-bound systems).
      2. The KV cache for the current sequence length.

    This is a conservative upper bound. Real implementations with operator
    fusion, weight tiling, and on-chip caching will do better, but for
    feasibility screening this is the right starting point.
    """
    weight_bytes = estimate_weight_memory(spec)

    # Average KV read per decode token: we read the full KV cache built so far.
    # For a rough estimate use the midpoint of the decode phase.
    avg_seq_len = spec.prefill_length + spec.decode_length // 2
    kv_read = int(
        2
        * spec.num_layers
        * spec.batch_size
        * avg_seq_len
        * spec.num_kv_heads
        * spec.head_dim
        * spec.bytes_per_kv
    )

    total_per_token = weight_bytes + kv_read
    required_bw = total_per_token * spec.target_tokens_per_sec

    return BandwidthProfile(
        weight_bytes_per_token=weight_bytes,
        kv_read_bytes_per_token=kv_read,
        total_bytes_per_token=total_per_token,
        required_bandwidth_bytes_per_sec=required_bw,
        target_tokens_per_sec=spec.target_tokens_per_sec,
    )
