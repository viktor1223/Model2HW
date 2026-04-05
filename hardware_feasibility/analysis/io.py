"""Host-device IO analysis: PCIe / link pressure estimation."""

from __future__ import annotations

from dataclasses import dataclass

from ..models.architecture_rules import ModelSpec


BYTES_PER_GB = 1 << 30


@dataclass
class IOProfile:
    """Host-to-device IO estimates per token."""

    # Bytes transferred per token over the host link
    input_embedding_bytes: int
    output_logits_bytes: int
    kv_sync_bytes_per_token: int  # non-zero only if KV cache is off-accelerator
    total_io_bytes_per_token: int

    # Sustained IO bandwidth required at target tok/s
    required_io_bandwidth_bytes_per_sec: float
    target_tokens_per_sec: float

    @property
    def required_io_bandwidth_gbps(self) -> float:
        return self.required_io_bandwidth_bytes_per_sec / BYTES_PER_GB


def analyze_io(spec: ModelSpec) -> IOProfile:
    """Estimate host-device link traffic per decode token.

    For a self-contained accelerator (weights + KV on device), the host link
    only carries:
      - input token IDs (negligible)
      - output logits or sampled token

    If the KV cache lives on the host, every decode step must stream the
    relevant KV slices across the link, which can be a major bottleneck.
    """
    bpw = spec.weight_precision.bytes_per_element

    # Input: one token embedding pushed to device
    # (in practice just a token ID, but we account for the looked-up embedding)
    input_bytes = spec.hidden_size * bpw

    # Output: logits over vocab (worst case) or just a sampled token ID
    # Use full logits as the conservative estimate
    output_bytes = spec.vocab_size * bpw

    # KV sync: if KV cache is NOT on accelerator, each decode step
    # must read the full KV cache from host
    if not spec.kv_on_accelerator:
        avg_seq = spec.prefill_length + spec.decode_length // 2
        kv_sync = int(
            2
            * spec.num_layers
            * spec.batch_size
            * avg_seq
            * spec.num_kv_heads
            * spec.head_dim
            * spec.bytes_per_kv
        )
    else:
        kv_sync = 0

    total = input_bytes + output_bytes + kv_sync
    required_bw = total * spec.target_tokens_per_sec

    return IOProfile(
        input_embedding_bytes=int(input_bytes),
        output_logits_bytes=int(output_bytes),
        kv_sync_bytes_per_token=kv_sync,
        total_io_bytes_per_token=int(total),
        required_io_bandwidth_bytes_per_sec=required_bw,
        target_tokens_per_sec=spec.target_tokens_per_sec,
    )
