"""Compute analysis: FLOPs per token and arithmetic intensity."""

from __future__ import annotations

from dataclasses import dataclass

from ..models.architecture_rules import ModelSpec


@dataclass
class ComputeProfile:
    """Compute estimates for a single decode token."""

    # FLOPs for one decode token (forward pass)
    flops_per_token: int

    # FLOPs for full prefill
    flops_prefill: int

    # Arithmetic intensity (FLOPs / byte of memory traffic)
    arithmetic_intensity: float

    # Whether the workload is likely compute-bound or memory-bound
    # on a system with the given bandwidth and compute
    roofline_note: str


def estimate_flops_per_token(spec: ModelSpec) -> int:
    """Rough FLOP count for a single autoregressive decode step.

    For a decoder-only transformer, the dominant operations per token are:

    1. Linear projections (Q, K, V, O, gate, up, down per layer):
       Each linear of shape (M, N) costs 2*M*N FLOPs per token.

    2. Attention: softmax(Q*K^T)*V
       QK^T: 2 * num_heads * head_dim * seq_len
       score*V: 2 * num_heads * seq_len * head_dim

    For decode (seq_len=1 for the new token), attention FLOPs scale
    with current sequence length but are typically small relative to
    the linear projections for large models.
    """
    h = spec.hidden_size
    n_heads = spec.num_attention_heads
    n_kv = spec.num_kv_heads
    head_dim = spec.head_dim
    inter = spec.intermediate_size
    n_layers = spec.num_layers
    avg_seq = spec.prefill_length + spec.decode_length // 2

    # Per-layer linear projection FLOPs (multiply-accumulate * 2 for mul+add)
    q_flops = 2 * h * (n_heads * head_dim)
    k_flops = 2 * h * (n_kv * head_dim)
    v_flops = 2 * h * (n_kv * head_dim)
    o_flops = 2 * (n_heads * head_dim) * h

    # SwiGLU MLP: gate, up each (h -> inter), then down (inter -> h)
    mlp_flops = 3 * 2 * h * inter

    # Attention score computation for one query token against avg_seq keys
    attn_qk = 2 * n_heads * head_dim * avg_seq
    attn_sv = 2 * n_heads * avg_seq * head_dim

    per_layer = q_flops + k_flops + v_flops + o_flops + mlp_flops + attn_qk + attn_sv
    total = n_layers * per_layer

    # LM head: hidden_size -> vocab_size
    lm_head = 2 * h * spec.vocab_size

    return total + lm_head


def estimate_flops_prefill(spec: ModelSpec) -> int:
    """Rough FLOP count for the entire prefill phase.

    Prefill processes all tokens in parallel, so linear projection FLOPs
    scale with prefill_length. Attention FLOPs scale quadratically.
    """
    h = spec.hidden_size
    n_heads = spec.num_attention_heads
    n_kv = spec.num_kv_heads
    head_dim = spec.head_dim
    inter = spec.intermediate_size
    n_layers = spec.num_layers
    s = spec.prefill_length

    # Linear projections scale with seq_len
    q_flops = 2 * h * (n_heads * head_dim) * s
    k_flops = 2 * h * (n_kv * head_dim) * s
    v_flops = 2 * h * (n_kv * head_dim) * s
    o_flops = 2 * (n_heads * head_dim) * h * s
    mlp_flops = 3 * 2 * h * inter * s

    # Attention: quadratic in seq_len
    # QK^T: for each head, (s, d) x (d, s) = 2 * s * s * d
    attn_qk = 2 * n_heads * s * s * head_dim
    attn_sv = 2 * n_heads * s * s * head_dim

    per_layer = q_flops + k_flops + v_flops + o_flops + mlp_flops + attn_qk + attn_sv
    total = n_layers * per_layer

    lm_head = 2 * h * spec.vocab_size * s

    return total + lm_head


def analyze_compute(spec: ModelSpec, bandwidth_bytes_per_sec: float) -> ComputeProfile:
    """Full compute analysis with roofline note."""
    flops_tok = estimate_flops_per_token(spec)
    flops_pre = estimate_flops_prefill(spec)

    # Arithmetic intensity: FLOPs per byte of memory traffic for one decode token
    from .bandwidth import analyze_bandwidth

    bw_profile = analyze_bandwidth(spec)
    bytes_per_tok = bw_profile.total_bytes_per_token
    if bytes_per_tok > 0:
        ai = flops_tok / bytes_per_tok
    else:
        ai = float("inf")

    # Roofline classification
    if ai < 1.0:
        note = "Heavily memory-bandwidth-bound (AI < 1). Faster memory helps more than faster compute."
    elif ai < 10.0:
        note = "Memory-bandwidth-bound (AI < 10). Typical for LLM decode on most hardware."
    elif ai < 100.0:
        note = "Balanced region. Both compute and bandwidth matter."
    else:
        note = "Compute-bound. Rare for LLM decode unless very long context."

    return ComputeProfile(
        flops_per_token=flops_tok,
        flops_prefill=flops_pre,
        arithmetic_intensity=ai,
        roofline_note=note,
    )
