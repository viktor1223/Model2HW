"""Human-readable report generation."""

from __future__ import annotations

from typing import Optional

from ..models.architecture_rules import ModelSpec
from ..analysis.memory import MemoryProfile
from ..analysis.bandwidth import BandwidthProfile
from ..analysis.compute import ComputeProfile
from ..analysis.io import IOProfile
from ..analysis.verdict import VerdictResult
from ..hardware.matcher import MatchResult


def _hr() -> str:
    return "-" * 72


def _section(title: str) -> str:
    return f"\n{_hr()}\n  {title}\n{_hr()}"


def _gb(val: float) -> str:
    return f"{val:.2f} GB"


def _params(val: int) -> str:
    if val >= 1e9:
        return f"{val / 1e9:.2f}B"
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    return str(val)


def generate_report(
    spec: ModelSpec,
    mem: MemoryProfile,
    bw: BandwidthProfile,
    compute: ComputeProfile,
    io: IOProfile,
    verdict: VerdictResult,
    matches: Optional[list[MatchResult]] = None,
) -> str:
    """Generate a full human-readable feasibility report."""
    lines: list[str] = []
    w = lines.append

    w("=" * 72)
    w("  HARDWARE FEASIBILITY REPORT")
    w(f"  Model: {spec.name}")
    w("=" * 72)

    # --- Model Architecture ---
    w(_section("MODEL ARCHITECTURE"))
    w(f"  Parameters:          {_params(spec.params)} ({spec.params:,})")
    w(f"  Layers:              {spec.num_layers}")
    w(f"  Hidden size:         {spec.hidden_size}")
    w(f"  Attention heads:     {spec.num_attention_heads}")
    w(f"  KV heads:            {spec.num_kv_heads} {'(GQA)' if spec.num_kv_heads < spec.num_attention_heads else '(MHA)'}")
    w(f"  Head dim:            {spec.head_dim}")
    w(f"  Intermediate size:   {spec.intermediate_size}")
    w(f"  Vocab size:          {spec.vocab_size:,}")

    # --- Runtime Assumptions ---
    w(_section("RUNTIME ASSUMPTIONS"))
    w(f"  Weight precision:    {spec.weight_precision.value}")
    w(f"  KV cache precision:  {spec.kv_precision.value}")
    w(f"  Batch size:          {spec.batch_size}")
    w(f"  Context length:      {spec.context_length:,}")
    w(f"  Prefill length:      {spec.prefill_length:,}")
    w(f"  Decode length:       {spec.decode_length:,}")
    w(f"  Target tok/s:        {spec.target_tokens_per_sec}")
    w(f"  KV on accelerator:   {'Yes' if spec.kv_on_accelerator else 'No (host memory)'}")

    # --- Memory Analysis ---
    w(_section("MEMORY ANALYSIS"))
    w(f"  Weight memory:       {_gb(mem.weight_gb)}")
    w(f"  KV cache (@{spec.context_length} ctx): {_gb(mem.kv_cache_gb)}")
    w(f"  KV per token:        {mem.kv_cache_per_token_bytes:,} bytes")
    w(f"  Activation buffers:  {_gb(mem.activation_buffer_gb)}")
    w(f"  --------------------------------")
    w(f"  TOTAL working memory:{_gb(mem.total_gb)}")

    # --- Bandwidth Analysis ---
    w(_section("BANDWIDTH ANALYSIS (DECODE)"))
    w(f"  Weights streamed/token:  {_gb(bw.weight_bytes_per_token_gb)}")
    w(f"  KV read/token (avg):     {bw.kv_read_bytes_per_token / (1 << 30):.3f} GB")
    w(f"  Total bytes/token:       {_gb(bw.total_bytes_per_token_gb)}")
    w(f"  Required for {spec.target_tokens_per_sec:.0f} tok/s:    {bw.required_bandwidth_gbps:.1f} GB/s")

    # --- Compute Analysis ---
    w(_section("COMPUTE ANALYSIS"))
    w(f"  FLOPs/token (decode):    {compute.flops_per_token / 1e9:.1f} GFLOPs")
    w(f"  FLOPs prefill total:     {compute.flops_prefill / 1e12:.2f} TFLOPs")
    w(f"  Arithmetic intensity:    {compute.arithmetic_intensity:.2f} FLOPs/byte")
    w(f"  Roofline:                {compute.roofline_note}")

    # --- Host IO Analysis ---
    w(_section("HOST-DEVICE IO"))
    w(f"  Input embedding/token:   {io.input_embedding_bytes:,} bytes")
    w(f"  Output logits/token:     {io.output_logits_bytes:,} bytes")
    if io.kv_sync_bytes_per_token > 0:
        w(f"  KV sync/token (off-chip):{io.kv_sync_bytes_per_token:,} bytes")
    w(f"  Total IO/token:          {io.total_io_bytes_per_token:,} bytes")
    w(f"  Required IO bandwidth:   {io.required_io_bandwidth_gbps:.4f} GB/s")

    # --- Verdict ---
    w(_section("VERDICT"))
    verdict_label = verdict.verdict.value.replace("_", " ").upper()
    w(f"  >> {verdict_label} <<")
    w("")
    for detail in verdict.details:
        w(f"  - {detail}")

    # --- Hardware Matches ---
    if matches:
        w(_section("HARDWARE MATCH RANKING"))
        for i, m in enumerate(matches, 1):
            status = "FITS" if m.fits else "NO FIT"
            tok_s = f"{m.estimated_tok_per_sec:.1f} tok/s" if m.estimated_tok_per_sec else "N/A"
            w(f"\n  {i}. {m.board.name} [{m.board.category}] - {status}")
            w(f"     Memory: {m.board.memory_gb} GB | BW: {m.board.memory_bandwidth_gbps} GB/s | Link: {m.board.host_link_bandwidth_gbps} GB/s")
            w(f"     Memory util: {m.memory_utilization * 100:.0f}% | BW util: {m.bandwidth_utilization * 100:.0f}%")
            w(f"     Est. decode speed: {tok_s}")
            w(f"     Verdict: {m.verdict.verdict.value.replace('_', ' ')}")
            for d in m.verdict.details:
                w(f"       - {d}")

    w("")
    w("=" * 72)
    w("  NOTE: These are static estimates based on model architecture.")
    w("  Real performance depends on kernel implementation, memory layout,")
    w("  operator fusion, and runtime scheduling. Use as feasibility bands.")
    w("=" * 72)

    return "\n".join(lines)
