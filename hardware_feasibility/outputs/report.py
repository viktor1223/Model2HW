"""Human-readable report generation."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ..models.architecture_rules import ModelSpec
from ..analysis.memory import MemoryProfile
from ..analysis.bandwidth import BandwidthProfile
from ..analysis.compute import ComputeProfile
from ..analysis.io import IOProfile
from ..analysis.verdict import VerdictResult
from ..hardware.matcher import MatchResult

if TYPE_CHECKING:
    from ..analysis.sweep import SweepResult
    from ..analysis.sensitivity import SensitivityResult
    from ..analysis.recommender import RecommendationResult
    from ..analysis.decomposition import DecompositionResult


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


def format_sweep_report(sweep: SweepResult) -> str:
    """Generate a human-readable precision sweep report."""
    from ..analysis.sweep import SweepResult as _SR  # noqa: F811 - runtime import

    lines: list[str] = []
    w = lines.append

    target_desc = ""
    if sweep.target_memory_gb is not None:
        target_desc += f"{sweep.target_memory_gb:.1f} GB"
    if sweep.target_bandwidth_gbps is not None:
        if target_desc:
            target_desc += ", "
        target_desc += f"{sweep.target_bandwidth_gbps:.1f} GB/s"
    if not target_desc:
        target_desc = "no hardware target"

    w("=" * 72)
    w("  PRECISION SWEEP RESULTS")
    w(f"  Model: {sweep.base_spec.name}  |  Target: {target_desc}")
    w("=" * 72)
    w("")
    w(f"  {'#':>3}  {'Weight':<8} {'KV':<8} {'Mem (GB)':>10} {'BW (GB/s)':>10}  {'Verdict':<25} {'Headroom':>10}")
    w(f"  {'---':>3}  {'------':<8} {'------':<8} {'--------':>10} {'---------':>10}  {'-------':<25} {'--------':>10}")

    for i, p in enumerate(sweep.points, 1):
        verdict_label = p.verdict.verdict.value.replace("_", " ")
        mem_head = f"{p.memory_headroom_gb:+.2f}" if p.memory_headroom_gb is not None else "-"
        w(f"  {i:3d}  {p.weight_precision.value:<8} {p.kv_precision.value:<8} "
          f"{p.memory.total_gb:10.2f} {p.bandwidth.required_bandwidth_gbps:10.1f}  "
          f"{verdict_label:<25} {mem_head:>10}")

    fitting_count = len(sweep.fitting_points)
    total_count = len(sweep.points)
    w("")
    w(f"  {fitting_count} of {total_count} configurations fit the target.")

    best = sweep.best_fitting
    if best is not None:
        w(f"  Best: {best.weight_precision.value} weights / {best.kv_precision.value} KV "
          f"({best.memory.total_gb:.2f} GB, {best.bandwidth.required_bandwidth_gbps:.1f} GB/s)")
    else:
        w("  No configuration fits the target hardware.")

    w("")
    w("=" * 72)

    return "\n".join(lines)


def format_sensitivity_report(result: SensitivityResult) -> str:
    """Generate a human-readable sensitivity analysis report."""
    from ..analysis.sensitivity import SensitivityResult as _SR  # noqa: F811

    lines: list[str] = []
    w = lines.append

    b = result.bottleneck
    w("=" * 72)
    w("  SENSITIVITY ANALYSIS")
    w("=" * 72)
    w("")
    w(f"  Primary bottleneck: {b.primary_bottleneck.upper()} "
      f"({max(b.memory_utilization, b.bandwidth_utilization, b.io_utilization, b.compute_utilization) * 100:.0f}% of available)")
    w("")
    w("  Resource utilization:")

    def _status(util: float) -> str:
        if util > 1.0:
            return "[BOTTLENECK]"
        if util > 0.85:
            return "[TIGHT]"
        return "[OK]"

    w(f"    Memory:     {b.memory_utilization * 100:5.0f}%  {_status(b.memory_utilization)}")
    w(f"    Bandwidth:  {b.bandwidth_utilization * 100:5.0f}%  {_status(b.bandwidth_utilization)}")
    w(f"    Host IO:    {b.io_utilization * 100:5.0f}%  {_status(b.io_utilization)}")
    w(f"    Compute:    {b.compute_utilization * 100:5.0f}%  {_status(b.compute_utilization)}")

    w("")
    w("  What-if scenarios:")
    for s in result.sensitivities:
        changed = "** VERDICT CHANGED **" if s.verdict_changed else "no change"
        verdict_str = s.modified_verdict.replace('_', ' ')
        w(f"    {s.parameter_name:<40} -> {verdict_str:<30} ({changed})")

    w("")
    w("=" * 72)

    return "\n".join(lines)


def format_recommend_report(result: RecommendationResult) -> str:
    """Generate a human-readable recommendation report."""
    from ..analysis.recommender import RecommendationResult as _RR  # noqa: F811

    lines: list[str] = []
    w = lines.append

    w("=" * 72)
    w("  CONFIGURATION RECOMMENDATIONS")
    w(f"  Model: {result.model_name}  |  Board: {result.target_board}")
    w("=" * 72)

    if result.infeasible_reason:
        w("")
        w(f"  INFEASIBLE: {result.infeasible_reason}")
        w("")
        w("=" * 72)
        return "\n".join(lines)

    for i, rec in enumerate(result.recommendations[:5], 1):
        w("")
        verdict_label = rec.verdict.verdict.value.replace('_', ' ')
        w(f"  #{i}: {rec.weight_precision.value} weights / {rec.kv_precision.value} KV | "
          f"ctx={rec.context_length} | batch={rec.batch_size} | "
          f"kv_on_device={'yes' if rec.kv_on_accelerator else 'no'}")
        w(f"      Est. tok/s: {rec.estimated_tok_per_sec:.1f} | "
          f"Memory: {rec.memory_utilization * 100:.0f}% | "
          f"BW: {rec.bandwidth_utilization * 100:.0f}% | "
          f"Verdict: {verdict_label}")
        if rec.estimated_perplexity is not None:
            w(f"      Perplexity: {rec.estimated_perplexity:.2f} ({rec.perplexity_source})")
        for r in rec.rationale:
            w(f"      - {r}")

    w("")
    w("=" * 72)

    return "\n".join(lines)


def format_decomposition_report(result: DecompositionResult) -> str:
    """Generate a human-readable decomposition report."""
    from ..analysis.decomposition import DecompositionResult as _DR  # noqa: F811

    lines: list[str] = []
    w = lines.append

    w("=" * 72)
    w("  DECOMPOSITION ANALYSIS")
    w(f"  Model: {result.model_name}")
    w("=" * 72)

    if result.single_device_feasible:
        w("")
        w("  Model fits on a single device. No decomposition needed.")
        w("")
        w("=" * 72)
        return "\n".join(lines)

    if not result.plans:
        w("")
        w("  No feasible decomposition plan found.")
        w("")
        w("=" * 72)
        return "\n".join(lines)

    for i, plan in enumerate(result.plans, 1):
        status = "FEASIBLE" if plan.feasible else "INFEASIBLE"
        w("")
        w(f"  Plan #{i}: {plan.strategy} ({plan.total_devices} devices) - {status}")
        w(f"    Est. tok/s: {plan.estimated_tok_per_sec:.1f} | "
          f"Bubble: {plan.pipeline_bubble_fraction * 100:.1f}% | "
          f"Inter-device: {plan.inter_device_transfer_bytes_per_token:,} bytes/token")
        for dev in plan.devices:
            w(f"    [{dev.device_name}] layers {dev.layer_start}-{dev.layer_end - 1}: "
              f"{dev.total_memory_gb:.2f} GB ({dev.memory_utilization * 100:.0f}% util)")
        for d in plan.details:
            w(f"    - {d}")

    if result.best_plan:
        w("")
        w(f"  >> Best: {result.best_plan.strategy} "
          f"({result.best_plan.total_devices} devices, "
          f"{result.best_plan.estimated_tok_per_sec:.1f} tok/s)")

    w("")
    w("=" * 72)

    return "\n".join(lines)


def format_pipeline_report(result: "PipelineResult") -> str:
    """Generate a comprehensive pipeline report combining all stage outputs."""
    from ..pipeline import PipelineResult as _PR  # noqa: F811

    lines: list[str] = []
    w = lines.append

    w("=" * 72)
    w("  FULL PIPELINE REPORT")
    w(f"  Model: {result.model_name}  |  Board: {result.target_board}")
    w("=" * 72)

    # Stage 1: Initial verdict
    w("")
    w("  STAGE 1: Initial Feasibility")
    w(f"    Verdict: {result.initial_verdict.verdict.value}")
    for d in result.initial_verdict.details:
        w(f"    - {d}")

    # Stage 1b: Precision sweep
    if result.precision_sweep is not None:
        w("")
        w("  STAGE 1b: Precision Sweep")
        fitting = result.precision_sweep.fitting_points
        if fitting:
            w(f"    {len(fitting)} fitting configuration(s)")
            best = fitting[0]
            w(f"    Best: {best.weight_precision.value}/{best.kv_precision.value}")
        else:
            w("    No single-device configuration fits.")

    # Stage 1c: Decomposition
    if result.decomposition is not None:
        w("")
        w("  STAGE 1c: Decomposition")
        if result.decomposition.plans:
            best = result.decomposition.best_plan
            if best:
                w(f"    Best plan: {best.strategy} ({best.total_devices} devices)")
        else:
            w("    No feasible decomposition found.")

    # Stage 2: Recommendation
    if result.recommendation is not None:
        w("")
        w("  STAGE 2: Configuration Recommendation")
        recs = result.recommendation.recommendations
        if recs:
            top = recs[0]
            w(f"    Top config: {top.weight_precision.value}/{top.kv_precision.value} "
              f"ctx={top.context_length} batch={top.batch_size}")
            w(f"    Est. tok/s: {top.estimated_tok_per_sec:.1f}")
            if top.estimated_perplexity is not None:
                w(f"    Perplexity: {top.estimated_perplexity:.2f} ({top.perplexity_source})")
            w(f"    Total configs evaluated: {len(recs)}")
        elif result.recommendation.infeasible_reason:
            w(f"    Infeasible: {result.recommendation.infeasible_reason}")

    # Stage 3: Sensitivity
    if result.sensitivity is not None:
        w("")
        w("  STAGE 3: Sensitivity Analysis")
        bd = result.sensitivity.bottleneck
        w(f"    Primary bottleneck: {bd.primary_bottleneck}")
        w(f"    Memory: {bd.memory_utilization * 100:.0f}% | "
          f"BW: {bd.bandwidth_utilization * 100:.0f}% | "
          f"Compute: {bd.compute_utilization * 100:.0f}% | "
          f"IO: {bd.io_utilization * 100:.0f}%")

    w("")
    w(f"  Pipeline completed in {result.total_runtime_seconds:.2f}s")
    w("=" * 72)

    return "\n".join(lines)
