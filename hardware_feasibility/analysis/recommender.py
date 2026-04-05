"""Configuration recommender: find the best operating point for a model-board pair."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..models.architecture_rules import ModelSpec, Precision
from ..hardware.board_specs import BoardSpec
from ..evaluation.accuracy_db import get_perplexity
from .memory import MemoryProfile, analyze_memory, BYTES_PER_GB
from .bandwidth import BandwidthProfile, analyze_bandwidth
from .compute import ComputeProfile, analyze_compute
from .io import IOProfile, analyze_io
from .verdict import FeasibilityVerdict, VerdictResult, render_verdict
from .sweep import run_precision_sweep


# Context lengths to sweep (tokens)
_CONTEXT_CANDIDATES = [512, 1024, 2048, 4096, 8192, 16384, 32768]

# Batch sizes to sweep
_BATCH_CANDIDATES = [1, 2, 4, 8]


@dataclass
class Recommendation:
    """A recommended operating configuration."""

    weight_precision: Precision
    kv_precision: Precision
    context_length: int
    batch_size: int
    kv_on_accelerator: bool

    estimated_tok_per_sec: float
    memory_utilization: float
    bandwidth_utilization: float
    verdict: VerdictResult

    rationale: list[str]

    estimated_perplexity: float | None = None
    perplexity_source: str = ""  # "lookup", "estimated", or ""


@dataclass
class RecommendationResult:
    """Full recommendation output."""

    target_board: str
    model_name: str
    recommendations: list[Recommendation]
    infeasible_reason: str | None


def _estimate_tok_per_sec(bw: BandwidthProfile, board: BoardSpec) -> float:
    """Estimate achievable tok/s based on bandwidth-limited model."""
    if bw.total_bytes_per_token <= 0:
        return 0.0
    board_bw_bytes = board.memory_bandwidth_gbps * BYTES_PER_GB
    return board_bw_bytes / bw.total_bytes_per_token


def _build_rationale(
    rec_spec: ModelSpec,
    original_spec: ModelSpec,
    board: BoardSpec,
    mem: MemoryProfile,
) -> list[str]:
    """Generate deterministic rationale strings for a recommendation."""
    rationale: list[str] = []

    if rec_spec.weight_precision != Precision.FP16:
        fp16_spec = replace(original_spec, weight_precision=Precision.FP16)
        fp16_weight_gb = analyze_memory(fp16_spec).weight_gb
        rec_weight_gb = mem.weight_gb
        rationale.append(
            f"{rec_spec.weight_precision.value.upper()} quantization reduces weight memory from "
            f"{fp16_weight_gb:.1f} GB to {rec_weight_gb:.1f} GB."
        )

    if rec_spec.kv_precision != rec_spec.weight_precision:
        rationale.append(
            f"KV cache uses {rec_spec.kv_precision.value} (vs {rec_spec.weight_precision.value} weights) "
            f"to balance memory and cache quality."
        )

    if rec_spec.context_length < original_spec.context_length:
        rationale.append(
            f"Context reduced from {original_spec.context_length:,} to {rec_spec.context_length:,} "
            f"to fit KV cache within memory budget."
        )

    if rec_spec.batch_size > 1:
        rationale.append(
            f"Batch size {rec_spec.batch_size} increases total throughput at the cost of "
            f"higher memory and per-request latency."
        )

    if not rec_spec.kv_on_accelerator:
        rationale.append(
            "KV cache offloaded to host memory to free device capacity for weights."
        )

    if not rationale:
        rationale.append("Default configuration fits the hardware target.")

    return rationale


def recommend_configuration(
    spec: ModelSpec,
    board: BoardSpec,
) -> RecommendationResult:
    """Recommend the best operating configurations for a model-board pair.

    Algorithm:
    1. Run precision sweep to find fitting precision combos.
    2. For each, sweep context lengths and batch sizes.
    3. Optionally try KV offload for tight fits.
    4. Rank by estimated tok/s.
    5. Generate rationale strings.
    """
    # Step 1: Precision sweep
    sweep = run_precision_sweep(
        spec,
        target_memory_gb=board.memory_gb,
        target_bandwidth_gbps=board.memory_bandwidth_gbps,
        target_link_bandwidth_gbps=board.host_link_bandwidth_gbps,
    )

    if not sweep.fitting_points:
        return RecommendationResult(
            target_board=board.name,
            model_name=spec.name,
            recommendations=[],
            infeasible_reason=(
                f"No precision configuration fits on {board.name} "
                f"({board.memory_gb} GB). Minimum memory needed at INT4: "
                f"{min(p.memory.total_gb for p in sweep.points):.2f} GB."
            ),
        )

    candidates: list[Recommendation] = []

    # Step 2-4: For each fitting precision, sweep context and batch
    for point in sweep.fitting_points:
        wp = point.weight_precision
        kp = point.kv_precision

        ctx_candidates = [c for c in _CONTEXT_CANDIDATES if c <= spec.context_length]
        if spec.context_length not in ctx_candidates:
            ctx_candidates.append(spec.context_length)

        for ctx in ctx_candidates:
            for bs in _BATCH_CANDIDATES:
                for kv_on_accel in [True, False]:
                    modified = replace(
                        spec,
                        weight_precision=wp,
                        kv_precision=kp,
                        context_length=ctx,
                        batch_size=bs,
                        kv_on_accelerator=kv_on_accel,
                    )

                    mem = analyze_memory(modified)
                    bw = analyze_bandwidth(modified)
                    compute = analyze_compute(modified, 0)
                    io = analyze_io(modified)
                    verdict = render_verdict(
                        modified, mem, bw, compute, io,
                        target_memory_gb=board.memory_gb,
                        target_bandwidth_gbps=board.memory_bandwidth_gbps,
                        target_link_bandwidth_gbps=board.host_link_bandwidth_gbps,
                    )

                    if verdict.verdict == FeasibilityVerdict.DOES_NOT_FIT:
                        continue

                    est_tok_s = _estimate_tok_per_sec(bw, board)
                    mem_util = mem.total_gb / board.memory_gb if board.memory_gb > 0 else 0.0
                    bw_util = bw.required_bandwidth_gbps / board.memory_bandwidth_gbps if board.memory_bandwidth_gbps > 0 else 0.0

                    rationale = _build_rationale(modified, spec, board, mem)

                    ppl, ppl_source = get_perplexity(spec.name, wp, kp)

                    candidates.append(Recommendation(
                        weight_precision=wp,
                        kv_precision=kp,
                        context_length=ctx,
                        batch_size=bs,
                        kv_on_accelerator=kv_on_accel,
                        estimated_tok_per_sec=est_tok_s,
                        memory_utilization=mem_util,
                        bandwidth_utilization=bw_util,
                        verdict=verdict,
                        rationale=rationale,
                        estimated_perplexity=ppl,
                        perplexity_source=ppl_source,
                    ))

    # Step 5: Sort by estimated tok/s descending
    candidates.sort(key=lambda r: -r.estimated_tok_per_sec)

    return RecommendationResult(
        target_board=board.name,
        model_name=spec.name,
        recommendations=candidates,
        infeasible_reason=None,
    )
