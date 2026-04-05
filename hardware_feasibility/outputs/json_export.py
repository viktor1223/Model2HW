"""JSON export for machine-readable output."""

from __future__ import annotations

import json
from typing import Any, Optional, TYPE_CHECKING

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


def _fmt(val: float, decimals: int = 2) -> float:
    return round(val, decimals)


def build_analysis_dict(
    spec: ModelSpec,
    mem: MemoryProfile,
    bw: BandwidthProfile,
    compute: ComputeProfile,
    io: IOProfile,
    verdict: VerdictResult,
    matches: Optional[list[MatchResult]] = None,
) -> dict[str, Any]:
    """Build a JSON-serializable dictionary of the full analysis."""
    result: dict[str, Any] = {
        "model": {
            "name": spec.name,
            "params": spec.params,
            "params_billions": _fmt(spec.params / 1e9),
            "num_layers": spec.num_layers,
            "hidden_size": spec.hidden_size,
            "num_attention_heads": spec.num_attention_heads,
            "num_kv_heads": spec.num_kv_heads,
            "head_dim": spec.head_dim,
            "intermediate_size": spec.intermediate_size,
            "vocab_size": spec.vocab_size,
        },
        "runtime_assumptions": {
            "weight_precision": spec.weight_precision.value,
            "kv_precision": spec.kv_precision.value,
            "batch_size": spec.batch_size,
            "context_length": spec.context_length,
            "prefill_length": spec.prefill_length,
            "decode_length": spec.decode_length,
            "target_tokens_per_sec": spec.target_tokens_per_sec,
            "kv_on_accelerator": spec.kv_on_accelerator,
        },
        "memory": {
            "weight_memory_gb": _fmt(mem.weight_gb),
            "kv_cache_gb": _fmt(mem.kv_cache_gb),
            "kv_cache_per_token_bytes": mem.kv_cache_per_token_bytes,
            "activation_buffer_gb": _fmt(mem.activation_buffer_gb),
            "total_working_memory_gb": _fmt(mem.total_gb),
        },
        "bandwidth": {
            "weight_bytes_per_token_gb": _fmt(bw.weight_bytes_per_token_gb, 3),
            "kv_read_bytes_per_token_gb": _fmt(bw.kv_read_bytes_per_token / (1 << 30), 3),
            "total_bytes_per_token_gb": _fmt(bw.total_bytes_per_token_gb, 3),
            "required_bandwidth_gbps": _fmt(bw.required_bandwidth_gbps, 1),
            "target_tokens_per_sec": bw.target_tokens_per_sec,
        },
        "compute": {
            "flops_per_token": compute.flops_per_token,
            "flops_per_token_gflops": _fmt(compute.flops_per_token / 1e9, 1),
            "flops_prefill": compute.flops_prefill,
            "flops_prefill_tflops": _fmt(compute.flops_prefill / 1e12, 2),
            "arithmetic_intensity": _fmt(compute.arithmetic_intensity, 2),
            "roofline_note": compute.roofline_note,
        },
        "host_io": {
            "input_embedding_bytes": io.input_embedding_bytes,
            "output_logits_bytes": io.output_logits_bytes,
            "kv_sync_bytes_per_token": io.kv_sync_bytes_per_token,
            "total_io_bytes_per_token": io.total_io_bytes_per_token,
            "required_io_bandwidth_gbps": _fmt(io.required_io_bandwidth_gbps, 4),
        },
        "verdict": {
            "verdict": verdict.verdict.value,
            "summary": verdict.summary,
            "details": verdict.details,
        },
    }

    if matches:
        result["hardware_matches"] = [
            {
                "board": m.board.name,
                "category": m.board.category,
                "memory_gb": m.board.memory_gb,
                "bandwidth_gbps": m.board.memory_bandwidth_gbps,
                "fits": m.fits,
                "verdict": m.verdict.verdict.value,
                "memory_utilization_pct": _fmt(m.memory_utilization * 100, 1),
                "bandwidth_utilization_pct": _fmt(m.bandwidth_utilization * 100, 1),
                "estimated_tok_per_sec": _fmt(m.estimated_tok_per_sec, 1)
                if m.estimated_tok_per_sec is not None
                else None,
                "details": m.verdict.details,
            }
            for m in matches
        ]

    return result


def export_json(
    spec: ModelSpec,
    mem: MemoryProfile,
    bw: BandwidthProfile,
    compute: ComputeProfile,
    io: IOProfile,
    verdict: VerdictResult,
    matches: Optional[list[MatchResult]] = None,
    *,
    indent: int = 2,
) -> str:
    """Return the full analysis as a JSON string."""
    d = build_analysis_dict(spec, mem, bw, compute, io, verdict, matches)
    return json.dumps(d, indent=indent)


def format_sweep_json(sweep: SweepResult) -> dict[str, Any]:
    """Build a JSON-serializable dictionary from a precision sweep result."""
    from ..analysis.sweep import SweepResult as _SR  # noqa: F811 - runtime import

    points_list = []
    for p in sweep.points:
        points_list.append({
            "weight_precision": p.weight_precision.value,
            "kv_precision": p.kv_precision.value,
            "fits": p.fits,
            "memory_total_gb": _fmt(p.memory.total_gb),
            "required_bandwidth_gbps": _fmt(p.bandwidth.required_bandwidth_gbps, 1),
            "memory_headroom_gb": _fmt(p.memory_headroom_gb, 2)
            if p.memory_headroom_gb is not None else None,
            "bandwidth_headroom_gbps": _fmt(p.bandwidth_headroom_gbps, 1)
            if p.bandwidth_headroom_gbps is not None else None,
            "verdict": p.verdict.verdict.value,
            "details": p.verdict.details,
        })

    best = sweep.best_fitting
    return {
        "model": sweep.base_spec.name,
        "target_memory_gb": sweep.target_memory_gb,
        "target_bandwidth_gbps": sweep.target_bandwidth_gbps,
        "target_link_bandwidth_gbps": sweep.target_link_bandwidth_gbps,
        "total_configurations": len(sweep.points),
        "fitting_configurations": len(sweep.fitting_points),
        "best_fitting": {
            "weight_precision": best.weight_precision.value,
            "kv_precision": best.kv_precision.value,
            "memory_total_gb": _fmt(best.memory.total_gb),
            "required_bandwidth_gbps": _fmt(best.bandwidth.required_bandwidth_gbps, 1),
        } if best is not None else None,
        "points": points_list,
    }


def format_sensitivity_json(result: SensitivityResult) -> dict[str, Any]:
    """Build a JSON-serializable dict from a sensitivity analysis result."""
    from ..analysis.sensitivity import SensitivityResult as _SR  # noqa: F811

    b = result.bottleneck
    return {
        "primary_bottleneck": b.primary_bottleneck,
        "utilization": {
            "memory": _fmt(b.memory_utilization, 3),
            "bandwidth": _fmt(b.bandwidth_utilization, 3),
            "host_io": _fmt(b.io_utilization, 3),
            "compute": _fmt(b.compute_utilization, 3),
        },
        "sensitivities": [
            {
                "parameter": s.parameter_name,
                "original_value": s.original_value,
                "modified_value": s.modified_value,
                "original_verdict": s.original_verdict,
                "modified_verdict": s.modified_verdict,
                "verdict_changed": s.verdict_changed,
            }
            for s in result.sensitivities
        ],
    }


def format_recommend_json(result: RecommendationResult) -> dict[str, Any]:
    """Build a JSON-serializable dict from a recommendation result."""
    from ..analysis.recommender import RecommendationResult as _RR  # noqa: F811

    recs = []
    for rec in result.recommendations:
        recs.append({
            "weight_precision": rec.weight_precision.value,
            "kv_precision": rec.kv_precision.value,
            "context_length": rec.context_length,
            "batch_size": rec.batch_size,
            "kv_on_accelerator": rec.kv_on_accelerator,
            "estimated_tok_per_sec": _fmt(rec.estimated_tok_per_sec, 1),
            "memory_utilization": _fmt(rec.memory_utilization, 3),
            "bandwidth_utilization": _fmt(rec.bandwidth_utilization, 3),
            "verdict": rec.verdict.verdict.value,
            "rationale": rec.rationale,
            "estimated_perplexity": _fmt(rec.estimated_perplexity, 2) if rec.estimated_perplexity is not None else None,
            "perplexity_source": rec.perplexity_source or None,
        })

    return {
        "model": result.model_name,
        "target_board": result.target_board,
        "infeasible_reason": result.infeasible_reason,
        "recommendations": recs,
    }


def format_decomposition_json(result: DecompositionResult) -> dict[str, Any]:
    """Build a JSON-serializable dict from a decomposition result."""
    from ..analysis.decomposition import DecompositionResult as _DR  # noqa: F811

    plans = []
    for plan in result.plans:
        devices = []
        for dev in plan.devices:
            devices.append({
                "device_name": dev.device_name,
                "layer_start": dev.layer_start,
                "layer_end": dev.layer_end,
                "weight_memory_gb": _fmt(dev.weight_memory_gb),
                "kv_cache_memory_gb": _fmt(dev.kv_cache_memory_gb),
                "total_memory_gb": _fmt(dev.total_memory_gb),
                "memory_utilization": _fmt(dev.memory_utilization, 3),
            })
        plans.append({
            "strategy": plan.strategy,
            "total_devices": plan.total_devices,
            "inter_device_transfer_bytes_per_token": plan.inter_device_transfer_bytes_per_token,
            "inter_device_bandwidth_required_gbps": _fmt(plan.inter_device_bandwidth_required_gbps, 4),
            "estimated_tok_per_sec": _fmt(plan.estimated_tok_per_sec, 1),
            "pipeline_bubble_fraction": _fmt(plan.pipeline_bubble_fraction, 4),
            "feasible": plan.feasible,
            "details": plan.details,
            "devices": devices,
        })

    return {
        "model": result.model_name,
        "single_device_feasible": result.single_device_feasible,
        "best_plan": result.best_plan.strategy if result.best_plan else None,
        "plans": plans,
    }


def format_pipeline_json(result: "PipelineResult") -> dict[str, Any]:
    """Build a JSON-serializable dict from a full pipeline result."""
    from ..pipeline import PipelineResult as _PR  # noqa: F811

    data: dict[str, Any] = {
        "model": result.model_name,
        "target_board": result.target_board,
        "pipeline_version": result.pipeline_version,
        "total_runtime_seconds": _fmt(result.total_runtime_seconds, 3),
        "initial_verdict": {
            "verdict": result.initial_verdict.verdict.value,
            "details": result.initial_verdict.details,
        },
    }

    if result.precision_sweep is not None:
        data["precision_sweep"] = format_sweep_json(result.precision_sweep)

    if result.decomposition is not None:
        data["decomposition"] = format_decomposition_json(result.decomposition)

    if result.recommendation is not None:
        data["recommendation"] = format_recommend_json(result.recommendation)

    if result.sensitivity is not None:
        data["sensitivity"] = format_sensitivity_json(result.sensitivity)

    return data
