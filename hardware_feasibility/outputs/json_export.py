"""JSON export for machine-readable output."""

from __future__ import annotations

import json
from typing import Any, Optional

from ..models.architecture_rules import ModelSpec
from ..analysis.memory import MemoryProfile
from ..analysis.bandwidth import BandwidthProfile
from ..analysis.compute import ComputeProfile
from ..analysis.io import IOProfile
from ..analysis.verdict import VerdictResult
from ..hardware.matcher import MatchResult


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
