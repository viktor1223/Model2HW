"""Sensitivity analysis: identify bottlenecks and quantify parameter sensitivity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..models.architecture_rules import ModelSpec, Precision
from .memory import MemoryProfile, analyze_memory
from .bandwidth import BandwidthProfile, analyze_bandwidth
from .compute import ComputeProfile, analyze_compute
from .io import IOProfile, analyze_io
from .verdict import FeasibilityVerdict, VerdictResult, render_verdict


# Precision ordering for "one step down" perturbation
_PRECISION_ORDER = [Precision.FP16, Precision.BF16, Precision.INT8, Precision.INT4]


def _next_lower_precision(p: Precision) -> Optional[Precision]:
    """Return the next lower precision, or None if already at INT4."""
    try:
        idx = _PRECISION_ORDER.index(p)
    except ValueError:
        return None
    if idx + 1 < len(_PRECISION_ORDER):
        return _PRECISION_ORDER[idx + 1]
    return None


@dataclass
class BottleneckBreakdown:
    """Quantifies how close each resource dimension is to the limit."""

    memory_utilization: float
    bandwidth_utilization: float
    io_utilization: float
    compute_utilization: float

    @property
    def primary_bottleneck(self) -> str:
        """Returns the name of the most constrained resource."""
        utils = {
            "memory": self.memory_utilization,
            "bandwidth": self.bandwidth_utilization,
            "host_io": self.io_utilization,
            "compute": self.compute_utilization,
        }
        return max(utils, key=utils.get)


@dataclass
class SensitivityPoint:
    """What happens if we change one parameter by a given delta."""

    parameter_name: str
    original_value: float
    modified_value: float
    original_verdict: str
    modified_verdict: str
    verdict_changed: bool


@dataclass
class SensitivityResult:
    """Full sensitivity analysis output."""

    bottleneck: BottleneckBreakdown
    sensitivities: list[SensitivityPoint]


def _run_verdict(
    spec: ModelSpec,
    target_memory_gb: float,
    target_bandwidth_gbps: float,
    target_link_bandwidth_gbps: float,
) -> VerdictResult:
    """Run full analysis pipeline and return verdict."""
    mem = analyze_memory(spec)
    bw = analyze_bandwidth(spec)
    compute = analyze_compute(spec, 0)
    io = analyze_io(spec)
    return render_verdict(
        spec, mem, bw, compute, io,
        target_memory_gb=target_memory_gb,
        target_bandwidth_gbps=target_bandwidth_gbps,
        target_link_bandwidth_gbps=target_link_bandwidth_gbps,
    )


def analyze_sensitivity(
    spec: ModelSpec,
    mem: MemoryProfile,
    bw: BandwidthProfile,
    compute: ComputeProfile,
    io: IOProfile,
    *,
    target_memory_gb: float,
    target_bandwidth_gbps: float,
    target_link_bandwidth_gbps: float,
    target_tops: float | None = None,
) -> SensitivityResult:
    """Identify bottlenecks and quantify sensitivity to parameter changes.

    Step 1: Compute bottleneck breakdown (utilization fractions).
    Step 2: Perturb hardware targets and model parameters, re-run verdict.
    """
    # --- Step 1: Bottleneck breakdown ---
    memory_util = mem.total_gb / target_memory_gb if target_memory_gb > 0 else 0.0
    bandwidth_util = bw.required_bandwidth_gbps / target_bandwidth_gbps if target_bandwidth_gbps > 0 else 0.0
    io_util = io.required_io_bandwidth_gbps / target_link_bandwidth_gbps if target_link_bandwidth_gbps > 0 else 0.0

    if target_tops is not None and target_tops > 0:
        peak_flops_per_sec = target_tops * 1e12
        required_flops_per_sec = compute.flops_per_token * spec.target_tokens_per_sec
        compute_util = required_flops_per_sec / peak_flops_per_sec
    else:
        compute_util = 0.0

    bottleneck = BottleneckBreakdown(
        memory_utilization=memory_util,
        bandwidth_utilization=bandwidth_util,
        io_utilization=io_util,
        compute_utilization=compute_util,
    )

    # --- Step 2: Perturbations ---
    original_verdict = render_verdict(
        spec, mem, bw, compute, io,
        target_memory_gb=target_memory_gb,
        target_bandwidth_gbps=target_bandwidth_gbps,
        target_link_bandwidth_gbps=target_link_bandwidth_gbps,
    ).verdict.value

    sensitivities: list[SensitivityPoint] = []

    # Hardware target perturbations (cheap: no re-analysis needed)
    for label, factor in [("+25%", 1.25), ("+50%", 1.50), ("-25%", 0.75)]:
        # Memory target
        new_mem = target_memory_gb * factor
        v = render_verdict(
            spec, mem, bw, compute, io,
            target_memory_gb=new_mem,
            target_bandwidth_gbps=target_bandwidth_gbps,
            target_link_bandwidth_gbps=target_link_bandwidth_gbps,
        ).verdict.value
        sensitivities.append(SensitivityPoint(
            parameter_name=f"target_memory_gb {label}",
            original_value=target_memory_gb,
            modified_value=round(new_mem, 2),
            original_verdict=original_verdict,
            modified_verdict=v,
            verdict_changed=(v != original_verdict),
        ))

        # Bandwidth target
        new_bw = target_bandwidth_gbps * factor
        v = render_verdict(
            spec, mem, bw, compute, io,
            target_memory_gb=target_memory_gb,
            target_bandwidth_gbps=new_bw,
            target_link_bandwidth_gbps=target_link_bandwidth_gbps,
        ).verdict.value
        sensitivities.append(SensitivityPoint(
            parameter_name=f"target_bandwidth_gbps {label}",
            original_value=target_bandwidth_gbps,
            modified_value=round(new_bw, 2),
            original_verdict=original_verdict,
            modified_verdict=v,
            verdict_changed=(v != original_verdict),
        ))

    # Model parameter perturbations (require re-analysis)
    for factor_label, factor in [("0.5x", 0.5), ("2x", 2), ("4x", 4)]:
        # Context length
        new_ctx = int(spec.context_length * factor)
        if new_ctx >= 64:  # minimum sensible context
            modified = replace(spec, context_length=new_ctx)
            v = _run_verdict(modified, target_memory_gb, target_bandwidth_gbps, target_link_bandwidth_gbps).verdict.value
            sensitivities.append(SensitivityPoint(
                parameter_name=f"context_length {factor_label}",
                original_value=spec.context_length,
                modified_value=new_ctx,
                original_verdict=original_verdict,
                modified_verdict=v,
                verdict_changed=(v != original_verdict),
            ))

    # Batch size perturbations
    for factor_label, factor in [("2x", 2), ("4x", 4)]:
        new_bs = spec.batch_size * factor
        modified = replace(spec, batch_size=new_bs)
        v = _run_verdict(modified, target_memory_gb, target_bandwidth_gbps, target_link_bandwidth_gbps).verdict.value
        sensitivities.append(SensitivityPoint(
            parameter_name=f"batch_size {factor_label}",
            original_value=spec.batch_size,
            modified_value=new_bs,
            original_verdict=original_verdict,
            modified_verdict=v,
            verdict_changed=(v != original_verdict),
        ))

    # Weight precision one step down
    next_wp = _next_lower_precision(spec.weight_precision)
    if next_wp is not None:
        modified = replace(spec, weight_precision=next_wp)
        v = _run_verdict(modified, target_memory_gb, target_bandwidth_gbps, target_link_bandwidth_gbps).verdict.value
        sensitivities.append(SensitivityPoint(
            parameter_name=f"weight_precision {spec.weight_precision.value}->{next_wp.value}",
            original_value=spec.weight_precision.bytes_per_element,
            modified_value=next_wp.bytes_per_element,
            original_verdict=original_verdict,
            modified_verdict=v,
            verdict_changed=(v != original_verdict),
        ))

    return SensitivityResult(
        bottleneck=bottleneck,
        sensitivities=sensitivities,
    )
