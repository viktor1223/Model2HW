"""Precision sweep: enumerate precision configurations and rank by feasibility."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..models.architecture_rules import ModelSpec, Precision
from .memory import MemoryProfile, analyze_memory
from .bandwidth import BandwidthProfile, analyze_bandwidth
from .compute import ComputeProfile, analyze_compute
from .io import IOProfile, analyze_io
from .verdict import FeasibilityVerdict, VerdictResult, render_verdict


@dataclass
class SweepPoint:
    """One configuration point in the precision sweep."""

    weight_precision: Precision
    kv_precision: Precision
    memory: MemoryProfile
    bandwidth: BandwidthProfile
    compute: ComputeProfile
    io: IOProfile
    verdict: VerdictResult

    @property
    def fits(self) -> bool:
        return self.verdict.verdict != FeasibilityVerdict.DOES_NOT_FIT

    @property
    def memory_headroom_gb(self) -> Optional[float]:
        """Positive = spare memory. Negative = overshoot. None = no target."""
        if self.verdict.target_memory_gb is None:
            return None
        return self.verdict.target_memory_gb - self.memory.total_gb

    @property
    def bandwidth_headroom_gbps(self) -> Optional[float]:
        if self.verdict.target_bandwidth_gbps is None:
            return None
        return self.verdict.target_bandwidth_gbps - self.bandwidth.required_bandwidth_gbps


@dataclass
class SweepResult:
    """Full result of a precision sweep."""

    base_spec: ModelSpec
    target_memory_gb: Optional[float]
    target_bandwidth_gbps: Optional[float]
    target_link_bandwidth_gbps: Optional[float]
    points: list[SweepPoint]

    @property
    def fitting_points(self) -> list[SweepPoint]:
        return [p for p in self.points if p.fits]

    @property
    def best_fitting(self) -> Optional[SweepPoint]:
        """Best fitting point: fits + highest bandwidth headroom (most throughput margin)."""
        fitting = self.fitting_points
        if not fitting:
            return None
        return max(fitting, key=lambda p: p.bandwidth_headroom_gbps or 0)


# Precision combinations to sweep. Order matters: leftmost = highest quality.
WEIGHT_PRECISIONS = [Precision.FP16, Precision.BF16, Precision.INT8, Precision.INT4]
KV_PRECISIONS = [Precision.FP16, Precision.INT8, Precision.INT4]


def run_precision_sweep(
    spec: ModelSpec,
    *,
    target_memory_gb: Optional[float] = None,
    target_bandwidth_gbps: Optional[float] = None,
    target_link_bandwidth_gbps: Optional[float] = None,
    weight_precisions: list[Precision] | None = None,
    kv_precisions: list[Precision] | None = None,
) -> SweepResult:
    """Enumerate precision combinations and rank by feasibility.

    For each (weight_precision, kv_precision) pair:
    1. Create a modified ModelSpec with those precisions
    2. Run the full analysis pipeline
    3. Render a verdict against hardware targets
    4. Collect into SweepResult

    The result is sorted: fitting points first (by bandwidth headroom descending),
    then non-fitting points (by memory overshoot ascending, i.e. closest to fitting first).
    """
    w_precs = weight_precisions or WEIGHT_PRECISIONS
    kv_precs = kv_precisions or KV_PRECISIONS
    points: list[SweepPoint] = []

    for wp in w_precs:
        for kp in kv_precs:
            # KV precision higher than weight precision makes no engineering sense
            if kp.bytes_per_element > wp.bytes_per_element:
                continue

            modified = replace(spec, weight_precision=wp, kv_precision=kp)
            mem = analyze_memory(modified)
            bw = analyze_bandwidth(modified)
            compute = analyze_compute(modified, 0)
            io = analyze_io(modified)
            verdict = render_verdict(
                modified, mem, bw, compute, io,
                target_memory_gb=target_memory_gb,
                target_bandwidth_gbps=target_bandwidth_gbps,
                target_link_bandwidth_gbps=target_link_bandwidth_gbps,
            )
            points.append(SweepPoint(
                weight_precision=wp,
                kv_precision=kp,
                memory=mem,
                bandwidth=bw,
                compute=compute,
                io=io,
                verdict=verdict,
            ))

    # Sort: fitting first (sorted by BW headroom desc), then non-fitting (sorted by memory gap asc)
    def sort_key(p: SweepPoint) -> tuple:
        fits = 0 if p.fits else 1
        bw_head = -(p.bandwidth_headroom_gbps or 0)
        mem_head = -(p.memory_headroom_gb or -1e9)
        return (fits, bw_head, mem_head)

    points.sort(key=sort_key)

    return SweepResult(
        base_spec=spec,
        target_memory_gb=target_memory_gb,
        target_bandwidth_gbps=target_bandwidth_gbps,
        target_link_bandwidth_gbps=target_link_bandwidth_gbps,
        points=points,
    )
