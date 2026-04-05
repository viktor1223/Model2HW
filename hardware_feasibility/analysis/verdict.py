"""Verdict engine: synthesizes all analysis into a feasibility verdict."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..models.architecture_rules import ModelSpec
from .memory import MemoryProfile
from .bandwidth import BandwidthProfile
from .compute import ComputeProfile
from .io import IOProfile


class FeasibilityVerdict(Enum):
    FITS_COMFORTABLY = "fits_comfortably"
    FITS_BANDWIDTH_LIMITED = "fits_but_bandwidth_limited"
    DOES_NOT_FIT = "does_not_fit"
    HOST_LINK_BOTTLENECK = "host_link_likely_bottleneck"


@dataclass
class VerdictResult:
    verdict: FeasibilityVerdict
    summary: str
    details: list[str]

    # Optional: verdict against a specific hardware target
    target_memory_gb: Optional[float] = None
    target_bandwidth_gbps: Optional[float] = None
    target_link_bandwidth_gbps: Optional[float] = None


def render_verdict(
    spec: ModelSpec,
    mem: MemoryProfile,
    bw: BandwidthProfile,
    compute: ComputeProfile,
    io: IOProfile,
    *,
    target_memory_gb: Optional[float] = None,
    target_bandwidth_gbps: Optional[float] = None,
    target_link_bandwidth_gbps: Optional[float] = None,
) -> VerdictResult:
    """Produce a feasibility verdict from analysis profiles.

    If hardware targets are provided, the verdict is specific.
    Otherwise it describes the model's demands generically.
    """
    details: list[str] = []
    verdict = FeasibilityVerdict.FITS_COMFORTABLY

    # --- Memory check ---
    if target_memory_gb is not None:
        headroom = target_memory_gb - mem.total_gb
        if headroom < 0:
            verdict = FeasibilityVerdict.DOES_NOT_FIT
            details.append(
                f"Total working memory ({mem.total_gb:.2f} GB) exceeds "
                f"target capacity ({target_memory_gb:.1f} GB) by {-headroom:.2f} GB."
            )
        elif headroom < mem.total_gb * 0.15:
            details.append(
                f"Memory is tight: {mem.total_gb:.2f} GB used of {target_memory_gb:.1f} GB "
                f"({headroom:.2f} GB headroom). OS/runtime overhead may push over."
            )
        else:
            details.append(
                f"Memory fits: {mem.total_gb:.2f} GB used of {target_memory_gb:.1f} GB "
                f"({headroom:.2f} GB headroom)."
            )

    # --- Bandwidth check ---
    if target_bandwidth_gbps is not None and verdict != FeasibilityVerdict.DOES_NOT_FIT:
        if bw.required_bandwidth_gbps > target_bandwidth_gbps:
            if verdict == FeasibilityVerdict.FITS_COMFORTABLY:
                verdict = FeasibilityVerdict.FITS_BANDWIDTH_LIMITED
            details.append(
                f"Bandwidth-limited: need {bw.required_bandwidth_gbps:.1f} GB/s "
                f"for {spec.target_tokens_per_sec:.0f} tok/s but target provides "
                f"{target_bandwidth_gbps:.1f} GB/s. "
                f"Achievable: ~{target_bandwidth_gbps / (bw.required_bandwidth_gbps / spec.target_tokens_per_sec):.1f} tok/s."
            )
        else:
            utilization = bw.required_bandwidth_gbps / target_bandwidth_gbps * 100
            details.append(
                f"Bandwidth sufficient: {bw.required_bandwidth_gbps:.1f} GB/s needed, "
                f"{target_bandwidth_gbps:.1f} GB/s available ({utilization:.0f}% utilization)."
            )

    # --- Host link check ---
    if target_link_bandwidth_gbps is not None and verdict != FeasibilityVerdict.DOES_NOT_FIT:
        if io.required_io_bandwidth_gbps > target_link_bandwidth_gbps:
            verdict = FeasibilityVerdict.HOST_LINK_BOTTLENECK
            details.append(
                f"Host link bottleneck: need {io.required_io_bandwidth_gbps:.2f} GB/s "
                f"over host link but only {target_link_bandwidth_gbps:.1f} GB/s available."
            )
        else:
            details.append(
                f"Host link OK: {io.required_io_bandwidth_gbps:.4f} GB/s needed, "
                f"{target_link_bandwidth_gbps:.1f} GB/s available."
            )

    # --- Generic summary if no targets given ---
    if target_memory_gb is None and target_bandwidth_gbps is None:
        details.append(f"Total working memory: {mem.total_gb:.2f} GB")
        details.append(f"Required bandwidth for {spec.target_tokens_per_sec:.0f} tok/s: {bw.required_bandwidth_gbps:.1f} GB/s")
        details.append(f"Arithmetic intensity: {compute.arithmetic_intensity:.2f} FLOPs/byte")
        details.append(compute.roofline_note)

        # Rough categorization
        if mem.total_gb <= 2:
            details.append("Likely feasible on small edge accelerators (2-4 GB class).")
        elif mem.total_gb <= 8:
            details.append("Requires a mid-range accelerator (8 GB class) or high-bandwidth FPGA board.")
        elif mem.total_gb <= 24:
            details.append("Requires a substantial accelerator (16-24 GB class).")
        else:
            details.append("Requires high-end or multi-device setup.")

    summary_lines = [details[0]] if details else ["Analysis complete."]
    summary = summary_lines[0]

    return VerdictResult(
        verdict=verdict,
        summary=summary,
        details=details,
        target_memory_gb=target_memory_gb,
        target_bandwidth_gbps=target_bandwidth_gbps,
        target_link_bandwidth_gbps=target_link_bandwidth_gbps,
    )
