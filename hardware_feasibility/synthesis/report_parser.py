"""Parse Vitis HLS synthesis report XML into structured data."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from .types import HLSSynthesisResult


def _safe_int(text: Optional[str]) -> Optional[int]:
    """Parse an integer from XML text, returning None on failure."""
    if text is None:
        return None
    try:
        return int(text)
    except (ValueError, TypeError):
        return None


def _safe_float(text: Optional[str]) -> Optional[float]:
    """Parse a float from XML text, returning None on failure."""
    if text is None:
        return None
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _find_text(root: ET.Element, *paths: str) -> Optional[str]:
    """Try multiple XPath patterns and return the first match's text."""
    for path in paths:
        elem = root.find(path)
        if elem is not None and elem.text is not None:
            return elem.text.strip()
    return None


def parse_synthesis_report(report_path: Path) -> HLSSynthesisResult:
    """Parse a Vitis HLS synthesis report XML file.

    Supports both Vitis HLS 2022.x and 2024.x XML formats.
    The parser tries multiple XPath patterns for each field to handle
    version differences gracefully.
    """
    raw_text = report_path.read_text()

    try:
        tree = ET.parse(report_path)
    except ET.ParseError as e:
        return HLSSynthesisResult(
            success=False,
            error_message=f"XML parse error: {e}",
            raw_report=raw_text,
        )

    root = tree.getroot()

    # --- Timing ---
    clock_target = _safe_float(_find_text(
        root,
        ".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod",
        ".//TimingReport/EstimatedClockPeriod",
        ".//Timing/EstimatedClockPeriod",
    ))
    clock_period = _safe_float(_find_text(
        root,
        ".//PerformanceEstimates/SummaryOfTimingAnalysis/AchievedClockPeriod",
        ".//TimingReport/AchievedClockPeriod",
        ".//Timing/AchievedClockPeriod",
    ))
    # If only target is found, also check for the <TargetClockPeriod> field
    if clock_target is None:
        clock_target = _safe_float(_find_text(
            root,
            ".//TargetClockPeriod",
            ".//UserAssignments/TargetClockPeriod",
        ))

    # --- Latency ---
    latency_cycles = _safe_int(_find_text(
        root,
        ".//PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency",
        ".//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency",
        ".//LatencyEstimates/SummaryOfOverallLatency/Best-caseLatency",
    ))
    latency_ns = None
    if latency_cycles is not None and clock_target is not None:
        latency_ns = latency_cycles * clock_target

    # --- Resources ---
    bram_used = _safe_int(_find_text(
        root,
        ".//AreaEstimates/Resources/BRAM_18K",
        ".//AreaEstimates/Resources/BRAM",
    ))
    bram_avail = _safe_int(_find_text(
        root,
        ".//AreaEstimates/AvailableResources/BRAM_18K",
        ".//AreaEstimates/AvailableResources/BRAM",
    ))
    dsp_used = _safe_int(_find_text(
        root,
        ".//AreaEstimates/Resources/DSP",
        ".//AreaEstimates/Resources/DSP48E",
    ))
    dsp_avail = _safe_int(_find_text(
        root,
        ".//AreaEstimates/AvailableResources/DSP",
        ".//AreaEstimates/AvailableResources/DSP48E",
    ))
    ff_used = _safe_int(_find_text(
        root,
        ".//AreaEstimates/Resources/FF",
    ))
    ff_avail = _safe_int(_find_text(
        root,
        ".//AreaEstimates/AvailableResources/FF",
    ))
    lut_used = _safe_int(_find_text(
        root,
        ".//AreaEstimates/Resources/LUT",
    ))
    lut_avail = _safe_int(_find_text(
        root,
        ".//AreaEstimates/AvailableResources/LUT",
    ))

    return HLSSynthesisResult(
        success=True,
        clock_period_ns=clock_period,
        clock_target_ns=clock_target,
        estimated_latency_cycles=latency_cycles,
        estimated_latency_ns=latency_ns,
        bram_used=bram_used,
        bram_available=bram_avail,
        dsp_used=dsp_used,
        dsp_available=dsp_avail,
        ff_used=ff_used,
        ff_available=ff_avail,
        lut_used=lut_used,
        lut_available=lut_avail,
        raw_report=raw_text,
    )
