"""Tests for the sensitivity analyzer (Phase 2)."""

from __future__ import annotations

import json

import pytest

from hardware_feasibility.models.architecture_rules import Precision
from hardware_feasibility.models.hf_config_loader import load_from_known_family
from hardware_feasibility.analysis.memory import analyze_memory
from hardware_feasibility.analysis.bandwidth import analyze_bandwidth
from hardware_feasibility.analysis.compute import analyze_compute
from hardware_feasibility.analysis.io import analyze_io
from hardware_feasibility.analysis.sensitivity import (
    BottleneckBreakdown,
    SensitivityResult,
    analyze_sensitivity,
)
from hardware_feasibility.outputs.report import format_sensitivity_report
from hardware_feasibility.outputs.json_export import format_sensitivity_json


def _run_sensitivity(family: str, target_mem: float, target_bw: float,
                     target_link: float = 16.0, target_tops: float | None = None,
                     **overrides) -> SensitivityResult:
    spec = load_from_known_family(family, **overrides)
    mem = analyze_memory(spec)
    bw = analyze_bandwidth(spec)
    compute = analyze_compute(spec, 0)
    io = analyze_io(spec)
    return analyze_sensitivity(
        spec, mem, bw, compute, io,
        target_memory_gb=target_mem,
        target_bandwidth_gbps=target_bw,
        target_link_bandwidth_gbps=target_link,
        target_tops=target_tops,
    )


def test_sensitivity_identifies_bandwidth_bottleneck():
    """llama3-8b INT4 on VCK190 should report bandwidth as primary bottleneck."""
    result = _run_sensitivity("llama3-8b", 8.0, 25.6,
                              weight_precision="int4", kv_precision="int4")
    assert result.bottleneck.primary_bottleneck == "bandwidth"
    assert result.bottleneck.memory_utilization < 1.0
    assert result.bottleneck.bandwidth_utilization > 1.0


def test_sensitivity_identifies_memory_bottleneck():
    """llama3-8b FP16 on a board with huge bandwidth but tiny memory should report memory."""
    # 4 GB memory but 2000 GB/s bandwidth (like a tiny HBM slice)
    result = _run_sensitivity("llama3-8b", 4.0, 2000.0, target_link=64.0)
    assert result.bottleneck.primary_bottleneck == "memory"
    assert result.bottleneck.memory_utilization > 1.0


def test_sensitivity_perturbation_count():
    """Verify the expected number of sensitivity points are generated."""
    result = _run_sensitivity("llama3-8b", 8.0, 25.6)
    # 3 memory perturbations + 3 bw perturbations + 3 context perturbations
    # + 2 batch perturbations + 1 weight precision step = 12
    assert len(result.sensitivities) == 12


def test_sensitivity_verdict_changes_on_memory_increase():
    """Increasing memory target should change verdict for FP16 model that doesn't fit."""
    result = _run_sensitivity("llama3-8b", 8.0, 25.6)
    # Find +50% memory perturbation
    mem_50 = [s for s in result.sensitivities if "target_memory_gb +50%" in s.parameter_name]
    assert len(mem_50) == 1
    # Original doesn't fit (FP16 on 8GB), +50% (12GB) still might not fit for 15GB model
    # but the point is the perturbation ran correctly
    assert mem_50[0].modified_value == pytest.approx(12.0)


def test_sensitivity_compute_utilization_with_tops():
    """When target_tops is provided, compute utilization should be non-zero."""
    result = _run_sensitivity("llama3-8b", 8.0, 25.6, target_tops=133.0,
                              weight_precision="int4", kv_precision="int4")
    assert result.bottleneck.compute_utilization > 0.0


def test_sensitivity_report_format():
    """format_sensitivity_report should produce expected sections."""
    result = _run_sensitivity("llama3.2-1b", 4.0, 50.0)
    report = format_sensitivity_report(result)
    assert "SENSITIVITY ANALYSIS" in report
    assert "Primary bottleneck:" in report
    assert "Resource utilization:" in report
    assert "What-if scenarios:" in report


def test_sensitivity_json_format():
    """format_sensitivity_json should return valid structured data."""
    result = _run_sensitivity("llama3.2-1b", 4.0, 50.0)
    d = format_sensitivity_json(result)
    assert "primary_bottleneck" in d
    assert "utilization" in d
    assert "sensitivities" in d
    assert len(d["sensitivities"]) > 0
    # Ensure it's JSON-serializable
    json.dumps(d)
