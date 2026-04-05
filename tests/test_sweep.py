"""Tests for the precision sweep engine (Phase 1)."""

from __future__ import annotations

import json

import pytest

from hardware_feasibility.models.architecture_rules import Precision
from hardware_feasibility.models.hf_config_loader import load_from_known_family
from hardware_feasibility.analysis.sweep import (
    SweepPoint,
    SweepResult,
    run_precision_sweep,
)
from hardware_feasibility.outputs.report import format_sweep_report
from hardware_feasibility.outputs.json_export import format_sweep_json


def test_sweep_llama3_8b_on_vck190():
    """INT4 weights should fit on VCK190 (8 GB); FP16 should not."""
    spec = load_from_known_family("llama3-8b")
    result = run_precision_sweep(spec, target_memory_gb=8.0, target_bandwidth_gbps=25.6)

    # At least one point should fit (INT4)
    assert len(result.fitting_points) > 0

    # INT4/INT4 should be the best fitting point
    best = result.best_fitting
    assert best is not None
    assert best.weight_precision == Precision.INT4

    # FP16/FP16 should NOT fit
    fp16_point = [p for p in result.points
                  if p.weight_precision == Precision.FP16 and p.kv_precision == Precision.FP16]
    assert len(fp16_point) == 1
    assert not fp16_point[0].fits


def test_sweep_skips_kv_higher_than_weight():
    """KV precision should never exceed weight precision."""
    spec = load_from_known_family("llama3.2-1b")
    result = run_precision_sweep(spec, target_memory_gb=4.0)
    for p in result.points:
        assert p.kv_precision.bytes_per_element <= p.weight_precision.bytes_per_element


def test_sweep_returns_sorted():
    """Fitting points come before non-fitting points."""
    spec = load_from_known_family("llama3-8b")
    result = run_precision_sweep(spec, target_memory_gb=8.0)
    seen_not_fit = False
    for p in result.points:
        if not p.fits:
            seen_not_fit = True
        if seen_not_fit:
            assert not p.fits, "Non-fitting point appeared before a fitting point"


def test_sweep_with_no_target():
    """Sweep without hardware targets should still produce points (all 'fit')."""
    spec = load_from_known_family("llama3.2-1b")
    result = run_precision_sweep(spec)
    assert len(result.points) > 0
    # Without targets, every config should have None headroom
    for p in result.points:
        assert p.memory_headroom_gb is None
        assert p.bandwidth_headroom_gbps is None


def test_sweep_custom_precisions():
    """Custom precision lists should restrict the sweep space."""
    spec = load_from_known_family("llama3.2-1b")
    result = run_precision_sweep(
        spec,
        target_memory_gb=4.0,
        weight_precisions=[Precision.INT8, Precision.INT4],
        kv_precisions=[Precision.INT8, Precision.INT4],
    )
    for p in result.points:
        assert p.weight_precision in (Precision.INT8, Precision.INT4)
        assert p.kv_precision in (Precision.INT8, Precision.INT4)


def test_sweep_report_format():
    """format_sweep_report should produce a non-empty string with expected sections."""
    spec = load_from_known_family("llama3.2-1b")
    result = run_precision_sweep(spec, target_memory_gb=4.0, target_bandwidth_gbps=50.0)
    report = format_sweep_report(result)
    assert "PRECISION SWEEP RESULTS" in report
    assert "llama3.2-1b" in report
    assert "configurations fit" in report


def test_sweep_json_format():
    """format_sweep_json should return a valid dict with expected keys."""
    spec = load_from_known_family("llama3.2-1b")
    result = run_precision_sweep(spec, target_memory_gb=4.0, target_bandwidth_gbps=50.0)
    d = format_sweep_json(result)
    assert d["model"] == "llama3.2-1b"
    assert d["total_configurations"] == len(result.points)
    assert d["fitting_configurations"] == len(result.fitting_points)
    assert isinstance(d["points"], list)
    # Ensure it's JSON-serializable
    json.dumps(d)
