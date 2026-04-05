"""Tests for the verdict engine."""

from __future__ import annotations

import pytest

from hardware_feasibility.models.architecture_rules import ModelSpec, Precision
from hardware_feasibility.analysis.memory import analyze_memory
from hardware_feasibility.analysis.bandwidth import analyze_bandwidth
from hardware_feasibility.analysis.compute import analyze_compute
from hardware_feasibility.analysis.io import analyze_io
from hardware_feasibility.analysis.verdict import (
    FeasibilityVerdict,
    VerdictResult,
    render_verdict,
)


def _full_profiles(spec: ModelSpec):
    """Convenience: run all four analyses."""
    mem = analyze_memory(spec)
    bw = analyze_bandwidth(spec)
    comp = analyze_compute(spec, 0)
    io = analyze_io(spec)
    return mem, bw, comp, io


# ---------------------------------------------------------------------------
# Verdict outcomes
# ---------------------------------------------------------------------------

class TestVerdictOutcomes:
    def test_fits_comfortably(self, qwen2_05b_int4: ModelSpec):
        """Small model on large board should fit comfortably."""
        mem, bw, comp, io = _full_profiles(qwen2_05b_int4)
        v = render_verdict(
            qwen2_05b_int4, mem, bw, comp, io,
            target_memory_gb=8.0,
            target_bandwidth_gbps=25.6,
            target_link_bandwidth_gbps=16.0,
        )
        assert v.verdict == FeasibilityVerdict.FITS_COMFORTABLY

    def test_does_not_fit_memory(self, llama3_8b_fp16: ModelSpec):
        """15.5 GB model on 8 GB board should not fit."""
        mem, bw, comp, io = _full_profiles(llama3_8b_fp16)
        v = render_verdict(
            llama3_8b_fp16, mem, bw, comp, io,
            target_memory_gb=8.0,
            target_bandwidth_gbps=2000.0,
            target_link_bandwidth_gbps=64.0,
        )
        assert v.verdict == FeasibilityVerdict.DOES_NOT_FIT

    def test_fits_bandwidth_limited(self, llama3_8b_int4: ModelSpec):
        """INT4 llama3-8b fits in 8GB memory but is bandwidth limited on VCK190."""
        mem, bw, comp, io = _full_profiles(llama3_8b_int4)
        v = render_verdict(
            llama3_8b_int4, mem, bw, comp, io,
            target_memory_gb=8.0,
            target_bandwidth_gbps=25.6,
            target_link_bandwidth_gbps=16.0,
        )
        assert v.verdict == FeasibilityVerdict.FITS_BANDWIDTH_LIMITED

    def test_host_link_bottleneck(self, llama3_8b_off_accel: ModelSpec):
        """Off-accelerator KV with tiny host link should be host-link bottleneck."""
        mem, bw, comp, io = _full_profiles(llama3_8b_off_accel)
        v = render_verdict(
            llama3_8b_off_accel, mem, bw, comp, io,
            target_memory_gb=64.0,
            target_bandwidth_gbps=77.0,
            target_link_bandwidth_gbps=0.1,  # very small link
        )
        assert v.verdict == FeasibilityVerdict.HOST_LINK_BOTTLENECK


# ---------------------------------------------------------------------------
# Verdict details
# ---------------------------------------------------------------------------

class TestVerdictDetails:
    def test_memory_exceeds_detail(self, llama3_8b_fp16: ModelSpec):
        mem, bw, comp, io = _full_profiles(llama3_8b_fp16)
        v = render_verdict(
            llama3_8b_fp16, mem, bw, comp, io,
            target_memory_gb=8.0,
        )
        assert any("exceeds" in d for d in v.details)

    def test_memory_fits_detail(self, qwen2_05b_int4: ModelSpec):
        mem, bw, comp, io = _full_profiles(qwen2_05b_int4)
        v = render_verdict(
            qwen2_05b_int4, mem, bw, comp, io,
            target_memory_gb=8.0,
        )
        assert any("fits" in d.lower() for d in v.details)

    def test_details_nonempty(self, llama3_8b_fp16: ModelSpec):
        mem, bw, comp, io = _full_profiles(llama3_8b_fp16)
        v = render_verdict(llama3_8b_fp16, mem, bw, comp, io)
        assert len(v.details) > 0

    def test_generic_summary_when_no_targets(self, llama3_8b_fp16: ModelSpec):
        """Without hardware targets, the verdict should give generic info."""
        mem, bw, comp, io = _full_profiles(llama3_8b_fp16)
        v = render_verdict(llama3_8b_fp16, mem, bw, comp, io)
        detail_text = " ".join(v.details)
        assert "working memory" in detail_text.lower() or "bandwidth" in detail_text.lower()


# ---------------------------------------------------------------------------
# VerdictResult structure
# ---------------------------------------------------------------------------

class TestVerdictResult:
    def test_returns_verdict_result(self, llama3_8b_fp16: ModelSpec):
        mem, bw, comp, io = _full_profiles(llama3_8b_fp16)
        v = render_verdict(llama3_8b_fp16, mem, bw, comp, io)
        assert isinstance(v, VerdictResult)

    def test_target_fields_stored(self, qwen2_05b_int4: ModelSpec):
        mem, bw, comp, io = _full_profiles(qwen2_05b_int4)
        v = render_verdict(
            qwen2_05b_int4, mem, bw, comp, io,
            target_memory_gb=8.0,
            target_bandwidth_gbps=25.6,
            target_link_bandwidth_gbps=16.0,
        )
        assert v.target_memory_gb == 8.0
        assert v.target_bandwidth_gbps == 25.6
        assert v.target_link_bandwidth_gbps == 16.0

    def test_summary_is_first_detail(self, llama3_8b_fp16: ModelSpec):
        mem, bw, comp, io = _full_profiles(llama3_8b_fp16)
        v = render_verdict(
            llama3_8b_fp16, mem, bw, comp, io,
            target_memory_gb=8.0,
        )
        assert v.summary == v.details[0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestVerdictEdgeCases:
    def test_tight_memory_reports_warning(self):
        """When headroom < 15% of total, a warning about tightness should appear."""
        from tests.conftest import _make_spec

        spec = _make_spec(
            "qwen2-0.5b",
            weight_precision=Precision.FP16,
            context_length=2048,
        )
        mem, bw, comp, io = _full_profiles(spec)
        # Set target just above total to trigger tight-memory path
        target = mem.total_gb * 1.10  # 10% headroom < 15% threshold
        v = render_verdict(
            spec, mem, bw, comp, io,
            target_memory_gb=target,
        )
        assert any("tight" in d.lower() for d in v.details)

    def test_no_targets_gives_generic_verdict(self, llama3_8b_fp16: ModelSpec):
        mem, bw, comp, io = _full_profiles(llama3_8b_fp16)
        v = render_verdict(llama3_8b_fp16, mem, bw, comp, io)
        # Should still be FITS_COMFORTABLY (default when no targets)
        assert v.verdict == FeasibilityVerdict.FITS_COMFORTABLY
