"""Tests for compute analysis module."""

from __future__ import annotations

import pytest

from hardware_feasibility.models.architecture_rules import ModelSpec
from hardware_feasibility.analysis.compute import (
    ComputeProfile,
    analyze_compute,
    estimate_flops_per_token,
    estimate_flops_prefill,
)


# ---------------------------------------------------------------------------
# FLOPs per token
# ---------------------------------------------------------------------------

class TestEstimateFlopsPerToken:
    def test_llama3_8b_reference(self, llama3_8b_fp16: ModelSpec):
        flops = estimate_flops_per_token(llama3_8b_fp16)
        assert flops == 15_344_861_184

    def test_positive(self, qwen2_05b_int4: ModelSpec):
        flops = estimate_flops_per_token(qwen2_05b_int4)
        assert flops > 0

    def test_larger_model_more_flops(self, llama3_8b_fp16: ModelSpec, qwen2_05b_int4: ModelSpec):
        flops_big = estimate_flops_per_token(llama3_8b_fp16)
        flops_small = estimate_flops_per_token(qwen2_05b_int4)
        assert flops_big > flops_small


# ---------------------------------------------------------------------------
# FLOPs prefill
# ---------------------------------------------------------------------------

class TestEstimateFlopsPrefix:
    def test_llama3_8b_reference(self, llama3_8b_fp16: ModelSpec):
        flops = estimate_flops_prefill(llama3_8b_fp16)
        assert flops == 7_822_209_187_840

    def test_prefill_exceeds_per_token(self, llama3_8b_fp16: ModelSpec):
        """Prefill processes many tokens, so total FLOPs >> single decode token."""
        per_tok = estimate_flops_per_token(llama3_8b_fp16)
        prefill = estimate_flops_prefill(llama3_8b_fp16)
        assert prefill > per_tok * 10

    def test_scales_with_prefill_length(self):
        from tests.conftest import _make_spec

        short = _make_spec("llama3-8b", prefill_length=128)
        long = _make_spec("llama3-8b", prefill_length=512)
        f_short = estimate_flops_prefill(short)
        f_long = estimate_flops_prefill(long)
        assert f_long > f_short


# ---------------------------------------------------------------------------
# Full compute analysis
# ---------------------------------------------------------------------------

class TestAnalyzeCompute:
    def test_arithmetic_intensity_llama3_8b(self, llama3_8b_fp16: ModelSpec):
        comp = analyze_compute(llama3_8b_fp16, 0)
        assert comp.arithmetic_intensity == pytest.approx(0.9505, abs=0.001)

    def test_roofline_note_bw_bound(self, llama3_8b_fp16: ModelSpec):
        comp = analyze_compute(llama3_8b_fp16, 0)
        assert "memory-bandwidth-bound" in comp.roofline_note.lower()

    def test_returns_compute_profile(self, llama3_8b_fp16: ModelSpec):
        comp = analyze_compute(llama3_8b_fp16, 0)
        assert isinstance(comp, ComputeProfile)

    def test_fields_populated(self, llama3_8b_fp16: ModelSpec):
        comp = analyze_compute(llama3_8b_fp16, 0)
        assert comp.flops_per_token > 0
        assert comp.flops_prefill > 0
        assert comp.arithmetic_intensity > 0
        assert len(comp.roofline_note) > 0
