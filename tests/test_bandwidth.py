"""Tests for bandwidth analysis module."""

from __future__ import annotations

import pytest

from hardware_feasibility.models.architecture_rules import ModelSpec
from hardware_feasibility.analysis.bandwidth import BandwidthProfile, analyze_bandwidth
from hardware_feasibility.analysis.memory import BYTES_PER_GB, estimate_weight_memory


# ---------------------------------------------------------------------------
# Bandwidth profile values
# ---------------------------------------------------------------------------

class TestAnalyzeBandwidth:
    def test_weight_bytes_per_token_equals_total_weights(self, llama3_8b_fp16: ModelSpec):
        """Worst-case: all weights are streamed per decode token."""
        bw = analyze_bandwidth(llama3_8b_fp16)
        w = estimate_weight_memory(llama3_8b_fp16)
        assert bw.weight_bytes_per_token == w

    def test_total_exceeds_weights(self, llama3_8b_fp16: ModelSpec):
        """Total bytes/token includes KV read, so must exceed weight-only."""
        bw = analyze_bandwidth(llama3_8b_fp16)
        assert bw.total_bytes_per_token > bw.weight_bytes_per_token

    def test_kv_read_positive(self, llama3_8b_fp16: ModelSpec):
        bw = analyze_bandwidth(llama3_8b_fp16)
        assert bw.kv_read_bytes_per_token > 0

    def test_required_bandwidth_is_total_times_tok_s(self, llama3_8b_fp16: ModelSpec):
        bw = analyze_bandwidth(llama3_8b_fp16)
        expected = bw.total_bytes_per_token * llama3_8b_fp16.target_tokens_per_sec
        assert bw.required_bandwidth_bytes_per_sec == pytest.approx(expected)

    def test_gbps_property(self, llama3_8b_fp16: ModelSpec):
        bw = analyze_bandwidth(llama3_8b_fp16)
        assert bw.required_bandwidth_gbps == pytest.approx(
            bw.required_bandwidth_bytes_per_sec / BYTES_PER_GB
        )

    def test_llama3_8b_fp16_reference(self, llama3_8b_fp16: ModelSpec):
        bw = analyze_bandwidth(llama3_8b_fp16)
        assert bw.required_bandwidth_gbps == pytest.approx(150.356, abs=0.1)

    def test_returns_bandwidth_profile(self, llama3_8b_fp16: ModelSpec):
        bw = analyze_bandwidth(llama3_8b_fp16)
        assert isinstance(bw, BandwidthProfile)

    def test_scales_linearly_with_target_tok_s(self):
        """Doubling target tok/s should double required bandwidth."""
        from hardware_feasibility.models.hf_config_loader import load_from_known_family

        spec_10 = load_from_known_family("llama3-8b", target_tokens_per_sec=10.0)
        spec_20 = load_from_known_family("llama3-8b", target_tokens_per_sec=20.0)
        bw_10 = analyze_bandwidth(spec_10)
        bw_20 = analyze_bandwidth(spec_20)
        assert bw_20.required_bandwidth_bytes_per_sec == pytest.approx(
            2 * bw_10.required_bandwidth_bytes_per_sec
        )

    def test_int4_reduces_weight_bandwidth(self, llama3_8b_fp16: ModelSpec, llama3_8b_int4: ModelSpec):
        bw_fp16 = analyze_bandwidth(llama3_8b_fp16)
        bw_int4 = analyze_bandwidth(llama3_8b_int4)
        assert bw_int4.weight_bytes_per_token < bw_fp16.weight_bytes_per_token
