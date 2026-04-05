"""Tests for host-device IO analysis module."""

from __future__ import annotations

import pytest

from hardware_feasibility.models.architecture_rules import ModelSpec
from hardware_feasibility.analysis.io import IOProfile, analyze_io
from hardware_feasibility.analysis.memory import BYTES_PER_GB


# ---------------------------------------------------------------------------
# On-accelerator KV (default)
# ---------------------------------------------------------------------------

class TestAnalyzeIOOnAccelerator:
    def test_kv_sync_zero_when_on_accelerator(self, llama3_8b_fp16: ModelSpec):
        io = analyze_io(llama3_8b_fp16)
        assert io.kv_sync_bytes_per_token == 0

    def test_input_embedding_bytes(self, llama3_8b_fp16: ModelSpec):
        io = analyze_io(llama3_8b_fp16)
        # hidden_size * bytes_per_weight = 4096 * 2 = 8192
        assert io.input_embedding_bytes == 8192

    def test_output_logits_bytes(self, llama3_8b_fp16: ModelSpec):
        io = analyze_io(llama3_8b_fp16)
        # vocab_size * bytes_per_weight = 128256 * 2 = 256512
        assert io.output_logits_bytes == 256_512

    def test_total_is_input_plus_output(self, llama3_8b_fp16: ModelSpec):
        io = analyze_io(llama3_8b_fp16)
        assert io.total_io_bytes_per_token == io.input_embedding_bytes + io.output_logits_bytes

    def test_reference_values(self, llama3_8b_fp16: ModelSpec):
        io = analyze_io(llama3_8b_fp16)
        assert io.total_io_bytes_per_token == 264_704
        assert io.required_io_bandwidth_bytes_per_sec == pytest.approx(2_647_040.0)

    def test_returns_io_profile(self, llama3_8b_fp16: ModelSpec):
        io = analyze_io(llama3_8b_fp16)
        assert isinstance(io, IOProfile)


# ---------------------------------------------------------------------------
# Off-accelerator KV
# ---------------------------------------------------------------------------

class TestAnalyzeIOOffAccelerator:
    def test_kv_sync_nonzero(self, llama3_8b_off_accel: ModelSpec):
        io = analyze_io(llama3_8b_off_accel)
        assert io.kv_sync_bytes_per_token > 0

    def test_kv_sync_reference(self, llama3_8b_off_accel: ModelSpec):
        io = analyze_io(llama3_8b_off_accel)
        assert io.kv_sync_bytes_per_token == 41_943_040

    def test_total_includes_kv_sync(self, llama3_8b_off_accel: ModelSpec):
        io = analyze_io(llama3_8b_off_accel)
        assert io.total_io_bytes_per_token == (
            io.input_embedding_bytes + io.output_logits_bytes + io.kv_sync_bytes_per_token
        )

    def test_off_accel_much_higher_io(self, llama3_8b_int4: ModelSpec, llama3_8b_off_accel: ModelSpec):
        io_on = analyze_io(llama3_8b_int4)
        io_off = analyze_io(llama3_8b_off_accel)
        assert io_off.total_io_bytes_per_token > 100 * io_on.total_io_bytes_per_token


# ---------------------------------------------------------------------------
# Bandwidth scaling
# ---------------------------------------------------------------------------

class TestIOBandwidthScaling:
    def test_required_bw_scales_with_tok_s(self):
        from hardware_feasibility.models.hf_config_loader import load_from_known_family

        spec_10 = load_from_known_family("llama3-8b", target_tokens_per_sec=10.0)
        spec_20 = load_from_known_family("llama3-8b", target_tokens_per_sec=20.0)
        io_10 = analyze_io(spec_10)
        io_20 = analyze_io(spec_20)
        assert io_20.required_io_bandwidth_bytes_per_sec == pytest.approx(
            2 * io_10.required_io_bandwidth_bytes_per_sec
        )

    def test_gbps_property(self, llama3_8b_fp16: ModelSpec):
        io = analyze_io(llama3_8b_fp16)
        assert io.required_io_bandwidth_gbps == pytest.approx(
            io.required_io_bandwidth_bytes_per_sec / BYTES_PER_GB
        )
