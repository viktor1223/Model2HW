"""Tests for memory analysis module."""

from __future__ import annotations

import pytest

from hardware_feasibility.models.architecture_rules import ModelSpec, Precision
from hardware_feasibility.analysis.memory import (
    BYTES_PER_GB,
    MemoryProfile,
    analyze_memory,
    estimate_activation_buffer,
    estimate_kv_cache,
    estimate_kv_cache_per_token,
    estimate_weight_memory,
)


# ---------------------------------------------------------------------------
# Weight memory
# ---------------------------------------------------------------------------

class TestEstimateWeightMemory:
    def test_llama3_8b_fp16(self, llama3_8b_fp16: ModelSpec):
        w = estimate_weight_memory(llama3_8b_fp16)
        assert w == 16_060_514_304  # 8.03B * 2 bytes

    def test_llama3_8b_int4(self, llama3_8b_int4: ModelSpec):
        w = estimate_weight_memory(llama3_8b_int4)
        # 8.03B * 0.5 bytes = ~4.015 GB
        assert w == 4_015_128_576

    def test_scales_linearly_with_precision(self, llama3_8b_fp16: ModelSpec, llama3_8b_int4: ModelSpec):
        w_fp16 = estimate_weight_memory(llama3_8b_fp16)
        w_int4 = estimate_weight_memory(llama3_8b_int4)
        assert w_fp16 == w_int4 * 4  # fp16=2B, int4=0.5B -> 4x ratio


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestEstimateKvCache:
    def test_kv_per_token_llama3_8b_fp16(self, llama3_8b_fp16: ModelSpec):
        kv_tok = estimate_kv_cache_per_token(llama3_8b_fp16)
        # 2 * 32 layers * 8 kv_heads * 128 head_dim * 2 bytes = 131072
        assert kv_tok == 131_072

    def test_kv_cache_full_context(self, llama3_8b_fp16: ModelSpec):
        kv = estimate_kv_cache(llama3_8b_fp16)
        # 131072 bytes/token * 4096 tokens * batch_size=1 = 536870912
        assert kv == 536_870_912

    def test_kv_per_token_int8(self, llama3_8b_int4: ModelSpec):
        kv_tok = estimate_kv_cache_per_token(llama3_8b_int4)
        # 2 * 32 * 8 * 128 * 1 = 65536
        assert kv_tok == 65_536

    def test_kv_scales_with_batch_size(self):
        """KV cache should scale linearly with batch size."""
        from hardware_feasibility.models.hf_config_loader import load_from_known_family

        spec1 = load_from_known_family("llama3-8b", batch_size=1)
        spec4 = load_from_known_family("llama3-8b", batch_size=4)
        assert estimate_kv_cache(spec4) == 4 * estimate_kv_cache(spec1)

    def test_kv_scales_with_context_length(self):
        from hardware_feasibility.models.hf_config_loader import load_from_known_family

        spec_2k = load_from_known_family("llama3-8b", context_length=2048)
        spec_4k = load_from_known_family("llama3-8b", context_length=4096)
        assert estimate_kv_cache(spec_4k) == 2 * estimate_kv_cache(spec_2k)


# ---------------------------------------------------------------------------
# Activation buffer
# ---------------------------------------------------------------------------

class TestEstimateActivationBuffer:
    def test_positive(self, llama3_8b_fp16: ModelSpec):
        act = estimate_activation_buffer(llama3_8b_fp16)
        assert act > 0

    def test_prefill_dominated(self, llama3_8b_fp16: ModelSpec):
        """Activation buffer should grow with prefill_length."""
        from hardware_feasibility.models.hf_config_loader import load_from_known_family

        short = load_from_known_family("llama3-8b", prefill_length=128)
        long = load_from_known_family("llama3-8b", prefill_length=1024)
        assert estimate_activation_buffer(long) > estimate_activation_buffer(short)


# ---------------------------------------------------------------------------
# Full memory analysis
# ---------------------------------------------------------------------------

class TestAnalyzeMemory:
    def test_total_is_sum_of_parts(self, llama3_8b_fp16: ModelSpec):
        mem = analyze_memory(llama3_8b_fp16)
        assert mem.total_bytes == mem.weight_bytes + mem.kv_cache_bytes + mem.activation_buffer_bytes

    def test_gb_properties_consistent(self, llama3_8b_fp16: ModelSpec):
        mem = analyze_memory(llama3_8b_fp16)
        assert mem.total_gb == pytest.approx(mem.total_bytes / BYTES_PER_GB)
        assert mem.weight_gb == pytest.approx(mem.weight_bytes / BYTES_PER_GB)

    def test_llama3_8b_fp16_total(self, llama3_8b_fp16: ModelSpec):
        mem = analyze_memory(llama3_8b_fp16)
        assert mem.total_gb == pytest.approx(15.477, abs=0.01)

    def test_qwen2_int4_total(self, qwen2_05b_int4: ModelSpec):
        mem = analyze_memory(qwen2_05b_int4)
        assert mem.total_gb == pytest.approx(0.306, abs=0.01)

    def test_returns_memory_profile(self, llama3_8b_fp16: ModelSpec):
        mem = analyze_memory(llama3_8b_fp16)
        assert isinstance(mem, MemoryProfile)
