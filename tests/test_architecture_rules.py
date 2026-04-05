"""Tests for architecture_rules: Precision, ModelSpec, estimate_param_count, KNOWN_FAMILIES."""

from __future__ import annotations

import pytest

from hardware_feasibility.models.architecture_rules import (
    KNOWN_FAMILIES,
    ModelSpec,
    Precision,
    estimate_param_count,
)


# ---------------------------------------------------------------------------
# Precision enum
# ---------------------------------------------------------------------------

class TestPrecision:
    @pytest.mark.parametrize(
        "prec, expected_bytes",
        [
            (Precision.FP32, 4.0),
            (Precision.FP16, 2.0),
            (Precision.BF16, 2.0),
            (Precision.INT8, 1.0),
            (Precision.INT4, 0.5),
        ],
    )
    def test_bytes_per_element(self, prec: Precision, expected_bytes: float):
        assert prec.bytes_per_element == expected_bytes

    def test_value_strings(self):
        assert Precision.FP16.value == "fp16"
        assert Precision.INT4.value == "int4"


# ---------------------------------------------------------------------------
# ModelSpec properties
# ---------------------------------------------------------------------------

class TestModelSpec:
    def test_head_dim(self, llama3_8b_fp16: ModelSpec):
        assert llama3_8b_fp16.head_dim == 128  # 4096 // 32

    def test_bytes_per_weight_fp16(self, llama3_8b_fp16: ModelSpec):
        assert llama3_8b_fp16.bytes_per_weight == 2.0

    def test_bytes_per_weight_int4(self, llama3_8b_int4: ModelSpec):
        assert llama3_8b_int4.bytes_per_weight == 0.5

    def test_bytes_per_kv_int8(self, llama3_8b_int4: ModelSpec):
        assert llama3_8b_int4.bytes_per_kv == 1.0


# ---------------------------------------------------------------------------
# estimate_param_count - gold standard values
# ---------------------------------------------------------------------------

class TestEstimateParamCount:
    def test_llama3_8b(self):
        """Llama-3-8B should estimate ~8.03B params (untied embeddings)."""
        params = estimate_param_count(**KNOWN_FAMILIES["llama3-8b"])
        assert params == 8_030_257_152

    def test_qwen2_05b(self):
        params = estimate_param_count(**KNOWN_FAMILIES["qwen2-0.5b"])
        assert params == 630_138_880

    def test_llama2_7b(self):
        params = estimate_param_count(**KNOWN_FAMILIES["llama2-7b"])
        # Llama-2-7B reference: ~6.74B (untied embed counted as 2x)
        assert params > 6_000_000_000
        assert params < 8_000_000_000

    def test_symmetry_of_projections(self):
        """Q/O projections should use num_attention_heads, K/V should use num_kv_heads."""
        # Compare MHA (kv=heads) vs GQA (kv < heads) with same dimensions otherwise
        base = dict(
            num_layers=1,
            hidden_size=512,
            intermediate_size=2048,
            vocab_size=1000,
            num_attention_heads=8,
        )
        mha = estimate_param_count(**base, num_kv_heads=8)
        gqa = estimate_param_count(**base, num_kv_heads=2)
        # GQA should have fewer params because K,V projections are smaller
        assert gqa < mha

    def test_scales_with_layers(self):
        base = dict(
            hidden_size=512,
            intermediate_size=2048,
            vocab_size=1000,
            num_attention_heads=8,
            num_kv_heads=8,
        )
        p1 = estimate_param_count(num_layers=1, **base)
        p2 = estimate_param_count(num_layers=2, **base)
        # Embedding is constant; per-layer cost should be p2-p1 ~= p1 - embed
        per_layer = p2 - p1
        assert per_layer > 0
        p4 = estimate_param_count(num_layers=4, **base)
        assert p4 - p2 == pytest.approx(2 * per_layer)


# ---------------------------------------------------------------------------
# KNOWN_FAMILIES coverage
# ---------------------------------------------------------------------------

class TestKnownFamilies:
    def test_all_families_have_required_keys(self):
        required = {
            "num_layers", "hidden_size", "num_attention_heads",
            "num_kv_heads", "intermediate_size", "vocab_size",
        }
        for name, arch in KNOWN_FAMILIES.items():
            missing = required - set(arch.keys())
            assert not missing, f"Family '{name}' missing keys: {missing}"

    def test_at_least_10_families(self):
        assert len(KNOWN_FAMILIES) >= 10

    def test_families_produce_positive_params(self):
        for name, arch in KNOWN_FAMILIES.items():
            params = estimate_param_count(**arch)
            assert params > 0, f"Family '{name}' yielded non-positive params"
