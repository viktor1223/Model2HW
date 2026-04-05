"""Tests for Phase 9: Accuracy-in-the-Loop."""

from __future__ import annotations

import pytest

from hardware_feasibility.evaluation.accuracy_db import (
    KNOWN_PERPLEXITY,
    estimate_perplexity_degradation,
    get_perplexity,
    lookup_perplexity,
)
from hardware_feasibility.models.architecture_rules import Precision


class TestKnownPerplexity:
    def test_minimum_entries(self) -> None:
        """Database must have at least 5 model-precision entries."""
        assert len(KNOWN_PERPLEXITY) >= 5

    def test_all_values_positive(self) -> None:
        for key, ppl in KNOWN_PERPLEXITY.items():
            assert ppl > 0, f"Perplexity for {key} must be positive"

    def test_quantization_increases_perplexity(self) -> None:
        """INT4 perplexity should be >= FP16 perplexity for the same model."""
        for family in ("llama3-8b", "llama2-7b", "mistral-7b"):
            fp16 = KNOWN_PERPLEXITY.get((family, "fp16", "fp16"))
            int4 = KNOWN_PERPLEXITY.get((family, "int4", "int4"))
            if fp16 is not None and int4 is not None:
                assert int4 >= fp16, f"{family} INT4 ppl should be >= FP16 ppl"


class TestLookupPerplexity:
    def test_known_lookup(self) -> None:
        result = lookup_perplexity("llama3-8b", Precision.FP16, Precision.FP16)
        assert result is not None
        assert result == pytest.approx(6.14)

    def test_unknown_lookup(self) -> None:
        result = lookup_perplexity("nonexistent-model", Precision.FP16, Precision.FP16)
        assert result is None


class TestEstimatePerplexity:
    def test_fp16_no_degradation(self) -> None:
        result = estimate_perplexity_degradation(6.14, Precision.FP16)
        assert result == pytest.approx(6.14)

    def test_int8_small_degradation(self) -> None:
        result = estimate_perplexity_degradation(6.14, Precision.INT8)
        assert result > 6.14
        assert result < 6.14 * 1.02  # Less than 2% increase

    def test_int4_larger_degradation(self) -> None:
        result = estimate_perplexity_degradation(6.14, Precision.INT4)
        assert result > 6.14
        assert result < 6.14 * 1.10  # Less than 10% increase


class TestGetPerplexity:
    def test_known_returns_lookup(self) -> None:
        ppl, source = get_perplexity("llama3-8b", Precision.FP16, Precision.FP16)
        assert ppl is not None
        assert source == "lookup"
        assert ppl == pytest.approx(6.14)

    def test_unknown_precision_returns_estimated(self) -> None:
        """BF16 for llama3-8b is not in database but FP16 baseline is."""
        ppl, source = get_perplexity("llama3-8b", Precision.BF16, Precision.BF16)
        assert ppl is not None
        assert source == "estimated"
        assert ppl == pytest.approx(6.14)  # BF16 has multiplier 1.0

    def test_unknown_model_returns_none(self) -> None:
        ppl, source = get_perplexity("nonexistent-99b", Precision.FP16, Precision.FP16)
        assert ppl is None
        assert source == ""


class TestRecommenderPerplexityIntegration:
    def test_recommendations_include_perplexity(self) -> None:
        """Recommendations for llama3-8b should include perplexity data."""
        from hardware_feasibility.analysis.recommender import recommend_configuration
        from hardware_feasibility.hardware.board_specs import BoardSpec

        board = BoardSpec(
            name="test board",
            category="fpga",
            memory_gb=32.0,
            memory_bandwidth_gbps=102.4,
            host_link_bandwidth_gbps=16.0,
        )
        from hardware_feasibility.models.architecture_rules import ModelSpec

        spec = ModelSpec(
            name="llama3-8b",
            params=8_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            intermediate_size=14336,
            vocab_size=128256,
            weight_precision=Precision.FP16,
        )

        result = recommend_configuration(spec, board)
        if result.recommendations:
            has_ppl = any(
                r.estimated_perplexity is not None
                for r in result.recommendations
            )
            assert has_ppl, "At least some recommendations should have perplexity data"
