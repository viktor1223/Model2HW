"""Tests for hf_config_loader: loading from JSON files and known families."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from hardware_feasibility.models.hf_config_loader import (
    _precision_from_str,
    load_from_hf_config,
    load_from_known_family,
)
from hardware_feasibility.models.architecture_rules import Precision


# ---------------------------------------------------------------------------
# Precision parsing
# ---------------------------------------------------------------------------

class TestPrecisionFromStr:
    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("fp16", Precision.FP16),
            ("FP16", Precision.FP16),
            ("float16", Precision.FP16),
            ("bf16", Precision.BF16),
            ("bfloat16", Precision.BF16),
            ("int8", Precision.INT8),
            ("int4", Precision.INT4),
            ("fp32", Precision.FP32),
            ("float32", Precision.FP32),
        ],
    )
    def test_valid_strings(self, input_str: str, expected: Precision):
        assert _precision_from_str(input_str) == expected

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown precision"):
            _precision_from_str("fp8")

    def test_normalizes_hyphens_underscores(self):
        assert _precision_from_str("b-f16") == Precision.BF16
        assert _precision_from_str("b_f16") == Precision.BF16


# ---------------------------------------------------------------------------
# load_from_known_family
# ---------------------------------------------------------------------------

class TestLoadFromKnownFamily:
    def test_loads_llama3_8b(self):
        spec = load_from_known_family("llama3-8b")
        assert spec.name == "llama3-8b"
        assert spec.num_layers == 32
        assert spec.hidden_size == 4096
        assert spec.num_kv_heads == 8
        assert spec.weight_precision == Precision.FP16

    def test_custom_precision(self):
        spec = load_from_known_family(
            "qwen2-0.5b",
            weight_precision="int4",
            kv_precision="int8",
        )
        assert spec.weight_precision == Precision.INT4
        assert spec.kv_precision == Precision.INT8

    def test_custom_runtime_params(self):
        spec = load_from_known_family(
            "llama3-8b",
            batch_size=4,
            context_length=8192,
            target_tokens_per_sec=30.0,
        )
        assert spec.batch_size == 4
        assert spec.context_length == 8192
        assert spec.target_tokens_per_sec == 30.0

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            load_from_known_family("nonexistent-model")

    def test_case_insensitive(self):
        spec = load_from_known_family("Llama3-8b")
        assert spec.num_layers == 32


# ---------------------------------------------------------------------------
# load_from_hf_config
# ---------------------------------------------------------------------------

class TestLoadFromHfConfig:
    def test_standard_config(self, tmp_path: Path):
        config = {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_hidden_layers": 24,
            "intermediate_size": 8192,
            "vocab_size": 32000,
            "_name_or_path": "test-model",
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        spec = load_from_hf_config(str(config_file))
        assert spec.name == "test-model"
        assert spec.hidden_size == 2048
        assert spec.num_attention_heads == 16
        assert spec.num_kv_heads == 4
        assert spec.num_layers == 24

    def test_gpt2_style_keys(self, tmp_path: Path):
        """Config using n_embd, n_head, n_layer (GPT-2 style)."""
        config = {
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "vocab_size": 50257,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        spec = load_from_hf_config(str(config_file))
        assert spec.hidden_size == 768
        assert spec.num_attention_heads == 12
        assert spec.num_kv_heads == 12  # MHA fallback
        assert spec.num_layers == 12
        # intermediate_size should default to 4*hidden
        assert spec.intermediate_size == 3072

    def test_name_override(self, tmp_path: Path):
        config = {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 6,
            "vocab_size": 10000,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        spec = load_from_hf_config(str(config_file), name_override="my-custom-model")
        assert spec.name == "my-custom-model"

    def test_precision_and_runtime_params(self, tmp_path: Path):
        config = {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_hidden_layers": 6,
            "vocab_size": 10000,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        spec = load_from_hf_config(
            str(config_file),
            weight_precision="int4",
            kv_precision="int8",
            batch_size=2,
            context_length=1024,
        )
        assert spec.weight_precision == Precision.INT4
        assert spec.kv_precision == Precision.INT8
        assert spec.batch_size == 2
        assert spec.context_length == 1024

    def test_params_are_estimated(self, tmp_path: Path):
        config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "vocab_size": 128256,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        spec = load_from_hf_config(str(config_file))
        # Should match the known llama3-8b param count
        assert spec.params == 8_030_257_152
