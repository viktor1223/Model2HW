"""Shared fixtures for hardware_feasibility tests."""

from __future__ import annotations

import pytest

from hardware_feasibility.models.architecture_rules import (
    KNOWN_FAMILIES,
    ModelSpec,
    Precision,
    estimate_param_count,
)


def _make_spec(family: str, **overrides) -> ModelSpec:
    """Helper to build a ModelSpec from a known family with optional overrides."""
    arch = KNOWN_FAMILIES[family]
    params = estimate_param_count(**arch)
    defaults = dict(
        name=family,
        params=params,
        num_layers=arch["num_layers"],
        hidden_size=arch["hidden_size"],
        num_attention_heads=arch["num_attention_heads"],
        num_kv_heads=arch["num_kv_heads"],
        intermediate_size=arch["intermediate_size"],
        vocab_size=arch["vocab_size"],
        weight_precision=Precision.FP16,
        kv_precision=Precision.FP16,
        batch_size=1,
        context_length=4096,
        prefill_length=512,
        decode_length=256,
        target_tokens_per_sec=10.0,
        kv_on_accelerator=True,
    )
    defaults.update(overrides)
    # Convert string precisions to Precision enum if needed
    for key in ("weight_precision", "kv_precision"):
        if isinstance(defaults[key], str):
            defaults[key] = Precision(defaults[key])

    return ModelSpec(**defaults)


@pytest.fixture()
def llama3_8b_fp16() -> ModelSpec:
    """Llama-3-8B with FP16 weights and KV cache, default runtime params."""
    return _make_spec("llama3-8b")


@pytest.fixture()
def llama3_8b_int4() -> ModelSpec:
    """Llama-3-8B with INT4 weights, INT8 KV, 2048 context."""
    return _make_spec(
        "llama3-8b",
        weight_precision=Precision.INT4,
        kv_precision=Precision.INT8,
        context_length=2048,
    )


@pytest.fixture()
def qwen2_05b_int4() -> ModelSpec:
    """Qwen2-0.5B with INT4 weights, INT8 KV, 2048 context."""
    return _make_spec(
        "qwen2-0.5b",
        weight_precision=Precision.INT4,
        kv_precision=Precision.INT8,
        context_length=2048,
        prefill_length=256,
        decode_length=128,
        target_tokens_per_sec=10.0,
    )


@pytest.fixture()
def llama3_8b_off_accel() -> ModelSpec:
    """Llama-3-8B INT4 with KV cache OFF the accelerator."""
    return _make_spec(
        "llama3-8b",
        weight_precision=Precision.INT4,
        kv_precision=Precision.INT8,
        context_length=2048,
        kv_on_accelerator=False,
    )
