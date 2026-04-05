"""Accuracy database: known perplexity values and estimation heuristics."""

from __future__ import annotations

from typing import Optional

from ..models.architecture_rules import Precision


# ---------------------------------------------------------------------------
# Known perplexity values (wikitext-2, published benchmarks)
# Key: (model_family, weight_precision_value, kv_precision_value)
# ---------------------------------------------------------------------------

KNOWN_PERPLEXITY: dict[tuple[str, str, str], float] = {
    # Llama 3-8B
    ("llama3-8b", "fp16", "fp16"): 6.14,
    ("llama3-8b", "int8", "int8"): 6.19,
    ("llama3-8b", "int4", "int4"): 6.52,
    ("llama3-8b", "int4", "int8"): 6.38,
    # Llama 3.1-8B (same arch, similar perplexity)
    ("llama3.1-8b", "fp16", "fp16"): 6.14,
    ("llama3.1-8b", "int8", "int8"): 6.20,
    ("llama3.1-8b", "int4", "int4"): 6.55,
    # Llama 3.2-1B
    ("llama3.2-1b", "fp16", "fp16"): 9.78,
    ("llama3.2-1b", "int8", "int8"): 9.84,
    ("llama3.2-1b", "int4", "int4"): 10.42,
    # Llama 3.2-3B
    ("llama3.2-3b", "fp16", "fp16"): 7.80,
    ("llama3.2-3b", "int8", "int8"): 7.87,
    ("llama3.2-3b", "int4", "int4"): 8.25,
    # Llama 2-7B
    ("llama2-7b", "fp16", "fp16"): 5.47,
    ("llama2-7b", "int8", "int8"): 5.53,
    ("llama2-7b", "int4", "int4"): 5.85,
    # Llama 2-13B
    ("llama2-13b", "fp16", "fp16"): 4.88,
    ("llama2-13b", "int8", "int8"): 4.93,
    ("llama2-13b", "int4", "int4"): 5.19,
    # Mistral-7B
    ("mistral-7b", "fp16", "fp16"): 5.25,
    ("mistral-7b", "int8", "int8"): 5.30,
    ("mistral-7b", "int4", "int4"): 5.62,
    # Qwen2-0.5B
    ("qwen2-0.5b", "fp16", "fp16"): 13.20,
    ("qwen2-0.5b", "int8", "int8"): 13.35,
    ("qwen2-0.5b", "int4", "int4"): 14.10,
    # Qwen2-1.5B
    ("qwen2-1.5b", "fp16", "fp16"): 9.60,
    ("qwen2-1.5b", "int8", "int8"): 9.72,
    ("qwen2-1.5b", "int4", "int4"): 10.25,
}


# Degradation multipliers by precision (relative to FP16 baseline)
_DEGRADATION_MULTIPLIERS: dict[Precision, float] = {
    Precision.FP32: 1.0,
    Precision.FP16: 1.0,
    Precision.BF16: 1.0,
    Precision.INT8: 1.008,   # ~0.5-1% increase
    Precision.INT4: 1.05,    # ~3-8% increase; conservative AWQ estimate
}


def lookup_perplexity(
    model_family: str,
    weight_precision: Precision,
    kv_precision: Precision,
) -> Optional[float]:
    """Look up known perplexity for a model-precision combination.

    Returns None if no empirical data is available.
    """
    key = (model_family, weight_precision.value, kv_precision.value)
    return KNOWN_PERPLEXITY.get(key)


def estimate_perplexity_degradation(
    base_ppl: float,
    precision: Precision,
) -> float:
    """Estimate perplexity increase from quantization.

    Uses published trends across Llama family models as rough multipliers.
    The *base_ppl* should be the FP16 perplexity for the model.
    """
    multiplier = _DEGRADATION_MULTIPLIERS.get(precision, 1.0)
    return base_ppl * multiplier


def get_perplexity(
    model_family: str,
    weight_precision: Precision,
    kv_precision: Precision,
) -> tuple[Optional[float], str]:
    """Get perplexity with source annotation.

    Returns (perplexity, source) where source is one of:
      - "lookup" - from the known perplexity database
      - "estimated" - estimated from FP16 baseline
      - source is "" and perplexity is None if no data available
    """
    # Try exact lookup first
    result = lookup_perplexity(model_family, weight_precision, kv_precision)
    if result is not None:
        return (result, "lookup")

    # Try to estimate from FP16 baseline
    fp16_key = (model_family, "fp16", "fp16")
    base = KNOWN_PERPLEXITY.get(fp16_key)
    if base is not None:
        # Use the worse of weight and kv precision
        worse = weight_precision if weight_precision.bytes_per_element <= kv_precision.bytes_per_element else kv_precision
        estimated = estimate_perplexity_degradation(base, worse)
        return (estimated, "estimated")

    return (None, "")
