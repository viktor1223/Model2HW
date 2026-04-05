"""Load model specifications from Hugging Face configs or local JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .architecture_rules import (
    KNOWN_FAMILIES,
    ModelSpec,
    Precision,
    estimate_param_count,
)


def _precision_from_str(s: str) -> Precision:
    s = s.lower().replace("-", "").replace("_", "")
    mapping = {
        "fp32": Precision.FP32,
        "float32": Precision.FP32,
        "fp16": Precision.FP16,
        "float16": Precision.FP16,
        "bf16": Precision.BF16,
        "bfloat16": Precision.BF16,
        "int8": Precision.INT8,
        "int4": Precision.INT4,
    }
    if s not in mapping:
        raise ValueError(f"Unknown precision '{s}'. Supported: {list(mapping.keys())}")
    return mapping[s]


def load_from_hf_config(
    config_path: str | Path,
    *,
    weight_precision: str = "fp16",
    kv_precision: str = "fp16",
    batch_size: int = 1,
    context_length: int = 4096,
    prefill_length: int = 512,
    decode_length: int = 256,
    target_tokens_per_sec: float = 10.0,
    kv_on_accelerator: bool = True,
    name_override: Optional[str] = None,
) -> ModelSpec:
    """Build a ModelSpec from a Hugging Face-style config.json file."""
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = json.load(f)

    # Resolve architecture-specific field names
    hidden = cfg.get("hidden_size") or cfg.get("d_model") or cfg["n_embd"]
    n_heads = (
        cfg.get("num_attention_heads")
        or cfg.get("n_head")
        or cfg.get("num_heads")
    )
    n_kv = (
        cfg.get("num_key_value_heads")
        or cfg.get("num_kv_heads")
        or n_heads  # MHA fallback
    )
    n_layers = (
        cfg.get("num_hidden_layers")
        or cfg.get("n_layer")
        or cfg.get("num_layers")
    )
    inter = (
        cfg.get("intermediate_size")
        or cfg.get("n_inner")
        or (4 * hidden)  # standard transformer default
    )
    vocab = cfg.get("vocab_size", 32000)
    name = name_override or cfg.get("_name_or_path", config_path.parent.name)

    params = estimate_param_count(
        num_layers=n_layers,
        hidden_size=hidden,
        intermediate_size=inter,
        vocab_size=vocab,
        num_kv_heads=n_kv,
        num_attention_heads=n_heads,
    )

    return ModelSpec(
        name=name,
        params=params,
        num_layers=n_layers,
        hidden_size=hidden,
        num_attention_heads=n_heads,
        num_kv_heads=n_kv,
        intermediate_size=inter,
        vocab_size=vocab,
        weight_precision=_precision_from_str(weight_precision),
        kv_precision=_precision_from_str(kv_precision),
        batch_size=batch_size,
        context_length=context_length,
        prefill_length=prefill_length,
        decode_length=decode_length,
        target_tokens_per_sec=target_tokens_per_sec,
        kv_on_accelerator=kv_on_accelerator,
    )


def load_from_known_family(
    family: str,
    *,
    weight_precision: str = "fp16",
    kv_precision: str = "fp16",
    batch_size: int = 1,
    context_length: int = 4096,
    prefill_length: int = 512,
    decode_length: int = 256,
    target_tokens_per_sec: float = 10.0,
    kv_on_accelerator: bool = True,
) -> ModelSpec:
    """Build a ModelSpec from a known architecture family name."""
    key = family.lower().strip()
    if key not in KNOWN_FAMILIES:
        available = ", ".join(sorted(KNOWN_FAMILIES.keys()))
        raise ValueError(
            f"Unknown family '{family}'. Available: {available}"
        )

    arch = KNOWN_FAMILIES[key]
    params = estimate_param_count(
        num_layers=arch["num_layers"],
        hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        vocab_size=arch["vocab_size"],
        num_kv_heads=arch["num_kv_heads"],
        num_attention_heads=arch["num_attention_heads"],
    )

    return ModelSpec(
        name=family,
        params=params,
        num_layers=arch["num_layers"],
        hidden_size=arch["hidden_size"],
        num_attention_heads=arch["num_attention_heads"],
        num_kv_heads=arch["num_kv_heads"],
        intermediate_size=arch["intermediate_size"],
        vocab_size=arch["vocab_size"],
        weight_precision=_precision_from_str(weight_precision),
        kv_precision=_precision_from_str(kv_precision),
        batch_size=batch_size,
        context_length=context_length,
        prefill_length=prefill_length,
        decode_length=decode_length,
        target_tokens_per_sec=target_tokens_per_sec,
        kv_on_accelerator=kv_on_accelerator,
    )


def load_from_hf_hub(
    model_id: str,
    *,
    revision: str = "main",
    weight_precision: str = "fp16",
    kv_precision: str = "fp16",
    batch_size: int = 1,
    context_length: int = 4096,
    prefill_length: int = 512,
    decode_length: int = 256,
    target_tokens_per_sec: float = 10.0,
    kv_on_accelerator: bool = True,
) -> ModelSpec:
    """Download config.json from HuggingFace Hub and build a ModelSpec.

    Requires the `huggingface_hub` package.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for loading from the Hub. "
            "Install it with: pip install huggingface_hub"
        ) from e

    config_file = hf_hub_download(
        repo_id=model_id,
        filename="config.json",
        revision=revision,
    )

    return load_from_hf_config(
        config_file,
        weight_precision=weight_precision,
        kv_precision=kv_precision,
        batch_size=batch_size,
        context_length=context_length,
        prefill_length=prefill_length,
        decode_length=decode_length,
        target_tokens_per_sec=target_tokens_per_sec,
        kv_on_accelerator=kv_on_accelerator,
        name_override=model_id,
    )
