"""Architecture rules and known model family defaults."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Precision(Enum):
    """Weight or KV-cache precision."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def bytes_per_element(self) -> float:
        return {
            Precision.FP32: 4.0,
            Precision.FP16: 2.0,
            Precision.BF16: 2.0,
            Precision.INT8: 1.0,
            Precision.INT4: 0.5,
        }[self]


@dataclass
class ModelSpec:
    """Full specification of a model needed for hardware feasibility analysis.

    This is the single source of truth that every analysis module reads from.
    """

    name: str
    params: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int

    # Runtime assumptions
    weight_precision: Precision = Precision.FP16
    kv_precision: Precision = Precision.FP16
    batch_size: int = 1
    context_length: int = 4096
    prefill_length: int = 512
    decode_length: int = 256
    target_tokens_per_sec: float = 10.0
    kv_on_accelerator: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def bytes_per_weight(self) -> float:
        return self.weight_precision.bytes_per_element

    @property
    def bytes_per_kv(self) -> float:
        return self.kv_precision.bytes_per_element


# ---------------------------------------------------------------------------
# Known architecture templates for quick estimation when only param count
# and family are known.
# ---------------------------------------------------------------------------

KNOWN_FAMILIES: dict[str, dict] = {
    "llama-7b": dict(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        intermediate_size=11008,
        vocab_size=32000,
    ),
    "llama2-7b": dict(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=32,
        intermediate_size=11008,
        vocab_size=32000,
    ),
    "llama2-13b": dict(
        num_layers=40,
        hidden_size=5120,
        num_attention_heads=40,
        num_kv_heads=40,
        intermediate_size=13824,
        vocab_size=32000,
    ),
    "llama3-8b": dict(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
    ),
    "llama3.1-8b": dict(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
    ),
    "llama3.2-1b": dict(
        num_layers=16,
        hidden_size=2048,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=8192,
        vocab_size=128256,
    ),
    "llama3.2-3b": dict(
        num_layers=28,
        hidden_size=3072,
        num_attention_heads=24,
        num_kv_heads=8,
        intermediate_size=8192,
        vocab_size=128256,
    ),
    "mistral-7b": dict(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=32000,
    ),
    "phi-2": dict(
        num_layers=32,
        hidden_size=2560,
        num_attention_heads=32,
        num_kv_heads=32,
        intermediate_size=10240,
        vocab_size=51200,
    ),
    "gemma-2b": dict(
        num_layers=18,
        hidden_size=2048,
        num_attention_heads=8,
        num_kv_heads=1,
        intermediate_size=16384,
        vocab_size=256000,
    ),
    "qwen2-0.5b": dict(
        num_layers=24,
        hidden_size=896,
        num_attention_heads=14,
        num_kv_heads=2,
        intermediate_size=4864,
        vocab_size=151936,
    ),
    "qwen2-1.5b": dict(
        num_layers=28,
        hidden_size=1536,
        num_attention_heads=12,
        num_kv_heads=2,
        intermediate_size=8960,
        vocab_size=151936,
    ),
}


def estimate_param_count(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
    num_kv_heads: int,
    num_attention_heads: int,
) -> int:
    """Rough parameter count for a decoder-only transformer.

    Accounts for:
      - embedding + output head (tied or untied, assume untied for safety)
      - per-layer: self-attention projections (Q, K, V, O)
      - per-layer: MLP (gate, up, down for SwiGLU-style)
      - per-layer: RMSNorm / LayerNorm weights
    """
    head_dim = hidden_size // num_attention_heads

    # Embedding + LM head (assume untied)
    embed = vocab_size * hidden_size * 2

    # Self-attention per layer
    q_proj = hidden_size * (num_attention_heads * head_dim)
    k_proj = hidden_size * (num_kv_heads * head_dim)
    v_proj = hidden_size * (num_kv_heads * head_dim)
    o_proj = (num_attention_heads * head_dim) * hidden_size
    attn_per_layer = q_proj + k_proj + v_proj + o_proj

    # MLP per layer (SwiGLU: gate_proj, up_proj, down_proj)
    mlp_per_layer = 3 * hidden_size * intermediate_size

    # Norms per layer (2 RMSNorms: attention + MLP)
    norms_per_layer = 2 * hidden_size

    per_layer = attn_per_layer + mlp_per_layer + norms_per_layer
    total = embed + num_layers * per_layer

    return total
