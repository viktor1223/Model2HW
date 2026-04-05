"""Multi-kernel orchestration for full-model HLS optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import TransformerOperatorSpec, KernelOptimizationResult
from ..hardware.board_specs import BoardSpec
from ..models.architecture_rules import ModelSpec, Precision


# ---------------------------------------------------------------------------
# Operator decomposition
# ---------------------------------------------------------------------------


def decompose_model_to_operators(
    spec: ModelSpec, board: BoardSpec, clock_mhz: int = 200
) -> list[TransformerOperatorSpec]:
    """Break down a model into its constituent transformer operators.

    For each of *num_layers* transformer layers, produces:
      - Q, K, V projections (GEMMs)
      - QK^T attention (batched GEMM)
      - Softmax
      - Score * V (batched GEMM)
      - O projection (GEMM)
      - Gate projection (GEMM) - SwiGLU MLP
      - Up projection (GEMM) - SwiGLU MLP
      - SiLU activation (element-wise)
      - Gate * Up multiply (element-wise, counted with SiLU)
      - Down projection (GEMM)
      - RMSNorm x2
      - Residual add x2 (counted with RMSNorm)

    Plus embedding lookup and LM head.
    """
    h = spec.hidden_size
    n_heads = spec.num_attention_heads
    n_kv = spec.num_kv_heads
    head_dim = spec.head_dim
    inter = spec.intermediate_size
    seq = 1  # decode phase: single token

    ops: list[TransformerOperatorSpec] = []

    common = dict(target_board=board, clock_mhz=clock_mhz)

    for layer in range(spec.num_layers):
        # --- Attention block ---

        # Q projection: (1, h) x (h, n_heads * head_dim)
        ops.append(TransformerOperatorSpec(
            op_type="gemm",
            input_shapes={"input": (seq, h), "weight": (h, n_heads * head_dim)},
            output_shapes={"output": (seq, n_heads * head_dim)},
            precision=spec.weight_precision,
            **common,
        ))

        # K projection: (1, h) x (h, n_kv * head_dim)
        ops.append(TransformerOperatorSpec(
            op_type="gemm",
            input_shapes={"input": (seq, h), "weight": (h, n_kv * head_dim)},
            output_shapes={"output": (seq, n_kv * head_dim)},
            precision=spec.weight_precision,
            **common,
        ))

        # V projection: (1, h) x (h, n_kv * head_dim)
        ops.append(TransformerOperatorSpec(
            op_type="gemm",
            input_shapes={"input": (seq, h), "weight": (h, n_kv * head_dim)},
            output_shapes={"output": (seq, n_kv * head_dim)},
            precision=spec.weight_precision,
            **common,
        ))

        # QK^T attention: batched (n_heads, 1, head_dim) x (n_heads, head_dim, ctx)
        ops.append(TransformerOperatorSpec(
            op_type="attention_qkv",
            input_shapes={"input": (n_heads, head_dim), "key": (n_heads, head_dim, spec.context_length)},
            output_shapes={"output": (n_heads, spec.context_length)},
            precision=spec.kv_precision,
            **common,
        ))

        # Softmax: (n_heads, 1, ctx)
        ops.append(TransformerOperatorSpec(
            op_type="softmax",
            input_shapes={"input": (n_heads, spec.context_length)},
            output_shapes={"output": (n_heads, spec.context_length)},
            precision=Precision.FP32,
            **common,
        ))

        # Score * V: batched (n_heads, 1, ctx) x (n_heads, ctx, head_dim)
        ops.append(TransformerOperatorSpec(
            op_type="attention_qkv",
            input_shapes={"input": (n_heads, spec.context_length), "key": (n_heads, spec.context_length, head_dim)},
            output_shapes={"output": (n_heads, head_dim)},
            precision=spec.kv_precision,
            **common,
        ))

        # O projection: (1, n_heads * head_dim) x (n_heads * head_dim, h)
        ops.append(TransformerOperatorSpec(
            op_type="gemm",
            input_shapes={"input": (seq, n_heads * head_dim), "weight": (n_heads * head_dim, h)},
            output_shapes={"output": (seq, h)},
            precision=spec.weight_precision,
            **common,
        ))

        # RMSNorm (pre-attention) - counted here
        ops.append(TransformerOperatorSpec(
            op_type="layernorm",
            input_shapes={"input": (seq, h)},
            output_shapes={"output": (seq, h)},
            precision=Precision.FP32,
            **common,
        ))

        # --- MLP block (SwiGLU) ---

        # Gate projection: (1, h) x (h, inter)
        ops.append(TransformerOperatorSpec(
            op_type="gemm",
            input_shapes={"input": (seq, h), "weight": (h, inter)},
            output_shapes={"output": (seq, inter)},
            precision=spec.weight_precision,
            **common,
        ))

        # Up projection: (1, h) x (h, inter)
        ops.append(TransformerOperatorSpec(
            op_type="gemm",
            input_shapes={"input": (seq, h), "weight": (h, inter)},
            output_shapes={"output": (seq, inter)},
            precision=spec.weight_precision,
            **common,
        ))

        # SiLU + gate*up (element-wise)
        ops.append(TransformerOperatorSpec(
            op_type="silu",
            input_shapes={"input": (seq, inter)},
            output_shapes={"output": (seq, inter)},
            precision=Precision.FP32,
            **common,
        ))

        # Down projection: (1, inter) x (inter, h)
        ops.append(TransformerOperatorSpec(
            op_type="gemm",
            input_shapes={"input": (seq, inter), "weight": (inter, h)},
            output_shapes={"output": (seq, h)},
            precision=spec.weight_precision,
            **common,
        ))

        # RMSNorm (pre-MLP)
        ops.append(TransformerOperatorSpec(
            op_type="layernorm",
            input_shapes={"input": (seq, h)},
            output_shapes={"output": (seq, h)},
            precision=Precision.FP32,
            **common,
        ))

    # Embedding lookup (not a GEMM but counted as one for resource purposes)
    ops.append(TransformerOperatorSpec(
        op_type="gemm",
        input_shapes={"input": (seq, 1), "weight": (spec.vocab_size, h)},
        output_shapes={"output": (seq, h)},
        precision=spec.weight_precision,
        **common,
    ))

    # LM head: (1, h) x (h, vocab_size)
    ops.append(TransformerOperatorSpec(
        op_type="gemm",
        input_shapes={"input": (seq, h), "weight": (h, spec.vocab_size)},
        output_shapes={"output": (seq, spec.vocab_size)},
        precision=spec.weight_precision,
        **common,
    ))

    return ops


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _op_key(op: TransformerOperatorSpec) -> tuple:
    """Generate a hashable key for operator equivalence."""
    inp = tuple(sorted((k, tuple(v)) for k, v in op.input_shapes.items()))
    out = tuple(sorted((k, tuple(v)) for k, v in op.output_shapes.items()))
    return (op.op_type, inp, out, op.precision)


def deduplicate_operators(
    ops: list[TransformerOperatorSpec],
) -> list[TransformerOperatorSpec]:
    """Group identical operators and return the unique set.

    Two operators are identical if they share (op_type, input_shapes,
    output_shapes, precision).  Target board and clock are constant
    across all operators in a single orchestration run.
    """
    seen: dict[tuple, TransformerOperatorSpec] = {}
    for op in ops:
        key = _op_key(op)
        if key not in seen:
            seen[key] = op
    return list(seen.values())


# ---------------------------------------------------------------------------
# Resource budget checking
# ---------------------------------------------------------------------------


def check_resource_budget(
    results: list[KernelOptimizationResult],
    board: BoardSpec,
) -> dict[str, dict[str, float]]:
    """Verify that total resource usage of all kernels fits on the FPGA.

    Returns a dict with keys ``"bram"``, ``"dsp"``, ``"lut"``, ``"ff"``
    each mapping to ``{"used": ..., "available": ..., "fits": ...}``.

    Assumes a static bitstream where all kernels coexist simultaneously.
    """
    totals: dict[str, float] = {"bram": 0, "dsp": 0, "lut": 0, "ff": 0}

    for r in results:
        if r.final_synthesis is None:
            continue
        s = r.final_synthesis
        totals["bram"] += s.bram_used or 0
        totals["dsp"] += s.dsp_used or 0
        totals["lut"] += s.lut_used or 0
        totals["ff"] += s.ff_used or 0

    # Board limits - use synthesis-level counts (not the _kb/_count board fields)
    # For boards with known FPGA fields, use them; otherwise report "unknown"
    limits = {
        "bram": float(board.bram_kb) if board.bram_kb is not None else 0.0,
        "dsp": float(board.dsp_slices) if board.dsp_slices is not None else 0.0,
        "lut": float(board.lut_count * 1000) if board.lut_count is not None else 0.0,
        "ff": float(board.lut_count * 2000) if board.lut_count is not None else 0.0,
    }

    budget: dict[str, dict[str, float]] = {}
    for resource in ("bram", "dsp", "lut", "ff"):
        used = totals[resource]
        available = limits[resource]
        budget[resource] = {
            "used": used,
            "available": available,
            "fits": float(used <= available) if available > 0 else 0.0,
        }
    return budget
