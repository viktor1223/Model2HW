"""Tests for Phase 8: Multi-Kernel Orchestration."""

from __future__ import annotations

import pytest

from hardware_feasibility.agents.orchestrator import (
    decompose_model_to_operators,
    deduplicate_operators,
    check_resource_budget,
)
from hardware_feasibility.agents.types import (
    KernelOptimizationResult,
    TransformerOperatorSpec,
)
from hardware_feasibility.hardware.board_specs import BoardSpec
from hardware_feasibility.models.architecture_rules import ModelSpec, Precision
from hardware_feasibility.synthesis.types import HLSSynthesisResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_board() -> BoardSpec:
    return BoardSpec(
        name="Xilinx VCK190",
        category="fpga",
        memory_gb=32.0,
        memory_bandwidth_gbps=102.4,
        peak_tops_int8=400.0,
        host_link="PCIe Gen4 x8",
        host_link_bandwidth_gbps=16.0,
        bram_kb=32400,
        uram_kb=36000,
        dsp_slices=1968,
        lut_count=899,
    )


def _make_qwen05b() -> ModelSpec:
    """Qwen2-0.5B - smallest model for fast tests."""
    return ModelSpec(
        name="qwen2-0.5b",
        params=500_000_000,
        num_layers=24,
        hidden_size=896,
        num_attention_heads=14,
        num_kv_heads=2,
        intermediate_size=4864,
        vocab_size=151936,
        weight_precision=Precision.INT8,
        kv_precision=Precision.INT8,
        context_length=2048,
    )


def _make_llama3_8b() -> ModelSpec:
    """Llama 3-8B for operator count verification."""
    return ModelSpec(
        name="llama3-8b",
        params=8_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
        weight_precision=Precision.INT8,
        kv_precision=Precision.INT8,
        context_length=4096,
    )


# ---------------------------------------------------------------------------
# Tests: decompose_model_to_operators
# ---------------------------------------------------------------------------


class TestDecomposeModel:
    def test_qwen05b_operator_count(self) -> None:
        """Qwen2-0.5B: 24 layers x 13 ops + 2 (embed + LM head) = 314."""
        spec = _make_qwen05b()
        board = _make_board()
        ops = decompose_model_to_operators(spec, board)
        # 13 operators per layer + 2 global
        assert len(ops) == 24 * 13 + 2

    def test_llama3_8b_operator_count(self) -> None:
        """Llama 3-8B: 32 layers x 13 ops + 2 = 418."""
        spec = _make_llama3_8b()
        board = _make_board()
        ops = decompose_model_to_operators(spec, board)
        assert len(ops) == 32 * 13 + 2

    def test_operator_types_present(self) -> None:
        spec = _make_qwen05b()
        board = _make_board()
        ops = decompose_model_to_operators(spec, board)
        types = {op.op_type for op in ops}
        assert "gemm" in types
        assert "attention_qkv" in types
        assert "softmax" in types
        assert "layernorm" in types
        assert "silu" in types

    def test_precision_assignment(self) -> None:
        """Weight projections use weight_precision; softmax uses FP32."""
        spec = _make_qwen05b()
        board = _make_board()
        ops = decompose_model_to_operators(spec, board)
        softmax_ops = [op for op in ops if op.op_type == "softmax"]
        assert all(op.precision == Precision.FP32 for op in softmax_ops)

        gemm_ops = [op for op in ops if op.op_type == "gemm"]
        # Most GEMMs use weight precision (INT8 for this spec)
        weight_gemms = [
            op for op in gemm_ops if op.precision == spec.weight_precision
        ]
        assert len(weight_gemms) > 0


# ---------------------------------------------------------------------------
# Tests: deduplicate_operators
# ---------------------------------------------------------------------------


class TestDeduplicateOperators:
    def test_qwen05b_dedup(self) -> None:
        """Qwen2-0.5B should deduplicate to roughly 8-12 unique kernels."""
        spec = _make_qwen05b()
        board = _make_board()
        ops = decompose_model_to_operators(spec, board)
        unique = deduplicate_operators(ops)
        # Should be much fewer than total operators
        assert len(unique) < len(ops)
        # Expect roughly 8-12 unique operator shapes
        assert 5 <= len(unique) <= 20

    def test_llama3_8b_dedup(self) -> None:
        """Llama 3-8B should deduplicate to roughly 8-12 unique kernels."""
        spec = _make_llama3_8b()
        board = _make_board()
        ops = decompose_model_to_operators(spec, board)
        unique = deduplicate_operators(ops)
        assert len(unique) < len(ops)
        assert 5 <= len(unique) <= 20

    def test_identical_ops_dedup(self) -> None:
        """Two identical operators should deduplicate to one."""
        board = _make_board()
        op = TransformerOperatorSpec(
            op_type="gemm",
            input_shapes={"input": (1, 896), "weight": (896, 896)},
            output_shapes={"output": (1, 896)},
            precision=Precision.INT8,
            target_board=board,
        )
        unique = deduplicate_operators([op, op, op])
        assert len(unique) == 1


# ---------------------------------------------------------------------------
# Tests: check_resource_budget
# ---------------------------------------------------------------------------


class TestCheckResourceBudget:
    def test_fits_within_budget(self) -> None:
        board = _make_board()
        synth = HLSSynthesisResult(
            success=True,
            bram_used=100, bram_available=32400,
            dsp_used=50, dsp_available=1968,
            lut_used=1000, lut_available=899000,
            ff_used=500, ff_available=1798000,
        )
        op = TransformerOperatorSpec(
            op_type="gemm",
            input_shapes={"input": (1, 896), "weight": (896, 896)},
            output_shapes={"output": (1, 896)},
            precision=Precision.INT8,
            target_board=board,
        )
        result = KernelOptimizationResult(
            operator=op,
            final_source="code",
            final_synthesis=synth,
            total_iterations=1,
            converged=True,
        )
        budget = check_resource_budget([result, result], board)
        assert budget["bram"]["fits"] == 1.0
        assert budget["dsp"]["fits"] == 1.0
        assert budget["bram"]["used"] == 200
        assert budget["dsp"]["used"] == 100

    def test_exceeds_dsp_budget(self) -> None:
        board = _make_board()
        synth = HLSSynthesisResult(
            success=True,
            bram_used=100, bram_available=32400,
            dsp_used=1500, dsp_available=1968,
            lut_used=1000, lut_available=899000,
            ff_used=500, ff_available=1798000,
        )
        op = TransformerOperatorSpec(
            op_type="gemm",
            input_shapes={"input": (1, 896), "weight": (896, 896)},
            output_shapes={"output": (1, 896)},
            precision=Precision.INT8,
            target_board=board,
        )
        result = KernelOptimizationResult(
            operator=op,
            final_source="code",
            final_synthesis=synth,
            total_iterations=1,
            converged=True,
        )
        # Two kernels each using 1500 DSPs = 3000 > 1968
        budget = check_resource_budget([result, result], board)
        assert budget["dsp"]["fits"] == 0.0
        assert budget["dsp"]["used"] == 3000
