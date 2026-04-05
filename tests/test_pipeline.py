"""Tests for Phase 10: End-to-End Pipeline."""

from __future__ import annotations

import json

import pytest

from hardware_feasibility.pipeline import PipelineResult, run_full_pipeline
from hardware_feasibility.hardware.board_specs import BoardSpec
from hardware_feasibility.models.architecture_rules import ModelSpec, Precision
from hardware_feasibility.models.hf_config_loader import load_from_known_family


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


def _make_small_board() -> BoardSpec:
    """A tiny board where most models won't fit."""
    return BoardSpec(
        name="Tiny Board",
        category="fpga",
        memory_gb=0.5,
        memory_bandwidth_gbps=10.0,
        host_link_bandwidth_gbps=2.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_result_fields(self) -> None:
        spec = load_from_known_family("qwen2-0.5b", weight_precision="int4")
        board = _make_board()
        result = run_full_pipeline(spec, board)
        assert result.model_name == "qwen2-0.5b"
        assert result.target_board == "Xilinx VCK190"
        assert result.initial_verdict is not None
        assert result.pipeline_version == "0.1.0"
        assert result.total_runtime_seconds >= 0


class TestPipelineStages:
    def test_fitting_model_skips_sweep(self) -> None:
        """When model fits at default precision, sweep should be None."""
        spec = load_from_known_family("qwen2-0.5b", weight_precision="int4")
        board = _make_board()
        result = run_full_pipeline(spec, board)
        # INT4 Qwen2-0.5B on 32GB board should fit
        if result.initial_verdict.verdict.value != "does_not_fit":
            assert result.precision_sweep is None

    def test_non_fitting_model_runs_sweep(self) -> None:
        """When model doesn't fit, pipeline should run precision sweep."""
        spec = load_from_known_family("llama3-8b", weight_precision="fp16")
        board = _make_small_board()
        result = run_full_pipeline(spec, board)
        # FP16 Llama3-8B on 0.5 GB board should not fit
        assert result.initial_verdict.verdict.value == "does_not_fit"
        assert result.precision_sweep is not None

    def test_recommendation_always_present(self) -> None:
        spec = load_from_known_family("qwen2-0.5b", weight_precision="int8")
        board = _make_board()
        result = run_full_pipeline(spec, board)
        assert result.recommendation is not None

    def test_sensitivity_always_present(self) -> None:
        spec = load_from_known_family("qwen2-0.5b", weight_precision="int8")
        board = _make_board()
        result = run_full_pipeline(spec, board)
        assert result.sensitivity is not None
        assert result.sensitivity.bottleneck is not None


class TestPipelineReport:
    def test_text_report(self) -> None:
        from hardware_feasibility.outputs.report import format_pipeline_report

        spec = load_from_known_family("qwen2-0.5b", weight_precision="int4")
        board = _make_board()
        result = run_full_pipeline(spec, board)
        report = format_pipeline_report(result)
        assert "FULL PIPELINE REPORT" in report
        assert "qwen2-0.5b" in report
        assert "Pipeline completed" in report

    def test_json_output(self) -> None:
        from hardware_feasibility.outputs.json_export import format_pipeline_json

        spec = load_from_known_family("qwen2-0.5b", weight_precision="int4")
        board = _make_board()
        result = run_full_pipeline(spec, board)
        data = format_pipeline_json(result)

        # Should be JSON-serializable
        serialized = json.dumps(data)
        assert len(serialized) > 0

        assert data["model"] == "qwen2-0.5b"
        assert data["target_board"] == "Xilinx VCK190"
        assert "initial_verdict" in data
        assert "recommendation" in data
        assert "sensitivity" in data


class TestPipelineDecomposition:
    def test_decomposition_for_large_model(self) -> None:
        """A model that doesn't fit at any precision triggers decomposition."""
        spec = load_from_known_family("llama3-8b", weight_precision="fp16")
        board = _make_small_board()
        result = run_full_pipeline(spec, board)
        # On a 0.5 GB board, even INT4 llama3-8b won't fit
        if result.precision_sweep is not None and not result.precision_sweep.fitting_points:
            assert result.decomposition is not None
