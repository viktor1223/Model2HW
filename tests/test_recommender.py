"""Tests for the configuration recommender (Phase 3)."""

from __future__ import annotations

import json

import pytest

from hardware_feasibility.models.architecture_rules import Precision
from hardware_feasibility.models.hf_config_loader import load_from_known_family
from hardware_feasibility.hardware.board_specs import get_board
from hardware_feasibility.analysis.recommender import recommend_configuration
from hardware_feasibility.outputs.report import format_recommend_report
from hardware_feasibility.outputs.json_export import format_recommend_json


def test_recommend_llama3_8b_on_vck190():
    """Should return at least one recommendation for llama3-8b on VCK190."""
    spec = load_from_known_family("llama3-8b")
    board = get_board("Xilinx VCK190")
    result = recommend_configuration(spec, board)
    assert result.infeasible_reason is None
    assert len(result.recommendations) >= 1
    # Best should be INT4 (smallest memory footprint, highest tok/s)
    best = result.recommendations[0]
    assert best.weight_precision == Precision.INT4


def test_recommend_infeasible_large_model():
    """llama2-13b on ZCU104 (2 GB) should return infeasible."""
    spec = load_from_known_family("llama2-13b")
    board = get_board("Xilinx ZCU104")
    result = recommend_configuration(spec, board)
    assert result.infeasible_reason is not None
    assert len(result.recommendations) == 0


def test_recommend_sorted_by_tok_s():
    """Recommendations should be sorted by estimated tok/s descending."""
    spec = load_from_known_family("llama3-8b")
    board = get_board("Xilinx VCK190")
    result = recommend_configuration(spec, board)
    tok_rates = [r.estimated_tok_per_sec for r in result.recommendations]
    assert tok_rates == sorted(tok_rates, reverse=True)


def test_recommend_rationale_populated():
    """Every recommendation should have at least one rationale string."""
    spec = load_from_known_family("llama3-8b")
    board = get_board("Xilinx VCK190")
    result = recommend_configuration(spec, board)
    for rec in result.recommendations[:5]:
        assert len(rec.rationale) >= 1


def test_recommend_report_format():
    """format_recommend_report should produce expected sections."""
    spec = load_from_known_family("llama3.2-1b")
    board = get_board("Xilinx VCK190")
    result = recommend_configuration(spec, board)
    report = format_recommend_report(result)
    assert "CONFIGURATION RECOMMENDATIONS" in report
    assert "llama3.2-1b" in report


def test_recommend_json_format():
    """format_recommend_json should return valid structured data."""
    spec = load_from_known_family("llama3.2-1b")
    board = get_board("Xilinx VCK190")
    result = recommend_configuration(spec, board)
    d = format_recommend_json(result)
    assert d["model"] == "llama3.2-1b"
    assert d["target_board"] == "Xilinx VCK190"
    assert isinstance(d["recommendations"], list)
    json.dumps(d)
