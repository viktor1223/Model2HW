"""Tests for board_specs and hardware matcher."""

from __future__ import annotations

import pytest

from hardware_feasibility.hardware.board_specs import (
    BOARD_DATABASE,
    BoardSpec,
    get_board,
    list_boards,
)
from hardware_feasibility.hardware.matcher import (
    MatchResult,
    match_board,
    rank_boards,
)
from hardware_feasibility.models.architecture_rules import ModelSpec
from hardware_feasibility.analysis.memory import analyze_memory
from hardware_feasibility.analysis.bandwidth import analyze_bandwidth
from hardware_feasibility.analysis.io import analyze_io
from hardware_feasibility.analysis.verdict import FeasibilityVerdict


# ---------------------------------------------------------------------------
# Board database
# ---------------------------------------------------------------------------

class TestBoardDatabase:
    def test_has_boards(self):
        assert len(BOARD_DATABASE) > 5

    def test_all_entries_are_board_spec(self):
        for name, board in BOARD_DATABASE.items():
            assert isinstance(board, BoardSpec), f"{name} is not a BoardSpec"

    def test_all_have_positive_memory(self):
        for name, board in BOARD_DATABASE.items():
            assert board.memory_gb >= 0, f"{name} has negative memory"

    def test_categories(self):
        categories = {b.category for b in BOARD_DATABASE.values()}
        assert "fpga" in categories
        assert "gpu" in categories


class TestGetBoard:
    def test_known_board(self):
        board = get_board("Xilinx VCK190")
        assert board.name == "Xilinx VCK190"
        assert board.memory_gb == 8.0
        assert board.category == "fpga"

    def test_unknown_board_raises(self):
        with pytest.raises(ValueError, match="Unknown board"):
            get_board("Nonexistent Board XYZ")


class TestListBoards:
    def test_returns_all_when_no_filter(self):
        boards = list_boards()
        assert len(boards) == len(BOARD_DATABASE)

    def test_filter_by_category(self):
        fpgas = list_boards(category="fpga")
        assert len(fpgas) > 0
        assert all(b.category == "fpga" for b in fpgas)

    def test_filter_empty_category(self):
        result = list_boards(category="nonexistent_category")
        assert result == []

    def test_sorted_by_memory(self):
        boards = list_boards()
        memories = [b.memory_gb for b in boards]
        assert memories == sorted(memories)


# ---------------------------------------------------------------------------
# Hardware matcher
# ---------------------------------------------------------------------------

class TestMatchBoard:
    def test_match_returns_match_result(self, qwen2_05b_int4: ModelSpec):
        board = get_board("Xilinx VCK190")
        mem = analyze_memory(qwen2_05b_int4)
        bw = analyze_bandwidth(qwen2_05b_int4)
        io = analyze_io(qwen2_05b_int4)
        result = match_board(qwen2_05b_int4, board, mem, bw, io)
        assert isinstance(result, MatchResult)

    def test_small_model_fits_vck190(self, qwen2_05b_int4: ModelSpec):
        board = get_board("Xilinx VCK190")
        mem = analyze_memory(qwen2_05b_int4)
        bw = analyze_bandwidth(qwen2_05b_int4)
        io = analyze_io(qwen2_05b_int4)
        result = match_board(qwen2_05b_int4, board, mem, bw, io)
        assert result.fits

    def test_large_model_no_fit_small_board(self, llama3_8b_fp16: ModelSpec):
        board = get_board("Xilinx ZCU104")  # 2 GB
        mem = analyze_memory(llama3_8b_fp16)
        bw = analyze_bandwidth(llama3_8b_fp16)
        io = analyze_io(llama3_8b_fp16)
        result = match_board(llama3_8b_fp16, board, mem, bw, io)
        assert not result.fits
        assert result.estimated_tok_per_sec == 0.0

    def test_memory_utilization(self, qwen2_05b_int4: ModelSpec):
        board = get_board("Xilinx VCK190")
        mem = analyze_memory(qwen2_05b_int4)
        bw = analyze_bandwidth(qwen2_05b_int4)
        io = analyze_io(qwen2_05b_int4)
        result = match_board(qwen2_05b_int4, board, mem, bw, io)
        expected_util = mem.total_gb / board.memory_gb
        assert result.memory_utilization == pytest.approx(expected_util)

    def test_estimated_tok_s_positive_when_fits(self, qwen2_05b_int4: ModelSpec):
        board = get_board("Xilinx VCK190")
        mem = analyze_memory(qwen2_05b_int4)
        bw = analyze_bandwidth(qwen2_05b_int4)
        io = analyze_io(qwen2_05b_int4)
        result = match_board(qwen2_05b_int4, board, mem, bw, io)
        assert result.estimated_tok_per_sec is not None
        assert result.estimated_tok_per_sec > 0


class TestRankBoards:
    def test_returns_list(self, qwen2_05b_int4: ModelSpec):
        mem = analyze_memory(qwen2_05b_int4)
        bw = analyze_bandwidth(qwen2_05b_int4)
        io = analyze_io(qwen2_05b_int4)
        results = rank_boards(qwen2_05b_int4, mem, bw, io)
        assert isinstance(results, list)
        assert len(results) == len(BOARD_DATABASE)

    def test_fitting_boards_first(self, qwen2_05b_int4: ModelSpec):
        mem = analyze_memory(qwen2_05b_int4)
        bw = analyze_bandwidth(qwen2_05b_int4)
        io = analyze_io(qwen2_05b_int4)
        results = rank_boards(qwen2_05b_int4, mem, bw, io)
        # Verify that fitting boards come before non-fitting
        saw_non_fit = False
        for r in results:
            if not r.fits:
                saw_non_fit = True
            elif saw_non_fit:
                pytest.fail("Fitting board found after non-fitting board in ranking")

    def test_category_filter(self, qwen2_05b_int4: ModelSpec):
        mem = analyze_memory(qwen2_05b_int4)
        bw = analyze_bandwidth(qwen2_05b_int4)
        io = analyze_io(qwen2_05b_int4)
        results = rank_boards(qwen2_05b_int4, mem, bw, io, category="fpga")
        assert all(r.board.category == "fpga" for r in results)
        assert len(results) > 0
        assert len(results) < len(BOARD_DATABASE)
