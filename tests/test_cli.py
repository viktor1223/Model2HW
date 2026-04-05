"""End-to-end CLI integration tests."""

from __future__ import annotations

import json
import pytest

from hardware_feasibility.cli import run


# ---------------------------------------------------------------------------
# Smoke tests: basic invocations should not raise
# ---------------------------------------------------------------------------

class TestCLISmoke:
    def test_list_families(self, capsys: pytest.CaptureFixture):
        run(["--list-families"])
        out = capsys.readouterr().out
        assert "llama3-8b" in out

    def test_list_boards(self, capsys: pytest.CaptureFixture):
        run(["--list-boards"])
        out = capsys.readouterr().out
        assert "Xilinx VCK190" in out

    def test_family_report(self, capsys: pytest.CaptureFixture):
        run([
            "--family", "llama3-8b",
            "--weight-precision", "int4",
            "--kv-precision", "int8",
            "--context-length", "2048",
            "--target-tok-s", "10",
            "--board", "Xilinx VCK190",
        ])
        out = capsys.readouterr().out
        assert "HARDWARE FEASIBILITY REPORT" in out
        assert "llama3-8b" in out

    def test_family_json(self, capsys: pytest.CaptureFixture):
        run([
            "--family", "qwen2-0.5b",
            "--json",
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "model" in data
        assert data["model"]["name"] == "qwen2-0.5b"


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------

class TestCLIReportContent:
    def test_verdict_in_report(self, capsys: pytest.CaptureFixture):
        run([
            "--family", "llama3-8b",
            "--weight-precision", "fp16",
            "--board", "NVIDIA A100 80GB",
        ])
        out = capsys.readouterr().out
        assert "VERDICT" in out

    def test_match_all(self, capsys: pytest.CaptureFixture):
        run([
            "--family", "qwen2-0.5b",
            "--weight-precision", "int4",
            "--kv-precision", "int8",
            "--match-all",
        ])
        out = capsys.readouterr().out
        assert "HARDWARE MATCH RANKING" in out

    def test_board_category_filter(self, capsys: pytest.CaptureFixture):
        run([
            "--family", "qwen2-0.5b",
            "--board-category", "fpga",
        ])
        out = capsys.readouterr().out
        assert "HARDWARE MATCH RANKING" in out


# ---------------------------------------------------------------------------
# JSON output structure
# ---------------------------------------------------------------------------

class TestCLIJsonOutput:
    def test_json_has_all_sections(self, capsys: pytest.CaptureFixture):
        run([
            "--family", "llama3-8b",
            "--weight-precision", "int4",
            "--board", "Xilinx VCK190",
            "--json",
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        for key in ("model", "runtime_assumptions", "memory", "bandwidth",
                     "compute", "host_io", "verdict"):
            assert key in data, f"Missing key: {key}"

    def test_json_match_all(self, capsys: pytest.CaptureFixture):
        run([
            "--family", "qwen2-0.5b",
            "--json",
            "--match-all",
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "hardware_matches" in data
        assert len(data["hardware_matches"]) > 0

    def test_json_verdict_value(self, capsys: pytest.CaptureFixture):
        run([
            "--family", "qwen2-0.5b",
            "--weight-precision", "int4",
            "--kv-precision", "int8",
            "--board", "Xilinx VCK190",
            "--json",
        ])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["verdict"]["verdict"] == "fits_comfortably"


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

class TestCLIFileOutput:
    def test_output_to_file(self, tmp_path, capsys: pytest.CaptureFixture):
        outfile = tmp_path / "report.txt"
        run([
            "--family", "qwen2-0.5b",
            "--output", str(outfile),
        ])
        assert outfile.exists()
        content = outfile.read_text()
        assert "HARDWARE FEASIBILITY REPORT" in content

    def test_json_output_to_file(self, tmp_path, capsys: pytest.CaptureFixture):
        outfile = tmp_path / "report.json"
        run([
            "--family", "qwen2-0.5b",
            "--json",
            "--output", str(outfile),
        ])
        assert outfile.exists()
        data = json.loads(outfile.read_text())
        assert "model" in data


# ---------------------------------------------------------------------------
# Custom hardware targets
# ---------------------------------------------------------------------------

class TestCLICustomTargets:
    def test_custom_memory_target(self, capsys: pytest.CaptureFixture):
        run([
            "--family", "qwen2-0.5b",
            "--target-memory-gb", "4.0",
        ])
        out = capsys.readouterr().out
        assert "VERDICT" in out

    def test_custom_bw_target(self, capsys: pytest.CaptureFixture):
        run([
            "--family", "qwen2-0.5b",
            "--target-memory-gb", "4.0",
            "--target-bw-gbps", "12.8",
        ])
        out = capsys.readouterr().out
        assert "VERDICT" in out


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestCLIErrors:
    def test_unknown_board_exits(self):
        with pytest.raises(ValueError, match="Unknown board"):
            run(["--family", "llama3-8b", "--board", "Fake Board 9000"])

    def test_unknown_family_exits(self):
        with pytest.raises(SystemExit):
            run(["--family", "not-a-real-model"])
