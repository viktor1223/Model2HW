"""Tests for HLS synthesis infrastructure (Phase 6)."""

import pytest
from pathlib import Path

from hardware_feasibility.synthesis.types import (
    KernelSpec,
    HLSSynthesisResult,
    HLSCoSimResult,
)
from hardware_feasibility.synthesis.report_parser import parse_synthesis_report
from hardware_feasibility.synthesis.hls_runner import (
    generate_synth_tcl,
    generate_cosim_tcl,
    HLSRunner,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _sample_kernel() -> KernelSpec:
    return KernelSpec(
        name="matmul_int8",
        source_code="void matmul_int8(int a[16][16], int b[16][16], int c[16][16]) { /* ... */ }",
        testbench_code="int main() { return 0; }",
        target_device="xcvc1902-vsva2197-2MP-e-S",
        clock_period_ns=5.0,
        top_function="matmul_int8",
    )


def test_hls_runner_init_fails_without_vitis():
    """Verify clean RuntimeError if vitis_hls is not installed."""
    with pytest.raises(RuntimeError, match="vitis_hls not found"):
        HLSRunner(vitis_hls_path="/nonexistent/vitis_hls")


def test_tcl_script_generation():
    """Verify the Tcl synthesis script is generated correctly."""
    kernel = _sample_kernel()
    tcl = generate_synth_tcl(kernel, "/tmp/matmul_int8.cpp", "test_project")

    assert "open_project -reset test_project" in tcl
    assert "set_top matmul_int8" in tcl
    assert "add_files /tmp/matmul_int8.cpp" in tcl
    assert "set_part {xcvc1902-vsva2197-2MP-e-S}" in tcl
    assert "create_clock -period 5.0 -name default" in tcl
    assert "csynth_design" in tcl
    assert "exit" in tcl


def test_cosim_tcl_script_generation():
    """Verify the Tcl co-sim script includes testbench."""
    kernel = _sample_kernel()
    tcl = generate_cosim_tcl(kernel, "/tmp/src.cpp", "/tmp/tb.cpp", "test_project")

    assert "add_files -tb /tmp/tb.cpp" in tcl
    assert "csim_design" in tcl
    assert "cosim_design" in tcl


def test_report_parser_sample():
    """Parse a sample Vitis HLS synthesis XML report."""
    report_path = FIXTURES_DIR / "sample_csynth.xml"
    result = parse_synthesis_report(report_path)

    assert result.success is True
    assert result.error_message is None

    # Timing
    assert result.clock_target_ns == 5.0
    assert result.clock_period_ns == 4.35
    assert result.meets_timing is True

    # Latency
    assert result.estimated_latency_cycles == 1024
    assert result.estimated_latency_ns == pytest.approx(5120.0)

    # Resources
    assert result.bram_used == 64
    assert result.bram_available == 1934
    assert result.bram_utilization == pytest.approx(64 / 1934)

    assert result.dsp_used == 128
    assert result.dsp_available == 1968
    assert result.dsp_utilization == pytest.approx(128 / 1968)

    assert result.ff_used == 45000
    assert result.lut_used == 32000
    assert result.lut_available == 899840

    # Raw report should be populated
    assert len(result.raw_report) > 0


def test_report_parser_malformed_xml(tmp_path):
    """Gracefully handle a malformed XML file."""
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text("<not><valid>xml")
    result = parse_synthesis_report(bad_xml)

    assert result.success is False
    assert "XML parse error" in result.error_message


def test_synthesis_result_utilization_properties():
    """Verify utilization property calculations."""
    r = HLSSynthesisResult(
        success=True,
        bram_used=100,
        bram_available=1000,
        dsp_used=50,
        dsp_available=200,
        ff_used=None,
        ff_available=None,
        lut_used=0,
        lut_available=0,
    )

    assert r.bram_utilization == pytest.approx(0.1)
    assert r.dsp_utilization == pytest.approx(0.25)
    assert r.ff_utilization is None
    assert r.lut_utilization is None  # available is 0


def test_synthesis_result_meets_timing():
    """Verify meets_timing property."""
    passing = HLSSynthesisResult(success=True, clock_period_ns=4.0, clock_target_ns=5.0)
    assert passing.meets_timing is True

    failing = HLSSynthesisResult(success=True, clock_period_ns=6.0, clock_target_ns=5.0)
    assert failing.meets_timing is False

    unknown = HLSSynthesisResult(success=True, clock_period_ns=None, clock_target_ns=None)
    assert unknown.meets_timing is None


def test_kernel_spec_construction():
    """Verify KernelSpec can be constructed with all fields."""
    kernel = _sample_kernel()
    assert kernel.name == "matmul_int8"
    assert kernel.clock_period_ns == 5.0
    assert kernel.testbench_code is not None


def test_cosim_result_construction():
    """Verify HLSCoSimResult dataclass."""
    r = HLSCoSimResult(passed=True, runtime_ms=1234.5)
    assert r.passed is True
    assert r.error_output is None
    assert r.runtime_ms == 1234.5
