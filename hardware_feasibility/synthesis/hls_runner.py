"""Invoke Vitis HLS programmatically and capture results."""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from .types import KernelSpec, HLSSynthesisResult, HLSCoSimResult
from .report_parser import parse_synthesis_report


# Default timeout for HLS synthesis (10 minutes)
_DEFAULT_TIMEOUT_SEC = 600

# Tcl script template for synthesis
_SYNTH_TCL_TEMPLATE = """\
open_project -reset {project_name}
set_top {top_function}
add_files {source_file}
open_solution -reset "solution1"
set_part {{{target_device}}}
create_clock -period {clock_period_ns} -name default
csynth_design
exit
"""

# Tcl script template for co-simulation
_COSIM_TCL_TEMPLATE = """\
open_project -reset {project_name}
set_top {top_function}
add_files {source_file}
add_files -tb {testbench_file}
open_solution -reset "solution1"
set_part {{{target_device}}}
create_clock -period {clock_period_ns} -name default
csim_design
csynth_design
cosim_design
exit
"""


def generate_synth_tcl(kernel: KernelSpec, source_file: str, project_name: str = "hls_project") -> str:
    """Generate a Vitis HLS Tcl script for synthesis."""
    return _SYNTH_TCL_TEMPLATE.format(
        project_name=project_name,
        top_function=kernel.top_function,
        source_file=source_file,
        target_device=kernel.target_device,
        clock_period_ns=kernel.clock_period_ns,
    )


def generate_cosim_tcl(
    kernel: KernelSpec,
    source_file: str,
    testbench_file: str,
    project_name: str = "hls_project",
) -> str:
    """Generate a Vitis HLS Tcl script for co-simulation."""
    return _COSIM_TCL_TEMPLATE.format(
        project_name=project_name,
        top_function=kernel.top_function,
        source_file=source_file,
        testbench_file=testbench_file,
        target_device=kernel.target_device,
        clock_period_ns=kernel.clock_period_ns,
    )


class HLSRunner:
    """Manages Vitis HLS invocations."""

    def __init__(self, vitis_hls_path: str = "vitis_hls"):
        self.vitis_hls_path = vitis_hls_path
        self._verify_installation()

    def _verify_installation(self) -> None:
        """Check that vitis_hls is available on PATH."""
        try:
            result = subprocess.run(
                [self.vitis_hls_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"vitis_hls --version failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"vitis_hls not found at '{self.vitis_hls_path}'. "
                "Install Vitis HLS or provide the correct path."
            )

    def synthesize(
        self,
        kernel: KernelSpec,
        work_dir: Optional[Path] = None,
        timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    ) -> HLSSynthesisResult:
        """Run HLS synthesis and return parsed results.

        1. Write kernel source to a temp directory.
        2. Generate a Tcl script.
        3. Invoke vitis_hls -f script.tcl.
        4. Parse the synthesis report XML.
        5. Return HLSSynthesisResult.
        """
        cleanup = work_dir is None
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="model2hw_hls_"))

        try:
            return self._run_synthesis(kernel, work_dir, timeout_sec)
        finally:
            if cleanup:
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)

    def _run_synthesis(
        self,
        kernel: KernelSpec,
        work_dir: Path,
        timeout_sec: int,
    ) -> HLSSynthesisResult:
        """Internal synthesis execution."""
        work_dir.mkdir(parents=True, exist_ok=True)
        project_name = "hls_project"

        # Write source file
        source_path = work_dir / f"{kernel.name}.cpp"
        source_path.write_text(kernel.source_code)

        # Generate and write Tcl script
        tcl_content = generate_synth_tcl(kernel, str(source_path), project_name)
        tcl_path = work_dir / "run_hls.tcl"
        tcl_path.write_text(tcl_content)

        # Run Vitis HLS
        try:
            result = subprocess.run(
                [self.vitis_hls_path, "-f", str(tcl_path)],
                capture_output=True,
                text=True,
                cwd=str(work_dir),
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return HLSSynthesisResult(
                success=False,
                error_message=f"HLS synthesis timed out after {timeout_sec}s.",
            )

        if result.returncode != 0:
            return HLSSynthesisResult(
                success=False,
                error_message=f"HLS synthesis failed (rc={result.returncode}): {result.stderr[:2000]}",
                raw_report=result.stdout[:5000],
            )

        # Parse report
        report_path = (
            work_dir / project_name / "solution1" / "syn" / "report"
            / f"{kernel.top_function}_csynth.xml"
        )
        if not report_path.exists():
            return HLSSynthesisResult(
                success=False,
                error_message=f"Synthesis report not found at {report_path}",
                raw_report=result.stdout[:5000],
            )

        return parse_synthesis_report(report_path)

    def cosim(
        self,
        kernel: KernelSpec,
        work_dir: Optional[Path] = None,
        timeout_sec: int = _DEFAULT_TIMEOUT_SEC,
    ) -> HLSCoSimResult:
        """Run HLS co-simulation for functional verification."""
        if kernel.testbench_code is None:
            return HLSCoSimResult(
                passed=False,
                error_output="No testbench provided for co-simulation.",
            )

        cleanup = work_dir is None
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="model2hw_cosim_"))

        try:
            return self._run_cosim(kernel, work_dir, timeout_sec)
        finally:
            if cleanup:
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)

    def _run_cosim(
        self,
        kernel: KernelSpec,
        work_dir: Path,
        timeout_sec: int,
    ) -> HLSCoSimResult:
        """Internal co-simulation execution."""
        work_dir.mkdir(parents=True, exist_ok=True)
        project_name = "hls_project"

        # Write source and testbench
        source_path = work_dir / f"{kernel.name}.cpp"
        source_path.write_text(kernel.source_code)
        tb_path = work_dir / f"{kernel.name}_tb.cpp"
        tb_path.write_text(kernel.testbench_code)

        # Generate and write Tcl script
        tcl_content = generate_cosim_tcl(
            kernel, str(source_path), str(tb_path), project_name
        )
        tcl_path = work_dir / "run_cosim.tcl"
        tcl_path.write_text(tcl_content)

        # Run
        start = time.monotonic()
        try:
            result = subprocess.run(
                [self.vitis_hls_path, "-f", str(tcl_path)],
                capture_output=True,
                text=True,
                cwd=str(work_dir),
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - start) * 1000
            return HLSCoSimResult(
                passed=False,
                error_output=f"Co-simulation timed out after {timeout_sec}s.",
                runtime_ms=elapsed,
            )

        elapsed = (time.monotonic() - start) * 1000
        passed = result.returncode == 0

        return HLSCoSimResult(
            passed=passed,
            error_output=result.stderr[:2000] if not passed else None,
            runtime_ms=elapsed,
        )
