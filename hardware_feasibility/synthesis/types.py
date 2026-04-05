"""Data models for HLS synthesis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KernelSpec:
    """Specification of an HLS kernel to synthesize."""

    name: str
    source_code: str                   # C++ source as a string
    testbench_code: Optional[str]      # C++ testbench (for co-sim)
    target_device: str                 # e.g., "xcvc1902-vsva2197-2MP-e-S"
    clock_period_ns: float             # e.g., 5.0 for 200 MHz
    top_function: str                  # Top-level function name


@dataclass
class HLSSynthesisResult:
    """Parsed output of an HLS synthesis run."""

    success: bool
    error_message: Optional[str] = None

    # Timing
    clock_period_ns: Optional[float] = None
    clock_target_ns: Optional[float] = None
    estimated_latency_cycles: Optional[int] = None
    estimated_latency_ns: Optional[float] = None

    # Resource utilization
    bram_used: Optional[int] = None
    bram_available: Optional[int] = None
    dsp_used: Optional[int] = None
    dsp_available: Optional[int] = None
    ff_used: Optional[int] = None
    ff_available: Optional[int] = None
    lut_used: Optional[int] = None
    lut_available: Optional[int] = None

    # Raw report text for agent consumption
    raw_report: str = ""

    @property
    def bram_utilization(self) -> Optional[float]:
        if self.bram_used is not None and self.bram_available and self.bram_available > 0:
            return self.bram_used / self.bram_available
        return None

    @property
    def dsp_utilization(self) -> Optional[float]:
        if self.dsp_used is not None and self.dsp_available and self.dsp_available > 0:
            return self.dsp_used / self.dsp_available
        return None

    @property
    def ff_utilization(self) -> Optional[float]:
        if self.ff_used is not None and self.ff_available and self.ff_available > 0:
            return self.ff_used / self.ff_available
        return None

    @property
    def lut_utilization(self) -> Optional[float]:
        if self.lut_used is not None and self.lut_available and self.lut_available > 0:
            return self.lut_used / self.lut_available
        return None

    @property
    def meets_timing(self) -> Optional[bool]:
        if self.clock_period_ns is not None and self.clock_target_ns is not None:
            return self.clock_period_ns <= self.clock_target_ns
        return None


@dataclass
class HLSCoSimResult:
    """Result of HLS co-simulation (functional correctness check)."""

    passed: bool
    error_output: Optional[str] = None
    runtime_ms: float = 0.0
