# Model2HW Agentic Co-Design System: Technical Design Document

## Preface

This document is the complete engineering specification for evolving Model2HW from a static feasibility analyzer into an agentic co-design system. It is written so that any engineer, including someone new to the codebase, can implement each feature without ambiguity about what to build, where to put it, how it connects to existing code, what the data shapes are, and what "done" looks like.

Read the feasibility study first (`docs/agentic-codesign-feasibility-study.md`). This document assumes you understand the "why." Here we only cover the "what" and "how."

**Companion documents**:

- `docs/implementation-checklist.md` - Running status of all 85 features across 10 phases
- `docs/progress-log.md` - Engineering decisions, experimental results, and open questions (paper-ready)

---

## Design Principle: LLM Only Where Absolutely Necessary

The system is split into a **deterministic layer** (Phases 1-5) and an **agentic layer** (Phases 6-10). Everything that can be expressed as a formula, bounded enumeration, or threshold comparison stays in the deterministic layer. The LLM enters only when the system needs to generate or modify HLS source code and interpret synthesis tool output.

```text
Deterministic layer (Phases 1-5)       Agentic layer (Phases 6-10)
Python, zero deps, no LLM              LLM + Vitis HLS feedback
───────────────────────────────        ─────────────────────────────
Precision sweep                        HLS kernel generation
Sensitivity analysis                   Pragma optimization
Configuration recommendation           Operator fusion proposals
Decomposition planning                 Synthesis failure diagnosis
Extended hardware model
Pareto frontiers
KV placement decisions
Layer-split enumeration
```

This boundary is a hard constraint, not a guideline. Using an LLM for tasks in the deterministic layer adds latency, cost, and hallucination risk for zero benefit. Phases 1-5 run in milliseconds, produce fully reproducible results, and cost nothing per invocation.

---

## Evaluation Tiers: Validate Before Real Hardware

Before any design is deployed to physical silicon, it must pass through a tiered evaluation pipeline. Each tier increases fidelity and cost. A design only advances to the next tier if it passes the current one. This eliminates wasted time and, critically, removes the need to purchase an FPGA during development.

| Tier | Tool | Cost | Runtime | Accuracy vs. Real HW | What It Validates |
|------|------|------|---------|----------------------|-------------------|
| 1 | Model2HW analytical models | Free | Milliseconds | Memory: exact. Bandwidth: conservative (10-20% margin). Compute: order-of-magnitude. | Does the model fit? Is bandwidth feasible? Initial precision selection. |
| 2 | Vitis HLS C-simulation + synthesis reports | Free (Vitis download) | Minutes per kernel | Resource utilization: <5% error. Latency estimates: 5-15% error. Timing: exact for target clock. | Per-operator resource consumption, cycle-accurate latency per kernel, timing closure at target frequency. |
| 3 | Full RTL simulation (Vivado XSIM / Verilator) | Free | Hours per operator | Cycle-exact for simulated operators | Bit-exact functional correctness, cycle-level pipeline behavior, memory access patterns. |
| 4 | FPGA-in-the-cloud (AWS F1, Nimbix, AMD University Program) | ~$1.65/hr (F1) | Real-time | Ground truth | End-to-end latency, actual throughput, thermal behavior, real memory controller effects. |

### Tier Advancement Rules

* Tier 1 is mandatory for every configuration. If Model2HW says it does not fit, stop.
* Tier 2 requires Vitis HLS installed locally (free download, no license for simulation and synthesis reports). Run C-simulation for functional correctness, then synthesis for resource and timing estimates.
* Tier 3 is optional and recommended only for individual operators where Tier 2 latency estimates seem suspect or where pipeline stalls need investigation.
* Tier 4 is the final validation before declaring a design production-ready. Use cloud FPGAs to avoid purchasing hardware.

### Where Each Tier Maps to Implementation Phases

* Phases 1-4 produce configurations validated at Tier 1 (analytical).
* Phase 6 (HLS Synthesis Feedback) produces Tier 2 validation data.
* Phase 7 (Agentic Kernel Optimizer) targets Tier 2 optimization with optional Tier 3 deep-dives.
* Phase 10 (End-to-End Pipeline) orchestrates full Tier 1 through Tier 4 progression.

### Accuracy Summary

| Metric | Tier 1 vs. Tier 4 | Tier 2 vs. Tier 4 |
|--------|--------------------|--------------------|
| Memory footprint | Exact (formula-based) | Exact (synthesis report) |
| Bandwidth requirement | Conservative, 10-20% overshoot | 5-15% error |
| Resource utilization (LUTs, DSPs, BRAM) | Not available | <5% error |
| Latency per operator | Not available | 5-15% error |
| End-to-end throughput | Order-of-magnitude estimate | Within 15-20% |
| Timing closure | Not available | Exact for target clock |

---

## Table of Contents

1. [Codebase Orientation](#1-codebase-orientation)
2. [Implementation Phases Overview](#2-implementation-phases-overview)
3. [Phase 1: Precision Sweep Engine](#3-phase-1-precision-sweep-engine)
4. [Phase 2: Sensitivity Analyzer](#4-phase-2-sensitivity-analyzer)
5. [Phase 3: Configuration Recommender](#5-phase-3-configuration-recommender)
6. [Phase 4: Decomposition Planner](#6-phase-4-decomposition-planner)
7. [Phase 5: Extended Hardware Model](#7-phase-5-extended-hardware-model)
8. [Phase 6: HLS Synthesis Feedback Integration](#8-phase-6-hls-synthesis-feedback-integration)
9. [Phase 7: Agentic Kernel Optimizer](#9-phase-7-agentic-kernel-optimizer)
10. [Phase 8: Multi-Kernel Orchestration](#10-phase-8-multi-kernel-orchestration)
11. [Phase 9: Accuracy-in-the-Loop](#11-phase-9-accuracy-in-the-loop)
12. [Phase 10: End-to-End Pipeline](#12-phase-10-end-to-end-pipeline)
13. [Testing Strategy](#13-testing-strategy)
14. [Coding Conventions](#14-coding-conventions)
15. [Dependency Policy](#15-dependency-policy)
16. [Glossary](#16-glossary)

---

## 1. Codebase Orientation

### 1.1 Project Structure (current)

```
Model2HW/
  pyproject.toml                       # Package config. Entry point: model2hw -> cli:main
  hardware_feasibility/
    __init__.py                        # __version__ = "0.1.0"
    __main__.py                        # `python -m hardware_feasibility` -> cli.main()
    cli.py                             # Argument parsing, orchestration, output routing
    models/
      architecture_rules.py            # ModelSpec dataclass, Precision enum, KNOWN_FAMILIES dict
      hf_config_loader.py              # load_from_hf_config(), load_from_hf_hub(), load_from_known_family()
    analysis/
      memory.py                        # MemoryProfile, estimate_weight_memory(), estimate_kv_cache(), analyze_memory()
      bandwidth.py                     # BandwidthProfile, analyze_bandwidth()
      compute.py                       # ComputeProfile, estimate_flops_per_token(), analyze_compute()
      io.py                            # IOProfile, analyze_io()
      verdict.py                       # FeasibilityVerdict enum, VerdictResult, render_verdict()
    hardware/
      board_specs.py                   # BoardSpec dataclass, BOARD_DATABASE dict, get_board(), list_boards()
      matcher.py                       # MatchResult, match_board(), rank_boards()
    outputs/
      json_export.py                   # export_json(), build_analysis_dict()
      report.py                        # generate_report() -> human-readable string
```

### 1.2 Key Data Flow

```
User CLI args
  -> build ModelSpec (from config.json / HF Hub / known family)
  -> analyze_memory(spec)       -> MemoryProfile
  -> analyze_bandwidth(spec)    -> BandwidthProfile
  -> analyze_compute(spec, bw)  -> ComputeProfile
  -> analyze_io(spec)           -> IOProfile
  -> render_verdict(spec, mem, bw, compute, io, targets) -> VerdictResult
  -> [optional] rank_boards(spec, mem, bw, io)           -> list[MatchResult]
  -> generate_report() or export_json()
  -> stdout or file
```

Every analysis function takes a `ModelSpec` and returns a typed dataclass. Verdict takes all profile dataclasses plus optional hardware targets. This is the pattern to follow for all new modules.

### 1.3 Key Dataclasses You Must Know

**`ModelSpec`** (in `architecture_rules.py`): The single source of truth for model configuration. Every field:

| Field | Type | Source | Meaning |
|---|---|---|---|
| `name` | `str` | Config/family | Human-readable model name |
| `params` | `int` | Computed | Total parameter count |
| `num_layers` | `int` | Config | Transformer layer count |
| `hidden_size` | `int` | Config | Hidden dimension |
| `num_attention_heads` | `int` | Config | Number of Q heads |
| `num_kv_heads` | `int` | Config | Number of KV heads (< attention_heads means GQA) |
| `intermediate_size` | `int` | Config | MLP intermediate dimension |
| `vocab_size` | `int` | Config | Vocabulary size |
| `weight_precision` | `Precision` | CLI | Weight quantization level |
| `kv_precision` | `Precision` | CLI | KV cache quantization level |
| `batch_size` | `int` | CLI | Batch size for inference |
| `context_length` | `int` | CLI | Maximum context window |
| `prefill_length` | `int` | CLI | Prompt token count |
| `decode_length` | `int` | CLI | Generation token count |
| `target_tokens_per_sec` | `float` | CLI | Throughput target |
| `kv_on_accelerator` | `bool` | CLI | Whether KV cache is on device |

Derived properties: `head_dim`, `bytes_per_weight`, `bytes_per_kv`.

**`BoardSpec`** (in `board_specs.py`): Hardware target. Fields: `name`, `category`, `memory_gb`, `memory_bandwidth_gbps`, `peak_tops_int8`, `peak_tflops_fp16`, `host_link`, `host_link_bandwidth_gbps`, `tdp_watts`, `notes`.

**`Precision`** (in `architecture_rules.py`): Enum with values FP32, FP16, BF16, INT8, INT4. Has a `bytes_per_element` property.

### 1.4 How the CLI Works

`cli.py` has three functions:
- `build_parser()` - returns `argparse.ArgumentParser` with all flags
- `run(argv)` - the actual logic: parse args, build ModelSpec, run analysis, format output
- `main()` - calls `run()` with no argv (uses sys.argv)

The `run()` function is 80 lines of sequential code. No classes, no state. This is intentional: the CLI is a thin orchestration layer. New features add new modules that `run()` calls.

### 1.5 How to Add a New Analysis Module

Follow this exact pattern:

1. Create `hardware_feasibility/analysis/new_module.py`
2. Define a `@dataclass` for the output (e.g., `SweepResult`)
3. Write a function that takes `ModelSpec` (and optionally profiles/board specs) and returns your dataclass
4. Import and call from `cli.py`
5. Add JSON serialization in `json_export.py`
6. Add report section in `report.py`
7. Add tests in `tests/test_new_module.py`

---

## 2. Implementation Phases Overview

Each phase builds on the previous. Do not skip phases. Do not start phase N+1 until phase N has passing tests.

| Phase | Name | Depends on | New files | Requires LLM? | Estimated scope |
|---|---|---|---|---|---|
| 1 | Precision Sweep Engine | Nothing (extends current) | `analysis/sweep.py` | No | ~200 lines |
| 2 | Sensitivity Analyzer | Phase 1 | `analysis/sensitivity.py` | No | ~150 lines |
| 3 | Configuration Recommender | Phases 1-2 | `analysis/recommender.py` | No | ~250 lines |
| 4 | Decomposition Planner | Phases 1-3 | `analysis/decomposition.py` | No | ~400 lines |
| 5 | Extended Hardware Model | Nothing (extends current) | modify `board_specs.py` | No | ~150 lines changed |
| 6 | HLS Synthesis Feedback | Phase 5 | `synthesis/` new package | No (infrastructure only) | ~500 lines |
| 7 | Agentic Kernel Optimizer | Phase 6 | `agents/` new package | **Yes** | ~800 lines |
| 8 | Multi-Kernel Orchestration | Phase 7 | `agents/orchestrator.py` | **Yes** | ~600 lines |
| 9 | Accuracy-in-the-Loop | Phase 3 | `evaluation/` new package | No | ~400 lines |
| 10 | End-to-End Pipeline | Phases 1-9 | `pipeline/` new package | Optional (only for Phases 7-8) | ~500 lines |

Phases 1 through 5 and Phase 9 require ZERO external dependencies and ZERO LLM calls. Phase 6 requires Vitis HLS (system install) but no LLM. Only Phases 7 and 8 invoke an LLM, and only for HLS code generation and optimization. Phase 10 orchestrates all phases but invokes the LLM only when running the optional kernel optimization path.

---

## 3. Phase 1: Precision Sweep Engine

### 3.1 Purpose

Given a model and a hardware target, enumerate all valid (weight_precision, kv_precision) combinations and for each one, compute the full analysis (memory, bandwidth, compute, IO, verdict). Return the results ranked by feasibility margin.

This replaces the current workflow where you manually run the CLI multiple times with different precision flags.

### 3.2 File: `hardware_feasibility/analysis/sweep.py`

```python
"""Precision sweep: enumerate precision configurations and rank by feasibility."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..models.architecture_rules import ModelSpec, Precision
from .memory import MemoryProfile, analyze_memory
from .bandwidth import BandwidthProfile, analyze_bandwidth
from .compute import ComputeProfile, analyze_compute
from .io import IOProfile, analyze_io
from .verdict import FeasibilityVerdict, VerdictResult, render_verdict


@dataclass
class SweepPoint:
    """One configuration point in the precision sweep."""

    weight_precision: Precision
    kv_precision: Precision
    memory: MemoryProfile
    bandwidth: BandwidthProfile
    compute: ComputeProfile
    io: IOProfile
    verdict: VerdictResult

    @property
    def fits(self) -> bool:
        return self.verdict.verdict != FeasibilityVerdict.DOES_NOT_FIT

    @property
    def memory_headroom_gb(self) -> Optional[float]:
        """Positive = spare memory. Negative = overshoot. None = no target."""
        if self.verdict.target_memory_gb is None:
            return None
        return self.verdict.target_memory_gb - self.memory.total_gb

    @property
    def bandwidth_headroom_gbps(self) -> Optional[float]:
        if self.verdict.target_bandwidth_gbps is None:
            return None
        return self.verdict.target_bandwidth_gbps - self.bandwidth.required_bandwidth_gbps


@dataclass
class SweepResult:
    """Full result of a precision sweep."""

    base_spec: ModelSpec
    target_memory_gb: Optional[float]
    target_bandwidth_gbps: Optional[float]
    target_link_bandwidth_gbps: Optional[float]
    points: list[SweepPoint]

    @property
    def fitting_points(self) -> list[SweepPoint]:
        return [p for p in self.points if p.fits]

    @property
    def best_fitting(self) -> Optional[SweepPoint]:
        """Best fitting point: fits + highest bandwidth headroom (most throughput margin)."""
        fitting = self.fitting_points
        if not fitting:
            return None
        return max(fitting, key=lambda p: p.bandwidth_headroom_gbps or 0)
```

### 3.3 Core Function: `run_precision_sweep`

```python
# Precision combinations to sweep. Order matters: leftmost = highest quality.
WEIGHT_PRECISIONS = [Precision.FP16, Precision.BF16, Precision.INT8, Precision.INT4]
KV_PRECISIONS = [Precision.FP16, Precision.INT8, Precision.INT4]


def run_precision_sweep(
    spec: ModelSpec,
    *,
    target_memory_gb: Optional[float] = None,
    target_bandwidth_gbps: Optional[float] = None,
    target_link_bandwidth_gbps: Optional[float] = None,
    weight_precisions: list[Precision] | None = None,
    kv_precisions: list[Precision] | None = None,
) -> SweepResult:
    """Enumerate precision combinations and rank by feasibility.

    For each (weight_precision, kv_precision) pair:
    1. Create a modified ModelSpec with those precisions
    2. Run the full analysis pipeline
    3. Render a verdict against hardware targets
    4. Collect into SweepResult

    The result is sorted: fitting points first (by bandwidth headroom descending),
    then non-fitting points (by memory overshoot ascending, i.e. closest to fitting first).
    """
    w_precs = weight_precisions or WEIGHT_PRECISIONS
    kv_precs = kv_precisions or KV_PRECISIONS
    points: list[SweepPoint] = []

    for wp in w_precs:
        for kp in kv_precs:
            # KV precision higher than weight precision makes no engineering sense
            if kp.bytes_per_element > wp.bytes_per_element:
                continue

            modified = replace(spec, weight_precision=wp, kv_precision=kp)
            mem = analyze_memory(modified)
            bw = analyze_bandwidth(modified)
            compute = analyze_compute(modified, 0)
            io = analyze_io(modified)
            verdict = render_verdict(
                modified, mem, bw, compute, io,
                target_memory_gb=target_memory_gb,
                target_bandwidth_gbps=target_bandwidth_gbps,
                target_link_bandwidth_gbps=target_link_bandwidth_gbps,
            )
            points.append(SweepPoint(
                weight_precision=wp,
                kv_precision=kp,
                memory=mem,
                bandwidth=bw,
                compute=compute,
                io=io,
                verdict=verdict,
            ))

    # Sort: fitting first (sorted by BW headroom desc), then non-fitting (sorted by memory gap asc)
    def sort_key(p: SweepPoint) -> tuple:
        fits = 0 if p.fits else 1
        bw_head = -(p.bandwidth_headroom_gbps or 0)
        mem_head = -(p.memory_headroom_gb or -1e9)
        return (fits, bw_head, mem_head)

    points.sort(key=sort_key)

    return SweepResult(
        base_spec=spec,
        target_memory_gb=target_memory_gb,
        target_bandwidth_gbps=target_bandwidth_gbps,
        target_link_bandwidth_gbps=target_link_bandwidth_gbps,
        points=points,
    )
```

### 3.4 CLI Integration

Add to `cli.py`:

New flag in `build_parser()`:

```python
p.add_argument("--sweep", action="store_true",
               help="Sweep all precision combinations and rank by feasibility.")
```

New block in `run()`, after the existing analysis and before output:

```python
if args.sweep:
    from .analysis.sweep import run_precision_sweep
    sweep = run_precision_sweep(
        spec,
        target_memory_gb=target_mem,
        target_bandwidth_gbps=target_bw,
        target_link_bandwidth_gbps=target_link,
    )
    # Use sweep-specific output formatters (see below)
```

### 3.5 Output Formats

Add a `format_sweep_report(sweep: SweepResult) -> str` function in `outputs/report.py`. Layout:

```
========================================================================
  PRECISION SWEEP RESULTS
  Model: llama3-8b  |  Target: Xilinx VCK190 (8.0 GB, 25.6 GB/s)
========================================================================

  #  Weight   KV      Mem (GB)   BW (GB/s)  Verdict              Est. tok/s
  -- ------   ------  --------   ---------  -------------------  ----------
  1  int4     int4    3.81       38.1       FITS (BW-limited)    6.7
  2  int4     int8    4.06       40.6       FITS (BW-limited)    6.3
  3  int8     int8    8.12       81.2       DOES NOT FIT         -
  4  fp16     int8    16.06      160.6      DOES NOT FIT         -
  ...
```

Add a `format_sweep_json(sweep: SweepResult) -> dict` in `outputs/json_export.py`.

### 3.6 Key Design Decision: Why `dataclasses.replace`

The `replace(spec, weight_precision=wp, kv_precision=kp)` call creates a shallow copy of `ModelSpec` with only the precision fields changed. This is safe because `ModelSpec` contains only immutable fields (ints, floats, enums, bools, strings). Do not add mutable fields (lists, dicts) to `ModelSpec` without updating this pattern.

### 3.7 Tests

File: `tests/test_sweep.py`

```python
def test_sweep_llama3_8b_on_vck190():
    """INT4 weights should fit on VCK190 (8 GB); FP16 should not."""
    spec = load_from_known_family("llama3-8b")
    result = run_precision_sweep(spec, target_memory_gb=8.0, target_bandwidth_gbps=25.6)

    # At least one point should fit (INT4)
    assert len(result.fitting_points) > 0

    # INT4/INT4 should be the best fitting point
    best = result.best_fitting
    assert best is not None
    assert best.weight_precision == Precision.INT4

    # FP16/FP16 should NOT fit
    fp16_point = [p for p in result.points
                  if p.weight_precision == Precision.FP16 and p.kv_precision == Precision.FP16]
    assert len(fp16_point) == 1
    assert not fp16_point[0].fits


def test_sweep_skips_kv_higher_than_weight():
    """KV precision should never exceed weight precision."""
    spec = load_from_known_family("llama3.2-1b")
    result = run_precision_sweep(spec, target_memory_gb=4.0)
    for p in result.points:
        assert p.kv_precision.bytes_per_element <= p.weight_precision.bytes_per_element


def test_sweep_returns_sorted():
    """Fitting points come before non-fitting points."""
    spec = load_from_known_family("llama3-8b")
    result = run_precision_sweep(spec, target_memory_gb=8.0)
    seen_not_fit = False
    for p in result.points:
        if not p.fits:
            seen_not_fit = True
        if seen_not_fit:
            assert not p.fits, "Non-fitting point appeared before a fitting point"
```

### 3.8 Definition of Done

- [ ] `run_precision_sweep` returns correct results for llama3-8b on VCK190
- [ ] CLI `--sweep` flag produces formatted output
- [ ] JSON output includes sweep data when `--sweep --json` is used
- [ ] All tests pass
- [ ] No new dependencies added

---

## 4. Phase 2: Sensitivity Analyzer

### 4.1 Purpose

For a given model-hardware configuration, identify which bottleneck dominates (memory capacity, memory bandwidth, host IO, compute) and quantify how sensitive the verdict is to changes in each parameter. Answers the question: "If I had 20% more bandwidth, would this model fit?"

### 4.2 File: `hardware_feasibility/analysis/sensitivity.py`

### 4.3 Data Model

```python
@dataclass
class BottleneckBreakdown:
    """Quantifies how close each resource dimension is to the limit."""

    memory_utilization: float        # 0.0 - inf. > 1.0 means over capacity.
    bandwidth_utilization: float     # 0.0 - inf. > 1.0 means under target tok/s.
    io_utilization: float            # 0.0 - inf. > 1.0 means host link saturated.
    compute_utilization: float       # 0.0 - inf. > 1.0 means compute-bound.

    @property
    def primary_bottleneck(self) -> str:
        """Returns the name of the most constrained resource."""
        utils = {
            "memory": self.memory_utilization,
            "bandwidth": self.bandwidth_utilization,
            "host_io": self.io_utilization,
            "compute": self.compute_utilization,
        }
        return max(utils, key=utils.get)


@dataclass
class SensitivityPoint:
    """What happens if we change one parameter by a given delta."""

    parameter_name: str              # e.g. "target_memory_gb", "context_length"
    original_value: float
    modified_value: float
    original_verdict: str            # FeasibilityVerdict.value
    modified_verdict: str
    verdict_changed: bool


@dataclass
class SensitivityResult:
    """Full sensitivity analysis output."""

    bottleneck: BottleneckBreakdown
    sensitivities: list[SensitivityPoint]
```

### 4.4 Core Function

```python
def analyze_sensitivity(
    spec: ModelSpec,
    mem: MemoryProfile,
    bw: BandwidthProfile,
    compute: ComputeProfile,
    io: IOProfile,
    *,
    target_memory_gb: float,
    target_bandwidth_gbps: float,
    target_link_bandwidth_gbps: float,
    target_tops: float | None = None,
) -> SensitivityResult:
```

**Step 1: Compute BottleneckBreakdown**

```python
memory_util = mem.total_gb / target_memory_gb
bandwidth_util = bw.required_bandwidth_gbps / target_bandwidth_gbps
io_util = io.required_io_bandwidth_gbps / target_link_bandwidth_gbps

# Compute utilization: requires knowing the board's peak TOPS
# If target_tops is provided, compute decode throughput limit
if target_tops is not None:
    # Convert TOPS to FLOPs/s (TOPS = 10^12 ops/s)
    peak_flops_per_sec = target_tops * 1e12
    # Time per token if compute-bound = flops_per_token / peak_flops_per_sec
    # Required flops/s = flops_per_token * target_tok_s
    required_flops_per_sec = compute.flops_per_token * spec.target_tokens_per_sec
    compute_util = required_flops_per_sec / peak_flops_per_sec
else:
    compute_util = 0.0  # Cannot assess without compute target
```

**Step 2: Test sensitivity by perturbing parameters**

For each of these dimensions, create a modified scenario and re-run the verdict:

| Parameter | Perturbation | How to modify |
|---|---|---|
| `target_memory_gb` | +25%, +50%, -25% | Change hardware target, not model |
| `target_bandwidth_gbps` | +25%, +50%, -25% | Change hardware target |
| `context_length` | 0.5x, 2x, 4x | `replace(spec, context_length=new_val)`, then re-analyze |
| `batch_size` | 2x, 4x | `replace(spec, batch_size=new_val)`, then re-analyze |
| `weight_precision` | One step down | `replace(spec, weight_precision=next_lower)`, then re-analyze |

For hardware target perturbations, re-run `render_verdict()` with modified targets (cheap, no re-analysis needed). For model parameter perturbations, re-run the full analysis pipeline.

Collect each perturbation as a `SensitivityPoint`.

### 4.5 CLI Integration

New flag:

```python
p.add_argument("--sensitivity", action="store_true",
               help="Run sensitivity analysis showing which bottleneck dominates.")
```

Requires `--board` or explicit `--target-memory-gb` and `--target-bw-gbps`.

### 4.6 Report Output Format

```
------------------------------------------------------------------------
  SENSITIVITY ANALYSIS
------------------------------------------------------------------------
  Primary bottleneck: BANDWIDTH (160% of available)

  Resource utilization:
    Memory:     56% of 8.0 GB  (4.5 GB used)     [OK]
    Bandwidth: 160% of 25.6 GB/s  (41.0 GB/s needed)  [BOTTLENECK]
    Host IO:     8% of 16.0 GB/s  (1.3 GB/s needed)   [OK]
    Compute:    12% of 133 TOPS                        [OK]

  What-if scenarios:
    +50% bandwidth (38.4 GB/s):  Still bandwidth-limited, ~9.3 tok/s
    +25% memory (10.0 GB):       No change (memory is not the bottleneck)
    2x context (4096 tokens):    KV cache doubles to 0.50 GB, still fits
    4x context (8192 tokens):    KV cache = 1.0 GB, still fits but BW worse
    INT4 -> INT8 weights:        DOES NOT FIT (8.0 GB weights alone)
```

### 4.7 Tests

- `test_sensitivity_identifies_bandwidth_bottleneck`: llama3-8b INT4 on VCK190 should report bandwidth as primary bottleneck
- `test_sensitivity_identifies_memory_bottleneck`: llama3-8b FP16 on VCK190 should report memory as primary bottleneck
- `test_sensitivity_perturbation_count`: Verify the expected number of sensitivity points are generated

### 4.8 Definition of Done

- [ ] `BottleneckBreakdown.primary_bottleneck` correctly identifies the dominant constraint
- [ ] Sensitivity points correctly show verdict changes
- [ ] Report output is human-readable and matches the format above
- [ ] All tests pass

---

## 5. Phase 3: Configuration Recommender

### 5.1 Purpose

Given a model and a hardware target, automatically recommend the best operating configuration: the combination of (weight_precision, kv_precision, context_length, batch_size, kv_placement) that maximizes throughput while fitting on the hardware to meet the user's target tok/s.

This is the first feature that does "optimization," not just "analysis."

### 5.2 File: `hardware_feasibility/analysis/recommender.py`

### 5.3 Data Model

```python
@dataclass
class Recommendation:
    """A recommended operating configuration."""

    weight_precision: Precision
    kv_precision: Precision
    context_length: int
    batch_size: int
    kv_on_accelerator: bool

    estimated_tok_per_sec: float
    memory_utilization: float          # fraction
    bandwidth_utilization: float       # fraction
    verdict: VerdictResult

    rationale: list[str]               # Human-readable explanation of why this config was chosen


@dataclass
class RecommendationResult:
    """Full recommendation output."""

    target_board: str
    model_name: str
    recommendations: list[Recommendation]  # Sorted: best first
    infeasible_reason: str | None          # If no configuration fits at all
```

### 5.4 Algorithm

The recommender performs a structured sweep, not an exhaustive search. The search space is intentionally constrained to keep runtime under 1 second.

```
Step 1: Run precision sweep (Phase 1) to find all fitting precision combos.
        If none fit, set infeasible_reason and return.

Step 2: For each fitting precision combo, sweep context_length:
        [512, 1024, 2048, 4096, 8192, 16384, 32768]
        but only values <= spec.context_length (the user's original request).
        For each, re-run analyze_memory and check if still fits.

Step 3: For each still-fitting (precision, context) pair, compute estimated tok/s:
        est_tok_s = board_bandwidth_bytes / total_bytes_per_token
        This uses bandwidth analysis.

Step 4: Optionally sweep batch_size [1, 2, 4, 8] for throughput optimization.
        Higher batch size = more total throughput but more memory and higher latency.

Step 5: For configs where model fits but bandwidth is the bottleneck,
        check if kv_on_accelerator=False helps (shifts KV to host, frees device memory
        for potential weight caching, but adds host link traffic).

Step 6: Rank all valid configurations by estimated_tok_per_sec descending.

Step 7: Generate rationale strings explaining each recommendation.
```

### 5.5 Rationale Generation

Do not use LLMs for this. Rationale is generated from template strings. This is deterministic text formatting, not a reasoning task:

```python
rationale = []
if rec.weight_precision == Precision.INT4:
    rationale.append(
        f"INT4 quantization reduces weight memory from "
        f"{fp16_weight_gb:.1f} GB to {rec_weight_gb:.1f} GB, "
        f"enabling this model to fit on {board.name}."
    )
if rec.kv_precision != rec.weight_precision:
    rationale.append(
        f"KV cache uses {rec.kv_precision.value} (vs {rec.weight_precision.value} weights) "
        f"to balance memory and cache quality."
    )
if rec.context_length < spec.context_length:
    rationale.append(
        f"Context reduced from {spec.context_length} to {rec.context_length} "
        f"to fit KV cache within memory budget."
    )
```

### 5.6 CLI Integration

New flag:

```python
p.add_argument("--recommend", action="store_true",
               help="Recommend the best configuration for the given model and board.")
```

Requires `--board`. Outputs the top 3 recommendations with rationale.

### 5.7 Definition of Done

- [ ] Returns >=1 recommendation for llama3-8b on VCK190
- [ ] Returns `infeasible_reason` for llama2-70b on VCK190 (nothing fits)
- [ ] Recommendations are sorted by estimated tok/s
- [ ] Rationale strings are populated and meaningful
- [ ] CLI `--recommend` produces formatted output

---

## 6. Phase 4: Decomposition Planner

### 6.1 Purpose

For models that do not fit on a single device, propose multi-device execution strategies: pipeline parallelism (layer splitting), KV offloading, and multi-board configurations.

### 6.2 File: `hardware_feasibility/analysis/decomposition.py`

### 6.3 Data Model

```python
@dataclass
class DeviceAssignment:
    """Assignment of model layers to a specific device."""

    device_name: str                   # Board name from BOARD_DATABASE
    layer_start: int                   # Inclusive
    layer_end: int                     # Exclusive
    weight_memory_gb: float            # Weights for assigned layers
    kv_cache_memory_gb: float          # KV cache for assigned layers
    total_memory_gb: float
    memory_utilization: float          # fraction of device capacity


@dataclass
class DecompositionPlan:
    """A complete multi-device execution plan."""

    strategy: str                       # "pipeline_parallel", "kv_offload", "hybrid"
    devices: list[DeviceAssignment]
    total_devices: int

    # Communication analysis
    inter_device_transfer_bytes_per_token: int
    inter_device_bandwidth_required_gbps: float

    # Performance
    estimated_tok_per_sec: float
    pipeline_bubble_fraction: float     # 0.0 - 1.0; fraction of time wasted in bubbles

    feasible: bool
    details: list[str]


@dataclass
class DecompositionResult:
    """All decomposition plans considered."""

    model_name: str
    single_device_feasible: bool
    plans: list[DecompositionPlan]
    best_plan: DecompositionPlan | None
```

### 6.4 Algorithm: Pipeline Parallel Decomposition

```python
def plan_pipeline_parallel(
    spec: ModelSpec,
    boards: list[BoardSpec],
) -> DecompositionPlan | None:
    """Try to split model layers across N identical boards.

    Strategy:
    1. Compute per-layer weight memory: total_weight_memory / num_layers
       (Approximate. Assumes uniform layers, which is true for standard
       decoder-only transformers where each layer has identical structure.)

    2. Compute per-layer KV cache memory at target context:
       kv_per_token * context_length / num_layers
       (This is exact: KV cache scales linearly with layer count.)

    3. For N = 2, 3, 4, ..., up to num_layers:
       - Divide layers into N contiguous groups, each fitting on one board
       - Check: per-group memory <= board.memory_gb
       - If yes, compute:
         a. Inter-device transfer = hidden_size * batch_size * bytes_per_weight
            (activation tensor passed between stages)
         b. Pipeline bubble fraction = (N - 1) / (N - 1 + decode_length)
            (simplified model; assumes uniform stage latency)
         c. Effective tok/s = base_tok_s * (1 - bubble_fraction)
            where base_tok_s is limited by the slowest stage's bandwidth

    4. Return the plan with minimum N that is feasible.
    """
```

Important detail: layers are NOT uniform in memory usage across all architectures. For standard decoder-only models (Llama family), they are. For mixture-of-experts models, they are not. The planner should compute per-layer memory when possible. For known families, use the uniform approximation. For HF configs, check if `num_experts` is present and handle differently.

### 6.5 Algorithm: KV Offload

```python
def plan_kv_offload(
    spec: ModelSpec,
    board: BoardSpec,
) -> DecompositionPlan | None:
    """Check if weights fit on device with KV cache on host.

    1. Compute weight_memory + activation_buffer (without KV cache)
    2. If this fits on board.memory_gb:
       a. KV cache lives on host memory (assumed unlimited)
       b. Each decode step transfers KV for all layers across host link
       c. Performance = min(bandwidth-limited tok/s, link-limited tok/s)
    3. Return plan if feasible.
    """
```

This uses the existing `analyze_io` logic (the `kv_on_accelerator=False` path) but wraps it in a plan structure.

### 6.6 CLI Integration

New flag:

```python
p.add_argument("--decompose", action="store_true",
               help="Propose multi-device decomposition if model doesn't fit on --board.")
```

Also useful as an automatic fallback: when `--recommend` finds no single-device config, it calls the decomposition planner.

### 6.7 Worked Example

Llama 2-13B INT4 on VCK190 (8 GB):

```
Per-layer weight memory: 6.5 GB / 40 layers = 0.163 GB/layer
Per-layer KV cache (2K ctx): 0.62 GB / 40 layers = 0.016 GB/layer
Total per-layer: 0.179 GB/layer

Single VCK190 can hold: 8.0 GB / 0.179 GB/layer ≈ 44 layers -> ALL 40 FIT (barely)
But total = 6.5 + 0.62 + activation = ~7.5 GB -> fits but tight.

If it did NOT fit:
2x VCK190: 20 layers each = 20 * 0.179 = 3.58 GB/device -> comfortable fit
Inter-device: 3072 * 1 * 0.5 = 1536 bytes/token (hidden_size * batch * bytes at INT4)
At 10 tok/s: 15.4 KB/s -> negligible on PCIe Gen4 x8
Pipeline bubble: (2-1) / (2-1 + 256) = 0.4% -> negligible
```

### 6.8 Tests

- `test_kv_offload_for_tight_fit`: Verify KV offload frees enough memory
- `test_pipeline_2way_split`: Verify 2-device split for a model that barely doesn't fit
- `test_no_decomposition_needed`: Verify early exit when model fits on one device

### 6.9 Definition of Done

- [ ] Pipeline parallel planner generates valid layer assignments
- [ ] KV offload planner correctly computes host link bandwidth requirements
- [ ] Communication overhead is included in tok/s estimate
- [ ] Plans are sorted by estimated performance
- [ ] CLI `--decompose` produces formatted output

---

## 7. Phase 5: Extended Hardware Model

### 7.1 Purpose

Add fields to `BoardSpec` that are needed by Phases 6-10: on-chip SRAM/BRAM capacity, DSP count, LUT count, and memory hierarchy type. These are required for the agentic kernel optimizer to reason about tiling and resource allocation.

### 7.2 Changes to `board_specs.py`

Add new optional fields to `BoardSpec`:

```python
@dataclass
class BoardSpec:
    # ... existing fields ...

    # On-chip resources (FPGA-specific)
    bram_kb: int | None = None          # Total Block RAM in KB
    uram_kb: int | None = None          # Total UltraRAM in KB (Xilinx specific)
    dsp_slices: int | None = None       # DSP slice count
    lut_count: int | None = None        # LUT count (in thousands)

    # Memory hierarchy
    memory_type: str = "ddr4"           # "ddr4", "ddr5", "hbm2", "hbm2e", "lpddr4", "lpddr5", "unified"
    memory_channels: int = 1            # Number of memory channels

    # Compute architecture
    has_ai_engines: bool = False         # Versal AI Engines or similar
    ai_engine_count: int | None = None
```

Update existing board registrations. For example for VCK190:

```python
_register(BoardSpec(
    name="Xilinx VCK190",
    category="fpga",
    memory_gb=8.0,
    memory_bandwidth_gbps=25.6,
    peak_tops_int8=133.0,
    host_link="PCIe Gen4 x8",
    host_link_bandwidth_gbps=16.0,
    tdp_watts=75,
    notes="Versal AI Core XCVC1902. 400 AI Engines. DDR4 + LPDDR4.",
    # New fields:
    bram_kb=967,                       # 967 x 36Kb BRAMs = ~4.3 MB
    uram_kb=0,                         # XCVC1902 does not have UltraRAM
    dsp_slices=1968,
    lut_count=899,                     # in thousands
    memory_type="ddr4",
    memory_channels=2,
    has_ai_engines=True,
    ai_engine_count=400,
))
```

### 7.3 Where to Find These Numbers

- Xilinx boards: Xilinx product selection guides (DS950 for Versal, DS890 for UltraScale+)
- Intel boards: Intel FPGA product tables
- NVIDIA GPUs: CUDA Compute Capability tables
- NPUs: Vendor datasheets

All values are from public datasheets. When a value is unknown, leave it as `None`. The code must handle `None` gracefully (skip FPGA-specific analysis for GPUs, for example).

### 7.4 Add Derived Property

```python
@property
def on_chip_memory_kb(self) -> int:
    """Total on-chip fast memory (BRAM + URAM) in KB."""
    bram = self.bram_kb or 0
    uram = self.uram_kb or 0
    return bram + uram
```

This is used by the kernel optimizer (Phase 7) to determine how much weight data can be tiled on-chip for reuse.

### 7.5 Backward Compatibility

All new fields have defaults (`None`, `False`, `0`, or `"ddr4"`). Existing code that only reads the original fields continues to work unchanged. No migration needed.

### 7.6 Tests

- `test_vck190_has_extended_fields`: Verify BRAM, DSP, AI engine fields are populated
- `test_generic_fpga_has_none_extended_fields`: Verify `None` defaults for unspecified boards
- `test_on_chip_memory_kb`: Verify the derived property

### 7.7 Definition of Done

- [ ] All Xilinx and Intel FPGA boards have BRAM, DSP, LUT populated
- [ ] GPU and NPU boards have `None` for FPGA-specific fields
- [ ] `on_chip_memory_kb` property works correctly
- [ ] All existing tests still pass (no regressions)

---

## 8. Phase 6: HLS Synthesis Feedback Integration

### 8.1 Purpose

Build the infrastructure to invoke Vitis HLS (or a compatible HLS tool) programmatically, parse synthesis reports, and return structured performance data. This is the feedback loop that the agentic optimizer (Phase 7) will use.

### 8.2 New Package: `hardware_feasibility/synthesis/`

```
hardware_feasibility/synthesis/
  __init__.py
  hls_runner.py          # Invoke Vitis HLS and capture output
  report_parser.py       # Parse synthesis reports into structured data
  types.py               # Data models for synthesis results
```

### 8.3 Data Model (`types.py`)

```python
@dataclass
class HLSSynthesisResult:
    """Parsed output of an HLS synthesis run."""

    success: bool
    error_message: str | None

    # Timing
    clock_period_ns: float | None
    clock_target_ns: float | None
    estimated_latency_cycles: int | None
    estimated_latency_ns: float | None

    # Resource utilization
    bram_used: int | None
    bram_available: int | None
    dsp_used: int | None
    dsp_available: int | None
    ff_used: int | None
    ff_available: int | None
    lut_used: int | None
    lut_available: int | None

    # Utilization fractions (convenience)
    @property
    def bram_utilization(self) -> float | None:
        if self.bram_used is not None and self.bram_available and self.bram_available > 0:
            return self.bram_used / self.bram_available
        return None

    @property
    def meets_timing(self) -> bool | None:
        if self.clock_period_ns is not None and self.clock_target_ns is not None:
            return self.clock_period_ns <= self.clock_target_ns
        return None

    # Raw report text for agent consumption
    raw_report: str = ""


@dataclass
class HLSCoSimResult:
    """Result of HLS co-simulation (functional correctness check)."""

    passed: bool
    error_output: str | None
    runtime_ms: float


@dataclass
class KernelSpec:
    """Specification of an HLS kernel to synthesize."""

    name: str
    source_code: str                   # C++ source as a string
    testbench_code: str | None         # C++ testbench (for co-sim)
    target_device: str                 # e.g., "xcvc1902-vsva2197-2MP-e-S"
    clock_period_ns: float             # e.g., 5.0 for 200 MHz
    top_function: str                  # Top-level function name
```

### 8.4 HLS Runner (`hls_runner.py`)

```python
import subprocess
import tempfile
from pathlib import Path


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
                "Install Vitis HLS or set the path explicitly."
            )

    def synthesize(self, kernel: KernelSpec, work_dir: Path | None = None) -> HLSSynthesisResult:
        """Run HLS synthesis for a kernel and return parsed results.

        Process:
        1. Write kernel source to a temp directory
        2. Generate a Tcl script that:
           a. Opens a project
           b. Sets the target device and clock
           c. Adds the source file
           d. Sets the top function
           e. Runs csynth_design
        3. Invoke vitis_hls -f script.tcl
        4. Parse the synthesis report XML/JSON
        5. Return HLSSynthesisResult
        """
        # Implementation details below

    def cosim(self, kernel: KernelSpec, work_dir: Path | None = None) -> HLSCoSimResult:
        """Run HLS co-simulation for functional verification."""
        # Similar to synthesize but adds run_csim step
```

**Tcl Script Template:**

```tcl
open_project -reset {project_name}
set_top {top_function}
add_files {source_file}
open_solution -reset "solution1"
set_part {target_device}
create_clock -period {clock_period_ns} -name default
csynth_design
exit
```

The runner writes this script to a temp directory, executes it, and reads the report from `{project_name}/solution1/syn/report/{top_function}_csynth.rpt`.

### 8.5 Report Parser (`report_parser.py`)

Vitis HLS outputs several report formats:
- `.rpt` (human-readable text)
- `.xml` (machine-readable, preferred)
- JSON (in newer versions)

Parse the XML report at `solution1/syn/report/{top_function}_csynth.xml`:

```python
import xml.etree.ElementTree as ET


def parse_synthesis_report(report_path: Path) -> HLSSynthesisResult:
    """Parse a Vitis HLS synthesis report XML file."""
    tree = ET.parse(report_path)
    root = tree.getroot()

    # Extract timing
    perf = root.find(".//PerformanceEstimates")
    timing = root.find(".//TimingReport") or root.find(".//Timing")

    # Extract resources
    resources = root.find(".//AreaEstimates/Resources")

    # Build result (field extraction depends on Vitis HLS version)
    # ...
```

The exact XML structure varies by Vitis HLS version. Ship a `_parse_vitis_2022` and `_parse_vitis_2024` implementation and select based on a version sniff on the XML root.

### 8.6 Sandboxing

HLS synthesis executes arbitrary C++ code (compilation). Security measures:
- Kernel source code is **only** written to a temporary directory that is cleaned up after synthesis
- The HLS process runs with the user's permissions (no elevation)
- Timeout: 10 minutes per synthesis run (configurable). Kill the process on timeout.
- No network access is needed or permitted during synthesis

### 8.7 Tests

- `test_hls_runner_init_fails_without_vitis`: Verify clean error if vitis_hls not installed
- `test_tcl_script_generation`: Verify the Tcl script is generated correctly for a given KernelSpec
- `test_report_parser_sample`: Ship a sample `.xml` report in `tests/fixtures/` and verify parsing

### 8.8 Dependency Policy

Vitis HLS is NOT a Python dependency. It is an external tool that must be installed separately. The `HLSRunner` constructor verifies its presence and raises `RuntimeError` with installation instructions if missing. Phases 1-5 do not require Vitis HLS. Phases 6+ require it only when `--synthesize` or `--optimize` flags are used.

### 8.9 Definition of Done

- [ ] `HLSRunner` can generate a Tcl script from a `KernelSpec`
- [ ] `parse_synthesis_report` extracts latency, resource, and timing data from a sample XML
- [ ] `HLSSynthesisResult` correctly computes utilization fractions and timing checks
- [ ] Errors (missing tool, synthesis failure, timeout) are handled gracefully
- [ ] No new Python dependencies added (only stdlib: subprocess, xml, tempfile, pathlib)

---

## 9. Phase 7: Agentic Kernel Optimizer

### 9.1 Purpose

Given a transformer operator specification (e.g., "GEMM of shape MxNxK for Llama 3-8B layer 0 attention Q projection on VCK190 at 200 MHz") and a target FPGA, generate and iteratively optimize HLS C++ code using an LLM-driven feedback loop, modeled after LAAFD.

**This is the first phase that requires an LLM.** Phases 1-6 are entirely deterministic. The LLM is justified here because there is no closed-form solution for generating optimized HLS code from an operator specification, and the combinatorial pragma space has no analytical cost model.

### 9.2 New Package: `hardware_feasibility/agents/`

```
hardware_feasibility/agents/
  __init__.py
  types.py               # Agent data models
  kernel_optimizer.py     # Core optimization loop
  prompts.py             # Prompt templates
  llm_client.py          # LLM API abstraction
```

### 9.3 Data Model (`types.py`)

```python
@dataclass
class TransformerOperatorSpec:
    """Specification of a single transformer operator to implement in HLS."""

    op_type: str                       # "gemm", "attention_qkv", "softmax", "layernorm", "mlp_swiglu"
    input_shapes: dict[str, tuple[int, ...]]   # {"input": (M, K), "weight": (K, N)}
    output_shapes: dict[str, tuple[int, ...]]  # {"output": (M, N)}
    precision: Precision
    target_board: BoardSpec
    clock_mhz: int = 200


@dataclass
class OptimizationIteration:
    """Record of a single optimization iteration."""

    iteration: int
    source_code: str
    synthesis_result: HLSSynthesisResult | None
    cosim_result: HLSCoSimResult | None
    judge_feedback: str                 # LLM judge's assessment
    action_taken: str                   # "optimize", "fix_compile", "fix_runtime", "accept"


@dataclass
class KernelOptimizationResult:
    """Final result of kernel optimization."""

    operator: TransformerOperatorSpec
    final_source: str
    final_synthesis: HLSSynthesisResult
    iterations: list[OptimizationIteration]
    total_iterations: int
    converged: bool                    # True if judge issued "pass"
    estimated_latency_cycles: int
    resource_utilization: dict[str, float]  # {"bram": 0.45, "dsp": 0.23, ...}
```

### 9.4 LLM Client Abstraction (`llm_client.py`)

```python
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract interface for LLM API calls."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt and return the LLM's text response."""
        ...


class OpenAIClient(LLMClient):
    """OpenAI API client (GPT-4, GPT-5, etc.)."""

    def __init__(self, model: str = "gpt-4", api_key: str | None = None):
        # api_key from parameter or OPENAI_API_KEY env var
        ...

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import openai
        # Standard chat completion call
        ...
```

The design uses dependency injection: `KernelOptimizer` takes an `LLMClient` in its constructor. This allows testing with a mock client.

### 9.5 Prompt Templates (`prompts.py`)

Store each prompt as a Python constant string. Do NOT load prompts from external files (keeps deployment simple; prompts are part of the code).

```python
TRANSLATOR_SYSTEM_PROMPT = """You are an expert FPGA engineer specializing in Vitis HLS.
Your task is to translate a high-level operator specification into optimized
Vitis HLS C++ code.

Rules:
- Use `extern "C"` for the top-level function
- Add #pragma HLS LOOP_TRIPCOUNT for all loops
- Include all necessary HLS headers (#include <hls_stream.h>, etc.)
- Use ap_uint / ap_int types for fixed-point when appropriate
- The generated code must be self-contained (no external dependencies)
- Preserve functional correctness: the output must match the specification exactly
"""

TRANSLATOR_USER_TEMPLATE = """Generate Vitis HLS C++ code for the following operator:

Operator type: {op_type}
Input shapes: {input_shapes}
Output shapes: {output_shapes}
Precision: {precision}
Target device: {target_device}
Clock: {clock_mhz} MHz

The top function must be named: {top_function}

Generate complete, compilable C++ code.
"""

JUDGE_SYSTEM_PROMPT = """You are an HLS performance evaluation expert.
Given a Vitis HLS kernel and its synthesis report, assess whether the
implementation is near-optimal.

Respond with EXACTLY one of:
- "PASS" if latency is within 5% of theoretical minimum
- "OPTIMIZE: <specific feedback>" if improvements are possible

Your feedback must be specific and actionable:
- Reference loop labels and cycle counts from the synthesis report
- Suggest specific HLS pragmas or code transformations
- Quantify expected improvement
"""

JUDGE_USER_TEMPLATE = """Evaluate this HLS kernel:

Source code:
```cpp
{source_code}
```

Synthesis report summary:
- Total latency: {latency_cycles} cycles
- Clock: {clock_period_ns} ns ({clock_mhz} MHz)
- BRAM: {bram_used}/{bram_available} ({bram_pct:.0f}%)
- DSP: {dsp_used}/{dsp_available} ({dsp_pct:.0f}%)
- LUT: {lut_used}/{lut_available} ({lut_pct:.0f}%)

Theoretical minimum latency for this operator: {theoretical_min} cycles

Available optimizations to consider:
1. Loop pipelining (#pragma HLS pipeline II=1)
2. Vectorization (wider data paths)
3. Dataflow (#pragma HLS dataflow)
4. Array partitioning (#pragma HLS array_partition)
5. Loop tiling / unrolling
6. Stream-based memory access
"""

OPTIMIZER_SYSTEM_PROMPT = """You are an HLS optimization expert.
Given a Vitis HLS kernel and specific optimization feedback, apply the
suggested transformations to reduce latency.

Rules:
- Apply ONLY the suggested optimizations
- Preserve functional correctness
- Keep code readable and maintainable
- Output the complete modified C++ file
"""
```

### 9.6 Core Optimization Loop (`kernel_optimizer.py`)

```python
class KernelOptimizer:
    """Iterative HLS kernel optimization using LLM agents."""

    def __init__(
        self,
        llm: LLMClient,
        hls: HLSRunner,
        max_iterations: int = 25,
        max_compile_retries: int = 3,
        max_runtime_retries: int = 3,
    ):
        self.llm = llm
        self.hls = hls
        self.max_iterations = max_iterations
        self.max_compile_retries = max_compile_retries
        self.max_runtime_retries = max_runtime_retries

    def optimize(self, op: TransformerOperatorSpec) -> KernelOptimizationResult:
        """Run the full optimization loop.

        Algorithm (matches LAAFD workflow):

        Phase 1 - Translation:
          1. Send operator spec to LLM with TRANSLATOR prompt
          2. Receive generated HLS C++ code

        Phase 2 - Validation:
          3. Attempt HLS synthesis
          4. If compilation fails:
             a. Send error + code to LLM with COMPILE_FIXER prompt
             b. Retry up to max_compile_retries times
             c. If all retries fail, return failure result
          5. Run co-simulation (if testbench provided)
          6. If co-sim fails:
             a. Send error + code + testbench to LLM with RUNTIME_FIXER prompt
             b. Retry up to max_runtime_retries times
          7. Record synthesis results

        Phase 3 - Optimization Loop:
          8. Send code + synthesis report to LLM JUDGE
          9. If JUDGE says "PASS", terminate successfully
          10. If JUDGE says "OPTIMIZE: ...", send feedback to LLM OPTIMIZER
          11. Receive optimized code
          12. Go to step 3 (validate optimized code)
          13. If validated, go to step 8 (judge again)
          14. If not validated, revert to last valid code
          15. Repeat until PASS or max_iterations reached

        Returns: KernelOptimizationResult with full iteration history.
        """
        iterations: list[OptimizationIteration] = []
        current_code = self._translate(op)
        best_code = None
        best_synthesis = None

        for i in range(self.max_iterations):
            # Validate
            synth = self._validate(op, current_code)
            if synth is None:
                # Validation failed; revert
                if best_code is not None:
                    current_code = best_code
                    continue
                else:
                    # No valid code at all
                    break

            # Record best
            if best_synthesis is None or (
                synth.estimated_latency_cycles is not None
                and (best_synthesis.estimated_latency_cycles is None
                     or synth.estimated_latency_cycles < best_synthesis.estimated_latency_cycles)
            ):
                best_code = current_code
                best_synthesis = synth

            # Judge
            feedback = self._judge(op, current_code, synth)
            if feedback.startswith("PASS"):
                iterations.append(OptimizationIteration(
                    iteration=i,
                    source_code=current_code,
                    synthesis_result=synth,
                    cosim_result=None,
                    judge_feedback=feedback,
                    action_taken="accept",
                ))
                return KernelOptimizationResult(
                    operator=op,
                    final_source=current_code,
                    final_synthesis=synth,
                    iterations=iterations,
                    total_iterations=i + 1,
                    converged=True,
                    estimated_latency_cycles=synth.estimated_latency_cycles or 0,
                    resource_utilization=self._extract_utilization(synth),
                )

            # Optimize
            current_code = self._optimize(current_code, feedback, synth)

            iterations.append(OptimizationIteration(
                iteration=i,
                source_code=current_code,
                synthesis_result=synth,
                cosim_result=None,
                judge_feedback=feedback,
                action_taken="optimize",
            ))

        # Didn't converge; return best result
        return KernelOptimizationResult(
            operator=op,
            final_source=best_code or current_code,
            final_synthesis=best_synthesis,
            iterations=iterations,
            total_iterations=len(iterations),
            converged=False,
            estimated_latency_cycles=(best_synthesis.estimated_latency_cycles or 0) if best_synthesis else 0,
            resource_utilization=self._extract_utilization(best_synthesis) if best_synthesis else {},
        )
```

### 9.7 Private Methods (the Four Agents)

Each private method maps to one of the LAAFD agents:

```python
def _translate(self, op: TransformerOperatorSpec) -> str:
    """Agent: Translator. Generates initial HLS code from operator spec."""
    prompt = TRANSLATOR_USER_TEMPLATE.format(
        op_type=op.op_type,
        input_shapes=op.input_shapes,
        output_shapes=op.output_shapes,
        precision=op.precision.value,
        target_device=op.target_board.name,
        clock_mhz=op.clock_mhz,
        top_function=f"kernel_{op.op_type}",
    )
    response = self.llm.generate(TRANSLATOR_SYSTEM_PROMPT, prompt)
    return self._extract_code_block(response)

def _validate(self, op: TransformerOperatorSpec, code: str) -> HLSSynthesisResult | None:
    """Agent: Validator. Synthesize and optionally co-simulate."""
    kernel = KernelSpec(
        name=f"kernel_{op.op_type}",
        source_code=code,
        testbench_code=None,  # Phase 8 adds testbenches
        target_device=self._resolve_device_string(op.target_board),
        clock_period_ns=1000.0 / op.clock_mhz,
        top_function=f"kernel_{op.op_type}",
    )
    result = self.hls.synthesize(kernel)
    if not result.success:
        # Try to fix compilation errors
        fixed = self._fix_compile(code, result.error_message)
        if fixed:
            kernel_fixed = replace(kernel, source_code=fixed)
            result = self.hls.synthesize(kernel_fixed)
        if not result.success:
            return None
    return result

def _judge(self, op: TransformerOperatorSpec, code: str, synth: HLSSynthesisResult) -> str:
    """Agent: Judge. Evaluate if kernel is near-optimal."""
    theoretical_min = self._compute_theoretical_min(op)
    prompt = JUDGE_USER_TEMPLATE.format(
        source_code=code,
        latency_cycles=synth.estimated_latency_cycles,
        clock_period_ns=synth.clock_period_ns,
        clock_mhz=int(1000 / synth.clock_period_ns) if synth.clock_period_ns else "?",
        bram_used=synth.bram_used or 0,
        bram_available=synth.bram_available or "?",
        bram_pct=(synth.bram_utilization or 0) * 100,
        dsp_used=synth.dsp_used or 0,
        dsp_available=synth.dsp_available or "?",
        lut_used=synth.lut_used or 0,
        lut_available=synth.lut_available or "?",
        theoretical_min=theoretical_min,
    )
    return self.llm.generate(JUDGE_SYSTEM_PROMPT, prompt)

def _optimize(self, code: str, feedback: str, synth: HLSSynthesisResult) -> str:
    """Agent: Optimizer. Apply suggested optimizations."""
    prompt = f"""Apply these optimizations to the kernel:

Feedback from judge:
{feedback}

Current synthesis latency: {synth.estimated_latency_cycles} cycles

Current source code:
```cpp
{code}
```

Output the complete optimized C++ file.
"""
    response = self.llm.generate(OPTIMIZER_SYSTEM_PROMPT, prompt)
    return self._extract_code_block(response)
```

### 9.8 Theoretical Minimum Computation

```python
def _compute_theoretical_min(self, op: TransformerOperatorSpec) -> int:
    """Compute theoretical minimum latency in cycles for an operator.

    For GEMM (M, K) x (K, N):
      - Minimum = M * N * K * 2 / (DSP_count * ops_per_dsp_per_cycle)
      - For INT8 on DSP48E2: 2 MACs per DSP per cycle
      - For INT4: 4 MACs per DSP per cycle

    For attention:
      - QK^T: 2 * num_heads * seq_len * head_dim / DSPs
      - softmax: seq_len * num_heads (comparatively cheap)
      - score * V: 2 * num_heads * seq_len * head_dim / DSPs

    This is a lower bound. Real implementations add pipeline fill/drain overhead.
    """
```

### 9.9 Testing Strategy

Testing the full agentic loop requires mocking:
- `MockLLMClient`: Returns predefined code strings based on prompt keywords
- `MockHLSRunner`: Returns predefined synthesis results

```python
class MockLLMClient(LLMClient):
    def __init__(self, responses: dict[str, str]):
        """responses maps prompt substrings to response strings."""
        self.responses = responses
        self.call_log: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.call_log.append((system_prompt, user_prompt))
        for key, response in self.responses.items():
            if key in system_prompt or key in user_prompt:
                return response
        return "PASS"  # Default: accept
```

Test cases:
- `test_optimizer_converges_on_simple_kernel`: Mock a scenario where translation + 1 optimization iteration gets a PASS
- `test_optimizer_retries_on_compile_error`: Mock a compilation failure followed by a fix
- `test_optimizer_hits_max_iterations`: Verify graceful termination when judge never says PASS
- `test_optimizer_reverts_on_validation_failure`: Verify revert to last valid code

### 9.10 Dependency Policy

The `openai` package is an optional dependency. Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
hub = ["huggingface_hub>=0.20"]
agents = ["openai>=1.0"]
dev = ["pytest>=7.0"]
all = ["huggingface_hub>=0.20", "openai>=1.0"]
```

Import `openai` only inside `OpenAIClient.__init__`. If the user tries to use agent features without installing the package, raise a clean error.

### 9.11 Definition of Done

- [ ] `KernelOptimizer.optimize()` runs to completion with mock LLM and HLS
- [ ] Iteration history is correctly recorded
- [ ] Compile-error retry logic works
- [ ] Revert-to-best logic works
- [ ] Max iteration limit is respected
- [ ] All prompt templates are populated without KeyError for any operator type

---

## 10. Phase 8: Multi-Kernel Orchestration

### 10.1 Purpose

Optimize all kernels in an LLM inference pipeline (not just one), managing inter-kernel dependencies and shared resource constraints.

### 10.2 File: `hardware_feasibility/agents/orchestrator.py`

### 10.3 Transformer Operator Decomposition

A single transformer layer in a Llama-family model produces these operators:

| Operator | GEMM shape | Precision | Notes |
|---|---|---|---|
| Q projection | (1, h) x (h, n_heads * head_dim) | weight_precision | Decode: seq=1 |
| K projection | (1, h) x (h, n_kv * head_dim) | weight_precision | |
| V projection | (1, h) x (h, n_kv * head_dim) | weight_precision | |
| QK^T attention | (n_heads, 1, head_dim) x (n_heads, head_dim, seq) | kv_precision | Batched |
| Softmax | (n_heads, 1, seq) | fp32 (always) | Numerical stability |
| Score * V | (n_heads, 1, seq) x (n_heads, seq, head_dim) | kv_precision | |
| O projection | (1, n_heads * head_dim) x (n_heads * head_dim, h) | weight_precision | |
| Gate projection | (1, h) x (h, inter) | weight_precision | SwiGLU MLP |
| Up projection | (1, h) x (h, inter) | weight_precision | SwiGLU MLP |
| SiLU activation | (1, inter) | fp32 | Element-wise |
| Gate * Up | (1, inter) | fp32 | Element-wise |
| Down projection | (1, inter) x (inter, h) | weight_precision | |
| RMSNorm (x2) | (1, h) | fp32 | Pre-attention and pre-MLP |
| Residual add (x2) | (1, h) | fp32 | Element-wise |

Total per layer: ~12 distinct operators. For 32 layers + embedding + LM head: ~390 operators.

### 10.4 Shared vs. Per-Layer Kernels

Most operators repeat identically across layers (same GEMM shape, same precision). Optimization strategy:

1. **Identify unique operator shapes**: Group the ~390 operators into equivalence classes by (op_type, shapes, precision). For Llama 3-8B, this yields ~8-10 unique kernels.

2. **Optimize each unique kernel once**: Run the Phase 7 optimizer on each unique kernel.

3. **Compose into full pipeline**: Verify that all kernels fit within the resource budget simultaneously.

```python
def decompose_model_to_operators(spec: ModelSpec) -> list[TransformerOperatorSpec]:
    """Break down a model into its constituent operators."""

def deduplicate_operators(ops: list[TransformerOperatorSpec]) -> list[TransformerOperatorSpec]:
    """Group identical operators and return unique set."""

def check_resource_budget(
    results: list[KernelOptimizationResult],
    board: BoardSpec,
) -> bool:
    """Verify that total resource usage of all kernels fits on the FPGA.

    Sum BRAM, DSP, LUT, FF across all kernels.
    Check each against board limits.
    """
```

### 10.5 Resource Budget Checking

Critical constraint: all kernels share the FPGA fabric. If kernel A uses 40% of BRAMs and kernel B uses 70% of BRAMs, they cannot coexist.

However, for inference, only ONE operator executes at a time (sequential execution per layer). This means kernels can be time-multiplexed on the same fabric if the FPGA supports dynamic partial reconfiguration, OR all kernels must fit simultaneously if using a static bitstream.

**Design decision**: Assume static bitstream (all kernels exist simultaneously). This is the common case for inference accelerators. Partial reconfiguration is an advanced feature for v3+.

For a static bitstream, the orchestrator must verify:

```
sum(kernel.bram_used for kernel in kernels) <= board.bram
sum(kernel.dsp_used for kernel in kernels) <= board.dsp_slices
sum(kernel.lut_used for kernel in kernels) <= board.lut_count * 1000
sum(kernel.ff_used for kernel in kernels) <= board.lut_count * 2000  # approximate
```

If the resource budget is exceeded, the orchestrator must:
1. Identify the most resource-hungry kernel
2. Request re-optimization with a stricter resource constraint
3. Accept higher latency in exchange for fewer resources

### 10.6 Definition of Done

- [ ] `decompose_model_to_operators` produces correct operator list for Llama 3-8B
- [ ] `deduplicate_operators` reduces ~390 operators to ~10 unique kernels
- [ ] `check_resource_budget` correctly sums resources and compares to board limits
- [ ] End-to-end orchestration works with mock LLM/HLS for a small model (Qwen2-0.5B)

---

## 11. Phase 9: Accuracy-in-the-Loop

### 11.1 Purpose

Integrate accuracy evaluation (perplexity on a calibration dataset) into the recommendation and sweep pipelines. This allows the system to answer: "What is the accuracy cost of INT4 quantization on this model?"

### 11.2 New Package: `hardware_feasibility/evaluation/`

```
hardware_feasibility/evaluation/
  __init__.py
  perplexity.py          # Perplexity evaluation
  accuracy_db.py         # Lookup table of known accuracy metrics
```

### 11.3 Accuracy Database (`accuracy_db.py`)

A lookup table of known perplexity values for common model-precision combinations:

```python
# Source: published quantization benchmarks
# Format: (model_family, weight_precision, kv_precision) -> perplexity on wikitext-2
KNOWN_PERPLEXITY: dict[tuple[str, str, str], float] = {
    ("llama3-8b", "fp16", "fp16"): 6.14,
    ("llama3-8b", "int8", "int8"): 6.19,
    ("llama3-8b", "int4", "int4"): 6.52,
    ("llama3-8b", "int4", "int8"): 6.38,
    ("llama3.2-1b", "fp16", "fp16"): 9.78,
    ("llama3.2-1b", "int8", "int8"): 9.84,
    ("llama3.2-1b", "int4", "int4"): 10.42,
    # ... extend as data becomes available
}
```

When empirical data is unavailable, provide estimates:

```python
def estimate_perplexity_degradation(base_ppl: float, precision: Precision) -> float:
    """Rough estimate of perplexity increase from quantization.

    Based on published trends across Llama family models:
    - INT8: ~0.5-1% perplexity increase
    - INT4 (GPTQ): ~3-8% perplexity increase
    - INT4 (AWQ): ~2-5% perplexity increase
    """
    multipliers = {
        Precision.FP16: 1.0,
        Precision.BF16: 1.0,
        Precision.INT8: 1.008,
        Precision.INT4: 1.05,  # Conservative; AWQ quality
    }
    return base_ppl * multipliers.get(precision, 1.0)
```

### 11.4 Integration with Recommender

The Phase 3 recommender adds accuracy data to each recommendation:

```python
@dataclass
class Recommendation:
    # ... existing fields ...
    estimated_perplexity: float | None
    perplexity_source: str             # "measured", "lookup", "estimated"
```

### 11.5 Definition of Done

- [ ] `KNOWN_PERPLEXITY` contains data for at least 5 model-precision combinations
- [ ] `estimate_perplexity_degradation` produces reasonable estimates
- [ ] The recommender includes perplexity in its output
- [ ] Perplexity appears in both report and JSON output formats

---

## 12. Phase 10: End-to-End Pipeline

### 12.1 Purpose

Combine all phases into a single command that takes a model ID and produces a complete deployment recommendation, including precision selection, hardware matching, optional decomposition, and (if Vitis HLS is available) optimized kernels.

### 12.2 CLI Integration

New top-level command:

```python
p.add_argument("--full-pipeline", action="store_true",
               help="Run the complete analysis + optimization pipeline.")
```

### 12.3 Pipeline Stages

```python
def run_full_pipeline(spec: ModelSpec, board: BoardSpec, llm: LLMClient | None) -> PipelineResult:
    """
    Stage 1: Feasibility screening (existing)
        -> If fits: proceed to Stage 2
        -> If does not fit: go to Stage 1b

    Stage 1b: Precision sweep (Phase 1)
        -> Find best fitting precision
        -> If none fit: go to Stage 1c

    Stage 1c: Decomposition planning (Phase 4)
        -> Propose multi-device plan
        -> If infeasible: return "model too large for this hardware class"

    Stage 2: Configuration recommendation (Phase 3)
        -> Select optimal (precision, context, batch) operating point

    Stage 3: Sensitivity analysis (Phase 2)
        -> Identify bottlenecks and headroom

    Stage 4 (optional, requires Vitis HLS + LLM):
        -> Kernel optimization (Phases 7-8)
        -> Generate optimized HLS for target board

    Stage 5: Final report
        -> Combine all results into comprehensive output
    """
```

### 12.4 Data Model

```python
@dataclass
class PipelineResult:
    """Complete pipeline output."""

    model_name: str
    target_board: str

    # Stage 1
    initial_verdict: VerdictResult

    # Stage 1b (if needed)
    precision_sweep: SweepResult | None

    # Stage 1c (if needed)
    decomposition: DecompositionResult | None

    # Stage 2
    recommendation: RecommendationResult

    # Stage 3
    sensitivity: SensitivityResult

    # Stage 4 (optional)
    kernel_results: list[KernelOptimizationResult] | None

    # Metadata
    pipeline_version: str
    total_runtime_seconds: float
```

### 12.5 Definition of Done

- [ ] `--full-pipeline` runs stages 1-3 without errors for any model-board pair
- [ ] JSON output contains all stage results
- [ ] Human-readable report includes a summary section explaining the recommendation
- [ ] Stage 4 is skipped cleanly if Vitis HLS or LLM is unavailable

---

## 13. Testing Strategy

### 13.1 Test Organization

```
tests/
  test_sweep.py               # Phase 1
  test_sensitivity.py          # Phase 2
  test_recommender.py          # Phase 3
  test_decomposition.py        # Phase 4
  test_board_specs_extended.py # Phase 5
  test_hls_runner.py           # Phase 6 (mostly mock-based)
  test_report_parser.py        # Phase 6
  test_kernel_optimizer.py     # Phase 7 (mock LLM + mock HLS)
  test_orchestrator.py         # Phase 8
  test_accuracy_db.py          # Phase 9
  test_pipeline.py             # Phase 10
  fixtures/
    sample_hls_report.xml      # For report parser tests
    sample_config.json         # For loader tests
```

### 13.2 Test Levels

**Unit tests**: Every function gets at least one test. Test with known inputs and verify exact outputs. Use `pytest.approx` for floating-point comparisons.

**Integration tests**: Test the full CLI with subprocess calls:

```python
def test_cli_sweep_json(tmp_path):
    result = subprocess.run(
        ["python", "-m", "hardware_feasibility",
         "--family", "llama3-8b", "--board", "Xilinx VCK190", "--sweep", "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "sweep" in data
    assert len(data["sweep"]["points"]) > 0
```

**Golden tests**: For critical calculations (memory, bandwidth, FLOPs), maintain golden output files that are checked into Git. Tests compare current output against golden values. Update golden files intentionally when formulas change.

### 13.3 Running Tests

```bash
# All tests
pytest

# Specific phase
pytest tests/test_sweep.py

# With coverage
pytest --cov=hardware_feasibility --cov-report=term-missing
```

---

## 14. Coding Conventions

### 14.1 Style

- Python 3.10+ (match statements allowed; `X | Y` union types allowed)
- Type annotations on all public functions
- Dataclasses for all structured data; no plain dicts crossing module boundaries
- `from __future__ import annotations` at the top of every file
- No global mutable state except `BOARD_DATABASE` and `KNOWN_FAMILIES` (both are populated at import time and then read-only)

### 14.2 File Layout

Every `.py` file follows this order:

```python
"""Module docstring."""

from __future__ import annotations

# stdlib imports
# third-party imports
# local imports

# Constants

# Dataclasses

# Functions (public, then private prefixed with _)
```

### 14.3 Error Handling

- Raise `ValueError` for bad user input (wrong precision string, unknown family name)
- Raise `RuntimeError` for infrastructure failures (Vitis HLS not found, synthesis timeout)
- Never catch broad `Exception` unless re-raising with additional context
- Use `Optional` / `None` returns for "not available" situations (e.g., no compute target provided)

### 14.4 Naming

- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Dataclass fields: `snake_case`
- CLI flags: `--kebab-case`

---

## 15. Dependency Policy

| Package | When needed | Install group | Import location | Phase |
|---|---|---|---|---|
| (none) | Phases 1-5, 9 | core | everywhere | 1-5, 9 |
| `huggingface_hub` | `--hf-model` flag | `pip install model2hw[hub]` | `hf_config_loader.py` only | 0 |
| `openai` | `--optimize` flag (Phase 7+) | `pip install model2hw[agents]` | `llm_client.py` only | 7-8 |
| Vitis HLS | `--synthesize`/`--optimize` | System install | `hls_runner.py` only | 6-8 |
| `pytest` | Testing | `pip install model2hw[dev]` | test files only | all |

Core functionality (Phases 1-5 and Phase 9) requires ZERO external dependencies and ZERO LLM calls. This is a hard constraint. Do not add dependencies to the core package. Do not add LLM calls to any module outside `hardware_feasibility/agents/`.

---

## 16. Glossary

| Term | Definition |
|---|---|
| **GQA** | Grouped Query Attention. num_kv_heads < num_attention_heads. Reduces KV cache size. |
| **MHA** | Multi-Head Attention. num_kv_heads == num_attention_heads. |
| **SwiGLU** | Swish-Gated Linear Unit. MLP style used in Llama. Three projections: gate, up, down. |
| **HLS** | High-Level Synthesis. Compiles C++ to FPGA hardware (RTL). |
| **RTL** | Register Transfer Level. Hardware description (Verilog/VHDL). |
| **BRAM** | Block RAM. On-chip memory on Xilinx FPGAs. 36 Kb per block. |
| **URAM** | UltraRAM. High-capacity on-chip memory on Xilinx UltraScale+. 288 Kb per block. |
| **DSP** | Digital Signal Processing slice. Fixed-function multiplier on FPGAs. |
| **LUT** | Look-Up Table. Basic logic element on FPGAs. |
| **Co-simulation** | Running C++ testbench against HLS-generated RTL to verify functional correctness. |
| **Roofline** | Performance model. Workload is either compute-bound or memory-bandwidth-bound. |
| **Arithmetic intensity** | FLOPs per byte of memory traffic. Low AI = memory-bound. |
| **Pipeline bubble** | Idle time in pipeline parallelism when stages wait for each other. |
| **Tensor parallelism** | Splitting a single layer's computation across multiple devices. |
| **Pipeline parallelism** | Assigning different layers to different devices in sequence. |
| **KV cache** | Key-Value cache. Stores attention keys and values from previous tokens to avoid recomputation. |
| **Prefill** | Phase where all prompt tokens are processed in parallel. Compute-bound. |
| **Decode** | Phase where tokens are generated one at a time. Memory-bandwidth-bound. |
| **tok/s** | Tokens per second. Throughput metric for LLM inference. |
| **TOPS** | Tera Operations Per Second. Compute throughput metric. |
