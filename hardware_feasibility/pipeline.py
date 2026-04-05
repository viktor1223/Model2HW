"""End-to-end analysis pipeline combining all phases."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .models.architecture_rules import ModelSpec
from .hardware.board_specs import BoardSpec
from .analysis.memory import analyze_memory
from .analysis.bandwidth import analyze_bandwidth
from .analysis.compute import analyze_compute
from .analysis.io import analyze_io
from .analysis.verdict import render_verdict, VerdictResult, FeasibilityVerdict
from .analysis.sweep import run_precision_sweep, SweepResult
from .analysis.sensitivity import analyze_sensitivity, SensitivityResult
from .analysis.recommender import recommend_configuration, RecommendationResult
from .analysis.decomposition import plan_decomposition, DecompositionResult


@dataclass
class PipelineResult:
    """Complete pipeline output."""

    model_name: str
    target_board: str

    # Stage 1: Initial feasibility
    initial_verdict: VerdictResult

    # Stage 1b: Precision sweep (if initial does not fit)
    precision_sweep: Optional[SweepResult] = None

    # Stage 1c: Decomposition (if no single-device config fits)
    decomposition: Optional[DecompositionResult] = None

    # Stage 2: Configuration recommendation
    recommendation: Optional[RecommendationResult] = None

    # Stage 3: Sensitivity analysis
    sensitivity: Optional[SensitivityResult] = None

    # Metadata
    pipeline_version: str = "0.1.0"
    total_runtime_seconds: float = 0.0


def run_full_pipeline(
    spec: ModelSpec,
    board: BoardSpec,
) -> PipelineResult:
    """Run the complete analysis pipeline (stages 1-3).

    Stage 1:  Feasibility screening
    Stage 1b: Precision sweep (if initial verdict is DOES_NOT_FIT)
    Stage 1c: Decomposition planning (if no precision fits)
    Stage 2:  Configuration recommendation
    Stage 3:  Sensitivity analysis
    """
    t0 = time.monotonic()

    mem = analyze_memory(spec)
    bw = analyze_bandwidth(spec)
    compute = analyze_compute(spec, 0)
    io = analyze_io(spec)

    # Stage 1: Initial feasibility check
    initial_verdict = render_verdict(
        spec, mem, bw, compute, io,
        target_memory_gb=board.memory_gb,
        target_bandwidth_gbps=board.memory_bandwidth_gbps,
        target_link_bandwidth_gbps=board.host_link_bandwidth_gbps,
    )

    result = PipelineResult(
        model_name=spec.name,
        target_board=board.name,
        initial_verdict=initial_verdict,
    )

    # Stage 1b: Precision sweep if initial config does not fit
    sweep: Optional[SweepResult] = None
    if initial_verdict.verdict == FeasibilityVerdict.DOES_NOT_FIT:
        sweep = run_precision_sweep(
            spec,
            target_memory_gb=board.memory_gb,
            target_bandwidth_gbps=board.memory_bandwidth_gbps,
            target_link_bandwidth_gbps=board.host_link_bandwidth_gbps,
        )
        result.precision_sweep = sweep

    # Stage 1c: Decomposition if no single precision fits
    if sweep is not None and not sweep.fitting_points:
        result.decomposition = plan_decomposition(spec, board)

    # Stage 2: Configuration recommendation
    result.recommendation = recommend_configuration(spec, board)

    # Stage 3: Sensitivity analysis
    target_tops = board.peak_tops_int8
    result.sensitivity = analyze_sensitivity(
        spec, mem, bw, compute, io,
        target_memory_gb=board.memory_gb,
        target_bandwidth_gbps=board.memory_bandwidth_gbps,
        target_link_bandwidth_gbps=board.host_link_bandwidth_gbps,
        target_tops=target_tops,
    )

    result.total_runtime_seconds = time.monotonic() - t0
    return result
