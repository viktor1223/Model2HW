"""Hardware matcher: rank boards against a model's requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models.architecture_rules import ModelSpec
from ..analysis.memory import MemoryProfile
from ..analysis.bandwidth import BandwidthProfile
from ..analysis.io import IOProfile
from ..analysis.verdict import FeasibilityVerdict, VerdictResult, render_verdict
from .board_specs import BoardSpec, list_boards


@dataclass
class MatchResult:
    """Result of matching a model against a specific board."""

    board: BoardSpec
    verdict: VerdictResult

    # Derived convenience fields
    memory_utilization: float  # fraction of board memory used
    bandwidth_utilization: float  # fraction of board bandwidth needed
    estimated_tok_per_sec: Optional[float]  # estimated achievable tok/s

    @property
    def fits(self) -> bool:
        return self.verdict.verdict != FeasibilityVerdict.DOES_NOT_FIT


def match_board(
    spec: ModelSpec,
    board: BoardSpec,
    mem: MemoryProfile,
    bw: BandwidthProfile,
    io: IOProfile,
) -> MatchResult:
    """Evaluate a single board against the model's requirements."""
    from ..analysis.compute import analyze_compute

    compute = analyze_compute(spec, board.memory_bandwidth_gbps * (1 << 30))

    v = render_verdict(
        spec,
        mem,
        bw,
        compute,
        io,
        target_memory_gb=board.memory_gb,
        target_bandwidth_gbps=board.memory_bandwidth_gbps,
        target_link_bandwidth_gbps=board.host_link_bandwidth_gbps,
    )

    mem_util = mem.total_gb / board.memory_gb if board.memory_gb > 0 else float("inf")
    bw_util = (
        bw.required_bandwidth_gbps / board.memory_bandwidth_gbps
        if board.memory_bandwidth_gbps > 0
        else float("inf")
    )

    # Estimate achievable tok/s (bandwidth-limited)
    if bw.total_bytes_per_token > 0 and board.memory_bandwidth_gbps > 0:
        board_bw_bytes = board.memory_bandwidth_gbps * (1 << 30)
        est_tok_s = board_bw_bytes / bw.total_bytes_per_token
        # Clamp to zero if memory doesn't even fit
        if v.verdict == FeasibilityVerdict.DOES_NOT_FIT:
            est_tok_s = 0.0
    else:
        est_tok_s = None

    return MatchResult(
        board=board,
        verdict=v,
        memory_utilization=mem_util,
        bandwidth_utilization=bw_util,
        estimated_tok_per_sec=est_tok_s,
    )


def rank_boards(
    spec: ModelSpec,
    mem: MemoryProfile,
    bw: BandwidthProfile,
    io: IOProfile,
    *,
    category: Optional[str] = None,
) -> list[MatchResult]:
    """Rank all known boards against the model requirements.

    Returns results sorted: fitting boards first (by best estimated tok/s),
    then non-fitting boards.
    """
    boards = list_boards(category=category)
    results = [match_board(spec, b, mem, bw, io) for b in boards]

    # Sort: fitting first, then by estimated tok/s descending
    def sort_key(r: MatchResult) -> tuple:
        fits = 0 if r.fits else 1
        tok_s = -(r.estimated_tok_per_sec or 0)
        return (fits, tok_s)

    results.sort(key=sort_key)
    return results
