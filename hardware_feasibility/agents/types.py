"""Data models for the agentic kernel optimization loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.architecture_rules import Precision
from ..hardware.board_specs import BoardSpec
from ..synthesis.types import HLSSynthesisResult, HLSCoSimResult


@dataclass
class TransformerOperatorSpec:
    """Specification of a single transformer operator to implement in HLS."""

    op_type: str  # "gemm", "attention_qkv", "softmax", "layernorm", "mlp_swiglu"
    input_shapes: dict[str, tuple[int, ...]]
    output_shapes: dict[str, tuple[int, ...]]
    precision: Precision
    target_board: BoardSpec
    clock_mhz: int = 200


@dataclass
class OptimizationIteration:
    """Record of a single optimization iteration."""

    iteration: int
    source_code: str
    synthesis_result: Optional[HLSSynthesisResult] = None
    cosim_result: Optional[HLSCoSimResult] = None
    judge_feedback: str = ""
    action_taken: str = ""  # "optimize", "fix_compile", "fix_runtime", "accept"


@dataclass
class KernelOptimizationResult:
    """Final result of kernel optimization."""

    operator: TransformerOperatorSpec
    final_source: str
    final_synthesis: Optional[HLSSynthesisResult]
    iterations: list[OptimizationIteration] = field(default_factory=list)
    total_iterations: int = 0
    converged: bool = False
    estimated_latency_cycles: int = 0
    resource_utilization: dict[str, float] = field(default_factory=dict)
