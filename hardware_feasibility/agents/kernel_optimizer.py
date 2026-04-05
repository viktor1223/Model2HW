"""Iterative HLS kernel optimization using LLM agents (LAAFD workflow)."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Optional

from .llm_client import LLMClient
from .prompts import (
    COMPILE_FIXER_SYSTEM_PROMPT,
    COMPILE_FIXER_USER_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    OPTIMIZER_SYSTEM_PROMPT,
    OPTIMIZER_USER_TEMPLATE,
    TRANSLATOR_SYSTEM_PROMPT,
    TRANSLATOR_USER_TEMPLATE,
)
from .types import (
    KernelOptimizationResult,
    OptimizationIteration,
    TransformerOperatorSpec,
)
from ..models.architecture_rules import Precision
from ..synthesis.hls_runner import HLSRunner
from ..synthesis.types import HLSSynthesisResult, KernelSpec


class KernelOptimizer:
    """Iterative HLS kernel optimization using LLM agents."""

    def __init__(
        self,
        llm: LLMClient,
        hls: HLSRunner,
        max_iterations: int = 25,
        max_compile_retries: int = 3,
        max_runtime_retries: int = 3,
    ) -> None:
        self.llm = llm
        self.hls = hls
        self.max_iterations = max_iterations
        self.max_compile_retries = max_compile_retries
        self.max_runtime_retries = max_runtime_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, op: TransformerOperatorSpec) -> KernelOptimizationResult:
        """Run the full LAAFD optimization loop.

        Phase 1 - Translation:
          Send operator spec to LLM with TRANSLATOR prompt.

        Phase 2 - Validation:
          Synthesize; retry compile errors up to *max_compile_retries*.

        Phase 3 - Optimization loop:
          Judge -> Optimize -> Validate, until PASS or *max_iterations*.
        """
        iterations: list[OptimizationIteration] = []
        current_code = self._translate(op)
        best_code: Optional[str] = None
        best_synthesis: Optional[HLSSynthesisResult] = None

        for i in range(self.max_iterations):
            # Validate current code
            synth = self._validate(op, current_code)

            if synth is None:
                # Validation failed after retries
                if best_code is not None:
                    current_code = best_code
                    iterations.append(OptimizationIteration(
                        iteration=i,
                        source_code=current_code,
                        judge_feedback="",
                        action_taken="revert",
                    ))
                    continue
                else:
                    # No valid code exists at all
                    break

            # Track best result
            if best_synthesis is None or (
                synth.estimated_latency_cycles is not None
                and (
                    best_synthesis.estimated_latency_cycles is None
                    or synth.estimated_latency_cycles
                    < best_synthesis.estimated_latency_cycles
                )
            ):
                best_code = current_code
                best_synthesis = synth

            # Judge
            feedback = self._judge(op, current_code, synth)

            if feedback.upper().startswith("PASS"):
                iterations.append(OptimizationIteration(
                    iteration=i,
                    source_code=current_code,
                    synthesis_result=synth,
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
            optimized_code = self._optimize(current_code, feedback, synth)
            iterations.append(OptimizationIteration(
                iteration=i,
                source_code=current_code,
                synthesis_result=synth,
                judge_feedback=feedback,
                action_taken="optimize",
            ))
            current_code = optimized_code

        # Did not converge - return best result
        return KernelOptimizationResult(
            operator=op,
            final_source=best_code or current_code,
            final_synthesis=best_synthesis,
            iterations=iterations,
            total_iterations=len(iterations),
            converged=False,
            estimated_latency_cycles=(
                (best_synthesis.estimated_latency_cycles or 0)
                if best_synthesis
                else 0
            ),
            resource_utilization=(
                self._extract_utilization(best_synthesis) if best_synthesis else {}
            ),
        )

    # ------------------------------------------------------------------
    # Agent 1: Translator
    # ------------------------------------------------------------------

    def _translate(self, op: TransformerOperatorSpec) -> str:
        """Generate initial HLS C++ code from operator spec."""
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

    # ------------------------------------------------------------------
    # Agent 2: Validator (synthesis + compile-error retry)
    # ------------------------------------------------------------------

    def _validate(
        self, op: TransformerOperatorSpec, code: str
    ) -> Optional[HLSSynthesisResult]:
        """Synthesize code, retrying compile errors with LLM fixes."""
        kernel = KernelSpec(
            name=f"kernel_{op.op_type}",
            source_code=code,
            testbench_code=None,
            target_device=self._resolve_device_string(op.target_board),
            clock_period_ns=1000.0 / op.clock_mhz,
            top_function=f"kernel_{op.op_type}",
        )

        result = self.hls.synthesize(kernel)
        if result.success:
            return result

        # Attempt compile-error fix retries
        current = code
        for _ in range(self.max_compile_retries):
            fixed = self._fix_compile(current, result.error_message or "")
            if fixed is None:
                return None
            fixed_kernel = replace(kernel, source_code=fixed)
            result = self.hls.synthesize(fixed_kernel)
            if result.success:
                return result
            current = fixed

        return None

    def _fix_compile(self, code: str, error: str) -> Optional[str]:
        """Ask LLM to fix a compilation error."""
        prompt = COMPILE_FIXER_USER_TEMPLATE.format(
            source_code=code,
            error_message=error,
        )
        response = self.llm.generate(COMPILE_FIXER_SYSTEM_PROMPT, prompt)
        extracted = self._extract_code_block(response)
        return extracted if extracted else None

    # ------------------------------------------------------------------
    # Agent 3: Judge
    # ------------------------------------------------------------------

    def _judge(
        self,
        op: TransformerOperatorSpec,
        code: str,
        synth: HLSSynthesisResult,
    ) -> str:
        """Evaluate whether the kernel is near-optimal."""
        theoretical_min = self._compute_theoretical_min(op)
        prompt = JUDGE_USER_TEMPLATE.format(
            source_code=code,
            latency_cycles=synth.estimated_latency_cycles or 0,
            clock_period_ns=synth.clock_period_ns or 0,
            clock_mhz=(
                int(1000 / synth.clock_period_ns) if synth.clock_period_ns else 0
            ),
            bram_used=synth.bram_used or 0,
            bram_available=synth.bram_available or "?",
            bram_pct=(synth.bram_utilization or 0) * 100,
            dsp_used=synth.dsp_used or 0,
            dsp_available=synth.dsp_available or "?",
            dsp_pct=(synth.dsp_utilization or 0) * 100,
            lut_used=synth.lut_used or 0,
            lut_available=synth.lut_available or "?",
            lut_pct=(synth.lut_utilization or 0) * 100,
            theoretical_min=theoretical_min,
        )
        return self.llm.generate(JUDGE_SYSTEM_PROMPT, prompt)

    # ------------------------------------------------------------------
    # Agent 4: Optimizer
    # ------------------------------------------------------------------

    def _optimize(
        self, code: str, feedback: str, synth: HLSSynthesisResult
    ) -> str:
        """Apply suggested optimizations."""
        prompt = OPTIMIZER_USER_TEMPLATE.format(
            feedback=feedback,
            latency_cycles=synth.estimated_latency_cycles or 0,
            source_code=code,
        )
        response = self.llm.generate(OPTIMIZER_SYSTEM_PROMPT, prompt)
        return self._extract_code_block(response)

    # ------------------------------------------------------------------
    # Theoretical minimum computation
    # ------------------------------------------------------------------

    def _compute_theoretical_min(self, op: TransformerOperatorSpec) -> int:
        """Compute theoretical minimum latency in cycles for an operator.

        Lower bound assuming perfect pipelining and full DSP utilization.
        """
        dsp_count = op.target_board.dsp_slices or 1

        # Operations per DSP per cycle depends on precision
        ops_per_dsp = {
            Precision.FP32: 1,
            Precision.FP16: 1,
            Precision.BF16: 1,
            Precision.INT8: 2,  # DSP48E2: 2 MACs per cycle
            Precision.INT4: 4,  # Packed: 4 MACs per cycle
        }.get(op.precision, 1)

        if op.op_type == "gemm":
            # GEMM (M, K) x (K, N): 2*M*N*K operations
            inp = op.input_shapes.get("input", (1, 1))
            wt = op.input_shapes.get("weight", (1, 1))
            m = inp[0] if len(inp) >= 1 else 1
            k = inp[1] if len(inp) >= 2 else 1
            n = wt[1] if len(wt) >= 2 else 1
            total_ops = 2 * m * n * k
            return max(1, total_ops // (dsp_count * ops_per_dsp))

        if op.op_type == "attention_qkv":
            # QK^T + softmax + score*V
            # Simplified: dominant term is 2 * num_heads * seq * head_dim GEMMs
            inp = op.input_shapes.get("input", (1, 1))
            seq_len = inp[0] if len(inp) >= 1 else 1
            hidden = inp[1] if len(inp) >= 2 else 1
            total_ops = 4 * seq_len * hidden  # rough lower bound
            return max(1, total_ops // (dsp_count * ops_per_dsp))

        if op.op_type in ("softmax", "layernorm", "silu"):
            # Element-wise: proportional to number of elements
            out = op.output_shapes.get("output", (1,))
            n_elements = 1
            for dim in out:
                n_elements *= dim
            # ~5 ops per element (exp, sum, div, etc.)
            return max(1, 5 * n_elements // dsp_count)

        if op.op_type == "mlp_swiglu":
            # Gate + Up + SiLU + multiply + Down: dominated by 3 GEMMs
            inp = op.input_shapes.get("input", (1, 1))
            wt = op.input_shapes.get("weight", (1, 1))
            m = inp[0] if len(inp) >= 1 else 1
            k = inp[1] if len(inp) >= 2 else 1
            n = wt[1] if len(wt) >= 2 else 1
            total_ops = 3 * 2 * m * n * k  # 3 GEMMs
            return max(1, total_ops // (dsp_count * ops_per_dsp))

        # Default: assume proportional to output size
        out = op.output_shapes.get("output", (1,))
        n_elements = 1
        for dim in out:
            n_elements *= dim
        return max(1, n_elements // dsp_count)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code_block(response: str) -> str:
        """Extract C++ code from a markdown code block in an LLM response."""
        # Try to find ```cpp ... ``` or ``` ... ```
        pattern = r"```(?:cpp|c\+\+)?\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fall back to the entire response (already code)
        return response.strip()

    @staticmethod
    def _resolve_device_string(board: BoardSpec) -> str:
        """Map a BoardSpec to a Vitis HLS part string."""
        # Known FPGA part numbers
        device_map = {
            "Xilinx ZCU104": "xczu7ev-ffvc1156-2-e",
            "Xilinx VCK190": "xcvc1902-vsva2197-2MP-e-S",
            "Xilinx Alveo U250": "xcu250-figd2104-2L-e",
            "Xilinx Alveo U55C": "xcu55c-fsvh2892-2L-e",
            "Xilinx VE2802": "xcve2802-vsvh1760-2MP-e-S",
        }
        return device_map.get(board.name, board.name)

    @staticmethod
    def _extract_utilization(synth: Optional[HLSSynthesisResult]) -> dict[str, float]:
        """Extract resource utilization percentages from synthesis result."""
        if synth is None:
            return {}
        util: dict[str, float] = {}
        if synth.bram_utilization is not None:
            util["bram"] = synth.bram_utilization
        if synth.dsp_utilization is not None:
            util["dsp"] = synth.dsp_utilization
        if synth.ff_utilization is not None:
            util["ff"] = synth.ff_utilization
        if synth.lut_utilization is not None:
            util["lut"] = synth.lut_utilization
        return util
