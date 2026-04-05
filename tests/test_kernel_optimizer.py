"""Tests for Phase 7: Agentic Kernel Optimizer."""

from __future__ import annotations

import pytest

from hardware_feasibility.agents.llm_client import LLMClient
from hardware_feasibility.agents.types import (
    KernelOptimizationResult,
    OptimizationIteration,
    TransformerOperatorSpec,
)
from hardware_feasibility.agents.kernel_optimizer import KernelOptimizer
from hardware_feasibility.agents.prompts import (
    TRANSLATOR_SYSTEM_PROMPT,
    TRANSLATOR_USER_TEMPLATE,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    OPTIMIZER_SYSTEM_PROMPT,
    OPTIMIZER_USER_TEMPLATE,
    COMPILE_FIXER_SYSTEM_PROMPT,
    COMPILE_FIXER_USER_TEMPLATE,
)
from hardware_feasibility.hardware.board_specs import BoardSpec
from hardware_feasibility.models.architecture_rules import Precision
from hardware_feasibility.synthesis.types import HLSSynthesisResult, KernelSpec


# ---------------------------------------------------------------------------
# Mock LLM Client
# ---------------------------------------------------------------------------

class MockLLMClient(LLMClient):
    """Returns predefined responses based on prompt substring matching."""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.call_log: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.call_log.append((system_prompt, user_prompt))
        for key, response in self.responses.items():
            if key in system_prompt or key in user_prompt:
                return response
        return "PASS"


# ---------------------------------------------------------------------------
# Mock HLS Runner
# ---------------------------------------------------------------------------

class MockHLSRunner:
    """Returns predefined synthesis results.

    Mimics the HLSRunner interface without requiring Vitis HLS.
    """

    def __init__(self, results: list[HLSSynthesisResult]) -> None:
        self._results = list(results)
        self._call_index = 0
        self.call_log: list[KernelSpec] = []

    def synthesize(self, kernel: KernelSpec, **kwargs) -> HLSSynthesisResult:
        self.call_log.append(kernel)
        if self._call_index < len(self._results):
            result = self._results[self._call_index]
            self._call_index += 1
            return result
        # Default: successful synthesis
        return HLSSynthesisResult(
            success=True,
            estimated_latency_cycles=1000,
            clock_period_ns=5.0,
            bram_used=10,
            bram_available=100,
            dsp_used=50,
            dsp_available=200,
            lut_used=5000,
            lut_available=50000,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_board() -> BoardSpec:
    return BoardSpec(
        name="Xilinx VCK190",
        category="fpga",
        memory_gb=32.0,
        memory_bandwidth_gbps=102.4,
        peak_tops_int8=400.0,
        host_link="PCIe Gen4 x8",
        host_link_bandwidth_gbps=16.0,
        bram_kb=32400,
        uram_kb=36000,
        dsp_slices=1968,
        lut_count=899,
    )


def _make_op(op_type: str = "gemm") -> TransformerOperatorSpec:
    return TransformerOperatorSpec(
        op_type=op_type,
        input_shapes={"input": (1, 4096), "weight": (4096, 4096)},
        output_shapes={"output": (1, 4096)},
        precision=Precision.INT8,
        target_board=_make_board(),
        clock_mhz=200,
    )


_SAMPLE_CODE = """\
extern "C" {
void kernel_gemm(int8_t* input, int8_t* weight, int32_t* output) {
    #pragma HLS LOOP_TRIPCOUNT min=4096 max=4096
    for (int i = 0; i < 4096; i++) {
        int32_t acc = 0;
        for (int j = 0; j < 4096; j++) {
            acc += input[j] * weight[j * 4096 + i];
        }
        output[i] = acc;
    }
}
}
"""

_GOOD_SYNTH = HLSSynthesisResult(
    success=True,
    estimated_latency_cycles=1000,
    clock_period_ns=5.0,
    bram_used=10,
    bram_available=100,
    dsp_used=50,
    dsp_available=200,
    lut_used=5000,
    lut_available=50000,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransformerOperatorSpec:
    def test_gemm_spec(self) -> None:
        op = _make_op("gemm")
        assert op.op_type == "gemm"
        assert op.precision == Precision.INT8
        assert op.clock_mhz == 200
        assert op.input_shapes["input"] == (1, 4096)

    def test_softmax_spec(self) -> None:
        op = TransformerOperatorSpec(
            op_type="softmax",
            input_shapes={"input": (32, 128)},
            output_shapes={"output": (32, 128)},
            precision=Precision.FP16,
            target_board=_make_board(),
        )
        assert op.op_type == "softmax"


class TestOptimizationIteration:
    def test_iteration_record(self) -> None:
        it = OptimizationIteration(
            iteration=0,
            source_code="code",
            synthesis_result=_GOOD_SYNTH,
            judge_feedback="PASS",
            action_taken="accept",
        )
        assert it.iteration == 0
        assert it.action_taken == "accept"


class TestKernelOptimizationResult:
    def test_result_fields(self) -> None:
        op = _make_op()
        r = KernelOptimizationResult(
            operator=op,
            final_source="code",
            final_synthesis=_GOOD_SYNTH,
            iterations=[],
            total_iterations=1,
            converged=True,
            estimated_latency_cycles=1000,
            resource_utilization={"bram": 0.1, "dsp": 0.25},
        )
        assert r.converged is True
        assert r.estimated_latency_cycles == 1000
        assert r.resource_utilization["dsp"] == 0.25


class TestKernelOptimizerConvergence:
    """Test: optimizer converges when judge says PASS on first evaluation."""

    def test_converges_on_simple_kernel(self) -> None:
        mock_llm = MockLLMClient({
            "Translator": f"```cpp\n{_SAMPLE_CODE}\n```",
            "Judge": "PASS",
        })
        # Rename keys to match system prompt content
        mock_llm.responses = {
            "translate a high-level operator": f"```cpp\n{_SAMPLE_CODE}\n```",
        }
        mock_hls = MockHLSRunner([_GOOD_SYNTH])

        optimizer = KernelOptimizer(llm=mock_llm, hls=mock_hls, max_iterations=5)
        result = optimizer.optimize(_make_op())

        assert result.converged is True
        assert result.total_iterations == 1
        assert result.estimated_latency_cycles == 1000
        assert len(result.iterations) == 1
        assert result.iterations[0].action_taken == "accept"


class TestKernelOptimizerCompileRetry:
    """Test: optimizer retries on compile error then succeeds."""

    def test_retries_on_compile_error(self) -> None:
        fail_synth = HLSSynthesisResult(
            success=False,
            error_message="error: unknown type name 'int8_t'",
        )
        mock_llm = MockLLMClient({
            "translate a high-level operator": f"```cpp\n{_SAMPLE_CODE}\n```",
            "compilation": f"```cpp\n{_SAMPLE_CODE}\n```",
        })
        # First synth fails, compile fix synth succeeds, then judge PASS
        mock_hls = MockHLSRunner([fail_synth, _GOOD_SYNTH])

        optimizer = KernelOptimizer(llm=mock_llm, hls=mock_hls, max_iterations=5)
        result = optimizer.optimize(_make_op())

        assert result.converged is True
        assert result.total_iterations == 1
        # LLM was called: once for translate, once for compile fix, once for judge
        assert len(mock_llm.call_log) >= 2


class TestKernelOptimizerMaxIterations:
    """Test: optimizer respects max_iterations when judge never says PASS."""

    def test_hits_max_iterations(self) -> None:
        mock_llm = MockLLMClient({
            "translate a high-level operator": f"```cpp\n{_SAMPLE_CODE}\n```",
            "evaluation expert": "OPTIMIZE: pipeline the inner loop",
            "optimization expert": f"```cpp\n{_SAMPLE_CODE}\n```",
        })
        mock_hls = MockHLSRunner([])  # Default: always succeeds

        optimizer = KernelOptimizer(llm=mock_llm, hls=mock_hls, max_iterations=3)
        result = optimizer.optimize(_make_op())

        assert result.converged is False
        assert result.total_iterations == 3
        assert len(result.iterations) == 3
        for it in result.iterations:
            assert it.action_taken == "optimize"


class TestKernelOptimizerRevertToBest:
    """Test: optimizer reverts to best valid code on validation failure."""

    def test_reverts_on_validation_failure(self) -> None:
        fail_synth = HLSSynthesisResult(
            success=False,
            error_message="synthesis error",
        )
        # Sequence: succeed, succeed (judge optimizes), fail+fail+fail+fail (retries exhausted),
        # revert to best, succeed, PASS
        mock_hls = MockHLSRunner([
            _GOOD_SYNTH,                    # iter 0: initial validate passes
            _GOOD_SYNTH,                    # iter 0: this is the initial valid result
            fail_synth, fail_synth, fail_synth, fail_synth,  # iter 1: optimized code fails all retries
            _GOOD_SYNTH,                    # iter 2: reverted code validates
        ])
        call_count = [0]

        class SequenceLLM(LLMClient):
            """LLM that returns specific responses based on call count."""

            def generate(self, system_prompt: str, user_prompt: str) -> str:
                call_count[0] += 1
                if "translate a high-level operator" in system_prompt:
                    return f"```cpp\n{_SAMPLE_CODE}\n```"
                if "compilation" in system_prompt:
                    return f"```cpp\n{_SAMPLE_CODE}\n```"
                if "evaluation expert" in system_prompt:
                    # First judge: optimize, second judge: pass
                    if call_count[0] <= 3:
                        return "OPTIMIZE: unroll loop"
                    return "PASS"
                if "optimization expert" in system_prompt:
                    return "```cpp\nbad code\n```"
                return "PASS"

        optimizer = KernelOptimizer(
            llm=SequenceLLM(), hls=mock_hls, max_iterations=10
        )
        result = optimizer.optimize(_make_op())

        # Should have reverted and eventually converged
        assert result.final_synthesis is not None
        assert result.final_synthesis.success is True


class TestTheoreticalMinimum:
    """Test: theoretical minimum computation for different operator types."""

    def test_gemm_theoretical_min(self) -> None:
        op = _make_op("gemm")
        optimizer = KernelOptimizer(
            llm=MockLLMClient({}),
            hls=MockHLSRunner([]),
        )
        # GEMM (1, 4096) x (4096, 4096): 2*1*4096*4096 = 33554432 ops
        # VCK190: 1968 DSPs, INT8: 2 ops/DSP/cycle
        # 33554432 / (1968 * 2) = 8527
        result = optimizer._compute_theoretical_min(op)
        assert result == 33554432 // (1968 * 2)

    def test_softmax_theoretical_min(self) -> None:
        op = TransformerOperatorSpec(
            op_type="softmax",
            input_shapes={"input": (32, 128)},
            output_shapes={"output": (32, 128)},
            precision=Precision.FP16,
            target_board=_make_board(),
        )
        optimizer = KernelOptimizer(
            llm=MockLLMClient({}),
            hls=MockHLSRunner([]),
        )
        # 32*128 = 4096 elements, 5 ops each, / 1968 DSPs
        result = optimizer._compute_theoretical_min(op)
        assert result == (5 * 4096) // 1968


class TestPromptTemplates:
    """Test: all prompt templates can be formatted without KeyError."""

    def test_translator_template(self) -> None:
        result = TRANSLATOR_USER_TEMPLATE.format(
            op_type="gemm",
            input_shapes={"input": (1, 4096)},
            output_shapes={"output": (1, 4096)},
            precision="int8",
            target_device="Xilinx VCK190",
            clock_mhz=200,
            top_function="kernel_gemm",
        )
        assert "gemm" in result
        assert "kernel_gemm" in result

    def test_judge_template(self) -> None:
        result = JUDGE_USER_TEMPLATE.format(
            source_code="void kernel() {}",
            latency_cycles=1000,
            clock_period_ns=5.0,
            clock_mhz=200,
            bram_used=10,
            bram_available=100,
            bram_pct=10.0,
            dsp_used=50,
            dsp_available=200,
            dsp_pct=25.0,
            lut_used=5000,
            lut_available=50000,
            lut_pct=10.0,
            theoretical_min=500,
        )
        assert "1000" in result
        assert "500" in result

    def test_optimizer_template(self) -> None:
        result = OPTIMIZER_USER_TEMPLATE.format(
            feedback="OPTIMIZE: pipeline inner loop",
            latency_cycles=1000,
            source_code="void kernel() {}",
        )
        assert "pipeline inner loop" in result

    def test_compile_fixer_template(self) -> None:
        result = COMPILE_FIXER_USER_TEMPLATE.format(
            source_code="void kernel() {}",
            error_message="error: unknown type",
        )
        assert "unknown type" in result


class TestExtractCodeBlock:
    """Test: code extraction from LLM responses."""

    def test_extract_cpp_block(self) -> None:
        response = "Here is the code:\n```cpp\nint main() {}\n```\nDone."
        result = KernelOptimizer._extract_code_block(response)
        assert result == "int main() {}"

    def test_extract_plain_block(self) -> None:
        response = "```\nint main() {}\n```"
        result = KernelOptimizer._extract_code_block(response)
        assert result == "int main() {}"

    def test_no_block_returns_full_response(self) -> None:
        response = "int main() {}"
        result = KernelOptimizer._extract_code_block(response)
        assert result == "int main() {}"
