# Model2HW Implementation Checklist

Master tracking document for all implementation phases. Updated after every coding session.

**Last updated**: 2026-04-06\
**Current phase**: Phase 10 (complete) - All phases implemented\
**Test suite**: 222/222 passing (0 failures)

---

## Status Legend

- DONE - Feature implemented, tested, and verified
- IN PROGRESS - Actively working on
- TODO - Not yet started
- BLOCKED - Waiting on a dependency

---

## Phase 0: Foundation (Pre-existing)

The static feasibility analyzer that all subsequent phases build on.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 0.1 | ModelSpec dataclass and Precision enum | DONE | `models/architecture_rules.py` | `test_architecture_rules.py` | 17 fields, 5 precision levels |
| 0.2 | Known model families database | DONE | `models/architecture_rules.py` | `test_architecture_rules.py` | 12 families (Llama, Mistral, Phi, Gemma, Qwen) |
| 0.3 | HuggingFace config loader | DONE | `models/hf_config_loader.py` | `test_hf_config_loader.py` | Local JSON, Hub API, known family |
| 0.4 | Memory analysis | DONE | `analysis/memory.py` | `test_memory.py` | Weights, KV cache, activation buffers |
| 0.5 | Bandwidth analysis | DONE | `analysis/bandwidth.py` | `test_bandwidth.py` | Decode-phase memory bandwidth demand |
| 0.6 | Compute analysis | DONE | `analysis/compute.py` | `test_compute.py` | FLOPs per token, prefill FLOPs, roofline |
| 0.7 | IO analysis | DONE | `analysis/io.py` | `test_io.py` | Host-device link traffic |
| 0.8 | Verdict engine | DONE | `analysis/verdict.py` | `test_verdict.py` | Memory/BW/IO checks, 4 verdict types |
| 0.9 | Board specs database | DONE | `hardware/board_specs.py` | `test_hardware.py` | 15+ boards: FPGA, GPU, Edge SoC, NPU |
| 0.10 | Hardware matcher and ranker | DONE | `hardware/matcher.py` | `test_hardware.py` | Match single board, rank all boards |
| 0.11 | Human-readable report | DONE | `outputs/report.py` | via `test_cli.py` | Full-text feasibility report |
| 0.12 | JSON export | DONE | `outputs/json_export.py` | via `test_cli.py` | Machine-readable output |
| 0.13 | CLI entry point | DONE | `cli.py` | `test_cli.py` | argparse, mutually exclusive model source |

**Phase 0 summary**: 132 tests passing. Zero external dependencies for core functionality.

---

## Phase 1: Precision Sweep Engine

Enumerate all valid (weight_precision, kv_precision) combinations and rank by feasibility.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 1.1 | SweepPoint dataclass | DONE | `analysis/sweep.py` | `test_sweep.py` | fits, memory_headroom_gb, bandwidth_headroom_gbps properties |
| 1.2 | SweepResult dataclass | DONE | `analysis/sweep.py` | `test_sweep.py` | fitting_points, best_fitting properties |
| 1.3 | run_precision_sweep() | DONE | `analysis/sweep.py` | `test_sweep.py` | Enumerates combos, skips kv > weight, sorts fitting-first |
| 1.4 | CLI --sweep flag | DONE | `cli.py` | `test_sweep.py` | Lazy import, early return before normal output |
| 1.5 | format_sweep_report() | DONE | `outputs/report.py` | `test_sweep.py::test_sweep_report_format` | Tabular text output |
| 1.6 | format_sweep_json() | DONE | `outputs/json_export.py` | `test_sweep.py::test_sweep_json_format` | Dict with points array |
| 1.7 | Test: llama3-8b on VCK190 | DONE | `tests/test_sweep.py` | 7/7 passing | INT4 fits, FP16 does not |

**Phase 1 summary**: 7 new tests, all passing. 132 existing tests unaffected. `--sweep` works end-to-end for both text and JSON output.

---

## Phase 2: Sensitivity Analyzer

Identify which bottleneck dominates and quantify how sensitive the verdict is to parameter changes.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 2.1 | BottleneckBreakdown dataclass | DONE | `analysis/sensitivity.py` | `test_sensitivity.py` | memory, bandwidth, io, compute utilization |
| 2.2 | SensitivityPoint dataclass | DONE | `analysis/sensitivity.py` | `test_sensitivity.py` | Parameter perturbation results |
| 2.3 | SensitivityResult dataclass | DONE | `analysis/sensitivity.py` | `test_sensitivity.py` | Breakdown + list of perturbations |
| 2.4 | analyze_sensitivity() | DONE | `analysis/sensitivity.py` | `test_sensitivity.py` | Bottleneck ID + what-if perturbations |
| 2.5 | CLI --sensitivity flag | DONE | `cli.py` | `test_sensitivity.py` | Requires --board or explicit targets |
| 2.6 | Sensitivity report formatter | DONE | `outputs/report.py` | `test_sensitivity.py` | Resource utilization + what-if table |
| 2.7 | Sensitivity JSON export | DONE | `outputs/json_export.py` | `test_sensitivity.py` | Structured perturbation data |

**Phase 2 summary**: 7 new tests, all passing. Perturbs memory/bandwidth targets and model params.

---

## Phase 3: Configuration Recommender

Automatically recommend the best operating configuration for a model-board pair.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 3.1 | Recommendation dataclass | DONE | `analysis/recommender.py` | `test_recommender.py` | Precision, context, batch, kv placement |
| 3.2 | RecommendationResult dataclass | DONE | `analysis/recommender.py` | `test_recommender.py` | Sorted list + infeasible_reason |
| 3.3 | Structured sweep algorithm | DONE | `analysis/recommender.py` | `test_recommender.py` | Precision x context x batch x kv_placement |
| 3.4 | Deterministic rationale generation | DONE | `analysis/recommender.py` | `test_recommender.py` | Template-based, no LLM |
| 3.5 | CLI --recommend flag | DONE | `cli.py` | `test_recommender.py` | Requires --board |
| 3.6 | Recommendation report formatter | DONE | `outputs/report.py` | `test_recommender.py` | Top-5 configs with rationale |
| 3.7 | Recommendation JSON export | DONE | `outputs/json_export.py` | `test_recommender.py` | Full recommendation data |

**Phase 3 summary**: 6 new tests, all passing. Sweeps precision x context x batch x kv_placement, ranks by tok/s.

---

## Phase 4: Decomposition Planner

Propose multi-device execution strategies for models that do not fit on a single device.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 4.1 | DeviceAssignment dataclass | DONE | `analysis/decomposition.py` | `test_decomposition.py` | Layer range, memory per device |
| 4.2 | DecompositionPlan dataclass | DONE | `analysis/decomposition.py` | `test_decomposition.py` | Strategy, devices, comms, performance |
| 4.3 | DecompositionResult dataclass | DONE | `analysis/decomposition.py` | `test_decomposition.py` | All plans + best plan |
| 4.4 | Pipeline parallel planner | DONE | `analysis/decomposition.py` | `test_decomposition.py` | Layer splitting across N identical boards |
| 4.5 | KV offload planner | DONE | `analysis/decomposition.py` | `test_decomposition.py` | Weights on device, KV on host |
| 4.6 | Communication overhead model | DONE | `analysis/decomposition.py` | `test_decomposition.py` | Inter-device transfer + bubble fraction |
| 4.7 | CLI --decompose flag | DONE | `cli.py` | `test_decomposition.py` | Requires --board |

**Phase 4 summary**: 7 new tests, all passing. Pipeline parallel (2-8 devices) and KV offload strategies.

---

## Phase 5: Extended Hardware Model

Add FPGA-specific fields to BoardSpec for kernel optimization phases.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 5.1 | Add BRAM/URAM/DSP/LUT fields | DONE | `hardware/board_specs.py` | `test_extended_hardware.py` | Optional fields with None defaults |
| 5.2 | Add memory_type and channels | DONE | `hardware/board_specs.py` | `test_extended_hardware.py` | DDR4, HBM2, HBM2e, LPDDR5, DDR6x |
| 5.3 | Add AI engine fields | DONE | `hardware/board_specs.py` | `test_extended_hardware.py` | Versal AI Engines |
| 5.4 | on_chip_memory_kb property | DONE | `hardware/board_specs.py` | `test_extended_hardware.py` | BRAM + URAM sum |
| 5.5 | Populate Xilinx board data | DONE | `hardware/board_specs.py` | `test_extended_hardware.py` | ZCU104, VCK190, Alveo U250, U55C, VE2802 |
| 5.6 | Populate Intel board data | DONE | `hardware/board_specs.py` | `test_extended_hardware.py` | Agilex 7, Stratix 10 NX |

**Phase 5 summary**: 8 new tests, all passing. GPU/NPU boards have None for FPGA-specific fields. Backward compatible.

---

## Phase 6: HLS Synthesis Feedback Integration

Infrastructure to invoke Vitis HLS programmatically and parse synthesis reports.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 6.1 | HLSSynthesisResult dataclass | DONE | `synthesis/types.py` | `test_hls_synthesis.py` | Timing, resources, utilization properties |
| 6.2 | HLSCoSimResult dataclass | DONE | `synthesis/types.py` | `test_hls_synthesis.py` | Pass/fail + error output + runtime |
| 6.3 | KernelSpec dataclass | DONE | `synthesis/types.py` | `test_hls_synthesis.py` | Source, testbench, target, clock |
| 6.4 | HLSRunner class | DONE | `synthesis/hls_runner.py` | `test_hls_synthesis.py` | Tcl script generation, subprocess invocation |
| 6.5 | Synthesis report XML parser | DONE | `synthesis/report_parser.py` | `test_hls_synthesis.py` | Vitis 2022 and 2024 format support |
| 6.6 | Sandboxing and timeout | DONE | `synthesis/hls_runner.py` | `test_hls_synthesis.py` | Configurable timeout, temp dir cleanup |
| 6.7 | Sample XML fixture | DONE | `tests/fixtures/sample_csynth.xml` | `test_hls_synthesis.py` | VCK190 matmul_int8 report structure |

**Phase 6 summary**: 9 new tests, all passing. No new Python deps (stdlib only: subprocess, xml, tempfile, pathlib).

---

## Phase 7: Agentic Kernel Optimizer (Requires LLM)

LLM-driven iterative HLS kernel optimization following the LAAFD workflow.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 7.1 | TransformerOperatorSpec dataclass | DONE | `agents/types.py` | `test_kernel_optimizer.py` | Op type, shapes, precision, target |
| 7.2 | OptimizationIteration dataclass | DONE | `agents/types.py` | `test_kernel_optimizer.py` | Per-iteration record |
| 7.3 | KernelOptimizationResult dataclass | DONE | `agents/types.py` | `test_kernel_optimizer.py` | Final source, synthesis, convergence |
| 7.4 | LLMClient abstract base class | DONE | `agents/llm_client.py` | `test_kernel_optimizer.py` | Dependency injection interface |
| 7.5 | OpenAIClient implementation | DONE | `agents/llm_client.py` | `test_kernel_optimizer.py` | Optional openai dependency |
| 7.6 | Prompt templates (4 agents) | DONE | `agents/prompts.py` | `test_kernel_optimizer.py` | Translator, Compile Fixer, Judge, Optimizer |
| 7.7 | KernelOptimizer.optimize() | DONE | `agents/kernel_optimizer.py` | `test_kernel_optimizer.py` | Full LAAFD loop: translate, validate, judge, optimize |
| 7.8 | Compile-error retry logic | DONE | `agents/kernel_optimizer.py` | `test_kernel_optimizer.py` | Up to 3 retries with LLM fix |
| 7.9 | Revert-to-best logic | DONE | `agents/kernel_optimizer.py` | `test_kernel_optimizer.py` | Keep best valid code across iterations |
| 7.10 | Theoretical minimum computation | DONE | `agents/kernel_optimizer.py` | `test_kernel_optimizer.py` | GEMM, attention, softmax, MLP bounds |

**Phase 7 summary**: 17 new tests, all passing. Mock LLM + mock HLS runner for full loop testing. Optional `openai` dep via `[agents]` extra.

---

## Phase 8: Multi-Kernel Orchestration (Requires LLM)

Optimize all kernels in an LLM inference pipeline with shared resource constraints.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 8.1 | decompose_model_to_operators() | DONE | `agents/orchestrator.py` | `test_orchestrator.py` | 13 ops/layer + embed + LM head |
| 8.2 | deduplicate_operators() | DONE | `agents/orchestrator.py` | `test_orchestrator.py` | ~418 -> ~10 unique kernels for Llama3-8B |
| 8.3 | check_resource_budget() | DONE | `agents/orchestrator.py` | `test_orchestrator.py` | Sum BRAM/DSP/LUT/FF, compare to board limits |
| 8.4 | Full pipeline orchestration | DONE | `agents/orchestrator.py` | `test_orchestrator.py` | Decompose + dedup + budget check |
| 8.5 | Resource-constrained re-optimization | DONE | `agents/orchestrator.py` | `test_orchestrator.py` | Budget fit/exceed detection |

**Phase 8 summary**: 9 new tests, all passing. Static bitstream assumption (all kernels coexist).

---

## Phase 9: Accuracy-in-the-Loop

Integrate accuracy evaluation into sweep and recommendation pipelines.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 9.1 | KNOWN_PERPLEXITY lookup table | DONE | `evaluation/accuracy_db.py` | `test_accuracy_db.py` | 24 entries across 8 model families |
| 9.2 | estimate_perplexity_degradation() | DONE | `evaluation/accuracy_db.py` | `test_accuracy_db.py` | Multiplier-based from FP16 baseline |
| 9.3 | Perplexity in Recommendation | DONE | `analysis/recommender.py` | `test_accuracy_db.py` | estimated_perplexity + perplexity_source fields |
| 9.4 | Perplexity in report/JSON output | DONE | `outputs/report.py`, `outputs/json_export.py` | `test_accuracy_db.py` | Perplexity line in reports, fields in JSON |

**Phase 9 summary**: 12 new tests, all passing. Perplexity data auto-injected into recommendations.

---

## Phase 10: End-to-End Pipeline

Single command that runs all phases in sequence.

| # | Feature | Status | Files | Tests | Notes |
|---|---------|--------|-------|-------|-------|
| 10.1 | PipelineResult dataclass | DONE | `pipeline.py` | `test_pipeline.py` | All stage results combined |
| 10.2 | run_full_pipeline() | DONE | `pipeline.py` | `test_pipeline.py` | Stages 1-3 deterministic, Stage 4 optional |
| 10.3 | CLI --full-pipeline flag | DONE | `cli.py` | `test_pipeline.py` | Single entry point, requires --board |
| 10.4 | Pipeline report formatter | DONE | `outputs/report.py` | `test_pipeline.py` | Multi-section text + JSON output |
| 10.5 | Graceful skip of HLS/LLM stages | DONE | `pipeline.py` | `test_pipeline.py` | Stages 1-3 always run, stage 4 skipped without LLM/HLS |

**Phase 10 summary**: 8 new tests, all passing. `--full-pipeline` runs feasibility + sweep + decomposition + recommendation + sensitivity.

---

## Cross-Cutting Concerns

| # | Feature | Status | Files | Notes |
|---|---------|--------|-------|-------|
| CC.1 | Evaluation tiers documented | DONE | `docs/technical-design-document.md` | 4-tier validation strategy |
| CC.2 | Design principle documented | DONE | `docs/technical-design-document.md` | LLM only where absolutely necessary |
| CC.3 | Feasibility study | DONE | `docs/agentic-codesign-feasibility-study.md` | Research grounding |
| CC.4 | Technical design document | DONE | `docs/technical-design-document.md` | Full 10-phase specification |
| CC.5 | Implementation checklist | DONE | `docs/implementation-checklist.md` | This document |
| CC.6 | Progress log (paper artifact) | DONE | `docs/progress-log.md` | Engineering decisions and results |
| CC.7 | pyproject.toml optional deps | DONE | `pyproject.toml` | Added [agents], [all] groups for openai |

---

## Summary

| Phase | Features | Done | TODO | Progress |
|-------|----------|------|------|----------|
| 0 - Foundation | 13 | 13 | 0 | 100% |
| 1 - Precision Sweep | 7 | 7 | 0 | 100% |
| 2 - Sensitivity Analyzer | 7 | 7 | 0 | 100% |
| 3 - Configuration Recommender | 7 | 7 | 0 | 100% |
| 4 - Decomposition Planner | 7 | 7 | 0 | 100% |
| 5 - Extended Hardware Model | 6 | 6 | 0 | 100% |
| 6 - HLS Synthesis Feedback | 7 | 7 | 0 | 100% |
| 7 - Agentic Kernel Optimizer | 10 | 10 | 0 | 100% |
| 8 - Multi-Kernel Orchestration | 5 | 5 | 0 | 100% |
| 9 - Accuracy-in-the-Loop | 4 | 4 | 0 | 100% |
| 10 - End-to-End Pipeline | 5 | 5 | 0 | 100% |
| Cross-Cutting | 7 | 7 | 0 | 100% |
| **Total** | **85** | **85** | **0** | **100%** |
