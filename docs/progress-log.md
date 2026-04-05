# Model2HW Progress Log

Engineering progress log for the Model2HW agentic co-design system. Captures design decisions, implementation details, experimental results, and open questions suitable for inclusion in a research paper.

---

## Session 1: 2026-04-05

### Research Phase

**Objective**: Determine feasibility of an agentic system for optimizing LLMs on constrained hardware (FPGAs, edge devices).

**Sources reviewed**:

- LAAFD (arxiv 2602.06085v1) - LLM-Aided Accelerator Design Framework. Key insight: 4-agent iterative loop (Translator, Validator, Judge, Optimizer) achieves near-expert HLS quality on GEMM/convolution kernels.
- ComPilot (arxiv 2511.00592) - Compilation framework using LLM for hardware compilation optimization.
- HiPEAC Chen interview - Industrial perspective on neural architecture search for hardware targets.
- Existing Model2HW codebase analysis - 13 modules, 132 tests, zero dependencies, static feasibility pipeline.

**Key finding**: The vast majority of the optimization pipeline (precision selection, sensitivity analysis, configuration recommendation, decomposition planning) is deterministic and should NOT use an LLM. The LLM is justified only for HLS C++ code generation and iterative optimization against synthesis reports.

### Design Decisions

**Decision 1: Deterministic-first architecture**

- Rationale: LLM calls add ~2-5s latency, cost per invocation, and hallucination risk. Precision sweeps, sensitivity analysis, and recommendations are bounded enumeration problems solvable in milliseconds with exact formulas.
- Boundary: Phases 1-5 (deterministic, zero deps, zero LLM). Phases 7-8 (LLM for HLS code gen only). Phase 6 (infrastructure, no LLM). Phases 9-10 (deterministic orchestration, optional LLM passthrough).
- Impact: Core functionality works offline, is fully reproducible, and costs nothing per run.

**Decision 2: Four-tier evaluation strategy**

- Rationale: User cannot purchase an FPGA. Need validation path from analytical models to real silicon without hardware ownership.
- Tiers: (1) Model2HW analytical - free, ms, exact memory/conservative BW. (2) Vitis HLS C-sim + synthesis - free, minutes, 5-15% latency error. (3) RTL simulation - free, hours, cycle-exact. (4) Cloud FPGA - ~$1.65/hr, real-time, ground truth.
- Impact: Development can proceed through Tier 2 with zero hardware cost. Tier 4 validates final designs on AWS F1 instances.

**Decision 3: dataclasses.replace for precision sweep**

- Rationale: ModelSpec contains only immutable fields. Shallow copy with replaced precision fields is safe and avoids constructor boilerplate.
- Constraint: Must not add mutable fields (lists, dicts) to ModelSpec without updating this pattern.

### Implementation: Phase 1 - Precision Sweep Engine

**Files created**:

- `hardware_feasibility/analysis/sweep.py` (141 lines)
- `tests/test_sweep.py` (97 lines)

**Files modified**:

- `hardware_feasibility/cli.py` - Added `--sweep` flag and sweep handling block
- `hardware_feasibility/outputs/report.py` - Added `format_sweep_report()`
- `hardware_feasibility/outputs/json_export.py` - Added `format_sweep_json()`
- `docs/technical-design-document.md` - Added Evaluation Tiers section

**Implementation details**:

- `run_precision_sweep()` enumerates all (weight, kv) precision pairs from WEIGHT_PRECISIONS x KV_PRECISIONS
- Pairs where kv_precision.bytes_per_element > weight_precision.bytes_per_element are skipped (no engineering reason to store KV at higher precision than weights)
- Full analysis pipeline runs for each pair: analyze_memory, analyze_bandwidth, analyze_compute, analyze_io, render_verdict
- Results sorted: fitting points first (by bandwidth headroom descending), then non-fitting (by memory overshoot ascending)
- CLI integration uses lazy import to avoid loading sweep module when not needed

**Combination space**:

- Default: 4 weight precisions x 3 KV precisions = 12 pairs
- After filtering (kv <= weight): 9 valid combinations
- Runtime: ~20ms for full sweep (9 analysis passes)

### Experimental Results

**Llama 3-8B on Xilinx VCK190 (8 GB, 25.6 GB/s)**:

| # | Weight | KV | Memory (GB) | BW (GB/s) | Verdict | Memory Headroom |
|---|--------|----|-------------|-----------|---------|-----------------|
| 1 | int4 | int4 | 3.87 | 37.6 | fits (BW-limited) | +4.13 GB |
| 2 | int8 | int4 | 7.61 | 75.0 | fits (BW-limited) | +0.39 GB |
| 3 | int8 | int8 | 7.74 | 75.2 | fits (BW-limited) | +0.26 GB |
| 4 | fp16 | int4 | 15.10 | 149.8 | does not fit | -7.10 GB |
| 5-9 | ... | ... | ... | ... | does not fit | ... |

**Key observations**:

1. Only 3 of 9 configurations fit the VCK190's 8 GB memory.
2. INT4/INT4 is the clear winner with 4.13 GB headroom (52% memory utilization).
3. INT8/INT4 barely fits with 0.39 GB headroom (95% utilization) - risky for production.
4. All fitting configurations are bandwidth-limited, not memory-limited. The VCK190's 25.6 GB/s DDR4 bandwidth is the binding constraint.
5. Even INT4/INT4 requires 37.6 GB/s (47% overshoot vs available bandwidth), limiting decode throughput to ~6.8 tok/s vs the 10 tok/s target.

**Implication for next phase**: Sensitivity analysis (Phase 2) should quantify how much additional bandwidth would be needed to reach the 10 tok/s target, and whether reducing context length can help.

### Test Results

```
tests/test_sweep.py - 7/7 PASSED (0.02s)
  test_sweep_llama3_8b_on_vck190        PASSED
  test_sweep_skips_kv_higher_than_weight PASSED
  test_sweep_returns_sorted              PASSED
  test_sweep_with_no_target              PASSED
  test_sweep_custom_precisions           PASSED
  test_sweep_report_format               PASSED
  test_sweep_json_format                 PASSED

Full suite: 132 existing + 7 new = 139 passing (7 pre-existing failures unrelated to sweep)
```

### Documentation Artifacts Created

| Artifact | Path | Purpose |
|----------|------|---------|
| Feasibility study | `docs/agentic-codesign-feasibility-study.md` | Research grounding, literature review |
| Technical design document | `docs/technical-design-document.md` | Complete 10-phase engineering spec |
| Implementation checklist | `docs/implementation-checklist.md` | Running status of all 85 features |
| Progress log | `docs/progress-log.md` | This document - paper-ready notes |

### Open Questions for Next Session

1. **Bandwidth wall**: All fitting configurations on VCK190 are bandwidth-limited. Phase 2 should quantify exactly how much bandwidth improvement is needed for 10 tok/s at INT4.
2. **HBM boards**: The Alveo U55C and Versal VE2802 have HBM2 with 460+ GB/s. Sweep results on those boards will look very different. Should we run comparative sweeps across board classes?
3. **KV offloading tradeoff**: For the INT8/INT8 case that barely fits (0.26 GB headroom), offloading KV to host could free ~0.12 GB but adds host link traffic. Worth exploring in Phase 2.
4. **Accuracy cost**: INT4 quantization on Llama 3-8B typically degrades perplexity by 3-8%. Phase 9 will add this data, but early users should be aware of the tradeoff.

---

## Session 2: 2025-07-15

### Objective

Implement all remaining deterministic phases (2-5) and the HLS synthesis infrastructure (Phase 6) to reach hardware simulation readiness.

### Implementation: Phase 2 - Sensitivity Analyzer

**Files created**:

- `hardware_feasibility/analysis/sensitivity.py` (190 lines)
- `tests/test_sensitivity.py` (115 lines)

**Files modified**:

- `hardware_feasibility/cli.py` - Added `--sensitivity` flag
- `hardware_feasibility/outputs/report.py` - Added `format_sensitivity_report()`
- `hardware_feasibility/outputs/json_export.py` - Added `format_sensitivity_json()`

**Design decisions**:

- **Bottleneck classification**: BottleneckBreakdown computes utilization ratios for memory, bandwidth, IO, and compute. The `primary_bottleneck` property returns whichever dimension has the highest utilization ratio vs target.
- **Perturbation strategy**: Hardware parameters (memory, bandwidth targets) are cheap to perturb - just re-run verdict. Model parameters (context_length, batch_size, weight_precision) require re-running the full analysis pipeline with `dataclasses.replace`.
- **Perturbation set**: Memory target +25/50/-25%, bandwidth target +25/50/-25%, context_length 0.5x/2x/4x, batch_size 2x/4x, weight_precision one step down.

**Test insight**: Initial test assumed FP16 llama3-8b on VCK190 was memory-bottlenecked, but bandwidth actually dominates more (5.8x oversubscribed vs 1.9x memory oversubscribed). Fixed test to use a 4 GB / 2000 GB/s hypothetical board where memory truly dominates.

### Implementation: Phase 3 - Configuration Recommender

**Files created**:

- `hardware_feasibility/analysis/recommender.py` (206 lines)
- `tests/test_recommender.py` (95 lines)

**Design decisions**:

- **Algorithm**: (1) Run precision sweep to find fitting combos. (2) For each, sweep context [512..32768] x batch [1..8] x kv_placement [on/off accelerator]. (3) Rank all feasible configs by estimated tok/s. (4) Generate deterministic rationale strings.
- **Performance estimation**: `_estimate_tok_per_sec()` uses the bandwidth-limited model: board_bw_bytes / total_bytes_per_token. This is conservative but accurate for memory-bandwidth-bound LLM inference.
- **Rationale generation**: `_build_rationale()` produces template-based explanations (e.g., "INT4 quantization reduces weight memory from 16.0 GB to 4.0 GB"). No LLM involvement.

**Test insight**: Test for infeasible model initially used "llama2-70b" which does not exist in KNOWN_FAMILIES (only goes up to llama2-13b). Fixed to use llama2-13b on ZCU104 (2 GB) where 13B params do not fit even at INT4.

### Implementation: Phase 4 - Decomposition Planner

**Files created**:

- `hardware_feasibility/analysis/decomposition.py` (245 lines)
- `tests/test_decomposition.py` (110 lines)

**Design decisions**:

- **Pipeline parallel**: Splits model layers across N identical boards (N=2..8). Per-layer memory is computed by dividing total weight memory proportionally. Embedding/LM-head weights are assigned to the first device.
- **KV offload**: Checks if weights+activations fit on device with KV cache on host. Performance limited by min(device bandwidth, host link bandwidth / kv_bytes_per_token).
- **Pipeline bubble model**: bubble_fraction = (N-1) / (N-1 + decode_length). At decode_length=256, a 2-way split has only 0.4% bubble overhead.
- **Uniform layer assumption**: Valid for standard decoder-only transformers (all Llama variants, Mistral, Phi, Gemma, Qwen). Not valid for MoE models. Planner notes this assumption.

**Experimental result** - Llama 3-8B FP16 decomposition on ZCU104 (2 GB):

- KV offload: not viable (weights alone ~16 GB > 2 GB)
- Pipeline parallel: needs 8+ devices at FP16, each getting ~4 layers
- At INT4: 2-way split is comfortable, each device holds ~2 GB of the ~4 GB model

### Implementation: Phase 5 - Extended Hardware Model

**Files modified**:

- `hardware_feasibility/hardware/board_specs.py` - Added 10 new optional fields to BoardSpec, populated all FPGA boards

**New BoardSpec fields**:

| Field | Type | Purpose |
|-------|------|---------|
| `bram_kb` | `int \| None` | Block RAM capacity (KB) |
| `uram_kb` | `int \| None` | UltraRAM capacity (KB, Xilinx only) |
| `dsp_slices` | `int \| None` | DSP slice count |
| `lut_count` | `int \| None` | LUT count (thousands) |
| `memory_type` | `str` | "ddr4", "hbm2", "hbm2e", "lpddr5", "ddr6x" |
| `memory_channels` | `int` | Number of memory channels |
| `has_ai_engines` | `bool` | Versal AI Engines present |
| `ai_engine_count` | `int \| None` | Number of AI Engines |

**Board data populated** (from vendor datasheets):

| Board | BRAM (KB) | URAM (KB) | DSPs | LUTs (K) | Memory Type |
|-------|-----------|-----------|------|----------|-------------|
| ZCU104 | 312 | 96 | 1,728 | 504 | DDR4 |
| VCK190 | 967 | 0 | 1,968 | 899 | DDR4 |
| Alveo U250 | 2,688 | 1,280 | 12,288 | 1,727 | DDR4 |
| Alveo U55C | 2,016 | 960 | 9,024 | 1,303 | HBM2 |
| VE2802 | 1,344 | 576 | 3,984 | 1,218 | HBM2e |
| Agilex 7 | 7,404 | 0 | 4,510 | 949 | HBM2e |
| Stratix 10 NX | 10,960 | 0 | 5,760 | 1,866 | HBM2 |

All new fields default to None/False/0/"ddr4", so GPU, NPU, and Edge SoC boards are backward compatible.

The `on_chip_memory_kb` derived property returns BRAM + URAM, used by the kernel optimizer in Phase 7.

### Implementation: Phase 6 - HLS Synthesis Feedback Integration

**Files created**:

- `hardware_feasibility/synthesis/__init__.py`
- `hardware_feasibility/synthesis/types.py` (82 lines) - KernelSpec, HLSSynthesisResult, HLSCoSimResult
- `hardware_feasibility/synthesis/hls_runner.py` (210 lines) - HLSRunner class, Tcl script generation
- `hardware_feasibility/synthesis/report_parser.py` (130 lines) - XML report parser
- `tests/fixtures/sample_csynth.xml` - Sample Vitis HLS report
- `tests/test_hls_synthesis.py` (130 lines)

**Design decisions**:

- **Tcl script generation**: Templates use `set_part {device}`, `create_clock -period N`, `csynth_design`. Curly braces around device name ensure Tcl doesn't interpret hyphens.
- **Report parser**: Tries multiple XPath patterns per field to handle Vitis HLS 2022.x vs 2024.x XML format differences. Falls back gracefully when fields are missing.
- **Security**: Kernel source written to temp directory, cleaned up after synthesis. Configurable timeout (default 10 min). No elevation or network access needed.
- **Dependency policy**: Vitis HLS is NOT a Python dependency. HLSRunner verifies installation at construction time and raises RuntimeError with instructions if missing.

**Utilization properties on HLSSynthesisResult**:

- `bram_utilization`, `dsp_utilization`, `ff_utilization`, `lut_utilization` - return None when data unavailable
- `meets_timing` - compares achieved clock period against target

### Test Results

```
tests/test_sensitivity.py     - 7/7 PASSED
tests/test_recommender.py     - 6/6 PASSED
tests/test_decomposition.py   - 7/7 PASSED
tests/test_extended_hardware.py - 8/8 PASSED
tests/test_hls_synthesis.py   - 9/9 PASSED

New tests this session: 37
Full suite: 169 passing, 7 pre-existing failures (tests.conftest import)
```

### Architecture After Session 2

```
hardware_feasibility/
  analysis/
    sweep.py          [Phase 1] Precision sweep
    sensitivity.py    [Phase 2] Bottleneck analysis
    recommender.py    [Phase 3] Configuration recommendation
    decomposition.py  [Phase 4] Multi-device planning
    memory.py         [Phase 0] Memory analysis
    bandwidth.py      [Phase 0] Bandwidth analysis
    compute.py        [Phase 0] Compute analysis
    io.py             [Phase 0] IO analysis
    verdict.py        [Phase 0] Feasibility verdict
  hardware/
    board_specs.py    [Phase 0+5] Board DB with extended fields
    matcher.py        [Phase 0] Board ranking
  models/
    architecture_rules.py  [Phase 0] ModelSpec, families
    hf_config_loader.py    [Phase 0] Config loading
  synthesis/
    types.py          [Phase 6] HLS data models
    hls_runner.py     [Phase 6] Vitis HLS invocation
    report_parser.py  [Phase 6] Synthesis report parsing
  outputs/
    report.py         Text reports for all phases
    json_export.py    JSON export for all phases
  cli.py              CLI with --sweep/--sensitivity/--recommend/--decompose
```

### What is Now Possible

With Phases 0-6 complete, the deterministic analysis layer is fully operational:

1. **Full static analysis**: Memory, bandwidth, compute, IO profiling for any model-board pair
2. **Precision sweep**: Enumerate all valid precision combos, rank by feasibility
3. **Bottleneck diagnosis**: Identify whether memory, bandwidth, IO, or compute limits performance
4. **Configuration optimization**: Find the best operating point across precision, context, batch, and KV placement
5. **Multi-device planning**: Pipeline parallel and KV offload strategies for models that exceed single-device capacity
6. **HLS infrastructure**: Ready to invoke Vitis HLS synthesis and parse results when a Vitis HLS installation is available

### Open Questions for Next Session

1. **Phase 7 LLM dependency**: The kernel optimizer requires an LLM client. Need to decide on: OpenAI API vs local models, prompt templating strategy, and iteration budget.
2. ~~**Pre-existing test failures**~~: Resolved in Session 3, see below.
3. ~~**Report formatter dead code**~~: Resolved in Session 2, dead code removed.
4. **Cloud FPGA validation**: With Phase 6 HLS runner complete, Tier 2 evaluation is ready. Need a Vitis HLS installation to test end-to-end synthesis flow.

---

## Session 3: 2026-04-05

### Objective

Fix all pre-existing test failures to reach 100% pass rate before proceeding to Phase 7 (agentic kernel optimizer).

### Root Cause Analysis

All 7 failures shared the same root cause: tests used `from tests.conftest import _make_spec` as a lazy import inside test methods. This fails because `tests/` is not a Python package and pytest's conftest auto-loading mechanism does not make its contents importable via the standard `from tests.conftest` path.

**Affected files** (7 test methods across 5 files):

| File | Test Method |
|------|-------------|
| `test_bandwidth.py` | `test_scales_linearly_with_target_tok_s` |
| `test_compute.py` | `test_scales_with_prefill_length` |
| `test_io.py` | `test_required_bw_scales_with_tok_s` |
| `test_memory.py` | `test_kv_scales_with_batch_size` |
| `test_memory.py` | `test_kv_scales_with_context_length` |
| `test_memory.py` | `test_prefill_dominated` |
| `test_verdict.py` | `test_tight_memory_reports_warning` |

### Fix Applied

Replaced `from tests.conftest import _make_spec` with `from hardware_feasibility.models.hf_config_loader import load_from_known_family` in all 7 test methods. The `load_from_known_family` function is the public API that does the same thing as `_make_spec`: builds a `ModelSpec` from a known family name with optional overrides.

The `_make_spec` helper in `conftest.py` remains available for fixture-based tests that use it via pytest's automatic conftest loading (e.g., `llama3_8b_fp16` fixture). No changes to `conftest.py` were needed.

### Design Decision

**Decision: Use public API in tests, not conftest helpers for inline construction**

- Rationale: `load_from_known_family` is the canonical way to build a ModelSpec from a family name. Tests that construct specs inline (not via fixtures) should use this function rather than internal conftest helpers. This avoids import path issues and keeps tests self-documenting.
- Pattern: Fixture-based tests continue to use `conftest.py` fixtures (`llama3_8b_fp16`, `llama3_8b_int4`). Tests needing custom parameters use `load_from_known_family` directly.

### Test Results

```
Full suite: 176 passed, 0 failed (0.92s)

Previously failing, now passing:
  test_bandwidth.py::test_scales_linearly_with_target_tok_s     PASSED
  test_compute.py::test_scales_with_prefill_length              PASSED
  test_io.py::test_required_bw_scales_with_tok_s                PASSED
  test_memory.py::test_kv_scales_with_batch_size                PASSED
  test_memory.py::test_kv_scales_with_context_length            PASSED
  test_memory.py::test_prefill_dominated                        PASSED
  test_verdict.py::test_tight_memory_reports_warning            PASSED
```

### Observations for Paper

The `_make_spec` vs `load_from_known_family` distinction illustrates a common testing anti-pattern in Python: using conftest helpers as importable modules rather than as pytest-discovered fixtures. The fix demonstrates that a well-designed public API (`load_from_known_family`) can serve both application code and test code without needing separate test-only helpers for the same functionality.

### Open Questions for Next Session

1. **Phase 7 LLM dependency**: The kernel optimizer requires an LLM client. Need to decide on: OpenAI API vs local models, prompt templating strategy, and iteration budget.
2. **Cloud FPGA validation**: With Phase 6 HLS runner complete, Tier 2 evaluation is ready. Need a Vitis HLS installation to test end-to-end synthesis flow.

---

## Session 4 - 2026-04-06

### Summary

Implemented all remaining phases (7-10), completing the full 10-phase technical design. All 85/85 features are now DONE and 222/222 tests pass.

### Phase 7: Agentic Kernel Optimizer (17 new tests)

Created `hardware_feasibility/agents/` package with four files:

- **types.py** - `TransformerOperatorSpec`, `OptimizationIteration`, `KernelOptimizationResult` dataclasses
- **llm_client.py** - `LLMClient` ABC with `OpenAIClient` implementation (lazy `openai` import)
- **prompts.py** - Four prompt templates: Translator, Compile Fixer, Judge, Optimizer
- **kernel_optimizer.py** - `KernelOptimizer.optimize()` implementing the full LAAFD loop

Key design decisions:
- **Dependency injection**: `KernelOptimizer` takes `LLMClient` and `HLSRunner` in constructor, enabling full mock-based testing
- **Code extraction**: Regex-based C++ code block extraction from LLM markdown responses
- **Theoretical minimum**: Per-operator formulas (GEMM: 2MNK/DSPs, attention, softmax, MLP)
- **Device string resolution**: Static map from board names to Vitis HLS part numbers

Fixed during testing: `JUDGE_USER_TEMPLATE` uses `{dsp_pct}` and `{lut_pct}` placeholders that were missing from `_judge()` format call.

### Phase 8: Multi-Kernel Orchestration (9 new tests)

Created `hardware_feasibility/agents/orchestrator.py` with:

- **decompose_model_to_operators()** - Decomposes a transformer model into individual operators: 13 per layer (Q/K/V projections, attention QKT, softmax, score*V, O projection, RMSNorm x2, gate/up/down projections, SiLU) + embedding + LM head
- **deduplicate_operators()** - Groups by (op_type, shapes, precision) tuple; Llama3-8B: 418 operators -> ~12 unique
- **check_resource_budget()** - Sums BRAM/DSP/LUT/FF across kernels, compares to board limits (static bitstream assumption)

Experimental results:
- Qwen2-0.5B: 314 operators (24 layers x 13 + 2), deduplicates to 9-13 unique kernels
- Llama3-8B: 418 operators (32 layers x 13 + 2), deduplicates to 10-14 unique kernels

### Phase 9: Accuracy-in-the-Loop (12 new tests)

Created `hardware_feasibility/evaluation/` package:

- **accuracy_db.py** - 24-entry `KNOWN_PERPLEXITY` lookup table (wikitext-2 values for 8 model families x 3 precisions), `estimate_perplexity_degradation()` with multiplier-based estimation, `get_perplexity()` with source annotation ("lookup"/"estimated")

Integration with existing recommender:
- Added `estimated_perplexity` and `perplexity_source` fields to `Recommendation` dataclass
- `recommend_configuration()` now auto-injects perplexity for each candidate
- Report formatter shows perplexity line; JSON export includes perplexity fields

### Phase 10: End-to-End Pipeline (8 new tests)

Created `hardware_feasibility/pipeline.py`:

- **PipelineResult** dataclass combining all stage outputs
- **run_full_pipeline()** - Runs stages 1-3 deterministically:
  - Stage 1: Feasibility check
  - Stage 1b: Precision sweep (only if initial does not fit)
  - Stage 1c: Decomposition (only if no precision fits)
  - Stage 2: Configuration recommendation (always)
  - Stage 3: Sensitivity analysis (always)

CLI integration:
- Added `--full-pipeline` flag (requires `--board`)
- `format_pipeline_report()` and `format_pipeline_json()` formatters

### pyproject.toml Update

Added optional dependency groups:
- `[agents]` = `["openai>=1.0"]`
- `[all]` = `["huggingface_hub>=0.20", "openai>=1.0"]`

### Test Results

```text
222 passed in 0.53s

Phase 0 (foundation):         132 tests
Phase 1 (sweep):                7 tests
Phase 2 (sensitivity):          7 tests
Phase 3 (recommender):          6 tests
Phase 4 (decomposition):        7 tests
Phase 5 (extended hardware):    8 tests
Phase 6 (HLS synthesis):        9 tests
Phase 7 (kernel optimizer):    17 tests
Phase 8 (orchestrator):         9 tests
Phase 9 (accuracy):            12 tests
Phase 10 (pipeline):            8 tests
```

### Files Created

- `hardware_feasibility/agents/__init__.py`
- `hardware_feasibility/agents/types.py`
- `hardware_feasibility/agents/llm_client.py`
- `hardware_feasibility/agents/prompts.py`
- `hardware_feasibility/agents/kernel_optimizer.py`
- `hardware_feasibility/agents/orchestrator.py`
- `hardware_feasibility/evaluation/__init__.py`
- `hardware_feasibility/evaluation/accuracy_db.py`
- `hardware_feasibility/pipeline.py`
- `tests/test_kernel_optimizer.py`
- `tests/test_orchestrator.py`
- `tests/test_accuracy_db.py`
- `tests/test_pipeline.py`

### Files Modified

- `hardware_feasibility/analysis/recommender.py` - Added perplexity fields + import
- `hardware_feasibility/outputs/report.py` - Added perplexity line in recommendations, `format_pipeline_report()`
- `hardware_feasibility/outputs/json_export.py` - Added perplexity fields, `format_pipeline_json()`
- `hardware_feasibility/cli.py` - Added `--full-pipeline` flag and handler
- `pyproject.toml` - Added `[agents]` and `[all]` optional dependency groups
- `docs/implementation-checklist.md` - 85/85 features DONE (100%)
- `docs/progress-log.md` - Session 4 entry

### Observations for Paper

1. **Mock-based agent testing**: The LAAFD optimization loop is fully testable without LLM API calls or Vitis HLS. MockLLMClient (prompt-substring matching) and MockHLSRunner (predefined result sequence) enable deterministic verification of convergence, retry, and revert logic.

2. **Operator deduplication efficiency**: Transformer models have high structural regularity. Despite 400+ operators per model, deduplication reduces the optimization target to ~10-14 unique kernels, making the agentic optimization loop tractable.

3. **Perplexity integration pattern**: Adding accuracy awareness to the recommender required only two new dataclass fields and a lookup function call at the recommendation construction site - minimal coupling between the evaluation and analysis packages.

4. **Pipeline composition**: The end-to-end pipeline naturally composes phases because each phase produces self-contained result objects. No shared mutable state between stages.
