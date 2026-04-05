# Agentic Co-Design for LLM-on-Constrained-Hardware: Feasibility Study

> "Can we build an agentic compiler for AI systems that maps models to constrained hardware?"

This document presents a critical, technically grounded analysis of the feasibility, architecture, and limitations of an agentic system for deploying large language models (1B - 70B parameters) on constrained hardware such as FPGAs, edge SoCs, and low-memory accelerators.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Feasibility of Agentic Optimization](#2-feasibility-of-agentic-optimization)
3. [Reference Architecture](#3-reference-architecture-for-an-agentic-co-design-loop)
4. [Key Bottlenecks in Scaling LLMs to Small Hardware](#4-key-bottlenecks-in-scaling-llms-to-small-hardware)
5. [Decomposition Strategies for Large Models](#5-decomposition-strategies-for-large-models)
6. [Agentic Optimization vs Traditional Compilers](#6-agentic-optimization-vs-traditional-compilers)
7. [Design Space Exploration](#7-design-space-exploration)
8. [Limits of Automation](#8-limits-of-automation)
9. [Taxonomy of Optimization Tasks](#9-taxonomy-of-optimization-tasks)
10. [Gap Analysis](#10-gap-analysis)
11. [Roadmap](#11-roadmap)
12. [Conclusions](#12-conclusions)
13. [References](#13-references)

---

## 1. Executive Summary

**Verdict: Conditionally feasible, with hard boundaries.**

An agentic system can meaningfully automate a subset of the LLM-to-hardware mapping problem, but it cannot replace the full stack. The core insight is that the problem decomposes into layers with very different automation profiles, and the majority of them do not require an LLM at all.

**Design principle: use the LLM only where absolutely necessary.** Everything that can be expressed as a formula, bounded enumeration, or threshold comparison stays deterministic. The LLM enters only when the system needs to generate or modify HLS source code and interpret synthesis tool output.

| Layer | Automatable? | Method | Requires LLM? |
|---|---|---|---|
| Model analysis (architecture extraction) | Fully | Deterministic JSON parsing | No |
| Memory/bandwidth feasibility screening | Fully | Rule-based arithmetic (Model2HW) | No |
| Quantization scheme selection | Fully | Bounded enum sweep over precisions | No |
| KV placement decision | Fully | Memory budget comparison (binary) | No |
| Layer-split / decomposition planning | Fully | 1D search over layer boundaries | No |
| Sensitivity analysis | Fully | Parameter perturbation + re-evaluation | No |
| Configuration recommendation | Fully | Structured sweep + Pareto ranking | No |
| HLS kernel generation | Partially | LLM-guided code generation with synthesis feedback | **Yes** |
| HLS pragma / kernel optimization | Partially | LLM-guided iterative refinement with HLS reports | **Yes** |
| Operator fusion proposals | Partially | LLM reasoning over dataflow + correctness verification | **Yes** |
| Synthesis failure diagnosis | Partially | LLM interpretation of HLS error reports | **Yes** |
| Hardware timing closure | Minimally | Requires physical design tools; no LLM-amenable abstraction | No (not automatable) |
| Full system integration | Minimally | Too many interacting constraints | No (human-driven) |

The strongest evidence for feasibility comes from LAAFD (LLM-based Agents for Accelerated FPGA Design), which achieves 99.9% geomean performance vs. hand-tuned HLS baselines on 15 HPC kernels using a closed-loop agentic workflow with Vitis HLS synthesis feedback [1]. ComPilot demonstrates that off-the-shelf LLMs, without fine-tuning, can achieve 2.66x - 3.54x speedups over unoptimized code through iterative compiler feedback loops [4]. However, both systems operate on individual kernels, not complete LLM inference pipelines. Scaling from "optimize one kernel" to "map an entire 8B-parameter model to a VCK190 FPGA" is an order-of-magnitude harder problem.

The existing Model2HW codebase already implements the deterministic foundation: architecture extraction, memory profiling, bandwidth analysis, compute estimation, IO analysis, and hardware matching. This is the right starting point. The question is how far an agentic layer on top of this can push.

---

## 2. Feasibility of Agentic Optimization

### 2.1 Existing Evidence

Three lines of work provide grounded evidence:

**LAAFD (2026)** [1]: A multi-agent workflow (translator, validator, judge, optimizer) transforms C++ into optimized Vitis HLS kernels. Key findings:
- Achieves near-theoretical latency (99.9% of hand-tuned baselines) using GPT-5
- The feedback loop (HLS synthesis reports providing cycle counts and resource utilization) is critical for convergence
- Smaller models (GPT-5-nano, o4-mini) achieve materially worse results (32.7% and 52.5% of optimal), demonstrating that agent capability is bottlenecked by model reasoning depth
- Complex kernels (stencils with data reuse) require 10+ runs with 25 optimization iterations each; only 1-2 out of 10 runs produce optimal results
- Scalability is limited by LLM context windows when processing large HLS reports

**ComPilot (2025)** [4]: An agentic auto-scheduling framework where an LLM proposes loop transformations and a compiler validates them. Key findings:
- 2.66x single-run, 3.54x best-of-5 speedups over unoptimized code on PolyBench
- Competitive with Pluto polyhedral optimizer (a state-of-the-art static optimizer)
- Zero-shot capability: no task-specific fine-tuning required
- The compiler feedback loop (legality + measured speedup) is what makes it work

**StreamTensor / A3C3 methodology** [5]: Inspirit IoT's compiler maps PyTorch LLM models directly to FPGA dataflow accelerators, achieving 0.64x lower latency and 1.99x higher energy efficiency vs. GPUs. This is not agentic per se, but demonstrates that automated end-to-end LLM-to-FPGA compilation is achievable with the right abstractions (iterative tensors, stream-oriented dataflow).

### 2.2 What Agents Can Realistically Do

**Learn memory constraints**: Yes, trivially. Memory constraints are codifiable as arithmetic over architecture parameters. Model2HW already does this. An agent's value here is in exploring quantization and layout alternatives that reduce memory, not in computing constraints.

**Optimize bandwidth usage**: Partially. An agent can propose tiling strategies, weight streaming schedules, and KV cache management policies, then validate them against bandwidth models. The Model2HW bandwidth analysis (weight bytes + KV read per decode token, multiplied by target tok/s) provides the exact feedback signal an agent needs. But optimal bandwidth utilization requires operator fusion and memory layout decisions that interact with hardware-specific memory hierarchies in ways that are hard to simulate without actual synthesis.

**Generate efficient execution schedules**: Yes, with caveats. ComPilot shows LLMs can propose competitive loop schedules when grounded by compiler feedback. However, LLM inference schedules are more complex than PolyBench loops. They involve:
- Interleaved prefill and decode phases with different compute/memory profiles
- Dynamic KV cache growth
- Speculative decoding branches
- Multi-head attention with varying group sizes (GQA vs. MHA)

These are structured enough for search but complex enough that an agent is unlikely to find globally optimal solutions without explicit domain knowledge injection.

### 2.3 Where Agents Fail

- **Stochastic convergence**: LAAFD requires multiple runs to find optimal kernels for complex patterns. For a full LLM pipeline with dozens of interacting kernels, the probability of all converging simultaneously is vanishingly small.
- **Context window limits**: HLS reports for complete LLM accelerator designs would vastly exceed current LLM context limits.
- **Physical design awareness**: Agents have no model of timing closure, routing congestion, or thermal constraints. These are determined by place-and-route tools that take hours to run.
- **Correctness verification at scale**: LAAFD uses C++ co-simulation per kernel. End-to-end correctness of a full LLM inference pipeline on hardware requires functional verification at a qualitatively different scale.

---

## 3. Reference Architecture for an Agentic Co-Design Loop

```
+------------------------------------------------------------------+
|                    AGENTIC CO-DESIGN SYSTEM                      |
+------------------------------------------------------------------+
|                                                                  |
|  +--------------+    +------------------+    +-----------------+ |
|  | Model        |    | Hardware         |    | Constraint      | |
|  | Analyzer     |--->| Constraint Model |--->| Synthesizer     | |
|  | (InferSpec)  |    | (BoardSpec DB)   |    | (Verdict)       | |
|  +--------------+    +------------------+    +-----------------+ |
|        |                     |                       |           |
|        v                     v                       v           |
|  +-------------------------------------------------------------+|
|  |              AGENT ORCHESTRATOR                              ||
|  |  +----------+  +-----------+  +----------+  +------------+  ||
|  |  | Quant    |  | Schedule  |  | Kernel   |  | Decomp     |  ||
|  |  | Explorer |  | Planner   |  | Generator|  | Planner    |  ||
|  |  +----------+  +-----------+  +----------+  +------------+  ||
|  +-------------------------------------------------------------+|
|        |                     |                       |           |
|        v                     v                       v           |
|  +-------------------------------------------------------------+|
|  |              FEEDBACK LOOP                                   ||
|  |  +----------+  +-----------+  +----------+  +------------+  ||
|  |  | HLS      |  | Cycle-    |  | Memory   |  | Functional |  ||
|  |  | Synthesis |  | Accurate  |  | Layout   |  | Validation |  ||
|  |  | Reports  |  | Simulator |  | Analyzer |  | (Co-sim)   |  ||
|  |  +----------+  +-----------+  +----------+  +------------+  ||
|  +-------------------------------------------------------------+|
|        |                                                         |
|        v                                                         |
|  +--------------+    +------------------+                        |
|  | Evaluator    |    | Decision         |                        |
|  | (Score:      |--->| (accept/reject/  |                        |
|  | latency,     |    |  iterate/stop)   |                        |
|  | memory,power)|    +------------------+                        |
|  +--------------+                                                |
+------------------------------------------------------------------+
```

### 3.1 Component Specifications

#### Model Analyzer (already implemented)

Extracts from HuggingFace config or known-family templates:
- Layer count, hidden size, attention heads, KV heads, intermediate size, vocab size
- Head dimension (derived: `hidden_size / num_attention_heads`)
- GQA ratio (`num_kv_heads < num_attention_heads`)

The existing `ModelSpec` dataclass is sufficient. Extension needed: attention pattern metadata (sliding window, cross-attention), activation function type, normalization placement.

#### Hardware Constraint Model (already implemented)

The `BoardSpec` database covers FPGAs (Xilinx ZCU104 through Versal VE2802), Intel FPGAs (Agilex 7, Stratix 10 NX), and edge SoCs (Jetson Orin NX). Key parameters: memory capacity, memory bandwidth, peak INT8 TOPS, host link bandwidth, TDP.

Extension needed: memory hierarchy detail (on-chip SRAM/BRAM capacity, HBM vs. DDR), DSP/LUT/BRAM counts, reconfigurability constraints, multi-die topology.

#### Execution Planner (new - agentic)

Proposes execution strategies for each transformer layer:
- **Tiling**: how to decompose matrix multiplications to fit on-chip memory
- **Scheduling**: operator fusion order, prefill vs. decode specialization
- **Memory layout**: weight storage format, KV cache placement (on-chip vs. off-chip)
- **Pipeline mapping**: which layers execute on which compute units; double-buffering

This is where agentic iteration provides the most value. The planner proposes a strategy, the feedback loop evaluates it, and the planner refines.

#### Kernel Generator (new - agentic)

Generates HLS C++ (for FPGAs) or optimized C/CUDA (for GPUs/NPUs) implementing the execution plan. LAAFD demonstrates this is achievable per-kernel with GPT-5-class models using synthesis feedback. The challenge is scaling to dozens of interconnected kernels.

#### Simulator + Feedback Loop (partially new)

The critical ingredient. Without concrete feedback, agents hallucinate. Required signals:
- **HLS synthesis reports**: cycle count, resource utilization, timing slack (from Vitis HLS / Intel Quartus)
- **Cycle-accurate simulation**: RTL co-simulation for functional correctness
- **Memory bandwidth simulation**: actual vs. theoretical bandwidth utilization
- **End-to-end latency estimation**: aggregated across all layers

Model2HW already provides analytical estimates (memory, bandwidth, compute, IO). These are fast but approximate. For agentic refinement, the loop needs actual synthesis/simulation data.

#### Evaluator

Multi-objective scoring:
- Latency (tok/s at target context length)
- Memory utilization (% of available memory consumed)
- Power efficiency (tok/s/W, estimated from resource utilization and TDP)
- Feasibility verdict (the existing `FeasibilityVerdict` enum: fits_comfortably, fits_but_bandwidth_limited, does_not_fit, host_link_likely_bottleneck)

### 3.2 Integration with Model2HW

The existing codebase provides the analytical backbone:

```
Model2HW (current)          Agentic Extension (proposed)
---------------------        ----------------------------
architecture_rules.py   -->  Input to agent orchestrator
memory.py              -->  Constraint for planner
bandwidth.py           -->  Constraint for planner
compute.py             -->  Constraint for planner
io.py                  -->  Constraint for planner
verdict.py             -->  Evaluator input
board_specs.py         -->  Hardware target definition
matcher.py             -->  Initial hardware screening
```

The agentic layer sits between the analytical models and the synthesis tools. Model2HW screens feasibility; the agent optimizes within feasible configurations.

---

## 4. Key Bottlenecks in Scaling LLMs to Small Hardware

### 4.1 Quantitative Analysis

Using Model2HW's analysis framework, here are concrete numbers for representative model-hardware pairs:

| Model | Precision | Weight Memory | KV Cache (2K ctx) | Total Working Set | VCK190 (8 GB) | Alveo U55C (16 GB HBM) |
|---|---|---|---|---|---|---|
| Llama 3.2-1B | INT4 | 0.76 GB | 0.13 GB | ~1.0 GB | FITS | FITS |
| Llama 3-8B | INT4 | 4.0 GB | 0.25 GB | ~4.5 GB | FITS (tight) | FITS |
| Llama 3-8B | INT8 | 8.0 GB | 0.50 GB | ~9.0 GB | DOES NOT FIT | FITS (tight) |
| Llama 2-13B | INT4 | 6.5 GB | 0.62 GB | ~7.5 GB | FITS (tight) | FITS |
| Llama 2-70B | INT4 | 35.0 GB | 1.25 GB | ~37 GB | DOES NOT FIT | DOES NOT FIT |

The memory wall is absolute: no amount of agentic optimization can fit a 70B INT4 model (35 GB weights) into 8 GB of on-board memory without multi-device decomposition. This is the first hard boundary.

### 4.2 Bottleneck Classification

#### Memory Capacity (codifiable, hard constraint)

- **Formula**: `params * bytes_per_weight + KV_cache + activation_buffers`
- **Status**: Fully codifiable. Model2HW implements this exactly.
- **Optimization approach**: Deterministic sweep over all `Precision` enum values (INT4, INT8, FP16, BF16), re-running `analyze_memory()` for each. This is a bounded enumeration, not a search problem. No LLM needed.

#### Memory Bandwidth (codifiable, soft constraint)

- **Formula**: `(weight_bytes + kv_read_bytes) * target_tok_s`
- **Example**: Llama 3-8B INT4 on VCK190: ~4.0 GB weights + ~0.12 GB KV read = 4.12 GB/token, times 10 tok/s = **41.2 GB/s required** vs. 25.6 GB/s available. **Bandwidth-limited to ~6.2 tok/s.**
- **Optimization approach (deterministic)**: The bandwidth computation itself is arithmetic. Sensitivity analysis (perturbing bandwidth targets by +25%, +50%) identifies how much headroom exists. This is a parameter sweep, no LLM needed.
- **Optimization approach (agentic, v2+)**: Weight tiling strategies that maximize on-chip BRAM/URAM reuse to reduce effective bandwidth demand. This requires generating HLS code with specific pragma combinations and measuring actual bandwidth utilization via synthesis. The LLM is needed here because the tiling code must be generated and iteratively refined against HLS feedback.

#### KV Cache Scaling (codifiable, exponential in context)

- **Per-token KV bytes**: `2 * num_layers * num_kv_heads * head_dim * bytes_per_kv`
- **Llama 3-8B INT8**: 2 * 32 * 8 * 128 * 1 = 65,536 bytes/token = 64 KB/token
- **At 128K context**: 64 KB * 128K = **8 GB** for KV cache alone
- **Optimization approach**: KV precision is part of the deterministic precision sweep (e.g., FP16 vs. INT8 vs. INT4 KV). The impact on memory is computed directly by `estimate_kv_cache()`. Context length sweep (512, 1024, 2048, ...) is also deterministic. No LLM needed for any of this.

#### Host-Device Communication (codifiable, architectural constraint)

- **When KV cache is on host**: Each decode step streams the full KV cache across PCIe. For Llama 3-8B at 2K context: ~0.13 GB per token over PCIe. At 10 tok/s on PCIe Gen4 x8 (16 GB/s): 1.3 GB/s - feasible but consuming 8% of link bandwidth.
- **At longer contexts, this becomes the bottleneck.** Model2HW's IO analysis already models this (`kv_on_accelerator` flag).
- **Optimization approach**: Binary decision (KV on device vs. host) based on comparing `kv_cache_gb` against remaining device memory after weights. Pure arithmetic. No LLM needed.

#### Execution Scheduling (requires search)

- Tiling dimensions, pipeline depth, operator fusion decisions, double-buffering strategies
- Interactions between these choices create a combinatorial design space
- **Optimization approach (deterministic, v1)**: Pipeline depth is a 1D search over layer boundaries with per-device memory as the constraint. This is a bounded enumeration, no LLM needed.
- **Optimization approach (agentic, v2+)**: Tiling dimensions, operator fusion, and double-buffering require generating HLS code and measuring synthesis results. This is where the LLM earns its cost, because there is no closed-form solution and the search requires creative code transformations grounded by compiler feedback.

---

## 5. Decomposition Strategies for Large Models

For models that exceed single-device capacity, decomposition is mandatory.

### 5.1 Layer-Level Splitting (Pipeline Parallelism)

Split transformer layers across devices. Device 1 runs layers 0-15, Device 2 runs layers 16-31.

**Constraints**:
- Each device must hold its layers' weights + KV cache + activations
- Activation tensors (hidden_size * batch_size * precision) must be transferred between devices at each stage boundary
- Pipeline bubbles reduce utilization proportionally to `1/num_stages`

**Automation potential**: High. Fully deterministic. The split point search is a 1D optimization over layer boundaries, constrained by per-device memory. The algorithm enumerates candidates N = 2, 3, 4, ... and for each computes per-device memory and communication cost analytically. No LLM needed.

### 5.2 Attention/MLP Splitting (Tensor Parallelism)

Split individual layers across devices. Attention heads on Device 1, MLP on Device 2, or split heads across devices.

**Constraints**:
- Requires all-reduce or point-to-point communication within each layer
- Communication volume: `hidden_size * batch_size * precision` per layer, twice (forward + backward for attention output and MLP output)
- Only viable if inter-device bandwidth is high (PCIe Gen4+ or NVLink-class)

**Automation potential**: Moderate. For simple topologies (homogeneous devices), the search space is manageable with brute-force enumeration. For complex heterogeneous systems, analytical models become harder to formulate. Not fundamentally an LLM problem - this is a combinatorial optimization problem solvable with classical methods (ILP, greedy, DP).

### 5.3 KV Cache Offloading

Keep weights on accelerator, stream KV cache from host memory as context grows.

**Constraints**:
- Decode latency becomes a function of host link bandwidth
- Model2HW's IO analysis models this exactly (the `kv_on_accelerator` flag)
- Viable when KV cache exceeds device memory but weights fit

**Automation potential**: High. Fully deterministic. Binary decision based on comparing weight memory against device capacity and KV cache against remaining space. Uses existing `analyze_io()` with `kv_on_accelerator=False`. No LLM needed.

### 5.4 Dynamic Scheduling Across Heterogeneous Devices

Assign different phases (prefill vs. decode) or different requests to different devices.

**Constraints**:
- Prefill is compute-bound; decode is memory-bandwidth-bound
- A high-compute device (GPU) handles prefill; a high-bandwidth device (HBM FPGA) handles decode
- Requires low-latency state transfer between devices

**Automation potential**: Moderate. The analytical modeling of prefill vs. decode assignment is deterministic (compare FLOP requirements vs. compute capacity, bandwidth requirements vs. memory bandwidth). An agent is not needed here - but real deployment requires integration with serving infrastructure, which is an engineering problem beyond this system's scope.

---

## 6. Agentic Optimization vs Traditional Compilers

### 6.1 Comparison Matrix

| Dimension | Static Compiler (TVM-style) | Auto-Scheduler (Ansor-style) | Agentic System |
|---|---|---|---|
| **Search strategy** | Fixed heuristics | Cost-model-guided random search | LLM-guided with synthesis feedback |
| **Optimization scope** | Per-operator | Per-operator + fusion | Cross-operator, potentially cross-layer |
| **Feedback signal** | Analytical cost model | Measured latency on target | Synthesis reports (cycles, resources, timing) |
| **Generalization** | Broad but shallow | Good within trained distribution | Potentially broad via LLM world knowledge |
| **Determinism** | Deterministic | Stochastic (seeded) | Stochastic (temperature-dependent) |
| **Cost per optimization** | Milliseconds | Minutes-hours | Minutes-hours (HLS synthesis per iteration) |
| **Failure mode** | Consistent, predictable | Occasional bad schedules | Hallucinated optimizations, correctness bugs |
| **State of the art** | Production-grade | Research-grade, some production | Research-grade only |

### 6.2 Where Agents Outperform Static Systems

1. **Novel pattern recognition**: LLMs can identify optimization opportunities (e.g., window buffering, data reuse patterns) that are not encoded in a compiler's fixed heuristic set. LAAFD demonstrates this for stencil kernels [1].

2. **Cross-abstraction reasoning**: An agent can reason about relationships between algorithmic structure, memory layout, and hardware resources simultaneously. Static compilers operate within a single abstraction level.

3. **Context-dependent adaptation**: An agent can tailor its strategy to specific hardware constraints described in natural language (e.g., "this FPGA has 400 AI Engines with limited local memory per engine"). Compilers require explicit target backends.

4. **Rapid prototyping**: For new hardware targets without existing compiler backends, an agentic workflow can produce working (if suboptimal) implementations faster than developing a new compiler backend.

### 6.3 Where Agents Fail

1. **Timing closure**: Agents have no model of wire delay, routing congestion, or clock domain crossing. These are determined by physical design tools. No amount of LLM reasoning can predict whether a design will meet timing at 250 MHz on a specific FPGA die.

2. **Resource balancing**: LAAFD-generated kernels consistently use more BRAM than hand-tuned equivalents (e.g., 576 BRAM blocks for S3D vs. lower for manual implementations) [1]. Agents optimize for latency but struggle with multi-objective tradeoffs involving physical resources.

3. **Memory layout precision**: The difference between row-major and column-major storage, between 64-byte and 128-byte cache line alignment, between bank-conflict-free and bank-conflicting access patterns - these are determined by single-bit decisions that have outsized performance impact. LLMs lack the precision to consistently get these right.

4. **Correctness guarantees**: Agents produce probabilistically correct code. For safety-critical or high-reliability deployments, this is insufficient without formal verification, which agents cannot perform.

5. **Reproducibility**: LAAFD reports that only 1-2 out of 10 runs produce optimal results for complex kernels [1]. This stochastic convergence is unacceptable for production deployment pipelines.

---

## 7. Design Space Exploration

### 7.1 The Design Space for LLM-on-FPGA

The design space for mapping an LLM to constrained hardware includes:

| Dimension | Options | Cardinality |
|---|---|---|
| Weight precision | FP16, BF16, INT8, INT4, NF4, mixed | ~10 |
| KV cache precision | FP16, INT8, INT4, per-head mixed | ~8 |
| Tiling (per operator) | Tile sizes along M, N, K | ~100 per operator |
| Operator fusion | Which adjacent ops to fuse | 2^n for n operators |
| Memory placement | On-chip, off-chip, host, per tensor | 3^(num_tensors) |
| Pipeline depth | 1 to num_layers stages | ~num_layers |
| Batch scheduling | Static, dynamic, speculative | ~5 |
| Hardware config | Board selection from database | ~10-20 |

For Llama 3-8B (32 layers, ~100 operators per layer), the combinatorial design space is conservatively 10^50+. Exhaustive search is impossible.

### 7.2 Can Agents Navigate This Space?

**Evidence for**: ComPilot achieves competitive-with-Pluto results through iterative refinement, suggesting LLMs can navigate large optimization spaces when given concrete feedback [4]. LAAFD converges to near-optimal HLS for individual kernels [1].

**Evidence against**: Both systems operate on small programs (single kernels, PolyBench loops). The state space for a complete LLM inference pipeline is orders of magnitude larger. Agent convergence time scales poorly with problem complexity (LAAFD needs 10 runs * 25 iterations for complex kernels).

**Realistic assessment**: An agentic system should decompose the design space into independent subproblems:
1. Model-level decisions (precision, KV strategy) - searched first
2. Layer-level decisions (tiling, fusion) - searched per-layer with shared heuristics
3. System-level decisions (pipeline depth, device assignment) - searched last

This hierarchical decomposition reduces the effective search space to manageable size but may miss globally optimal configurations that require coordinated decisions across levels.

---

## 8. Limits of Automation

### 8.1 What Can Be Fully Automated (Deterministic, No LLM)

- **Model architecture extraction**: Parsing HuggingFace configs, estimating parameter counts, computing KV cache sizes. Already done in Model2HW.
- **Feasibility screening**: Memory/bandwidth/compute analysis against hardware specs. Already done in Model2HW.
- **Precision sweep**: Enumerating all (weight_precision, kv_precision) combinations and ranking by feasibility margin. Bounded loop over `Precision` enum values. No LLM needed.
- **KV placement decision**: Comparing KV cache size against remaining device memory. Binary arithmetic. No LLM needed.
- **Layer-split planning**: 1D search over layer boundaries constrained by per-device memory. No LLM needed.
- **Sensitivity analysis**: Perturbing parameters (+25% bandwidth, 2x context, etc.) and re-running verdicts. No LLM needed.
- **Configuration recommendation**: Structured sweep over (precision, context, batch, KV placement) ranked by estimated tok/s. No LLM needed.
- **Quantization selection**: Given a lookup table of known precision-accuracy tradeoffs, selecting the best precision is a table lookup. No LLM needed.
- **Simple kernel generation**: For standard operations (GEMM, softmax, layer norm), template-based generation with parameter specialization is reliable. No LLM needed.

### 8.2 What Requires an LLM Agent

The LLM agent is justified only when the task has **no closed-form solution** and **requires creative code generation or free-form search with external tool feedback**:

- **HLS kernel generation**: Translating a transformer operator spec into optimized Vitis HLS C++. No template covers every variant. The LLM generates code; HLS synthesis validates it.
- **HLS pragma optimization**: Combinatorial pragma space with no analytical cost model. The agent proposes pragma combinations; Vitis HLS reports back cycle counts and resource usage. LAAFD proves this works [1].
- **Operator fusion proposals (non-standard)**: For novel or non-standard architectures, deciding which adjacent ops to fuse requires reasoning about dataflow that goes beyond pattern matching.
- **Synthesis failure diagnosis**: When HLS reports timing violations or resource overflows, diagnosing the root cause and proposing a fix is a natural-language reasoning task over structured reports.

Everything else stays deterministic.

### 8.3 What Remains Human-Driven

- **Architecture-level HW decisions**: Choosing between FPGA fabric families, memory technologies, interconnect topologies. These require domain expertise and cost/business tradeoff analysis.
- **Physical design closure**: Place-and-route, timing closure, power grid design. These tools are not language-model-amenable.
- **Accuracy-hardware tradeoff decisions**: Deciding how much accuracy loss is acceptable for a given deployment scenario is a product/business decision, not an engineering optimization.
- **Novel accelerator architecture design**: Designing new datapath architectures (e.g., systolic arrays optimized for GQA attention) requires creative engineering that current LLMs cannot perform from scratch.
- **Safety and certification**: For safety-critical deployments, formal verification, testing, and certification are inherently human-supervised processes.

### 8.4 Where Current Agentic Systems Fail Hard

1. **Memory layout sensitivity**: A 1-bit difference in address alignment can cause 10x performance variation on FPGAs due to bank conflicts. LLMs lack the bit-level precision to consistently reason about this.

2. **Hardware timing constraints**: Meeting setup/hold time requirements at high clock frequencies depends on physical placement. No LLM-accessible feedback signal exists for this until after place-and-route completes (which takes hours).

3. **Lack of ground-truth feedback signals**: The most critical limitation. LAAFD works because Vitis HLS provides deterministic cycle counts and resource estimates. For many aspects of the LLM deployment problem (accuracy impact, real-world latency under load, thermal behavior), no fast, reliable feedback signal exists.

4. **Long feedback loops**: HLS synthesis for a large design takes minutes to hours. RTL simulation of a full LLM inference step could take days. This limits the number of iterations an agent can perform in practical time.

---

## 9. Taxonomy of Optimization Tasks

### 9.1 Deterministic (rule-based, no search needed, no LLM)

| Task | Input | Output | Implementation | Requires LLM? |
|---|---|---|---|---|
| Parameter count estimation | Architecture params | Integer | `estimate_param_count()` in Model2HW | No |
| Weight memory calculation | params, precision | Bytes | `estimate_weight_memory()` in Model2HW | No |
| KV cache sizing | layers, heads, head_dim, precision, context | Bytes | `estimate_kv_cache()` in Model2HW | No |
| Bandwidth requirement | weight + KV bytes, target tok/s | GB/s | `analyze_bandwidth()` in Model2HW | No |
| IO requirement | embedding/logit sizes, KV placement | GB/s | `analyze_io()` in Model2HW | No |
| Feasibility verdict | memory, BW, IO vs. hardware specs | Verdict enum | `render_verdict()` in Model2HW | No |
| Hardware matching/ranking | model requirements vs. board DB | Ranked list | `rank_boards()` in Model2HW | No |
| Precision sweep | model + board | Ranked configs | Nested loop over `Precision` enum (v1) | No |
| KV placement decision | weight memory + device capacity | Boolean | Memory arithmetic (v1) | No |
| Layer split planning | model + N devices | Layer assignments | 1D enumeration (v1) | No |
| Sensitivity analysis | model + board + perturbation deltas | Bottleneck report | Parameter perturbation + re-eval (v1) | No |
| Configuration recommendation | model + board | Top-K configs | Structured sweep + Pareto ranking (v1) | No |
| Pareto frontier | precision * context * batch * board | Dominated set | Grid search + domination filter (v1) | No |

### 9.2 Heuristic (good solutions from known patterns, no LLM)

| Task | Method | Quality guarantee | Requires LLM? |
|---|---|---|---|
| Quantization scheme selection | Lookup table of known precision-accuracy tradeoffs per model family | ~90% of optimal; validated by perplexity | No |
| Operator fusion (standard patterns) | Pattern matching (GEMM + bias + activation; QKV projection fusion) | Well-understood for standard architectures | No |
| Memory tiling for standard GEMMs | Tile size = sqrt(on-chip memory / 3) heuristic | Typically within 20% of optimal | No |
| Pipeline stage assignment | Balanced load across stages by parameter count | Near-optimal for uniform architectures | No |

### 9.3 Search-Based (require exploration with feedback, LLM needed only for code generation tasks)

| Task | Search space | Feedback signal | Requires LLM? | Rationale |
|---|---|---|---|---|
| HLS pragma selection | Pragma combinations per loop | HLS cycle count | **Yes** | No closed-form solution; requires generating and modifying C++ code |
| Custom kernel optimization | Code transformations | Synthesis reports | **Yes** | Requires creative code refactoring grounded by HLS feedback |
| Operator fusion (non-standard) | Fusion candidates | Correctness + latency | **Yes** | Requires reasoning about novel dataflow patterns |
| Synthesis failure diagnosis | Error report space | Next synthesis pass/fail | **Yes** | Natural-language interpretation of HLS error messages |
| Mixed-precision assignment | Per-layer precision | Accuracy + memory | No | Bounded enumeration; test each combo and evaluate perplexity |
| Multi-device partitioning | Layer-to-device assignment | Communication + compute balance | No | Analytical model sufficient; ILP or greedy search |

### 9.4 Non-Automatable (require human judgment or infeasible tools)

| Task | Why not automatable |
|---|---|
| Physical design closure | Requires iterative P&R with hours-long runtimes; no LLM-amenable abstraction |
| Novel accelerator architecture | Creative design; not a search problem |
| Accuracy-deployment tradeoff decisions | Business/product judgment |
| Safety certification | Regulatory and legal requirements |
| Hardware procurement decisions | Cost, supply chain, vendor relationships |

---

## 10. Gap Analysis

### 10.1 Missing Tooling

| Gap | Impact | Difficulty to fill |
|---|---|---|
| Fast, accurate LLM inference simulator for FPGAs | Cannot close the agentic loop without fast feedback | High - requires FPGA-specific architectural simulator |
| Automated accuracy evaluation pipeline | Cannot evaluate quantization tradeoffs in-loop | Medium - requires calibration dataset and perplexity eval |
| HLS template library for transformer operators | Agents must generate from scratch each time | Medium - LAAFD kernels could be adapted |
| Multi-device communication cost model | Cannot evaluate decomposition strategies | Medium - extend Model2HW IO analysis |
| Resource-aware HLS feedback | LAAFD optimizes for latency only; no resource budget | Low - already available in Vitis HLS reports, just not used |

### 10.2 Missing Data

| Gap | Impact |
|---|---|
| Accuracy vs. precision lookup tables per model family | Must evaluate accuracy empirically for each quantization choice |
| Real benchmark data for LLM inference on FPGAs | Analytical models (Model2HW) are unvalidated against real hardware |
| Performance profiles of existing FPGA-based LLM accelerators | No baseline to compare against |

### 10.3 Missing Feedback Loops

| Gap | Impact |
|---|---|
| End-to-end latency measurement (not just per-kernel) | Agents optimize kernels independently; system-level performance unknown |
| Power measurement integration | Cannot optimize for efficiency without power data |
| Thermal monitoring in-loop | Sustained inference may throttle; not modeled |

### 10.4 Missing Abstractions

| Gap | Impact |
|---|---|
| Standard IR for "transformer execution plan" | No formal representation for agent proposals; each agent speaks its own language |
| Hardware-agnostic scheduling language | Cannot retarget schedules across FPGA families, NPUs, and GPUs without reimplementation |
| Decomposition specification format | No standard way to describe multi-device execution plans |

---

## 11. Roadmap

### v0: Analytical Screening (current state - Model2HW)

**Capabilities**:
- Extract model architecture from HuggingFace configs or known families
- Compute memory, bandwidth, compute, IO requirements
- Match against hardware database
- Produce feasibility verdicts and human-readable reports

**Limitations**:
- No optimization - only analysis
- No feedback loop
- No actionable recommendations beyond "fits" / "does not fit"

**Status**: Implemented and functional.

### v1: Constrained Search with Analytical Feedback (6-12 months)

**New capabilities**:
- **Quantization explorer**: search over weight/KV precision combinations, evaluate memory/bandwidth impact using existing analytical models, rank by feasibility margin
- **Decomposition planner**: given a model that does not fit on a single device, propose layer-split and KV-offload strategies with communication cost estimates
- **Configuration recommender**: for a given model-hardware pair, recommend the optimal (precision, context length, batch size) operating point
- **Sensitivity analysis**: identify which bottleneck (memory, bandwidth, compute, IO) dominates and by how much

**Implementation approach**: No LLM agents needed for v1. Zero. All optimization is analytical: sweep functions iterate over precision/context/batch combinations, evaluate each with existing Model2HW analysis modules, and report Pareto frontiers. The entire v1 runs in milliseconds, requires zero external dependencies, produces fully reproducible results, and costs nothing per invocation.

**Validation**: Compare recommendations against known-good configurations from published FPGA LLM deployments.

### v2: Agentic Kernel Optimization (12-24 months)

**New capabilities**:
- **HLS kernel generation agent**: given a transformer operator specification and target FPGA, generate optimized Vitis HLS code using LAAFD-style workflow
- **Synthesis feedback integration**: parse Vitis HLS reports, feed cycle counts and resource utilization back to the agent
- **Template library**: curated, verified HLS kernels for standard transformer operators (GEMM, softmax, layer norm, GQA attention), used as starting points for agent refinement
- **Per-kernel optimization loop**: iterative refinement with synthesis validation

**Prerequisites**:
- Vitis HLS (or equivalent) installed and scriptable
- GPT-5-class model access for agent reasoning
- Test infrastructure for functional correctness validation (co-simulation testbenches)

**Validation**: Compare agent-generated kernels against hand-tuned implementations on VCK190 and Alveo U55C targets. Target: within 80% of hand-tuned performance for standard operators.

**Hard constraints**:
- Each synthesis iteration takes minutes. Budget for 50-100 iterations per kernel.
- Context window limits constrain report size. May need report summarization.
- Functional correctness must be verified per iteration (co-simulation).

### v3: End-to-End Agentic Pipeline (24-36+ months)

**New capabilities**:
- Full model-to-bitstream automation for supported model-hardware pairs
- Multi-kernel orchestration with inter-kernel communication optimization
- Multi-device execution plan generation and validation
- Power and thermal awareness in the optimization loop
- Continuous improvement: agent learns from previously successful configurations

**Prerequisites**:
- Validated v2 kernel library covering >90% of transformer operators
- Full-system simulation capability (or instrumented hardware)
- Accuracy evaluation pipeline integrated with the feedback loop

**Open research questions**:
- Can agents coordinate optimization across dozens of kernels?
- How do we propagate physical design constraints back to the algorithmic level?
- What is the right representation for "transformer execution plan" that is both agent-readable and compiler-consumable?

**Realistic expectation**: v3 will be partially achievable. Full automation of the model-to-hardware pipeline is unlikely within this timeframe. The most likely outcome is a semi-automated system where agents handle kernel optimization and initial scheduling, and human engineers handle system integration, physical design, and validation.

---

## 12. Conclusions

### Can agentic systems meaningfully compress large-model deployment onto small hardware?

**Yes, but with explicit boundaries. Use the LLM only where absolutely necessary.**

1. **Agents cannot violate physics.** If a model's minimum memory footprint exceeds hardware capacity, no optimization helps. The first step is always analytical screening (Model2HW's current capability). This is deterministic arithmetic, not an LLM task.

2. **The vast majority of the optimization stack is deterministic.** Precision sweeps, KV placement decisions, layer-split planning, sensitivity analysis, configuration recommendation, and Pareto frontier computation are all bounded enumerations or closed-form calculations. They run in milliseconds, produce reproducible results, and cost nothing. Using an LLM for any of these tasks adds latency, cost, and hallucination risk for zero benefit.

3. **The LLM earns its cost only when generating or modifying HLS source code.** Specifically: kernel generation, pragma optimization, non-standard operator fusion, and synthesis failure diagnosis. These are the only tasks where there is no closed-form solution and the search requires creative code transformations grounded by compiler feedback.

4. **The feedback loop is everything.** LAAFD works because HLS reports provide fast, deterministic, quantitative feedback. ComPilot works because compiler messages provide legality + speedup data. Where such feedback does not exist (physical design, accuracy impact, thermal behavior), agents will not converge.

5. **Stochastic convergence is the primary practical barrier.** For production use, the system must produce consistently good results, not occasionally optimal ones. This is why the deterministic layers matter: they provide the reproducible foundation that the stochastic agentic layer builds on.

6. **The right mental model is "deterministic optimizer with an agentic HLS backend."** Phases 1-5 (v1) are pure Python with zero dependencies and zero LLM calls. Phase 6+ (v2) adds the LLM only for the HLS kernel optimization loop. This keeps cost low, results reproducible, and failure modes predictable.

### Bottom Line

The Model2HW codebase is the correct foundation. Extending it from "feasibility analysis" to "feasibility + constrained optimization" (v1) is achievable with conventional Python engineering and requires no LLM at all. This covers precision sweeps, decomposition planning, sensitivity analysis, configuration recommendations, and Pareto frontiers. Adding agentic kernel optimization (v2) is plausible given LAAFD's results but requires HLS tool integration and significant infrastructure, and this is the first point where an LLM is justified. Full end-to-end automation (v3) remains a research challenge.

---

## 13. References

[1] M. Moraru, K. Kamalakkannan, J. Dominguez-Trujillo, P. Diehl, A. Barai, J. Loiseau, Z. K. Baker, H. Pritchard, G. M. Shipman. "LAAFD: LLM-based Agents for Accelerated FPGA Design." arXiv:2602.06085, 2026. Los Alamos National Laboratory. [https://arxiv.org/html/2602.06085v1](https://arxiv.org/html/2602.06085v1)

[2] "LLM aided design." Wikipedia. [https://en.wikipedia.org/wiki/LLM_aided_design](https://en.wikipedia.org/wiki/LLM_aided_design)

[3] "Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices." ResearchGate, 2025. [https://www.researchgate.net/publication/396291193](https://www.researchgate.net/publication/396291193)

[4] M. Merouani, I. Kara Bernou, R. Baghdadi. "Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization." PACT 2025. arXiv:2511.00592. [https://arxiv.org/abs/2511.00592](https://arxiv.org/abs/2511.00592)

[5] D. Chen. "While AI-assisted hardware design is far from a solved problem, it is well positioned to fundamentally reshape how hardware systems are conceived, built, verified, and deployed." HiPEAC News, January 2026. [https://www.hipeac.net/news/7129/](https://www.hipeac.net/news/7129/while-ai-assisted-hardware-design-far-solved-problem-well-positioned/)

[6] F. Fahim et al. "hls4ml: An open-source codesign workflow to empower scientific low-power machine learning devices." arXiv:2103.05579, 2021.

[7] Y. Chi, J. Cong, P. Wei, P. Zhou. "SODA: Stencil with optimized dataflow architecture." ICCAD, 2018.

[8] C. Xiong, C. Liu, H. Li, X. Li. "HLSPilot: LLM-based High-Level Synthesis." ICCAD, 2025.

[9] L. Collini, S. Garg, R. Karri. "C2HLSC: Leveraging Large Language Models to Bridge the Software-to-Hardware Design Gap." ACM TODAES, 2025.
