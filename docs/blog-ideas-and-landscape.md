---
title: Blog Ideas and Competitive Landscape
description: State-of-the-art survey of LLM-on-FPGA work, gap analysis showing where Model2HW fits, and six blog post outlines
author: Viktor Ciroski
ms.date: 2026-04-05
ms.topic: overview
keywords:
  - FPGA
  - LLM inference
  - hardware accelerator
  - blog content
  - competitive landscape
estimated_reading_time: 12
---

## State of the Art

The LLM-on-FPGA space is active but fragmented. Existing work falls into five
categories, each solving a different slice of the problem. No single project
spans the full pipeline from model analysis through verified hardware handoff.

### Hand-Built FPGA Accelerators

These projects produce working hardware for specific models through manual
engineering. They prove feasibility but do not generalize.

culurciello/LLM-HW-accelerator is a SystemVerilog transformer that simulates
with Verilator. It runs TinyStories (3M params) and SmolLM2 (135M params) at
toy dimensions (`D_MODEL=4, D_FF=8`), down-projecting real model weights to
fit. Outputs are produced but accuracy is unverified. No synthesis reports, no
bitstream, simulation-only.

Chen et al. (FCCM'24), "Understanding the Potential of FPGA-Based Spatial
Acceleration for LLM Inference," presents analytical models plus an HLS kernel
library for BERT and GPT-2 on the Alveo U280. The spatial (dataflow)
architecture achieves 13.4x over prior FPGA work and 1.9x over the NVIDIA A100
in decode latency. The analytical model estimates performance given on-chip
resources. However, the accelerator is manually designed and model-specific.
There is no reusable design tool.

FlightLLM (FPGA'24) maps LLaMA2-7B to the Alveo U280 and Versal VHK158 with a
configurable sparse DSP chain, always-on-chip decode, and mixed-precision
support. It achieves 6.0x higher energy efficiency than the V100S and 1.2x
higher throughput than the A100 on the VHK158. The "complete mapping flow"
covers compilation and scheduling, but it remains a fixed accelerator
architecture rather than a design automation tool.

### LLM-for-Chip-Design (AI generating hardware)

These projects use LLMs to write or optimize hardware descriptions. They
demonstrate that agentic EDA is viable for individual kernels but have not
scaled to full inference pipelines.

ChipNeMo (NVIDIA, 2024) domain-adapts LLaMA2 for chip design tasks: an
engineering assistant chatbot, EDA script generation, and bug summarization.
ChipNeMo-70B outperforms GPT-4 on two of three use cases. The focus is general
ASIC design, not LLM-specific accelerators, and there is no synthesis feedback
loop.

LAAFD (2026), "LLM-based Agents for Accelerated FPGA Design," uses a
multi-agent workflow (translator, validator, judge, optimizer) to transform C++
into optimized Vitis HLS kernels. With GPT-5, it achieves 99.9% of hand-tuned
baselines on 15 HPC kernels. Key limitations: smaller models degrade to
32-52% of optimal, complex kernels require 10+ runs with 25 iterations each,
and the approach operates on individual kernels rather than complete LLM
inference pipelines.

ComPilot (2025) is an agentic auto-scheduling framework where an LLM proposes
loop transformations and a compiler validates them. It achieves 2.66x
single-run and 3.54x best-of-5 speedups over unoptimized code on PolyBench,
competitive with the Pluto polyhedral optimizer. The approach requires no
fine-tuning, but targets CPU loop optimization rather than FPGA synthesis.

### Hardware-Model Co-Design Studies

"The Case for Co-Designing Model Architectures with Hardware" (Anthony et al.,
2024) provides guidelines for making transformer shapes GPU-efficient. Models
with hardware-aware shapes achieve up to 39% higher throughput at similar
parameter counts. The work is GPU-centric, retrospective (guidelines rather
than automation), and does not address FPGAs.

### Heterogeneous Accelerator Systems

Chameleon (VLDB 2025) combines FPGA vector search accelerators with GPU LLM
inference for retrieval-augmented generation. It achieves 2.16x lower latency
and 3.18x higher throughput versus CPU-GPU baselines. FPGAs handle retrieval,
not LLM inference itself, so the work is adjacent rather than directly
comparable.

### Commercial and Industrial Compilers

StreamTensor from Inspirit IoT maps PyTorch LLM models directly to FPGA
dataflow accelerators, achieving 0.64x lower latency and 1.99x higher energy
efficiency versus GPUs. This demonstrates that automated end-to-end
LLM-to-FPGA compilation is achievable. The compiler is closed-source and
proprietary.

## Gap Analysis

The table below maps what existing work provides against what Model2HW
provides. The central gap: nobody has built the full vertical stack from
analytical screening through verified hardware handoff.

| Existing work | Model2HW |
|---|---|
| Builds one accelerator for one model | Analyzes any model on any board |
| Generates individual HLS kernels | Plans the full pipeline (decomposition, precision, scheduling) |
| Uses LLMs for hardware design generically | Uses LLMs specifically for the model-to-FPGA mapping problem |
| Provides deployment tools for GPUs | Targets FPGAs and constrained edge hardware |
| Performs manual design space exploration | Automates sweep, sensitivity, and recommendation |
| Open-loop: generate and hope | Closed-loop: generate, synthesize, evaluate, iterate, verify, hand off |

### The Unique Contribution

Deterministic Tier 1 screening as a gatekeeper before expensive synthesis.
FlightLLM and Chen et al. include analytical models, but those are integral to
their specific accelerator designs. They are not reusable tools you can point
at an arbitrary model and board combination.

Model2HW's analytical layer answers "should you even attempt this?" in
milliseconds, for free, before committing hours of synthesis time. Combined
with the planned agentic layer (Phases 6-10), it becomes the only system that
connects the feasibility question to automated hardware generation with
synthesis validation.

### Missing Pieces Across the Field

No existing tool provides:

* Fast, accurate LLM inference simulation for FPGAs (required to close the
  agentic feedback loop without real hardware)
* An automated accuracy evaluation pipeline (quantization tradeoff assessment
  in-loop)
* A reusable HLS template library for transformer operators (agents currently
  generate from scratch every time)
* A multi-device communication cost model (needed for decomposition across
  multiple FPGAs or FPGA+host splits)
* Resource-aware HLS optimization (LAAFD optimizes for latency only, ignoring
  resource budgets)

## Blog Post Ideas

### Post 1: Why Your LLM Doesn't Fit on That FPGA (and How to Find One That Does)

Hook: Everyone is excited about FPGA inference. Nobody checks the math first.

Content: Walk through the memory, bandwidth, and compute analysis for
LLaMA 3-8B against various boards. Show how most configurations fail and why.
Demonstrate Model2HW's Tier 1 screening with real CLI output. Reveal the
surprising result that context length and KV precision matter more than weight
precision for many board/model pairs.

Gap it highlights: No existing tool performs this quick sanity check. Engineers
either guess or spend weeks on synthesis to discover the design does not fit.

Target audience: ML engineers evaluating FPGA deployment, hardware engineers
scoping new projects.

### Post 2: The LLM-on-FPGA Landscape: What Works, What Doesn't, and What's Missing

Hook: Survey the field honestly. FlightLLM, Chen et al., culurciello,
StreamTensor, LAAFD, all in one place.

Content: Categorize approaches (hand-built RTL, analytical models, automated
HLS, commercial compilers). Show what each achieves and where each stops.
Include a comparison table. Identify the missing middle: nobody connects the
analytical "can it fit?" question to the synthesis "make it work" pipeline.

Gap it highlights: The field is fragmented. Individual pieces exist but nothing
ties them together. Model2HW's roadmap is the integration thesis.

Target audience: Researchers, hardware architects, investors tracking the FPGA
AI accelerator space.

### Post 3: My PhD Advisor and I Are Solving the Same Problem From Opposite Ends

Hook: Personal narrative. Your former advisor hand-writes SystemVerilog
transformers; you are building the automated pipeline. Neither knew about the
other's work until you found his repo.

Content: Compare bottom-up (RTL-first, manually tune dimensions, simulate,
iterate) versus top-down (analyze model requirements, sweep design space,
generate optimized hardware). Show the tradeoffs in concreteness versus
generality, speed of first prototype versus scalability. Argue that automation
is inevitable because hand-design does not scale: a seasoned hardware designer
is still manually adjusting parameters for a 3M-parameter toy model.

Gap it highlights: The field needs automation. One proof point is that
experienced researchers default to manual processes because the tooling does
not exist yet.

Target audience: Broad technical audience, good for personal blog or
newsletter.

### Post 4: LLMs Designing LLM Hardware: Where Agentic EDA Actually Works and Where It Doesn't

Hook: LAAFD achieves 99.9% of optimal on individual kernels. ChipNeMo
outperforms GPT-4 on EDA scripts. Are we done?

Content: Honest analysis of what agentic HLS generation can do (single
kernels, constrained operators, well-defined fitness functions) versus what it
cannot (full pipeline orchestration, timing closure, thermal modeling, system
integration). Walk through Model2HW's explicit boundary between deterministic
and agentic layers. Explain why using an LLM for tasks that can be expressed as
arithmetic formulas adds latency, cost, and hallucination risk for zero
benefit.

Gap it highlights: The industry conflates "LLM can write Verilog" with "LLM
can design an accelerator." The hard part is knowing where the boundary is.
Model2HW draws this line explicitly.

Target audience: AI/ML researchers interested in agentic systems, EDA
professionals.

### Post 5: From "Does It Fit?" to "Flash the Board": Building an Automated FPGA Deployment Pipeline for LLMs

Hook: Existing tools give you one piece of the puzzle. Here is what the full
pipeline looks like.

Content: Walk through Model2HW's 10-phase roadmap and the four evaluation
tiers (analytical, HLS synthesis, RTL simulation, cloud FPGA). Show the
current state (analytical screening with 100% test coverage) and the path to
closed-loop agentic optimization. Include the architecture diagram from the
feasibility study. Discuss tier advancement rules: a design only moves to the
next evaluation tier if it passes the current one, eliminating wasted synthesis
time and removing the need to purchase an FPGA during development.

Gap it highlights: Nobody has articulated this end-to-end vision publicly.
This is the manifesto post.

Target audience: Hardware engineers, ML infrastructure teams, anyone building
AI deployment pipelines.

### Post 6: The Precision Paradox: Why INT4 LLMs on FPGAs Beat FP16 GPUs in Tok/s per Watt

Hook: Counterintuitive result that falls directly out of the bandwidth
arithmetic.

Content: Use Model2HW's precision sweep to show how INT4 on a mid-range FPGA
(VCK190) can match or beat an A100 in energy efficiency for decode-bound
workloads. Walk through the math: decode is memory-bandwidth-bound, INT4
reduces bytes per weight by 4x, FPGA power budget is 10-20x lower than a GPU.
The tok/s per watt comparison inverts the conventional wisdom that GPUs always
win.

Gap it highlights: This analysis takes seconds with Model2HW and is impossible
without it (or a spreadsheet and significant patience). The tool makes
non-obvious insights accessible.

Target audience: ML engineers evaluating inference cost, edge deployment teams,
sustainability-focused audiences.

## Recommended Publishing Order

1. Post 1 (practical hook, demonstrates the tool, attracts practitioners)
2. Post 3 (personal story, broad appeal, builds the author's profile)
3. Post 2 (landscape survey, establishes authority, useful as reference)
4. Post 6 (counterintuitive result, shareable, drives tool adoption)
5. Post 4 (deeper technical content for the agentic AI audience)
6. Post 5 (manifesto, best published after the earlier posts build credibility)

## References

* Chen, H. et al. "Understanding the Potential of FPGA-Based Spatial
  Acceleration for Large Language Model Inference." FCCM'24 / ACM TRETS.
  arXiv:2312.15159.
* Zeng, S. et al. "FlightLLM: Efficient Large Language Model Inference with a
  Complete Mapping Flow on FPGAs." FPGA'24. arXiv:2401.03868.
* Liu, M. et al. "ChipNeMo: Domain-Adapted LLMs for Chip Design." 2024.
  arXiv:2311.00176.
* Anthony, Q. et al. "The Case for Co-Designing Model Architectures with
  Hardware." 2024. arXiv:2401.14489.
* Jiang, W. et al. "Chameleon: a Heterogeneous and Disaggregated Accelerator
  System for Retrieval-Augmented Language Models." VLDB 2025.
  arXiv:2310.09949.
* culurciello. "LLM-HW-accelerator." GitHub, 2026.
  `https://github.com/culurciello/LLM-HW-accelerator`
* LAAFD. "LLM-based Agents for Accelerated FPGA Design." 2026. Referenced in
  Model2HW feasibility study.
* ComPilot. "Agentic Auto-Scheduling Framework." 2025. Referenced in Model2HW
  feasibility study.
* StreamTensor / Inspirit IoT. Referenced in Model2HW feasibility study.
