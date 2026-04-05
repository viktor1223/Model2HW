# Model2HW

Hardware feasibility analyzer for LLM inference on constrained accelerators.

Model2HW takes a transformer model specification and evaluates whether it can run on a given hardware target by analyzing memory capacity, memory bandwidth, compute throughput, and host-device IO requirements. It produces a feasibility verdict and a detailed report.

## Features

* Extract model architecture from HuggingFace `config.json` files, Hub model IDs, or built-in family templates
* Compute weight memory, KV cache, and activation buffer requirements across precisions (FP32, FP16, BF16, INT8, INT4)
* Estimate decode-phase bandwidth demand and compare against hardware limits
* Model host-device IO for KV-on-host scenarios
* Rank all boards in the built-in database against a model's requirements
* Output human-readable reports or structured JSON

## Supported Hardware

The built-in board database covers:

* AMD/Xilinx FPGAs (ZCU104, VCK190, Alveo U55C, Versal VE2802)
* Intel FPGAs (Agilex 7, Stratix 10 NX)
* NVIDIA GPUs (Jetson Orin NX, A100, H100)
* Edge SoCs and NPUs

Run `model2hw --list-boards` to see the full list.

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

To load models directly from the HuggingFace Hub:

```bash
pip install -e ".[hub]"
```

## Quick Start

Analyze Llama 3-8B in INT4 against the Xilinx VCK190:

```bash
model2hw --family llama3-8b \
  --weight-precision int4 \
  --kv-precision int8 \
  --context-length 2048 \
  --board "AMD Versal VCK190" \
  --target-tok-s 10
```

Rank all boards for a model loaded from a local config file:

```bash
model2hw --config path/to/config.json \
  --weight-precision fp16 \
  --match-all
```

List built-in model families:

```bash
model2hw --list-families
```

Export results as JSON:

```bash
model2hw --family llama3.2-1b --board "AMD Versal VCK190" --json
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Project Structure

```text
hardware_feasibility/
  analysis/       Memory, bandwidth, compute, IO, and verdict engines
  hardware/       Board specs database and hardware matcher
  models/         Architecture rules, known families, HF config loader
  outputs/        Report and JSON export
  cli.py          Command-line entry point
tests/            Test suite
docs/             Design documents and feasibility study
```

## License

See the project repository for license details.
