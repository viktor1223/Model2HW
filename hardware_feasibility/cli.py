"""CLI entry point for the Hardware Feasibility Analyzer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models.architecture_rules import KNOWN_FAMILIES, ModelSpec
from .models.hf_config_loader import (
    load_from_hf_config,
    load_from_hf_hub,
    load_from_known_family,
)
from .analysis.memory import analyze_memory
from .analysis.bandwidth import analyze_bandwidth
from .analysis.compute import analyze_compute
from .analysis.io import analyze_io
from .analysis.verdict import render_verdict
from .hardware.board_specs import BOARD_DATABASE, get_board
from .hardware.matcher import rank_boards
from .outputs.json_export import export_json
from .outputs.report import generate_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="model2hw",
        description=(
            "Hardware Feasibility Analyzer for LLM inference.\n"
            "Analyzes a model's memory, bandwidth, compute, and IO requirements\n"
            "and produces a feasibility verdict for target hardware."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Model source (mutually exclusive) ---
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--config",
        type=str,
        help="Path to a HuggingFace-style config.json file.",
    )
    src.add_argument(
        "--hf-model",
        type=str,
        help="HuggingFace Hub model ID (e.g. meta-llama/Llama-3.2-1B). Requires huggingface_hub.",
    )
    src.add_argument(
        "--family",
        type=str,
        choices=sorted(KNOWN_FAMILIES.keys()),
        help="Use a known model family template.",
    )
    src.add_argument(
        "--list-families",
        action="store_true",
        help="List available built-in model families and exit.",
    )
    src.add_argument(
        "--list-boards",
        action="store_true",
        help="List available board specs and exit.",
    )

    # --- Precision ---
    p.add_argument("--weight-precision", type=str, default="fp16",
                    help="Weight precision: fp32, fp16, bf16, int8, int4 (default: fp16)")
    p.add_argument("--kv-precision", type=str, default="fp16",
                    help="KV cache precision (default: fp16)")

    # --- Runtime assumptions ---
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    p.add_argument("--context-length", type=int, default=4096,
                    help="Target context length in tokens (default: 4096)")
    p.add_argument("--prefill-length", type=int, default=512,
                    help="Prefill / prompt length in tokens (default: 512)")
    p.add_argument("--decode-length", type=int, default=256,
                    help="Decode / generation length in tokens (default: 256)")
    p.add_argument("--target-tok-s", type=float, default=10.0,
                    help="Target decode tokens/sec (default: 10)")
    p.add_argument("--kv-off-accelerator", action="store_true",
                    help="KV cache resides on host, not accelerator.")

    # --- Hardware target ---
    p.add_argument("--board", type=str, default=None,
                    help="Evaluate against a specific board from the database.")
    p.add_argument("--board-category", type=str, default=None,
                    help="Filter board ranking to a category: fpga, gpu, npu, edge_soc")
    p.add_argument("--match-all", action="store_true",
                    help="Rank the model against all known boards.")

    # --- Custom hardware target ---
    p.add_argument("--target-memory-gb", type=float, default=None,
                    help="Custom target memory in GB.")
    p.add_argument("--target-bw-gbps", type=float, default=None,
                    help="Custom target memory bandwidth in GB/s.")
    p.add_argument("--target-link-gbps", type=float, default=None,
                    help="Custom target host-link bandwidth in GB/s.")

    # --- Output ---
    p.add_argument("--json", action="store_true", help="Output as JSON instead of report.")
    p.add_argument("--output", "-o", type=str, default=None,
                    help="Write output to a file instead of stdout.")

    return p


def run(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- List modes ---
    if args.list_families:
        print("Available model families:\n")
        for name in sorted(KNOWN_FAMILIES.keys()):
            f = KNOWN_FAMILIES[name]
            print(f"  {name:20s}  layers={f['num_layers']:3d}  hidden={f['hidden_size']:5d}  "
                  f"heads={f['num_attention_heads']:3d}  kv_heads={f['num_kv_heads']:3d}")
        return

    if args.list_boards:
        print("Available board specs:\n")
        for name, b in sorted(BOARD_DATABASE.items()):
            print(f"  {name:35s}  [{b.category:8s}]  mem={b.memory_gb:6.1f}GB  "
                  f"bw={b.memory_bandwidth_gbps:7.1f}GB/s  link={b.host_link_bandwidth_gbps:.1f}GB/s  "
                  f"tdp={b.tdp_watts or '?':>4}W")
        return

    # --- Build model spec ---
    common = dict(
        weight_precision=args.weight_precision,
        kv_precision=args.kv_precision,
        batch_size=args.batch_size,
        context_length=args.context_length,
        prefill_length=args.prefill_length,
        decode_length=args.decode_length,
        target_tokens_per_sec=args.target_tok_s,
        kv_on_accelerator=not args.kv_off_accelerator,
    )

    if args.config:
        spec = load_from_hf_config(args.config, **common)
    elif args.hf_model:
        spec = load_from_hf_hub(args.hf_model, **common)
    elif args.family:
        spec = load_from_known_family(args.family, **common)
    else:
        parser.error("Provide --config, --hf-model, or --family.")
        return  # unreachable

    # --- Run analysis ---
    mem = analyze_memory(spec)
    bw = analyze_bandwidth(spec)
    compute = analyze_compute(spec, 0)  # generic; board-specific in matcher
    io = analyze_io(spec)

    # --- Determine hardware targets ---
    target_mem = args.target_memory_gb
    target_bw = args.target_bw_gbps
    target_link = args.target_link_gbps

    if args.board:
        board = get_board(args.board)
        target_mem = target_mem or board.memory_gb
        target_bw = target_bw or board.memory_bandwidth_gbps
        target_link = target_link or board.host_link_bandwidth_gbps

    verdict = render_verdict(
        spec, mem, bw, compute, io,
        target_memory_gb=target_mem,
        target_bandwidth_gbps=target_bw,
        target_link_bandwidth_gbps=target_link,
    )

    # --- Hardware matching ---
    matches = None
    if args.match_all or args.board_category:
        matches = rank_boards(spec, mem, bw, io, category=args.board_category)

    # --- Output ---
    if args.json:
        output = export_json(spec, mem, bw, compute, io, verdict, matches)
    else:
        output = generate_report(spec, mem, bw, compute, io, verdict, matches)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
