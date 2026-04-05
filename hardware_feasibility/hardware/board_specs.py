"""Board / accelerator hardware specifications database."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BoardSpec:
    """Hardware specification for a target accelerator board."""

    name: str
    category: str  # "fpga", "gpu", "npu", "edge_soc", "custom"

    # Memory
    memory_gb: float  # total on-board / on-chip memory accessible to accelerator
    memory_bandwidth_gbps: float  # peak memory bandwidth in GB/s

    # Compute
    peak_tops_int8: Optional[float] = None  # INT8 TOPS
    peak_tflops_fp16: Optional[float] = None  # FP16 TFLOPS

    # Host link
    host_link: str = "PCIe Gen3 x4"  # human-readable
    host_link_bandwidth_gbps: float = 4.0  # effective GB/s

    # Power
    tdp_watts: Optional[float] = None

    # Notes
    notes: str = ""


# ---------------------------------------------------------------------------
# Built-in board database.  Expand as needed.
# ---------------------------------------------------------------------------

BOARD_DATABASE: dict[str, BoardSpec] = {}


def _register(spec: BoardSpec) -> None:
    BOARD_DATABASE[spec.name] = spec


# ---- AMD/Xilinx FPGAs ----

_register(BoardSpec(
    name="Xilinx ZCU104",
    category="fpga",
    memory_gb=2.0,
    memory_bandwidth_gbps=4.2,
    peak_tops_int8=1.2,
    host_link="PCIe Gen2 x4",
    host_link_bandwidth_gbps=2.0,
    tdp_watts=30,
    notes="Zynq UltraScale+ ZU7EV. 504 DSP slices. Typical edge FPGA.",
))

_register(BoardSpec(
    name="Xilinx VCK190",
    category="fpga",
    memory_gb=8.0,
    memory_bandwidth_gbps=25.6,
    peak_tops_int8=133.0,
    host_link="PCIe Gen4 x8",
    host_link_bandwidth_gbps=16.0,
    tdp_watts=75,
    notes="Versal AI Core XCVC1902. 400 AI Engines. DDR4 + LPDDR4.",
))

_register(BoardSpec(
    name="Xilinx Alveo U250",
    category="fpga",
    memory_gb=64.0,
    memory_bandwidth_gbps=77.0,
    peak_tops_int8=33.0,
    peak_tflops_fp16=16.5,
    host_link="PCIe Gen3 x16",
    host_link_bandwidth_gbps=16.0,
    tdp_watts=225,
    notes="Datacenter FPGA. 4x DDR4 banks. Large fabric.",
))

_register(BoardSpec(
    name="Xilinx Alveo U55C",
    category="fpga",
    memory_gb=16.0,  # HBM2
    memory_bandwidth_gbps=460.0,
    peak_tops_int8=33.0,
    peak_tflops_fp16=16.5,
    host_link="PCIe Gen4 x16",
    host_link_bandwidth_gbps=32.0,
    tdp_watts=150,
    notes="HBM2-based Alveo. High bandwidth for streaming workloads.",
))

_register(BoardSpec(
    name="AMD Versal VE2802",
    category="fpga",
    memory_gb=32.0,
    memory_bandwidth_gbps=819.0,
    peak_tops_int8=256.0,
    host_link="PCIe Gen5 x16",
    host_link_bandwidth_gbps=64.0,
    tdp_watts=150,
    notes="Versal AI Edge series with HBM and AI Engines.",
))

# ---- Intel FPGAs ----

_register(BoardSpec(
    name="Intel Agilex 7 AGF027",
    category="fpga",
    memory_gb=32.0,
    memory_bandwidth_gbps=460.0,
    peak_tops_int8=70.0,
    host_link="PCIe Gen5 x16",
    host_link_bandwidth_gbps=64.0,
    tdp_watts=150,
    notes="Agilex 7 FPGA with HBM2e. High-bandwidth FPGA for AI inference.",
))

_register(BoardSpec(
    name="Intel Stratix 10 NX",
    category="fpga",
    memory_gb=32.0,
    memory_bandwidth_gbps=460.0,
    peak_tops_int8=143.0,
    host_link="PCIe Gen3 x16",
    host_link_bandwidth_gbps=16.0,
    tdp_watts=225,
    notes="AI-optimized FPGA with HBM2 and dedicated AI tensor blocks.",
))

# ---- NVIDIA GPUs (reference points) ----

_register(BoardSpec(
    name="NVIDIA Jetson Orin NX 16GB",
    category="edge_soc",
    memory_gb=16.0,
    memory_bandwidth_gbps=102.4,
    peak_tops_int8=100.0,
    peak_tflops_fp16=18.0,
    host_link="PCIe Gen4 x4 + USB",
    host_link_bandwidth_gbps=8.0,
    tdp_watts=25,
    notes="Edge SoC. Unified memory (shared CPU/GPU). Ampere GPU + DLA.",
))

_register(BoardSpec(
    name="NVIDIA Jetson AGX Orin 64GB",
    category="edge_soc",
    memory_gb=64.0,
    memory_bandwidth_gbps=204.8,
    peak_tops_int8=275.0,
    peak_tflops_fp16=40.0,
    host_link="PCIe Gen4 x8",
    host_link_bandwidth_gbps=16.0,
    tdp_watts=60,
    notes="Top-end Jetson. Unified LPDDR5 memory.",
))

_register(BoardSpec(
    name="NVIDIA A100 80GB",
    category="gpu",
    memory_gb=80.0,
    memory_bandwidth_gbps=2039.0,
    peak_tops_int8=624.0,
    peak_tflops_fp16=312.0,
    host_link="PCIe Gen4 x16 / NVLink",
    host_link_bandwidth_gbps=64.0,
    tdp_watts=300,
    notes="Datacenter GPU. Reference high-end point.",
))

_register(BoardSpec(
    name="NVIDIA RTX 4090",
    category="gpu",
    memory_gb=24.0,
    memory_bandwidth_gbps=1008.0,
    peak_tops_int8=660.0,
    peak_tflops_fp16=165.0,
    host_link="PCIe Gen4 x16",
    host_link_bandwidth_gbps=32.0,
    tdp_watts=450,
    notes="Consumer GPU. High compute and bandwidth.",
))

# ---- Edge AI NPUs ----

_register(BoardSpec(
    name="Google Coral Edge TPU",
    category="npu",
    memory_gb=0.008,  # 8 MB SRAM
    memory_bandwidth_gbps=8.0,
    peak_tops_int8=4.0,
    host_link="USB 3.0 / PCIe Gen2 x1",
    host_link_bandwidth_gbps=0.5,
    tdp_watts=2,
    notes="Tiny NPU. Only INT8. Weight streaming from host. Not suitable for LLMs.",
))

_register(BoardSpec(
    name="Hailo-8",
    category="npu",
    memory_gb=0.0,  # No on-board memory; uses host
    memory_bandwidth_gbps=0.0,
    peak_tops_int8=26.0,
    host_link="PCIe Gen3 x4",
    host_link_bandwidth_gbps=4.0,
    tdp_watts=5.5,
    notes="Edge NPU. Fully host-dependent for memory. Weight streaming architecture.",
))

_register(BoardSpec(
    name="Qualcomm Cloud AI 100",
    category="npu",
    memory_gb=16.0,
    memory_bandwidth_gbps=134.0,
    peak_tops_int8=400.0,
    host_link="PCIe Gen4 x16",
    host_link_bandwidth_gbps=32.0,
    tdp_watts=75,
    notes="Cloud inference accelerator. LPDDR4X on-board.",
))

# ---- Custom / RISC-V ----

_register(BoardSpec(
    name="Generic FPGA 4GB DDR4",
    category="fpga",
    memory_gb=4.0,
    memory_bandwidth_gbps=12.8,
    peak_tops_int8=5.0,
    host_link="PCIe Gen3 x4",
    host_link_bandwidth_gbps=4.0,
    tdp_watts=25,
    notes="Generic mid-range FPGA with single DDR4 channel. Rough baseline.",
))

_register(BoardSpec(
    name="Generic FPGA 8GB DDR4",
    category="fpga",
    memory_gb=8.0,
    memory_bandwidth_gbps=25.6,
    peak_tops_int8=10.0,
    host_link="PCIe Gen3 x8",
    host_link_bandwidth_gbps=8.0,
    tdp_watts=40,
    notes="Generic FPGA with dual DDR4 channels.",
))


def list_boards(category: Optional[str] = None) -> list[BoardSpec]:
    """Return all boards, optionally filtered by category."""
    boards = list(BOARD_DATABASE.values())
    if category:
        boards = [b for b in boards if b.category == category]
    return sorted(boards, key=lambda b: b.memory_gb)


def get_board(name: str) -> BoardSpec:
    if name not in BOARD_DATABASE:
        available = ", ".join(sorted(BOARD_DATABASE.keys()))
        raise ValueError(f"Unknown board '{name}'. Available: {available}")
    return BOARD_DATABASE[name]
