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

    # On-chip resources (FPGA-specific)
    bram_kb: Optional[int] = None       # Total Block RAM in KB
    uram_kb: Optional[int] = None       # Total UltraRAM in KB (Xilinx specific)
    dsp_slices: Optional[int] = None    # DSP slice count
    lut_count: Optional[int] = None     # LUT count (in thousands)

    # Memory hierarchy
    memory_type: str = "ddr4"           # "ddr4", "ddr5", "hbm2", "hbm2e", "lpddr4", "lpddr5", "unified"
    memory_channels: int = 1            # Number of memory channels

    # Compute architecture
    has_ai_engines: bool = False         # Versal AI Engines or similar
    ai_engine_count: Optional[int] = None

    @property
    def on_chip_memory_kb(self) -> int:
        """Total on-chip fast memory (BRAM + URAM) in KB."""
        bram = self.bram_kb or 0
        uram = self.uram_kb or 0
        return bram + uram


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
    bram_kb=312,
    uram_kb=96,
    dsp_slices=1728,
    lut_count=504,
    memory_type="ddr4",
    memory_channels=1,
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
    bram_kb=967,
    uram_kb=0,
    dsp_slices=1968,
    lut_count=899,
    memory_type="ddr4",
    memory_channels=2,
    has_ai_engines=True,
    ai_engine_count=400,
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
    bram_kb=2688,
    uram_kb=1280,
    dsp_slices=12288,
    lut_count=1727,
    memory_type="ddr4",
    memory_channels=4,
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
    bram_kb=2016,
    uram_kb=960,
    dsp_slices=9024,
    lut_count=1303,
    memory_type="hbm2",
    memory_channels=32,
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
    bram_kb=1344,
    uram_kb=576,
    dsp_slices=3984,
    lut_count=1218,
    memory_type="hbm2e",
    memory_channels=8,
    has_ai_engines=True,
    ai_engine_count=304,
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
    bram_kb=7404,
    uram_kb=0,
    dsp_slices=4510,
    lut_count=949,
    memory_type="hbm2e",
    memory_channels=16,
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
    bram_kb=10960,
    uram_kb=0,
    dsp_slices=5760,
    lut_count=1866,
    memory_type="hbm2",
    memory_channels=32,
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
    memory_type="lpddr5",
    memory_channels=8,
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
    memory_type="lpddr5",
    memory_channels=16,
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
    memory_type="hbm2e",
    memory_channels=5,
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
    memory_type="ddr6x",
    memory_channels=12,
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
    memory_type="ddr4",
    memory_channels=1,
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
    memory_type="ddr4",
    memory_channels=2,
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
