"""Tests for extended hardware model (Phase 5)."""

from hardware_feasibility.hardware.board_specs import get_board, list_boards


def test_vck190_has_extended_fields():
    """Verify VCK190 has BRAM, DSP, AI engine fields populated."""
    b = get_board("Xilinx VCK190")
    assert b.bram_kb == 967
    assert b.dsp_slices == 1968
    assert b.lut_count == 899
    assert b.has_ai_engines is True
    assert b.ai_engine_count == 400
    assert b.memory_type == "ddr4"
    assert b.memory_channels == 2


def test_alveo_u55c_hbm():
    """Verify Alveo U55C has HBM memory type and on-chip resources."""
    b = get_board("Xilinx Alveo U55C")
    assert b.memory_type == "hbm2"
    assert b.memory_channels == 32
    assert b.bram_kb == 2016
    assert b.uram_kb == 960
    assert b.dsp_slices == 9024


def test_gpu_has_none_for_fpga_fields():
    """Verify GPU boards have None for FPGA-specific fields."""
    b = get_board("NVIDIA A100 80GB")
    assert b.bram_kb is None
    assert b.uram_kb is None
    assert b.dsp_slices is None
    assert b.lut_count is None
    assert b.has_ai_engines is False
    assert b.ai_engine_count is None


def test_npu_has_none_for_fpga_fields():
    """Verify NPU boards have None for FPGA-specific fields."""
    b = get_board("Google Coral Edge TPU")
    assert b.bram_kb is None
    assert b.dsp_slices is None
    assert b.lut_count is None


def test_on_chip_memory_kb():
    """Verify the derived on_chip_memory_kb property."""
    b = get_board("Xilinx Alveo U250")
    # BRAM 2688 + URAM 1280 = 3968
    assert b.on_chip_memory_kb == 3968


def test_on_chip_memory_kb_none_fields():
    """Verify on_chip_memory_kb returns 0 when BRAM/URAM are None."""
    b = get_board("NVIDIA A100 80GB")
    assert b.on_chip_memory_kb == 0


def test_all_fpga_boards_have_memory_type():
    """Every FPGA board should have a non-default memory_type or be ddr4."""
    fpga_boards = list_boards(category="fpga")
    for b in fpga_boards:
        assert b.memory_type in ("ddr4", "ddr5", "hbm2", "hbm2e", "lpddr4", "lpddr5"), \
            f"{b.name} has unexpected memory_type: {b.memory_type}"
        assert b.memory_channels >= 1, f"{b.name} has invalid memory_channels"


def test_intel_fpga_has_extended_fields():
    """Verify Intel FPGAs have on-chip resources populated."""
    b = get_board("Intel Agilex 7 AGF027")
    assert b.bram_kb == 7404
    assert b.dsp_slices == 4510
    assert b.memory_type == "hbm2e"

    b2 = get_board("Intel Stratix 10 NX")
    assert b2.bram_kb == 10960
    assert b2.dsp_slices == 5760
    assert b2.memory_type == "hbm2"
