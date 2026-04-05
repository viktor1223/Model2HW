"""Tests for the decomposition planner."""

from hardware_feasibility.models.hf_config_loader import load_from_known_family
from hardware_feasibility.hardware.board_specs import get_board, BoardSpec
from hardware_feasibility.analysis.decomposition import (
    plan_decomposition,
    plan_kv_offload,
    plan_pipeline_parallel,
)


def test_no_decomposition_needed():
    """Small model on large board should report single_device_feasible."""
    spec = load_from_known_family("qwen2-0.5b", weight_precision="int4", context_length=2048)
    board = get_board("Xilinx VCK190")  # 8 GB
    result = plan_decomposition(spec, board)

    assert result.single_device_feasible is True


def test_kv_offload_for_tight_fit():
    """Model where weights fit but total doesn't - KV offload should help."""
    # llama3-8b FP16 on Alveo U250 (64 GB) fits easily, but let's use a
    # constrained custom board to force KV offload.
    spec = load_from_known_family(
        "llama3-8b",
        weight_precision="fp16",
        context_length=4096,
        batch_size=1,
    )
    # Custom board: 17 GB memory (weights ~16 GB, KV ~2 GB, total ~18 GB)
    board = BoardSpec(
        name="Test Board 17GB",
        category="fpga",
        memory_gb=17.0,
        memory_bandwidth_gbps=100.0,
        host_link="PCIe Gen4 x8",
        host_link_bandwidth_gbps=16.0,
    )
    plan = plan_kv_offload(spec, board)

    assert plan is not None
    assert plan.strategy == "kv_offload"
    assert plan.feasible is True
    assert plan.total_devices == 1
    assert plan.devices[0].kv_cache_memory_gb == 0.0  # KV on host
    assert plan.estimated_tok_per_sec > 0


def test_kv_offload_returns_none_if_weights_dont_fit():
    """If weights alone exceed board memory, KV offload can't help."""
    spec = load_from_known_family("llama2-13b", weight_precision="fp16")
    board = get_board("Xilinx ZCU104")  # 2 GB - weights alone are ~26 GB
    plan = plan_kv_offload(spec, board)

    assert plan is None


def test_pipeline_2way_split():
    """Verify 2-device split for a model that barely doesn't fit on one board."""
    spec = load_from_known_family(
        "llama3-8b",
        weight_precision="int8",
        context_length=2048,
    )
    # ~8 GB weights at INT8, plus KV => doesn't fit on 8 GB board
    board = get_board("Xilinx VCK190")  # 8 GB

    plans = plan_pipeline_parallel(spec, board)
    # Should have plans for 2..8 devices
    assert len(plans) >= 2

    # Find the first feasible plan
    feasible = [p for p in plans if p.feasible]
    assert len(feasible) > 0
    first_feasible = feasible[0]
    assert first_feasible.total_devices >= 2
    assert first_feasible.estimated_tok_per_sec > 0
    assert first_feasible.pipeline_bubble_fraction > 0


def test_pipeline_layer_assignments_cover_all_layers():
    """Layer assignments should cover all layers without gaps or overlaps."""
    spec = load_from_known_family("llama3-8b", weight_precision="int8")
    board = get_board("Xilinx VCK190")

    plans = plan_pipeline_parallel(spec, board)
    for plan in plans:
        # Check contiguous coverage
        expected_start = 0
        for dev in plan.devices:
            assert dev.layer_start == expected_start
            expected_start = dev.layer_end
        assert expected_start == spec.num_layers


def test_decomposition_report_format():
    """Verify report formatter works with decomposition result."""
    from hardware_feasibility.outputs.report import format_decomposition_report

    spec = load_from_known_family("llama3-8b", weight_precision="fp16", context_length=4096)
    board = get_board("Xilinx ZCU104")  # 2 GB - won't fit, triggers decomposition
    result = plan_decomposition(spec, board)

    report = format_decomposition_report(result)
    assert "DECOMPOSITION ANALYSIS" in report
    assert "pipeline_parallel" in report


def test_decomposition_json_format():
    """Verify JSON formatter works with decomposition result."""
    from hardware_feasibility.outputs.json_export import format_decomposition_json

    spec = load_from_known_family("llama3-8b", weight_precision="fp16", context_length=4096)
    board = get_board("Xilinx ZCU104")  # 2 GB - won't fit
    result = plan_decomposition(spec, board)

    data = format_decomposition_json(result)
    assert "model" in data
    assert "plans" in data
    assert isinstance(data["plans"], list)
    assert len(data["plans"]) > 0
