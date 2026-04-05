"""Decomposition planner: multi-device and KV offload strategies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..models.architecture_rules import ModelSpec
from ..hardware.board_specs import BoardSpec
from .memory import analyze_memory, estimate_weight_memory, estimate_kv_cache, BYTES_PER_GB
from .bandwidth import analyze_bandwidth
from .compute import analyze_compute
from .io import analyze_io
from .verdict import FeasibilityVerdict, render_verdict


@dataclass
class DeviceAssignment:
    """Assignment of model layers to a specific device."""

    device_name: str
    layer_start: int       # inclusive
    layer_end: int         # exclusive
    weight_memory_gb: float
    kv_cache_memory_gb: float
    total_memory_gb: float
    memory_utilization: float  # fraction of device capacity


@dataclass
class DecompositionPlan:
    """A complete multi-device execution plan."""

    strategy: str  # "pipeline_parallel", "kv_offload", "hybrid"
    devices: list[DeviceAssignment]
    total_devices: int

    # Communication analysis
    inter_device_transfer_bytes_per_token: int
    inter_device_bandwidth_required_gbps: float

    # Performance
    estimated_tok_per_sec: float
    pipeline_bubble_fraction: float  # 0.0 - 1.0

    feasible: bool
    details: list[str]


@dataclass
class DecompositionResult:
    """All decomposition plans considered."""

    model_name: str
    single_device_feasible: bool
    plans: list[DecompositionPlan]
    best_plan: DecompositionPlan | None


def _weight_gb_for_layers(spec: ModelSpec, layer_start: int, layer_end: int) -> float:
    """Estimate weight memory for a contiguous slice of layers.

    For standard decoder-only transformers (uniform layers), this is
    simply (layer_count / total_layers) * total_weight_memory.
    Embedding/LM-head weights are assigned to the first device.
    """
    total_weight_bytes = estimate_weight_memory(spec)
    # Embedding + LM head account for vocab_size * hidden_size * 2 * bytes_per_weight
    embed_bytes = spec.vocab_size * spec.hidden_size * 2 * spec.bytes_per_weight
    layer_weight_bytes = total_weight_bytes - embed_bytes
    per_layer_bytes = layer_weight_bytes / spec.num_layers if spec.num_layers > 0 else 0

    slice_bytes = per_layer_bytes * (layer_end - layer_start)
    # Add embedding weights to the first slice
    if layer_start == 0:
        slice_bytes += embed_bytes

    return slice_bytes / BYTES_PER_GB


def _kv_gb_for_layers(spec: ModelSpec, layer_start: int, layer_end: int) -> float:
    """KV cache memory for a layer slice at the target context length."""
    total_kv_bytes = estimate_kv_cache(spec)
    if spec.num_layers == 0:
        return 0.0
    per_layer_kv = total_kv_bytes / spec.num_layers
    return per_layer_kv * (layer_end - layer_start) / BYTES_PER_GB


def plan_pipeline_parallel(
    spec: ModelSpec,
    board: BoardSpec,
) -> list[DecompositionPlan]:
    """Try splitting layers across N identical copies of a board.

    For N = 2..max_devices, divide layers evenly and check fit on each device.
    """
    max_devices = min(spec.num_layers, 8)  # cap at 8 devices
    plans: list[DecompositionPlan] = []

    for n_devices in range(2, max_devices + 1):
        layers_per_device = spec.num_layers // n_devices
        remainder = spec.num_layers % n_devices

        devices: list[DeviceAssignment] = []
        cursor = 0
        feasible = True

        for d in range(n_devices):
            # Distribute remainder layers to the first `remainder` devices
            count = layers_per_device + (1 if d < remainder else 0)
            layer_start = cursor
            layer_end = cursor + count
            cursor = layer_end

            w_gb = _weight_gb_for_layers(spec, layer_start, layer_end)
            kv_gb = _kv_gb_for_layers(spec, layer_start, layer_end)
            total = w_gb + kv_gb
            util = total / board.memory_gb if board.memory_gb > 0 else float("inf")

            if total > board.memory_gb:
                feasible = False

            devices.append(DeviceAssignment(
                device_name=board.name,
                layer_start=layer_start,
                layer_end=layer_end,
                weight_memory_gb=w_gb,
                kv_cache_memory_gb=kv_gb,
                total_memory_gb=total,
                memory_utilization=min(util, 9.99),
            ))

        # Inter-device transfer: activation tensor between pipeline stages
        # hidden_size * batch_size * bytes_per_weight
        inter_bytes = int(spec.hidden_size * spec.batch_size * spec.bytes_per_weight)

        # Pipeline bubble: (N-1) / (N-1 + decode_length)
        bubble = (n_devices - 1) / (n_devices - 1 + spec.decode_length) if spec.decode_length > 0 else 0.0

        # Bandwidth-limited tok/s for slowest stage
        # Each stage processes its share of weights per token
        slowest_weight_gb = max(d.weight_memory_gb for d in devices)
        slowest_bytes = slowest_weight_gb * BYTES_PER_GB
        base_tok_s = (board.memory_bandwidth_gbps * BYTES_PER_GB) / slowest_bytes if slowest_bytes > 0 else 0.0
        est_tok_s = base_tok_s * (1 - bubble)

        # Inter-device bandwidth requirement at est_tok_s
        inter_bw_gbps = (inter_bytes * est_tok_s) / BYTES_PER_GB if est_tok_s > 0 else 0.0

        details: list[str] = []
        if feasible:
            details.append(f"{n_devices}-way pipeline split across identical {board.name} boards.")
            details.append(f"Pipeline bubble overhead: {bubble * 100:.1f}%.")
        else:
            over_devices = [d for d in devices if d.total_memory_gb > board.memory_gb]
            details.append(
                f"{n_devices}-way split does not fit: "
                f"{len(over_devices)} device(s) exceed {board.memory_gb:.1f} GB capacity."
            )

        plans.append(DecompositionPlan(
            strategy="pipeline_parallel",
            devices=devices,
            total_devices=n_devices,
            inter_device_transfer_bytes_per_token=inter_bytes,
            inter_device_bandwidth_required_gbps=inter_bw_gbps,
            estimated_tok_per_sec=est_tok_s if feasible else 0.0,
            pipeline_bubble_fraction=bubble,
            feasible=feasible,
            details=details,
        ))

    return plans


def plan_kv_offload(
    spec: ModelSpec,
    board: BoardSpec,
) -> DecompositionPlan | None:
    """Check if weights fit on device with KV cache offloaded to host.

    If the model does not fit because of KV cache, offloading KV to host
    memory may be viable. The cost is host link bandwidth for KV streaming.
    """
    mem = analyze_memory(spec)
    weight_gb = mem.weight_gb
    act_gb = mem.activation_buffer_gb
    on_device_gb = weight_gb + act_gb

    if on_device_gb > board.memory_gb:
        return None  # weights alone don't fit

    # KV offloaded: use kv_on_accelerator=False analysis
    offload_spec = replace(spec, kv_on_accelerator=False)
    io = analyze_io(offload_spec)
    bw = analyze_bandwidth(offload_spec)

    # Bandwidth-limited tok/s (device memory bandwidth)
    bw_bytes = board.memory_bandwidth_gbps * BYTES_PER_GB
    weight_bytes = mem.weight_bytes
    base_tok_s = bw_bytes / weight_bytes if weight_bytes > 0 else 0.0

    # Host link limited tok/s
    link_bytes_per_sec = board.host_link_bandwidth_gbps * BYTES_PER_GB
    kv_bytes_per_token = io.kv_sync_bytes_per_token
    link_tok_s = link_bytes_per_sec / kv_bytes_per_token if kv_bytes_per_token > 0 else float("inf")

    est_tok_s = min(base_tok_s, link_tok_s)

    feasible = on_device_gb <= board.memory_gb
    details: list[str] = [
        f"Weights + activations: {on_device_gb:.2f} GB on {board.name} ({board.memory_gb:.1f} GB).",
        f"KV cache offloaded to host memory.",
        f"Device bandwidth-limited: {base_tok_s:.1f} tok/s.",
        f"Host link-limited: {link_tok_s:.1f} tok/s (KV sync: {kv_bytes_per_token:,} bytes/token).",
        f"Effective: {est_tok_s:.1f} tok/s (limited by {'host link' if link_tok_s < base_tok_s else 'device bandwidth'}).",
    ]

    return DecompositionPlan(
        strategy="kv_offload",
        devices=[DeviceAssignment(
            device_name=board.name,
            layer_start=0,
            layer_end=spec.num_layers,
            weight_memory_gb=weight_gb,
            kv_cache_memory_gb=0.0,  # KV is on host
            total_memory_gb=on_device_gb,
            memory_utilization=on_device_gb / board.memory_gb if board.memory_gb > 0 else 0.0,
        )],
        total_devices=1,
        inter_device_transfer_bytes_per_token=kv_bytes_per_token,
        inter_device_bandwidth_required_gbps=(kv_bytes_per_token * est_tok_s) / BYTES_PER_GB,
        estimated_tok_per_sec=est_tok_s,
        pipeline_bubble_fraction=0.0,
        feasible=feasible,
        details=details,
    )


def plan_decomposition(
    spec: ModelSpec,
    board: BoardSpec,
) -> DecompositionResult:
    """Run all decomposition strategies and return the best plan.

    1. Check if the model fits on a single device.
    2. Try KV offload (1 device, KV on host).
    3. Try pipeline parallel (2..N devices).
    4. Sort feasible plans by estimated tok/s, pick best.
    """
    # Check single-device feasibility
    mem = analyze_memory(spec)
    bw = analyze_bandwidth(spec)
    compute = analyze_compute(spec, 0)
    io = analyze_io(spec)
    verdict = render_verdict(
        spec, mem, bw, compute, io,
        target_memory_gb=board.memory_gb,
        target_bandwidth_gbps=board.memory_bandwidth_gbps,
        target_link_bandwidth_gbps=board.host_link_bandwidth_gbps,
    )

    single_fits = verdict.verdict != FeasibilityVerdict.DOES_NOT_FIT

    plans: list[DecompositionPlan] = []

    # KV offload
    kv_plan = plan_kv_offload(spec, board)
    if kv_plan is not None:
        plans.append(kv_plan)

    # Pipeline parallel
    pp_plans = plan_pipeline_parallel(spec, board)
    plans.extend(pp_plans)

    # Sort feasible plans by estimated tok/s descending
    feasible_plans = [p for p in plans if p.feasible]
    feasible_plans.sort(key=lambda p: -p.estimated_tok_per_sec)
    best = feasible_plans[0] if feasible_plans else None

    return DecompositionResult(
        model_name=spec.name,
        single_device_feasible=single_fits,
        plans=plans,
        best_plan=best,
    )
