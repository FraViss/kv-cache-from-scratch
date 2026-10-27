"""Measure memory bandwidth, compute throughput and CUDA launch overhead.

Units are decimal: GB/s is 1e9 bytes/s, TFLOP/s is 1e12 FLOP/s. The copy and
read buffers are 1 GiB (2**30 bytes) of float32 unless a caller passes a
smaller ``copy_bytes`` (the tests do). Rates are the quartiles of the per-sample
rates, not the rate of the median time.

Run: ``uv run python -m benchmarks.calibrate --device {cpu,cuda,all}
--out benchmarks/results/calibration.json``
"""

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

from benchmarks.machine import (
    machine_info,
    nvidia_smi_query,
    require_clean_tree,
    set_benchmark_numerics,
)
from benchmarks.timing import Measurement, time_fn

_GIB = 1 << 30


def _summary(measurement: Measurement, value_of: Callable[[float], float]) -> dict[str, float]:
    values = Measurement(tuple(value_of(sample) for sample in measurement.samples_s))
    return {"median": values.median_s, "p25": values.p25_s, "p75": values.p75_s}


def _empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _calibrate_device(
    device: torch.device,
    warmup: int,
    repeats: int,
    copy_bytes: int,
    matmul_n: int,
    launch_calls: int,
) -> dict[str, object]:
    if copy_bytes % 4 != 0:
        raise ValueError(f"copy_bytes must be a multiple of 4, got {copy_bytes}")
    n_elem = copy_bytes // 4
    src = torch.empty(n_elem, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    src.fill_(1)
    copy_time = time_fn(lambda: dst.copy_(src), device, warmup, repeats)
    copy_bytes_moved = 2 * src.nbytes
    copy = _summary(copy_time, lambda t: copy_bytes_moved / t / 1e9)
    del dst
    _empty_cache(device)

    read_time = time_fn(lambda: src.sum(), device, warmup, repeats)
    read_bytes = src.nbytes
    read = _summary(read_time, lambda t: read_bytes / t / 1e9)
    del src
    _empty_cache(device)

    left = torch.randn(matmul_n, matmul_n, device=device, dtype=torch.float32)
    right = torch.randn(matmul_n, matmul_n, device=device, dtype=torch.float32)
    flops = 2 * matmul_n**3
    f32_time = time_fn(lambda: left @ right, device, warmup, repeats)
    matmul_f32 = _summary(f32_time, lambda t: flops / t / 1e12)
    del left, right
    _empty_cache(device)

    result: dict[str, object] = {
        "copy_bandwidth_GBs": copy,
        "read_bandwidth_GBs": read,
        "matmul_f32_TFLOPs": matmul_f32,
        "ridge_point_flop_per_byte": (matmul_f32["median"] * 1e12) / (read["median"] * 1e9),
    }

    if device.type != "cuda":
        return result

    left_bf = torch.randn(matmul_n, matmul_n, device=device, dtype=torch.bfloat16)
    right_bf = torch.randn(matmul_n, matmul_n, device=device, dtype=torch.bfloat16)
    bf16_time = time_fn(lambda: left_bf @ right_bf, device, warmup, repeats)
    result["matmul_bf16_TFLOPs"] = _summary(bf16_time, lambda t: flops / t / 1e12)
    del left_bf, right_bf
    _empty_cache(device)

    scalar = torch.zeros(1, device=device)
    block = time_fn(
        lambda: [scalar.add_(1) for _ in range(launch_calls)],
        device,
        warmup,
        repeats,
    )
    result["launch_overhead_us"] = _summary(block, lambda t: t / launch_calls * 1e6)
    del scalar
    _empty_cache(device)
    return result


def _devices(choice: str) -> list[torch.device]:
    if choice == "all":
        wanted = ["cpu", "cuda"]
    else:
        wanted = [choice]
    devices = []
    for name in wanted:
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        devices.append(torch.device(name))
    return devices


def calibrate(
    devices: list[str] | str,
    out: Path,
    warmup: int = 5,
    repeats: int = 20,
    *,
    copy_bytes: int = _GIB,
    matmul_n: int = 4096,
    launch_calls: int = 1000,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Calibrate each device and write the JSON record to ``out``.

    ``copy_bytes``, ``matmul_n`` and ``launch_calls`` default to the full
    benchmark sizes. Tests pass smaller ones.
    """
    require_clean_tree(allow_dirty)
    set_benchmark_numerics()
    if isinstance(devices, str):
        selected = _devices(devices)
    else:
        selected = []
        for name in devices:
            selected.extend(_devices(name))

    payload: dict[str, object] = {
        "units": {
            "bandwidth": "GB/s = 1e9 bytes/s",
            "compute": "TFLOP/s = 1e12 FLOP/s",
            "launch_overhead": "microseconds per kernel launch",
            "ridge_point": "FLOP/byte = float32 peak FLOP/s / read bandwidth in bytes/s",
            "time_base": "quartiles are taken on the per-sample rates, not on the times",
        },
        "machine": machine_info(),
        "settings": {
            "warmup": warmup,
            "repeats": repeats,
            "copy_bytes": copy_bytes,
            "matmul_n": matmul_n,
            "launch_calls": launch_calls,
            "allow_dirty": allow_dirty,
            "copy_dtype": "float32",
            "read_op": "sum",
            "matmul_flop": "2 * n**3",
        },
        "results": {
            device.type: _calibrate_device(
                device, warmup, repeats, copy_bytes, matmul_n, launch_calls
            )
            for device in selected
        },
        "machine_after": nvidia_smi_query(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_table(payload)
    return payload


def _print_table(payload: dict[str, object]) -> None:
    machine = payload["machine"]
    after = payload["machine_after"]
    assert isinstance(machine, dict)
    smi = machine.get("nvidia_smi")
    print(f"commit {machine['git_commit']} dirty={machine['dirty']}")
    print(
        "float32_matmul_precision="
        f"{machine['float32_matmul_precision']} "
        f"allow_tf32_matmul={machine['allow_tf32_matmul']} "
        f"allow_tf32_cudnn={machine['allow_tf32_cudnn']}"
    )
    if isinstance(smi, dict) and isinstance(after, dict):
        print(
            "gpu before  "
            f"temp={smi['temperature.gpu']} C  "
            f"sm={smi['clocks.sm']} MHz  max_sm={smi['clocks.max.sm']} MHz  "
            f"power={smi['power.draw']}/{smi['power.limit']} W"
        )
        print(
            "gpu after   "
            f"temp={after['temperature.gpu']} C  "
            f"sm={after['clocks.sm']} MHz  max_sm={after['clocks.max.sm']} MHz  "
            f"power={after['power.draw']}/{after['power.limit']} W"
        )
    results = payload["results"]
    assert isinstance(results, dict)
    print(f"{'device':<8} {'measure':<24} {'median':>12} {'p25':>12} {'p75':>12}")
    for device, measures in results.items():
        assert isinstance(measures, dict)
        for name, summary in measures.items():
            if not isinstance(summary, dict):
                print(f"{device:<8} {name:<24} {summary:12.4g}")
                continue
            print(
                f"{device:<8} {name:<24} {summary['median']:12.4g} "
                f"{summary['p25']:12.4g} {summary['p75']:12.4g}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "all"), required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/calibration.json"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    calibrate(
        args.device,
        args.out,
        warmup=args.warmup,
        repeats=args.repeats,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()
