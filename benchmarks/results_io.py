"""Read committed benchmark JSON and derive the quantities that figures and summary share.

Nothing here measures anything. Every number comes from a JSON file in
``benchmarks/results/`` or from the analytic cost model in ``benchmarks.costs``.
"""

import json
from pathlib import Path
from typing import Any

from benchmarks.costs import (
    decode_step_bytes,
    flops_per_token,
    kv_bytes_per_token,
    weight_bytes,
)
from kvcache import GPT2Config

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DTYPES = ("float32", "bfloat16")
ITEMSIZE = {"float32": 4, "bfloat16": 2}
GPT2_SMALL = GPT2Config()


def load(results_dir: Path, name: str) -> dict[str, Any]:
    return json.loads((results_dir / name).read_text(encoding="utf-8"))


def gpu_file(dtype: str) -> str:
    return f"all_cuda_{dtype}.json"


def suite_records(payload: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    return payload["suites"][suite]["records"]


def commit(payload: dict[str, Any]) -> str:
    return str(payload["machine"]["git_commit"])[:7]


def device_name(payload: dict[str, Any]) -> str:
    gpu = payload["machine"].get("gpu")
    if payload.get("settings", {}).get("device") == "cpu" or not gpu:
        return str(payload["machine"]["cpu"])
    return str(gpu["name"])


def calibration_values(results_dir: Path) -> dict[str, Any]:
    payload = load(results_dir, "calibration.json")
    results = payload["results"]
    cuda = results["cuda"]
    cpu = results["cpu"]
    return {
        "payload": payload,
        "cuda_read_GBs": cuda["read_bandwidth_GBs"]["median"],
        "cpu_read_GBs": cpu["read_bandwidth_GBs"]["median"],
        "peak_TFLOPs": {
            "float32": cuda["matmul_f32_TFLOPs"]["median"],
            "bfloat16": cuda["matmul_bf16_TFLOPs"]["median"],
        },
        "cpu_peak_TFLOPs": cpu["matmul_f32_TFLOPs"]["median"],
    }


def _decode_record(payload: dict[str, Any], variant: str, context: int) -> dict[str, Any]:
    for record in suite_records(payload, "decode"):
        if record["variant"] == variant and record["context_len"] == context:
            return record
    raise KeyError(f"no decode record for {variant} at T={context}")


def _overhead_time(payload: dict[str, Any], model: str, variant: str) -> float:
    for record in suite_records(payload, "overhead"):
        if record["model"] == model and record["variant"] == variant:
            return record["total_median_s_per_token"]
    raise KeyError(f"no overhead record for {model}:{variant}")


def time_breakdown(results_dir: Path, dtype: str, context: int = 512) -> dict[str, Any]:
    """Where a B=1 decode step goes, at one context length, in seconds per token.

    Eager static: the CPU overhead is the structural model in eager static mode; the
    rest is the measured step minus that overhead (it can be negative).
    Graph: the kernel floor is the structural model in the graph; weights and cache
    are their bytes over the calibrated read bandwidth (the graph reads the whole
    cache buffer); the residual is the measured step minus those three.
    """
    payload = load(results_dir, gpu_file(dtype))
    calibration = calibration_values(results_dir)
    bandwidth = calibration["cuda_read_GBs"] * 1e9
    itemsize = ITEMSIZE[dtype]
    static = _decode_record(payload, "static", context)
    graph = _decode_record(payload, "graph", context)
    capacity = graph["cache_capacity"]
    eager_overhead = _overhead_time(payload, "structural", "static")
    floor = _overhead_time(payload, "structural", "graph")
    weights = weight_bytes(GPT2_SMALL, itemsize) / bandwidth
    cache = kv_bytes_per_token(GPT2_SMALL, itemsize) * capacity / bandwidth
    return {
        "dtype": dtype,
        "context_len": context,
        "read_bandwidth_GBs": calibration["cuda_read_GBs"],
        "eager": {
            "measured": static["median_s"],
            "cpu_overhead": eager_overhead,
            "rest": static["median_s"] - eager_overhead,
        },
        "graph": {
            "measured": graph["median_s"],
            "kernel_floor": floor,
            "weights": weights,
            "cache": cache,
            "cache_capacity": capacity,
            "residual": graph["median_s"] - floor - weights - cache,
        },
    }


def roofline_points(results_dir: Path) -> list[dict[str, Any]]:
    """Arithmetic intensity and achieved FLOP/s of measured GPU runs.

    Decode points come from the batch suite (T=512), eager static and graph, at
    B = 1, 8 and 32. FLOPs and bytes are the cost-model means over the timed steps
    (the graph reads and attends over its whole buffer). Prefill points are the
    uncached prefill at every length, with weight bytes as traffic.
    """
    points: list[dict[str, Any]] = []
    for dtype in DTYPES:
        payload = load(results_dir, gpu_file(dtype))
        itemsize = ITEMSIZE[dtype]
        for record in suite_records(payload, "batch"):
            if record.get("out_of_memory") or record["variant"] not in ("static", "graph"):
                continue
            batch = record["batch_size"]
            if batch not in (1, 8, 32):
                continue
            context, steps = record["context_len"], record["steps"]
            flops = bytes_ = 0.0
            for index in range(steps):
                length = context + 1 + index
                if record["variant"] == "graph":
                    capacity = record["cache_capacity"]
                    flops += batch * flops_per_token(GPT2_SMALL, capacity)
                    bytes_ += decode_step_bytes(
                        GPT2_SMALL, itemsize, length, batch, "capturable", capacity=capacity
                    )
                else:
                    flops += batch * flops_per_token(GPT2_SMALL, length)
                    bytes_ += decode_step_bytes(GPT2_SMALL, itemsize, length, batch, "static")
            flops /= steps
            bytes_ /= steps
            step_s = record["median_s"] / steps
            points.append(
                {
                    "dtype": dtype,
                    "kind": f"decode {record['variant']}",
                    "label": f"B={batch}",
                    "intensity": flops / bytes_,
                    "achieved_TFLOPs": flops / step_s / 1e12,
                }
            )
        for record in suite_records(payload, "prefill"):
            if record["variant"] != "uncached":
                continue
            points.append(
                {
                    "dtype": dtype,
                    "kind": "prefill",
                    "label": f"T={record['prompt_len']}",
                    "intensity": record["flops"] / weight_bytes(GPT2_SMALL, itemsize),
                    "achieved_TFLOPs": record["achieved_TFLOPs"],
                }
            )
    return points
