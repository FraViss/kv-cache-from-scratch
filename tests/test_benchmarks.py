import json
import time

import pytest
import torch

from benchmarks.calibrate import calibrate
from benchmarks.costs import (
    decode_step_bytes,
    flops_per_token,
    kv_bytes_per_token,
    matmul_weights,
    num_params,
    prefill_flops,
    roofline_time_s,
    weight_bytes,
)
from benchmarks.machine import machine_info
from benchmarks.timing import time_fn, time_interleaved
from kvcache import GPT, GPT2Config

CONFIG = GPT2Config()


def test_cost_model_matches_the_closed_form() -> None:
    assert matmul_weights(CONFIG) == 123_532_032
    assert flops_per_token(CONFIG, 0) == 247_064_064
    assert num_params(CONFIG) == 124_439_808
    with torch.device("meta"):
        counted = sum(p.numel() for p in GPT(GPT2Config()).parameters())
    assert counted == 124_439_808
    assert weight_bytes(CONFIG, 4) == 494_613_504
    assert kv_bytes_per_token(CONFIG, 4) == 73_728
    context_len = 100
    gap = decode_step_bytes(CONFIG, 4, context_len, 1, "dynamic") - decode_step_bytes(
        CONFIG, 4, context_len, 1, "static"
    )
    assert gap == 2 * 73_728 * (context_len - 1)
    assert prefill_flops(CONFIG, 1) == flops_per_token(CONFIG, 1)


def test_roofline_takes_the_slower_roof() -> None:
    assert roofline_time_s(100, 10, peak_flops=50, bandwidth=100) == 2
    assert roofline_time_s(10, 100, peak_flops=50, bandwidth=10) == 10


def test_time_fn_discards_warmup_and_keeps_repeats() -> None:
    calls = {"n": 0}

    def count() -> None:
        calls["n"] += 1

    measurement = time_fn(count, "cpu", warmup=2, repeats=3)
    assert calls["n"] == 5
    assert len(measurement.samples_s) == 3


def test_time_fn_on_cpu_sees_a_short_sleep() -> None:
    measurement = time_fn(lambda: time.sleep(0.01), "cpu", warmup=1, repeats=3)
    assert 0.009 <= measurement.median_s <= 0.1


def test_time_interleaved_is_round_robin_after_warmup() -> None:
    order: list[str] = []

    def record(name: str) -> None:
        order.append(name)

    fns = {
        "A": lambda: record("A"),
        "B": lambda: record("B"),
        "C": lambda: record("C"),
    }
    time_interleaved(fns, "cpu", warmup=2, repeats=3)
    assert order[:6] == ["A", "A", "B", "B", "C", "C"]
    assert order[6:] == ["A", "B", "C", "A", "B", "C", "A", "B", "C"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_time_fn_on_cuda_returns_positive_samples() -> None:
    left = torch.randn(1024, 1024, device="cuda")
    right = torch.randn(1024, 1024, device="cuda")
    measurement = time_fn(lambda: left @ right, "cuda", warmup=1, repeats=3)
    assert all(sample > 0 for sample in measurement.samples_s)


def test_machine_info_is_json_and_records_numerics() -> None:
    info = machine_info()
    json.dumps(info)
    assert "float32_matmul_precision" in info
    assert "allow_tf32_matmul" in info
    assert "allow_tf32_cudnn" in info


def test_calibrate_cpu_writes_the_expected_keys(tmp_path) -> None:
    out = tmp_path / "calibration.json"
    calibrate(
        ["cpu"],
        out,
        warmup=1,
        repeats=2,
        copy_bytes=1 << 20,
        matmul_n=64,
        launch_calls=4,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) >= {"machine", "machine_after", "settings", "results"}
    assert payload["settings"]["copy_bytes"] == 1 << 20
    cpu = payload["results"]["cpu"]
    for name in ("copy_bandwidth_GBs", "read_bandwidth_GBs", "matmul_f32_TFLOPs"):
        assert set(cpu[name]) == {"median", "p25", "p75"}
    assert cpu["ridge_point_flop_per_byte"] > 0
    assert "matmul_bf16_TFLOPs" not in cpu
    assert "launch_overhead_us" not in cpu
