import json
import time
from datetime import datetime
import weakref

import pytest
import torch

import benchmarks.machine as machine_module
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
from benchmarks.machine import machine_info, require_clean_tree
from benchmarks.opcount import count_ops
from benchmarks.run import (
    _decode_prediction,
    run_batch_suite,
    run_capacity_suite,
    run_decode_suite,
    run_graph_isolation_suite,
    run_overhead_suite,
    run_prefill_suite,
    run_shapes_suite,
    run_suites,
    validate_benchmark_request,
)
from benchmarks.timing import time_fn, time_interleaved, time_interleaved_staged
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


def test_capturable_step_reads_the_whole_buffer() -> None:
    assert decode_step_bytes(CONFIG, 4, 1, 1, "capturable", capacity=1024) == 570_110_976
    assert decode_step_bytes(CONFIG, 4, 512, 8, "capturable", capacity=1024) == (
        494_613_504 + 8 * 75_497_472
    )
    # Independent of how much of the buffer is written.
    assert decode_step_bytes(CONFIG, 4, 1, 1, "capturable", capacity=1024) == decode_step_bytes(
        CONFIG, 4, 1000, 1, "capturable", capacity=1024
    )
    # The eager caches ignore capacity.
    assert decode_step_bytes(CONFIG, 4, 100, 1, "static", capacity=1024) == decode_step_bytes(
        CONFIG, 4, 100, 1, "static"
    )
    for capacity in (None, 99):
        with pytest.raises(ValueError, match="capacity"):
            decode_step_bytes(CONFIG, 4, 100, 1, "capturable", capacity=capacity)


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


def test_time_interleaved_staged_excludes_setup_and_is_round_robin() -> None:
    order: list[str] = []
    setup_calls = {"A": 0, "B": 0}
    run_calls = {"A": 0, "B": 0}

    def case(name: str):
        def setup() -> str:
            order.append(f"setup-{name}")
            setup_calls[name] += 1
            time.sleep(0.05)
            return name

        def run(state: str) -> None:
            assert state == name
            order.append(f"run-{name}")
            run_calls[name] += 1
            time.sleep(0.005)

        return setup, run

    measurements = time_interleaved_staged(
        {"A": case("A"), "B": case("B")}, "cpu", warmup=1, repeats=2
    )
    assert setup_calls == run_calls == {"A": 3, "B": 3}
    assert all(measurement.median_s < 0.03 for measurement in measurements.values())
    assert order == [
        "setup-A",
        "run-A",
        "setup-B",
        "run-B",
    ] * 3


def test_time_interleaved_staged_releases_state_before_next_setup() -> None:
    previous: list[weakref.ReferenceType[object]] = []

    class State:
        pass

    def setup() -> State:
        assert all(reference() is None for reference in previous)
        state = State()
        previous.append(weakref.ref(state))
        return state

    measurements = time_interleaved_staged(
        {"tracked": (setup, lambda state: None)},
        "cpu",
        warmup=1,
        repeats=2,
    )
    assert len(measurements["tracked"].samples_s) == 2
    assert all(reference() is None for reference in previous)


def test_count_ops_partitions_model_forward_and_counts_linears() -> None:
    config = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)
    model = GPT(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    counts = count_ops(lambda: model(input_ids))
    assert counts["total"] > 0
    assert counts["view"] + counts["compute"] == counts["total"]
    assert counts["by_name"]["aten.addmm"] == 4 * config.n_layer


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
    assert "dirty_outside_results" in info


def test_require_clean_tree_ignores_results_but_rejects_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        machine_module,
        "_git_porcelain",
        lambda *, outside_results=False: "" if outside_results else " M benchmarks/results/x.json",
    )
    require_clean_tree(False)

    monkeypatch.setattr(
        machine_module,
        "_git_porcelain",
        lambda *, outside_results=False: " M benchmarks/run.py",
    )
    with pytest.raises(RuntimeError, match="benchmarks/run.py"):
        require_clean_tree(False)
    require_clean_tree(True)


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
        allow_dirty=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) >= {"machine", "machine_after", "settings", "results"}
    assert payload["settings"]["copy_bytes"] == 1 << 20
    assert payload["settings"]["allow_dirty"] is True
    cpu = payload["results"]["cpu"]
    for name in ("copy_bandwidth_GBs", "read_bandwidth_GBs", "matmul_f32_TFLOPs"):
        assert set(cpu[name]) == {"median", "p25", "p75"}
    assert cpu["ridge_point_flop_per_byte"] > 0
    assert "matmul_bf16_TFLOPs" not in cpu
    assert "launch_overhead_us" not in cpu


def _small_benchmark_model() -> GPT:
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)
    return GPT(config).to(torch.float64).eval()


def _small_calibration() -> dict[str, float]:
    return {
        "read_bandwidth_GBs": 10.0,
        "peak_TFLOPs": 1.0,
        "launch_overhead_us": 1.0,
    }


def test_decode_suite_runs_on_small_cpu_model() -> None:
    result = run_decode_suite(
        _small_benchmark_model(),
        "cpu",
        torch.float64,
        _small_calibration(),
        contexts=(4,),
        steps=2,
        warmup=0,
        repeats=2,
        op_context=8,
    )
    assert set(result) == {"op_context", "opcounts", "records"}
    assert len(result["records"]) == 3
    assert all(record["sequences_match"] is True for record in result["records"])
    for record in result["records"]:
        assert set(record) >= {
            "variant",
            "median_s",
            "roofline_per_token_s",
            "generated_tokens",
        }


def test_prefill_suite_runs_on_small_cpu_model() -> None:
    result = run_prefill_suite(
        _small_benchmark_model(),
        "cpu",
        torch.float64,
        _small_calibration(),
        lengths=(4,),
        warmup=0,
        repeats=2,
    )
    assert len(result["records"]) == 2
    for record in result["records"]:
        assert set(record) >= {
            "variant",
            "median_s",
            "achieved_TFLOPs",
            "peak_fraction",
            "roofline_s",
        }


def test_overhead_suite_runs_on_small_cpu_models() -> None:
    pretrained = _small_benchmark_model()
    structural = _small_benchmark_model()
    result = run_overhead_suite(
        pretrained,
        structural,
        "cpu",
        torch.float64,
        context=4,
        steps=2,
        warmup=0,
        repeats=2,
    )
    assert set(result) == {"structural_config", "records"}
    assert len(result["records"]) == 4
    for record in result["records"]:
        assert set(record) >= {
            "model",
            "variant",
            "total_median_s_per_token",
            "host_median_s_per_token",
            "host_to_total_ratio",
            "ops",
            "structural_cost_per_op_s",
            "gpu_work_delta_vs_structural_s_per_token",
        }
        assert record["ops"]["total"] > 0
        assert record["ops"]["view"] + record["ops"]["compute"] == record["ops"]["total"]


def test_cpu_bfloat16_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="ADR-0009"):
        validate_benchmark_request("decode", "cpu", "bfloat16")


def test_cpu_overhead_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        validate_benchmark_request("overhead", "cpu", "float32")


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def test_decode_suite_on_cpu_has_no_graph_variant() -> None:
    result = run_decode_suite(
        _small_benchmark_model(),
        "cpu",
        torch.float64,
        _small_calibration(),
        contexts=(4,),
        steps=2,
        warmup=0,
        repeats=2,
        op_context=8,
    )
    assert [record["variant"] for record in result["records"]] == ["uncached", "dynamic", "static"]
    assert set(result["opcounts"]) == {"dynamic", "static"}
    for record in result["records"]:
        assert len(record["generated_tokens"][0]) == 2
        assert record["matches_uncached"] is True


def test_graph_prediction_does_not_depend_on_the_step() -> None:
    model = _small_benchmark_model()
    calibration = _small_calibration()
    short = _decode_prediction(model, "graph", 2, 2, 1, 8, calibration, capacity=32)
    long = _decode_prediction(model, "graph", 20, 2, 1, 8, calibration, capacity=32)
    assert short == long
    assert short["mean_flops_per_token"] == flops_per_token(model.config, 32)
    with pytest.raises(ValueError, match="capacity"):
        _decode_prediction(model, "graph", 2, 2, 1, 8, calibration)


def test_capacity_suite_requires_cuda() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        run_capacity_suite(_small_benchmark_model(), "cpu", torch.float64, _small_calibration())
    with pytest.raises(ValueError, match="requires CUDA"):
        validate_benchmark_request("capacity", "cpu", "float32")


def _small_cuda_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    config = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)
    return GPT(config).to(dtype=torch.float64, device="cuda").eval()


@requires_cuda
def test_decode_suite_graph_variant_matches_in_float64() -> None:
    result = run_decode_suite(
        _small_cuda_model(),
        "cuda",
        torch.float64,
        _small_calibration(),
        contexts=(4, 8),
        steps=3,
        warmup=1,
        repeats=2,
        op_context=8,
        graph_floor_s=1e-3,
    )
    graph = [record for record in result["records"] if record["variant"] == "graph"]
    assert len(graph) == 2
    assert set(result["opcounts"]) == {"dynamic", "static", "graph"}
    assert all(record["sequences_match"] is True for record in result["records"])
    for record in graph:
        assert record["cache_capacity"] == 32
        assert record["graph_floor_s"] == 1e-3
        assert record["graph_floor_plus_bandwidth_per_token_s"] > 1e-3
        assert record["launch_limited_per_step_s"] is None
        assert record["median_s"] > 0
        assert len(record["generated_tokens"][0]) == 3


@requires_cuda
def test_batch_suite_graph_variant_uses_the_smallest_capacity() -> None:
    context, steps = 6, 3
    result = run_batch_suite(
        _small_cuda_model(),
        "cuda",
        torch.float64,
        _small_calibration(),
        batches=(1, 2),
        context=context,
        steps=steps,
        warmup=1,
        repeats=2,
    )
    assert result["graph_capacity"] == context + steps
    graph = [record for record in result["records"] if record["variant"] == "graph"]
    assert [record["batch_size"] for record in graph] == [1, 2]
    for record in result["records"]:
        assert record["sequences_match"] is True
        assert set(record) >= {
            "tokens_per_s",
            "roofline_per_step_s",
            "max_memory_allocated",
            "max_memory_reserved",
            "cache_nbytes",
            "cache_used_nbytes",
            "cache_capacity",
        }
    for record in graph:
        assert record["cache_capacity"] == context + steps
        assert record["cache_used_nbytes"] == record["cache_nbytes"]


@requires_cuda
def test_overhead_suite_on_cuda_includes_graph_records() -> None:
    result = run_overhead_suite(
        _small_cuda_model(),
        _small_cuda_model(seed=1),
        "cuda",
        torch.float64,
        context=4,
        steps=2,
        warmup=1,
        repeats=2,
    )
    keys = {(record["model"], record["variant"]) for record in result["records"]}
    assert keys == {
        (model, variant)
        for model in ("pretrained", "structural")
        for variant in ("dynamic", "static", "graph")
    }
    for record in result["records"]:
        assert record["total_median_s_per_token"] > 0
        assert record["host_median_s_per_token"] > 0


@requires_cuda
def test_capacity_suite_records_each_capacity() -> None:
    model = _small_cuda_model()
    batch, context, steps = 2, 4, 3
    result = run_capacity_suite(
        model,
        "cuda",
        torch.float64,
        _small_calibration(),
        batch=batch,
        context=context,
        steps=steps,
        capacities=(8, 16, 32),
        warmup=1,
        repeats=2,
    )
    assert [record["cache_capacity"] for record in result["records"]] == [8, 16, 32]
    per_token = kv_bytes_per_token(model.config, 8, batch)
    for record in result["records"]:
        assert record["sequences_match"] is True
        assert record["cache_used_nbytes"] == per_token * (context + steps)
        assert record["cache_nbytes"] == per_token * record["cache_capacity"]
        assert record["step_bytes"] == decode_step_bytes(
            model.config, 8, context + 1, batch, "capturable", capacity=record["cache_capacity"]
        )
        assert record["roofline_per_token_s"] > 0
    rooflines = [record["roofline_per_token_s"] for record in result["records"]]
    assert rooflines == sorted(rooflines)


@requires_cuda
def test_run_suites_feeds_the_graph_floor_from_overhead_to_decode() -> None:
    results = run_suites(
        _small_cuda_model(),
        _small_cuda_model(seed=1),
        "all",
        "cuda",
        torch.float64,
        _small_calibration(),
        warmup=1,
        repeats=2,
        steps=2,
        decode_contexts=(4,),
        batch_sizes=(1,),
        batch_context=4,
        prefill_lengths=(4,),
        memory_batches=(1,),
        memory_prompt_len=4,
        memory_new_tokens=4,
        memory_sample_every=2,
        op_context=4,
        overhead_context=4,
        capacity_batch=2,
        capacity_context=4,
        capacities=(8, 16),
    )
    assert list(results) == ["overhead", "decode", "batch", "prefill", "memory", "capacity"]
    floor = next(
        record["total_median_s_per_token"]
        for record in results["overhead"]["records"]
        if record["model"] == "structural" and record["variant"] == "graph"
    )
    graph = next(record for record in results["decode"]["records"] if record["variant"] == "graph")
    assert graph["graph_floor_s"] == floor


@pytest.mark.parametrize("suite", ["graph_isolation", "shapes"])
def test_diagnostic_suites_are_rejected_on_cpu(suite: str) -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        validate_benchmark_request(suite, "cpu", "float32")


def test_diagnostic_suites_require_cuda() -> None:
    model = _small_benchmark_model()
    with pytest.raises(ValueError, match="requires CUDA"):
        run_graph_isolation_suite(model, model, "cpu", torch.float64)
    with pytest.raises(ValueError, match="requires CUDA"):
        run_shapes_suite("cpu", torch.float64, {"small": model.config})


@requires_cuda
def test_graph_isolation_suite_repeats_the_same_work_in_every_mode() -> None:
    contexts = (4, 12)
    result = run_graph_isolation_suite(
        _small_cuda_model(),
        _small_cuda_model(seed=1),
        "cuda",
        torch.float64,
        contexts=contexts,
        steps=3,
        warmup=1,
        repeats=3,
        heat_seconds=0.0,
        idle_s=0.0,
    )
    modes = (
        "isolated_ascending",
        "isolated_descending",
        "after_idle",
        "interleaved_with_eager",
    )
    assert result["modes"] == list(modes)
    assert [block["mode"] for block in result["blocks"]] == list(modes)
    assert result["blocks"][1]["order"] == [12, 4]
    for block in result["blocks"]:
        assert block["heat_up_started_at"] <= block["started_at"] <= block["finished_at"]
    assert len(result["records"]) == len(modes) * 2 * len(contexts)
    tokens: dict[tuple[str, int], list[list[int]]] = {}
    for record in result["records"]:
        assert set(record) >= {
            "model",
            "mode",
            "context_len",
            "cache_capacity",
            "median_s",
            "p25_s",
            "p75_s",
            "repeats_identical",
            "nvidia_smi_before_block",
            "nvidia_smi_after_block",
            "started_at",
            "finished_at",
        }
        started = datetime.fromisoformat(record["started_at"])
        assert started.tzinfo is not None
        assert started <= datetime.fromisoformat(record["finished_at"])
        assert record["median_s"] > 0
        assert record["repeats_identical"] is True
        tokens.setdefault((record["model"], record["context_len"]), []).append(
            record["generated_tokens"]
        )
    # The same (model, T) replays the same work in every mode.
    for generated in tokens.values():
        assert all(entry == generated[0] for entry in generated)


@requires_cuda
def test_shapes_suite_records_every_shape_batch_and_size() -> None:
    config = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)
    result = run_shapes_suite(
        "cuda",
        torch.float32,
        {"small": config},
        batches=(1, 4),
        calls=4,
        sizes_mib=(1, 2),
        warmup=1,
        repeats=2,
        heat_seconds=0.0,
    )
    layers = {record["layer"] for record in result["records"]}
    assert layers == {"c_attn", "attn.c_proj", "c_fc", "mlp.c_proj", "lm_head"}
    assert len(result["records"]) == 5 * 2
    for record in result["records"]:
        fan_in, fan_out = record["in_features"], record["out_features"]
        bias = fan_out if record["bias"] else 0
        assert record["weight_bytes"] == (fan_in * fan_out + bias) * 4
        per_call = record["median_s"]
        assert record["effective_bandwidth_GBs"] == pytest.approx(
            record["weight_bytes"] / per_call / 1e9
        )
        assert record["effective_TFLOPs"] == pytest.approx(
            2 * record["batch_size"] * fan_in * fan_out / per_call / 1e12
        )
    lm_head = [record for record in result["records"] if record["layer"] == "lm_head"]
    assert all(record["bias"] is False and record["out_features"] == 100 for record in lm_head)
    assert [record["size_mib"] for record in result["bandwidth_records"]] == [1, 2]
    for record in result["bandwidth_records"]:
        assert record["nbytes"] == record["size_mib"] * 2**20
        assert record["effective_bandwidth_GBs"] > 0
    assert set(result["blocks"]) == {"shapes", "bandwidth"}
