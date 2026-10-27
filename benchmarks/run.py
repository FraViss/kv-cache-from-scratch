"""Run generation benchmark suites and save reproducible JSON records.

Examples:
    uv run python -m benchmarks.run --suite all --device cuda --dtype float32
    uv run python -m benchmarks.run --suite decode --device cpu --dtype float32
"""

import argparse
import gc
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from benchmarks.costs import (
    decode_step_bytes,
    flops_per_token,
    prefill_flops,
    roofline_time_s,
    weight_bytes,
)
from benchmarks.gpu import heat_up, nvidia_smi_snapshot
from benchmarks.machine import machine_info, require_clean_tree, set_benchmark_numerics
from benchmarks.opcount import count_ops
from benchmarks.timing import Measurement, time_interleaved_staged
from kvcache import GPT, GPT2Config, load_pretrained_gpt2
from kvcache.cache import DynamicKVCache, KVCache, StaticKVCache

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / "benchmarks" / "results" / "calibration.json"
RESULTS_DIR = ROOT / "benchmarks" / "results"
SEED = 20261027


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _measurement(measurement: Measurement, divisor: int = 1) -> dict[str, float]:
    return {
        "median_s": measurement.median_s / divisor,
        "p25_s": measurement.p25_s / divisor,
        "p75_s": measurement.p75_s / divisor,
    }


def _prompt(model: GPT, batch: int, length: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(SEED + batch * 10_000 + length)
    return torch.randint(
        0,
        model.config.vocab_size,
        (batch, length),
        generator=generator,
        device=device,
    )


def _static_cache(model: GPT, batch: int, dtype: torch.dtype, device: torch.device) -> StaticKVCache:
    return StaticKVCache.from_config(
        model.config,
        batch_size=batch,
        max_seq_len=model.config.n_positions,
        dtype=dtype,
        device=device,
    )


def _clear_cuda_cache(device: torch.device) -> None:
    """Release cached allocator blocks between variants, outside timed regions."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _new_cache(
    variant: str, model: GPT, batch: int, dtype: torch.dtype, device: torch.device
) -> KVCache:
    if variant == "dynamic":
        return DynamicKVCache(model.config.n_layer)
    if variant == "static":
        return _static_cache(model, batch, dtype, device)
    raise ValueError(f"unknown cache variant {variant!r}")


def _generate_uncached(model: GPT, prompt: Tensor, steps: int) -> Tensor:
    sequence = prompt
    with torch.inference_mode():
        for _ in range(steps):
            token = model(sequence)[:, -1, :].argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, token), dim=1)
    return sequence[:, prompt.shape[1] :]


def _setup_cached(
    model: GPT,
    prompt: Tensor,
    variant: str,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[KVCache, Tensor]:
    cache = _new_cache(variant, model, prompt.shape[0], dtype, device)
    with torch.inference_mode():
        first = model(prompt, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
    return cache, first


def _run_cached(model: GPT, state: tuple[KVCache, Tensor], steps: int) -> Tensor:
    cache, token = state
    generated = [token]
    with torch.inference_mode():
        for _ in range(steps - 1):
            token = model(token, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(token)
    return torch.cat(generated, dim=1)


def _calibration_values(
    calibration: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, float]:
    values = calibration["results"][device.type]
    peak_name = "matmul_bf16_TFLOPs" if dtype == torch.bfloat16 else "matmul_f32_TFLOPs"
    return {
        "read_bandwidth_GBs": values["read_bandwidth_GBs"]["median"],
        "peak_TFLOPs": values[peak_name]["median"],
        "launch_overhead_us": values.get("launch_overhead_us", {}).get("median", 0.0),
    }


def _roofline(
    flops: float, bytes_: float, calibration: dict[str, float]
) -> float:
    return roofline_time_s(
        flops,
        bytes_,
        calibration["peak_TFLOPs"] * 1e12,
        calibration["read_bandwidth_GBs"] * 1e9,
    )


def _decode_prediction(
    model: GPT,
    variant: str,
    context: int,
    steps: int,
    batch: int,
    itemsize: int,
    calibration: dict[str, float],
) -> dict[str, float]:
    times = []
    for index in range(steps):
        length = context + index
        if variant == "uncached":
            flops = batch * prefill_flops(model.config, length)
            bytes_ = weight_bytes(model.config, itemsize)
        else:
            flops = batch * flops_per_token(model.config, length)
            bytes_ = decode_step_bytes(
                model.config, itemsize, length, batch, variant
            )
        times.append(_roofline(flops, bytes_, calibration))
    return {
        "roofline_per_token_s": sum(times) / len(times),
        "mean_flops_per_token": sum(
            (
                batch * prefill_flops(model.config, context + index)
                if variant == "uncached"
                else batch * flops_per_token(model.config, context + index)
            )
            for index in range(steps)
        )
        / steps,
    }


def _decode_opcounts(
    model: GPT,
    dtype: torch.dtype,
    device: torch.device,
    context: int,
    launch_overhead_us: float,
) -> dict[str, Any]:
    prompt = _prompt(model, 1, context, device)
    result: dict[str, Any] = {}
    for variant in ("dynamic", "static"):
        cache = _new_cache(variant, model, 1, dtype, device)
        with torch.inference_mode():
            token = model(prompt, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
            counts = count_ops(lambda: model(token, cache=cache))
        counts["launch_limited_s"] = counts["compute"] * launch_overhead_us / 1e6
        result[variant] = counts
    return result


def run_decode_suite(
    model: GPT,
    device: str | torch.device,
    dtype: torch.dtype,
    calibration: dict[str, float],
    *,
    contexts: Sequence[int] | None = None,
    steps: int = 16,
    warmup: int = 3,
    repeats: int = 10,
    op_context: int = 512,
) -> dict[str, Any]:
    """Benchmark uncached, dynamic and static greedy decoding at B=1."""
    resolved = torch.device(device)
    if contexts is None:
        contexts = (64, 128, 256, 512, 768, 1000) if resolved.type == "cuda" else (
            64,
            256,
            512,
            1000,
        )
    records: list[dict[str, Any]] = []
    itemsize = _itemsize(dtype)
    op_len = min(op_context, model.config.n_positions - 1)
    opcounts = _decode_opcounts(
        model, dtype, resolved, op_len, calibration["launch_overhead_us"]
    )
    for context in contexts:
        if context + steps > model.config.n_positions:
            raise ValueError(
                f"decode context {context} + steps {steps} exceeds "
                f"n_positions {model.config.n_positions}"
            )
        prompt = _prompt(model, 1, context, resolved)
        outputs: dict[str, Tensor] = {}

        def uncached_setup() -> Tensor:
            _clear_cuda_cache(resolved)
            return prompt

        def uncached_run(state: Tensor) -> Tensor:
            outputs["uncached"] = _generate_uncached(model, state, steps)
            return outputs["uncached"]

        cases: dict[str, tuple[Callable[[], object], Callable[[object], object]]] = {
            "uncached": (uncached_setup, uncached_run),
        }
        for variant in ("dynamic", "static"):
            def setup(variant: str = variant) -> tuple[KVCache, Tensor]:
                _clear_cuda_cache(resolved)
                return _setup_cached(model, prompt, variant, dtype, resolved)

            def run(state: object, variant: str = variant) -> Tensor:
                assert isinstance(state, tuple)
                outputs[variant] = _run_cached(model, state, steps)
                return outputs[variant]

            cases[variant] = (setup, run)

        timings = time_interleaved_staged(cases, resolved, warmup, repeats)
        sequences_match = all(
            torch.equal(outputs["uncached"], outputs[name]) for name in ("dynamic", "static")
        )
        for variant, measurement in timings.items():
            prediction = _decode_prediction(
                model, variant, context, steps, 1, itemsize, calibration
            )
            launch_limited = (
                None
                if variant == "uncached"
                else opcounts[variant]["launch_limited_s"]
            )
            records.append(
                {
                    "variant": variant,
                    "context_len": context,
                    "steps": steps,
                    **_measurement(measurement, steps),
                    **prediction,
                    "launch_limited_per_step_s": launch_limited,
                    "generated_tokens": outputs[variant].cpu().tolist(),
                    "sequences_match": sequences_match,
                }
            )
        if resolved.type == "cuda":
            torch.cuda.empty_cache()
    return {"op_context": op_len, "opcounts": opcounts, "records": records}


def run_batch_suite(
    model: GPT,
    device: str | torch.device,
    dtype: torch.dtype,
    calibration: dict[str, float],
    *,
    batches: Sequence[int] = (1, 2, 4, 8, 16, 32),
    context: int = 512,
    steps: int = 16,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, Any]:
    """Benchmark cached CUDA decode throughput and peak memory by batch."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("batch suite requires CUDA")
    records: list[dict[str, Any]] = []
    itemsize = _itemsize(dtype)
    for batch in batches:
        prompt = _prompt(model, batch, context, resolved)
        outputs: dict[str, Tensor] = {}
        memory: dict[str, dict[str, int]] = {}
        cases: dict[str, tuple[Callable[[], object], Callable[[object], object]]] = {}
        for variant in ("dynamic", "static"):
            def setup(variant: str = variant) -> tuple[KVCache, Tensor]:
                _clear_cuda_cache(resolved)
                state = _setup_cached(model, prompt, variant, dtype, resolved)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                return state

            def run(state: object, variant: str = variant) -> Tensor:
                assert isinstance(state, tuple)
                outputs[variant] = _run_cached(model, state, steps)
                cache = state[0]
                memory[variant] = {
                    "max_memory_allocated": torch.cuda.max_memory_allocated(),
                    "max_memory_reserved": torch.cuda.max_memory_reserved(),
                    "cache_nbytes": cache.nbytes,
                    "cache_used_nbytes": cache.used_nbytes,
                }
                return outputs[variant]

            cases[variant] = (setup, run)
        try:
            timings = time_interleaved_staged(cases, resolved, warmup, repeats)
        except torch.OutOfMemoryError as error:
            records.append(
                {
                    "batch_size": batch,
                    "out_of_memory": True,
                    "error": str(error),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()
            continue
        for variant, measurement in timings.items():
            prediction = _decode_prediction(
                model, variant, context, steps, batch, itemsize, calibration
            )
            records.append(
                {
                    "variant": variant,
                    "batch_size": batch,
                    "context_len": context,
                    "steps": steps,
                    "out_of_memory": False,
                    **_measurement(measurement),
                    "tokens_per_s": batch * steps / measurement.median_s,
                    "roofline_per_step_s": prediction["roofline_per_token_s"],
                    **memory[variant],
                }
            )
        del prompt
        gc.collect()
        torch.cuda.empty_cache()
    return {"records": records}


def run_prefill_suite(
    model: GPT,
    device: str | torch.device,
    dtype: torch.dtype,
    calibration: dict[str, float],
    *,
    lengths: Sequence[int] = (64, 128, 256, 512, 1024),
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, Any]:
    """Benchmark one full prompt forward with and without a static cache."""
    resolved = torch.device(device)
    records: list[dict[str, Any]] = []
    itemsize = _itemsize(dtype)
    for length in lengths:
        prompt = _prompt(model, 1, length, resolved)

        def uncached_setup() -> Tensor:
            _clear_cuda_cache(resolved)
            return prompt

        def static_setup() -> tuple[StaticKVCache, Tensor]:
            _clear_cuda_cache(resolved)
            return _static_cache(model, 1, dtype, resolved), prompt

        cases: dict[str, tuple[Callable[[], object], Callable[[object], object]]] = {
            "uncached": (
                uncached_setup,
                lambda state: model(state),
            ),
            "static": (
                static_setup,
                lambda state: model(state[1], cache=state[0]),
            ),
        }
        with torch.inference_mode():
            timings = time_interleaved_staged(cases, resolved, warmup, repeats)
        flops = prefill_flops(model.config, length)
        predicted = _roofline(
            flops, weight_bytes(model.config, itemsize), calibration
        )
        for variant, measurement in timings.items():
            achieved = flops / measurement.median_s / 1e12
            records.append(
                {
                    "variant": variant,
                    "prompt_len": length,
                    **_measurement(measurement),
                    "flops": flops,
                    "achieved_TFLOPs": achieved,
                    "peak_fraction": achieved / calibration["peak_TFLOPs"],
                    "roofline_s": predicted,
                }
            )
        if resolved.type == "cuda":
            torch.cuda.empty_cache()
    return {"records": records}


def _memory_values(cache: KVCache, generated: int) -> dict[str, int]:
    return {
        "generated_tokens": generated,
        "memory_allocated": torch.cuda.memory_allocated(),
        "memory_reserved": torch.cuda.memory_reserved(),
        "cache_nbytes": cache.nbytes,
        "cache_used_nbytes": cache.used_nbytes,
    }


def run_memory_suite(
    model: GPT,
    device: str | torch.device,
    dtype: torch.dtype,
    *,
    batches: Sequence[int] = (1, 8),
    prompt_len: int = 32,
    new_tokens: int = 992,
    sample_every: int = 32,
) -> dict[str, Any]:
    """Trace CUDA allocator and cache memory through an untimed generation."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("memory suite requires CUDA")
    records: list[dict[str, Any]] = []
    for batch in batches:
        prompt = _prompt(model, batch, prompt_len, resolved)
        for variant in ("dynamic", "static"):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                cache, token = _setup_cached(model, prompt, variant, dtype, resolved)
                snapshots = [_memory_values(cache, 0)]
                with torch.inference_mode():
                    for generated in range(1, new_tokens + 1):
                        if generated > 1:
                            token = model(token, cache=cache)[:, -1, :].argmax(
                                dim=-1, keepdim=True
                            )
                        if generated % sample_every == 0 or generated == new_tokens:
                            torch.cuda.synchronize()
                            snapshots.append(_memory_values(cache, generated))
                records.append(
                    {
                        "variant": variant,
                        "batch_size": batch,
                        "prompt_len": prompt_len,
                        "new_tokens": new_tokens,
                        "out_of_memory": False,
                        "max_memory_allocated": torch.cuda.max_memory_allocated(),
                        "max_memory_reserved": torch.cuda.max_memory_reserved(),
                        "snapshots": snapshots,
                    }
                )
                del cache, token
            except torch.OutOfMemoryError as error:
                records.append(
                    {
                        "variant": variant,
                        "batch_size": batch,
                        "out_of_memory": True,
                        "error": str(error),
                    }
                )
            gc.collect()
            torch.cuda.empty_cache()
        del prompt
    return {"records": records}


def _time_host_interleaved_staged(
    cases: dict[str, tuple[Callable[[], object], Callable[[object], object]]],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, Measurement]:
    """Measure CPU enqueue time, synchronizing only outside the timed region."""
    samples: dict[str, list[float]] = {name: [] for name in cases}

    def invoke(name: str, measured: bool) -> None:
        setup, run = cases[name]
        state = setup()
        try:
            if device.type == "cuda":
                torch.cuda.synchronize()
            if measured:
                started = time.perf_counter_ns()
                run(state)
                samples[name].append((time.perf_counter_ns() - started) / 1e9)
            else:
                run(state)
            # Intentionally outside perf_counter: host time is the time needed
            # to enqueue the work, not the time needed for the GPU to finish it.
            if device.type == "cuda":
                torch.cuda.synchronize()
        finally:
            del state

    for _ in range(warmup):
        for name in cases:
            invoke(name, measured=False)
    for _ in range(repeats):
        for name in cases:
            invoke(name, measured=True)
    return {name: Measurement(tuple(values)) for name, values in samples.items()}


def run_overhead_suite(
    pretrained_model: GPT,
    structural_model: GPT,
    device: str | torch.device,
    dtype: torch.dtype,
    *,
    context: int = 512,
    steps: int = 16,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, Any]:
    """Separate host/dispatch overhead from useful GPU work during decode.

    The structural model has the same 12-layer control flow as GPT-2 small but
    tiny matrices, so its CUDA-event time approximates the Python, dispatcher,
    launch and small-kernel floor. Host timings stop before synchronization.
    """
    resolved = torch.device(device)
    models = {
        "pretrained": pretrained_model,
        "structural": structural_model,
    }
    prompts = {
        name: _prompt(model, 1, context, resolved)
        for name, model in models.items()
    }
    cases: dict[str, tuple[Callable[[], object], Callable[[object], object]]] = {}
    for model_name, model in models.items():
        for variant in ("dynamic", "static"):
            case_name = f"{model_name}:{variant}"

            def setup(
                model: GPT = model,
                model_name: str = model_name,
                variant: str = variant,
            ) -> tuple[KVCache, Tensor]:
                _clear_cuda_cache(resolved)
                return _setup_cached(
                    model, prompts[model_name], variant, dtype, resolved
                )

            def run(state: object, model: GPT = model) -> Tensor:
                assert isinstance(state, tuple)
                return _run_cached(model, state, steps)

            cases[case_name] = (setup, run)

    total = time_interleaved_staged(cases, resolved, warmup, repeats)
    host = _time_host_interleaved_staged(cases, resolved, warmup, repeats)
    opcounts = {
        model_name: _decode_opcounts(
            model, dtype, resolved, context, launch_overhead_us=0.0
        )
        for model_name, model in models.items()
    }
    per_token = {
        name: measurement.median_s / steps for name, measurement in total.items()
    }
    records: list[dict[str, Any]] = []
    for model_name in models:
        for variant in ("dynamic", "static"):
            case_name = f"{model_name}:{variant}"
            total_summary = _measurement(total[case_name], steps)
            host_summary = _measurement(host[case_name], steps)
            structural_name = f"structural:{variant}"
            structural_ops = opcounts["structural"][variant]["total"]
            structural_cost = per_token[structural_name] / structural_ops
            records.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "context_len": context,
                    "steps": steps,
                    "total_median_s_per_token": total_summary["median_s"],
                    "total_p25_s_per_token": total_summary["p25_s"],
                    "total_p75_s_per_token": total_summary["p75_s"],
                    "host_median_s_per_token": host_summary["median_s"],
                    "host_p25_s_per_token": host_summary["p25_s"],
                    "host_p75_s_per_token": host_summary["p75_s"],
                    "host_to_total_ratio": (
                        host_summary["median_s"] / total_summary["median_s"]
                    ),
                    "ops": opcounts[model_name][variant],
                    "structural_cost_per_op_s": structural_cost,
                    "gpu_work_delta_vs_structural_s_per_token": (
                        per_token[case_name] - per_token[structural_name]
                    ),
                }
            )
    return {
        "structural_config": {
            "vocab_size": structural_model.config.vocab_size,
            "n_positions": structural_model.config.n_positions,
            "n_embd": structural_model.config.n_embd,
            "n_layer": structural_model.config.n_layer,
            "n_head": structural_model.config.n_head,
        },
        "records": records,
    }


def run_suites(
    model: GPT,
    structural_model: GPT | None,
    suite: str,
    device: str | torch.device,
    dtype: torch.dtype,
    calibration: dict[str, float],
    *,
    warmup: int,
    repeats: int,
    steps: int,
    decode_contexts: Sequence[int] | None = None,
    batch_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32),
    batch_context: int = 512,
    prefill_lengths: Sequence[int] = (64, 128, 256, 512, 1024),
    memory_batches: Sequence[int] = (1, 8),
    memory_prompt_len: int = 32,
    memory_new_tokens: int = 992,
    memory_sample_every: int = 32,
    op_context: int = 512,
    overhead_context: int = 512,
) -> dict[str, Any]:
    """Run selected suites. Small dimensions can be injected by unit tests."""
    resolved = torch.device(device)
    selected = (
        ("decode", "batch", "prefill", "memory", "overhead")
        if suite == "all"
        else (suite,)
    )
    results: dict[str, Any] = {}
    for name in selected:
        started = time.perf_counter()
        if name == "decode":
            results[name] = run_decode_suite(
                model,
                resolved,
                dtype,
                calibration,
                contexts=decode_contexts,
                steps=steps,
                warmup=warmup,
                repeats=repeats,
                op_context=op_context,
            )
        elif name == "batch":
            results[name] = run_batch_suite(
                model,
                resolved,
                dtype,
                calibration,
                batches=batch_sizes,
                context=batch_context,
                steps=steps,
                warmup=warmup,
                repeats=repeats,
            )
        elif name == "prefill":
            results[name] = run_prefill_suite(
                model,
                resolved,
                dtype,
                calibration,
                lengths=prefill_lengths,
                warmup=warmup,
                repeats=repeats,
            )
        elif name == "memory":
            results[name] = run_memory_suite(
                model,
                resolved,
                dtype,
                batches=memory_batches,
                prompt_len=memory_prompt_len,
                new_tokens=memory_new_tokens,
                sample_every=memory_sample_every,
            )
        elif name == "overhead":
            if structural_model is None:
                raise ValueError("overhead suite requires a structural model")
            results[name] = run_overhead_suite(
                model,
                structural_model,
                resolved,
                dtype,
                context=overhead_context,
                steps=steps,
                warmup=warmup,
                repeats=repeats,
            )
        else:
            raise ValueError(f"unknown suite {name!r}")
        results[name]["duration_s"] = time.perf_counter() - started
    return results


def validate_benchmark_request(suite: str, device: str, dtype: str) -> None:
    """Reject CLI combinations excluded by ADR-0009 or by a suite's contract."""
    if device == "cpu" and dtype == "bfloat16":
        raise ValueError("bfloat16 on CPU is rejected by ADR-0009; use float32")
    if device == "cpu" and suite in ("batch", "memory", "overhead"):
        raise ValueError(f"{suite} suite requires CUDA")
    if device == "cpu" and suite == "all":
        raise ValueError("--suite all includes CUDA-only batch and memory suites")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("decode", "batch", "prefill", "memory", "overhead", "all"),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    try:
        validate_benchmark_request(args.suite, args.device, args.dtype)
    except ValueError as error:
        parser.error(str(error))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")

    require_clean_tree(args.allow_dirty)
    set_benchmark_numerics()
    dtype = _dtype(args.dtype)
    device = torch.device(args.device)
    steps = args.steps if args.steps is not None else (16 if device.type == "cuda" else 8)
    calibration_payload = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    calibration = _calibration_values(calibration_payload, device, dtype)
    model = load_pretrained_gpt2(dtype=dtype, device=device)
    structural_model = None
    if args.suite in ("overhead", "all"):
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        with torch.device(device):
            structural_model = GPT(
                GPT2Config(
                    vocab_size=256,
                    n_positions=1024,
                    n_embd=48,
                    n_layer=12,
                    n_head=12,
                )
            ).to(dtype)
        structural_model.eval()
    heat_up(device)
    before = nvidia_smi_snapshot()
    started = time.perf_counter()
    results = run_suites(
        model,
        structural_model,
        args.suite,
        device,
        dtype,
        calibration,
        warmup=args.warmup,
        repeats=args.repeats,
        steps=steps,
    )
    duration = time.perf_counter() - started
    after = nvidia_smi_snapshot()

    payload = {
        "machine": machine_info(),
        "settings": {
            "suite": args.suite,
            "device": args.device,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "steps": steps,
            "seed": SEED,
            "allow_dirty": args.allow_dirty,
        },
        "calibration": {
            "path": str(CALIBRATION_PATH.relative_to(ROOT)).replace("\\", "/"),
            "values_used": calibration,
        },
        "nvidia_smi_before": before,
        "nvidia_smi_after": after,
        "duration_s": duration,
        "suites": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.suite}_{args.device}_{args.dtype}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"duration_s={duration:.3f}")
    print(f"nvidia_smi_before={before}")
    print(f"nvidia_smi_after={after}")
    for name, result in results.items():
        print(
            f"{name}: records={len(result['records'])} "
            f"duration_s={result['duration_s']:.3f}"
        )


if __name__ == "__main__":
    main()
