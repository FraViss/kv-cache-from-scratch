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
from benchmarks.timing import Measurement, time_fn, time_interleaved_staged
from kvcache import GPT, GPT2Config, load_pretrained_gpt2
from kvcache.cache import CapturableKVCache, DynamicKVCache, KVCache, StaticKVCache
from kvcache.graphs import CUDAGraphDecoder

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


def _setup_uncached(model: GPT, prompt: Tensor) -> Tensor:
    """Untimed first forward over the prompt; returns the prompt plus its first token."""
    with torch.inference_mode():
        first = model(prompt)[:, -1, :].argmax(dim=-1, keepdim=True)
    return torch.cat((prompt, first), dim=1)


def _generate_uncached(model: GPT, sequence: Tensor, steps: int) -> Tensor:
    """``steps`` decode steps without a cache, each forwarding the whole sequence."""
    start = sequence.shape[1]
    with torch.inference_mode():
        for _ in range(steps):
            token = model(sequence)[:, -1, :].argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, token), dim=1)
    return sequence[:, start:]


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
    """``steps`` decode steps after the prefill, one token each; returns their tokens."""
    cache, token = state
    generated = []
    with torch.inference_mode():
        for _ in range(steps):
            token = model(token, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(token)
    return torch.cat(generated, dim=1)


def _setup_graph(
    model: GPT,
    prompt: Tensor,
    dtype: torch.dtype,
    device: torch.device,
    capacity: int,
    steps: int,
) -> CUDAGraphDecoder:
    """Untimed: allocate a capturable cache, run an eager prefill, capture one step.

    The cache is built outside inference mode so its buffers stay writable by the
    graph. The decoder holds ``steps + 1`` output slots: slot 0 is the prefill token,
    and each replay fills the next one.
    """
    batch, context = prompt.shape
    if context + steps > capacity:
        raise ValueError(
            f"graph capacity {capacity} cannot hold context {context} + steps {steps}"
        )
    cache = CapturableKVCache.from_config(
        model.config, batch, max_seq_len=capacity, dtype=dtype, device=device
    )
    with torch.no_grad():
        first = model(prompt, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
    decoder = CUDAGraphDecoder(model, cache, steps + 1)
    decoder.out_tokens[:, :1].copy_(first)
    decoder.prepare(first, position=context, first_step=1)
    decoder.capture()
    return decoder


def _run_graph(decoder: CUDAGraphDecoder, steps: int) -> Tensor:
    """``steps`` replays with no synchronisation; returns a view of their tokens.

    ``advance`` only updates Python lengths (no kernel), so ``used_nbytes`` and
    ``seq_len`` match the device afterwards.
    """
    decoder.replay(steps)
    decoder.cache.advance(steps)
    return decoder.out_tokens[:, 1:]


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
    capacity: int | None = None,
) -> dict[str, float]:
    """Roofline per decode step, averaged over the ``steps`` steps after the prefill.

    Step ``index`` feeds the token at position ``context + index`` and attends over
    ``context + 1 + index`` keys. The uncached step forwards the whole sequence of
    that length. The graph step reads the whole buffer of ``capacity`` slots and does
    attention over all of them, so its cost does not depend on the step.
    """
    times = []
    flops_total = 0
    for index in range(steps):
        length = context + 1 + index
        if variant == "uncached":
            flops = batch * prefill_flops(model.config, length)
            bytes_ = weight_bytes(model.config, itemsize)
        elif variant == "graph":
            if capacity is None:
                raise ValueError("the graph prediction needs the cache capacity")
            flops = batch * flops_per_token(model.config, capacity)
            bytes_ = decode_step_bytes(
                model.config, itemsize, length, batch, "capturable", capacity=capacity
            )
        else:
            flops = batch * flops_per_token(model.config, length)
            bytes_ = decode_step_bytes(model.config, itemsize, length, batch, variant)
        flops_total += flops
        times.append(_roofline(flops, bytes_, calibration))
    return {
        "roofline_per_token_s": sum(times) / len(times),
        "mean_flops_per_token": flops_total / steps,
    }


def _graph_bytes(
    model: GPT, itemsize: int, context: int, batch: int, capacity: int
) -> int:
    return decode_step_bytes(
        model.config, itemsize, context + 1, batch, "capturable", capacity=capacity
    )


def _decode_opcounts(
    model: GPT,
    dtype: torch.dtype,
    device: torch.device,
    context: int,
    launch_overhead_us: float,
    variants: Sequence[str] = ("dynamic", "static"),
) -> dict[str, Any]:
    """aten operations of one eager decode step per variant.

    ``"graph"`` counts the eager step that the graph captures: a forward with a
    capturable cache and positions held in a tensor. A replay launches the same
    kernels from one host call, so no per-kernel launch cost is charged for it.
    """
    prompt = _prompt(model, 1, context, device)
    result: dict[str, Any] = {}
    for variant in variants:
        if variant == "graph":
            cache = CapturableKVCache.from_config(
                model.config, 1, dtype=dtype, device=device
            )
            positions = torch.tensor([context], device=device)
            with torch.no_grad():
                token = model(prompt, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
                counts = count_ops(
                    lambda: model(token, cache=cache, positions=positions)
                )
            counts["launch_limited_s"] = None
        else:
            cache = _new_cache(variant, model, 1, dtype, device)
            with torch.inference_mode():
                token = model(prompt, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
                counts = count_ops(lambda: model(token, cache=cache))
            counts["launch_limited_s"] = counts["compute"] * launch_overhead_us / 1e6
        result[variant] = counts
    return result


def _cached_variants(device: torch.device) -> tuple[str, ...]:
    """Cache variants for a device: the graph variant needs CUDA."""
    return ("dynamic", "static", "graph") if device.type == "cuda" else ("dynamic", "static")


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
    graph_capacity: int | None = None,
    graph_floor_s: float | None = None,
) -> dict[str, Any]:
    """Benchmark uncached, dynamic, static and (on CUDA) graph greedy decoding at B=1.

    Every variant runs ``steps`` decode steps after an untimed prefill, and the
    timing is divided by ``steps``. The graph cache holds ``graph_capacity`` tokens
    (default ``n_positions``). ``graph_floor_s`` is the per-token time of the
    structural model in the graph, from the overhead suite of the same run; when it
    is given, graph records also carry the "graph floor + bandwidth" prediction.
    """
    resolved = torch.device(device)
    if contexts is None:
        contexts = (64, 128, 256, 512, 768, 1000) if resolved.type == "cuda" else (
            64,
            256,
            512,
            1000,
        )
    capacity = model.config.n_positions if graph_capacity is None else graph_capacity
    variants = _cached_variants(resolved)
    records: list[dict[str, Any]] = []
    itemsize = _itemsize(dtype)
    op_len = min(op_context, model.config.n_positions - 1)
    opcounts = _decode_opcounts(
        model, dtype, resolved, op_len, calibration["launch_overhead_us"], variants
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
            return _setup_uncached(model, prompt)

        def uncached_run(state: object) -> Tensor:
            assert isinstance(state, Tensor)
            outputs["uncached"] = _generate_uncached(model, state, steps)
            return outputs["uncached"]

        cases: dict[str, tuple[Callable[[], object], Callable[[object], object]]] = {
            "uncached": (uncached_setup, uncached_run),
        }
        for variant in variants:
            if variant == "graph":
                def setup() -> CUDAGraphDecoder:
                    _clear_cuda_cache(resolved)
                    return _setup_graph(model, prompt, dtype, resolved, capacity, steps)

                def run(state: object) -> Tensor:
                    assert isinstance(state, CUDAGraphDecoder)
                    outputs["graph"] = _run_graph(state, steps)
                    return outputs["graph"]
            else:
                def setup(variant: str = variant) -> tuple[KVCache, Tensor]:
                    _clear_cuda_cache(resolved)
                    return _setup_cached(model, prompt, variant, dtype, resolved)

                def run(state: object, variant: str = variant) -> Tensor:
                    assert isinstance(state, tuple)
                    outputs[variant] = _run_cached(model, state, steps)
                    return outputs[variant]

            cases[variant] = (setup, run)

        timings = time_interleaved_staged(cases, resolved, warmup, repeats)
        matches = {
            name: torch.equal(outputs["uncached"], outputs[name]) for name in variants
        }
        sequences_match = all(matches.values())
        for variant, measurement in timings.items():
            prediction = _decode_prediction(
                model, variant, context, steps, 1, itemsize, calibration, capacity
            )
            launch_limited = (
                None
                if variant in ("uncached", "graph")
                else opcounts[variant]["launch_limited_s"]
            )
            record: dict[str, Any] = {
                "variant": variant,
                "context_len": context,
                "steps": steps,
                **_measurement(measurement, steps),
                **prediction,
                "launch_limited_per_step_s": launch_limited,
                "generated_tokens": outputs[variant].cpu().tolist(),
                "sequences_match": sequences_match,
                "matches_uncached": True if variant == "uncached" else matches[variant],
            }
            if variant == "graph":
                bandwidth = calibration["read_bandwidth_GBs"] * 1e9
                record["cache_capacity"] = capacity
                record["graph_floor_s"] = graph_floor_s
                record["graph_floor_plus_bandwidth_per_token_s"] = (
                    None
                    if graph_floor_s is None
                    else graph_floor_s
                    + _graph_bytes(model, itemsize, context, 1, capacity) / bandwidth
                )
            records.append(record)
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
    """Benchmark cached CUDA decode throughput and peak memory by batch.

    The graph variant gets the smallest capacity that fits, ``context + steps``.
    Its memory figures include the private pool of the captured graph.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("batch suite requires CUDA")
    records: list[dict[str, Any]] = []
    itemsize = _itemsize(dtype)
    graph_capacity = context + steps
    capacities = {"dynamic": None, "static": model.config.n_positions, "graph": graph_capacity}
    for batch in batches:
        prompt = _prompt(model, batch, context, resolved)
        outputs: dict[str, Tensor] = {}
        memory: dict[str, dict[str, int]] = {}
        cases: dict[str, tuple[Callable[[], object], Callable[[object], object]]] = {}

        def record_memory(variant: str, cache: KVCache) -> None:
            memory[variant] = {
                "max_memory_allocated": torch.cuda.max_memory_allocated(),
                "max_memory_reserved": torch.cuda.max_memory_reserved(),
                "cache_nbytes": cache.nbytes,
                "cache_used_nbytes": cache.used_nbytes,
            }

        for variant in ("dynamic", "static", "graph"):
            if variant == "graph":
                def setup() -> CUDAGraphDecoder:
                    _clear_cuda_cache(resolved)
                    state = _setup_graph(model, prompt, dtype, resolved, graph_capacity, steps)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    return state

                def run(state: object) -> Tensor:
                    assert isinstance(state, CUDAGraphDecoder)
                    outputs["graph"] = _run_graph(state, steps)
                    record_memory("graph", state.cache)
                    return outputs["graph"]
            else:
                def setup(variant: str = variant) -> tuple[KVCache, Tensor]:
                    _clear_cuda_cache(resolved)
                    state = _setup_cached(model, prompt, variant, dtype, resolved)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    return state

                def run(state: object, variant: str = variant) -> Tensor:
                    assert isinstance(state, tuple)
                    outputs[variant] = _run_cached(model, state, steps)
                    record_memory(variant, state[0])
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
        sequences_match = all(
            torch.equal(outputs["dynamic"], outputs[name]) for name in ("static", "graph")
        )
        for variant, measurement in timings.items():
            prediction = _decode_prediction(
                model,
                variant,
                context,
                steps,
                batch,
                itemsize,
                calibration,
                capacities[variant],
            )
            records.append(
                {
                    "variant": variant,
                    "batch_size": batch,
                    "context_len": context,
                    "steps": steps,
                    "cache_capacity": capacities[variant],
                    "out_of_memory": False,
                    **_measurement(measurement),
                    "tokens_per_s": batch * steps / measurement.median_s,
                    "roofline_per_step_s": prediction["roofline_per_token_s"],
                    "sequences_match": sequences_match,
                    **memory[variant],
                }
            )
        del prompt
        gc.collect()
        torch.cuda.empty_cache()
    return {"graph_capacity": graph_capacity, "records": records}


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

    On CUDA a graph variant replays a captured step (cache capacity
    ``n_positions``). Its host time is the time to enqueue ``steps`` replays.
    """
    resolved = torch.device(device)
    models = {
        "pretrained": pretrained_model,
        "structural": structural_model,
    }
    variants = _cached_variants(resolved)
    prompts = {
        name: _prompt(model, 1, context, resolved)
        for name, model in models.items()
    }
    cases: dict[str, tuple[Callable[[], object], Callable[[object], object]]] = {}
    for model_name, model in models.items():
        for variant in variants:
            case_name = f"{model_name}:{variant}"
            if variant == "graph":
                def setup(
                    model: GPT = model, model_name: str = model_name
                ) -> CUDAGraphDecoder:
                    _clear_cuda_cache(resolved)
                    return _setup_graph(
                        model,
                        prompts[model_name],
                        dtype,
                        resolved,
                        model.config.n_positions,
                        steps,
                    )

                def run(state: object) -> Tensor:
                    assert isinstance(state, CUDAGraphDecoder)
                    return _run_graph(state, steps)
            else:
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
            model, dtype, resolved, context, launch_overhead_us=0.0, variants=variants
        )
        for model_name, model in models.items()
    }
    per_token = {
        name: measurement.median_s / steps for name, measurement in total.items()
    }
    records: list[dict[str, Any]] = []
    for model_name in models:
        for variant in variants:
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


def _graph_floor_s(overhead: dict[str, Any] | None) -> float | None:
    """Per-token time of the structural model in the graph, from an overhead result."""
    if overhead is None:
        return None
    for record in overhead["records"]:
        if record["model"] == "structural" and record["variant"] == "graph":
            return record["total_median_s_per_token"]
    return None


def run_capacity_suite(
    model: GPT,
    device: str | torch.device,
    dtype: torch.dtype,
    calibration: dict[str, float],
    *,
    batch: int = 8,
    context: int = 64,
    steps: int = 16,
    capacities: Sequence[int] = (128, 256, 512, 1024),
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, Any]:
    """Graph decode time against cache capacity, at a fixed batch and context.

    The graph step reads and attends over the whole buffer, so its cost should grow
    with the capacity even though the number of written tokens is the same.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("capacity suite requires CUDA")
    itemsize = _itemsize(dtype)
    prompt = _prompt(model, batch, context, resolved)
    outputs: dict[int, Tensor] = {}
    memory: dict[int, dict[str, int]] = {}
    cases: dict[str, tuple[Callable[[], object], Callable[[object], object]]] = {}
    for capacity in capacities:
        def setup(capacity: int = capacity) -> CUDAGraphDecoder:
            _clear_cuda_cache(resolved)
            return _setup_graph(model, prompt, dtype, resolved, capacity, steps)

        def run(state: object, capacity: int = capacity) -> Tensor:
            assert isinstance(state, CUDAGraphDecoder)
            outputs[capacity] = _run_graph(state, steps)
            memory[capacity] = {
                "cache_nbytes": state.cache.nbytes,
                "cache_used_nbytes": state.cache.used_nbytes,
            }
            return outputs[capacity]

        cases[str(capacity)] = (setup, run)
    timings = time_interleaved_staged(cases, resolved, warmup, repeats)
    first = outputs[capacities[0]]
    sequences_match = all(torch.equal(first, output) for output in outputs.values())
    records: list[dict[str, Any]] = []
    for capacity in capacities:
        prediction = _decode_prediction(
            model, "graph", context, steps, batch, itemsize, calibration, capacity
        )
        records.append(
            {
                "variant": "graph",
                "cache_capacity": capacity,
                "batch_size": batch,
                "context_len": context,
                "steps": steps,
                **_measurement(timings[str(capacity)], steps),
                "roofline_per_token_s": prediction["roofline_per_token_s"],
                "step_bytes": _graph_bytes(model, itemsize, context, batch, capacity),
                "sequences_match": sequences_match,
                **memory[capacity],
            }
        )
    del prompt
    _clear_cuda_cache(resolved)
    return {"records": records}


GRAPH_ISOLATION_MODES = (
    "isolated_ascending",
    "isolated_descending",
    "after_idle",
    "interleaved_with_eager",
)


def run_graph_isolation_suite(
    pretrained_model: GPT,
    structural_model: GPT,
    device: str | torch.device,
    dtype: torch.dtype,
    *,
    contexts: Sequence[int] = (64, 512, 1000),
    steps: int = 16,
    capacity: int | None = None,
    warmup: int = 3,
    repeats: int = 10,
    heat_seconds: float = 3.0,
    idle_s: float = 0.2,
) -> dict[str, Any]:
    """Time graph replays alone, to separate GPU clock effects from the graph itself.

    For each (model, T) the prefill and the capture happen once. Before every
    repetition ``prepare`` restores tokens, positions and step index, and the cache is
    never advanced, so every repetition replays the same ``steps`` steps over the same
    data. Only ``replay(steps)`` is timed, with CUDA events.

    Modes, each a block preceded by ``heat_up`` and bracketed by nvidia-smi snapshots:
    ``isolated_ascending`` and ``isolated_descending`` run T in each order;
    ``after_idle`` sleeps ``idle_s`` before every repetition (untimed);
    ``interleaved_with_eager`` runs ``steps`` untimed eager decode steps with a
    ``StaticKVCache``, setup included, before every repetition, as the alternating
    suites did.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("graph_isolation suite requires CUDA")
    models = {"pretrained": pretrained_model, "structural": structural_model}
    states: dict[tuple[str, int], tuple[CUDAGraphDecoder, Tensor, Tensor]] = {}
    for model_name, model in models.items():
        graph_capacity = model.config.n_positions if capacity is None else capacity
        for context in contexts:
            prompt = _prompt(model, 1, context, resolved)
            decoder = _setup_graph(model, prompt, dtype, resolved, graph_capacity, steps)
            first = decoder.out_tokens[:, :1].clone()
            states[(model_name, context)] = (decoder, first, prompt)

    def measure(model_name: str, context: int, mode: str) -> dict[str, Any]:
        decoder, first, prompt = states[(model_name, context)]
        model = models[model_name]
        produced: list[Tensor] = []

        def setup() -> CUDAGraphDecoder:
            # Untimed: keep the tokens of the previous repetition before restoring.
            produced.append(decoder.out_tokens[:, 1:].clone())
            decoder.prepare(first, position=context, first_step=1)
            if mode == "after_idle":
                time.sleep(idle_s)
            elif mode == "interleaved_with_eager":
                _run_cached(
                    model, _setup_cached(model, prompt, "static", dtype, resolved), steps
                )
            return decoder

        def run(state: object) -> None:
            assert isinstance(state, CUDAGraphDecoder)
            state.replay(steps)

        measurement = time_interleaved_staged(
            {"graph": (setup, run)}, resolved, warmup, repeats
        )["graph"]
        torch.cuda.synchronize()
        produced.append(decoder.out_tokens[:, 1:].clone())
        # produced[0] holds the capture-time contents; compare the replays only.
        replays = produced[1:]
        return {
            "model": model_name,
            "mode": mode,
            "context_len": context,
            "steps": steps,
            "cache_capacity": decoder.cache.capacity,
            **_measurement(measurement, steps),
            "repeats_identical": all(torch.equal(replays[0], t) for t in replays[1:]),
            "generated_tokens": replays[-1].cpu().tolist(),
        }

    records: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for mode in GRAPH_ISOLATION_MODES:
        order = tuple(reversed(contexts)) if mode == "isolated_descending" else tuple(contexts)
        heat_up(resolved, heat_seconds)
        before = nvidia_smi_snapshot()
        mode_records = [
            measure(model_name, context, mode) for model_name in models for context in order
        ]
        after = nvidia_smi_snapshot()
        for record in mode_records:
            record["nvidia_smi_before_block"] = before
            record["nvidia_smi_after_block"] = after
        records.extend(mode_records)
        blocks.append({"mode": mode, "order": list(order), "before": before, "after": after})
    for decoder, _, _ in states.values():
        decoder.graph = None
    states.clear()
    _clear_cuda_cache(resolved)
    return {
        "capacity": capacity,
        "contexts": list(contexts),
        "modes": list(GRAPH_ISOLATION_MODES),
        "blocks": blocks,
        "records": records,
    }


def _decode_linear_shapes(config: GPT2Config) -> dict[str, tuple[int, int, bool]]:
    """The linear layers of one decode step as (in, out, has_bias), per GPT2Config."""
    width = config.n_embd
    return {
        "c_attn": (width, 3 * width, True),
        "attn.c_proj": (width, width, True),
        "c_fc": (width, 4 * width, True),
        "mlp.c_proj": (4 * width, width, True),
        "lm_head": (width, config.vocab_size, False),
    }


def _capture_calls(call: Callable[[], object], calls: int, warmup: int = 3) -> torch.cuda.CUDAGraph:
    """Capture ``calls`` consecutive calls in one graph, after a side-stream warmup."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            call()
    return graph


def run_shapes_suite(
    device: str | torch.device,
    dtype: torch.dtype,
    configs: dict[str, GPT2Config],
    *,
    batches: Sequence[int] = (1, 8, 32),
    calls: int = 20,
    sizes_mib: Sequence[int] = (1, 4, 16, 64, 256, 1024),
    warmup: int = 3,
    repeats: int = 10,
    heat_seconds: float = 3.0,
) -> dict[str, Any]:
    """Time the decode matmuls at their real shapes, inside a CUDA graph.

    For every model family in ``configs`` (name to config) and every linear layer of
    a decode step, ``calls`` consecutive linears on a (B, in) input are captured in
    one graph and the replay is timed with CUDA events. A call is
    ``torch.addmm(bias, x, weight.t(), out=out)``, or ``torch.mm`` without a bias:
    the kernel ``F.linear`` dispatches to for a 2D input, writing into a static
    output buffer. Bandwidth counts the weight and bias bytes; TFLOP/s counts
    ``2 * B * in * out``.

    A read-bandwidth curve follows: ``x.sum()`` on float32 tensors of each size in
    ``sizes_mib``, ``calls`` calls per graph, bytes = ``x.nbytes``.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("shapes suite requires CUDA")
    itemsize = _itemsize(dtype)
    generator = torch.Generator(device=resolved).manual_seed(SEED)

    heat_up(resolved, heat_seconds)
    shapes_before = nvidia_smi_snapshot()
    shape_records: list[dict[str, Any]] = []
    for family, config in configs.items():
        for layer, (fan_in, fan_out, has_bias) in _decode_linear_shapes(config).items():
            weight = torch.randn(
                fan_out, fan_in, generator=generator, dtype=dtype, device=resolved
            )
            bias = (
                torch.randn(fan_out, generator=generator, dtype=dtype, device=resolved)
                if has_bias
                else None
            )
            weight_t = weight.t()
            for batch in batches:
                x = torch.randn(batch, fan_in, generator=generator, dtype=dtype, device=resolved)
                out = torch.empty(batch, fan_out, dtype=dtype, device=resolved)
                if bias is not None:
                    def call(x: Tensor = x, out: Tensor = out, bias: Tensor = bias) -> None:
                        torch.addmm(bias, x, weight_t, out=out)
                else:
                    def call(x: Tensor = x, out: Tensor = out) -> None:
                        torch.mm(x, weight_t, out=out)

                graph = _capture_calls(call, calls)
                measurement = time_fn(graph.replay, resolved, warmup, repeats)
                per_call = measurement.median_s / calls
                bytes_ = (fan_in * fan_out + (fan_out if has_bias else 0)) * itemsize
                shape_records.append(
                    {
                        "family": family,
                        "layer": layer,
                        "in_features": fan_in,
                        "out_features": fan_out,
                        "bias": has_bias,
                        "batch_size": batch,
                        "calls": calls,
                        **_measurement(measurement, calls),
                        "weight_bytes": bytes_,
                        "effective_bandwidth_GBs": bytes_ / per_call / 1e9,
                        "effective_TFLOPs": 2 * batch * fan_in * fan_out / per_call / 1e12,
                    }
                )
                del graph, x, out
            del weight, weight_t, bias
            _clear_cuda_cache(resolved)
    shapes_after = nvidia_smi_snapshot()

    heat_up(resolved, heat_seconds)
    bandwidth_before = nvidia_smi_snapshot()
    bandwidth_records: list[dict[str, Any]] = []
    for size in sizes_mib:
        x = torch.ones(size * 2**20 // 4, dtype=torch.float32, device=resolved)
        graph = _capture_calls(x.sum, calls)
        measurement = time_fn(graph.replay, resolved, warmup, repeats)
        per_call = measurement.median_s / calls
        bandwidth_records.append(
            {
                "size_mib": size,
                "nbytes": x.nbytes,
                "calls": calls,
                **_measurement(measurement, calls),
                "effective_bandwidth_GBs": x.nbytes / per_call / 1e9,
            }
        )
        del graph, x
        _clear_cuda_cache(resolved)
    bandwidth_after = nvidia_smi_snapshot()

    return {
        "configs": {
            family: {"n_embd": config.n_embd, "vocab_size": config.vocab_size}
            for family, config in configs.items()
        },
        "call": "addmm(bias, x, weight.t(), out=out) or mm(x, weight.t(), out=out)",
        "blocks": {
            "shapes": {"before": shapes_before, "after": shapes_after},
            "bandwidth": {"before": bandwidth_before, "after": bandwidth_after},
        },
        "records": shape_records,
        "bandwidth_records": bandwidth_records,
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
    graph_capacity: int | None = None,
    capacity_batch: int = 8,
    capacity_context: int = 64,
    capacities: Sequence[int] = (128, 256, 512, 1024),
    isolation_contexts: Sequence[int] = (64, 512, 1000),
    isolation_heat_seconds: float = 3.0,
    isolation_idle_s: float = 0.2,
    shapes_batches: Sequence[int] = (1, 8, 32),
    shapes_sizes_mib: Sequence[int] = (1, 4, 16, 64, 256, 1024),
    shapes_heat_seconds: float = 3.0,
) -> dict[str, Any]:
    """Run selected suites. Small dimensions can be injected by unit tests.

    With ``suite="all"`` the overhead suite runs first, so decode can use the graph
    floor measured in the same run.
    """
    resolved = torch.device(device)
    selected = (
        ("overhead", "decode", "batch", "prefill", "memory", "capacity")
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
                graph_capacity=graph_capacity,
                graph_floor_s=_graph_floor_s(results.get("overhead")),
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
        elif name == "capacity":
            results[name] = run_capacity_suite(
                model,
                resolved,
                dtype,
                calibration,
                batch=capacity_batch,
                context=capacity_context,
                steps=steps,
                capacities=capacities,
                warmup=warmup,
                repeats=repeats,
            )
        elif name == "graph_isolation":
            if structural_model is None:
                raise ValueError("graph_isolation suite requires a structural model")
            results[name] = run_graph_isolation_suite(
                model,
                structural_model,
                resolved,
                dtype,
                contexts=isolation_contexts,
                steps=steps,
                capacity=graph_capacity,
                warmup=warmup,
                repeats=repeats,
                heat_seconds=isolation_heat_seconds,
                idle_s=isolation_idle_s,
            )
        elif name == "shapes":
            if structural_model is None:
                raise ValueError("shapes suite requires a structural model")
            results[name] = run_shapes_suite(
                resolved,
                dtype,
                {"real": model.config, "structural": structural_model.config},
                batches=shapes_batches,
                sizes_mib=shapes_sizes_mib,
                warmup=warmup,
                repeats=repeats,
                heat_seconds=shapes_heat_seconds,
            )
        else:
            raise ValueError(f"unknown suite {name!r}")
        results[name]["duration_s"] = time.perf_counter() - started
    return results


def validate_benchmark_request(suite: str, device: str, dtype: str) -> None:
    """Reject CLI combinations excluded by ADR-0009 or by a suite's contract."""
    if device == "cpu" and dtype == "bfloat16":
        raise ValueError("bfloat16 on CPU is rejected by ADR-0009; use float32")
    if device == "cpu" and suite in (
        "batch",
        "memory",
        "overhead",
        "capacity",
        "graph_isolation",
        "shapes",
    ):
        raise ValueError(f"{suite} suite requires CUDA")
    if device == "cpu" and suite == "all":
        raise ValueError("--suite all includes CUDA-only batch, memory and capacity suites")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=(
            "decode",
            "batch",
            "prefill",
            "memory",
            "overhead",
            "capacity",
            "graph_isolation",
            "shapes",
            "all",
        ),
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
    if args.suite in ("overhead", "all", "graph_isolation", "shapes"):
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
