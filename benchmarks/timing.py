"""Timers for CPU and GPU benchmarks.

A Python timer around a CUDA call measures only the time to enqueue the kernels:
the CPU continues as soon as the launches have been queued, while the GPU may
still be running them. GPU intervals are therefore taken with CUDA events, and
the device is synchronized before the elapsed time is read.
"""

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

_QUARTILE_METHOD = "exclusive"
S = TypeVar("S")


@dataclass(frozen=True)
class Measurement:
    """Repeated timings of one callable, in seconds.

    p25, median and p75 are the three cut points of
    ``statistics.quantiles(samples, n=4, method="exclusive")``. Exclusive mode
    is the standard-library default: it interpolates linearly at the positions
    ``(N + 1) * k / 4`` of the sorted sample (1-based). The median is that
    second cut point, so all three numbers come from one method. At least two
    samples are required.
    """

    samples_s: tuple[float, ...]

    def __post_init__(self) -> None:
        samples = tuple(self.samples_s)
        if len(samples) < 2:
            raise ValueError(f"at least 2 samples are required for quartiles, got {len(samples)}")
        object.__setattr__(self, "samples_s", samples)

    @property
    def _quartiles(self) -> list[float]:
        return statistics.quantiles(self.samples_s, n=4, method=_QUARTILE_METHOD)

    @property
    def p25_s(self) -> float:
        return self._quartiles[0]

    @property
    def median_s(self) -> float:
        return self._quartiles[1]

    @property
    def p75_s(self) -> float:
        return self._quartiles[2]

    @property
    def min_s(self) -> float:
        return min(self.samples_s)


def _on_cuda(device: str | torch.device) -> bool:
    return torch.device(device).type == "cuda"


def _elapsed_s(fn: Callable[[], object], device: str | torch.device) -> float:
    """Time a single call. CUDA events on GPU, perf_counter_ns on CPU."""
    if _on_cuda(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        # elapsed_time is measured from this event to the one passed in.
        return start.elapsed_time(end) / 1000.0
    t0 = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - t0) / 1e9


def time_fn(
    fn: Callable[[], object],
    device: str | torch.device,
    warmup: int = 5,
    repeats: int = 20,
) -> Measurement:
    """Time ``fn`` after discarding ``warmup`` calls.

    On CUDA each repeat gets its own pair of ``torch.cuda.Event(enable_timing=True)``.
    ``record()`` runs before and after ``fn``, and ``torch.cuda.synchronize()`` runs
    before ``elapsed_time`` is read (milliseconds, converted to seconds). Events
    measure GPU stream time between the two records. A Python timer without that
    synchronization returns as soon as the kernels have been queued, so it measures
    the launch, not the work.

    On CPU each repeat is bracketed by ``time.perf_counter_ns()``.
    """
    for _ in range(warmup):
        fn()
    return Measurement(tuple(_elapsed_s(fn, device) for _ in range(repeats)))


def time_interleaved(
    fns: dict[str, Callable[[], object]],
    device: str | torch.device,
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, Measurement]:
    """Time several callables so a thermal drift does not always hit the same one.

    Each callable is warmed up ``warmup`` times, in dict order. The measured calls
    then run round-robin: for every repeat, every callable runs once, in the same
    order (A, B, C, A, B, C, ...). Each call is timed like ``time_fn``.
    """
    names = tuple(fns)
    for name in names:
        for _ in range(warmup):
            fns[name]()
    samples: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(repeats):
        for name in names:
            samples[name].append(_elapsed_s(fns[name], device))
    return {name: Measurement(tuple(samples[name])) for name in names}


def time_interleaved_staged(
    cases: dict[str, tuple[Callable[[], S], Callable[[S], object]]],
    device: str | torch.device,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, Measurement]:
    """Time staged cases round-robin while excluding setup from every sample.

    A case is ``(setup, run)``. For every warmup and measured invocation,
    ``setup()`` creates fresh state outside the timed region. CUDA is synchronized
    after setup so queued setup work cannot leak into the interval, then only
    ``run(state)`` is measured by the same ``_elapsed_s`` used by ``time_fn`` and
    ``time_interleaved``.

    Cases run round-robin in dict order for both warmup and measurement:
    A, B, C, A, B, C, ... . This keeps stateful benchmarks independent while
    spreading thermal drift across variants.
    """
    names = tuple(cases)
    samples: dict[str, list[float]] = {name: [] for name in names}

    def invoke(name: str, *, measured: bool) -> None:
        setup, run = cases[name]
        state = setup()
        if _on_cuda(device):
            torch.cuda.synchronize()
        if measured:
            samples[name].append(_elapsed_s(lambda: run(state), device))
        else:
            run(state)

    for _ in range(warmup):
        for name in names:
            invoke(name, measured=False)
    for _ in range(repeats):
        for name in names:
            invoke(name, measured=True)
    return {name: Measurement(tuple(samples[name])) for name in names}
