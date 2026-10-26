# ADR-0009: Benchmark methodology

## Context

M4 compares measured tokens/s and memory with the analytic cost model, on CPU and GPU. The numbers are only meaningful if the timing, the precision and the machine state are fixed and recorded. A laptop GPU does not run at the peak printed on a datasheet: the manufacturer power limit moves the clocks.

## Options considered

- **(A)** Measure bandwidth, compute and launch overhead on this machine, and time GPU work with CUDA events.
- **(B)** Take peak FLOP/s and bandwidth from the GPU datasheet, and time GPU work with a Python timer.

## Decision

(A), with the rules below.

- GPU benchmarks run in float32 and bfloat16. CPU benchmarks run in float32 only, because bfloat16 on CPU is often emulated and would not measure the CPU.
- TF32 is off (`torch.set_float32_matmul_precision("highest")`), and the precision flags are recorded in the result file.
- CUDA timings use a pair of CUDA events per repeat, with a device synchronization before the elapsed time is read. A Python timer without that synchronization measures the kernel launch, not the GPU work.
- Warmup calls are discarded. The reported value is the median, with p25 and p75 from `statistics.quantiles` (method `exclusive`).
- Variants that will be compared are run interleaved (round-robin), so a thermal drift does not always land on the same variant.
- Peaks come from a calibration run on this machine, not from a datasheet.
- Each result file is JSON, committed, and contains the machine record, the settings and the git commit of the run.
- Benchmark code lives in `benchmarks/`, outside `src/`. `benchmarks` may import `kvcache`; `kvcache` does not import `benchmarks`.
- Sequence checks (same tokens) are only valid at the same precision. Correctness stays with the float64 tests.
- CUDA graphs are deferred until the first GPU measurements show whether launch overhead is a large part of a decode step.

## Rationale

Datasheet peaks ignore the laptop power limit, so a roofline built from them would not be a bound for this machine. CUDA events are the timing source that sees the GPU rather than the launch queue. The median and the quartile band describe a run without letting one slow outlier set the result, and interleaving keeps the variants under the same thermal conditions. Committing the JSON with the commit hash is what makes a later number reproducible.

## Consequences

- `benchmarks/calibrate.py` writes `benchmarks/results/calibration.json`. Later benchmark scripts call `set_benchmark_numerics()` before measuring and follow the same timer.
- A result compared across float32 and float64 is a timing result only. Token identity is checked at one precision.
- Whether decode is launch-bound or bandwidth-bound is an empirical question. CUDA graphs stay out until that measurement exists.
