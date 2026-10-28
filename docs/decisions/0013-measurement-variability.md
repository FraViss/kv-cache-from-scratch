# ADR-0013: Measurement variability

## Context

At batch size 1, graph decode in float32 shows two time levels for the same captured graph: about 2.17 and about 2.54 ms per token. The re-measured decode suite (`all_cuda_float32.json`, commit `b0a0cde`) has 2.541–2.548 ms at T = 64…768 and 2.194 ms at T = 1000. A graph-only preview had given 2.11 ms. Two diagnostic suites and a clock-logged run were added to find where the difference comes from, and to decide how the README presents graph timings. This record states what was measured. It does not claim a cause the data do not show.

Sources:

- `graph_isolation` and `shapes`, measured at commit `eb3b6b8` and committed in `e71febc`. The float32 isolation file from that run was later replaced, and is still available in git history at `e71febc`.
- A second float32 `graph_isolation` run with an nvidia-smi log every 100 ms, measured at commit `bb8f03a`: `benchmarks/results/graph_isolation_cuda_float32.json` and `benchmarks/results/nvsmi_graph_isolation_float32.csv`.

## Options considered

- **(A)** Attribute the two levels to GPU memory-clock states (hypothesis H3), if the clock-logged run confirms it under a rule fixed before the analysis.
- **(B)** Report the two levels as variability with an unidentified cause, and give float32 graph timings as a range.

## Decision

(B). H3 is not confirmed by the rule below.

## Rationale

### Interleaving with eager variants does not explain the two levels

In the first isolation run (`e71febc`), each (model, T) was captured once and replayed in four modes. Pretrained model, float32, ms per token:

| T | isolated ascending | isolated descending | after 0.2 s idle | interleaved with eager |
| --- | --- | --- | --- | --- |
| 64 | 2.170 | 2.544 | 2.542 | 2.543 |
| 512 | 2.166 | 2.538 | 2.535 | 2.118 |
| 1000 | 2.165 | 2.529 | 2.527 | 2.153 |

The same graph is fast in one block and slow in another. Two modes with no eager work at all (descending, after idle) are uniformly slow, and the interleaved mode contains fast values. The structural model in the graph stays at 0.40–0.48 ms in every mode except after idle at T = 1000 (0.860 ms). bfloat16 shows the same pattern on a smaller scale: 1.486 and 1.540 ms against 1.677–1.711 ms elsewhere.

### H3 (memory-clock states) is not confirmed

The rule was fixed before the clock data were examined.

- Pretrained records count as fast at a median of at most 2.30 ms and as slow at a median of at least 2.40 ms.
- A record lies in the high memory-clock interval if every `clocks.mem` sample within its start and end timestamps equals the highest value seen during the run. It lies in a low interval if every sample is below that value.
- H3 is confirmed only if `clocks.mem` takes at least two values, every fast record is high and every slow record is low.

Run `bb8f03a`, 445 samples between the first heat-up and the last record:

- `clocks.mem` took four values during the run: 810 MHz (23 samples), 9001 MHz (329), 11001 MHz (50) and 12001 MHz (43). 405 MHz also appears outside the run window.
- Eleven pretrained records were slow and one was fast.

Pretrained records (mean memory and SM clocks over each record's interval):

| mode | T | median ms | samples | mean mem MHz (range) | mean SM MHz | class | interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| isolated ascending | 64 | 2.545 | 5 | 9801 (9001–11001) | 1660 | slow | low |
| isolated ascending | 512 | 2.537 | 5 | 9001 | 2820 | slow | low |
| isolated ascending | 1000 | 2.531 | 5 | 9001 | 2820 | slow | low |
| isolated descending | 1000 | 2.529 | 5 | 9001 | 1815 | slow | low |
| isolated descending | 512 | 2.537 | 6 | 9001 | 2629 | slow | low |
| isolated descending | 64 | 2.624 | 6 | 9001 | 2805 | slow | low |
| after idle | 64 | 2.585 | 28 | 9001 | 1055 | slow | low |
| after idle | 512 | 2.710 | 29 | 9001 | 891 | slow | low |
| after idle | 1000 | 2.589 | 28 | 9001 | 887 | slow | low |
| interleaved with eager | 64 | 2.546 | 23 | 9001 | 2590 | slow | low |
| interleaved with eager | 512 | 2.537 | 23 | 9001 | 2753 | slow | low |
| interleaved with eager | 1000 | 2.235 | 27 | 10260 (9001–12001) | 2672 | fast | mixed |

Every slow record is low, but the only fast record has mixed samples, so the rule's condition fails: **H3 not confirmed**. Other facts from the same run:

- The only fast record is the only pretrained record whose interval reaches 12001 MHz.
- The slow records span mean SM clocks from 887 to 2820 MHz while their times stay between 2.529 and 2.710 ms.
- The isolated records are about 0.5 s long, so each has only 5–6 clock samples.

### Repeated-matmul calibration measures weights resident in L2

The `shapes` suite repeats each linear 20 times on one weight. The read-bandwidth curve (`x.sum()`, float32) shows the cache effect. Two runs are reported: the shapes run in float32, and the shapes run in bfloat16, whose curve also uses float32 tensors.

| tensor size | float32 run | bfloat16 run |
| --- | --- | --- |
| 16 MiB | 824.4 GB/s | 764.4 GB/s |
| 64 MiB | 274.6 GB/s | 273.3 GB/s |
| 1024 MiB | 347.0 GB/s | 347.9 GB/s |

The calibrated DRAM read bandwidth is 344.2 GB/s. At B = 1 in float32 the four projections measure 390.7–845.1 GB/s, above DRAM. In decode every layer has its own weights, about 495 MB in total in float32, far beyond L2, so these per-call times understate the real weight reads.

Component prediction for float32, B = 1, T = 512, against 2.1663 ms measured (isolated ascending, `e71febc`):

- As computed from the per-call times: 0.4464 + 0.8412 + 0.2750 = 1.5625 ms. The terms are the structural graph, Σ real − Σ structural linears, and cache bytes at the 64 MiB bandwidth.
- With the 340,070,400 bytes of projection weights read at 344.243 GB/s (0.9879 ms) instead of the 0.5221 ms measured with repeated calls: 2.0283 ms, a ratio of 1.068.

In bfloat16 the prediction is 1.2466 ms against 1.6871 ms measured, a gap of 0.4405 ms that stays unexplained. In bfloat16 at B = 1 some matmuls are slower than at B = 8, so the kernel choice depends on the shape:

| layer | B = 1 | B = 8 |
| --- | --- | --- |
| c_attn | 17.79 µs | 6.58 µs |
| attn.c_proj | 7.07 µs | 4.80 µs |
| c_fc | 9.28 µs | 8.27 µs |

### The square-matmul peak does not represent decode shapes

At B = 32, effective throughput as a share of the peak measured with a 4096×4096 matmul (8.938 TFLOP/s in float32, 32.099 in bfloat16):

| layer | float32 | bfloat16 |
| --- | --- | --- |
| c_attn | 26.6% | 39.7% |
| attn.c_proj | 17.6% | 18.4% |
| c_fc | 30.2% | 45.1% |
| mlp.c_proj | 29.4% | 38.5% |
| lm_head | 33.4% | 21.5% |

## Consequences

- The README reports the two float32 graph levels as machine variability with an unidentified cause. It gives graph timings in float32 as a range: 2.19–2.55 ms per token in the decode suite, and up to 2.71 ms in the clock-logged isolation run. It does not attribute them to memory clocks.
- Calibration for a cost model has to use the real shapes, and has to account for L2 residency when one weight is reused.
- Locking clocks was not attempted: it needs administrator rights. The power and clock states of this laptop remain an uncontrolled variable, and the clock log is committed so the data can be re-examined.
