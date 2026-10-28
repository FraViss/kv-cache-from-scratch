# Status

Date: 2026-10-28

State: the project is complete. M1–M5 are done. The README presents the results and the lessons, [results.md](results.md) holds every table, generated from the committed JSON in `benchmarks/results/`, and [figures/](figures/) holds the figures drawn from the same files. `notebooks/demo.ipynb` is an executed demonstration.

## Milestones

| ID | Milestone | Status |
| --- | --- | --- |
| M1 | GPT-2 decoder from scratch, weight loader, greedy generation without a cache, logits checked against Hugging Face | done |
| M2 | Dynamic cache (concatenation) with logit-equivalence tests against the uncached path | done |
| M3 | Static preallocated cache with a write index | done |
| M4 | Benchmarks (tokens/s versus length, memory versus length, CPU and GPU) compared with the theoretical cost model | done |
| M5 | Decode captured in a CUDA graph with the static cache (ADR-0010) | done |
| — | Memory and serving (block allocator in the style of PagedAttention, batching with mixed lengths) | deferred, see ADR-0002 |

## Summary by milestone

- **M1.** GPT-2 small from scratch, with a strict safetensors loader (checkpoint pinned to `607a30d783dfa663caf39e06633721c8d4cfcd7e`) and an uncached greedy baseline. All three match Hugging Face within 1.6e-13 in float64 (ADR-0001, 0003, 0004, 0005, 0006).
- **M2.** A mutable `KVCache` interface and `DynamicKVCache`. Cached and uncached logits agree within 8.3e-13 on a 1000-token prompt (ADR-0007).
- **M3.** `StaticKVCache`: preallocated, written in place, same interface (ADR-0008).
- **M4.** Calibration, cost model, timing harness and suites for decode, batch, prefill, memory and overhead (ADR-0009). Eager decode on the GPU turned out to be bound by host-side overhead.
- **M5.** Attention mask built from positions, `CapturableKVCache`, and a CUDA graph decoder with graph variants in the benchmarks (ADR-0010, 0011, 0012). Diagnostic suites (`graph_isolation` with a GPU clock log, `shapes`) documented the variability between runs and the limits of the calibration (ADR-0013).

All results were re-measured from clean commits; each JSON records its commit. The float32 graph runs at two levels whose cause was not identified (ADR-0013), so float32 graph times are reported as a range.

## Possible next steps (not planned)

- The deferred serving block: a block allocator in the style of PagedAttention, and batching with mixed lengths (ADR-0002).
- A Llama-style extension: RoPE, GQA, RMSNorm (ADR-0001).
- Quantisation of the cache (int8 or fp8 keys and values), measured with the same suites.

## How to get back up to speed

1. Read `docs/README.md`.
2. Read the decision records in order.
3. Read this file, then `README.md` and `docs/results.md`.
