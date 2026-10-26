# Status

Date: 2026-10-26

State: M4 in progress. Part 1 is done: the timing harness (CUDA events, interleaved runs), the machine record, the analytic FLOP and byte model, and a calibration of bandwidth, compute and launch overhead on this machine (`benchmarks/results/calibration.json`). No generation benchmark yet.

M3, for reference: `StaticKVCache` (ADR-0008) preallocates its buffers once and writes new keys and values in place. It implements the same `KVCache` interface as `DynamicKVCache`. Preview on the 1000-token prompt plus 24 steps, CPU, float64: about 2.3 s with the static cache against 2.7 s with the dynamic one.

M2, for reference: the model takes an optional `KVCache` (ADR-0007), and `DynamicKVCache` grows it by concatenation. With the pretrained model in float64, on a 1000-token prompt plus 24 decode steps, cached and uncached logits differ by at most 8.3e-13. On that case, on CPU in float64, cached generation takes about 2.5 s against 38.5 s without a cache.

M1, for reference: the GPT-2 small model, the strict safetensors loader (checkpoint pinned to `607a30d783dfa663caf39e06633721c8d4cfcd7e`) and the uncached greedy baseline match Hugging Face within 1.6e-13 in float64.

## Milestones

| ID | Milestone | Status |
| --- | --- | --- |
| M1 | GPT-2 decoder from scratch, weight loader, greedy generation without a cache, logits checked against Hugging Face | done |
| M2 | Dynamic cache (concatenation) with logit-equivalence tests against the uncached path | done |
| M3 | Static preallocated cache with a write index | done |
| M4 | Benchmarks (tokens/s versus length, memory versus length, CPU and GPU) compared with the theoretical cost model | in progress (part 1 done: harness, cost model, calibration) |
| — | Memory and serving (block allocator in the style of PagedAttention, batching with mixed lengths) | deferred, see ADR-0002 |

## Next step

M4 part 2: generation benchmarks. Tokens/s and memory against length, for the uncached baseline and both caches, on CPU and GPU, compared with the cost model.

## How to get back up to speed

1. Read `docs/README.md`.
2. Read the decision records in order.
3. Read this file.
