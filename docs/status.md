# Status

Date: 2026-10-28

State: M5 in progress; parts 1 and 2 are done. Part 2 added `CapturableKVCache`, which writes at device-side positions and returns its whole buffer, and `CUDAGraphDecoder` with `generate_greedy_graph`: an eager prefill, then a self-feeding decode step captured once and replayed (ADR-0012). The capturable cache passes the full equivalence suite in eager mode, and graph decoding matches the uncached baseline on CUDA and Hugging Face in the integration tests. A one-off preview (pretrained, float32, batch 1, 512-token prompt, 65 new tokens) gave 2.11 ms per token with the graph against 9.95 ms in eager mode with the static cache. That is not yet a measurement.

M5 part 1, for reference: The M4 overhead suite showed that decode at batch size 1 on the GPU sits on a floor of about 10 ms per token set by host-side overhead, since a model of the same structure but tiny dimensions takes as long. M5 captures the decode step in a CUDA graph (ADR-0010). Part 1 checked that capture and replay work on this machine (`tests/test_cuda_graphs.py`). It also rebuilt the attention mask from absolute query positions (ADR-0011), so attention can later read the whole static buffer. No result changed.

M4, for reference: parts 1 and 2 are done; part 3 is now the write-up of M4 and M5 together. Part 1 added the timing harness (CUDA events, interleaved runs), machine record, analytic FLOP and byte model, and machine calibration. Part 2 measured decode, batch scaling, prefill and memory for the uncached baseline, `DynamicKVCache` and `StaticKVCache`, in float32 and bfloat16 on CUDA and in float32 on CPU. The committed JSON records include roofline predictions, aten operation counts, sequence comparisons, allocator measurements and GPU snapshots.

M3, for reference: `StaticKVCache` (ADR-0008) preallocates its buffers once and writes new keys and values in place. It implements the same `KVCache` interface as `DynamicKVCache`. Preview on the 1000-token prompt plus 24 steps, CPU, float64: about 2.3 s with the static cache against 2.7 s with the dynamic one.

M2, for reference: the model takes an optional `KVCache` (ADR-0007), and `DynamicKVCache` grows it by concatenation. With the pretrained model in float64, on a 1000-token prompt plus 24 decode steps, cached and uncached logits differ by at most 8.3e-13. On that case, on CPU in float64, cached generation takes about 2.5 s against 38.5 s without a cache.

M1, for reference: the GPT-2 small model, the strict safetensors loader (checkpoint pinned to `607a30d783dfa663caf39e06633721c8d4cfcd7e`) and the uncached greedy baseline match Hugging Face within 1.6e-13 in float64.

## Milestones

| ID | Milestone | Status |
| --- | --- | --- |
| M1 | GPT-2 decoder from scratch, weight loader, greedy generation without a cache, logits checked against Hugging Face | done |
| M2 | Dynamic cache (concatenation) with logit-equivalence tests against the uncached path | done |
| M3 | Static preallocated cache with a write index | done |
| M4 | Benchmarks (tokens/s versus length, memory versus length, CPU and GPU) compared with the theoretical cost model | in progress (parts 1–2 done: harness, calibration and raw measurement series; part 3, the write-up, waits for M5) |
| M5 | Decode captured in a CUDA graph with the static cache (ADR-0010) | in progress (parts 1–2 done: capture feasibility, position-based mask, capturable cache and graph decoder) |
| — | Memory and serving (block allocator in the style of PagedAttention, batching with mixed lengths) | deferred, see ADR-0002 |

## Next step

M5 part 3: benchmark graph decoding, and re-measure every suite from the same commit, so all M4 and M5 results describe one code base. Then M4 part 3, the write-up of M4 and M5: plots, analysis in the README, and a demonstration notebook.

## How to get back up to speed

1. Read `docs/README.md`.
2. Read the decision records in order.
3. Read this file.
