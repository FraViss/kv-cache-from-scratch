# Status

Date: 2026-10-26

State: M1 in progress. Part 1 is done: the GPT-2 small model from scratch, the strict safetensors weight loader, and the logits equivalence test against Hugging Face (float64, CPU, `atol=1e-9`).

## Milestones

| ID | Milestone | Status |
| --- | --- | --- |
| M1 | GPT-2 decoder from scratch, weight loader, greedy generation without a cache, logits checked against Hugging Face | in progress (part 1 done: model, loader, equivalence) |
| M2 | Dynamic cache (concatenation) with logit-equivalence tests against the uncached path | not started |
| M3 | Static preallocated cache with a write index | not started |
| M4 | Benchmarks (tokens/s versus length, memory versus length, CPU and GPU) compared with the theoretical cost model | not started |
| — | Memory and serving (block allocator in the style of PagedAttention, batching with mixed lengths) | deferred, see ADR-0002 |

## Next step

M1 part 2: greedy generation without a cache, as the baseline.

## How to get back up to speed

1. Read `docs/README.md`.
2. Read the decision records in order.
3. Read this file.
