# Status

Date: 2026-10-26

State: M1 done. The GPT-2 small model, the strict safetensors loader (checkpoint pinned to `607a30d783dfa663caf39e06633721c8d4cfcd7e`) and the uncached greedy baseline all match Hugging Face. Logits agree within 1.6e-13 in float64, and greedy sequences are identical.

## Milestones

| ID | Milestone | Status |
| --- | --- | --- |
| M1 | GPT-2 decoder from scratch, weight loader, greedy generation without a cache, logits checked against Hugging Face | done |
| M2 | Dynamic cache (concatenation) with logit-equivalence tests against the uncached path | not started |
| M3 | Static preallocated cache with a write index | not started |
| M4 | Benchmarks (tokens/s versus length, memory versus length, CPU and GPU) compared with the theoretical cost model | not started |
| — | Memory and serving (block allocator in the style of PagedAttention, batching with mixed lengths) | deferred, see ADR-0002 |

## Next step

M2: dynamic cache by concatenation, with logit-equivalence tests against the uncached path.

## How to get back up to speed

1. Read `docs/README.md`.
2. Read the decision records in order.
3. Read this file.
