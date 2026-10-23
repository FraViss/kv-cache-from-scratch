# ADR-0002: Core scope and deferred serving

## Context

The possible scope runs from a basic cache all the way to a serving layer.

## Options considered

- **(A)** A core of four milestones that is complete and publishable on its own, with the memory-and-serving block deferred until after the benchmarks.
- **(B)** Include the serving block in the initial scope: a block allocator in the style of PagedAttention, and batching with mixed sequence lengths.

## Decision

(A). The core is four milestones:

- **M1** — GPT-2 decoder from scratch, weight loader, and greedy generation without a cache (baseline), with logits checked against Hugging Face.
- **M2** — Dynamic cache (concatenation), with tests that logits match with and without the cache.
- **M3** — Static preallocated cache with a write index.
- **M4** — Benchmarks (tokens/s versus length, memory versus length, CPU and GPU) compared with the theoretical cost model.

Deferred: the "memory & serving" block (a block allocator in the style of PagedAttention, and batching with mixed sequence lengths). That block will be decided after the M4 benchmarks.

## Rationale

The four milestones are enough to implement the cache, prove it against a reference, and measure it. Serving is a separate design, and it should wait until M4 shows where time and memory actually go.

## Consequences

The project can be closed and published after M4. If the deferred block is approved later, it gets its own decision record.
