# KV Cache From Scratch

A from-scratch implementation of key-value caching for autoregressive inference of a decoder-only GPT, checked against a reference and measured with benchmarks.

**Status: work in progress**

## Planned milestones

1. **M1** — GPT-2 decoder from scratch, weight loader, and greedy generation without a cache (baseline), with logits checked against Hugging Face.
2. **M2** — Dynamic cache (concatenation), with tests that logits match the uncached path.
3. **M3** — Static preallocated cache with a write index.
4. **M4** — Benchmarks (tokens/s versus length, memory versus length, CPU and GPU) compared with the theoretical cost model.

## Development

```bash
uv sync
uv run pytest
```

## License

MIT
