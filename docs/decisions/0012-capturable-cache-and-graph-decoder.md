# ADR-0012: Capturable cache and graph decoder

## Context

ADR-0010 opened M5: capture the decode step in a CUDA graph to remove the host-side floor that M4 measured, about 10 ms per token at batch size 1. ADR-0011 moved the attention mask onto absolute query positions, so attention can read a buffer longer than the tokens written. A graph replays fixed kernels on fixed addresses: every shape is frozen at capture, and so is every Python value used in it. `tests/test_cuda_graphs.py` shows that a Python integer index is frozen too. `StaticKVCache` writes at its Python length and returns a view whose length grows by one token per step, so it cannot be captured as it is.

## Options considered

- **Where the capturable behaviour lives.** Either a mode flag on `StaticKVCache`, or a subclass `CapturableKVCache(StaticKVCache)`.
- **How the cache learns where to write.** Either a write-index tensor owned by the cache, or the positions tensor the model already computes, passed to `update`.
- **What the graph covers.** Either prefill and decode, or decode only, with an eager prefill.

## Decision

- **A subclass.** `CapturableKVCache` keeps the constructor and `from_config` of `StaticKVCache`. `StaticKVCache`, and with it the M4 results measured on it, is unchanged.
- **Positions in `update`.** `KVCache.update(layer_idx, k, v, positions)` receives the long tensor of absolute positions of the new tokens, the same tensor the mask uses. `DynamicKVCache` and `StaticKVCache` ignore it. `CapturableKVCache` writes with `index_copy_` at those positions and returns the whole buffer of the layer, so its shape is constant. `KVCache.accepts_device_positions` is True only for the capturable cache, and `GPT.forward(..., positions=...)` accepts positions on the device only with such a cache.
- **Two sets of books.** The cache keeps Python lengths per layer, which drive validation, `seq_len` and the capacity checks. During a replay the Python body of `update` does not run: positions advance on the device, and the Python lengths stay behind. `advance(n)` brings them back in line after n replays.
- **`truncate(seq_len)`** on `StaticKVCache`, inherited by the capturable cache, cuts every layer back to seq_len tokens without touching the buffers. As with `reset`, only the written part of a layer is ever visible, so stale data cannot leak.
- **A self-feeding graph.** `CUDAGraphDecoder` captures one step: forward of the current token at the current position, argmax, write into `out_tokens[:, step_idx]`, then advance tokens, positions and step_idx on the device. Consecutive replays need no host work and no synchronisation in between.
- **Eager prefill.** `generate_greedy_graph` runs the prefill eagerly with the capturable cache, then captures and replays the decode step N - 1 times.
- **Validation on the host only, before capture.** Shapes, P >= 1, P + N <= n_positions, an empty cache, `n_layer`, batch size, dtype, the CUDA device and a capacity of at least P + N - 1 are all checked before the first forward pass. Nothing inside the graph can check values: positions are never read on the host.
- **Warmup with a restore after every step.** Warmup runs on a side stream, as PyTorch documents. After each warmup step the decoder restores tokens, positions, step_idx and the cache length (`truncate`), so warmup always writes at the prepared position and never runs past the capacity or the width of `out_tokens`. Capture records without running kernels, but the Python body of `update` still advances the lengths, so the state is restored once more after capture.
- **A capture per generation call.** The graph is captured on every call to `generate_greedy_graph`, so it always matches the cache and the number of tokens. The capture cost is excluded from the M5 part 3 measurements, which time the replays only.

## Rationale

A subclass keeps the measured static cache exactly as M4 measured it, and puts the changed contract (full buffer returned, positions required) in a type of its own instead of behind a flag. Passing the positions the model already computes avoids a second write index that could disagree with the mask and the position embedding. The Python lengths cannot be dropped, because every host check depends on them, so they are kept and resynchronised explicitly. Prefill runs once per generation and has a different shape from decode, so capturing it would add a second graph for little gain. Restoring after each warmup step makes warmup independent of the number of steps left, which matters when only a few tokens remain before the capacity or `out_tokens` is full.

## Consequences

- Inside the graph, attention reads all `max_seq_len` keys at every step, written or not (ADR-0011). Cost per step no longer depends on the number of tokens so far.
- After `generate_greedy_graph`, `cache.seq_len` is P + N - 1, the same as with `generate_greedy_cached`. An eager forward can continue from that cache.
- A wrong position fed to a capturable cache is not caught on the host. On the eager path the model derives the positions from `seq_len`; on the graph path the decoder derives them from P and checks the capacity before capture. An index past the buffer is caught only by a device-side assert of `index_copy_`, which leaves the CUDA context unusable.
- The capturable cache passes the full equivalence suite in eager mode on CPU, and the graph decoder matches the uncached baseline on CUDA and Hugging Face in the integration tests.
- One-off preview, not committed and not a benchmark: pretrained GPT-2 small, float32, CUDA, batch size 1, prompt of 512 tokens, 65 new tokens. Replaying the 64 decode steps took a median of 2.11 ms per token (synchronisation included, capture excluded), against 9.95 ms per decode token with `generate_greedy_cached` and `StaticKVCache`. The prediction recorded in ADR-0010 was about 1.69 ms. The proper measurement is M5 part 3.
