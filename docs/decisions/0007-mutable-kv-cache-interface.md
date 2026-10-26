# ADR-0007: Mutable KV cache interface

## Context

M2 adds a KV cache that grows by concatenation. M3 will replace it with a static cache that writes into a preallocated buffer at a write index. The model, the generation loop and the equivalence tests should not change when that happens. The interface chosen now decides how much of M3 is a drop-in replacement and how much is a rewrite.

## Options considered

- **(A)** A mutable cache object passed into the forward pass. Each attention layer calls `update(layer_idx, k, v)`, which stores the new keys and values and returns the layer's full keys and values. The cache also exposes `seq_len`, `nbytes` and `reset()`.
- **(B)** A functional style: the forward pass takes the past keys and values as an argument and returns the updated ones (`past_key_values`), and the caller threads them from one step to the next.

## Decision

(A). `KVCache` is an abstract base class, and `DynamicKVCache` is its first implementation, growing each layer with `torch.cat` along the token dimension.

## Rationale

The static cache of M3 writes in place into a buffer allocated once. With (A) it is a second implementation of the same interface: `update` writes at the current index and returns a view of the filled part of the buffer. Model, generation and tests take a `KVCache` and stay unchanged, and the same equivalence tests can run against both implementations. With (B) the forward pass returns new tensors at every step, which is the natural shape of concatenation but not of writing in place: a static cache would either have to pretend to return fresh tensors or change the signature of the model and of every caller.

## Consequences

- With a cache, `GPT.forward` takes only the new tokens, and the position of the first one comes only from `cache.seq_len`. Passing a nonzero `position_offset` together with a cache raises `ValueError`. Having a single source of truth for the position prevents the classic bug where decode steps are embedded at position 0.
- `seq_len` is read once per forward pass, before any block runs. Layer 0 is updated first, so reading it during the forward would give a value that is already ahead.
- The cache owns its memory. On the first update of a layer it stores a contiguous copy, never a view of the `c_attn` output, which would also keep the queries alive.
- `DynamicKVCache` validates every update (layer index, shapes, dtype, device) and checks that every layer ends up with the same length as layer 0, which catches a layer updated twice or skipped. `torch.cat` would otherwise promote a mixed dtype silently.
- The last generated token never enters the cache: after generating N tokens from a prompt of length P, `cache.seq_len` is P + N - 1. The token is returned, but no logits are needed after it.
- Growing by `torch.cat` copies the whole past of every layer at every step, so memory traffic over a generation is quadratic in its length, and the allocator sees a new tensor of a different size at every step. That cost is the motivation for M3.
