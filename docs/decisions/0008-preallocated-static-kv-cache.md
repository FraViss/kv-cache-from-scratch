# ADR-0008: Preallocated static KV cache

## Context

`DynamicKVCache` (ADR-0007) grows every layer with `torch.cat`. Each decode step allocates new tensors and copies the whole past, so memory traffic is quadratic in the length of the generation and the allocator sees a tensor of a new size at every step. M3 replaces the growth with buffers allocated once, which are written in place at each layer's current length. `StaticKVCache` implements the same `KVCache` interface, so the model, the generation loop and the equivalence tests stay as they are.

The main open question is what `update` returns.

## Options considered

- **(A)** A view of the filled part only: `keys[layer, :, :, :len]` and the same for values. Its shape grows by one token per step, exactly as with `DynamicKVCache`.
- **(B)** The whole buffer, always `max_seq_len` tokens long, with the attention mask extended to hide the positions not yet written. Shapes are constant from one step to the next, which is a prerequisite for CUDA graphs and for `torch.compile` without recompilation. But attention then always runs over `max_seq_len` keys, however few are filled, and `attend` needs the valid lengths passed in to build the mask.

## Decision

(A). (B) is noted as a possible M4 experiment.

## Rationale

With (A) the static cache has the same interface and passes the same tests as `DynamicKVCache` (ADR-0007), and `attend` is unchanged. The difference between the two caches is then only in-place writing against reallocation and copying, so M4 can measure that effect in isolation. (B) changes two things at once: the memory scheme and the amount of attention work per step, which is fixed at `max_seq_len`. It also widens the contract of `attend`. Its constant shapes only pay off together with CUDA graphs or `torch.compile`, both outside the core scope.

Other decisions:

- **Explicit allocation in the constructor.** `StaticKVCache(n_layer, batch_size, n_head, head_dim, max_seq_len, dtype, device)` allocates the two buffers with `torch.zeros`, and nothing allocates afterwards. The cache is built outside `torch.inference_mode()`, so the buffers are normal tensors that can still be written in place outside that context, for example in a later forward under `torch.no_grad()`. The memory cost is visible before generation starts, and `generate_greedy_cached` rejects a cache that is too small (it needs P + N - 1 tokens, since the last token never enters the cache) before the first forward pass.
- **Two memory figures.** `nbytes` is the memory the cache has allocated: for a static cache, the full capacity whatever its use. `used_nbytes` is the memory taken by the tokens actually cached. For `DynamicKVCache` the two coincide. `capacity` is the maximum number of tokens per sequence, `None` for a cache without a limit.
- **Reset without clearing.** `reset()` sets the lengths back to zero and leaves the buffers untouched. This is safe because `update` only ever returns the filled part of a layer, so data from an earlier sequence can never be seen.
- **Strict dtype and device.** The new keys and values must have exactly the dtype and device of the buffer, because `copy_` would convert them silently. The shared validation of ADR-0007 (layer index, shapes, layer lengths) applies unchanged, and a rejected update, including one over the capacity, leaves the cache as it was.

## Consequences

- A short sequence still reserves the full capacity. With sequences of different lengths, every slot is sized for the longest one, and the waste grows with the spread of lengths. That waste is the motivation for the deferred serving block (a block allocator in the style of PagedAttention, ADR-0002).
- Expected in M4: a small gain at batch size 1 with short sequences, where reading the weights dominates each step, and a gain that grows with length and batch size, as copying the past becomes a larger share of the work.
- GPT-2 small at full context (1024 tokens) and batch size 1 allocates 72 MiB of cache in float32 (75,497,472 bytes).
