# ADR-0011: Position-based attention mask

## Context

`attend` built its causal mask as `ones(Tq, Tk).tril(diagonal=Tk - Tq)`. That rule assumes the queries are the last Tq of exactly Tk keys. It holds for every path up to M4: the uncached forward, `DynamicKVCache`, and `StaticKVCache`, whose `update` returns only the filled part of the buffer (ADR-0008).

The captured decode of M5 (ADR-0010) needs constant shapes, so attention will read the whole static buffer, written or not. Tk is then `max_seq_len`, not the number of tokens, and the `tril` rule would expose every unwritten slot to the query.

## Options considered

- **(A)** Build the mask from the absolute position of each query: `arange(Tk)[None, :] <= q_positions[:, None]`, with `q_positions` the long tensor of positions that `GPT.forward` already computes for the position embedding.
- **(B)** Keep `tril` and add a separate mask for the valid length of the buffer.

## Decision

(A). `attend(q, k, v, q_positions)` takes the positions as a long tensor of shape (Tq,), the same for every sequence in the batch (ADR-0006). `GPT.forward` passes one positions tensor to both `wpe` and every block.

## Rationale

One rule covers both cases. A key at index j is visible to a query at position p when j <= p. The position comes from the same tensor as the position embedding, so the mask and the embedding cannot disagree. (B) would need a second piece of state, the valid length, that must be kept in step with the positions.

## Consequences

- **Equivalence with the old rule.** On every existing path the keys are exactly the tokens so far, Tk of them, and the queries sit at positions Tk - Tq to Tk - 1. Query i is at position i + Tk - Tq, so `j <= p` is `j <= i + (Tk - Tq)`, which is the `tril(diagonal=Tk - Tq)` mask. No result changes: the full test suite passes unchanged, apart from the two direct calls that now pass positions, and so do the Hugging Face integration tests.
- **Partially written buffer.** Unwritten slots have an index above every query position, so they are excluded with no extra state. A test checks this: keys and values 16 long with 8 written and large finite garbage in the rest give the same output as the same call on the first 8 only.
- **Finite values in masked slots.** A masked slot gets weight 0 after the softmax, but `0 * NaN` and `0 * inf` are both NaN, so the masked value slots must hold finite numbers. Keys are less strict: `masked_fill` overwrites the score of a masked key whatever it was, and a check with NaN and inf in the masked key slots gave an unchanged output. The `attend` docstring asks for finite values in all masked slots, which is the simpler and stricter rule. The static cache allocates with `torch.zeros` and only ever writes model outputs, so the invariant holds. Any future buffer has to keep it.
- The mask is now built with `arange` and a comparison, instead of `ones` and `tril`. The aten operation counts recorded in M4 change accordingly. The committed results describe the commit that measured them.
