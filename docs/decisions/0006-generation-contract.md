# ADR-0006: Generation contract

## Context

The uncached baseline and the cached path have to stay comparable token by token. A difference in the generated sequence must mean a bug in the cache, not a difference in the decoding rules.

## Options considered

- **(A)** Greedy decoding only, against sampling as well.
- **(B)** Always emit exactly N tokens, against stopping at the end-of-text token.
- **(C)** Refuse to run when the prompt plus the new tokens exceed `n_positions`, against a sliding window that drops the oldest tokens.
- **(D)** Batches of equal-length prompts only, against padding and mixed lengths.

## Decision

(A), (B), (C) and (D).

- Greedy only. Sampling is out of scope and can be added later without touching the cache.
- Exactly N tokens, with no stop at the end-of-text token, so benchmarks are deterministic.
- No sliding window: raise an error when the prompt length plus N exceeds `n_positions`.
- Batches only when every prompt has the same length. Padding and mixed lengths are deferred to the serving block (ADR-0002).

## Rationale

GPT-2 uses learned absolute positions. Dropping the oldest tokens renumbers every remaining token, which invalidates every key and value already in the cache. Past the context limit the cached and uncached paths would no longer be computing the same thing, so a comparison between them would be meaningless. Stopping at the end-of-text token would make the sequence length depend on the sample, which makes a benchmark non-deterministic. Sampling changes which token is chosen without saying anything about the cache.

## Consequences

- `generate_greedy_uncached` raises `ValueError` when `P + N > n_positions`, and the cached generators will use the same rule.
- The Hugging Face oracle is compared with its end-of-text stop turned off, and the test fails if the reference sequence is shorter than `P + N`.
- `min_new_tokens` is not used: it changes the logits and would make the comparison false.
- Padding and mixed-length batches stay out until the serving block gets its own decision record.
