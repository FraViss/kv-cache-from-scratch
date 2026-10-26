# ADR-0005: Token-id API and tokenizer

## Context

The model works on token ids. Demos and benchmarks need text turned into GPT-2 ids and back, but the cache and the correctness tests do not depend on text at all.

## Options considered

- **(A)** The API takes and returns token ids only. `tiktoken` is used only for demos and benchmarks.
- **(B)** Use the Hugging Face tokenizer for demos.
- **(C)** Adapt the BPE tokenizer from the earlier `bpe-tokenizer` project to load the GPT-2 vocabulary and merges.

## Decision

(A). `tiktoken` is not added yet; it comes in when a demo or benchmark first needs text.

## Rationale

Token ids keep the core small and free of tokenizer dependencies, and the tests already use random ids. (B) would pull Hugging Face tooling outside the tests, against the line set in ADR-0001. (C) gives the most continuity, but GPT-2 needs a byte-level pre-tokenizer and its own regex split, which make it a project of its own.

## Consequences

- `GPT.forward` takes a (B, T) `torch.long` tensor of ids. Text handling stays outside `kvcache`.
- (C) is recorded as a possible extension at the end of the project.
