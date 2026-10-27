# Project logbook

`docs/` is the project logbook and the source of truth for anyone, person or agent, who picks this work back up.

## Index

- [status.md](status.md) — current state and the next step
- [decisions/](decisions/) — architecture decision records, in numerical order
  - [0001](decisions/0001-gpt2-decoder-as-base-model.md) — GPT-2 decoder as the base model
  - [0002](decisions/0002-core-scope-and-deferred-serving.md) — core scope and deferred serving
  - [0003](decisions/0003-load-weights-from-safetensors.md) — load weights from safetensors
  - [0004](decisions/0004-float64-for-correctness-tests.md) — float64 for correctness tests
  - [0005](decisions/0005-token-id-api-and-tokenizer.md) — token-id API and tokenizer
  - [0006](decisions/0006-generation-contract.md) — generation contract
  - [0007](decisions/0007-mutable-kv-cache-interface.md) — mutable KV cache interface
  - [0008](decisions/0008-preallocated-static-kv-cache.md) — preallocated static KV cache
  - [0009](decisions/0009-benchmark-methodology.md) — benchmark methodology

## Decision records

Files in `docs/decisions/` use four-digit numbering (`0001`, `0002`, and so on). Each record has these sections:

- **Context** — the problem and the constraints
- **Options considered** — the alternatives that were on the table
- **Decision** — the option that was chosen
- **Rationale** — why that option won
- **Consequences** — what the choice commits the project to, and what it rules out

## Project conventions

- Python 3.12, managed with uv
- Tests run with pytest. `uv run pytest` runs the fast tests only; `uv run pytest -m integration` runs the Hugging Face integration tests and must pass before every push
- Commits follow Conventional Commits and stay granular
- All public content is written in English
- No agent-specific instruction files
- Commits are attributed only to the author
- Agents stop at commits and never push: the author pushes after reviewing
- Agents never rewrite existing commits: no rebase, amend, squash or autosquash. A correction is a new commit.
- Benchmarks run only from committed code. Every result JSON records the commit that measured its numbers.
