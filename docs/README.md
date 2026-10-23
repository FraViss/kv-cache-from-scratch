# Project logbook

`docs/` is the project logbook and the source of truth for anyone, person or agent, who picks this work back up.

## Index

- [status.md](status.md) — current state and the next step
- [decisions/](decisions/) — architecture decision records, in numerical order

## Decision records

Files in `docs/decisions/` use four-digit numbering (`0001`, `0002`, and so on). Each record has these sections:

- **Context** — the problem and the constraints
- **Options considered** — the alternatives that were on the table
- **Decision** — the option that was chosen
- **Rationale** — why that option won
- **Consequences** — what the choice commits the project to, and what it rules out

## Project conventions

- Python 3.12, managed with uv
- Tests run with pytest
- Commits follow Conventional Commits and stay granular
- All public content is written in English
- No agent-specific instruction files
- Commits are attributed only to the author
