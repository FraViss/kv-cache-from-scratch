# ADR-0004: float64 for correctness tests

## Context

Correctness is checked by comparing our logits with an oracle: Hugging Face now, the uncached path from M2 on. The comparison needs a tolerance tight enough to catch real bugs, such as a wrong mask offset or a wrong position, without failing on rounding noise. Floating-point results also depend on the order of operations, which differs between implementations and devices.

## Options considered

- **(A)** Run correctness tests in float64 on CPU, with `atol=1e-9` and `rtol=0`.
- **(B)** Run correctness tests in float32, on CPU or GPU, with a tolerance loose enough to absorb float32 rounding.

## Decision

(A). Tests that compare greedy token sequences also run in float64 on CPU. float32 and float16 on GPU are reserved for the benchmarks (M4).

## Rationale

In float64, the rounding differences between two correct implementations sit several orders of magnitude below 1e-9. A real bug shows up far above it. In float32 the noise is around 1e-4, so the tolerance would have to be loose enough to hide small bugs. Greedy decoding is also fragile in float32: when two logits are nearly equal, rounding can flip the argmax, and the two sequences diverge from that token onward with no bug at all.

Measured maximum absolute difference between our logits and `GPT2LMHeadModel` (eager attention), on a fixed random input of shape (2, 64). The largest absolute logit is about 110.

| dtype | device | max abs difference | argmax equal |
| --- | --- | --- | --- |
| float64 | CPU | 1.6e-13 | yes |
| float32 | CPU | 6.1e-5 | yes |
| float32 | CUDA (RTX 5060 Laptop) | 1.0e-4 | yes |

## Consequences

- The integration test loads both models in float64 on CPU and asserts `atol=1e-9`, `rtol=0`, on a (2, 64) batch and on a full-context (1, 1024) sequence.
- A failing equivalence test is investigated, never fixed by widening the tolerance.
- float64 on CPU is slow, so the correctness tests stay small and the speed measurements live in the benchmarks.
