# ADR-0010: CUDA graph decode milestone

## Context

The M4 overhead suite (`benchmarks/results/all_cuda_float32.json` and `all_cuda_bfloat16.json`, suite `overhead`, measured at commit `0274162`) timed a decode step at a context of 512 tokens, batch size 1, on the RTX 5060 Laptop GPU. It compared pretrained GPT-2 small with a structural twin: the same 12 layers and 12 heads, but `n_embd` 48 and a vocabulary of 256, so its GPU work is close to nothing.

Median time per token:

| model | cache | float32 | bfloat16 |
| --- | --- | --- | --- |
| pretrained | dynamic | 9.08 ms | 10.08 ms |
| pretrained | static | 10.18 ms | 9.71 ms |
| structural | dynamic | 10.31 ms | 9.96 ms |
| structural | static | 10.33 ms | 10.08 ms |

The tiny model takes as long as the real one, and the host time per token is 0.97 to 1.29 times the total. A decode step issues 354 aten operations with the dynamic cache and 462 with the static cache, 234 of which launch compute, at roughly 22 to 29 µs of structural cost per operation. The calibration measured 16.5 µs per kernel launch (median). Decode at batch size 1 sits on a floor of about 10 ms set by host-side overhead: Python, the dispatcher and kernel launches. The GPU work underneath is far smaller, so neither cache can show its memory behaviour on the GPU, and the comparison of M4 would end there.

ADR-0009 deferred CUDA graphs until this measurement existed. It now exists.

## Options considered

The scope question is whether to add the work at all. The decision below adds it. The architecture question has two options:

- **(A)** One model. The attention mask is built from absolute query positions, the static cache gets a write index held in a tensor on the GPU, and a new module `graphs.py` captures a decode step of the existing model.
- **(B)** A separate decode path that reimplements the block for capture, next to the existing model.

## Decision

A new milestone, M5: decode captured in a CUDA graph, using the static cache. It comes before the final write-up, so M4 part 3 becomes the write-up of everything, M5 included. This extends the core scope of ADR-0002.

Architecture: (A).

## Rationale

The overhead floor is the main finding of M4 on the GPU, and capturing the step is the direct way to test it: a replay launches the whole step from one host call. Writing up M4 before that would describe a floor without testing the explanation.

With (A) there is a single implementation, already protected by the full equivalence suite: Hugging Face logits, cached against uncached, dynamic against static. Every change needed for capture goes through those tests. (B) would need two implementations kept in step, and the captured one would need its own equivalence tests against the first.

## Consequences

- The dynamic cache cannot be captured: its shapes change at every step and it allocates. M5 captures the static cache only.
- Inside the graph, attention reads the whole buffer of `max_seq_len` keys, with the unwritten slots masked by position (ADR-0011). The write index can no longer be a Python integer, because an index on the host is fixed at capture. That is proved by `tests/test_cuda_graphs.py::test_python_int_index_is_frozen_at_capture`.
- Prediction, recorded before any measurement, from the calibration and cost model at the current commit. In float32, at batch size 1, a captured step reads the weights without the position table (494,613,504 B) and the full static cache at 1024 tokens (75,497,472 B). At the measured read bandwidth of 337.6 GB/s that is about (494,613,504 + 75,497,472) B / 337.6 GB/s ≈ 1.69 ms per token. bfloat16 should take about half of that. Both should be nearly constant in the context length, since the step always reads the full buffer.
- Feasibility on this machine (Windows 11, RTX 5060 Laptop GPU, compute capability 12.0, torch 2.11.0+cu128, CUDA 12.8, driver 610.74): capture and replay work. The three tests in `tests/test_cuda_graphs.py` pass. They cover replay with static input buffers against eager results, a counter on the GPU that advances on each replay, and the frozen Python index.
