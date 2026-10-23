# ADR-0001: GPT-2 decoder as the base model

## Context

The project needs a decoder-only model on which to implement and measure the KV cache. The model must be verifiable, must run in 8 GB of VRAM, and must have realistic dimensions.

## Options considered

- **(A)** A GPT decoder-only written from scratch that loads the pretrained weights of GPT-2 small (124M).
- **(B)** Reuse the decoder from the previous project `transformer-from-scratch` (a toy encoder-decoder of about 932K parameters, trained on 10 sentence pairs).
- **(C)** A Llama-style architecture (RoPE, GQA, RMSNorm) with pretrained weights.

## Decision

(A).

## Rationale

(A) has an external reference (Hugging Face `transformers`) for checking logits, and it has real dimensions, so the benchmarks mean something. (B) gives the most continuity with earlier work, but the benchmarks would be worthless. (C) is more current, but the implementation and the weight loader cost much more.

## Consequences

`transformers` is used only as an oracle in the tests, never in the inference code. Position embeddings are learned and absolute (no RoPE), and attention is standard multi-head attention (no GQA). The code is structured so a later Llama-style extension stays possible. The no-cache baseline is, conceptually, the greedy decoding of `transformer-from-scratch`.
