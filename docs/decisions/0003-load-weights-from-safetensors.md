# ADR-0003: Load weights from safetensors

## Context

The from-scratch `GPT` needs the pretrained GPT-2 small weights. ADR-0001 keeps Hugging Face `transformers` out of the inference code and allows it only as a test oracle.

## Options considered

- **(A)** Download `model.safetensors` with `huggingface_hub` and read it with `safetensors`, then map the tensors onto our parameters.
- **(B)** Build `GPT2LMHeadModel.from_pretrained("gpt2")` and copy its `state_dict()` into our model.

## Decision

(A).

## Rationale

(A) keeps `transformers` out of the inference code, as ADR-0001 requires. `huggingface_hub` and `safetensors` are small, and they only fetch and read the file. (B) would make `transformers` a runtime dependency just to obtain tensors that are already on disk.

## Consequences

- Our parameter names mirror the checkpoint (`wte`, `wpe`, `h.N.ln_1`, `h.N.attn.c_attn`, and so on), so the mapping is the identity, apart from the transpositions below.
- HF stores the attention and MLP projections as Conv1D with weights laid out (in, out). The loader transposes `attn.c_attn.weight`, `attn.c_proj.weight`, `mlp.c_fc.weight` and `mlp.c_proj.weight` to the (out, in) layout of `nn.Linear`.
- The causal-mask buffers (`.attn.bias`, `.attn.masked_bias`) and `lm_head.weight` are dropped explicitly, by name. `lm_head` is tied to `wte`. The `gpt2` checkpoint has 160 keys, none of them prefixed with `transformer.`. Its 12 dropped keys are the `h.N.attn.bias` buffers, and it has no `lm_head.weight`. That leaves 148 keys, one for each parameter.
- Loading is strict: missing keys, unexpected keys and shape mismatches all raise `ValueError`. A parameter left at its random initialization raises no error on its own, it only makes the output worse, so a lenient loader would hide the bug.
