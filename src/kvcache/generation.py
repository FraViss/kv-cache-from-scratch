import torch
from torch import Tensor

from kvcache.cache import DynamicKVCache, KVCache
from kvcache.model import GPT


def _validate_generation_args(model: GPT, input_ids: Tensor, max_new_tokens: int) -> None:
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if input_ids.dim() != 2 or input_ids.shape[1] == 0:
        raise ValueError(
            f"input_ids must have shape (B, P) with P >= 1, got {tuple(input_ids.shape)}"
        )
    prompt_length = input_ids.shape[1]
    total = prompt_length + max_new_tokens
    if total > model.config.n_positions:
        raise ValueError(
            f"prompt length + max_new_tokens = {prompt_length} + {max_new_tokens} = {total} "
            f"exceeds n_positions ({model.config.n_positions}). There is no sliding window: "
            "GPT-2 uses learned absolute positions, so dropping earlier tokens would renumber "
            "the context and make the cached and uncached paths incomparable."
        )


def generate_greedy_uncached(model: GPT, input_ids: Tensor, max_new_tokens: int) -> Tensor:
    """Greedy decoding with no KV cache: every step forwards the whole prefix.

    input_ids has shape (B, P) with P >= 1. The result has shape (B, P + max_new_tokens)
    and starts with a copy of the prompt. input_ids itself is not modified, and the
    model's train/eval mode is left as it was. Generation always runs for exactly
    max_new_tokens steps; there is no end-of-text stop.
    """
    _validate_generation_args(model, input_ids, max_new_tokens)
    if max_new_tokens == 0:
        return input_ids.clone()

    with torch.inference_mode():
        sequence = input_ids
        for _ in range(max_new_tokens):
            next_token = model(sequence)[:, -1, :].argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, next_token), dim=1)
    # A clone made outside inference_mode is a normal tensor that autograd can use.
    return sequence.clone()


def generate_greedy_cached(
    model: GPT, input_ids: Tensor, max_new_tokens: int, cache: KVCache | None = None
) -> Tensor:
    """Greedy decoding with a KV cache: a prefill on the prompt, then one token per step.

    Same contract as `generate_greedy_uncached`. cache must be empty; when it is None a
    `DynamicKVCache` is created. The last generated token is never fed to the model, so
    afterwards cache.seq_len is P + max_new_tokens - 1 (0 when max_new_tokens is 0).
    """
    _validate_generation_args(model, input_ids, max_new_tokens)
    if cache is None:
        cache = DynamicKVCache(model.config.n_layer)
    elif cache.seq_len != 0:
        raise ValueError(f"cache must be empty, it holds {cache.seq_len} tokens")
    if max_new_tokens == 0:
        return input_ids.clone()

    with torch.inference_mode():
        next_token = model(input_ids, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]
        for _ in range(max_new_tokens - 1):
            next_token = model(next_token, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
        sequence = torch.cat((input_ids, *generated), dim=1)
    # A clone made outside inference_mode is a normal tensor that autograd can use.
    return sequence.clone()
