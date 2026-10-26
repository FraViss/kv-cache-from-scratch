import torch
from torch import Tensor

from kvcache.model import GPT


def generate_greedy_uncached(model: GPT, input_ids: Tensor, max_new_tokens: int) -> Tensor:
    """Greedy decoding with no KV cache: every step forwards the whole prefix.

    input_ids has shape (B, P) with P >= 1. The result has shape (B, P + max_new_tokens)
    and starts with a copy of the prompt. input_ids itself is not modified, and the
    model's train/eval mode is left as it was. Generation always runs for exactly
    max_new_tokens steps; there is no end-of-text stop.
    """
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

    with torch.inference_mode():
        sequence = input_ids
        for _ in range(max_new_tokens):
            next_token = model(sequence)[:, -1, :].argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, next_token), dim=1)
        token_ids = sequence.tolist()
    return torch.tensor(token_ids, dtype=input_ids.dtype, device=input_ids.device)
