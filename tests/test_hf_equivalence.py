from collections.abc import Callable
from copy import copy

import pytest
import torch
from transformers import GPT2LMHeadModel

from kvcache import (
    GPT,
    DynamicKVCache,
    generate_greedy_cached,
    generate_greedy_uncached,
    load_pretrained_gpt2,
)
from kvcache.weights import GPT2_REVISION

pytestmark = pytest.mark.integration

GENERATION_CASES = pytest.mark.parametrize(
    ("prompt_shape", "new_tokens"),
    [((2, 16), 48), ((1, 1000), 24)],
    ids=["prompt-2x16-plus-48", "prompt-1x1000-plus-24"],
)


@pytest.fixture(scope="module")
def models() -> tuple[GPT, GPT2LMHeadModel]:
    ours = load_pretrained_gpt2(dtype=torch.float64, device="cpu", revision=GPT2_REVISION)
    reference = GPT2LMHeadModel.from_pretrained(
        "gpt2", revision=GPT2_REVISION, dtype=torch.float64, attn_implementation="eager"
    ).eval()
    return ours, reference


@pytest.mark.parametrize("shape", [(2, 64), (1, 1024)], ids=["batch-2x64", "full-context-1024"])
def test_logits_match_hugging_face(models: tuple[GPT, GPT2LMHeadModel], shape: tuple[int, int]) -> None:
    ours, reference = models
    generator = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, ours.config.vocab_size, shape, generator=generator)
    with torch.no_grad():
        our_logits = ours(input_ids)
        hf_logits = reference(input_ids).logits
    assert our_logits.dtype == hf_logits.dtype == torch.float64
    torch.testing.assert_close(our_logits, hf_logits, rtol=0, atol=1e-9)


def hugging_face_greedy(
    reference: GPT2LMHeadModel, prompt: torch.Tensor, new_tokens: int
) -> torch.Tensor:
    # Set every generation parameter on a copy, so the shared oracle keeps its config and
    # generate receives no loose generation arguments next to the config. eos_token_id=None
    # disables the end-of-text stop.
    generation_config = copy(reference.generation_config)
    generation_config.max_new_tokens = new_tokens
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.pad_token_id = 50256
    generation_config.eos_token_id = None
    hf_output = reference.generate(
        prompt,
        attention_mask=torch.ones_like(prompt),
        generation_config=generation_config,
    )

    expected_length = prompt.shape[1] + new_tokens
    if hf_output.shape[1] != expected_length:
        raise AssertionError(
            f"Hugging Face stopped early: got length {hf_output.shape[1]}, "
            f"expected {expected_length}. The end-of-text stop is still active."
        )
    return hf_output


@GENERATION_CASES
@pytest.mark.parametrize(
    "generate", [generate_greedy_uncached, generate_greedy_cached], ids=["uncached", "cached"]
)
def test_greedy_generation_matches_hugging_face(
    models: tuple[GPT, GPT2LMHeadModel],
    prompt_shape: tuple[int, int],
    new_tokens: int,
    generate: Callable[[GPT, torch.Tensor, int], torch.Tensor],
) -> None:
    ours, reference = models
    generator = torch.Generator().manual_seed(1234)
    prompt = torch.randint(0, ours.config.vocab_size, prompt_shape, generator=generator)
    our_output = generate(ours, prompt, new_tokens)
    assert torch.equal(our_output, hugging_face_greedy(reference, prompt, new_tokens))


def test_pretrained_cached_logits_match_uncached_forward(
    models: tuple[GPT, GPT2LMHeadModel],
) -> None:
    ours, _ = models
    generator = torch.Generator().manual_seed(1234)
    prompt = torch.randint(0, ours.config.vocab_size, (1, 1000), generator=generator)
    decode_steps = 24

    cache = DynamicKVCache(ours.config.n_layer)
    with torch.no_grad():
        logits = ours(prompt, cache=cache)[:, -1:, :].clone()
        step_logits = [logits]
        sequence = [prompt]
        for _ in range(decode_steps):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sequence.append(next_token)
            logits = ours(next_token, cache=cache)
            step_logits.append(logits)
        # The 1024 tokens fed to the model: the prompt and the 24 decoded tokens.
        full = ours(torch.cat(sequence, dim=1))[:, prompt.shape[1] - 1 :, :]

    assert cache.seq_len == ours.config.n_positions
    torch.testing.assert_close(torch.cat(step_logits, dim=1), full, atol=1e-9, rtol=0)
