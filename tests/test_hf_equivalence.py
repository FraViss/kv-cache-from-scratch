from copy import copy

import pytest
import torch
from transformers import GPT2LMHeadModel

from kvcache import GPT, generate_greedy_uncached, load_pretrained_gpt2
from kvcache.weights import GPT2_REVISION

pytestmark = pytest.mark.integration


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


@pytest.mark.parametrize(
    ("prompt_shape", "new_tokens"),
    [((2, 16), 48), ((1, 1000), 24)],
    ids=["prompt-2x16-plus-48", "prompt-1x1000-plus-24"],
)
def test_greedy_generation_matches_hugging_face(
    models: tuple[GPT, GPT2LMHeadModel], prompt_shape: tuple[int, int], new_tokens: int
) -> None:
    ours, reference = models
    generator = torch.Generator().manual_seed(1234)
    prompt = torch.randint(0, ours.config.vocab_size, prompt_shape, generator=generator)
    our_output = generate_greedy_uncached(ours, prompt, new_tokens)

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

    expected_length = prompt_shape[1] + new_tokens
    if hf_output.shape[1] != expected_length:
        raise AssertionError(
            f"Hugging Face stopped early: got length {hf_output.shape[1]}, "
            f"expected {expected_length}. The end-of-text stop is still active."
        )
    assert torch.equal(our_output, hf_output)
