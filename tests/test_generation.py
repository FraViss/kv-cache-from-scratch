from collections.abc import Callable

import pytest
import torch

from kvcache import (
    GPT,
    DynamicKVCache,
    GPT2Config,
    generate_greedy_cached,
    generate_greedy_uncached,
)

SMALL = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)
GENERATORS = pytest.mark.parametrize(
    "generate", [generate_greedy_uncached, generate_greedy_cached], ids=["uncached", "cached"]
)


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(SMALL).to(torch.float64).eval()


def random_ids(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, SMALL.vocab_size, (batch, length), generator=generator)


def test_shape_and_prompt_are_preserved(model: GPT) -> None:
    prompt = random_ids(2, 5)
    original = prompt.clone()
    output = generate_greedy_uncached(model, prompt, 4)
    assert output.shape == (2, 9)
    assert torch.equal(output[:, :5], original)
    assert torch.equal(prompt, original)


def test_each_generated_token_is_the_argmax(model: GPT) -> None:
    prompt_length = 5
    output = generate_greedy_uncached(model, random_ids(2, prompt_length), 4)
    with torch.no_grad():
        for i in range(4):
            expected = model(output[:, : prompt_length + i])[:, -1, :].argmax(dim=-1)
            assert torch.equal(output[:, prompt_length + i], expected)


def test_zero_new_tokens_returns_a_copy(model: GPT) -> None:
    prompt = random_ids(2, 5)
    output = generate_greedy_uncached(model, prompt, 0)
    assert torch.equal(output, prompt)
    assert output is not prompt


@GENERATORS
def test_rejects_negative_max_new_tokens(model: GPT, generate: Callable) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(model, random_ids(1, 4), -1)


@GENERATORS
def test_rejects_empty_prompt(model: GPT, generate: Callable) -> None:
    empty = torch.zeros(2, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="P >= 1"):
        generate(model, empty, 1)


@GENERATORS
def test_rejects_generation_past_n_positions(model: GPT, generate: Callable) -> None:
    with pytest.raises(ValueError, match="no sliding window"):
        generate(model, random_ids(1, 30), 3)


@GENERATORS
def test_accepts_generation_up_to_n_positions(model: GPT, generate: Callable) -> None:
    output = generate(model, random_ids(1, 30), 2)
    assert output.shape == (1, SMALL.n_positions)


@GENERATORS
def test_model_stays_in_eval(model: GPT, generate: Callable) -> None:
    assert not model.training
    generate(model, random_ids(1, 4), 2)
    assert not model.training


@GENERATORS
def test_output_is_not_an_inference_tensor(model: GPT, generate: Callable) -> None:
    assert not generate(model, random_ids(2, 4), 3).is_inference()
    assert not generate(model, random_ids(2, 4), 0).is_inference()


def test_cached_logits_match_uncached_forward(model: GPT) -> None:
    prompt_length, new_tokens = 5, 6
    output = generate_greedy_cached(model, random_ids(2, prompt_length), new_tokens)
    # The last generated token is never fed to the model.
    fed = output[:, :-1]

    cache = DynamicKVCache(SMALL.n_layer)
    with torch.no_grad():
        full = model(fed)
        prefill = model(fed[:, :prompt_length], cache=cache)
        torch.testing.assert_close(prefill, full[:, :prompt_length], atol=1e-10, rtol=0)
        for position in range(prompt_length, fed.shape[1]):
            step = model(fed[:, position : position + 1], cache=cache)
            assert step.shape == (2, 1, SMALL.vocab_size)
            torch.testing.assert_close(step, full[:, position : position + 1], atol=1e-10, rtol=0)


def test_chunked_prefill_matches_full_forward(model: GPT) -> None:
    ids = random_ids(2, 12)
    cache = DynamicKVCache(SMALL.n_layer)
    with torch.no_grad():
        full = model(ids)
        chunks = [model(chunk, cache=cache) for chunk in ids.split([5, 4, 3], dim=1)]
    assert cache.seq_len == 12
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-10, rtol=0)


@pytest.mark.parametrize("new_tokens", [1, 4, SMALL.n_positions - 5], ids=["n1", "n4", "to-limit"])
def test_cached_generation_matches_uncached(model: GPT, new_tokens: int) -> None:
    prompt = random_ids(2, 5)
    cached = generate_greedy_cached(model, prompt, new_tokens)
    uncached = generate_greedy_uncached(model, prompt, new_tokens)
    assert torch.equal(cached, uncached)


def test_cache_holds_all_but_the_last_token(model: GPT) -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    generate_greedy_cached(model, random_ids(2, 5), 4, cache=cache)
    assert cache.seq_len == 5 + 4 - 1


def test_cached_zero_new_tokens_runs_no_forward(model: GPT) -> None:
    calls = []
    model.register_forward_hook(lambda *_: calls.append(1))
    prompt = random_ids(2, 5)
    cache = DynamicKVCache(SMALL.n_layer)
    output = generate_greedy_cached(model, prompt, 0, cache=cache)
    assert calls == []
    assert cache.seq_len == 0
    assert cache.nbytes == 0
    assert torch.equal(output, prompt)
    assert output is not prompt


def test_cached_rejects_non_empty_cache(model: GPT) -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    with torch.no_grad():
        model(random_ids(2, 3), cache=cache)
    with pytest.raises(ValueError, match="cache must be empty"):
        generate_greedy_cached(model, random_ids(2, 5), 2, cache=cache)
