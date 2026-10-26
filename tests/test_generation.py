import pytest
import torch

from kvcache import GPT, GPT2Config, generate_greedy_uncached

SMALL = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)


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


def test_rejects_negative_max_new_tokens(model: GPT) -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_greedy_uncached(model, random_ids(1, 4), -1)


def test_rejects_empty_prompt(model: GPT) -> None:
    empty = torch.zeros(2, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="P >= 1"):
        generate_greedy_uncached(model, empty, 1)


def test_rejects_generation_past_n_positions(model: GPT) -> None:
    with pytest.raises(ValueError, match="no sliding window"):
        generate_greedy_uncached(model, random_ids(1, 30), 3)


def test_accepts_generation_up_to_n_positions(model: GPT) -> None:
    output = generate_greedy_uncached(model, random_ids(1, 30), 2)
    assert output.shape == (1, SMALL.n_positions)


def test_model_stays_in_eval(model: GPT) -> None:
    assert not model.training
    generate_greedy_uncached(model, random_ids(1, 4), 2)
    assert not model.training
