import pytest
import torch

from kvcache import GPT, GPT2Config

SMALL = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(SMALL).to(torch.float64).eval()


def random_ids(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, SMALL.vocab_size, (batch, length), generator=generator)


def test_logits_shape(model: GPT) -> None:
    with torch.no_grad():
        logits = model(random_ids(2, 5))
    assert logits.shape == (2, 5, SMALL.vocab_size)


def test_causality(model: GPT) -> None:
    ids = random_ids(2, 5)
    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % SMALL.vocab_size
    with torch.no_grad():
        logits = model(ids)
        logits_changed = model(changed)
    torch.testing.assert_close(logits_changed[:, :-1], logits[:, :-1], atol=1e-12, rtol=0)
    assert not torch.allclose(logits_changed[:, -1], logits[:, -1])


def test_lm_head_is_tied_to_wte(model: GPT) -> None:
    assert model.lm_head.weight is model.wte.weight


def test_gpt2_small_parameter_count() -> None:
    with torch.device("meta"):
        model = GPT(GPT2Config())
    assert sum(p.numel() for p in model.parameters()) == 124_439_808


def test_rejects_sequence_past_n_positions(model: GPT) -> None:
    with pytest.raises(ValueError, match="n_positions"):
        model(random_ids(1, 4), position_offset=SMALL.n_positions - 3)


def test_rejects_negative_position_offset(model: GPT) -> None:
    with pytest.raises(ValueError, match="position_offset"):
        model(random_ids(1, 4), position_offset=-1)


@pytest.mark.parametrize(
    "input_ids",
    [
        torch.zeros(2, 5, dtype=torch.int32),
        torch.zeros(2, 5, dtype=torch.float64),
        torch.zeros(5, dtype=torch.long),
        torch.zeros(1, 2, 5, dtype=torch.long),
    ],
    ids=["int32", "float64", "1d", "3d"],
)
def test_rejects_malformed_input_ids(model: GPT, input_ids: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="input_ids"):
        model(input_ids)
