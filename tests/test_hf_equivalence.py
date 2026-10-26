import pytest
import torch
from transformers import GPT2LMHeadModel

from kvcache import GPT, load_pretrained_gpt2

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def models() -> tuple[GPT, GPT2LMHeadModel]:
    ours = load_pretrained_gpt2(dtype=torch.float64, device="cpu")
    reference = GPT2LMHeadModel.from_pretrained(
        "gpt2", dtype=torch.float64, attn_implementation="eager"
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
