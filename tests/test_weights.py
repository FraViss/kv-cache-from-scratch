import pytest
import torch
from torch import Tensor

from kvcache import GPT, GPT2Config
from kvcache.weights import convert_hf_gpt2_state_dict, load_gpt2_weights

SMALL = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)
CONV1D_WEIGHTS = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def build_model(seed: int) -> GPT:
    torch.manual_seed(seed)
    return GPT(SMALL).to(torch.float64).eval()


def hf_layout_state_dict(model: GPT, prefix: str = "") -> dict[str, Tensor]:
    """Export model in the Hugging Face GPT-2 layout, with the extras HF checkpoints carry."""
    sd: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        tensor = param.detach().clone()
        if name.endswith(CONV1D_WEIGHTS):
            tensor = tensor.t().contiguous()
        sd[prefix + name] = tensor
    n = SMALL.n_positions
    sd[prefix + "h.0.attn.bias"] = torch.ones(n, n, dtype=torch.bool).tril().view(1, 1, n, n)
    sd["lm_head.weight"] = model.wte.weight.detach().clone()
    return sd


def converted_reference() -> dict[str, Tensor]:
    return convert_hf_gpt2_state_dict(hf_layout_state_dict(build_model(seed=0)))


def assert_same_logits(a: GPT, b: GPT) -> None:
    ids = torch.randint(0, SMALL.vocab_size, (2, 7), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        torch.testing.assert_close(b(ids), a(ids), atol=1e-12, rtol=0)


@pytest.mark.parametrize("prefix", ["", "transformer."], ids=["no-prefix", "transformer-prefix"])
def test_round_trip_reproduces_logits(prefix: str) -> None:
    reference = build_model(seed=0)
    target = build_model(seed=1)
    load_gpt2_weights(target, convert_hf_gpt2_state_dict(hf_layout_state_dict(reference, prefix)))
    assert_same_logits(reference, target)


def test_converter_drops_mask_buffer_and_lm_head() -> None:
    sd = converted_reference()
    assert "h.0.attn.bias" not in sd
    assert "lm_head.weight" not in sd
    assert sd.keys() == dict(build_model(seed=0).named_parameters()).keys()


def test_missing_keys_are_all_reported() -> None:
    sd = converted_reference()
    del sd["ln_f.bias"], sd["h.1.mlp.c_fc.weight"]
    with pytest.raises(ValueError, match="missing") as excinfo:
        load_gpt2_weights(build_model(seed=1), sd)
    assert "ln_f.bias" in str(excinfo.value)
    assert "h.1.mlp.c_fc.weight" in str(excinfo.value)


def test_unexpected_key_is_rejected() -> None:
    sd = converted_reference()
    sd["h.0.attn.extra"] = torch.zeros(1)
    with pytest.raises(ValueError, match=r"unexpected keys \['h\.0\.attn\.extra'\]"):
        load_gpt2_weights(build_model(seed=1), sd)


def test_shape_mismatch_is_rejected() -> None:
    sd = converted_reference()
    sd["h.0.attn.c_proj.weight"] = sd["h.0.attn.c_proj.weight"][:-1]
    with pytest.raises(ValueError, match=r"h\.0\.attn\.c_proj\.weight.*\(64, 64\).*\(63, 64\)"):
        load_gpt2_weights(build_model(seed=1), sd)
