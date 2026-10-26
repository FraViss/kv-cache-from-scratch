import math

import pytest
import torch
from torch import Tensor

from kvcache import GPT, DynamicKVCache, GPT2Config

SMALL = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)
B = 2


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(SMALL).to(torch.float64).eval()


def random_ids(batch: int, length: int, seed: int = 0) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, SMALL.vocab_size, (batch, length), generator=generator)


def random_kv(length: int, seed: int = 0, **shape_overrides: int) -> tuple[Tensor, Tensor]:
    """Random k and v shaped (B, n_head, length, head_dim), with optional dimension overrides."""
    shape = (
        shape_overrides.get("batch", B),
        shape_overrides.get("n_head", SMALL.n_head),
        length,
        shape_overrides.get("head_dim", SMALL.head_dim),
    )
    generator = torch.Generator().manual_seed(seed)
    k = torch.randn(shape, generator=generator, dtype=torch.float64)
    v = torch.randn(shape, generator=generator, dtype=torch.float64)
    return k, v


def test_update_on_empty_cache_copies_the_input(model: GPT) -> None:
    x = torch.randn(B, 5, SMALL.n_embd, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        _, k, v = model.h[0].attn.project_qkv(x)
    cache = DynamicKVCache(SMALL.n_layer)
    cached_k, cached_v = cache.update(0, k, v)
    assert torch.equal(cached_k, k)
    assert torch.equal(cached_v, v)
    assert cache.seq_len == 5
    # k and v are views into the c_attn output, which also holds q. The cache must not
    # keep that storage alive.
    assert cached_k.untyped_storage().data_ptr() != k.untyped_storage().data_ptr()
    assert cached_v.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()


def test_successive_updates_concatenate() -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    pieces = [random_kv(length, seed=length) for length in (3, 1, 4)]
    total = 0
    for i, (k, v) in enumerate(pieces):
        total += k.shape[-2]
        for layer_idx in range(SMALL.n_layer):
            cached_k, cached_v = cache.update(layer_idx, k, v)
            assert torch.equal(cached_k, torch.cat([p[0] for p in pieces[: i + 1]], dim=-2))
            assert torch.equal(cached_v, torch.cat([p[1] for p in pieces[: i + 1]], dim=-2))
        assert cache.seq_len == total


def test_nbytes_after_model_forward(model: GPT) -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    T = 7
    with torch.no_grad():
        model(random_ids(B, T), cache=cache)
    expected = 2 * SMALL.n_layer * B * SMALL.n_head * T * SMALL.head_dim * torch.float64.itemsize
    assert cache.nbytes == expected


def test_reset_empties_the_cache() -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    for layer_idx in range(SMALL.n_layer):
        cache.update(layer_idx, *random_kv(4))
    assert cache.seq_len == 4
    cache.reset()
    assert cache.seq_len == 0
    assert cache.nbytes == 0


@pytest.mark.parametrize("layer_idx", [-1, SMALL.n_layer])
def test_rejects_layer_idx_out_of_range(layer_idx: int) -> None:
    with pytest.raises(ValueError, match="layer_idx"):
        DynamicKVCache(SMALL.n_layer).update(layer_idx, *random_kv(3))


def test_rejects_non_4d_tensors() -> None:
    k, v = random_kv(3)
    with pytest.raises(ValueError, match="4D"):
        DynamicKVCache(SMALL.n_layer).update(0, k[0], v[0])


def test_rejects_k_and_v_with_different_shapes() -> None:
    k, v = random_kv(3)
    with pytest.raises(ValueError, match="same shape"):
        DynamicKVCache(SMALL.n_layer).update(0, k, v[..., :-1])


def test_rejects_k_and_v_with_different_dtypes() -> None:
    k, v = random_kv(3)
    with pytest.raises(ValueError, match="share dtype and device"):
        DynamicKVCache(SMALL.n_layer).update(0, k, v.float())


@pytest.mark.parametrize("dimension", ["batch", "n_head", "head_dim"])
def test_rejects_shape_different_from_the_cache(dimension: str) -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    cache.update(0, *random_kv(3))
    other = {"batch": B + 1, "n_head": SMALL.n_head + 1, "head_dim": SMALL.head_dim + 1}
    name = "B" if dimension == "batch" else dimension
    with pytest.raises(ValueError, match=f"^{name} of the new keys"):
        cache.update(0, *random_kv(1, **{dimension: other[dimension]}))


def test_rejects_dtype_different_from_the_cache() -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    cache.update(0, *random_kv(3))
    k, v = random_kv(1)
    with pytest.raises(ValueError, match="dtype torch.float32 does not match"):
        cache.update(0, k.float(), v.float())


def test_rejects_device_different_from_the_cache() -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    cache.update(0, *random_kv(3))
    k, v = random_kv(1)
    with pytest.raises(ValueError, match="device meta does not match"):
        cache.update(0, k.to("meta"), v.to("meta"))


def test_rejects_layer_updated_twice() -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    for layer_idx in range(SMALL.n_layer):
        cache.update(layer_idx, *random_kv(3))
    nbytes = cache.nbytes
    with pytest.raises(ValueError, match="updated twice or skipped"):
        cache.update(1, *random_kv(3))
    # A rejected update leaves the cache as it was.
    assert cache.nbytes == nbytes


def test_rejects_layer_skipped() -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    for layer_idx in range(SMALL.n_layer):
        cache.update(layer_idx, *random_kv(3))
    cache.update(0, *random_kv(1))  # layer 1 is skipped in this forward
    cache.update(0, *random_kv(1))
    with pytest.raises(ValueError, match="updated twice or skipped"):
        cache.update(1, *random_kv(1))


def test_rejects_later_layer_before_layer_0() -> None:
    with pytest.raises(ValueError, match="updated twice or skipped"):
        DynamicKVCache(SMALL.n_layer).update(1, *random_kv(3))


def test_model_rejects_position_offset_with_cache(model: GPT) -> None:
    with pytest.raises(ValueError, match="position_offset must be 0"):
        model(random_ids(1, 3), position_offset=2, cache=DynamicKVCache(SMALL.n_layer))


def test_model_rejects_cache_past_n_positions(model: GPT) -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    with torch.no_grad():
        model(random_ids(1, 30), cache=cache)
        with pytest.raises(ValueError, match="30 \\+ 3 = 33 exceeds n_positions"):
            model(random_ids(1, 3, seed=1), cache=cache)


def test_attention_requires_layer_idx_with_cache(model: GPT) -> None:
    x = torch.zeros(1, 3, SMALL.n_embd, dtype=torch.float64)
    with pytest.raises(ValueError, match="layer_idx"):
        model.h[0].attn(x, DynamicKVCache(SMALL.n_layer))


def test_attend_with_more_keys_than_queries(model: GPT) -> None:
    Tq, Tk = 3, 8
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(B, SMALL.n_head, Tq, SMALL.head_dim, generator=generator, dtype=torch.float64)
    k = torch.randn(B, SMALL.n_head, Tk, SMALL.head_dim, generator=generator, dtype=torch.float64)
    v = torch.randn(B, SMALL.n_head, Tk, SMALL.head_dim, generator=generator, dtype=torch.float64)

    # Query i sits at absolute position i + Tk - Tq and sees keys 0 through that position.
    expected = torch.empty(B, SMALL.n_head, Tq, SMALL.head_dim, dtype=torch.float64)
    for i in range(Tq):
        visible = i + Tk - Tq + 1
        scores = q[:, :, i, :].unsqueeze(-2) @ k[:, :, :visible, :].transpose(-2, -1)
        weights = torch.softmax(scores / math.sqrt(SMALL.head_dim), dim=-1)
        expected[:, :, i, :] = (weights @ v[:, :, :visible, :]).squeeze(-2)
    expected = expected.transpose(1, 2).reshape(B, Tq, SMALL.n_embd)

    torch.testing.assert_close(model.h[0].attn.attend(q, k, v), expected, atol=1e-12, rtol=0)
