import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from kvcache import (
    GPT,
    DynamicKVCache,
    GPT2Config,
    CapturableKVCache,
    KVCache,
    StaticKVCache,
    generate_greedy_cached,
    generate_greedy_uncached,
)

SMALL = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)
B = 2
CacheFactory = Callable[[GPT2Config, int], KVCache]


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


def update(cache: KVCache, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """cache.update with the positions the new tokens would get in a forward pass.

    Layer 0 is updated first in a forward, so the other layers start where layer 0 did.
    """
    new = k.shape[-2]
    start = cache.seq_len if layer_idx == 0 else cache.seq_len - new
    return cache.update(layer_idx, k, v, torch.arange(start, start + new, device=k.device))


def filled(t: Tensor, cache: KVCache) -> Tensor:
    """The written part of keys or values returned by update. A capturable cache returns
    its whole buffer; the other caches return exactly the written tokens."""
    return t[:, :, : cache.seq_len]


def test_update_on_empty_cache_copies_the_input(model: GPT, cache_factory: CacheFactory) -> None:
    x = torch.randn(B, 5, SMALL.n_embd, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        _, k, v = model.h[0].attn.project_qkv(x)
    cache = cache_factory(SMALL, B)
    cached_k, cached_v = update(cache, 0, k, v)
    assert torch.equal(filled(cached_k, cache), k)
    assert torch.equal(filled(cached_v, cache), v)
    assert cache.seq_len == 5
    # k and v are views into the c_attn output, which also holds q. The cache must not
    # keep that storage alive.
    assert cached_k.untyped_storage().data_ptr() != k.untyped_storage().data_ptr()
    assert cached_v.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()


def test_successive_updates_concatenate(cache_factory: CacheFactory) -> None:
    cache = cache_factory(SMALL, B)
    pieces = [random_kv(length, seed=length) for length in (3, 1, 4)]
    total = 0
    for i, (k, v) in enumerate(pieces):
        total += k.shape[-2]
        for layer_idx in range(SMALL.n_layer):
            cached_k, cached_v = update(cache, layer_idx, k, v)
            expected_k = torch.cat([p[0] for p in pieces[: i + 1]], dim=-2)
            expected_v = torch.cat([p[1] for p in pieces[: i + 1]], dim=-2)
            assert torch.equal(filled(cached_k, cache), expected_k)
            assert torch.equal(filled(cached_v, cache), expected_v)
        assert cache.seq_len == total


def test_nbytes_after_model_forward(model: GPT) -> None:
    cache = DynamicKVCache(SMALL.n_layer)
    T = 7
    with torch.no_grad():
        model(random_ids(B, T), cache=cache)
    expected = 2 * SMALL.n_layer * B * SMALL.n_head * T * SMALL.head_dim * torch.float64.itemsize
    assert cache.nbytes == expected
    assert cache.used_nbytes == expected


def test_reset_empties_the_cache(cache_factory: CacheFactory) -> None:
    cache = cache_factory(SMALL, B)
    for layer_idx in range(SMALL.n_layer):
        update(cache, layer_idx, *random_kv(4))
    assert cache.seq_len == 4
    nbytes = cache.nbytes
    cache.reset()
    assert cache.seq_len == 0
    assert cache.used_nbytes == 0
    # A dynamic cache frees its tensors; a static one keeps its buffers allocated.
    assert cache.nbytes == (0 if cache.capacity is None else nbytes)
    k, v = random_kv(2, seed=1)
    cached_k, cached_v = update(cache, 0, k, v)
    assert torch.equal(filled(cached_k, cache), k)
    assert torch.equal(filled(cached_v, cache), v)


@pytest.mark.parametrize("layer_idx", [-1, SMALL.n_layer])
def test_rejects_layer_idx_out_of_range(cache_factory: CacheFactory, layer_idx: int) -> None:
    with pytest.raises(ValueError, match="layer_idx"):
        update(cache_factory(SMALL, B), layer_idx, *random_kv(3))


def test_rejects_non_4d_tensors(cache_factory: CacheFactory) -> None:
    k, v = random_kv(3)
    with pytest.raises(ValueError, match="4D"):
        update(cache_factory(SMALL, B), 0, k[0], v[0])


def test_rejects_k_and_v_with_different_shapes(cache_factory: CacheFactory) -> None:
    k, v = random_kv(3)
    with pytest.raises(ValueError, match="same shape"):
        update(cache_factory(SMALL, B), 0, k, v[..., :-1])


def test_rejects_k_and_v_with_different_dtypes(cache_factory: CacheFactory) -> None:
    k, v = random_kv(3)
    with pytest.raises(ValueError, match="share dtype and device"):
        update(cache_factory(SMALL, B), 0, k, v.float())


@pytest.mark.parametrize("dimension", ["batch", "n_head", "head_dim"])
def test_rejects_shape_different_from_the_cache(cache_factory: CacheFactory, dimension: str) -> None:
    cache = cache_factory(SMALL, B)
    update(cache, 0, *random_kv(3))
    other = {"batch": B + 1, "n_head": SMALL.n_head + 1, "head_dim": SMALL.head_dim + 1}
    name = "B" if dimension == "batch" else dimension
    with pytest.raises(ValueError, match=f"^{name} of the new keys"):
        update(cache, 0, *random_kv(1, **{dimension: other[dimension]}))


def test_rejects_dtype_different_from_the_cache(cache_factory: CacheFactory) -> None:
    cache = cache_factory(SMALL, B)
    update(cache, 0, *random_kv(3))
    k, v = random_kv(1)
    with pytest.raises(ValueError, match="dtype torch.float32 does not match"):
        update(cache, 0, k.float(), v.float())


def test_rejects_device_different_from_the_cache(cache_factory: CacheFactory) -> None:
    cache = cache_factory(SMALL, B)
    update(cache, 0, *random_kv(3))
    k, v = random_kv(1)
    with pytest.raises(ValueError, match="device meta does not match"):
        update(cache, 0, k.to("meta"), v.to("meta"))


def test_rejects_layer_updated_twice(cache_factory: CacheFactory) -> None:
    cache = cache_factory(SMALL, B)
    for layer_idx in range(SMALL.n_layer):
        update(cache, layer_idx, *random_kv(3))
    used_nbytes = cache.used_nbytes
    with pytest.raises(ValueError, match="updated twice or skipped"):
        update(cache, 1, *random_kv(3))
    # A rejected update leaves the cache as it was.
    assert cache.seq_len == 3
    assert cache.used_nbytes == used_nbytes


def test_rejects_layer_skipped(cache_factory: CacheFactory) -> None:
    cache = cache_factory(SMALL, B)
    for layer_idx in range(SMALL.n_layer):
        update(cache, layer_idx, *random_kv(3))
    update(cache, 0, *random_kv(1))  # layer 1 is skipped in this forward
    update(cache, 0, *random_kv(1))
    with pytest.raises(ValueError, match="updated twice or skipped"):
        update(cache, 1, *random_kv(1))


def test_rejects_later_layer_before_layer_0(cache_factory: CacheFactory) -> None:
    with pytest.raises(ValueError, match="updated twice or skipped"):
        update(cache_factory(SMALL, B), 1, *random_kv(3))


def test_model_rejects_position_offset_with_cache(model: GPT, cache_factory: CacheFactory) -> None:
    with pytest.raises(ValueError, match="position_offset must be 0"):
        model(random_ids(1, 3), position_offset=2, cache=cache_factory(SMALL, 1))


def test_model_rejects_cache_past_n_positions(model: GPT, cache_factory: CacheFactory) -> None:
    cache = cache_factory(SMALL, 1)
    with torch.no_grad():
        model(random_ids(1, 30), cache=cache)
        with pytest.raises(ValueError, match="30 \\+ 3 = 33 exceeds n_positions"):
            model(random_ids(1, 3, seed=1), cache=cache)


def test_attention_requires_layer_idx_with_cache(model: GPT) -> None:
    x = torch.zeros(1, 3, SMALL.n_embd, dtype=torch.float64)
    with pytest.raises(ValueError, match="layer_idx"):
        model.h[0].attn(x, torch.arange(3), DynamicKVCache(SMALL.n_layer))


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

    q_positions = torch.arange(Tk - Tq, Tk)
    actual = model.h[0].attn.attend(q, k, v, q_positions)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def static_cache(batch_size: int = B, max_seq_len: int | None = None) -> StaticKVCache:
    return StaticKVCache.from_config(SMALL, batch_size, max_seq_len, dtype=torch.float64)


def test_static_cache_allocates_full_capacity_up_front() -> None:
    cache = static_cache()
    itemsize = torch.float64.itemsize
    max_seq_len = SMALL.n_positions
    assert cache.capacity == max_seq_len
    per_token = B * SMALL.n_head * SMALL.head_dim * itemsize
    assert cache.nbytes == 2 * SMALL.n_layer * max_seq_len * per_token
    assert cache.used_nbytes == 0
    assert cache.seq_len == 0


def test_static_cache_used_nbytes_after_model_forward(model: GPT) -> None:
    cache = static_cache()
    nbytes = cache.nbytes
    T = 7
    with torch.no_grad():
        model(random_ids(B, T), cache=cache)
    itemsize = torch.float64.itemsize
    assert cache.used_nbytes == 2 * SMALL.n_layer * B * SMALL.n_head * T * SMALL.head_dim * itemsize
    assert cache.nbytes == nbytes


def test_static_cache_writes_in_place(model: GPT) -> None:
    cache = static_cache()
    pointers = []
    original_update = cache.update

    def recording_update(
        layer_idx: int, k: Tensor, v: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        cached_k, cached_v = original_update(layer_idx, k, v, positions)
        if layer_idx == 0:
            pointers.append(cached_k.data_ptr())
        return cached_k, cached_v

    cache.update = recording_update
    ids = random_ids(B, 15)
    with torch.no_grad():
        model(ids[:, :10], cache=cache)
        for position in range(10, 15):
            model(ids[:, position : position + 1], cache=cache)
    assert len(pointers) == 6
    assert set(pointers) == {cache._keys.data_ptr()}


def test_static_cache_rejects_update_past_capacity() -> None:
    cache = static_cache(max_seq_len=5)
    for layer_idx in range(SMALL.n_layer):
        update(cache, layer_idx, *random_kv(4))
    keys, values = cache._keys.clone(), cache._values.clone()
    with pytest.raises(ValueError, match="over the cache capacity of 5"):
        update(cache, 0, *random_kv(2, seed=1))
    assert cache.seq_len == 4
    assert torch.equal(cache._keys, keys)
    assert torch.equal(cache._values, values)


def test_static_cache_rejects_dtype_different_from_the_buffer() -> None:
    cache = static_cache()
    k, v = random_kv(3)
    with pytest.raises(ValueError, match="float32 does not match the cache dtype torch.float64"):
        update(cache, 0, k.float(), v.float())
    assert cache.seq_len == 0


def test_static_cache_hides_stale_data_after_reset(model: GPT) -> None:
    reused = static_cache()
    fresh = static_cache()
    sequence_b = random_ids(B, 6, seed=2)
    with torch.no_grad():
        model(random_ids(B, 20, seed=1), cache=reused)
        reused.reset()
        logits_reused = model(sequence_b, cache=reused)
        logits_fresh = model(sequence_b, cache=fresh)
    assert torch.equal(logits_reused, logits_fresh)


def test_static_cache_is_reusable_outside_inference_mode(model: GPT) -> None:
    cache = static_cache()
    generate_greedy_cached(model, random_ids(B, 5), 4, cache=cache)
    cache.reset()
    with torch.no_grad():
        model(random_ids(B, 5, seed=1), cache=cache)
    assert cache.seq_len == 5
    assert not cache._keys.is_inference()
    assert not cache._values.is_inference()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_static_cache_on_cuda(model: GPT) -> None:
    model = model.to("cuda")
    prompt = random_ids(B, 5).to("cuda")
    cache = StaticKVCache.from_config(SMALL, B, dtype=torch.float64, device="cuda")
    cached = generate_greedy_cached(model, prompt, 20, cache=cache)
    assert torch.equal(cached, generate_greedy_uncached(model, prompt, 20))


@pytest.mark.parametrize("query_positions", [[5, 6, 7], [7]], ids=["tq3", "tq1"])
def test_attend_ignores_unwritten_slots_of_a_buffer(
    model: GPT, query_positions: list[int]
) -> None:
    written, capacity = 8, 16
    generator = torch.Generator().manual_seed(0)
    shape = (B, SMALL.n_head, capacity, SMALL.head_dim)
    Tq = len(query_positions)
    q = torch.randn(B, SMALL.n_head, Tq, SMALL.head_dim, generator=generator, dtype=torch.float64)
    k = torch.randn(shape, generator=generator, dtype=torch.float64)
    v = torch.randn(shape, generator=generator, dtype=torch.float64)
    # Slots past the written tokens hold large but finite garbage, as a reused buffer would.
    garbage = (B, SMALL.n_head, capacity - written, SMALL.head_dim)
    k[:, :, written:] = torch.randn(garbage, generator=generator, dtype=torch.float64) * 1e3
    v[:, :, written:] = torch.randn(garbage, generator=generator, dtype=torch.float64) * 1e3
    q_positions = torch.tensor(query_positions)

    attn = model.h[0].attn
    over_buffer = attn.attend(q, k, v, q_positions)
    over_written = attn.attend(q, k[:, :, :written], v[:, :, :written], q_positions)
    torch.testing.assert_close(over_buffer, over_written, atol=1e-12, rtol=0)


@pytest.mark.parametrize("cache_class", [StaticKVCache, CapturableKVCache])
def test_truncate_keeps_the_first_tokens(cache_class: type[StaticKVCache]) -> None:
    cache = cache_class.from_config(SMALL, B, dtype=torch.float64)
    pieces = [random_kv(4), random_kv(3, seed=1)]
    for k, v in pieces:
        for layer_idx in range(SMALL.n_layer):
            update(cache, layer_idx, k, v)
    cache.truncate(4)
    assert cache.seq_len == 4
    itemsize = torch.float64.itemsize
    assert cache.used_nbytes == 2 * SMALL.n_layer * 4 * B * SMALL.n_head * SMALL.head_dim * itemsize
    k, v = random_kv(2, seed=2)
    for layer_idx in range(SMALL.n_layer):
        cached_k, cached_v = update(cache, layer_idx, k, v)
        assert torch.equal(filled(cached_k, cache), torch.cat((pieces[0][0], k), dim=-2))
        assert torch.equal(filled(cached_v, cache), torch.cat((pieces[0][1], v), dim=-2))
    cache.truncate(0)
    assert cache.seq_len == 0


@pytest.mark.parametrize("cache_class", [StaticKVCache, CapturableKVCache])
def test_truncate_rejects_invalid_lengths(cache_class: type[StaticKVCache]) -> None:
    cache = cache_class.from_config(SMALL, B, dtype=torch.float64)
    for layer_idx in range(SMALL.n_layer):
        update(cache, layer_idx, *random_kv(4))
    for seq_len in (-1, 5):
        with pytest.raises(ValueError, match=r"seq_len must be in \[0, 4\]"):
            cache.truncate(seq_len)
    update(cache, 0, *random_kv(1))  # layer 0 now runs ahead of layer 1
    with pytest.raises(ValueError, match="different lengths"):
        cache.truncate(2)
    assert cache.seq_len == 5


@pytest.mark.parametrize("cache_class", [StaticKVCache, CapturableKVCache])
def test_truncate_then_forward_matches_a_fresh_cache(
    model: GPT, cache_class: type[StaticKVCache]
) -> None:
    reused = cache_class.from_config(SMALL, B, dtype=torch.float64)
    fresh = cache_class.from_config(SMALL, B, dtype=torch.float64)
    prefix = random_ids(B, 6, seed=1)
    new_tokens = random_ids(B, 4, seed=3)
    with torch.no_grad():
        model(prefix, cache=reused)
        model(random_ids(B, 9, seed=2), cache=reused)
        reused.truncate(6)
        model(prefix, cache=fresh)
        assert torch.equal(model(new_tokens, cache=reused), model(new_tokens, cache=fresh))


def test_advance_moves_every_layer() -> None:
    cache = CapturableKVCache.from_config(SMALL, B, max_seq_len=10, dtype=torch.float64)
    for layer_idx in range(SMALL.n_layer):
        update(cache, layer_idx, *random_kv(4))
    cache.advance(0)
    assert cache.seq_len == 4
    cache.advance(6)
    assert cache.seq_len == 10
    assert cache.used_nbytes == cache.nbytes


def test_advance_rejects_invalid_requests() -> None:
    cache = CapturableKVCache.from_config(SMALL, B, max_seq_len=10, dtype=torch.float64)
    for layer_idx in range(SMALL.n_layer):
        update(cache, layer_idx, *random_kv(4))
    with pytest.raises(ValueError, match="n must be >= 0"):
        cache.advance(-1)
    with pytest.raises(ValueError, match="exceeds the cache capacity of 10"):
        cache.advance(7)
    update(cache, 0, *random_kv(1))  # layer 0 now runs ahead of layer 1
    with pytest.raises(ValueError, match="different lengths"):
        cache.advance(1)
    assert cache.seq_len == 5


def test_capturable_update_returns_the_whole_buffer() -> None:
    cache = CapturableKVCache.from_config(SMALL, B, max_seq_len=10, dtype=torch.float64)
    k, v = random_kv(3)
    cached_k, cached_v = cache.update(0, k, v, torch.arange(3))
    assert cached_k.shape == cached_v.shape == (B, SMALL.n_head, 10, SMALL.head_dim)
    assert torch.equal(cached_k[:, :, :3], k)
    assert torch.count_nonzero(cached_k[:, :, 3:]) == 0


@pytest.mark.parametrize(
    "positions",
    [torch.arange(3).view(1, 3), torch.arange(3, dtype=torch.int32), torch.arange(2)],
    ids=["2d", "int32", "wrong-length"],
)
def test_capturable_update_rejects_malformed_positions(positions: Tensor) -> None:
    cache = CapturableKVCache.from_config(SMALL, B, dtype=torch.float64)
    with pytest.raises(ValueError, match="positions"):
        cache.update(0, *random_kv(3), positions)
    assert cache.seq_len == 0


def test_capturable_update_rejects_positions_on_another_device() -> None:
    cache = CapturableKVCache.from_config(SMALL, B, dtype=torch.float64)
    with pytest.raises(ValueError, match="positions are on meta"):
        cache.update(0, *random_kv(3), torch.arange(3, device="meta"))
    assert cache.seq_len == 0


def test_only_the_capturable_cache_accepts_device_positions() -> None:
    assert not DynamicKVCache(SMALL.n_layer).accepts_device_positions
    assert not StaticKVCache.from_config(SMALL, B).accepts_device_positions
    assert CapturableKVCache.from_config(SMALL, B).accepts_device_positions


def test_forward_with_positions_matches_forward_from_seq_len(model: GPT) -> None:
    by_seq_len = CapturableKVCache.from_config(SMALL, B, dtype=torch.float64)
    by_positions = CapturableKVCache.from_config(SMALL, B, dtype=torch.float64)
    prompt, new_tokens = random_ids(B, 5), random_ids(B, 2, seed=1)
    with torch.no_grad():
        model(prompt, cache=by_seq_len)
        model(prompt, cache=by_positions, positions=torch.arange(5))
        expected = model(new_tokens, cache=by_seq_len)
        actual = model(new_tokens, cache=by_positions, positions=torch.arange(5, 7))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("cache_kind", ["none", "dynamic", "static"])
def test_forward_rejects_positions_without_a_capturable_cache(model: GPT, cache_kind: str) -> None:
    caches = {
        "none": None,
        "dynamic": DynamicKVCache(SMALL.n_layer),
        "static": StaticKVCache.from_config(SMALL, B, dtype=torch.float64),
    }
    with pytest.raises(ValueError, match="cache that accepts them"):
        model(random_ids(B, 3), cache=caches[cache_kind], positions=torch.arange(3))


def test_forward_rejects_positions_with_position_offset(model: GPT) -> None:
    cache = CapturableKVCache.from_config(SMALL, B, dtype=torch.float64)
    with pytest.raises(ValueError, match="position_offset must be 0 when positions are given"):
        model(random_ids(B, 3), position_offset=1, cache=cache, positions=torch.arange(3))


@pytest.mark.parametrize(
    ("positions", "match"),
    [
        (torch.arange(3).view(1, 3), "1D long tensor of length 3"),
        (torch.arange(3, dtype=torch.int32), "1D long tensor of length 3"),
        (torch.arange(4), "1D long tensor of length 3"),
        (torch.arange(3, device="meta"), "positions are on meta"),
    ],
    ids=["2d", "int32", "wrong-length", "wrong-device"],
)
def test_forward_rejects_malformed_positions(model: GPT, positions: Tensor, match: str) -> None:
    cache = CapturableKVCache.from_config(SMALL, B, dtype=torch.float64)
    with pytest.raises(ValueError, match=match):
        model(random_ids(B, 3), cache=cache, positions=positions)
    assert cache.seq_len == 0
