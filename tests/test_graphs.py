import pytest
import torch
from torch import Tensor

from kvcache import (
    GPT,
    CapturableKVCache,
    CUDAGraphDecoder,
    GPT2Config,
    StaticKVCache,
    generate_greedy_graph,
    generate_greedy_uncached,
)

SMALL = GPT2Config(vocab_size=100, n_positions=32, n_embd=64, n_layer=2, n_head=4)
B, P = 2, 5
CAPACITY = 12
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def build_model(device: str) -> GPT:
    torch.manual_seed(0)
    return GPT(SMALL).to(dtype=torch.float64, device=device).eval()


@pytest.fixture
def cuda_model() -> GPT:
    return build_model("cuda")


def random_ids(batch: int, length: int, device: str = "cuda", seed: int = 0) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, SMALL.vocab_size, (batch, length), generator=generator)
    return ids.to(device)


def capturable_cache(
    capacity: int = CAPACITY, device: str = "cuda", batch_size: int = B
) -> CapturableKVCache:
    return CapturableKVCache.from_config(
        SMALL, batch_size, max_seq_len=capacity, dtype=torch.float64, device=device
    )


@requires_cuda
@pytest.mark.parametrize(
    "new_tokens", [1, 2, 5, CAPACITY - P + 1], ids=["n1", "n2", "n5", "to-capacity"]
)
def test_graph_generation_matches_uncached(cuda_model: GPT, new_tokens: int) -> None:
    prompt = random_ids(B, P)
    cache = capturable_cache()
    output = generate_greedy_graph(cuda_model, prompt, new_tokens, cache)
    assert torch.equal(output, generate_greedy_uncached(cuda_model, prompt, new_tokens))
    assert cache.seq_len == P + new_tokens - 1
    assert not output.is_inference()


@requires_cuda
def test_graph_logits_match_uncached_forward_step_by_step(cuda_model: GPT) -> None:
    new_tokens = 6
    prompt = random_ids(B, P)
    cache = capturable_cache()
    with torch.no_grad():
        first_token = cuda_model(prompt, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
    decoder = CUDAGraphDecoder(cuda_model, cache, new_tokens)
    decoder.out_tokens[:, :1].copy_(first_token)
    decoder.prepare(first_token, position=P, first_step=1)
    decoder.capture()

    for step in range(1, new_tokens):
        decoder.replay(1)
        torch.cuda.synchronize()
        # Replay number `step` fed out_tokens[:, step - 1], the last token of this prefix.
        prefix = torch.cat((prompt, decoder.out_tokens[:, :step]), dim=1)
        with torch.no_grad():
            expected = cuda_model(prefix)[:, -1, :]
        torch.testing.assert_close(decoder.logits, expected, atol=1e-10, rtol=0)


@requires_cuda
def test_eager_forward_continues_after_graph_generation(cuda_model: GPT) -> None:
    new_tokens = 5
    prompt = random_ids(B, P)
    cache = capturable_cache()
    output = generate_greedy_graph(cuda_model, prompt, new_tokens, cache)
    # The last generated token never entered the cache; feeding it continues the sequence.
    with torch.no_grad():
        continued = cuda_model(output[:, -1:], cache=cache)
        expected = cuda_model(output)[:, -1:, :]
    torch.testing.assert_close(continued, expected, atol=1e-10, rtol=0)
    assert cache.seq_len == P + new_tokens


@requires_cuda
def test_graph_generation_rejects_too_small_a_cache_before_any_forward(cuda_model: GPT) -> None:
    calls = []
    cuda_model.register_forward_hook(lambda *_: calls.append(1))
    new_tokens = 4
    cache = capturable_cache(capacity=P + new_tokens - 2)
    with pytest.raises(ValueError, match="capacity of 8 tokens.*cache capacity is 7"):
        generate_greedy_graph(cuda_model, random_ids(B, P), new_tokens, cache)
    assert calls == []
    assert cache.seq_len == 0


def test_graph_generation_rejects_a_cpu_cache() -> None:
    model = build_model("cpu")
    calls = []
    model.register_forward_hook(lambda *_: calls.append(1))
    prompt = random_ids(B, P, device="cpu")
    with pytest.raises(ValueError, match="needs a CUDA cache"):
        generate_greedy_graph(model, prompt, 4, capturable_cache(device="cpu"))
    assert calls == []


def test_graph_generation_rejects_a_non_capturable_cache() -> None:
    cache = StaticKVCache.from_config(SMALL, B, dtype=torch.float64)
    with pytest.raises(ValueError, match="must be a CapturableKVCache"):
        generate_greedy_graph(build_model("cpu"), random_ids(B, P, device="cpu"), 4, cache)


def test_graph_generation_rejects_a_batch_size_mismatch() -> None:
    cache = capturable_cache(device="cpu", batch_size=B + 1)
    with pytest.raises(ValueError, match="cache batch size 3 does not match the batch of 2"):
        generate_greedy_graph(build_model("cpu"), random_ids(B, P, device="cpu"), 4, cache)


def test_graph_decoder_rejects_a_cpu_cache() -> None:
    with pytest.raises(ValueError, match="same CUDA device"):
        CUDAGraphDecoder(build_model("cpu"), capturable_cache(device="cpu"), 4)


@requires_cuda
def test_graph_generation_at_the_edge_of_capacity(cuda_model: GPT) -> None:
    """Two new tokens with a capacity of exactly P + 1.

    Warmup runs the step three times. If the state were restored only once at the end,
    the Python lengths would reach P + 2 during the second warmup step, past the
    capacity, and this test would fail with a clean ValueError from the capacity check
    in update.
    """
    new_tokens = 2
    prompt = random_ids(B, P)
    cache = capturable_cache(capacity=P + 1)
    output = generate_greedy_graph(cuda_model, prompt, new_tokens, cache)
    assert torch.equal(output, generate_greedy_uncached(cuda_model, prompt, new_tokens))
    assert cache.seq_len == cache.capacity
