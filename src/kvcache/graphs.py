import torch
from torch import Tensor

from kvcache.cache import CapturableKVCache
from kvcache.generation import _validate_generation_args
from kvcache.model import GPT


class CUDAGraphDecoder:
    """Greedy decode step captured once in a CUDA graph, then replayed one token at a time.

    The graph feeds itself. Each replay reads the current token from the tokens buffer
    and its position from the positions buffer, writes keys and values into the cache at
    that position, stores the argmax at out_tokens[:, step_idx], and advances tokens,
    positions and step_idx on the device. Replays need no host work between them.

    All buffers are static: their addresses are baked into the graph, so inputs are set
    with copy_ or fill_, never by rebinding.
    """

    def __init__(self, model: GPT, cache: CapturableKVCache, max_new_tokens: int) -> None:
        device = cache.device
        model_device = model.lm_head.weight.device
        if device.type != "cuda" or model_device != device:
            raise ValueError(
                f"model and cache must be on the same CUDA device, got the model on "
                f"{model_device} and the cache on {device}"
            )
        self.model = model
        self.cache = cache
        batch_size = cache.batch_size
        # Graph buffers must be normal tensors, writable in place outside inference mode.
        with torch.inference_mode(False):
            self.tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            self.positions = torch.zeros(1, dtype=torch.long, device=device)
            self.step_idx = torch.zeros(1, dtype=torch.long, device=device)
            self.out_tokens = torch.zeros(
                batch_size, max_new_tokens, dtype=torch.long, device=device
            )
            self.logits = torch.zeros(
                batch_size,
                model.config.vocab_size,
                dtype=model.lm_head.weight.dtype,
                device=device,
            )
        self.graph: torch.cuda.CUDAGraph | None = None
        self._start: tuple[Tensor, int, int, int] | None = None

    def _step(self) -> None:
        with torch.no_grad():
            logits_all = self.model(self.tokens, cache=self.cache, positions=self.positions)
            self.logits.copy_(logits_all[:, -1, :])
            next_token = self.logits.argmax(dim=-1, keepdim=True)
            self.out_tokens.index_copy_(1, self.step_idx, next_token)
            self.tokens.copy_(next_token)
            self.positions.add_(1)
            self.step_idx.add_(1)

    def prepare(self, first_token: Tensor, position: int, first_step: int) -> None:
        """Set the buffers for the first replay and remember the state for restores.

        first_token (B, 1) is the token the first replay feeds, at absolute position
        position; its output goes to out_tokens[:, first_step].
        """
        self._start = (first_token.clone(), position, first_step, self.cache.seq_len)
        self._restore()

    def _restore(self) -> None:
        if self._start is None:
            raise RuntimeError("call prepare before capture")
        first_token, position, first_step, seq_len = self._start
        self.tokens.copy_(first_token)
        self.positions.fill_(position)
        self.step_idx.fill_(first_step)
        self.cache.truncate(seq_len)

    def capture(self, warmup: int = 3) -> None:
        """Warm the step up on a side stream, then capture one step into a graph.

        The state is restored after every warmup step, so warmup always writes at the
        prepared position and never runs past the cache capacity. Capture records
        kernels without running them, but the Python body of update still advances the
        Python lengths, so the state is restored once more afterwards.
        """
        side = torch.cuda.Stream(device=self.cache.device)
        side.wait_stream(torch.cuda.current_stream(self.cache.device))
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._step()
                self._restore()
        torch.cuda.current_stream(self.cache.device).wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step()
        self.graph = graph
        self._restore()

    def replay(self, n: int) -> None:
        """Replay the captured step n times, with no synchronisation in between."""
        if self.graph is None:
            raise RuntimeError("call capture before replay")
        for _ in range(n):
            self.graph.replay()


def generate_greedy_graph(
    model: GPT, input_ids: Tensor, max_new_tokens: int, cache: CapturableKVCache
) -> Tensor:
    """Greedy decoding with an eager prefill and a decode step captured in a CUDA graph.

    Same contract and results as `generate_greedy_cached`. Every check runs on the host
    before the first forward pass, since nothing can be checked inside the graph. The
    graph is captured on every call; afterwards cache.seq_len is P + max_new_tokens - 1.
    """
    _validate_generation_args(model, input_ids, max_new_tokens)
    if not isinstance(cache, CapturableKVCache):
        raise ValueError(f"cache must be a CapturableKVCache, got {type(cache).__name__}")
    if cache.seq_len != 0:
        raise ValueError(f"cache must be empty, it holds {cache.seq_len} tokens")
    if cache.n_layer != model.config.n_layer:
        raise ValueError(
            f"cache has {cache.n_layer} layers but the model has {model.config.n_layer}"
        )
    batch_size, prompt_length = input_ids.shape
    if cache.batch_size != batch_size:
        raise ValueError(
            f"cache batch size {cache.batch_size} does not match the batch of {batch_size}"
        )
    model_dtype = model.lm_head.weight.dtype
    if cache.dtype != model_dtype:
        raise ValueError(f"cache dtype {cache.dtype} does not match the model dtype {model_dtype}")
    device = cache.device
    if device.type != "cuda":
        raise ValueError(f"graph decoding needs a CUDA cache, got one on {device}")
    if model.lm_head.weight.device != device or input_ids.device != device:
        raise ValueError(
            f"model, input_ids and cache must share one CUDA device, got "
            f"{model.lm_head.weight.device}, {input_ids.device} and {device}"
        )
    if max_new_tokens == 0:
        return input_ids.clone()
    needed = prompt_length + max_new_tokens - 1
    if needed > cache.capacity:
        raise ValueError(
            f"generation needs a cache capacity of {needed} tokens (prompt length "
            f"{prompt_length} + max_new_tokens {max_new_tokens} - 1, the last token "
            f"never enters the cache), but the cache capacity is {cache.capacity}"
        )

    with torch.no_grad():
        first_token = model(input_ids, cache=cache)[:, -1, :].argmax(dim=-1, keepdim=True)
    if max_new_tokens == 1:
        return torch.cat((input_ids, first_token), dim=1)

    decoder = CUDAGraphDecoder(model, cache, max_new_tokens)
    decoder.out_tokens[:, :1].copy_(first_token)
    decoder.prepare(first_token, position=prompt_length, first_step=1)
    decoder.capture()
    decoder.replay(max_new_tokens - 1)
    torch.cuda.synchronize(device)
    cache.advance(max_new_tokens - 1)
    return torch.cat((input_ids, decoder.out_tokens), dim=1).clone()
