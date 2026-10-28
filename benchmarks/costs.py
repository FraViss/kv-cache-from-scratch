"""Analytic FLOP and byte counts for a GPT-2 decoder.

Pure Python: no tensors. A multiply-add counts as two FLOPs. Biases, layer
norms and the residual adds are left out of the FLOP count; they are negligible
next to the matmuls. Activation traffic is left out of the byte count except
where a formula below says otherwise. ``itemsize`` is the bytes per element of
the dtype under test (4 for float32, 2 for bfloat16).
"""

from typing import Protocol


class _GPT2Shape(Protocol):
    """The fields of kvcache.GPT2Config that the formulas read."""

    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int


def num_params(config: _GPT2Shape) -> int:
    """Parameter count with the tied output head counted once.

    Closed form, matching ``sum(p.numel() for p in model.parameters())``:
    token embedding ``vocab_size * n_embd`` (shared with ``lm_head``), position
    embedding ``n_positions * n_embd``, per layer ``12 * n_embd**2`` projection
    weights plus ``13 * n_embd`` of biases and layer-norm scales, and the final
    layer norm ``2 * n_embd``.
    """
    width = config.n_embd
    per_layer = 12 * width * width + 13 * width
    return (
        config.vocab_size * width
        + config.n_positions * width
        + config.n_layer * per_layer
        + 2 * width
    )


def matmul_weights(config: _GPT2Shape) -> int:
    """Parameters that dominate the matmul FLOPs.

    Per layer: ``c_attn`` is ``3 C**2``, ``attn.c_proj`` is ``C**2``, ``c_fc`` is
    ``4 C**2`` and ``mlp.c_proj`` is ``4 C**2``, so ``12 C**2``. The output head
    adds ``vocab_size * C`` and is the same storage as ``wte``. Biases and layer
    norms are excluded.
    """
    return config.n_layer * 12 * config.n_embd**2 + config.vocab_size * config.n_embd


def flops_per_token(config: _GPT2Shape, context_len: int) -> int:
    """FLOPs to decode one new token that attends over ``context_len`` keys.

    ``2 * matmul_weights`` is the linear algebra of every projection and of the
    output head, each weight used once. ``4 * n_layer * n_embd * context_len``
    is QKᵀ plus the attention-weighted values, for one query against
    ``context_len`` keys: each of those two matmuls does ``n_embd * context_len``
    multiply-adds per layer.
    """
    attention = 4 * config.n_layer * config.n_embd * context_len
    return 2 * matmul_weights(config) + attention


def prefill_flops(config: _GPT2Shape, prompt_len: int) -> int:
    """FLOPs of a full forward over a prompt of length T.

    Linear layers scale with T. Attention does not: this implementation builds
    the whole score matrix and masks it afterwards, so the count is ``T * T``,
    not the ``T * T / 2`` of a causal implementation that skips the upper
    triangle. ``prefill_flops(config, 1)`` equals ``flops_per_token(config, 1)``.
    """
    width = config.n_embd
    attention = 4 * config.n_layer * width * prompt_len * prompt_len
    return 2 * matmul_weights(config) * prompt_len + attention


def weight_bytes(config: _GPT2Shape, itemsize: int) -> int:
    """Bytes of parameters a forward reads, excluding the position table.

    ``wpe`` has ``n_positions * n_embd`` parameters, but a forward reads only one
    row per token, so the whole table is not traffic. Everything else is read
    (the tied head once). ``num_params`` counts the tied weights once.
    """
    position_table = config.n_positions * config.n_embd
    return (num_params(config) - position_table) * itemsize


def kv_bytes_per_token(config: _GPT2Shape, itemsize: int, batch_size: int = 1) -> int:
    """Bytes of keys plus values for one token, across every layer.

    Two tensors (K and V), each ``n_layer * n_embd`` elements per sequence,
    times the batch. ``n_embd`` equals ``n_head * head_dim``.
    """
    return 2 * config.n_layer * config.n_embd * itemsize * batch_size


def decode_step_bytes(
    config: _GPT2Shape,
    itemsize: int,
    context_len: int,
    batch_size: int,
    cache: str,
    capacity: int | None = None,
) -> int:
    """Bytes moved by one decode step that attends over ``context_len`` keys.

    ``"static"`` and ``"dynamic"`` read the weights (``weight_bytes``) and the cache
    (``kv_bytes_per_token * context_len``). ``"dynamic"`` also pays for
    ``torch.cat``: the ``context_len - 1`` tokens already stored are read and
    written again, which is ``2 * kv_bytes_per_token * (context_len - 1)``.
    ``"static"`` writes the new token in place and that write is neglected.

    ``"capturable"`` is the cache of a decode step captured in a CUDA graph. Its
    attention reads the whole buffer, written or not, so the cache term is
    ``kv_bytes_per_token * capacity`` whatever ``context_len`` is; the write of
    the new token is neglected. ``capacity`` is required for it and ignored by the
    other caches. Activation traffic is neglected in all three.

    The matching FLOP count for ``"capturable"`` is
    ``batch_size * flops_per_token(config, capacity)``: attention works on every
    position of the buffer, masked or not.
    """
    if cache not in ("static", "dynamic", "capturable"):
        raise ValueError(f"cache must be 'static', 'dynamic' or 'capturable', got {cache!r}")
    if context_len < 1:
        raise ValueError(f"context_len must be >= 1, got {context_len}")
    per_token = kv_bytes_per_token(config, itemsize, batch_size)
    if cache == "capturable":
        if capacity is None or capacity < context_len:
            raise ValueError(
                f"capturable needs a capacity >= context_len ({context_len}), got {capacity}"
            )
        return weight_bytes(config, itemsize) + per_token * capacity
    traffic = weight_bytes(config, itemsize) + per_token * context_len
    if cache == "dynamic":
        traffic += 2 * per_token * (context_len - 1)
    return traffic


def roofline_time_s(flops: float, bytes_: float, peak_flops: float, bandwidth: float) -> float:
    """The roofline lower bound: the slower of the compute roof and the bandwidth roof.

    ``peak_flops`` is FLOP/s and ``bandwidth`` is bytes/s. The bound ignores
    latency, launch overhead and any traffic the byte count left out, so a
    measured step can only be slower.
    """
    return max(flops / peak_flops, bytes_ / bandwidth)
