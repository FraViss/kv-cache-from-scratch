from abc import ABC, abstractmethod

import torch
from torch import Tensor


class KVCache(ABC):
    """Per-layer store of attention keys and values, shared by every layer of one model.

    A forward pass with a cache feeds only the new tokens. Each attention layer hands its
    new keys and values to `update` and attends over the full keys and values it gets
    back. Implementations differ in how they hold memory (growing by concatenation, or a
    preallocated buffer with a write index), not in this interface.
    """

    @abstractmethod
    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append k and v, shaped (B, n_head, T_new, head_dim), to layer layer_idx.

        Returns the layer's full keys and values, shaped (B, n_head, T_tot, head_dim).
        """

    @property
    @abstractmethod
    def seq_len(self) -> int:
        """Number of tokens in the cache, 0 when empty."""

    @property
    @abstractmethod
    def nbytes(self) -> int:
        """Bytes actually held by every cached key and value tensor."""

    @abstractmethod
    def reset(self) -> None:
        """Empty the cache."""


class DynamicKVCache(KVCache):
    """KV cache that grows by concatenating new keys and values along the token dimension.

    Every update copies the whole past of the layer into a new tensor, so the memory
    traffic of a generation grows quadratically with its length.
    """

    def __init__(self, n_layer: int) -> None:
        if n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {n_layer}")
        self.n_layer = n_layer
        self._keys: list[Tensor | None] = [None] * n_layer
        self._values: list[Tensor | None] = [None] * n_layer

    @property
    def seq_len(self) -> int:
        """Number of tokens in layer 0.

        A forward pass updates the layers in order, so during a forward layer 0 is
        already ahead of the others. Read seq_len before the forward starts, which is
        what `GPT.forward` does to get the position of the first new token.
        """
        keys = self._keys[0]
        return 0 if keys is None else keys.shape[-2]

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size() for t in self._keys + self._values if t is not None
        )

    def reset(self) -> None:
        self._keys = [None] * self.n_layer
        self._values = [None] * self.n_layer

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(layer_idx, k, v)
        past_k, past_v = self._keys[layer_idx], self._values[layer_idx]
        if past_k is None or past_v is None:
            # k and v are usually views into the output of c_attn, which also holds q.
            # Keeping a view would keep q alive too, so the cache takes its own copy.
            new_k = k.clone(memory_format=torch.contiguous_format)
            new_v = v.clone(memory_format=torch.contiguous_format)
        else:
            new_k = torch.cat((past_k, k), dim=-2)
            new_v = torch.cat((past_v, v), dim=-2)
        self._keys[layer_idx], self._values[layer_idx] = new_k, new_v
        return new_k, new_v

    def _validate(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
        """Raise ValueError before any state changes, so a rejected update leaves the cache as it was."""
        if not 0 <= layer_idx < self.n_layer:
            raise ValueError(f"layer_idx must be in [0, {self.n_layer}), got {layer_idx}")
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                "k and v must be 4D (B, n_head, T_new, head_dim), "
                f"got shapes {tuple(k.shape)} and {tuple(v.shape)}"
            )
        if k.shape != v.shape:
            raise ValueError(
                f"k and v must have the same shape, got {tuple(k.shape)} and {tuple(v.shape)}"
            )
        if k.dtype != v.dtype or k.device != v.device:
            raise ValueError(
                "k and v must share dtype and device, "
                f"got {k.dtype} on {k.device} and {v.dtype} on {v.device}"
            )

        # Layer 0 is written first in every forward, so it is the reference for the
        # whole cache: every layer must match its B, n_head, head_dim, dtype and device.
        reference = self._keys[0]
        if reference is not None:
            for dim, name in ((0, "B"), (1, "n_head"), (3, "head_dim")):
                if k.shape[dim] != reference.shape[dim]:
                    raise ValueError(
                        f"{name} of the new keys ({k.shape[dim]}) does not match the cache "
                        f"({reference.shape[dim]}); got shape {tuple(k.shape)}, cached "
                        f"shape {tuple(reference.shape)}"
                    )
            # torch.cat would silently promote a mixed dtype instead of failing.
            if k.dtype != reference.dtype:
                raise ValueError(f"dtype {k.dtype} does not match the cache dtype {reference.dtype}")
            if k.device != reference.device:
                raise ValueError(
                    f"device {k.device} does not match the cache device {reference.device}"
                )

        if layer_idx > 0:
            past = self._keys[layer_idx]
            new_len = (0 if past is None else past.shape[-2]) + k.shape[-2]
            layer0_len = 0 if reference is None else reference.shape[-2]
            if new_len != layer0_len:
                raise ValueError(
                    f"layer {layer_idx} would hold {new_len} tokens but layer 0 holds "
                    f"{layer0_len}: a layer was updated twice or skipped in a forward pass"
                )
