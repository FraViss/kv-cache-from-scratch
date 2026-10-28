from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor

from kvcache.config import GPT2Config


class KVCache(ABC):
    """Per-layer store of attention keys and values, shared by every layer of one model.

    A forward pass with a cache feeds only the new tokens. Each attention layer hands its
    new keys and values to `update` and attends over the full keys and values it gets
    back. Implementations differ in how they hold memory (growing by concatenation, or a
    preallocated buffer with a write index), not in this interface.
    """

    @abstractmethod
    def update(
        self, layer_idx: int, k: Tensor, v: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Append k and v, shaped (B, n_head, T_new, head_dim), to layer layer_idx.

        positions is a long tensor of shape (T_new,), on the device of k, with the
        absolute position of each new token: the same tensor the attention mask uses.
        A cache that tracks its own write index may ignore it.

        Returns the layer's full keys and values, shaped (B, n_head, T_tot, head_dim).
        """

    @property
    def accepts_device_positions(self) -> bool:
        """Whether the cache writes at the positions it is given, so a forward pass can
        take its positions from a tensor on the device instead of from seq_len."""
        return False

    @property
    @abstractmethod
    def n_layer(self) -> int:
        """Number of layers the cache holds keys and values for."""

    @property
    @abstractmethod
    def seq_len(self) -> int:
        """Number of tokens in the cache, 0 when empty."""

    @property
    @abstractmethod
    def capacity(self) -> int | None:
        """Maximum number of tokens per sequence, or None when the cache can grow without limit."""

    @property
    @abstractmethod
    def nbytes(self) -> int:
        """Bytes of memory the cache has actually allocated for keys and values.

        For a static cache this is the full capacity, however much of it is in use.
        """

    @property
    @abstractmethod
    def used_nbytes(self) -> int:
        """Bytes taken by the keys and values of the tokens currently in the cache."""

    @abstractmethod
    def reset(self) -> None:
        """Empty the cache."""


def _validate_update(
    n_layer: int,
    lengths: Sequence[int],
    reference: Tensor | None,
    layer_idx: int,
    k: Tensor,
    v: Tensor,
) -> None:
    """Checks shared by every cache, run before an update changes any state.

    lengths holds the current number of tokens of each layer. reference is a tensor laid
    out like the cache, (B, n_head, any, head_dim), with the cache's dtype and device, or
    None while nothing fixes them yet.
    """
    if not 0 <= layer_idx < n_layer:
        raise ValueError(f"layer_idx must be in [0, {n_layer}), got {layer_idx}")
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

    if reference is not None:
        for dim, name in ((0, "B"), (1, "n_head"), (3, "head_dim")):
            if k.shape[dim] != reference.shape[dim]:
                raise ValueError(
                    f"{name} of the new keys ({k.shape[dim]}) does not match the cache "
                    f"({reference.shape[dim]}); got shape {tuple(k.shape)}, cached "
                    f"shape {tuple(reference.shape)}"
                )
        # Neither torch.cat nor copy_ fails on a mixed dtype: one promotes, the other
        # converts, silently.
        if k.dtype != reference.dtype:
            raise ValueError(f"dtype {k.dtype} does not match the cache dtype {reference.dtype}")
        if k.device != reference.device:
            raise ValueError(f"device {k.device} does not match the cache device {reference.device}")

    # Layer 0 is written first in every forward, so after an update every other layer
    # must hold exactly as many tokens as layer 0.
    if layer_idx > 0:
        new_len = lengths[layer_idx] + k.shape[-2]
        if new_len != lengths[0]:
            raise ValueError(
                f"layer {layer_idx} would hold {new_len} tokens but layer 0 holds "
                f"{lengths[0]}: a layer was updated twice or skipped in a forward pass"
            )


class DynamicKVCache(KVCache):
    """KV cache that grows by concatenating new keys and values along the token dimension.

    Every update copies the whole past of the layer into a new tensor, so the memory
    traffic of a generation grows quadratically with its length.
    """

    def __init__(self, n_layer: int) -> None:
        if n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {n_layer}")
        self._n_layer = n_layer
        self._keys: list[Tensor | None] = [None] * n_layer
        self._values: list[Tensor | None] = [None] * n_layer

    @property
    def n_layer(self) -> int:
        return self._n_layer

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
    def capacity(self) -> None:
        return None

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size() for t in self._keys + self._values if t is not None
        )

    @property
    def used_nbytes(self) -> int:
        """Same as nbytes: the cache holds exactly the tokens in use."""
        return self.nbytes

    def reset(self) -> None:
        self._keys = [None] * self._n_layer
        self._values = [None] * self._n_layer

    def update(
        self, layer_idx: int, k: Tensor, v: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Append along the token dimension. positions is ignored: concatenation always
        appends after the tokens already cached."""
        lengths = [0 if t is None else t.shape[-2] for t in self._keys]
        # Layer 0 fixes B, n_head, head_dim, dtype and device for the whole cache.
        _validate_update(self._n_layer, lengths, self._keys[0], layer_idx, k, v)
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


class StaticKVCache(KVCache):
    """KV cache backed by buffers preallocated once, written in place at each layer's length.

    The constructor allocates keys and values of shape
    (n_layer, batch_size, n_head, max_seq_len, head_dim), and nothing allocates again
    afterwards. Build the cache outside torch.inference_mode(): buffers created inside it
    would be inference tensors, which can no longer be written in place outside it.
    """

    def __init__(
        self,
        n_layer: int,
        batch_size: int,
        n_head: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        sizes = {
            "n_layer": n_layer,
            "batch_size": batch_size,
            "n_head": n_head,
            "head_dim": head_dim,
            "max_seq_len": max_seq_len,
        }
        for name, size in sizes.items():
            if size < 1:
                raise ValueError(f"{name} must be >= 1, got {size}")
        shape = (n_layer, batch_size, n_head, max_seq_len, head_dim)
        self._keys = torch.zeros(shape, dtype=dtype, device=device)
        self._values = torch.zeros(shape, dtype=dtype, device=device)
        self._lengths = [0] * n_layer

    @classmethod
    def from_config(
        cls,
        config: GPT2Config,
        batch_size: int,
        max_seq_len: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "StaticKVCache":
        """Size the cache for a model; max_seq_len defaults to config.n_positions."""
        return cls(
            n_layer=config.n_layer,
            batch_size=batch_size,
            n_head=config.n_head,
            head_dim=config.head_dim,
            max_seq_len=config.n_positions if max_seq_len is None else max_seq_len,
            dtype=dtype,
            device=device,
        )

    @property
    def n_layer(self) -> int:
        return self._keys.shape[0]

    @property
    def seq_len(self) -> int:
        """Number of tokens in layer 0.

        A forward pass updates the layers in order, so during a forward layer 0 is
        already ahead of the others. Read seq_len before the forward starts, which is
        what `GPT.forward` does to get the position of the first new token.
        """
        return self._lengths[0]

    @property
    def capacity(self) -> int:
        return self._keys.shape[3]

    @property
    def nbytes(self) -> int:
        return 2 * self._keys.numel() * self._keys.element_size()

    @property
    def used_nbytes(self) -> int:
        _, batch_size, n_head, _, head_dim = self._keys.shape
        per_token = batch_size * n_head * head_dim * self._keys.element_size()
        return 2 * sum(self._lengths) * per_token

    def reset(self) -> None:
        """Forget every token. The buffers keep their old contents, which is safe because
        update only ever returns the filled part of a layer."""
        self._lengths = [0] * self.n_layer

    def truncate(self, seq_len: int) -> None:
        """Keep only the first seq_len tokens of every layer.

        Like reset, this leaves the buffers untouched: only the written part of a layer
        is ever visible, so the dropped tokens are simply overwritten later.
        """
        if len(set(self._lengths)) != 1:
            raise ValueError(f"cannot truncate: the layers have different lengths {self._lengths}")
        if not 0 <= seq_len <= self._lengths[0]:
            raise ValueError(
                f"seq_len must be in [0, {self._lengths[0]}] to truncate, got {seq_len}"
            )
        self._lengths = [seq_len] * self.n_layer

    def update(
        self, layer_idx: int, k: Tensor, v: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Write at the layer's current length. positions is ignored: the write index is
        the Python length of the layer."""
        # The buffer fixes B, n_head, head_dim, dtype and device from the start.
        _validate_update(self.n_layer, self._lengths, self._keys[0], layer_idx, k, v)
        start = self._lengths[layer_idx]
        end = start + k.shape[-2]
        if end > self.capacity:
            raise ValueError(
                f"layer {layer_idx} would hold {end} tokens ({start} + {k.shape[-2]}), "
                f"over the cache capacity of {self.capacity}"
            )
        self._keys[layer_idx, :, :, start:end].copy_(k)
        self._values[layer_idx, :, :, start:end].copy_(v)
        self._lengths[layer_idx] = end
        # Only the filled part is returned, so attention never sees stale or zero slots.
        return self._keys[layer_idx, :, :, :end], self._values[layer_idx, :, :, :end]
