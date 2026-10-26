import math

import torch
from torch import Tensor, nn

from kvcache.cache import KVCache
from kvcache.config import GPT2Config


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project (B, T, C) into q, k, v, each shaped (B, n_head, T, head_dim)."""
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return q, k, v

    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Causal attention over (B, n_head, T, head_dim) inputs, returned as (B, Tq, C).

        Keys may be longer than queries: the queries are taken to be the last Tq
        positions, so query i sees keys j <= i + (Tk - Tq).
        """
        B, _, Tq, _ = q.shape
        Tk = k.shape[-2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        return out.transpose(1, 2).contiguous().view(B, Tq, self.n_head * self.head_dim)

    def forward(
        self, x: Tensor, cache: KVCache | None = None, layer_idx: int | None = None
    ) -> Tensor:
        q, k, v = self.project_qkv(x)
        if cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx is required when a cache is given")
            # Update before attending: the new tokens must see themselves.
            k, v = cache.update(layer_idx, k, v)
        return self.c_proj(self.attend(q, k, v))


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self, x: Tensor, cache: KVCache | None = None, layer_idx: int | None = None
    ) -> Tensor:
        x = x + self.attn(self.ln_1(x), cache, layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2 decoder-only language model, inference only.

    Module names mirror the GPT-2 checkpoint, so loading pretrained weights is an
    identity mapping apart from transposing the Conv1D weights.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(
        self, input_ids: Tensor, position_offset: int = 0, cache: KVCache | None = None
    ) -> Tensor:
        """Return logits of shape (B, T, vocab_size) for input_ids of shape (B, T).

        Without a cache, input_ids is the whole sequence and position_offset is the
        absolute position of its first token (0 for a full forward pass).

        With a cache, input_ids holds only the new tokens: the whole prompt for a
        prefill, one token per sequence for a decode step, or any chunk in between.
        Their keys and values are appended to the cache. The position of the first new
        token is cache.seq_len, the cache being the only source of truth for it, so
        position_offset must be left at 0.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D (B, T), got shape {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            raise ValueError(f"input_ids must have dtype torch.long, got {input_ids.dtype}")
        if position_offset < 0:
            raise ValueError(f"position_offset must be >= 0, got {position_offset}")
        if cache is not None:
            if position_offset != 0:
                raise ValueError(
                    "position_offset must be 0 when a cache is given: the position "
                    f"comes from cache.seq_len, got position_offset={position_offset}"
                )
            # Read once, before any block updates the cache.
            position_offset = cache.seq_len
        T = input_ids.shape[1]
        if position_offset + T > self.config.n_positions:
            raise ValueError(
                f"position_offset + sequence length = {position_offset} + {T} = "
                f"{position_offset + T} exceeds n_positions ({self.config.n_positions})"
            )

        positions = torch.arange(position_offset, position_offset + T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(positions)
        for layer_idx, block in enumerate(self.h):
            x = block(x, cache, layer_idx)
        return self.lm_head(self.ln_f(x))
