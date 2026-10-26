import math

import torch
from torch import Tensor, nn

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

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(self.attend(*self.project_qkv(x)))


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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
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

    def forward(self, input_ids: Tensor, position_offset: int = 0) -> Tensor:
        """Return logits of shape (B, T, vocab_size) for input_ids of shape (B, T).

        position_offset is the absolute position of the first token in input_ids.
        It is 0 for a full forward pass; cached decoding (M2) feeds only the new
        tokens and sets it to the number of tokens already in the cache.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D (B, T), got shape {tuple(input_ids.shape)}")
        if input_ids.dtype != torch.long:
            raise ValueError(f"input_ids must have dtype torch.long, got {input_ids.dtype}")
        if position_offset < 0:
            raise ValueError(f"position_offset must be >= 0, got {position_offset}")
        T = input_ids.shape[1]
        if position_offset + T > self.config.n_positions:
            raise ValueError(
                f"position_offset + sequence length = {position_offset} + {T} = "
                f"{position_offset + T} exceeds n_positions ({self.config.n_positions})"
            )

        positions = torch.arange(position_offset, position_offset + T, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(positions)
        for block in self.h:
            x = block(x)
        return self.lm_head(self.ln_f(x))
