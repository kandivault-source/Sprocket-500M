"""Homebrew LLM — model architecture (from scratch).

A modern decoder-only transformer in the Llama family:
  - RoPE rotary position embeddings (no learned position table)
  - RMSNorm (pre-norm)
  - SwiGLU feed-forward
  - Grouped-query attention (n_kv_heads <= n_heads) for cheap inference KV cache
  - Weight-tied token embedding / LM head
  - Optional gradient checkpointing (trades compute for VRAM)

One config scales the same code from a ~50M local prototype to the ~1B cloud run.
Attention uses torch SDPA, which auto-selects the memory-efficient kernel on this box.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None      # None -> = n_heads (standard MHA); fewer -> GQA
    max_seq_len: int = 1024
    ffn_multiple_of: int = 256         # round SwiGLU hidden dim up to a multiple of this
    ffn_dim_multiplier: float | None = None  # override the 8/3 Llama rule if set
    rope_theta: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5
    grad_checkpoint: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even (RoPE)"
        if self.n_kv_heads is not None:
            assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def precompute_rope(head_dim: int, seq_len: int, theta: float = 10000.0):
    """Return (cos, sin) each shaped (seq_len, head_dim/2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)          # (seq_len, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim). Split-half (GPT-NeoX/Llama) rotation."""
    T, D = x.shape[-2], x.shape[-1]
    cos = cos[:T].to(x.dtype)[None, None]     # (1,1,T,D/2)
    sin = sin[:T].to(x.dtype)[None, None]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rep = self.n_heads // self.n_kv_heads
        self.dropout = cfg.dropout
        self.wq = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.rep > 1:                       # expand KV heads for GQA
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 / 3 * cfg.dim)
        if cfg.ffn_dim_multiplier is not None:
            hidden = int(cfg.ffn_dim_multiplier * cfg.dim)
        m = cfg.ffn_multiple_of
        hidden = m * ((hidden + m - 1) // m)
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight             # weight tying

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual output projections
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()   # head is tied to tok_emb, count once
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"seq len {T} > max {self.cfg.max_seq_len}"
        cos, sin = self.rope_cos, self.rope_sin
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            if self.cfg.grad_checkpoint and self.training:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def configure_optimizers(self, weight_decay, lr, betas=(0.9, 0.95), device_type="cuda"):
        """AdamW with weight decay on 2D matrices only (not norms/embeddings/biases)."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# Named size presets. head_dim is even and 64 or 128 throughout.
PRESETS = {
    "proto-50m":  ModelConfig(dim=512,  n_layers=8,  n_heads=8,  max_seq_len=1024),
    "proto-75m":  ModelConfig(dim=640,  n_layers=11, n_heads=10, max_seq_len=1024),
    "small-125m": ModelConfig(dim=768,  n_layers=12, n_heads=12, max_seq_len=1024),
    "mid-350m":   ModelConfig(dim=1024, n_layers=24, n_heads=16, max_seq_len=2048),
    # Flagship #1. 501.1M params (verify with scripts/flagship_plan.py).
    # GQA 20:4 keeps the KV cache small — this model is meant to run on a laptop
    # or phone, where KV cache, not weights, is what bites at long context.
    "500m":       ModelConfig(dim=1280, n_layers=26, n_heads=20, n_kv_heads=4,
                              max_seq_len=2048),
    "large-1b":   ModelConfig(dim=2048, n_layers=22, n_heads=16, n_kv_heads=8, max_seq_len=2048),
}
