"""End-to-end GPU sanity check: prove the 4060 can actually train, not just import CUDA.

Builds a small causal transformer block and overfits one synthetic batch in bf16.
If the loss drops sharply, the full training stack (SDPA/flash, autocast, backward,
optimizer, GPU memory) is working.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Environment ===")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available — GPU training would not work."
dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("compute capability:", f"{cap[0]}.{cap[1]}")
print("bf16 supported:", torch.cuda.is_bf16_supported())
print("total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

# --- a minimal causal self-attention + MLP block (the real workload) ---
class Block(nn.Module):
    def __init__(self, d=512, n_head=8):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # scaled_dot_product_attention uses flash / mem-efficient kernels on Ada
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab=256, d=512, n_layer=4, block=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

torch.manual_seed(0)
model = TinyGPT().to(dev)
n_params = sum(p.numel() for p in model.parameters())
print("test model params:", f"{n_params/1e6:.1f}M")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

# one fixed synthetic batch — a real model MUST be able to memorize it
B, T = 16, 128
x = torch.randint(0, 256, (B, T), device=dev)
y = torch.randint(0, 256, (B, T), device=dev)

print("\n=== Training (overfit one batch, bf16 autocast) ===")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
losses = []
for step in range(60):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 10 == 0 or step == 59:
        losses.append((step, loss.item()))
        print(f"  step {step:3d}  loss {loss.item():.4f}")
torch.cuda.synchronize()
dt = time.time() - t0

first, last = losses[0][1], losses[-1][1]
peak_gb = torch.cuda.max_memory_allocated() / 1e9
toks = 60 * B * T
print("\n=== Results ===")
print(f"loss: {first:.3f} -> {last:.3f}  (drop {first-last:.3f})")
print(f"peak VRAM used: {peak_gb:.2f} GB")
print(f"throughput: {toks/dt:,.0f} tokens/sec (tiny model, warmup included)")

# SDPA backend check
from torch.nn.attention import SDPBackend, sdpa_kernel
try:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        qq = torch.randn(2, 8, 128, 64, device=dev, dtype=torch.bfloat16)
        F.scaled_dot_product_attention(qq, qq, qq, is_causal=True)
    print("flash-attention kernel: available")
except Exception as e:
    print("flash-attention kernel: NOT available ->", type(e).__name__)

ok = last < first - 2.0 and torch.cuda.is_available()
print("\nVERDICT:", "PASS — GPU trains correctly" if ok else "FAIL — investigate")
