"""Prove the architecture: exact param counts for every preset, a real GPU training
run of the 125M model at full 1024 context, a gradient-checkpointing memory comparison,
and a generation smoke test.
"""
import os
import sys
import gc
import time
from dataclasses import replace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model.model import GPT, ModelConfig, PRESETS  # noqa: E402

print("=== Parameter counts (vocab=32000) ===")
for name, cfg in PRESETS.items():
    m = GPT(cfg)
    tot, non = m.num_params(), m.num_params(non_embedding=True)
    print(f"  {name:11s} dim={cfg.dim:5d} L={cfg.n_layers:2d} H={cfg.n_heads:2d} "
          f"kv={cfg.n_kv_heads or cfg.n_heads:2d} ctx={cfg.max_seq_len:5d}  ->  "
          f"{tot/1e6:7.1f}M total ({non/1e6:6.1f}M non-embedding)")
    del m
    gc.collect()

dev = torch.device("cuda")


def train_peak(grad_ckpt, steps=30, B=8, T=1024, report=False):
    cfg = replace(PRESETS["small-125m"], grad_checkpoint=grad_ckpt)
    model = GPT(cfg).to(dev)
    opt = model.configure_optimizers(weight_decay=0.1, lr=3e-4)
    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.time()
    for step in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if report and (step % 6 == 0 or step == steps - 1):
            print(f"  step {step:3d}  loss {loss.item():.4f}")
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    tps = steps * B * T / dt
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return peak, tps, loss.item()


print("\n=== GPU training: small-125m, batch 8 x 1024 tokens, bf16 (overfit one batch) ===")
peak, tps, final = train_peak(grad_ckpt=False, report=True)
print(f"peak VRAM: {peak:.2f} GB   throughput: {tps:,.0f} tok/s")
assert final < 5.0, "model failed to memorize a fixed batch"

print("\n=== Gradient checkpointing memory tradeoff (same 125m, 1 comparison) ===")
p_off, t_off, _ = train_peak(grad_ckpt=False, steps=8)
p_on,  t_on,  _ = train_peak(grad_ckpt=True,  steps=8)
print(f"  checkpoint OFF: {p_off:.2f} GB   {t_off:,.0f} tok/s")
print(f"  checkpoint ON : {p_on:.2f} GB   {t_on:,.0f} tok/s   "
      f"(saves {(1-p_on/p_off)*100:.0f}% VRAM, ~{(1-t_on/t_off)*100:.0f}% slower)")

print("\n=== generate() smoke test ===")
cfg = PRESETS["small-125m"]
model = GPT(cfg).to(dev).eval()
start = torch.zeros(1, 1, dtype=torch.long, device=dev)
out = model.generate(start, max_new_tokens=20, temperature=0.8, top_k=50)
print("  generated token ids shape:", tuple(out.shape), "->", out[0].tolist())

print("\nVERDICT: architecture builds, scales, trains on the 4060, and generates.")
