"""Honest throughput/VRAM benchmark for the 125M prototype on the 4060.

Windows spills to shared system RAM instead of OOM-ing, which silently destroys
throughput. So we sweep configs, flag any that exceed the safe VRAM budget, and
report SUSTAINED tokens/sec (warmup excluded) only for configs that fit natively.
"""
import os
import sys
import gc
import time
import statistics
from dataclasses import replace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model.model import GPT, PRESETS  # noqa: E402

dev = torch.device("cuda")
free0, total0 = torch.cuda.mem_get_info()
print(f"VRAM total {total0/1e9:.2f} GB | free at start {free0/1e9:.2f} GB", flush=True)
SAFE_GB = free0 / 1e9 - 0.4   # leave headroom below what's actually free
print(f"Safe torch-allocation budget: <= {SAFE_GB:.2f} GB\n", flush=True)

SEQ = 1024
WARMUP, TIMED = 4, 12


def bench(batch, grad_ckpt):
    cfg = replace(PRESETS["small-125m"], grad_checkpoint=grad_ckpt, dropout=0.0)
    model = GPT(cfg).to(dev)
    opt = model.configure_optimizers(weight_decay=0.1, lr=3e-4)
    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (batch, SEQ), device=dev)
    y = torch.randint(0, cfg.vocab_size, (batch, SEQ), device=dev)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    dts = []
    for step in range(WARMUP + TIMED):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        if step >= WARMUP:
            dts.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    tps = batch * SEQ / statistics.median(dts)
    del model, opt, x, y
    gc.collect()
    torch.cuda.empty_cache()
    return peak, tps


print(f"{'ckpt':>5} {'batch':>6} {'peakGB':>8} {'tok/s':>10}  fit", flush=True)
results = []
for ckpt in (False, True):
    for batch in (2, 4, 6, 8, 12, 16):
        try:
            peak, tps = bench(batch, ckpt)
        except RuntimeError as e:
            print(f"{str(ckpt):>5} {batch:>6}  OOM/{type(e).__name__}", flush=True)
            break
        fits = peak <= SAFE_GB
        tag = "OK" if fits else "SPILL (invalid tok/s)"
        print(f"{str(ckpt):>5} {batch:>6} {peak:>8.2f} {tps:>10,.0f}  {tag}", flush=True)
        if fits:
            results.append((ckpt, batch, peak, tps))

# pick the fastest config that fits natively
best = max(results, key=lambda r: r[3])
ckpt, batch, peak, tps = best
overnight = tps * 3600 * 10
print(f"\nBest native-fit config: checkpoint={ckpt}, micro-batch={batch}, "
      f"{peak:.2f} GB, {tps:,.0f} tok/s", flush=True)
print(f"=> ~{overnight/1e9:.2f}B tokens in a 10-hour overnight run", flush=True)
print(f"=> Chinchilla-optimal for 125M (~2.2B tokens) reachable in "
      f"~{2.2e9/tps/3600:.1f} GPU-hours", flush=True)
