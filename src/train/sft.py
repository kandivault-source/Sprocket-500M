"""SFT / instruct stage â€” fine-tune a pretrained base checkpoint on the chat corpus.

Differs from pretraining in four ways that matter:
  1. Starts from a base checkpoint (--base), with a FRESH optimizer. Carrying the
     pretrain Adam state into SFT drags the model back toward the web distribution.
  2. Much lower LR (2e-5 vs 6e-4). SFT is style/format adaptation, not learning
     language; a pretrain LR here will wreck the base model's knowledge.
  3. Loss on assistant spans only â€” see src/train/sft_data.py. That module's
     --self-test is the guard; run it if you touch the template.
  4. Epochs over a small corpus (2-4), not a single pass over a huge one. For
     style, repeated exposure matters more than unique tokens.

  py -m src.train.sft --base checkpoints/500m_final.pt --preset 500m --epochs 3
  py -m src.train.sft --base ... --resume auto          # same semantics as train.py
"""
import argparse
import glob
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from model.model import GPT, PRESETS  # noqa: E402
from train.sft_data import ChatTemplate, load_conversations, pack  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=None, help="pretrained checkpoint to start from")
    p.add_argument("--preset", default="proto-75m", choices=sorted(PRESETS))
    p.add_argument("--data", nargs="*", default=["data/synthetic/sprocket_sft.jsonl",
                                                 "data/synthetic/sprocket_instruct.jsonl"])
    p.add_argument("--cache", default="data/processed/sft_packed.npz")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--tokenizer", default="config/tokenizer/tokenizer.json")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--metrics", default="dashboard/data/metrics.json")
    p.add_argument("--tag", default="sft")

    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--peak-lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.02)

    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=12)
    p.add_argument("--sample-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", default=None)
    p.add_argument("--ckpt-minutes", type=float, default=15.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile â€” measured 1.7x on H100. Linux only; "
                        "inductor needs a C compiler, so it is a no-op risk on Windows.")
    return p.parse_args()


A = parse_args()
os.makedirs(A.ckpt_dir, exist_ok=True)
if os.path.dirname(A.metrics):
    os.makedirs(os.path.dirname(A.metrics), exist_ok=True)
if os.path.dirname(A.cache):
    os.makedirs(os.path.dirname(A.cache), exist_ok=True)

device = A.device
torch.manual_seed(A.seed)
np.random.seed(A.seed)
random.seed(A.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

RUN = f"{A.preset}_{A.tag}"
LATEST = os.path.join(A.ckpt_dir, f"{RUN}_latest.pt")

# ------------------------------------------------------------------ data
tmpl = ChatTemplate(A.tokenizer)
if os.path.exists(A.cache) and not A.rebuild_cache:
    z = np.load(A.cache)
    X, Y = z["x"], z["y"]
    if X.shape[1] != A.ctx:
        print(f"cache ctx={X.shape[1]} != --ctx {A.ctx}; rebuilding", flush=True)
        X = None
    else:
        print(f"loaded packed cache {A.cache}: {X.shape[0]:,} blocks", flush=True)
else:
    X = None

if X is None:
    convos = load_conversations(A.data)
    print(f"rendering {len(convos):,} conversations -> ctx {A.ctx} blocks...", flush=True)
    X, Y, dropped = pack(convos, tmpl, A.ctx, report_every=5000)
    np.savez(A.cache, x=X, y=Y)
    print(f"  packed {X.shape[0]:,} blocks ({dropped} dropped), cached -> {A.cache}", flush=True)

trainable = int((Y != -1).sum())
print(f"corpus: {X.shape[0]:,} blocks x {A.ctx} = {X.size:,} tokens, "
      f"{trainable:,} trainable ({100*trainable/X.size:.1f}%)", flush=True)

rng = np.random.default_rng(A.seed)
perm = rng.permutation(X.shape[0])
n_val = max(1, int(X.shape[0] * A.val_frac))
val_idx, train_idx = perm[:n_val], perm[n_val:]
Xtr = torch.from_numpy(X[train_idx].astype(np.int64))
Ytr = torch.from_numpy(Y[train_idx].astype(np.int64))
Xva = torch.from_numpy(X[val_idx].astype(np.int64)).to(device)
Yva = torch.from_numpy(Y[val_idx].astype(np.int64)).to(device)
print(f"  train {len(train_idx):,} blocks / val {len(val_idx):,} blocks", flush=True)

STEPS_PER_EPOCH = max(1, len(train_idx) // (A.micro_batch * A.grad_accum))
MAX_ITERS = int(STEPS_PER_EPOCH * A.epochs)
WARMUP = max(1, int(MAX_ITERS * A.warmup_frac))
print(f"  {STEPS_PER_EPOCH:,} steps/epoch x {A.epochs} epochs = {MAX_ITERS:,} steps "
      f"(warmup {WARMUP})", flush=True)


def get_batch(split="train"):
    if split == "val":
        i = torch.randint(0, Xva.shape[0], (A.micro_batch,))
        return Xva[i], Yva[i]
    i = torch.randint(0, Xtr.shape[0], (A.micro_batch,))
    return Xtr[i].to(device, non_blocking=True), Ytr[i].to(device, non_blocking=True)


# ------------------------------------------------------------------ model
cfg = replace(PRESETS[A.preset], max_seq_len=A.ctx, dropout=0.0)
model = GPT(cfg).to(device)
raw_model = model          # state_dict/generate must go through the uncompiled module
n_params = model.num_params()
print(f"model {A.preset}: {n_params/1e6:.1f}M params", flush=True)

start_it, history = 0, []
if A.base and not A.resume:
    ck = torch.load(A.base, map_location=device, weights_only=False)
    saved = ck.get("cfg", {})
    for k in ("dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size"):
        if k in saved and saved[k] != getattr(cfg, k):
            sys.exit(f"base checkpoint mismatch: {k}={saved[k]} vs preset {getattr(cfg,k)}")
    raw_model.load_state_dict(ck["model"])
    print(f"loaded BASE weights from {A.base} (iter {ck.get('it')}) â€” fresh optimizer",
          flush=True)
elif not A.base and not A.resume:
    print("! no --base: fine-tuning from RANDOM init. Fine for a smoke test, "
          "useless for a real model.", flush=True)

opt = model.configure_optimizers(A.weight_decay, A.peak_lr, device_type="cuda")


def lr_at(it):
    if it < WARMUP:
        return A.peak_lr * (it + 1) / WARMUP
    r = min(1.0, (it - WARMUP) / max(MAX_ITERS - WARMUP, 1))
    return A.min_lr + 0.5 * (A.peak_lr - A.min_lr) * (1 + math.cos(math.pi * r))


# ------------------------------------------------------------------ resume
def resolve(spec):
    if spec is None:
        return None
    if spec != "auto":
        return spec
    return LATEST if os.path.exists(LATEST) else None


rp = resolve(A.resume)
if rp:
    ck = torch.load(rp, map_location=device, weights_only=False)
    raw_model.load_state_dict(ck["model"])
    if "opt" in ck:
        opt.load_state_dict(ck["opt"])
    start_it = ck.get("it", 0) + 1
    history = ck.get("history", [])
    try:
        r = ck.get("rng", {})
        if "torch" in r:
            torch.set_rng_state(r["torch"].cpu().to(torch.uint8))
        if r.get("cuda"):
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in r["cuda"]])
        if "numpy" in r:
            np.random.set_state(r["numpy"])
        if "python" in r:
            random.setstate(r["python"])
    except Exception as e:
        print(f"  ! RNG restore failed ({e})", flush=True)
    print(f"RESUMED {rp} at step {start_it}/{MAX_ITERS}", flush=True)


if A.compile:
    model = torch.compile(model)
    print("torch.compile enabled (first steps will be slow while it compiles)", flush=True)


@torch.no_grad()
def evaluate():
    model.eval()
    tot, n = 0.0, 0
    for _ in range(A.eval_batches):
        xb, yb = get_batch("val")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l = model(xb, yb)
        tot += l.item()
        n += 1
    model.train()
    return tot / n


@torch.no_grad()
def sample(prompt="What are you?"):
    """Generate through the real chat template â€” same bytes the app will send."""
    model.eval()
    ids = tmpl.render_prompt([{"role": "user", "content": prompt}])
    idx = torch.tensor([ids], device=device)
    out = raw_model.generate(idx, 120, temperature=0.8, top_k=50)
    model.train()
    gen = out[0].tolist()[len(ids):]
    if tmpl.end in gen:
        gen = gen[:gen.index(tmpl.end)]
    return tmpl.tok.decode(gen)


def write_metrics(step, tr, val, lr, tps, peak, sample_text, eta, status="training"):
    if val is not None:
        history.append({"step": step, "train": round(tr, 4), "val": round(val, 4)})
    m = {"preset": f"{A.preset} (SFT)", "params_m": round(n_params / 1e6, 1),
         "step": step, "max_iters": MAX_ITERS, "lr": lr,
         "train_loss": round(tr, 4), "val_loss": round(val, 4) if val else None,
         "tok_per_s": round(tps), "peak_gb": round(peak, 2), "eta_min": round(eta / 60, 1),
         "batch_tokens": A.micro_batch * A.grad_accum * A.ctx, "ctx": A.ctx,
         "sample": sample_text, "history": history[-500:], "status": status}
    tmp = A.metrics + ".tmp"
    json.dump(m, open(tmp, "w"))
    os.replace(tmp, A.metrics)


def save_ckpt(path, it, include_opt=True):
    blob = {"model": raw_model.state_dict(), "it": it, "cfg": cfg.__dict__,
            "history": history, "args": vars(A), "stage": "sft",
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy": np.random.get_state(), "python": random.getstate()}}
    if include_opt:
        blob["opt"] = opt.state_dict()
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


STOP = {"now": False}
for _s in ("SIGTERM", "SIGINT", "SIGBREAK"):
    if hasattr(signal, _s):
        try:
            signal.signal(getattr(signal, _s), lambda *_: STOP.update(now=True))
        except (ValueError, OSError):
            pass

# ------------------------------------------------------------------ loop
model.train()
t0 = time.time()
seen = 0
last_ckpt = time.time()
last_sample = "(warming up...)"
torch.cuda.reset_peak_memory_stats()
loss = torch.tensor(float("nan"))

for it in range(start_it, MAX_ITERS + 1):
    lr = lr_at(it)
    for g in opt.param_groups:
        g["lr"] = lr
    for _ in range(A.grad_accum):
        xb, yb = get_batch("train")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(xb, yb)
            loss = loss / A.grad_accum
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), A.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    seen += A.micro_batch * A.grad_accum * A.ctx

    if it % A.eval_every == 0:
        torch.cuda.synchronize()
        dt = time.time() - t0
        val = evaluate()
        if it % A.sample_every == 0:
            last_sample = sample()
        peak = torch.cuda.max_memory_allocated() / 1e9
        eta = (MAX_ITERS - it) * (dt / max(it - start_it, 1))
        write_metrics(it, loss.item() * A.grad_accum, val, lr,
                      seen / max(dt, 1e-9), peak, last_sample, eta)
        print(f"step {it:5d}/{MAX_ITERS} | train {loss.item()*A.grad_accum:.3f} | "
              f"val {val:.3f} | lr {lr:.2e} | ep {it/STEPS_PER_EPOCH:.2f} | "
              f"{peak:.2f} GB", flush=True)

    if (time.time() - last_ckpt) >= A.ckpt_minutes * 60 or STOP["now"]:
        save_ckpt(LATEST, it)
        last_ckpt = time.time()
        print(f"  checkpoint @ step {it}", flush=True)
        if STOP["now"]:
            print("stopped by signal; resume with --resume auto", flush=True)
            sys.exit(130)

save_ckpt(os.path.join(A.ckpt_dir, f"{RUN}_final.pt"), MAX_ITERS, include_opt=False)
save_ckpt(LATEST, MAX_ITERS)
write_metrics(MAX_ITERS, loss.item() * A.grad_accum, evaluate(), lr_at(MAX_ITERS),
              seen / max(time.time() - t0, 1e-9),
              torch.cuda.max_memory_allocated() / 1e9, sample(), 0, status="done")
print(f"\nSFT complete -> {A.ckpt_dir}/{RUN}_final.pt", flush=True)
print(f"sample: {sample()!r}", flush=True)
