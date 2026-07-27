"""Train a model from scratch on the tokenized corpus.

Streams live metrics (loss curve, throughput, sample generations) to
dashboard/data/metrics.json so the dashboard can show training in real time.

RESUME (the thing that makes a multi-day spot run survivable):
    A checkpoint carries model + optimizer + LR-schedule position + RNG states +
    token count + dashboard history. Resuming continues the run as if it had never
    stopped, rather than silently restarting the cosine schedule at peak LR — which
    is the classic way to quietly ruin a long run.

    py -m src.train.train --preset 500m --resume auto

DATA LOADING:
    Small corpora are held resident on the GPU (fastest — batching becomes a pure
    on-device gather). Large ones are sampled from a memmap, because a 20-50B token
    train.bin is 40-100 GB and will not fit in VRAM or RAM. Chosen automatically;
    force with --loader {gpu,memmap}.

Examples:
    py -m src.train.train                                   # prototype defaults
    py -m src.train.train --preset 500m --max-iters 200000 --ctx 2048
    py -m src.train.train --resume auto                     # newest ckpt for preset
    py -m src.train.train --resume checkpoints/500m_latest.pt
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
from tokenizers import Tokenizer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="proto-75m", choices=sorted(PRESETS))
    p.add_argument("--data", default="data/processed/train.bin")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--metrics", default="dashboard/data/metrics.json")
    p.add_argument("--tokenizer", default="config/tokenizer/tokenizer.json")

    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--micro-batch", type=int, default=6)
    p.add_argument("--grad-accum", type=int, default=10)
    p.add_argument("--max-iters", type=int, default=800)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--peak-lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--sample-every", type=int, default=80)
    p.add_argument("--val-frac", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--resume", default=None,
                   help="'auto' for the newest checkpoint of this preset, or a path")
    p.add_argument("--ckpt-every", type=int, default=500, help="milestone save, in iters")
    p.add_argument("--ckpt-minutes", type=float, default=15.0,
                   help="wall-clock save interval; the real protection on spot instances")
    p.add_argument("--keep-last", type=int, default=3, help="milestone checkpoints to retain")
    p.add_argument("--loader", default="auto", choices=["auto", "gpu", "memmap"])
    p.add_argument("--gpu-resident-max-gb", type=float, default=4.0,
                   help="corpora smaller than this are held in VRAM")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


A = parse_args()
os.makedirs(A.ckpt_dir, exist_ok=True)
if os.path.dirname(A.metrics):
    os.makedirs(os.path.dirname(A.metrics), exist_ok=True)

device = A.device
torch.manual_seed(A.seed)
np.random.seed(A.seed)
random.seed(A.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LATEST = os.path.join(A.ckpt_dir, f"{A.preset}_latest.pt")

# ------------------------------------------------------------------ data
data_np = np.memmap(A.data, dtype=np.uint16, mode="r")
n_val = int(len(data_np) * A.val_frac)
n_train = len(data_np) - n_val
corpus_gb = data_np.nbytes / 1e9

use_gpu_resident = (A.loader == "gpu" or
                    (A.loader == "auto" and corpus_gb <= A.gpu_resident_max_gb))

if use_gpu_resident:
    # int16 is safe: vocab 32000 < 32767. Batching becomes a pure on-device gather.
    full = torch.from_numpy(np.asarray(data_np, dtype=np.int16)).to(device)
    train_t, val_t = full[:n_train], full[n_train:]
    mode = f"GPU-resident ({corpus_gb:.2f} GB)"

    def get_batch(split):
        d = train_t if split == "train" else val_t
        ix = torch.randint(0, d.numel() - A.ctx - 1, (A.micro_batch,), device=device)
        idx = ix[:, None] + torch.arange(A.ctx + 1, device=device)
        seq = d[idx].long()
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()
else:
    # Memmap path — the OS page cache does the work. Required above a few GB:
    # a 20B-token corpus is ~40 GB and fits in neither VRAM nor RAM.
    train_np, val_np = data_np[:n_train], data_np[n_train:]
    mode = f"memmap ({corpus_gb:.2f} GB, streamed)"

    def get_batch(split):
        d = train_np if split == "train" else val_np
        ix = np.random.randint(0, len(d) - A.ctx - 1, size=A.micro_batch)
        batch = np.stack([np.asarray(d[i:i + A.ctx + 1], dtype=np.int64) for i in ix])
        seq = torch.from_numpy(batch).pin_memory().to(device, non_blocking=True)
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

print(f"corpus {len(data_np):,} tokens (train {n_train:,} / val {n_val:,}) — {mode}", flush=True)

tok = Tokenizer.from_file(A.tokenizer)
EOT = tok.token_to_id("<|endoftext|>")

# ------------------------------------------------------------------ model
cfg = replace(PRESETS[A.preset], max_seq_len=A.ctx, dropout=0.0)
model = GPT(cfg).to(device)
n_params = model.num_params()
opt = model.configure_optimizers(A.weight_decay, A.peak_lr, device_type="cuda")
raw_model = model
if A.compile:
    model = torch.compile(model)
print(f"model {A.preset}: {n_params/1e6:.1f}M params", flush=True)


def lr_at(it):
    if it < A.warmup:
        return A.peak_lr * (it + 1) / A.warmup
    if it > A.max_iters:
        return A.min_lr
    r = (it - A.warmup) / max(A.max_iters - A.warmup, 1)
    return A.min_lr + 0.5 * (A.peak_lr - A.min_lr) * (1 + math.cos(math.pi * r))


# ------------------------------------------------------------------ resume
start_it, tokens_done, history = 0, 0, []


def resolve_resume(spec):
    if spec is None:
        return None
    if spec != "auto":
        return spec
    if os.path.exists(LATEST):
        return LATEST
    cands = glob.glob(os.path.join(A.ckpt_dir, f"{A.preset}_*.pt"))
    cands = [c for c in cands if not c.endswith("_final.pt")]
    return max(cands, key=os.path.getmtime) if cands else None


rpath = resolve_resume(A.resume)
if rpath:
    if not os.path.exists(rpath):
        sys.exit(f"resume checkpoint not found: {rpath}")
    ck = torch.load(rpath, map_location=device, weights_only=False)

    # Refuse a silent architecture mismatch — loading 500m weights into proto-75m
    # would either explode or, worse, partially succeed.
    saved_cfg = ck.get("cfg", {})
    for k in ("dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size"):
        if k in saved_cfg and saved_cfg[k] != getattr(cfg, k):
            sys.exit(f"config mismatch on resume: {k} checkpoint={saved_cfg[k]} "
                     f"current={getattr(cfg, k)}. Wrong --preset?")

    raw_model.load_state_dict(ck["model"])
    if "opt" in ck:
        opt.load_state_dict(ck["opt"])
    else:
        print("  ! checkpoint has no optimizer state — Adam moments restart from zero",
              flush=True)
    start_it = ck.get("it", 0) + 1
    tokens_done = ck.get("tokens_done", 0)
    history = ck.get("history", [])

    # Restoring RNG makes the resumed run follow the same data order it would have.
    rng = ck.get("rng", {})
    try:
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["cuda"]])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "python" in rng:
            random.setstate(rng["python"])
    except Exception as e:  # never let RNG restore kill a resume
        print(f"  ! RNG restore failed ({e}) — continuing with a fresh stream", flush=True)

    if start_it > A.max_iters:
        sys.exit(f"checkpoint is at iter {ck.get('it')} but --max-iters is {A.max_iters}; "
                 f"raise --max-iters to continue")
    print(f"RESUMED {rpath} at iter {start_it}/{A.max_iters} "
          f"({tokens_done/1e9:.3f}B tokens done, lr will be {lr_at(start_it):.2e})", flush=True)
else:
    print("starting fresh (no --resume)", flush=True)


# ------------------------------------------------------------------ eval / sample
@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for _ in range(A.eval_batches):
        x_, y_ = get_batch("val")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss_ = model(x_, y_)
        losses.append(loss_.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(n=100):
    model.eval()
    idx = torch.tensor([[EOT]], device=device)
    out = raw_model.generate(idx, n, temperature=0.8, top_k=50)
    model.train()
    return tok.decode([i for i in out[0].tolist() if i > 15])


def write_metrics(step, tr, val, lr, tps, peak, sample_text, eta, status="training"):
    if val is not None:
        history.append({"step": step, "train": round(tr, 4), "val": round(val, 4)})
    m = {
        "preset": A.preset, "params_m": round(n_params / 1e6, 1),
        "step": step, "max_iters": A.max_iters, "lr": lr,
        "train_loss": round(tr, 4), "val_loss": round(val, 4) if val else None,
        "tok_per_s": round(tps), "peak_gb": round(peak, 2), "eta_min": round(eta / 60, 1),
        "batch_tokens": A.micro_batch * A.grad_accum * A.ctx, "ctx": A.ctx,
        "sample": sample_text, "history": history[-500:], "status": status,
    }
    tmp = A.metrics + ".tmp"
    json.dump(m, open(tmp, "w"))
    os.replace(tmp, A.metrics)


def save_ckpt(path, it, include_opt=True):
    """Atomic: a preemption mid-write must not leave a truncated checkpoint where
    a good one used to be."""
    blob = {
        "model": raw_model.state_dict(),
        "it": it,
        "cfg": cfg.__dict__,
        "tokens_done": tokens_done,
        "history": history,
        "args": vars(A),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    if include_opt:
        blob["opt"] = opt.state_dict()
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def rotate(keep):
    ms = sorted(glob.glob(os.path.join(A.ckpt_dir, f"{A.preset}_[0-9]*.pt")),
                key=os.path.getmtime)
    for old in ms[:-keep] if keep > 0 else []:
        try:
            os.remove(old)
        except OSError:
            pass


# ------------------------------------------------------------------ preemption
# Spot/preemptible instances send SIGTERM with ~30-120s of notice. Python's default
# handler exits immediately, so without this the run dies between checkpoints and
# loses up to --ckpt-minutes of work. Set a flag and let the loop stop cleanly at
# the next iteration boundary (safer than raising out of a CUDA call).
STOP = {"now": False, "sig": None}


def _graceful(signum, _frame):
    STOP["now"], STOP["sig"] = True, signum
    print(f"\n[signal {signum}] preemption notice — finishing this iteration, "
          f"then checkpointing.", flush=True)


for _s in ("SIGTERM", "SIGINT", "SIGBREAK"):
    if hasattr(signal, _s):
        try:
            signal.signal(getattr(signal, _s), _graceful)
        except (ValueError, OSError):
            pass   # not the main thread, or unsupported on this platform

# ------------------------------------------------------------------ train loop
model.train()
x, y = get_batch("train")
t0 = time.time()
session_tokens = 0
last_ckpt_t = time.time()
torch.cuda.reset_peak_memory_stats()
last_sample = "(warming up...)"
loss = torch.tensor(float("nan"))

try:
    for it in range(start_it, A.max_iters + 1):
        it_t0 = time.time()
        lr = lr_at(it)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(A.grad_accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / A.grad_accum
            x, y = get_batch("train")
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), A.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step_tokens = A.micro_batch * A.grad_accum * A.ctx
        tokens_done += step_tokens
        session_tokens += step_tokens

        if start_it + 1 <= it <= start_it + 5:
            torch.cuda.synchronize()
            it_dt = time.time() - it_t0
            print(f"  iter {it}: {step_tokens/it_dt:,.0f} tok/s  ({it_dt:.2f}s/iter)", flush=True)

        if it % A.eval_every == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            tps = session_tokens / max(dt, 1e-9)
            peak = torch.cuda.max_memory_allocated() / 1e9
            val = evaluate()
            eta = (A.max_iters - it) * (dt / max(it - start_it, 1))
            if it % A.sample_every == 0:
                last_sample = sample(100)
            write_metrics(it, loss.item() * A.grad_accum, val, lr, tps, peak, last_sample, eta)
            print(f"iter {it:6d} | train {loss.item()*A.grad_accum:.3f} | val {val:.3f} | "
                  f"lr {lr:.2e} | {tps:,.0f} tok/s | {peak:.2f} GB | "
                  f"{tokens_done/1e9:.3f}B tok", flush=True)

        # Wall-clock save is what actually protects a spot run; iteration-count
        # saves alone can be hours apart on a big model.
        due_time = (time.time() - last_ckpt_t) >= A.ckpt_minutes * 60
        due_iter = it > start_it and A.ckpt_every > 0 and it % A.ckpt_every == 0
        if due_time or due_iter:
            save_ckpt(LATEST, it)
            last_ckpt_t = time.time()
            if due_iter:
                save_ckpt(os.path.join(A.ckpt_dir, f"{A.preset}_{it}.pt"), it)
                rotate(A.keep_last)
            print(f"  checkpoint @ iter {it} -> {LATEST}", flush=True)

        if STOP["now"]:
            save_ckpt(LATEST, it)
            print(f"stopped at iter {it} (signal {STOP['sig']}) -> {LATEST}\n"
                  f"resume with: --resume auto", flush=True)
            sys.exit(130)

except KeyboardInterrupt:
    save_ckpt(LATEST, it)
    print(f"\ninterrupted — checkpointed at iter {it} -> {LATEST}\n"
          f"resume with: --resume auto", flush=True)
    sys.exit(130)

save_ckpt(os.path.join(A.ckpt_dir, f"{A.preset}_final.pt"), A.max_iters, include_opt=False)
save_ckpt(LATEST, A.max_iters)
write_metrics(A.max_iters, loss.item() * A.grad_accum, evaluate(), lr_at(A.max_iters),
              session_tokens / max(time.time() - t0, 1e-9),
              torch.cuda.max_memory_allocated() / 1e9, last_sample, 0, status="done")
print("training complete", flush=True)
