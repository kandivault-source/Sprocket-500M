"""Pull the cloud run's train.log down from the HF repo to logs/, and report
exactly which graphable series it contains.

WHY: train.log is the only record of throughput, the loss curves, the LR
schedule and the corpus-build tokenization rates. It is written with `tee -a`
on the network volume, so it is CUMULATIVE across every pod that has ever run -
smoke runs, the launch that died on a stale manifest, the 20B pretrain, and the
SFT stage all append to the same file. Newest copy is always a superset.

    py scripts/fetch_run_log.py              # -> logs/train.log
    py scripts/fetch_run_log.py --csv        # also writes logs/train_metrics.csv

Needs HF_TOKEN (already a User env var on this machine).
"""
import argparse
import csv
import os
import re
import sys
import urllib.request

REPO = os.environ.get("HF_REPO", "kandivault/sprocket-500m")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# Pretrain: iter N | train X | val Y | lr Z | N tok/s | N GB | N.NNNB tok
ITER_RX = re.compile(
    r"iter\s+(\d+) \| train ([\d.]+) \| val ([\d.]+) \| lr ([\d.e+-]+) \| "
    r"([\d,]+) tok/s \| ([\d.]+) GB \| ([\d.]+)B tok")
# SFT: step N/M | train X | val Y | lr Z | ep E | N GB
STEP_RX = re.compile(
    r"step\s+(\d+)/(\d+) \| train ([\d.]+) \| val ([\d.]+) \| lr ([\d.e+-]+) \| "
    r"ep ([\d.]+)")
# Corpus build: [ n/N] shard.parquet D docs T tok in Ss (dl Ss) | ... | R tok/s
SHARD_RX = re.compile(
    r"\[\s*(\d+)/\s*(\d+)\]\s+(\S+)\s+([\d,]+) docs\s+([\d.]+)M tok in ([\d.]+)s"
    r".*?\|\s*([\d.]+[KM]?) tok/s")


def fetch(path):
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set")
    url = f"https://huggingface.co/{REPO}/resolve/main/{path}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/train.log")
    ap.add_argument("--csv", action="store_true",
                    help="also emit a tidy CSV of the pretrain curve")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(f"fetching debug/train.log from {REPO} ...")
    raw = fetch("debug/train.log")
    with open(a.out, "wb") as f:
        f.write(raw)
    txt = raw.decode("utf-8", "replace")
    lines = txt.splitlines()
    print(f"  wrote {a.out}  ({len(raw)/1e6:.2f} MB, {len(lines):,} lines)")

    iters = [m.groups() for l in lines for m in [ITER_RX.search(l)] if m]
    steps = [m.groups() for l in lines for m in [STEP_RX.search(l)] if m]
    shards = [m.groups() for l in lines for m in [SHARD_RX.search(l)] if m]

    print("\ngraphable series:")
    print(f"  {len(iters):6,}  pretrain: iter / train / val / lr / tok-s / GB / tokens")
    print(f"  {len(steps):6,}  SFT: step / train / val / lr / epoch")
    print(f"  {len(shards):6,}  corpus build: per-shard docs, tokens, seconds, tok/s")
    print(f"  {sum('PARITY' in l for l in lines):6,}  export logit-parity checks")
    print(f"  {sum('checkpoint @' in l for l in lines):6,}  checkpoint events")

    if iters:
        tps = [int(p[4].replace(",", "")) for p in iters]
        print(f"\npretrain span: iter {iters[0][0]} (train {iters[0][1]}) -> "
              f"iter {iters[-1][0]} (train {iters[-1][1]}, val {iters[-1][2]})")
        print(f"  tokens seen : {iters[-1][6]}B")
        print(f"  tok/s       : min {min(tps):,}  max {max(tps):,}")

    if a.csv and iters:
        p = os.path.join(os.path.dirname(a.out) or ".", "train_metrics.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iter", "train_loss", "val_loss", "lr",
                        "tok_per_s", "vram_gb", "tokens_b"])
            for g in iters:
                w.writerow([int(g[0]), float(g[1]), float(g[2]), float(g[3]),
                            int(g[4].replace(",", "")), float(g[5]), float(g[6])])
        print(f"\n  wrote {p}  ({len(iters):,} rows) - ready to plot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
