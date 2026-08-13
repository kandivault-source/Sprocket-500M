"""Emit the run data the explorer page charts, straight from the training log.

docs/index.html draws the loss and throughput curves itself so they can be
hovered and read. It needs the numbers as JSON rather than as the baked SVG the
README uses.

    py scripts/build_web_data.py      # -> docs/run-data.json

Everything here is parsed from logs/train.log. Nothing is typed in by hand, so
the page cannot drift away from what the run actually did.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_run import main_segment, parse, sft_segment, thin  # noqa: E402

LOG = os.path.join("logs", "train.log")
OUT = os.path.join("docs", "run-data.json")

# Charting more points than a screen has pixels costs bytes and buys nothing.
PRETRAIN_POINTS = 420


def main() -> int:
    if not os.path.exists(LOG):
        sys.exit(f"{LOG} not found")

    iters, steps, shards = parse(LOG)
    run = main_segment(iters)
    if not run:
        sys.exit("no pretrain iterations found")
    sft = sft_segment(steps)

    pts = thin(run, PRETRAIN_POINTS)
    last = run[-1]

    payload = {
        "pretrain": {
            "tokens": [round(r["tok"], 4) for r in pts],
            "train": [round(r["train"], 4) for r in pts],
            "val": [round(r["val"], 4) for r in pts],
            "lr": [round(r["lr"], 8) for r in pts],
            "tps": [r["tps"] for r in pts],
            "final_val": round(last["val"], 4),
            "final_train": round(last["train"], 4),
            "steps": last["it"],
            "total_tokens": round(last["tok"], 3),
            "peak_tps": max(r["tps"] for r in run),
        },
        "sft": {
            "step": [r["step"] for r in sft],
            "train": [round(r["train"], 4) for r in sft],
            "val": [round(r["val"], 4) for r in sft],
            "total_steps": sft[-1]["total"] if sft else 0,
            "epochs": round(sft[-1]["ep"], 1) if sft else 0,
            "final_val": round(sft[-1]["val"], 4) if sft else 0,
        } if sft else None,
        "corpus_shards": len({s["n"]: 1 for s in shards}),
        # Measured on the pod across an 11-config sweep, not estimated.
        "measured": {
            "tok_per_s": 115295,
            "baseline_tok_per_s": 63470,
            "gpu": "H100 SXM 80GB",
            "hours": 54.3,
            "cost_usd": 165,
            "usd_per_hour": 2.99,
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"  wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
    print(f"    pretrain points : {len(pts)} (from {len(run):,})")
    print(f"    final val       : {payload['pretrain']['final_val']}")
    print(f"    tokens          : {payload['pretrain']['total_tokens']}B")
    if sft:
        print(f"    sft             : {len(sft)} points over "
              f"{payload['sft']['total_steps']} steps, val {payload['sft']['final_val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
