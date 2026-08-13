"""Check the page's parameter arithmetic against the real PyTorch model.

docs/modelmath.js reproduces the model's size calculation so the explorer page
can respond to someone changing the width or the layer count. If that formula
drifts from src/model/model.py, the page quietly reports wrong numbers about the
model it is describing.

    py scripts/verify_web_modelmath.py

Builds each preset for real and compares totals. Needs a node binary; uses the
embedded one shipped with LM Studio unless $NODE says otherwise.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

JS = os.path.join("docs", "modelmath.js")

NODE_CANDIDATES = [
    os.environ.get("NODE", ""),
    r"C:\Users\Daniel\.lmstudio\.internal\utils\node.exe",
    "node",
]

HARNESS = r"""
const path = require("path");
const M = require(path.resolve(process.argv[2]));
const cfgs = JSON.parse(process.argv[3]);
process.stdout.write(JSON.stringify(cfgs.map((c) => M.paramCount(c))));
"""


def find_node() -> str:
    for cand in NODE_CANDIDATES:
        if not cand:
            continue
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except (OSError, subprocess.CalledProcessError):
            continue
    sys.exit("No node binary found. Set $NODE to one.")


def main() -> int:
    sys.path.insert(0, "src")
    from model.model import GPT, PRESETS  # noqa: E402

    names = list(PRESETS.keys())
    cfgs = []
    for n in names:
        c = PRESETS[n]
        cfgs.append({
            "vocab": c.vocab_size, "dim": c.dim, "layers": c.n_layers,
            "heads": c.n_heads, "kvHeads": c.n_kv_heads or c.n_heads,
            "ffnMultiple": c.ffn_multiple_of,
        })

    node = find_node()
    with tempfile.TemporaryDirectory() as td:
        h = os.path.join(td, "h.js")
        with open(h, "w", encoding="utf-8") as f:
            f.write(HARNESS)
        proc = subprocess.run(
            [node, h, os.path.abspath(JS), json.dumps(cfgs)],
            capture_output=True, text=True, encoding="utf-8",
        )
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
        sys.exit("node harness failed")

    js = json.loads(proc.stdout)

    failures = 0
    for name, got in zip(names, js):
        model = GPT(PRESETS[name])
        want_total = model.num_params()
        want_nonemb = model.num_params(non_embedding=True)
        del model

        ok = got["total"] == want_total and got["nonEmbedding"] == want_nonemb
        if ok:
            print(f"  ok   {name:<11} {want_total:>13,} params "
                  f"({want_nonemb:,} non-embedding)")
        else:
            failures += 1
            print(f"  FAIL {name}")
            print(f"       pytorch: total {want_total:,}  non-emb {want_nonemb:,}")
            print(f"       js     : total {got['total']:,}  non-emb {got['nonEmbedding']:,}")

    print(f"\n  {len(names) - failures}/{len(names)} presets agree")
    if failures:
        sys.exit(f"{failures} preset(s) diverged")
    print("  page arithmetic matches the real model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
