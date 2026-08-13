"""Emit a compact tokenizer payload for the static web explorer.

docs/index.html reimplements this tokenizer in JavaScript so the page can encode
text with no server behind it. That reimplementation needs the vocabulary and the
merge list; it does not need the training-time scaffolding that tokenizer.json
also carries.

    py scripts/build_web_tokenizer.py     # -> docs/tokenizer-data.json

Run scripts/verify_web_tokenizer.py afterwards to confirm the JavaScript encoder
still agrees with the real one.
"""
from __future__ import annotations

import json
import os

SRC = os.path.join("config", "tokenizer", "tokenizer.json")
OUT = os.path.join("docs", "tokenizer-data.json")


def main() -> int:
    with open(SRC, encoding="utf-8") as f:
        tk = json.load(f)

    model = tk["model"]
    vocab = model["vocab"]
    merges = model["merges"]

    # id -> token string, so the page can look up by index instead of shipping
    # the mapping twice.
    by_id = [""] * (max(vocab.values()) + 1)
    for tok, i in vocab.items():
        by_id[i] = tok

    specials = {t["content"]: t["id"] for t in tk.get("added_tokens", [])}

    payload = {
        "vocab_size": len(vocab),
        "tokens": by_id,
        # Stored as "a b" strings: about 40% smaller than nested arrays.
        "merges": [f"{a} {b}" for a, b in merges],
        "specials": specials,
        "pre_tokenizer": tk["pre_tokenizer"]["type"],
        "add_prefix_space": tk["pre_tokenizer"].get("add_prefix_space", False),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    print(f"  wrote {OUT}")
    print(f"    vocab   : {payload['vocab_size']:,}")
    print(f"    merges  : {len(payload['merges']):,}")
    print(f"    specials: {len(specials)}")
    print(f"    size    : {os.path.getsize(OUT)/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
