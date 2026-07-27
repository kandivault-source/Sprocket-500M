"""Tokenize the raw corpus into a flat uint16 binary of token ids — the format the
training loop memory-maps. Documents are separated by the <|endoftext|> token so the
model learns where documents begin and end.

vocab 32000 < 65536, so uint16 (2 bytes/token) is exact and half the size of int32.
"""
import os
import numpy as np
from tokenizers import Tokenizer

INP = "data/raw/fineweb_000.txt"
OUT = "data/processed/train.bin"
BATCH = 2000
os.makedirs("data/processed", exist_ok=True)

tok = Tokenizer.from_file("config/tokenizer/tokenizer.json")
EOT = tok.token_to_id("<|endoftext|>")

total = 0
lines = []


def flush(lines, w):
    if not lines:
        return 0
    out = []
    for enc in tok.encode_batch(lines):
        out.extend(enc.ids)
        out.append(EOT)
    np.asarray(out, dtype=np.uint16).tofile(w)
    return len(out)


with open(INP, encoding="utf-8") as f, open(OUT, "wb") as w:
    for line in f:
        line = line.rstrip("\n")
        if line:
            lines.append(line)
        if len(lines) >= BATCH:
            total += flush(lines, w)
            lines = []
            if total % 20_000_000 < BATCH * 8:
                print(f"  {total:,} tokens...", flush=True)
    total += flush(lines, w)

mb = os.path.getsize(OUT) / 1e6
print(f"done: {total:,} tokens -> {OUT} ({mb:.0f} MB)", flush=True)
