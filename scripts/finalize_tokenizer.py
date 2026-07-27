"""finalize_tokenizer.py — bind the locked special tokens to their reserved ids.

Locked decision #6: <think> -> id 8, </think> -> id 9,
                   <|memory_read|> -> id 10, <|memory_write|> -> id 11.

The 32k vocab was trained with 8 placeholder slots at ids 8-15
(<|reserved_0|>..<|reserved_7|>). This RENAMES four of them in place. It does not
add tokens, so:
  * vocab stays exactly 32000  -> no embedding resize, ever
  * every other token id is untouched -> data/processed/train.bin stays valid
  * no SFT data regeneration is needed -> the literal "<think>" string in the
    corpus now encodes to the single id 8 instead of ['<','think','>']

WHY IT MATTERS BEYOND US: without this, anyone who downloads the released model
gets a tokenizer that splits <think> into 3 tokens the model was never trained on,
and reasoning output silently breaks. It must be baked into the artifact, not
applied as a private encode-time hack.

  py scripts/finalize_tokenizer.py            # dry run
  py scripts/finalize_tokenizer.py --apply    # rewrite (backs up first)
"""
import json
import os
import shutil
import sys

import numpy as np

TOK = "config/tokenizer/tokenizer.json"
TRAIN_BIN = "data/processed/train.bin"
APPLY = "--apply" in sys.argv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RENAME = {8: "<think>", 9: "</think>", 10: "<|memory_read|>", 11: "<|memory_write|>"}

j = json.load(open(TOK, encoding="utf-8"))
vocab = j["model"]["vocab"]
inv = {v: k for k, v in vocab.items()}

print("=" * 72)
print("FINALIZE TOKENIZER — bind locked special tokens to reserved ids")
print("=" * 72)
print(f"  vocab size before: {len(vocab):,}")

todo = {}
for tid, new in RENAME.items():
    cur = inv.get(tid)
    if cur == new:
        print(f"  id {tid:2d}: already {new!r}")
    elif cur is None or not cur.startswith("<|reserved_"):
        sys.exit(f"  ABORT: id {tid} holds {cur!r}, not a reserved slot. "
                 f"Refusing to clobber a real token.")
    else:
        todo[tid] = (cur, new)
        print(f"  id {tid:2d}: {cur!r} -> {new!r}")
    if new in vocab and vocab[new] != tid:
        sys.exit(f"  ABORT: {new!r} already exists at id {vocab[new]}")

# Is anything in the tokenized corpus using ids 8-15? If so a rename would
# retroactively change what those tokens mean. Verify rather than assume.
if os.path.exists(TRAIN_BIN):
    arr = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    hits = int(((arr >= 8) & (arr <= 15)).sum())
    print(f"\n  {TRAIN_BIN}: {len(arr):,} tokens, occurrences of ids 8-15 = {hits}")
    if hits:
        print("  ! non-zero — renaming changes the meaning of already-tokenized data.")
    else:
        print("  reserved slots unused in the corpus -> rename is provably harmless.")
else:
    print(f"\n  ({TRAIN_BIN} not found — skipping corpus check)")

if not todo:
    print("\n  nothing to do.")
    sys.exit(0)
if not APPLY:
    print("\n  DRY RUN — re-run with --apply")
    sys.exit(0)

shutil.copy2(TOK, TOK + ".bak")
for tid, (cur, new) in todo.items():
    del vocab[cur]
    vocab[new] = tid
for at in j.get("added_tokens", []):
    if at["id"] in RENAME:
        at["content"] = RENAME[at["id"]]
        at["special"] = True

assert len(vocab) == 32000, f"vocab drifted to {len(vocab)}"
with open(TOK, "w", encoding="utf-8") as f:
    json.dump(j, f, ensure_ascii=False)

from tokenizers import Tokenizer  # noqa: E402
T = Tokenizer.from_file(TOK)
print(f"\n  backed up -> {TOK}.bak")
print(f"  vocab size after : {T.get_vocab_size():,}")
ok = True
for tid, new in RENAME.items():
    ids = T.encode(new, add_special_tokens=False).ids
    good = ids == [tid]
    ok &= good
    print(f"    {new:18s} -> {ids}  {'OK' if good else 'FAILED'}")
probe = "<think>reason</think>Oi. Sprocket."
e = T.encode(probe, add_special_tokens=False)
print(f"  round-trip: {probe!r} -> {e.ids[:6]}... -> {T.decode(e.ids)!r}")
sys.exit(0 if ok else 1)
