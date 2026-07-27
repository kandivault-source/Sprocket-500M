"""build_corpus.py — download FineWeb-Edu and tokenize it straight to train.bin.

Designed for a rented box on a metered clock, so:
  * RESUMABLE. A manifest records every finished shard and the exact byte length
    of train.bin. Re-running continues; it never re-tokenizes or double-appends.
    (The pod WILL get preempted. This is not optional.)
  * DISK-FRUGAL. Each parquet is deleted after tokenizing (--keep-parquet to
    override). Peak disk = train.bin + one shard, NOT the whole raw dataset:
    50B tokens is ~225 GB of parquet but only ~100 GB of .bin, so deleting as we
    go is the difference between a $10 volume and a $50 one.
  * NO INTERMEDIATE .txt. Straight parquet -> token ids -> uint16.

Subsets (FineWeb-Edu is ~1.3T tokens total; pick the smallest that covers you):
    sample/10BT   ~10B GPT-2 tokens   (good for a first real run)
    sample/100BT  ~100B
    sample/350BT  ~350B
    data          everything

NOTE ON COUNTS: HuggingFace's "10BT" is measured in GPT-2 tokens. Our 32k vocab
is smaller, so the same text yields MORE tokens — expect roughly +10-25%. Always
size by --target-tokens (measured with OUR tokenizer), never by the subset name.

  py scripts/build_corpus.py --target-tokens 200_000_000        # local dry run
  py scripts/build_corpus.py --target-tokens 50_000_000_000 --subset sample/100BT
"""
import argparse
import json
import os
import sys
import time

import numpy as np

DEFAULT_TOKENIZER = "config/tokenizer/tokenizer.json"


def human(n):
    for u in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{u}"
        n /= 1000
    return f"{n:.1f}P"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-tokens", type=float, default=200e6)
    ap.add_argument("--subset", default="sample/10BT")
    ap.add_argument("--repo", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--out", default="data/processed/train.bin")
    ap.add_argument("--manifest", default=None, help="default: <out>.manifest.json")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--scratch", default="data/raw/_shards")
    ap.add_argument("--keep-parquet", action="store_true")
    ap.add_argument("--batch-docs", type=int, default=2000)
    ap.add_argument("--flush-tokens", type=int, default=16_000_000,
                    help="buffer this many tokens (~32MB) before writing; large "
                         "values matter enormously on network storage")
    ap.add_argument("--max-shards", type=int, default=0, help="0 = unlimited")
    a = ap.parse_args()

    target = int(a.target_tokens)
    manifest_path = a.manifest or (a.out + ".manifest.json")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(a.scratch, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    EOT = tok.token_to_id("<|endoftext|>")
    if EOT is None:
        sys.exit("tokenizer has no <|endoftext|>")
    if tok.get_vocab_size() > 65535:
        sys.exit("vocab > 65535 does not fit uint16; change the bin dtype")

    # ---------------------------------------------------------------- resume
    man = {"done": [], "tokens": 0, "bytes": 0, "subset": a.subset}
    if os.path.exists(manifest_path):
        man = json.load(open(manifest_path, encoding="utf-8"))
        if man.get("subset") != a.subset:
            sys.exit(f"manifest is for subset {man.get('subset')!r}, you asked for "
                     f"{a.subset!r}. Use a different --out, or delete the manifest.")
    # Truncate any partial tail from a kill mid-append, so we never double-write.
    if os.path.exists(a.out):
        actual = os.path.getsize(a.out)
        if actual != man["bytes"]:
            print(f"  train.bin is {actual:,} B but manifest says {man['bytes']:,} — "
                  f"truncating partial shard", flush=True)
            with open(a.out, "r+b") as f:
                f.truncate(man["bytes"])
    elif man["bytes"]:
        sys.exit(f"{a.out} is missing but the manifest claims {man['bytes']:,} bytes. "
                 f"Delete {manifest_path} to start over.")

    print("=" * 74)
    print(f"BUILD CORPUS  {a.repo}:{a.subset}  ->  {a.out}")
    print("=" * 74)
    print(f"  target {human(target)} tokens (our 32k tokenizer)")
    if man["tokens"]:
        print(f"  RESUMING: {len(man['done'])} shards done, {human(man['tokens'])} tokens "
              f"({100*man['tokens']/target:.1f}% of target)")

    api = HfApi()
    files = api.list_repo_files(a.repo, repo_type="dataset")
    prefix = a.subset.rstrip("/") + "/"
    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not shards:
        sys.exit(f"no parquet files under {prefix!r} in {a.repo}")
    todo = [s for s in shards if s not in man["done"]]
    print(f"  {len(shards)} shards in subset, {len(todo)} remaining\n")

    t0 = time.time()
    processed = 0
    out_f = open(a.out, "ab")
    try:
        for shard in todo:
            if man["tokens"] >= target:
                break
            if a.max_shards and processed >= a.max_shards:
                print(f"  stopping: --max-shards {a.max_shards} reached")
                break
            s_t0 = time.time()
            path = hf_hub_download(a.repo, shard, repo_type="dataset",
                                   local_dir=a.scratch)
            dl = time.time() - s_t0

            n_tok, n_doc = 0, 0
            pf = pq.ParquetFile(path)
            # Buffer before writing. Measured on RunPod: writing once per document
            # (194k tiny writes) ran at 453K tok/s against a NETWORK volume, vs
            # 2.86M tok/s on a local SSD — a 6x penalty that a local dry run cannot
            # reveal. At 50B tokens that is ~30h vs ~5h, i.e. ~$75 of wasted H100.
            buf = []
            buf_tok = 0

            def flush():
                nonlocal buf, buf_tok
                if buf:
                    np.concatenate(buf).tofile(out_f)
                    buf, buf_tok = [], 0

            for batch in pf.iter_batches(batch_size=a.batch_docs, columns=["text"]):
                texts = [t for t in batch.column("text").to_pylist() if t]
                if not texts:
                    continue
                # encode_batch is multithreaded in Rust — no multiprocessing needed.
                for enc in tok.encode_batch(texts, add_special_tokens=False):
                    ids = enc.ids
                    ids.append(EOT)          # document separator
                    buf.append(np.asarray(ids, dtype=np.uint16))
                    buf_tok += len(ids)
                    n_tok += len(ids)
                n_doc += len(texts)
                if buf_tok >= a.flush_tokens:
                    flush()
                if man["tokens"] + n_tok >= target:
                    break
            flush()

            out_f.flush()
            os.fsync(out_f.fileno())
            man["done"].append(shard)
            man["tokens"] += n_tok
            man["bytes"] = out_f.tell()
            # Manifest is written AFTER fsync so it can never claim data that
            # isn't durably on disk.
            tmp = manifest_path + ".tmp"
            json.dump(man, open(tmp, "w"), indent=1)
            os.replace(tmp, manifest_path)
            processed += 1

            el = time.time() - s_t0
            rate = man["tokens"] / max(time.time() - t0, 1e-9)
            eta = (target - man["tokens"]) / max(rate, 1e-9)
            print(f"  [{len(man['done']):4d}/{len(shards)}] {os.path.basename(shard)} "
                  f"{n_doc:,} docs {human(n_tok)} tok in {el:.0f}s (dl {dl:.0f}s) | "
                  f"total {human(man['tokens'])} ({100*man['tokens']/target:.1f}%) | "
                  f"{human(rate)} tok/s | ETA {eta/3600:.1f}h", flush=True)

            if not a.keep_parquet:
                try:
                    os.remove(path)
                except OSError:
                    pass
    finally:
        out_f.close()

    gb = man["bytes"] / 1e9
    print(f"\n  DONE: {human(man['tokens'])} tokens -> {a.out} ({gb:.2f} GB)")
    print(f"  shards used: {len(man['done'])}")
    if man["tokens"] < target:
        print(f"  ! short of target {human(target)} — subset exhausted; "
              f"use a larger --subset")
    print(f"\n  train with:  py -m src.train.train --preset 500m --data {a.out} "
          f"--loader memmap --resume auto")


if __name__ == "__main__":
    main()
