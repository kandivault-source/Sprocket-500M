"""Download one FineWeb-Edu sample shard (cleaned Common Crawl) and extract its
text to a plain .txt file for training the tokenizer + prototype.

FineWeb-Edu IS Common Crawl, already deduplicated and education-quality-filtered —
so we get Common Crawl's volume without building the cleaning pipeline ourselves.
"""
import os
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

REPO = "HuggingFaceFW/fineweb-edu"
OUT_DIR = "data/raw"
CHAR_CAP = 800_000_000  # ~200M tokens — plenty for a tokenizer + prototype
os.makedirs(OUT_DIR, exist_ok=True)

api = HfApi()
print("listing repo files...", flush=True)
files = api.list_repo_files(REPO, repo_type="dataset")
cands = sorted(f for f in files if f.startswith("sample/10BT/") and f.endswith(".parquet"))
if not cands:
    cands = sorted(f for f in files if f.endswith(".parquet"))
assert cands, "no parquet files found in repo"
target = cands[0]
print(f"downloading: {target}", flush=True)
path = hf_hub_download(REPO, target, repo_type="dataset", local_dir=OUT_DIR)
print(f"downloaded -> {path}", flush=True)

out_txt = os.path.join(OUT_DIR, "fineweb_000.txt")
pf = pq.ParquetFile(path)
nchars, ndocs = 0, 0
with open(out_txt, "w", encoding="utf-8") as f:
    for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
        for t in batch.column("text").to_pylist():
            if t:
                f.write(t)
                f.write("\n")
                nchars += len(t)
                ndocs += 1
        if nchars >= CHAR_CAP:
            break
print(f"extracted {ndocs:,} documents, {nchars:,} chars (~{nchars//4:,} tokens)", flush=True)
print(f"saved -> {out_txt}", flush=True)
