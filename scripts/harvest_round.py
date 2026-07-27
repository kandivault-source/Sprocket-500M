"""Harvest one or more production-round workflow journals into master corpus
files, with global dedup against everything already banked.

  SFT (Opus, kind='know')          -> data/synthetic/sprocket_sft.jsonl   ({turns})
  pretrain docs (Haiku, story/convo/prose) -> data/synthetic/pretrain_synth.jsonl ({source,text,...})

Dedup:
  SFT      keyed on first user-turn text (also preloaded from the legacy
           sprocket_instruct.jsonl so we never re-add an existing convo).
  pretrain keyed on the first 120 chars of the doc text.

Usage:  py scripts/harvest_round.py <run_id> [<run_id> ...]
"""
import json
import os
import sys

if len(sys.argv) < 2:
    sys.exit("usage: harvest_round.py <run_id> [<run_id> ...]")
RUN_IDS = sys.argv[1:]

import glob as _glob
PROJECT_ROOT = r"C:\Users\Daniel\.claude\projects\D--Razer-3D-Models-Etsy-KandiVaultLLC-AI-Claude-Artifacts---Web-projects-LLM"


def find_journal(rid):
    """Locate a run's journal.jsonl. Accepts a bare run id (searched across ALL
    session dirs, not just one), a workflow dir, or a direct path to journal.jsonl."""
    if os.path.isfile(rid):
        return rid
    if os.path.isdir(rid):
        jp = os.path.join(rid, "journal.jsonl")
        return jp if os.path.isfile(jp) else None
    hits = _glob.glob(os.path.join(PROJECT_ROOT, "*", "subagents", "workflows", rid, "journal.jsonl"))
    return hits[0] if hits else None


SYN = "data/synthetic"
SFT_MASTER = f"{SYN}/sprocket_sft.jsonl"
PRE_MASTER = f"{SYN}/pretrain_synth.jsonl"
LEGACY_SFT = f"{SYN}/sprocket_instruct.jsonl"
os.makedirs(SYN, exist_ok=True)


def walk(obj, know, docs):
    if isinstance(obj, dict):
        k = obj.get("kind")
        if k == "know" and isinstance(obj.get("examples"), list):
            for e in obj["examples"]:
                if isinstance(e, dict) and isinstance(e.get("turns"), list) and e["turns"]:
                    know.append(e)
        elif k and k not in ("know", "brief") and isinstance(obj.get("docs"), list):
            # any doc-bearing pretrain kind: story / convo / prose / article / ...
            for d in obj["docs"]:
                if isinstance(d, dict) and d.get("text"):
                    d = dict(d)
                    d["source"] = k
                    if not d.get("genre"):
                        d["genre"] = d.get("topic") or k  # normalize to {text, genre, source}
                    docs.append(d)
        for v in obj.values():
            walk(v, know, docs)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, know, docs)


def sft_key(e):
    return next((t["content"] for t in e["turns"] if t.get("role") == "user"), "").strip().lower()


def doc_key(d):
    return (d.get("text") or "").strip()[:120].lower()


# ---- preload existing dedup keys + counts ----
sft_seen, sft_existing = set(), 0
for path in (LEGACY_SFT, SFT_MASTER):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(e.get("turns"), list) and e["turns"]:
                sft_seen.add(sft_key(e))
                if path == SFT_MASTER:
                    sft_existing += 1

pre_seen, pre_existing = set(), 0
if os.path.exists(PRE_MASTER):
    for line in open(PRE_MASTER, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        pre_seen.add(doc_key(d))
        pre_existing += 1

# ---- collect from runs ----
know_raw, docs_raw = [], []
for rid in RUN_IDS:
    jp = find_journal(rid)
    if not jp:
        print(f"  ! journal not found for {rid}")
        continue
    for line in open(jp, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                walk(json.loads(line), know_raw, docs_raw)
            except json.JSONDecodeError:
                pass

# ---- dedup + append ----
new_know = []
for e in know_raw:
    kk = sft_key(e)
    if kk and kk not in sft_seen:
        sft_seen.add(kk)
        new_know.append({"turns": e["turns"]})
if new_know:
    with open(SFT_MASTER, "a", encoding="utf-8") as f:
        for e in new_know:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

new_docs, by_src = [], {}
for d in docs_raw:
    dk = doc_key(d)
    if dk and dk not in pre_seen:
        pre_seen.add(dk)
        new_docs.append(d)
        by_src[d["source"]] = by_src.get(d["source"], 0) + 1
if new_docs:
    with open(PRE_MASTER, "a", encoding="utf-8") as f:
        for d in new_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def toks_chat(items):
    return sum(sum(len(t.get("content", "")) for t in e["turns"]) for e in items) // 4


def toks_docs(items):
    return sum(len(d.get("text", "")) for d in items) // 4


def rate(new, raw):
    return f"{100 * (1 - new / raw):.0f}% dup" if raw else "-"


print("=" * 64)
print(f"HARVEST ROUND  runs={','.join(RUN_IDS)}")
print("=" * 64)
print(f"raw collected : know={len(know_raw)}  docs={len(docs_raw)}")
print(f"dedup dropped : know={len(know_raw) - len(new_know)} ({rate(len(new_know), len(know_raw))})  "
      f"docs={len(docs_raw) - len(new_docs)} ({rate(len(new_docs), len(docs_raw))})")
print(f"SFT   (Opus)  : +{len(new_know):4d} new  (~{toks_chat(new_know):,} tok)  "
      f"| master now {sft_existing + len(new_know)}  [+ legacy {os.path.basename(LEGACY_SFT)}]")
print(f"pretrain docs : +{len(new_docs):4d} new  (~{toks_docs(new_docs):,} tok)  "
      f"| master now {pre_existing + len(new_docs)}")
if by_src:
    print("   by source :", ", ".join(f"{k}=+{v}" for k, v in sorted(by_src.items())))
