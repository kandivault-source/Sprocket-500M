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
import re
import sys

if len(sys.argv) < 2:
    sys.exit("usage: harvest_round.py <run_id> [<run_id> ...]")
RUN_IDS = sys.argv[1:]

import glob as _glob
from _journal import project_root

PROJECT_ROOT = project_root()


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


# Conversation kinds that are SFT data. "safety" comes from the separate
# safety round; without it listed here every safety record falls through to the
# doc branch, finds no "docs" key, and is silently discarded.
SFT_KINDS = ("know", "safety")

# Roles src/train/sft_data.py can actually render. Anything else makes
# ChatTemplate.render() raise, and pack() catches that and drops the
# conversation with no error and no loss-curve signal - so reject it loudly
# HERE instead of losing it invisibly at training time.
RENDERABLE = {"system", "user", "assistant", "memory", "tool"}


def walk(obj, know, docs):
    if isinstance(obj, dict):
        k = obj.get("kind")
        if k in SFT_KINDS and isinstance(obj.get("examples"), list):
            for e in obj["examples"]:
                if isinstance(e, dict) and isinstance(e.get("turns"), list) and e["turns"]:
                    know.append(e)
        elif k and k not in SFT_KINDS + ("brief",) and isinstance(obj.get("docs"), list):
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


THINK_CLOSE = "</think>"


def marker_placement_ok(turns):
    """Reject conversations where a control marker sits in the wrong place.

    The host parses these positionally, so placement IS the contract:
      * <|tool_call|> must be the ENTIRE assistant turn (optionally after a
        think block). Prose before it means the host sees chatter where it
        expects JSON, and one such example teaches the model to narrate before
        calling. Measured rate in round 14: 1 in 881.
      * <|memory_write|> must open the turn, so the host can strip it before
        display.
      * Neither marker may appear inside a think block, and host-only markers
        (<|tool_result|>) may never appear in an assistant turn at all.
    """
    for t in turns:
        if t.get("role") != "assistant":
            # A model-emitted marker in a host turn is nonsense.
            if any(m in (t.get("content") or "")
                   for m in ("<|tool_call|>", "<|memory_write|>")):
                return False
            continue
        c = t.get("content") or ""
        if "<|tool_result|>" in c:
            return False

        # No marker may appear INSIDE the reasoning block. Note this is strictly
        # the text between the tags - a <|memory_write|> that precedes <think>
        # is the correct, documented layout, so "everything before </think>" is
        # the wrong region to test.
        for m in re.finditer(r"<think>(.*?)</think>", c, re.S):
            if any(k in m.group(1) for k in ("<|tool_call|>", "<|memory_write|>",
                                             "<|memory_read|>")):
                return False

        rest = c.lstrip()
        if "<|memory_write|>" in c:
            if not rest.startswith("<|memory_write|>"):
                return False
            rest = rest.split("\n", 1)[1].lstrip() if "\n" in rest else ""
        # A think block may sit between the save and the rest of the reply.
        if rest.startswith("<think>") and THINK_CLOSE in rest:
            rest = rest.split(THINK_CLOSE, 1)[1].lstrip()
        if "<|tool_call|>" in c and not rest.startswith("<|tool_call|>"):
            return False
        if "<|memory_read|>" in c and not rest.startswith("<|memory_read|>"):
            return False
    return True


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

# ---- clean, validate, dedup, append ----
# Generators occasionally emit a blank turn (seen: an empty second user turn
# from a double-texting example). It renders as a bare header+<|end|> pair that
# teaches nothing, so drop the turn but keep the conversation.
new_know, bad_role, emptied, misplaced = [], 0, 0, 0
for e in know_raw:
    turns = [t for t in e["turns"]
             if isinstance(t, dict) and (t.get("content") or "").strip()]
    if len(turns) != len(e["turns"]):
        emptied += 1
    if not turns:
        continue
    roles = {t.get("role") for t in turns}
    if not roles <= RENDERABLE:
        bad_role += 1
        print(f"  ! dropped: unrenderable role(s) {sorted(roles - RENDERABLE)}")
        continue
    if not any(t.get("role") == "assistant" for t in turns):
        continue                      # nothing trainable in it
    if not marker_placement_ok(turns):
        misplaced += 1
        continue
    kk = sft_key({"turns": turns})
    if kk and kk not in sft_seen:
        sft_seen.add(kk)
        new_know.append({"turns": turns})
if emptied:
    print(f"  stripped blank turns from {emptied} conversations")
if bad_role:
    print(f"  DROPPED {bad_role} conversations with unrenderable roles")
if misplaced:
    print(f"  DROPPED {misplaced} conversations with a misplaced control marker")
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
