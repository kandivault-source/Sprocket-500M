"""Harvest the corpus pilot workflow journal into separated, deduped files.

Routes by the 'kind' tag each agent returns:
  know  -> chat-format SFT (turns)   -> data/synthetic/pilot_know.jsonl
  story -> pretrain doc              -> data/synthetic/pilot_pretrain_story.jsonl
  convo -> pretrain doc              -> data/synthetic/pilot_pretrain_convo.jsonl
  prose -> pretrain doc              -> data/synthetic/pilot_pretrain_prose.jsonl
  brief -> the Opus generation brief -> data/synthetic/pilot_brief.txt

Usage:  py scripts/harvest_pilot.py <run_id>
        (default run id is the pilot below)
"""
import json
import os
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "wf_37758c5f-25d"
BASE = r"C:\Users\Daniel\.claude\projects\D--Razer-3D-Models-Etsy-KandiVaultLLC-AI-Claude-Artifacts---Web-projects-LLM\bca2d753-d961-4fe8-93a1-62487f1ab070\subagents\workflows"
JOURNAL = os.path.join(BASE, RUN, "journal.jsonl")
SYN = "data/synthetic"
os.makedirs(SYN, exist_ok=True)

buckets = {"know": [], "story": [], "convo": [], "prose": [], "brief": []}


def walk(obj):
    """Collect every dict carrying a recognized 'kind' tag with its payload."""
    if isinstance(obj, dict):
        k = obj.get("kind")
        if k == "know" and isinstance(obj.get("examples"), list):
            buckets["know"].extend(obj["examples"])
        elif k in ("story", "convo", "prose") and isinstance(obj.get("docs"), list):
            for d in obj["docs"]:
                if isinstance(d, dict) and d.get("text"):
                    buckets[k].append(d)
        elif k == "brief" and obj.get("brief"):
            buckets["brief"].append(obj)
        for v in obj.values():
            walk(v)
    elif isinstance(obj, list):
        for v in obj:
            walk(v)


with open(JOURNAL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                walk(json.loads(line))
            except json.JSONDecodeError:
                pass


def dedup(items, keyfn):
    seen, out = set(), []
    for it in items:
        k = keyfn(it)
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


# ---- know (chat turns) ----
know = dedup(
    [e for e in buckets["know"] if isinstance(e.get("turns"), list) and e["turns"]],
    lambda e: next((t["content"] for t in e["turns"] if t.get("role") == "user"), "").strip().lower(),
)
with open(f"{SYN}/pilot_know.jsonl", "w", encoding="utf-8") as f:
    for e in know:
        f.write(json.dumps({"turns": e["turns"]}, ensure_ascii=False) + "\n")

# ---- pretrain docs ----
doc_counts = {}
for kind in ("story", "convo", "prose"):
    docs = dedup(buckets[kind], lambda d: (d.get("text") or "").strip()[:120].lower())
    doc_counts[kind] = docs
    with open(f"{SYN}/pilot_pretrain_{kind}.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

# ---- brief ----
if buckets["brief"]:
    with open(f"{SYN}/pilot_brief.txt", "w", encoding="utf-8") as f:
        f.write(buckets["brief"][0]["brief"])


def toks_chat(items):
    return sum(sum(len(t.get("content", "")) for t in e["turns"]) for e in items) // 4


def toks_docs(items):
    return sum(len(d.get("text", "")) for d in items) // 4


print("=" * 60)
print(f"HARVEST {RUN}")
print("=" * 60)
print(f"know (Opus SFT)   : {len(know):4d} examples  ~{toks_chat(know):,} tok")
for kind in ("story", "convo", "prose"):
    d = doc_counts[kind]
    print(f"{kind:5s} (Haiku doc)  : {len(d):4d} docs      ~{toks_docs(d):,} tok")
tot = toks_chat(know) + sum(toks_docs(doc_counts[k]) for k in ("story", "convo", "prose"))
print(f"{'TOTAL':17s}: ~{tot:,} tok")
print(f"brief captured    : {'yes' if buckets['brief'] else 'NO'}")

# ---- print samples for eyeball verification ----
print("\n" + "=" * 60 + "\nSAMPLES\n" + "=" * 60)
if know:
    print("\n--- KNOW (first) ---")
    for t in know[0]["turns"]:
        print(f"[{t['role']}] {t['content'][:500]}")
for kind in ("story", "convo", "prose"):
    if doc_counts[kind]:
        d = doc_counts[kind][0]
        print(f"\n--- {kind.upper()} (first, genre/topic={d.get('genre') or d.get('topic','')}) ---")
        print(d["text"][:600])
