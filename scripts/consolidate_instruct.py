"""Harvest the 100-agent run, then consolidate ALL synthetic batches into one
deduped chat-format file: data/synthetic/sprocket_instruct.jsonl"""
import json
import os

from _journal import journal

SYN = "data/synthetic"
JOURNAL = journal("92af27b2-e7da-4cf2-a2fc-e61dc39b750d", "wf_44a22f59-fe6")


def find_convos(obj, acc):
    if isinstance(obj, dict):
        ex = obj.get("examples")
        if isinstance(ex, list):
            for e in ex:
                if isinstance(e, dict) and isinstance(e.get("turns"), list) and e["turns"]:
                    acc.append(e)
        for v in obj.values():
            find_convos(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_convos(v, acc)


raw = []
with open(JOURNAL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                find_convos(json.loads(line), raw)
            except json.JSONDecodeError:
                pass
with open(f"{SYN}/batch03.jsonl", "w", encoding="utf-8") as f:
    for e in raw:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
print(f"batch03: {len(raw)} conversations")


def to_turns(e):
    if isinstance(e.get("turns"), list):
        return e["turns"]
    if "user" in e and "assistant" in e:
        return [{"role": "user", "content": e["user"]}, {"role": "assistant", "content": e["assistant"]}]
    return None


convos = []
for fn in ["pilot.jsonl", "batch01.jsonl", "batch02.jsonl", "batch03.jsonl"]:
    p = f"{SYN}/{fn}"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                t = to_turns(json.loads(line))
                if t:
                    convos.append({"turns": t})

seen, uniq = set(), []
for e in convos:
    first = next((t["content"] for t in e["turns"] if t.get("role") == "user"), "")
    k = first.strip().lower()
    if k and k not in seen:
        seen.add(k)
        uniq.append(e)

with open(f"{SYN}/sprocket_instruct.jsonl", "w", encoding="utf-8") as f:
    for e in uniq:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
tok = sum(sum(len(t["content"]) for t in e["turns"]) for e in uniq) // 4
multi = sum(1 for e in uniq if len(e["turns"]) > 2)
print(f"CONSOLIDATED: {len(uniq)} unique conversations ({multi} multi-turn), "
      f"~{tok:,} tokens -> {SYN}/sprocket_instruct.jsonl")
