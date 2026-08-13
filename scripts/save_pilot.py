"""Pull the Sprocket instruct examples out of the workflow journal and save as JSONL."""
import json
import os

from _journal import journal

JOURNAL = journal("92af27b2-e7da-4cf2-a2fc-e61dc39b750d", "wf_85e660a0-bd8")
OUT_DIR = "data/synthetic"
os.makedirs(OUT_DIR, exist_ok=True)


def find_examples(obj, acc):
    if isinstance(obj, dict):
        ex = obj.get("examples")
        if isinstance(ex, list):
            for e in ex:
                if isinstance(e, dict) and "user" in e and "assistant" in e:
                    acc.append(e)
        for v in obj.values():
            find_examples(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_examples(v, acc)


raw = []
with open(JOURNAL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            find_examples(json.loads(line), raw)
        except json.JSONDecodeError:
            continue

seen, uniq = set(), []
for e in raw:
    k = e["user"].strip().lower()
    if k and k not in seen:
        seen.add(k)
        uniq.append(e)

path = os.path.join(OUT_DIR, "pilot.jsonl")
with open(path, "w", encoding="utf-8") as f:
    for e in uniq:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

chars = sum(len(e["user"]) + len(e["assistant"]) for e in uniq)
print(f"saved {len(uniq)} unique examples ({len(raw)} raw incl. duplicates), "
      f"~{chars // 4:,} usable tokens -> {path}")
