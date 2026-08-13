"""Save the mass Sprocket run (turns format) from the workflow journal + preview."""
import json
import os

from _journal import journal

JOURNAL = journal("92af27b2-e7da-4cf2-a2fc-e61dc39b750d", "wf_4a6959ad-f70")
OUT = "data/synthetic/batch02.jsonl"
os.makedirs("data/synthetic", exist_ok=True)


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

seen, uniq = set(), []
for e in raw:
    first = next((t["content"] for t in e["turns"] if t.get("role") == "user"), "")
    k = first.strip().lower()
    if k and k not in seen:
        seen.add(k)
        uniq.append(e)

with open(OUT, "w", encoding="utf-8") as f:
    for e in uniq:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

multi = [e for e in uniq if len(e["turns"]) > 2]
print(f"saved {len(uniq)} conversations ({len(multi)} multi-turn) -> {OUT}")
if multi:
    print("\n===== MULTI-TURN SAMPLE =====")
    for t in multi[0]["turns"]:
        print(f"[{t['role']}] {t['content'][:350]}")
