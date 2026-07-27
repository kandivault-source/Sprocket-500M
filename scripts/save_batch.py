"""Save the scaled Sprocket batch from the workflow journal + preview the voice range."""
import json
import os

JOURNAL = r"C:\Users\Daniel\.claude\projects\D--Razer-3D-Models-Etsy-KandiVaultLLC-AI-Claude-Artifacts---Web-projects-LLM\92af27b2-e7da-4cf2-a2fc-e61dc39b750d\subagents\workflows\wf_dbbc25fd-cf3\journal.jsonl"
OUT = "data/synthetic/batch01.jsonl"
os.makedirs("data/synthetic", exist_ok=True)


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
        if line:
            try:
                find_examples(json.loads(line), raw)
            except json.JSONDecodeError:
                pass

seen, uniq = set(), []
for e in raw:
    k = e["user"].strip().lower()
    if k and k not in seen:
        seen.add(k)
        uniq.append(e)

with open(OUT, "w", encoding="utf-8") as f:
    for e in uniq:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

by_len = sorted(uniq, key=lambda e: len(e["assistant"]))
print(f"SAVED {len(uniq)} unique examples -> {OUT}\n")
print("===== TERSEST (gruff mood check) =====")
for e in by_len[:4]:
    print(f"U: {e['user']}\nS: {e['assistant']}\n")
print("===== LONGEST (expansive mood check) =====")
for e in by_len[-2:]:
    print(f"U: {e['user']}\nS: {e['assistant'][:700]}\n")
