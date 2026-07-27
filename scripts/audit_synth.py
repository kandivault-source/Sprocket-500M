"""Audit the synthetic corpus the parallel session produced: counts, format, quality, dedup."""
import json
from collections import Counter


def load(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


print("========== pretrain_synth.jsonl (Phase A raw docs) ==========")
pt = load("data/synthetic/pretrain_synth.jsonl")
print("count:", len(pt), "| keys:", list(pt[0].keys()))
print("kinds:", dict(Counter(d.get("kind", "?") for d in pt)))
print("model source:", dict(Counter(d.get("model", d.get("gen", "?")) for d in pt)))
lens = [len(d.get("text", "")) for d in pt]
print(f"avg {sum(lens)//len(lens)} chars/doc | ~{sum(lens)//4:,} tokens | unique texts {len(set(d.get('text','') for d in pt))}/{len(pt)}")
persona_leak = sum(1 for d in pt if "sprocket" in d.get("text", "").lower() or "goblin" in d.get("text", "").lower())
print(f"persona leak into pretrain docs (should be ~0): {persona_leak}")
for d in pt[:3]:
    print(f"\n[{d.get('kind','?')}] {d.get('text','')[:260]}")

print("\n\n========== sprocket_sft.jsonl (Phase B Opus instruct/persona) ==========")
sft = load("data/synthetic/sprocket_sft.jsonl")
print("count:", len(sft), "| keys:", list(sft[0].keys()))
with_turns = sum(1 for e in sft if isinstance(e.get("turns"), list))
multi = sum(1 for e in sft if len(e.get("turns", [])) > 2)
think = sum(1 for e in sft if "<think>" in json.dumps(e, ensure_ascii=False))
print(f"with turns: {with_turns} | multi-turn: {multi} | contain <think>: {think}")
firsts = [next((t.get("content", "") for t in e.get("turns", []) if t.get("role") == "user"), "") for e in sft]
print(f"unique first-user turns: {len(set(firsts))}/{len(firsts)}")
for e in sft[:2]:
    for t in e.get("turns", [])[:4]:
        print(f"  [{t.get('role')}] {t.get('content','')[:220]}")
    print("  ---")
