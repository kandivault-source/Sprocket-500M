"""scrub_sft.py — remove records whose SYSTEM prompt names the persona.

The locked system-prompt policy: a system turn may carry behavioural/formatting/
audience modifiers, or assign a different role to play, but it must NEVER name
Sprocket, the goblin, or the persona. A prompt that does — even as a negative
constraint like "no goblin dialect" — presupposes the persona as something the
prompt knows about and controls, which is precisely the "costume" failure mode:
it makes the goblin attributable to, and therefore removable with, the prompt.

Generation is ~99.94% compliant, so this catches rare stragglers rather than a
systemic problem. Run after every harvest of a `focus:"system"` round.

  py scripts/scrub_sft.py            # dry run — report only
  py scripts/scrub_sft.py --apply    # rewrite the master (backs up first)
"""
import json
import os
import re
import shutil
import sys

MASTER = "data/synthetic/sprocket_sft.jsonl"
APPLY = "--apply" in sys.argv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Naming the persona in a system prompt is fatal regardless of polarity.
# Abstract mentions ("no persona", "set aside any character voice") are FINE —
# they negate without naming, and the override stream needs them.
FATAL = re.compile(r"\b(sprocket|goblin|engineer-sage|loupe|tinkerer|cogs?)\b", re.I)

if not os.path.exists(MASTER):
    sys.exit(f"missing {MASTER}")

kept, dropped = [], []
for line in open(MASTER, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        continue
    sys_text = " ".join(
        t.get("content", "")
        for t in (rec.get("turns") or [])
        if isinstance(t, dict) and t.get("role") == "system"
    )
    (dropped if sys_text and FATAL.search(sys_text) else kept).append(rec)

print("=" * 70)
print(f"SCRUB  {MASTER}")
print("=" * 70)
print(f"  records          : {len(kept) + len(dropped):,}")
print(f"  FATAL system leak: {len(dropped)}")
for r in dropped:
    st = next(t["content"] for t in r["turns"] if t.get("role") == "system")
    print(f"    - {st[:160]!r}")

if not dropped:
    print("\n  nothing to scrub.")
elif APPLY:
    backup = MASTER + ".bak"
    shutil.copy2(MASTER, backup)
    with open(MASTER, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  backed up -> {backup}")
    print(f"  rewrote   -> {MASTER}  ({len(kept):,} records, {len(dropped)} removed)")
else:
    print("\n  DRY RUN — re-run with --apply to remove.")
