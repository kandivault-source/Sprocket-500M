"""inspect_round.py — read a generation round's ACTUAL output before trusting it.

Automated flags catch candidates; the flags are NOT the verdict. Past rounds showed
the regexes fire mostly on FALSE POSITIVES (a user asking about *their own* GPU, the
words "gear"/"cog"/"potato" appearing as legitimate problem content, an anti-disavowal
sentence matching a disavowal pattern). ALWAYS read the printed samples yourself.

  py scripts/inspect_round.py <run_id> [<run_id> ...] [--n 10] [--only system|think|multi|ident|flagged]

Checks (defects, each must be 0):
  hardware leak     specific GPU/model numbers in Sprocket's self-description
  disavowal         "just a persona/costume/act/roleplay/character" — identity must be OWNED
  persona in think   goblin voice inside a <think> block (it is a neutral scratchpad)
  unclosed think     <think> with no </think>, or </think> with no <think>
  role order         turns must alternate user/assistant after an optional leading system turn
System-round checks:
  sys persona leak   the system prompt names Sprocket/goblin/persona/accent — FORBIDDEN,
                     that would make the persona attributable to (and removable with) the prompt
  sys not first      a system turn appearing anywhere but position 0
"""
import argparse
import glob as _glob
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = (r"C:\Users\Daniel\.claude\projects"
                r"\D--Razer-3D-Models-Etsy-KandiVaultLLC-AI-Claude-Artifacts---Web-projects-LLM")

ap = argparse.ArgumentParser()
ap.add_argument("run_ids", nargs="+")
ap.add_argument("--n", type=int, default=10, help="samples to print (default 10)")
ap.add_argument("--only", default="", help="system | think | multi | ident | flagged")
A = ap.parse_args()


def find_journal(rid):
    if os.path.isfile(rid):
        return rid
    if os.path.isdir(rid):
        jp = os.path.join(rid, "journal.jsonl")
        return jp if os.path.isfile(jp) else None
    hits = _glob.glob(os.path.join(PROJECT_ROOT, "*", "subagents", "workflows", rid, "journal.jsonl"))
    return hits[0] if hits else None


def walk(obj, out):
    if isinstance(obj, dict):
        if obj.get("kind") == "know" and isinstance(obj.get("examples"), list):
            for e in obj["examples"]:
                if isinstance(e, dict) and isinstance(e.get("turns"), list) and e["turns"]:
                    out.append(e)
        for v in obj.values():
            walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, out)


THINK_RX = re.compile(r"<think>(.*?)</think>", re.DOTALL)
HW_RX = re.compile(r"\b(rtx|gtx|geforce|radeon|rx\s?\d|\d{3,4}\s?ti\b|4060|3090|4090|a100|h100)\b", re.I)
DISAVOW_RX = re.compile(
    r"\b(just|only|merely|simply)\s+(a\s+)?(persona|character|costume|act|role[- ]?play|voice|gimmick|shtick)\b"
    r"|\b(persona|character|costume|act)\s+is\s+(just|only|merely)\b"
    r"|\b(underneath|beneath|behind)\s+(the|this|my)\s+(persona|character|costume|act)\b"
    r"|\b(drop|break|set aside|remove)\s+(the|my)\s+(persona|character|costume|act)\b", re.I)
# goblin voice inside a <think> block — the scratchpad must be plain neutral English
VOICE_RX = re.compile(
    r"\b(oi\b|goblin|sprocket|cog|clockwork|loupe|tinker|whatcha|buildin['’]|diggin['’]|"
    r"tinkerin['’]|yer\b|ain['’]t\b|well-oiled|stripped gear|potato)\b", re.I)
# FATAL only: the system prompt NAMES the persona. Naming it — even to forbid it
# ("no goblin dialect") — presupposes the prompt controls it, which is the costume
# failure mode. Deliberately EXCLUDED, all verified false positives by reading:
#   "accent"/"dialect"  -> negative style constraints in the tone-clamp stream
#   "persona"/"in character" -> abstract override prompts ("set aside any persona")
#                               which negate without naming, and are intended.
SYS_LEAK_RX = re.compile(
    r"\b(sprocket|goblin|engineer-sage|loupe|tinkerer|cogs?)\b", re.I)

examples, missing = [], []
for rid in A.run_ids:
    jp = find_journal(rid)
    if not jp:
        missing.append(rid)
        continue
    with open(jp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    walk(json.loads(line), examples)
                except json.JSONDecodeError:
                    pass

for rid in missing:
    print(f"  ! journal not found for {rid}")
if not examples:
    sys.exit("no examples found")


def sys_turns(e):
    return [t for t in e["turns"] if t.get("role") == "system"]


def asst(e):
    return " ".join(t.get("content", "") for t in e["turns"] if t.get("role") == "assistant")


def flags(e):
    f = []
    a = asst(e)
    if HW_RX.search(a):
        f.append("hardware")
    if DISAVOW_RX.search(a):
        f.append("disavow")
    for blk in THINK_RX.findall(a):
        if VOICE_RX.search(blk):
            f.append("persona-in-think")
            break
    opens, closes = a.count("<think>"), a.count("</think>")
    if opens != closes:
        f.append("unclosed-think")
    st = sys_turns(e)
    if st:
        if any(SYS_LEAK_RX.search(t.get("content", "")) for t in st):
            f.append("SYS-PERSONA-LEAK")
        if e["turns"][0].get("role") != "system" or len(st) > 1:
            f.append("sys-not-first")
    roles = [t.get("role") for t in e["turns"]]
    body = roles[1:] if roles and roles[0] == "system" else roles
    if body != ["user" if i % 2 == 0 else "assistant" for i in range(len(body))]:
        f.append("role-order")
    return f


n_sys = sum(1 for e in examples if sys_turns(e))
n_think = sum(1 for e in examples if "<think>" in asst(e))
n_multi = sum(1 for e in examples if len([t for t in e["turns"] if t.get("role") != "system"]) > 2)
flagged = [(e, f) for e in examples for f in [flags(e)] if f]

print("=" * 74)
print(f"INSPECT  runs={','.join(A.run_ids)}")
print("=" * 74)
print(f"  examples            : {len(examples):,}")
print(f"  with system turn    : {n_sys:,}  ({100*n_sys/len(examples):.0f}%)")
print(f"  with <think>        : {n_think:,}  ({100*n_think/len(examples):.0f}%)")
print(f"  multi-turn          : {n_multi:,}  ({100*n_multi/len(examples):.0f}%)")
print("  flag counts (CANDIDATES — read the samples, most are false positives):")
counts = {}
for _, f in flagged:
    for x in f:
        counts[x] = counts.get(x, 0) + 1
print("   ", counts if counts else "none")

pool = examples
if A.only == "system":
    pool = [e for e in examples if sys_turns(e)]
elif A.only == "think":
    pool = [e for e in examples if "<think>" in asst(e)]
elif A.only == "multi":
    pool = [e for e in examples if len([t for t in e["turns"] if t.get("role") != "system"]) > 2]
elif A.only == "ident":
    pool = [e for e in examples if re.search(r"\b(who|what) (are|r) (you|u)\b|made you|are you real|"
                                             r"how big|pretend|just an ai", asst(e) + " " +
                                             " ".join(t.get("content", "") for t in e["turns"]), re.I)]
elif A.only == "flagged":
    pool = [e for e, _ in flagged]

if not pool:
    sys.exit(f"\nno examples match --only {A.only}")

step = max(1, len(pool) // max(A.n, 1))
picks = pool[::step][:A.n]
print(f"\n{'=' * 74}\nSAMPLES  ({len(picks)} of {len(pool)} matching, evenly spaced)\n{'=' * 74}")
for i, e in enumerate(picks, 1):
    f = flags(e)
    print(f"\n--- [{i}] {len(e['turns'])} turns" + (f"   FLAGS: {','.join(f)}" if f else "") + " " + "-" * 30)
    for t in e["turns"]:
        print(f"  <{t.get('role')}> {t.get('content','')}")
