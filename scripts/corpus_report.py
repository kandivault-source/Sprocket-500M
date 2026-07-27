"""corpus_report.py — authoritative state of the synthetic corpus.

Counts docs + tokens with the REAL trained 32k tokenizer (config/tokenizer/tokenizer.json),
separates MASTERS from the already-consolidated SOURCE files so nothing double-counts,
and breaks the two training layers down by area:

  PRETRAIN (Phase A, raw docs, NO persona)  : pretrain_synth.jsonl        -> by source/genre
  SFT/instruct (Phase B, Opus, persona)     : sprocket_sft.jsonl (new) +
                                              sprocket_instruct.jsonl (legacy), deduped
                                              -> single/multi-turn, think vs answer, persona density

Usage:  py scripts/corpus_report.py
"""
import json
import os
import re
from collections import Counter
from tokenizers import Tokenizer

SYN = "data/synthetic"
TOK = Tokenizer.from_file("config/tokenizer/tokenizer.json")

# goblin / mechanical persona markers (lowercased assistant text)
PERSONA_RX = re.compile(
    r"\b(sprocket|goblin|cog|gear|thread|tinker|clockwork|loupe|crank|gizmo|contraption|"
    r"potato|oi[ ,.]|whatcha|buildin|diggin|jam(?:med|s)?|stripped|well-oiled|bench|toolkit|"
    r"wrench|workbench|bolt|spanner|grease|greas|solder|weld|rust|gadget|widget|"
    r"wearin|leakin|tinkerin|crankin|riggin|riggin|re-?thread|spare part|gears?)\b"
    r"|(?:in['’]|\bme\b(?! and| or)|\byer\b|\bya\b|\bain['’]t\b|\bgonna\b|\bwee\b)"
)
THINK_RX = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# A system prompt must NEVER hand down the persona — that would make the goblin
# attributable to (and therefore removable with) the prompt. Any hit here is a defect.
# FATAL: the system prompt NAMES the persona — that would make the goblin
# attributable to (and removable with) the prompt. Must be 0.
# "accent"/"dialect"/"persona" are deliberately NOT here: they appear legitimately
# as negative constraints ("no regional dialect", "no persona") in the tone-clamp
# and override streams, which negate rather than grant. Verified by reading r10.
SYS_PERSONA_LEAK_RX = re.compile(
    r"\b(sprocket|goblin|engineer-sage|cogs?|loupe|tinkerer)\b", re.I)
# Expected non-zero: abstract "set aside any persona" override prompts.
SYS_ABSTRACT_RX = re.compile(r"\b(persona|personality|character voice|in character)\b", re.I)


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def ntok(texts):
    """Exact token count over a list of strings, batched."""
    if isinstance(texts, str):
        texts = [texts]
    texts = [t for t in texts if t]
    total = 0
    B = 2000
    for i in range(0, len(texts), B):
        for enc in TOK.encode_batch(texts[i:i + B]):
            total += len(enc.ids)
    return total


def rec_texts(rec):
    """Every text string in a record, whatever its schema."""
    if isinstance(rec.get("turns"), list):
        return [t.get("content", "") for t in rec["turns"] if isinstance(t, dict)]
    if "text" in rec:
        return [rec.get("text", "")]
    if "assistant" in rec or "user" in rec:
        return [rec.get("user", ""), rec.get("assistant", "")]
    return []


def sft_key(rec):
    if isinstance(rec.get("turns"), list):
        return next((t.get("content", "") for t in rec["turns"] if t.get("role") == "user"), "").strip().lower()
    return (rec.get("user", "") or "").strip().lower()


def assistant_text(rec):
    if isinstance(rec.get("turns"), list):
        return " ".join(t.get("content", "") for t in rec["turns"] if t.get("role") == "assistant")
    return rec.get("assistant", "") or ""


# ------------------------------------------------------------------ per-file
print("=" * 74)
print("PER-FILE  (raw, before dedup — some SOURCE files are folded into masters)")
print("=" * 74)
files = sorted(f for f in os.listdir(SYN) if f.endswith(".jsonl"))
per_file = {}
for fn in files:
    recs = load(os.path.join(SYN, fn))
    toks = ntok([t for r in recs for t in rec_texts(r)])
    schema = ("turns" if any("turns" in r for r in recs[:5])
              else "text" if any("text" in r for r in recs[:5])
              else "user/assistant" if recs else "-")
    per_file[fn] = (len(recs), toks)
    print(f"  {fn:34s} {len(recs):6,d} recs  ~{toks:9,d} tok   [{schema}]")

# ------------------------------------------------------------------ PRETRAIN
print("\n" + "=" * 74)
print("PRETRAIN  (Phase A raw docs — the FineWeb replacement layer)")
print("=" * 74)
pt = load(f"{SYN}/pretrain_synth.jsonl")
by_src, tok_src = Counter(), Counter()
by_model = Counter()
leak = 0
for d in pt:
    src = d.get("source") or d.get("kind") or d.get("genre") or "?"
    txt = d.get("text", "")
    by_src[src] += 1
    tok_src[src] += ntok(txt)
    by_model[d.get("model", d.get("gen", "?"))] += 1
    if "sprocket" in txt.lower() or "goblin" in txt.lower():
        leak += 1
pt_tok = sum(tok_src.values())
uniq = len({(d.get("text", "") or "")[:120].lower() for d in pt})
print(f"  master pretrain_synth.jsonl : {len(pt):,} docs   ~{pt_tok:,} tokens")
print(f"  unique (first 120 chars)    : {uniq:,}/{len(pt):,}   ({100*(1-uniq/max(len(pt),1)):.1f}% dup)")
print(f"  persona leak into pretrain  : {leak}  (must be ~0)")
print(f"  avg tokens/doc              : {pt_tok//max(len(pt),1):,}")
print("  by source/area:")
for src, n in by_src.most_common():
    print(f"    {src:12s} {n:6,d} docs   ~{tok_src[src]:9,d} tok   ({100*tok_src[src]/max(pt_tok,1):4.1f}%)")
print("  by generator model:", dict(by_model))

# pilot_pretrain_* overlap with master
pilot_pre = []
for fn in ("pilot_pretrain_story.jsonl", "pilot_pretrain_convo.jsonl", "pilot_pretrain_prose.jsonl"):
    pilot_pre += load(f"{SYN}/{fn}")
if pilot_pre:
    master_keys = {(d.get("text", "") or "")[:120].lower() for d in pt}
    novel = [d for d in pilot_pre if (d.get("text", "") or "")[:120].lower() not in master_keys]
    print(f"  pilot_pretrain_* files      : {len(pilot_pre):,} docs, of which {len(novel):,} NOT in master "
          f"(~{ntok([d.get('text','') for d in novel]):,} extra tok if added)")

# ------------------------------------------------------------------ SFT
print("\n" + "=" * 74)
print("SFT / INSTRUCT  (Phase B — Opus, persona lives here)")
print("=" * 74)
legacy = load(f"{SYN}/sprocket_instruct.jsonl")
newfmt = load(f"{SYN}/sprocket_sft.jsonl")

# authoritative union, deduped on first-user-turn (same key harvest uses)
seen, union = set(), []
src_tag = []
for tag, recs in (("new", newfmt), ("legacy", legacy)):
    for r in recs:
        k = sft_key(r)
        if k and k not in seen:
            seen.add(k)
            union.append(r)
            src_tag.append(tag)

sft_tok = ntok([t for r in union for t in rec_texts(r)])


def convo_turns(r):
    """Turns excluding a leading system turn — a system+user+assistant example is
    single-turn, not multi. Counting the system turn inflated multi-turn by exactly
    the number of system examples."""
    turns = r.get("turns")
    if not isinstance(turns, list):
        return []
    return [t for t in turns if isinstance(t, dict) and t.get("role") != "system"]


multi = sum(1 for r in union if len(convo_turns(r)) > 2)
single = len(union) - multi
think_recs = [r for r in union if THINK_RX.search(assistant_text(r))]
# think vs answer token split
think_tok = 0
for r in think_recs:
    for m in THINK_RX.findall(assistant_text(r)):
        think_tok += ntok(m)
persona_hits = sum(1 for r in union if PERSONA_RX.search(assistant_text(r).lower()))

# system-turn coverage (trains the <|system|> embedding; target ~15% of the corpus)
sys_recs, sys_leak, sys_abstract, sys_not_first, sys_tok = [], 0, 0, 0, 0
for r in union:
    turns = r.get("turns")
    if not isinstance(turns, list):
        continue
    sys_turns = [t for t in turns if isinstance(t, dict) and t.get("role") == "system"]
    if not sys_turns:
        continue
    sys_recs.append(r)
    sys_tok += ntok([t.get("content", "") for t in sys_turns])
    if turns[0].get("role") != "system":
        sys_not_first += 1
    if any(SYS_PERSONA_LEAK_RX.search(t.get("content", "")) for t in sys_turns):
        sys_leak += 1
    elif any(SYS_ABSTRACT_RX.search(t.get("content", "")) for t in sys_turns):
        sys_abstract += 1
sys_multi = sum(1 for r in sys_recs if len(convo_turns(r)) > 2)
sys_think = sum(1 for r in sys_recs if THINK_RX.search(assistant_text(r)))
sys_voice = sum(1 for r in sys_recs if PERSONA_RX.search(assistant_text(r).lower()))

print(f"  legacy sprocket_instruct    : {len(legacy):,} convos")
print(f"  new    sprocket_sft         : {len(newfmt):,} convos")
print(f"  DEDUPED UNION (authoritative): {len(union):,} convos   ~{sft_tok:,} tokens")
print(f"    from new={src_tag.count('new'):,}  from legacy={src_tag.count('legacy'):,}  "
      f"(overlap dropped {len(newfmt)+len(legacy)-len(union):,})")
print(f"  single-turn / multi-turn    : {single:,} / {multi:,}   ({100*multi/max(len(union),1):.0f}% multi)")
print(f"  with <think> reasoning      : {len(think_recs):,}   ({100*len(think_recs)/max(len(union),1):.0f}%)   "
      f"~{think_tok:,} think tok")
print(f"  persona markers in answer   : {persona_hits:,}/{len(union):,}   "
      f"({100*persona_hits/max(len(union),1):.0f}% of replies visibly in-voice)")
print(f"  avg tokens/convo            : {sft_tok//max(len(union),1):,}")

n_sys = len(sys_recs)
print(f"  with a <|system|> turn      : {n_sys:,}   ({100*n_sys/max(len(union),1):.0f}%)   "
      f"~{sys_tok:,} system tok   [target ~15%]")
if n_sys:
    print(f"    of those: multi-turn={sys_multi:,} ({100*sys_multi/n_sys:.0f}%)  "
          f"<think>={sys_think:,} ({100*sys_think/n_sys:.0f}%)  "
          f"in-voice reply={sys_voice:,} ({100*sys_voice/n_sys:.0f}%)")
    print(f"    DEFECTS: persona NAMED in system prompt={sys_leak}  "
          f"system turn not first={sys_not_first}   (both must be 0)")
    print(f"    ok: abstract 'set aside any persona' override prompts={sys_abstract}  "
          f"(expected non-zero — these negate, they don't grant)")

# ------------------------------------------------------------------ TOTALS
print("\n" + "=" * 74)
print("GRAND TOTAL (authoritative, deduped)")
print("=" * 74)
total = pt_tok + sft_tok
print(f"  pretrain synthetic : {len(pt):,} docs    ~{pt_tok:,} tok")
print(f"  SFT synthetic      : {len(union):,} convos  ~{sft_tok:,} tok")
print(f"  ------------------------------------------------")
print(f"  TOTAL synthetic    : ~{total:,} tokens")
for target, label in ((8_000_000_000, "8B"), (10_000_000_000, "10B")):
    print(f"    = {100*pt_tok/target:.2f}% of a {label}-token pretrain (pretrain layer alone), "
          f"or {(target)//max(pt_tok,1)}x upweight to fill it")
