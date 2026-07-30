"""Extract the CAPABILITY conversations into their own file so SFT can oversample them.

WHY: the first real SFT fit produced zero tool calls, zero memory writes and no
refusals. Tool data was 4.4% of the corpus over 3 epochs - about 2,800 gradient
exposures of a brand-new control token, which is not enough for a 501M model to
learn a fixed output format. Upweighting costs nothing at 6M tokens and is the
only lever left now that the pretrain budget is closed.

Passing the output to --data N times repeats it N times, because
load_conversations() simply concatenates every path it is given.

    py scripts/build_upweighted.py
    -> data/synthetic/sprocket_caps.jsonl

Detection:
  tool    - a "tool" role, a <|tool_call|>, OR a system turn carrying a tool
            manifest. The manifest clause is what catches the NEGATIVES (tools
            offered, none needed); upweighting only the positives would train a
            model that calls a tool for everything.
  memory  - a "memory" role or a model-emitted memory token. Again includes the
            silent/restraint cases.
  safety  - recovered from the safety workflow journals, which is the only way
            to catch BOTH the refusals and the over-refusal counterweights.
            Upweighting refusals alone would make it refuse ordinary requests.
"""
import argparse
import glob
import json
import os
import re
import sys

PROJECT_ROOT = (r"C:\Users\Daniel\.claude\projects"
                r"\D--Razer-3D-Models-Etsy-KandiVaultLLC-AI-Claude-Artifacts---Web-projects-LLM")
SAFETY_RUNS = ["wf_bb14b306-418", "wf_11501c44-c1a"]


def key(turns):
    for t in turns:
        if t.get("role") == "user":
            return (t.get("content") or "").strip().lower()
    return ""


def load_jsonl(p):
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r.get("turns"), list) and r["turns"]:
            out.append(r["turns"])
    return out


def safety_keys():
    keys = set()
    for rid in SAFETY_RUNS:
        hits = glob.glob(os.path.join(PROJECT_ROOT, "*", "subagents", "workflows",
                                      rid, "journal.jsonl"))
        if not hits:
            print(f"  ! journal not found for {rid}")
            continue
        for line in open(hits[0], encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "result":
                continue
            val = rec.get("result") or rec.get("value") or {}
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    continue
            for e in val.get("examples") or []:
                if isinstance(e.get("turns"), list) and e["turns"]:
                    keys.add(key(e["turns"]))
    return keys


def classify(turns, safe):
    roles = {t.get("role") for t in turns}
    asst = " ".join(t.get("content", "") for t in turns
                    if t.get("role") == "assistant")
    sysm = " ".join(t.get("content", "") for t in turns
                    if t.get("role") == "system")
    if "tool" in roles or "<|tool_call|>" in asst or '"name":' in sysm:
        return "tool"
    if "memory" in roles or "<|memory_write|>" in asst or "<|memory_read|>" in asst:
        return "memory"
    if key(turns) in safe:
        return "safety"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/sprocket_caps.jsonl")
    ap.add_argument("--data", nargs="*", default=["data/synthetic/sprocket_sft.jsonl",
                                                  "data/synthetic/sprocket_instruct.jsonl"])
    a = ap.parse_args()

    print("recovering safety conversation keys from the generation journals...")
    safe = safety_keys()
    print(f"  {len(safe):,} safety keys")

    convos, seen = [], set()
    for p in a.data:
        for turns in load_jsonl(p):
            k = key(turns)
            if k in seen:
                continue
            seen.add(k)
            convos.append(turns)
    print(f"deduped corpus: {len(convos):,} conversations")

    counts, out = {}, []
    for turns in convos:
        c = classify(turns, safe)
        if c:
            counts[c] = counts.get(c, 0) + 1
            out.append(turns)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for turns in out:
            f.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")

    # A SECOND, narrower file: only the conversations that actually EMIT a
    # control token. Measured after the 4x run - tool calling imprinted and
    # memory writing did not, and the reason is exposure count, not weighting:
    #   <|tool_call|>    590 emitting convos  x4 = 2,360
    #   <|memory_write|> 237 emitting convos  x4 =   948
    # Tools got 2.5x the gradient signal. Listing this file a few extra times
    # lifts the emitters without touching the restraint/negative balance, since
    # the negatives all stay at their base weight in the main caps file.
    emit = [t for t in out
            if any(m in " ".join(x.get("content", "") for x in t
                                 if x.get("role") == "assistant")
                   for m in ("<|memory_write|>", "<|memory_read|>", "<|tool_call|>"))]
    ep = a.out.replace(".jsonl", "_emit.jsonl")
    with open(ep, "w", encoding="utf-8") as f:
        for turns in emit:
            f.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")
    n_tool = sum(1 for t in emit if any("<|tool_call|>" in x.get("content", "")
                 for x in t if x.get("role") == "assistant"))
    n_mw = sum(1 for t in emit if any("<|memory_write|>" in x.get("content", "")
               for x in t if x.get("role") == "assistant"))
    n_mr = sum(1 for t in emit if any("<|memory_read|>" in x.get("content", "")
               for x in t if x.get("role") == "assistant"))
    print(f"\nwrote {ep}: {len(emit):,} token-emitting conversations")
    print(f"  tool_call {n_tool:,}   memory_write {n_mw:,}   memory_read {n_mr:,}")

    tok = sum(len(t.get("content", "")) for c in out for t in c) // 4
    print(f"\nwrote {a.out}: {len(out):,} conversations (~{tok:,} tok)")
    for k in sorted(counts):
        print(f"  {counts[k]:6,}  {k}   ({100*counts[k]/max(len(convos),1):.1f}% of corpus)")
    print(f"\n  capability share now : {100*len(out)/max(len(convos),1):.1f}%")
    for n in (3, 4, 5):
        share = n * len(out) / (len(convos) + (n - 1) * len(out))
        print(f"  at {n}x upweight       : {100*share:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
