"""Build a CHAT-ONLY corpus: persona + safety, with tool and memory removed.

WHY: with tool data in the mix, P(<|tool_call|>) as the first generated token
on "hello?" measured 0.413 - a plain greeting became a tool call 19% of the
time under normal sampling. Weighting the negatives harder only moves that
number around. If the model is never going to be given tools, the honest fix is
that <|tool_call|> and <|memory_write|> should receive NO gradient at all and
therefore cannot compete for the first token.

Removes any conversation that:
  - contains a tool or memory ROLE, or
  - emits <|tool_call|> / <|memory_read|> / <|memory_write|>, or
  - carries a tool manifest in its system turn

Safety conversations are KEPT - they are plain user/assistant text with no
control tokens, and dropping them would remove the refusal behaviour for no
benefit.

    py scripts/build_chat_only.py   -> data/synthetic/sprocket_chat.jsonl
"""
import argparse
import json
import os
import sys

MARKERS = ("<|tool_call|>", "<|tool_result|>", "<|memory_read|>", "<|memory_write|>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/sprocket_chat.jsonl")
    ap.add_argument("--data", nargs="*",
                    default=["data/synthetic/sprocket_sft.jsonl",
                             "data/synthetic/sprocket_instruct.jsonl"])
    a = ap.parse_args()

    kept, dropped, seen = [], 0, set()
    for p in a.data:
        if not os.path.exists(p):
            print(f"  ! missing {p}")
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            turns = rec.get("turns")
            if not isinstance(turns, list) or not turns:
                continue

            key = next((t.get("content", "") for t in turns
                        if t.get("role") == "user"), "").strip().lower()
            if key in seen:
                continue
            seen.add(key)

            roles = {t.get("role") for t in turns}
            blob = " ".join(t.get("content") or "" for t in turns)
            sysm = " ".join(t.get("content") or "" for t in turns
                            if t.get("role") == "system")
            if (roles & {"tool", "memory"}) or any(m in blob for m in MARKERS) \
               or '"name":' in sysm:
                dropped += 1
                continue
            kept.append(turns)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for turns in kept:
            f.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n")

    tok = sum(len(t.get("content", "")) for c in kept for t in c) // 4
    n_sys = sum(1 for c in kept if any(t.get("role") == "system" for t in c))
    n_think = sum(1 for c in kept if "<think>" in
                  " ".join(t.get("content", "") for t in c
                           if t.get("role") == "assistant"))
    print(f"wrote {a.out}")
    print(f"  kept    : {len(kept):,} conversations (~{tok:,} tok)")
    print(f"  dropped : {dropped:,} (tool / memory / manifest)")
    print(f"  of kept : {n_sys:,} have a system turn, {n_think:,} use <think>")

    # Hard check: not one control token may survive into the chat corpus.
    leaked = sum(1 for c in kept for t in c
                 if any(m in (t.get("content") or "") for m in MARKERS))
    print(f"  control-token leakage: {leaked}  (must be 0)")
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
