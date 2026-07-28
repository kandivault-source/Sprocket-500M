"""Chat-template rendering, loss masking, and packing for the SFT stage.

WIRE FORMAT
    <|system|>text<|end|>          role "system"  — input, masked OUT
    <|memory_read|>text<|end|>     role "memory"  — host-injected recall, masked OUT
    <|tool_result|>text<|end|>     role "tool"    — host-injected tool output, masked OUT
    <|user|>text<|end|>            role "user"    — input, masked OUT
    <|assistant|>text<|end|>       role "assistant" — TRAINED (content + closing <|end|>)

LOSS MASKING — the single most important thing in this file.
    Train on assistant CONTENT and its closing <|end|>. Nothing else.
      * The <|assistant|> header is masked out: the host emits it to prompt
        generation, so the model never needs to produce it.
      * The closing <|end|> is masked IN. If you mask it out the model never
        learns to stop and generates until it hits max_tokens, forever.
      * Everything a host supplies — system, user, host-injected memory, tool
        results — is masked out. Training on user turns teaches the model to
        write the user's side of the conversation, the classic naive-SFT failure.

HOST-EMITS vs MODEL-EMITS — the asymmetry, and why it is built this way.
    Every token the HOST produces is a role header + masked content. Every token
    the MODEL must produce lives inside assistant content and is trained.

      host-injected (masked)          model-emitted (trained)
      ----------------------          -----------------------
      role "memory"  -> <|memory_read|>   <|memory_read|>  in assistant content
                                          = model ASKS to search its memory
      role "tool"    -> <|tool_result|>   <|tool_call|>    in assistant content
                                          = model REQUESTS a tool

    <|tool_call|> and <|memory_write|>/<|memory_read|> are therefore NOT roles.
    Making <|tool_call|> a role would be a silent disaster: role headers are
    emitted by put([head], False), so the model would never be trained to
    produce the very token that triggers a tool call, and tool use would be
    dead on arrival with a perfectly healthy-looking loss curve.

    An assistant turn that ends in a tool call still ends with a TRAINED
    <|end|> — that is what teaches the model to stop and hand control back to
    the host, which then executes the tool and injects role "tool".

    None of this shows up in the loss curve when it is wrong. Hence --self-test.

PACKING
    Conversations are concatenated, separated by <|endoftext|>, and cut into
    fixed ctx+1 blocks. No padding, no wasted compute, and no model change —
    GPT.forward already ignores label -1 via cross_entropy(ignore_index=-1).
    Cross-document attention within a block is accepted (standard practice —
    GPT-3/Llama pretraining packs the same way); the <|endoftext|> separator is
    the signal the model uses to reset.

Self-test:  py -m src.train.sft_data --self-test
Stats:      py -m src.train.sft_data --stats data/synthetic/sprocket_sft.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tokenizers import Tokenizer  # noqa: E402

DEFAULT_TOKENIZER = "config/tokenizer/tokenizer.json"

# Roles whose text the model must LEARN TO PRODUCE. Everything else is context.
TRAINED_ROLES = {"assistant"}
# Role -> opening special token. Only "assistant" is model-emitted; every other
# role is something the HOST supplies, so its header and body are masked out.
# "memory" = host-injected recall, "tool" = host-injected tool output.
ROLE_TOKEN = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "memory": "<|memory_read|>",
    "tool": "<|tool_result|>",
}


class ChatTemplate:
    def __init__(self, tokenizer_path=DEFAULT_TOKENIZER):
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.id = {name: self.tok.token_to_id(name) for name in
                   ["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>",
                    "<|end|>", "<think>", "</think>", "<|memory_read|>", "<|memory_write|>",
                    "<|tool_call|>", "<|tool_result|>"]}
        missing = [k for k, v in self.id.items() if v is None]
        if missing:
            raise SystemExit(
                f"tokenizer is missing {missing}. Run: py scripts/finalize_tokenizer.py --apply")
        self.eot = self.id["<|endoftext|>"]
        self.end = self.id["<|end|>"]

    def encode_text(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def render(self, turns):
        """turns -> (ids, train_mask). train_mask[i] is True iff the model should
        be trained to produce ids[i]."""
        ids, mask = [], []

        def put(seq, trained):
            ids.extend(seq)
            mask.extend([trained] * len(seq))

        for t in turns:
            role = t.get("role")
            content = t.get("content", "")
            if role not in ROLE_TOKEN:
                raise ValueError(f"unknown role {role!r}")
            head = self.id[ROLE_TOKEN[role]]
            body = self.encode_text(content)
            trained = role in TRAINED_ROLES
            # Header is always context (the host emits it to cue generation).
            put([head], False)
            # Content + closing <|end|> are trained only for assistant turns.
            put(body, trained)
            put([self.end], trained)
        return ids, mask

    def render_prompt(self, turns):
        """Inference-side: render context and leave a dangling <|assistant|> so the
        model generates the reply. Mirrors render() exactly — if they drift, the
        model sees a different format at test time than it trained on."""
        ids, _ = self.render(turns)
        return ids + [self.id["<|assistant|>"]]


def load_conversations(paths):
    convos = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  ! missing {p}", file=sys.stderr)
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
            if isinstance(turns, list) and turns:
                convos.append(turns)
    return convos


def pack(convos, tmpl, ctx, report_every=0):
    """Render + concatenate + cut into ctx+1 blocks. Returns (x, y) int arrays
    where y is -1 wherever the token must not contribute to the loss."""
    ids_buf, mask_buf = [], []
    dropped = 0
    for i, turns in enumerate(convos):
        try:
            ids, mask = tmpl.render(turns)
        except ValueError:
            dropped += 1
            continue
        if not any(mask):        # nothing to learn from — skip
            dropped += 1
            continue
        # Separator resets context between conversations.
        ids_buf.extend(ids + [tmpl.eot])
        mask_buf.extend(mask + [False])
        if report_every and (i + 1) % report_every == 0:
            print(f"    rendered {i+1:,}/{len(convos):,}", flush=True)

    n_blocks = len(ids_buf) // (ctx + 1)
    if n_blocks == 0:
        raise SystemExit(f"corpus too small to fill one {ctx+1}-token block")
    usable = n_blocks * (ctx + 1)
    arr = np.asarray(ids_buf[:usable], dtype=np.int32).reshape(n_blocks, ctx + 1)
    msk = np.asarray(mask_buf[:usable], dtype=bool).reshape(n_blocks, ctx + 1)

    x = arr[:, :-1].astype(np.int32)
    # Shift: y[i] is the token predicted FROM position i, i.e. ids[i+1].
    # Keep it only if ids[i+1] is itself a trained token.
    y = np.where(msk[:, 1:], arr[:, 1:], -1).astype(np.int32)
    return x, y, dropped


# ------------------------------------------------------------------ self-test
def self_test():
    tmpl = ChatTemplate()
    I = tmpl.id
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}")
        if not good:
            print(f"          got  {got}\n          want {want}")

    # 1. Single turn: only assistant content + closing <|end|> are trained.
    ids, mask = tmpl.render([{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": "yo"}])
    u, a = tmpl.encode_text("hi"), tmpl.encode_text("yo")
    check("layout",
          ids, [I["<|user|>"]] + u + [I["<|end|>"]] + [I["<|assistant|>"]] + a + [I["<|end|>"]])
    check("mask: user turn fully masked out",
          mask[:1 + len(u) + 1], [False] * (1 + len(u) + 1))
    check("mask: <|assistant|> header masked out", mask[1 + len(u) + 1], False)
    check("mask: assistant content trained", mask[2 + len(u) + 1: 2 + len(u) + 1 + len(a)],
          [True] * len(a))
    check("mask: closing <|end|> TRAINED (or the model never stops)", mask[-1], True)

    # 2. System and host-injected memory are input, never targets.
    ids, mask = tmpl.render([
        {"role": "system", "content": "Be concise."},
        {"role": "memory", "content": "User likes tea."},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"}])
    n_sys = 1 + len(tmpl.encode_text("Be concise.")) + 1
    n_mem = 1 + len(tmpl.encode_text("User likes tea.")) + 1
    check("system turn masked out", any(mask[:n_sys]), False)
    check("host-injected memory masked out", any(mask[n_sys:n_sys + n_mem]), False)
    check("memory uses reserved id 10", ids[n_sys], 10)

    # 3. A model-emitted memory_write is inside assistant content -> trained.
    ids, mask = tmpl.render([{"role": "user", "content": "remember: I like tea"},
                             {"role": "assistant", "content": "<|memory_write|>likes tea"}])
    w = ids.index(I["<|memory_write|>"])
    check("model-emitted <|memory_write|> IS trained", mask[w], True)

    # 3b. TOOL CALLING. Same host/model asymmetry as memory, and the failure
    #     mode is silent: if <|tool_call|> is ever masked out the model simply
    #     never calls a tool, with a perfectly normal-looking loss curve.
    call = '<|tool_call|>{"name":"get_weather","arguments":{"city":"Boston"}}'
    ids, mask = tmpl.render([
        {"role": "user", "content": "weather in boston?"},
        {"role": "assistant", "content": call},
        {"role": "tool", "content": '{"temp_f":54}'},
        {"role": "assistant", "content": "54 an' rainin'."}])
    c = ids.index(I["<|tool_call|>"])
    check("model-emitted <|tool_call|> IS trained", mask[c], True)
    check("<|tool_call|> is one token id 6", I["<|tool_call|>"], 6)
    check("<|tool_result|> is one token id 7", I["<|tool_result|>"], 7)
    # The <|end|> that closes the tool-call turn is what hands control back to
    # the host. Mask it out and the model runs straight past its own tool call.
    r = ids.index(I["<|tool_result|>"])
    check("<|end|> closing the tool-call turn IS trained (hands off to host)",
          mask[r - 1], True)
    check("role 'tool' header masked out (host emits it)", mask[r], False)
    n_res = 1 + len(tmpl.encode_text('{"temp_f":54}')) + 1
    check("host-injected tool result fully masked out",
          any(mask[r:r + n_res]), False)

    # 3c. A model-emitted <|memory_read|> (asking to search) must be trained,
    #     even though the SAME token is a masked host header for role "memory".
    #     Position is what disambiguates them, so prove both directions hold.
    ids, mask = tmpl.render([
        {"role": "memory", "content": "user drinks tea"},
        {"role": "user", "content": "what do i drink?"},
        {"role": "assistant", "content": "<|memory_read|>drink preference"}])
    first = ids.index(I["<|memory_read|>"])
    second = ids.index(I["<|memory_read|>"], first + 1)
    check("host <|memory_read|> header masked, model-emitted one trained",
          (mask[first], mask[second]), (False, True))

    # 4. <think>/</think> must be single ids 8/9, not 3 ASCII tokens each.
    ids, _ = tmpl.render([{"role": "user", "content": "x"},
                          {"role": "assistant", "content": "<think>r</think>ans"}])
    check("<think> is one token id 8", I["<think>"], 8)
    check("</think> is one token id 9", I["</think>"], 9)
    check("think tokens present in rendered assistant turn",
          (8 in ids and 9 in ids), True)

    # 5. Consecutive user turns (~0.2% of the corpus — real double-texting)
    #    must render without assuming strict alternation.
    try:
        ids, mask = tmpl.render([{"role": "user", "content": "a"},
                                 {"role": "user", "content": "b"},
                                 {"role": "assistant", "content": "c"}])
        check("consecutive user turns render", True, True)
        check("  ...and both stay masked out",
              any(mask[:len(ids) - 1 - len(tmpl.encode_text("c")) - 1]), False)
    except Exception as e:
        check(f"consecutive user turns render ({e})", False, True)

    # 6. The shift. This is the bug that silently trains on the wrong token.
    convo = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    x, y, _ = pack([convo] * 400, tmpl, ctx=64)
    kept = y[y != -1]
    a_ids = set(tmpl.encode_text("yo")) | {I["<|end|>"]}
    stray = sorted(set(kept.tolist()) - a_ids)
    check("only assistant tokens survive the shift into y", stray, [])
    # And prove alignment: every kept y[i] equals x[i+1] within the row.
    aligned = all(y[r, i] == -1 or y[r, i] == x[r, i + 1]
                  for r in range(min(5, x.shape[0])) for i in range(x.shape[1] - 1))
    check("y[i] == x[i+1] wherever y is kept (no off-by-one)", aligned, True)

    # 7. Inference prompt must end with a dangling <|assistant|>.
    p = tmpl.render_prompt([{"role": "user", "content": "hi"}])
    check("render_prompt ends with <|assistant|>", p[-1], I["<|assistant|>"])
    check("render_prompt has no assistant body yet",
          p[:-1], tmpl.render([{"role": "user", "content": "hi"}])[0])

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


def stats(paths, ctx):
    tmpl = ChatTemplate()
    convos = load_conversations(paths)
    print(f"conversations: {len(convos):,}")
    x, y, dropped = pack(convos, tmpl, ctx, report_every=5000)
    total = x.size
    trained = int((y != -1).sum())
    print(f"  dropped (unrenderable/no-target): {dropped}")
    print(f"  packed blocks: {x.shape[0]:,} x {ctx} = {total:,} tokens")
    print(f"  trainable target tokens: {trained:,} ({100*trained/total:.1f}%)")
    print(f"  -> {total - trained:,} tokens are context the model is NOT trained to emit")
    print(f"  epochs over {x.shape[0]:,} blocks at micro-batch 8 = "
          f"{x.shape[0]//8:,} steps/epoch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--stats", nargs="*", default=None)
    ap.add_argument("--ctx", type=int, default=1024)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if a.stats is not None:
        paths = a.stats or ["data/synthetic/sprocket_sft.jsonl",
                            "data/synthetic/sprocket_instruct.jsonl"]
        stats(paths, a.ctx)
    else:
        ap.print_help()
