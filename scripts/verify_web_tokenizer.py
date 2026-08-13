"""Check the browser tokenizer against the real one, token id for token id.

docs/tokenizer.js reimplements the model's tokenizer so the explorer page can run
without a server. A reimplementation that silently disagrees would put wrong token
ids in front of every visitor, so this runs both over the same strings and exits
non-zero on any divergence.

    py scripts/verify_web_tokenizer.py

Needs a node binary. This box has no standalone node by design; it uses the
embedded one shipped with LM Studio, overridable with $NODE.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

TOKENIZER = os.path.join("config", "tokenizer", "tokenizer.json")
WEB_DATA = os.path.join("docs", "tokenizer-data.json")
WEB_JS = os.path.join("docs", "tokenizer.js")

NODE_CANDIDATES = [
    os.environ.get("NODE", ""),
    r"C:\Users\Daniel\.lmstudio\.internal\utils\node.exe",
    "node",
]

CASES = [
    "The goblin tinkered with the machine until it finally worked.",
    "antidisestablishmentarianism",
    "12,847 + 936 = 13,783",
    "def tokenize(text): return [vocab[t] for t in split(text)]",
    "<|user|>what are you?<|end|><|assistant|>",
    "<|system|>Be brief.<|end|><|user|>hi<|end|><|assistant|>hello<|end|>",
    "<think>reasoning goes here</think>",
    "<|tool_call|>{\"name\":\"search\",\"args\":{\"q\":\"weather\"}}<|tool_result|>",
    "",
    " ",
    "   leading and trailing   ",
    "\n\nnewlines\n\tand tabs\n",
    "MiXeD CaSe AnD  DoUbLe  SpAcEs",
    "Ünïcödé, naïve café, Straße",
    "emoji: 🐉🔧 and math: ∑∫≈",
    "日本語のテキストもトークン化できる",
    "a" * 200,
    "https://example.com/path?q=1&r=2#frag",
    "e.g. i.e. Dr. Smith's co-worker's 3rd-party API's",
    "1234567890 0.0001 -42 1e-9 0xFF",
    "'quoted' \"double\" `back` (paren) [brack] {brace}",
    "Trailing space ",
    " Leading space",
    "\u200bzero width\u200b",
    "line1\r\nline2",
]


def find_node() -> str:
    for cand in NODE_CANDIDATES:
        if not cand:
            continue
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except (OSError, subprocess.CalledProcessError):
            continue
    sys.exit("No node binary found. Set $NODE to one.")


JS_HARNESS = r"""
const fs = require("fs");
const path = require("path");
const { Tokenizer } = require(path.resolve(process.argv[2]));
const data = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const cases = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const tk = new Tokenizer(data);
process.stdout.write(JSON.stringify(cases.map((c) => tk.ids_of(c))));
"""


def main() -> int:
    for p in (TOKENIZER, WEB_DATA, WEB_JS):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    from tokenizers import Tokenizer as HFTokenizer

    hf = HFTokenizer.from_file(TOKENIZER)
    expected = [hf.encode(c, add_special_tokens=False).ids for c in CASES]

    node = find_node()
    with tempfile.TemporaryDirectory() as td:
        harness = os.path.join(td, "harness.js")
        cases_f = os.path.join(td, "cases.json")
        with open(harness, "w", encoding="utf-8") as f:
            f.write(JS_HARNESS)
        with open(cases_f, "w", encoding="utf-8") as f:
            json.dump(CASES, f)

        proc = subprocess.run(
            [node, harness, os.path.abspath(WEB_JS),
             os.path.abspath(WEB_DATA), cases_f],
            capture_output=True, text=True, encoding="utf-8",
        )
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        sys.exit("node harness failed")

    actual = json.loads(proc.stdout)

    failures = 0
    for case, want, got in zip(CASES, expected, actual):
        label = repr(case if len(case) <= 46 else case[:43] + "...")
        if want == got:
            print(f"  ok   {len(want):>4} tok  {label}")
        else:
            failures += 1
            print(f"  FAIL          {label}")
            print(f"       python: {want}")
            print(f"       js    : {got}")
            for i, (a, b) in enumerate(zip(want, got)):
                if a != b:
                    print(f"       first divergence at index {i}: {a} vs {b}")
                    break

    total_tokens = sum(len(w) for w in expected)
    print(f"\n  {len(CASES) - failures}/{len(CASES)} cases agree "
          f"({total_tokens:,} tokens compared)")
    if failures:
        sys.exit(f"{failures} case(s) diverged")
    print("  browser tokenizer matches the real one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
