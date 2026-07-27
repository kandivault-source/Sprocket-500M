"""Analyze the trained tokenizer and export stats for the dashboard panel."""
import json
import os
from collections import Counter

from tokenizers import Tokenizer

TOK = "config/tokenizer/tokenizer.json"
CORPUS = "data/raw/fineweb_000.txt"
OUT = "dashboard/data/tokenizer.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

tok = Tokenizer.from_file(TOK)
vocab = tok.get_vocab()
vocab_size = tok.get_vocab_size()

# readable length of every token id (byte-level pieces decoded to text)
piece_len = {i: len(tok.decode([i])) for i in vocab.values()}

# real tokenizations to display
examples = [
    "The cat sat on the mat.",
    "Photosynthesis converts sunlight into chemical energy.",
    "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "The witch cackled: 'Foolish mortal, your code has bugs.'",
    "antidisestablishmentarianism",
]
ex_out = []
for s in examples:
    enc = tok.encode(s)
    ex_out.append({
        "text": s,
        "pieces": [tok.decode([i]) for i in enc.ids],
        "ids": enc.ids,
        "n": len(enc.ids),
        "chars": len(s),
    })

specials = []
for name in ["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>",
             "<|end|>", "<|tool_call|>", "<|tool_result|>"]:
    tid = tok.token_to_id(name)
    if tid is not None:
        specials.append({"tok": name, "id": tid})

# compression + token-length histogram over a corpus sample
with open(CORPUS, encoding="utf-8") as f:
    sample = f.read(2_000_000)
enc = tok.encode(sample)
compression = len(sample) / len(enc.ids)
hist_c = Counter(piece_len[i] for i in enc.ids)
hist = [{"len": k, "count": hist_c[k]} for k in sorted(hist_c) if k <= 15]

# longest learned tokens (whole words the model merged into one unit)
readable = [(tok.decode([i]).strip(), i) for i in vocab.values()]
longest = [{"tok": r, "id": i} for r, i in
           sorted(readable, key=lambda x: len(x[0]), reverse=True) if r.isalpha()][:15]

out = {
    "vocab_size": vocab_size,
    "specials": specials,
    "examples": ex_out,
    "compression": round(compression, 2),
    "hist": hist,
    "longest": longest,
    "corpus_docs": 168_000,
    "corpus_tokens": 200_220_599,
}
json.dump(out, open(OUT, "w"), indent=2)
print(f"vocab_size={vocab_size}  compression={compression:.2f} chars/token")
print(f"wrote {OUT}")
