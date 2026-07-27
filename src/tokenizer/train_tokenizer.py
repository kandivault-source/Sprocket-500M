"""Train our own byte-level BPE tokenizer from scratch.

Byte-level BPE (same family GPT-2 / Llama use): every byte is in the base alphabet, so
there is no unknown token and any text — code, emoji, other languages — round-trips.

We reserve the chat / tool / persona special tokens NOW, at fixed low ids. They must exist
in the vocab from the first day of pretraining, because adding tokens later means resizing
the model's embedding matrix and re-learning them. Reserved slots leave room to grow.
"""
import argparse
import glob
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# Fixed, ordered — ids 0..N. Do not reorder once a model is trained against them.
SPECIAL_TOKENS = [
    "<|endoftext|>",    # 0  document boundary / end of sequence
    "<|pad|>",          # 1  padding
    "<|system|>",       # 2  chat: system turn
    "<|user|>",         # 3  chat: user turn
    "<|assistant|>",    # 4  chat: assistant turn
    "<|end|>",          # 5  end of a chat turn
    "<|tool_call|>",    # 6  model is emitting a tool/function call
    "<|tool_result|>",  # 7  a tool result is being fed back in
] + [f"<|reserved_{i}|>" for i in range(8)]  # 8..15  future use, no re-resize needed


def train(input_glob: str, out_path: str, vocab_size: int = 32000, min_frequency: int = 2):
    files = sorted(glob.glob(input_glob))
    assert files, f"no files match {input_glob!r}"
    total_gb = sum(os.path.getsize(f) for f in files) / 1e9
    print(f"training on {len(files)} file(s), {total_gb:.2f} GB of text")

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
        show_progress=True,
    )
    tok.train(files, trainer)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tok.save(out_path)
    print(f"final vocab size: {tok.get_vocab_size()}")
    print(f"saved -> {out_path}")

    # quick round-trip sanity check
    sample = "The goblin tinkerer cackled: 'Ooh, shiny! def fix(x): return x+1  🛠️'"
    ids = tok.encode(sample).ids
    back = tok.decode(ids)
    print(f"round-trip ok: {back == sample}  ({len(ids)} tokens for {len(sample)} chars)")
    return tok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of .txt shards, e.g. data/raw/*.txt")
    ap.add_argument("--out", default="config/tokenizer/tokenizer.json")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()
    train(args.input, args.out, args.vocab, args.min_frequency)
