"""Audit where the 32k vocab slots actually go — to answer 'are we wasting slots?'"""
from tokenizers import Tokenizer

tok = Tokenizer.from_file("config/tokenizer/tokenizer.json")
V = tok.get_vocab_size()

dec = {i: tok.decode([i]) for i in range(V)}
readable = set(dec.values())

single_char = sum(1 for r in dec.values() if len(r) == 1)
space_prefixed = sum(1 for r in dec.values() if r.startswith(" "))
single_digit = sum(1 for r in dec.values() if r.strip().isdigit() and len(r.strip()) == 1)
multi_digit = sum(1 for r in dec.values() if r.strip().isdigit() and len(r.strip()) >= 2)
has_any_digit = sum(1 for r in dec.values() if any(c.isdigit() for c in r))

# "duplicate" pairs: a token 'w' AND ' w' both exist
words_no_space = {r for r in readable if r and not r.startswith(" ")}
both_forms = sum(1 for w in words_no_space if (" " + w) in readable)

# tokenization efficiency: space-attached vs. space-as-its-own-token
sentence = "the quick brown fox jumps over the lazy dog and runs away fast"
n_ours = len(tok.encode(sentence).ids)
n_if_space_separate = n_ours + sentence.count(" ")  # rough: each space would add a token

print(f"vocab size:                 {V}")
print(f"single-char (base bytes):   {single_char}")
print(f"space-prefixed tokens:      {space_prefixed}  ({space_prefixed/V*100:.0f}%)")
print(f"single-digit tokens (0-9):  {single_digit}")
print(f"multi-digit tokens (45,2020): {multi_digit}  <- reclaimable with digit-splitting")
print(f"tokens containing any digit:  {has_any_digit}")
print(f"words that exist as BOTH 'w' and ' w': {both_forms}")
print()
print(f"'{sentence}'")
print(f"  our tokenizer:            {n_ours} tokens")
print(f"  if space were its own token: ~{n_if_space_separate} tokens ({(n_if_space_separate/n_ours-1)*100:.0f}% more)")
