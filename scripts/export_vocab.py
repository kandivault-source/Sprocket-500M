"""Export the full 32k vocabulary (readable form + id) for the dashboard vocab browser."""
import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file("config/tokenizer/tokenizer.json")
vocab = tok.get_vocab()  # token_string -> id

items = []
for tokstr, i in vocab.items():
    readable = tok.decode([i])
    if not readable:            # special tokens decode to empty; show their literal name
        readable = tokstr
    items.append({"id": i, "t": readable})
items.sort(key=lambda x: x["id"])

json.dump({"size": len(items), "tokens": items}, open("dashboard/data/vocab.json", "w"))
print(f"exported {len(items)} tokens -> dashboard/data/vocab.json")
