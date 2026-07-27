"""Extract real internals from the trained model for the dashboard:
  - embed3d.json: token embeddings projected to 3D (PCA/SVD) -> a real semantic map
  - weights.json: per-component weight statistics + histograms (real numbers, not mocked)
CPU only; no GPU needed.
"""
import json
import sys
import numpy as np
import torch

sys.path.insert(0, "src")
from model.model import GPT, ModelConfig  # noqa
from tokenizers import Tokenizer  # noqa

ck = torch.load("checkpoints/proto-75m_final.pt", map_location="cpu", weights_only=False)
cfg = ModelConfig(**ck["cfg"])
model = GPT(cfg)
model.load_state_dict(ck["model"])
model.eval()
tok = Tokenizer.from_file("config/tokenizer/tokenizer.json")

# ---- 3D embedding projection ----
emb = model.tok_emb.weight.detach().numpy().astype(np.float64)
words, ids = [], []
for i in range(cfg.vocab_size):
    r = tok.decode([i]).strip()
    if r and len(r) <= 14 and r.isprintable():
        words.append(r); ids.append(i)
step = max(1, len(ids) // 1600)          # sample ~1600 across the whole vocab
ids, words = ids[::step][:1600], words[::step][:1600]

X = emb[ids]
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
P = Xc @ Vt[:3].T
P = P / np.abs(P).max()
var = [round(float(v), 1) for v in (S[:3] ** 2 / (S ** 2).sum() * 100)]


def cat(w):
    if w.isdigit():
        return "num"
    if w[0].isupper():
        return "cap"
    if w.isalpha():
        return "word"
    return "sym"


pts = [{"w": words[k], "x": round(float(P[k, 0]), 3), "y": round(float(P[k, 1]), 3),
        "z": round(float(P[k, 2]), 3), "c": cat(words[k])} for k in range(len(ids))]
json.dump({"points": pts, "variance": var}, open("dashboard/data/embed3d.json", "w"))

# ---- per-component weight stats ----
comps = []
for name, p in model.named_parameters():
    a = p.detach().numpy().ravel().astype(np.float64)
    hist, edges = np.histogram(a, bins=21)
    comps.append({
        "name": name, "shape": list(p.shape), "n": int(a.size),
        "mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
        "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
        "hist": [int(h) for h in hist], "edges": [round(float(e), 3) for e in edges],
        "sample": [round(float(x), 4) for x in a[:10]],
    })
json.dump({"total": int(sum(c["n"] for c in comps)), "components": comps},
          open("dashboard/data/weights.json", "w"))

print(f"embed3d.json: {len(pts)} points, top-3 PCs explain {sum(var):.1f}% variance")
print(f"weights.json: {len(comps)} components, {sum(c['n'] for c in comps):,} total params")
