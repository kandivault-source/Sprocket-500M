"""Predicted quality vs training tokens for a 500M model, from the published Chinchilla
scaling law (Hoffmann et al. 2022): L(N,D) = E + A/N^a + B/D^b.
These are PREDICTIONS from real research — accurate shape + ballpark magnitude, not
measured on our exact model. Cost anchored to train_cost.py (~$5.1 per 1B tokens @ 500M).
"""
import math
import json

E, A, a, B, b = 1.69, 406.4, 0.34, 410.7, 0.28
N = 500e6
COST_PER_B = 5.1  # dollars per 1B tokens for a 500M model (cloud)

def loss(D):
    return E + A / N ** a + B / D ** b

pts = []
for Db in [2, 5, 10, 15, 20, 30, 50, 75, 100, 150]:
    D = Db * 1e9
    L = loss(D)
    pts.append({"tokens_b": Db, "loss": round(L, 3), "ppl": round(math.exp(L), 1),
                "cost": round(COST_PER_B * Db)})

print(f"{'tok':>5}{'loss':>8}{'ppl':>8}{'cost':>7}{'ppl vs 10B':>12}")
base = next(p for p in pts if p["tokens_b"] == 10)["ppl"]
for p in pts:
    gain = (1 - p["ppl"] / base) * 100
    print(f"{p['tokens_b']:>4}B{p['loss']:>8}{p['ppl']:>8}{'$'+str(p['cost']):>7}{gain:>+10.1f}%")

print("\n10B -> 50B:  ppl", next(p for p in pts if p['tokens_b']==10)['ppl'],
      "->", next(p for p in pts if p['tokens_b']==50)['ppl'],
      "(~21% lower) for 5x the cost. Sublinear = diminishing returns.")
print(json.dumps(pts))
