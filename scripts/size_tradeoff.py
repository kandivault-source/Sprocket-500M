"""At a FIXED budget, which model size wins? Bigger model = more capacity but FEWER
tokens for the same $ (starved). Smaller = more tokens but a higher capacity floor.
There's an optimal size per budget (the Chinchilla trough) — and it moves right as you spend more.

L(N,D) = E + A/N^a + B/D^b  (Chinchilla, Hoffmann 2022)
cost($) = 5.1 * (N/500e6) * (D/1e9)   (anchored to train_cost.py: 500M @ 10B = $51)
"""
import math
import json

E, A, al, B, be = 1.69, 406.4, 0.34, 410.7, 0.28

def loss(N, D):
    return E + A / N ** al + B / D ** be

def tokens_for(C, N):
    return (C / 5.1) * (500e6 / N) * 1e9

SIZES = [("100M", 100e6), ("200M", 200e6), ("350M", 350e6), ("500M", 500e6),
         ("1B", 1e9), ("2B", 2e9), ("3B", 3e9), ("5B", 5e9), ("7B", 7e9), ("13B", 13e9)]

out = {}
for C in [100, 250, 500]:
    rows, best = [], None
    for name, N in SIZES:
        D = tokens_for(C, N)
        L = loss(N, D)
        rows.append({"size": name, "N": N, "tok_b": round(D / 1e9, 1), "loss": round(L, 3)})
        if best is None or L < best["loss"]:
            best = {"size": name, "loss": round(L, 3)}
    out[C] = rows
    print(f"\n=== budget ${C}  (optimal: {best['size']} @ loss {best['loss']}) ===")
    for r in rows:
        star = "  <<<" if r["size"] == best["size"] else ""
        print(f"  {r['size']:5} {r['tok_b']:8.1f}B tok   loss {r['loss']}   ppl {math.exp(r['loss']):.1f}{star}")

print("\nJSON:", json.dumps(out))
