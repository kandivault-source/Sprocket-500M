"""Scaling math, comparing the current RTX 4060 to an RTX 5090 upgrade.

Anchor: measured 76M @ 24,700 tok/s on the 4060 => 4060 effective ~11.3 TFLOPS.
5090 ~ 7x the 4060's compute (21760 vs 3072 cores, ~1.8TB/s vs 272GB/s bw) => ~80 TFLOPS eff.
5090 has 32GB VRAM (vs 8GB) => local training ceiling jumps ~340M -> ~1.75B (16 bytes/param).
Cloud reference: 1x H100 ~350 TFLOPS eff, ~$2.5/hr.
(CPU 14900KS barely matters for training — it's GPU-bound; it helps data prep + CPU-offload.)
"""
GPUS = [                       # (name, effective flops/s, local training ceiling params)
    ("4060 8GB", 11.3e12, 340e6),
    ("5090 32GB", 80e12, 1.75e9),
]
H100_EFF, H100_HR = 350e12, 2.5
VOCAB = 32000
SIZES = [("76M", 76e6, 640), ("200M", 200e6, 1024), ("300M", 300e6, 1024),
         ("500M", 500e6, 1280), ("1B", 1.1e9, 2048), ("1.5B", 1.5e9, 2048), ("3B", 3.0e9, 2560)]

def dur(sec):
    d = sec / 86400
    return f"{d:.1f}d" if d >= 1 else f"{sec/3600:.1f}h"

def local(N, tokens, eff, ceil):
    return dur(6 * N * tokens / eff) if N <= ceil else "cloud"

print(f"{'size':6}{'tokens':>8}", end="")
for name, _, _ in GPUS:
    print(f"{name:>12}", end="")
print(f"{'cloud H100':>12}{'cloud $':>9}")
print("-" * 66)
for name, N, dim in SIZES:
    tokens = 20 * N
    row = f"{name:6}{tokens/1e9:6.0f}B "
    for _, eff, ceil in GPUS:
        row += f"{local(N, tokens, eff, ceil):>12}"
    cloud_s = 6 * N * tokens / H100_EFF
    row += f"{dur(cloud_s):>12}{'$'+format(cloud_s/3600*H100_HR,'.0f'):>9}"
    print(row)

print("\nKey shifts with a 5090:")
print("- Local training ceiling ~340M -> ~1.75B: the 1B/1.5B 'real model' becomes LOCAL.")
print("- ~7x faster: 300M drops from 11 days -> ~1.6 days; 1B ~3 weeks local (vs impossible on 4060).")
print("- 32GB also allows bigger batches + longer training context locally.")
print("- tokens = Chinchilla (20x); a punch-above-weight small model wants 2-4x that (x the time).")
