"""What drives training cost: GPU-HOURS (compute), not VRAM. Show it for a 500M model
at 10B vs 50B tokens across GPUs of different VRAM/speed/price.

cost = time * $/hr ;  time = total_FLOPs / effective_throughput ;  total_FLOPs = 6 * params * tokens
VRAM only decides whether the model FITS — a 500M model needs the same ~15-25GB whether you
feed it 10B or 50B tokens.
"""
N = 500e6  # 500M model — same VRAM footprint regardless of token count

# (name, VRAM GB, effective training TFLOPS, spot/community $/hr)
GPUS = [
    ("RTX 4090", 24, 65e12, 0.40),
    ("L40S",     48, 75e12, 0.95),
    ("H100",     80, 380e12, 2.30),
]

for tok_b in (10, 50):
    flops = 6 * N * tok_b * 1e9
    print(f"\n=== 500M model, {tok_b}B tokens  (total {flops:.1e} FLOPs) ===")
    print(f"{'GPU':10}{'VRAM':>6}{'fits 500M?':>12}{'time':>10}{'cost':>8}")
    for name, vram, eff, hr in GPUS:
        secs = flops / eff
        cost = secs / 3600 * hr
        print(f"{name:10}{vram:>4}GB{'yes (~20GB)':>12}{secs/86400:>8.1f}d ${cost:>6.0f}")

print("\nRead the columns:")
print("- DOWN a column (same tokens): VRAM rises 24->48->80GB but total $ barely changes")
print("  -> more VRAM ~= free for total cost (you pay for SPEED per hour, not memory).")
print("- ACROSS a row (same GPU): 10B -> 50B is ~5x the cost -> that's GPU-HOURS, from tokens.")
print("- So 50B costs more because it's ~5x the compute-TIME, NOT more VRAM.")
