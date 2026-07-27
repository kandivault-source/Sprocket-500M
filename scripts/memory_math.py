"""Exact memory math for two decisions: (1) how big a model can train on the 4060,
and (2) what long context actually costs. No hand-waving.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model.model import GPT, PRESETS  # noqa

VRAM = 8.59
SAFE = 7.0  # usable after desktop

print("=" * 70)
print("TRAINING memory per model (fp32 AdamW = ~16 bytes/param for params+grad+opt)")
print("=" * 70)
print(f"{'model':11} {'params':>9} {'weights+opt':>12} {'+~1.5GB act':>12}  local?")
for name, cfg in PRESETS.items():
    m = GPT(cfg); n = m.num_params(); del m
    opt_gb = 16 * n / 1e9
    total = opt_gb + 1.5
    ok = "YES" if total <= SAFE else ("TIGHT" if total <= SAFE + 2 else "NO - cloud only")
    print(f"{name:11} {n/1e6:7.0f}M {opt_gb:10.1f}GB {total:10.1f}GB  {ok}")

ceiling = (SAFE - 1.5) * 1e9 / 16
print(f"\n-> Local training ceiling (fp32 AdamW): ~{ceiling/1e6:.0f}M params")
print("   8-bit optimizer (bitsandbytes) would push this higher but is painful on Windows.")
print("   The 1B model MUST train in the cloud — optimizer states alone exceed 8 GB.")

print("\n" + "=" * 70)
print("LONG-CONTEXT cost — KV cache at inference for the 1B config (GQA: 8 kv-heads)")
print("=" * 70)
cfg = PRESETS["large-1b"]
kv_bytes_per_tok = 2 * cfg.n_layers * (cfg.n_kv_heads or cfg.n_heads) * cfg.head_dim * 2  # k+v, bf16
print(f"1B config: {cfg.n_layers} layers, {cfg.n_kv_heads} kv-heads, head_dim {cfg.head_dim}")
print(f"KV cache = {kv_bytes_per_tok/1024:.0f} KB per token\n")
print(f"{'context':>10} {'KV cache':>12} {'fits 4060 (8GB)?':>18} {'fits A100 (80GB)?':>18}")
for T in (8_192, 32_768, 128_000, 256_000, 700_000, 1_000_000):
    gb = kv_bytes_per_tok * T / 1e9
    print(f"{T:>10,} {gb:>10.1f}GB {('yes' if gb<7 else 'NO'):>18} {('yes' if gb<78 else 'NO'):>18}")

print("\nAttention COMPUTE scales O(T^2). Relative to an 8k-token forward pass:")
for T in (8_192, 32_768, 128_000, 256_000, 1_000_000):
    print(f"  {T:>10,} tokens: {(T/8192)**2:>10,.0f}x the attention FLOPs")

print("\n-> Sliding-window attention (e.g. window=4k) caps the KV cache at the window")
print("   size REGARDLESS of total context, and cuts attention compute to O(T*W).")
print("   That is the real lever for long context on limited memory.")
