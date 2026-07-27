"""Determine which scaled_dot_product_attention backends actually run on this box,
and measure the memory cost of attention at a realistic context length.

This decides our max local context length. Flash/mem-efficient keep attention memory
~O(T); the math fallback materializes the full T x T matrix (~O(T^2)).
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

dev = torch.device("cuda")
B, H, T, D = 8, 12, 1024, 64  # realistic-ish shapes for a small model

def try_backend(name, backend):
    q = torch.randn(B, H, T, D, device=dev, dtype=torch.bfloat16)
    try:
        torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel(backend):
            out = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {name:16s}: OK    peak {mem:.2f} GB")
        return True
    except Exception as e:
        print(f"  {name:16s}: FAIL  ({type(e).__name__})")
        return False

print("=== SDPA backends (B=8 H=12 T=1024 D=64, bf16, causal) ===")
avail = {}
avail["flash"] = try_backend("flash", SDPBackend.FLASH_ATTENTION)
avail["mem_efficient"] = try_backend("mem_efficient", SDPBackend.EFFICIENT_ATTENTION)
avail["math"] = try_backend("math", SDPBackend.MATH)
try:
    avail["cudnn"] = try_backend("cudnn", SDPBackend.CUDNN_ATTENTION)
except Exception:
    avail["cudnn"] = False

print("\n=== Default SDPA (auto-select) memory at increasing context ===")
for T2 in (512, 1024, 2048, 4096):
    q = torch.randn(4, 12, T2, 64, device=dev, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    out = F.scaled_dot_product_attention(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    print(f"  T={T2:5d}: peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

usable = [k for k, v in avail.items() if v and k != "math"]
print("\nMemory-efficient backends available:", usable if usable else "NONE (math fallback only)")
