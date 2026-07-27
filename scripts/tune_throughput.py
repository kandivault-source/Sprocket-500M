"""tune_throughput.py — find the fastest training config on THIS GPU, empirically.

The first smoke run measured 63,470 tok/s for the 500m on an H100 at mb=24/ctx=1024
— about 19% MFU, which is low. Rather than guess, sweep the knobs that plausibly
matter and print what actually wins. At $2.99/hr this pays for itself immediately:
a 1.5x speedup on a 50B-token run is worth ~$190.

Knobs swept:
  micro-batch     small models are launch-overhead bound; bigger batches amortise
  context length  longer ctx costs O(n^2) attention but improves matmul shapes
  torch.compile   fuses elementwise chains; usually the single biggest win, and
                  cannot be tested on Windows (inductor needs a C compiler)

  py scripts/tune_throughput.py --preset 500m --minutes 10
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from model.model import GPT, PRESETS  # noqa: E402
import model.model as M  # noqa: E402


def bench(preset, ctx, mb, compile_mode, steps, vocab=32000):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    label = f"mb={mb:<3d} ctx={ctx:<5d} " + (f"compile={compile_mode}" if compile_mode else "eager       ")
    try:
        cfg = replace(PRESETS[preset], max_seq_len=ctx)
        m = GPT(cfg).cuda()
        m.train()
        opt = m.configure_optimizers(0.1, 1e-4, device_type="cuda")
        run = torch.compile(m, mode=compile_mode) if compile_mode else m
        x = torch.randint(0, vocab, (mb, ctx), device="cuda")
        y = x.clone()

        t_compile = time.time()
        for _ in range(3):                      # warmup; compilation happens here
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, l = run(x, y)
            l.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        warm = time.time() - t_compile

        t0 = time.time()
        for _ in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, l = run(x, y)
            l.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0

        tps = steps * mb * ctx / dt
        peak = torch.cuda.max_memory_allocated() / 1e9
        tflops = 6 * m.num_params() * tps / 1e12
        res = {"mb": mb, "ctx": ctx, "compile": compile_mode, "tok_s": tps,
               "peak_gb": peak, "tflops": tflops, "warmup_s": warm}
        print(f"  {label}  {tps:9,.0f} tok/s  {peak:5.1f} GB  {tflops:6.1f} TFLOPS"
              f"  (warmup {warm:.0f}s)", flush=True)
        del m, opt, x, y, l, run
        torch.cuda.empty_cache()
        return res
    except torch.cuda.OutOfMemoryError:
        print(f"  {label}  OOM", flush=True)
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"  {label}  FAILED: {type(e).__name__}: {str(e)[:60]}", flush=True)
        torch.cuda.empty_cache()
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="500m")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default="/workspace/tune_results.json")
    ap.add_argument("--upload", action="store_true", help="push results to HF_REPO")
    a = ap.parse_args()

    free, total = torch.cuda.mem_get_info()
    name = torch.cuda.get_device_name(0)
    print("=" * 78)
    print(f"THROUGHPUT TUNING  {a.preset}  on  {name}")
    print(f"  torch {torch.__version__} | {total/1e9:.0f} GB | SDPA enable_gqa={M._SDPA_HAS_GQA}")
    print("=" * 78)

    # compile already proven ~1.7x, so sweep COMPILED configs only and settle both
    # context lengths in one pass. Anything left unmeasured here becomes a guess
    # baked into a multi-hundred-dollar run, so cover the whole decision space.
    results = []
    for ctx in (1024, 2048):
        print(f"\n--- ctx {ctx}: micro-batch scaling (compiled) ---")
        oom = False
        for mb in (8, 12, 16, 24, 32, 48, 64):
            if oom:
                break
            r = bench(a.preset, ctx, mb, "default", a.steps)
            if r:
                results.append(r)
            else:
                oom = True   # bigger will only be worse

    if not results:
        sys.exit("every config failed")

    # max-autotune is slow to compile; only worth it on the winner.
    top = max(results, key=lambda r: r["tok_s"])
    print(f"\n--- max-autotune on the winner (mb={top['mb']} ctx={top['ctx']}) ---")
    r = bench(a.preset, top["ctx"], top["mb"], "max-autotune", a.steps)
    if r:
        results.append(r)

    best = max(results, key=lambda r: r["tok_s"])
    peak_tflops = {"H100": 989, "H200": 989, "A100": 312, "4090": 165}
    ref = next((v for k, v in peak_tflops.items() if k in name), None)

    print("\n" + "=" * 78)
    print("BEST CONFIG")
    print("=" * 78)
    print(f"  micro-batch {best['mb']}, ctx {best['ctx']}, "
          f"compile={best['compile'] or 'off'}")
    print(f"  {best['tok_s']:,.0f} tok/s   {best['peak_gb']:.1f} GB   "
          f"{best['tflops']:.1f} TFLOPS effective"
          + (f"  ({100*best['tflops']/ref:.0f}% MFU)" if ref else ""))
    base = next((r for r in results if r["mb"] == 24 and r["ctx"] == 1024
                 and r["compile"] is None), None)
    if base:
        print(f"  vs the smoke-run config (mb24/ctx1024/eager): "
              f"{best['tok_s']/base['tok_s']:.2f}x")
    for D, lbl in ((20e9, "20B"), (50e9, "50B")):
        h = D / best["tok_s"] / 3600
        print(f"    {lbl:>4s}: {h:6.1f} h  =  ${h*2.99:7.0f} at $2.99/hr")

    json.dump({"gpu": name, "torch": torch.__version__, "results": results,
               "best": best}, open(a.out, "w"), indent=2)
    print(f"\n  wrote {a.out}")

    if a.upload and os.environ.get("HF_TOKEN") and os.environ.get("HF_REPO"):
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=a.out, path_in_repo="debug/tune_results.json",
            repo_id=os.environ["HF_REPO"], repo_type="model")
        print("  uploaded to HF")


if __name__ == "__main__":
    main()
