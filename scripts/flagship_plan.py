"""flagship_plan.py — LOCAL vs CLOUD decision math for the ~500M flagship (Sprocket).

Everything here is computed from the REAL model shapes in src/model/model.py and
calibrated against the MEASURED prototype run (76.4M @ 24,694 tok/s on the RTX 4060),
so the throughput numbers are anchored to this machine, not vendor marketing.

  params      : exact formula matching model.py (SwiGLU 8/3 -> multiple_of 256, GQA, tied head)
  VRAM        : fp32 master weights + fp32 grads + AdamW m/v (the real AdamW cost)
  time/cost   : FLOPs = 6*N*D, tokens/s = MFU * peak_bf16 / (6*N)
  data        : how much web corpus must actually be downloaded + tokenized

Usage:  py scripts/flagship_plan.py
"""

VOCAB = 32000

# ---------------------------------------------------------------- param math
def ffn_hidden(dim, multiple_of=256, mult=None):
    h = int(mult * dim) if mult else int(8 / 3 * dim)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


def params(dim, n_layers, n_heads, n_kv_heads=None, vocab=VOCAB):
    """Exact param count matching model.py (weight-tied head => embedding counted once)."""
    head_dim = dim // n_heads
    n_kv = n_kv_heads or n_heads
    emb = vocab * dim
    attn = dim * dim + 2 * (dim * n_kv * head_dim) + dim * dim   # wq, wk, wv, wo
    hid = ffn_hidden(dim)
    ffn = 3 * dim * hid                                          # w1, w3, w2
    norms = 2 * dim
    per_layer = attn + ffn + norms
    return emb + n_layers * per_layer + dim, hid                 # + final norm


# ---------------------------------------------------------------- candidates
CANDIDATES = [
    ("500m-a", 1280, 26, 20, 4),
    ("500m-b", 1280, 24, 20, 5),
    ("500m-c", 1536, 18, 12, 4),
    ("500m-d", 1024, 40, 16, 4),
]

print("=" * 78)
print("CANDIDATE ~500M PRESETS  (exact params from model.py formula)")
print("=" * 78)
print(f"{'name':9s} {'dim':>5s} {'L':>3s} {'H':>3s} {'KV':>3s} {'ffn_hid':>8s} {'params':>12s}")
for name, dim, L, H, KV in CANDIDATES:
    n, hid = params(dim, L, H, KV)
    print(f"{name:9s} {dim:5d} {L:3d} {H:3d} {KV:3d} {hid:8d} {n/1e6:9.1f}M")

# pick the primary
DIM, LAYERS, HEADS, KV = 1280, 26, 20, 4
N, HID = params(DIM, LAYERS, HEADS, KV)
print(f"\nPRIMARY: dim={DIM} layers={LAYERS} heads={HEADS} kv={KV} -> {N/1e6:.1f}M params")
print(f"  (embedding {VOCAB*DIM/1e6:.1f}M = {100*VOCAB*DIM/N:.1f}% of params — tied head, counted once)")

# ---------------------------------------------------------------- local VRAM
print("\n" + "=" * 78)
print("LOCAL FEASIBILITY — RTX 4060 8GB (~7.0 GB usable)")
print("=" * 78)
USABLE = 7.0
for label, opt_bytes_per_param in (("AdamW fp32 (current train.py)", 16), ("8-bit Adam (bitsandbytes)", 10)):
    states = N * opt_bytes_per_param / 1e9
    print(f"  {label:32s}: {states:5.2f} GB  (params+grads+opt states, before activations)")
    verdict = "FITS (barely, tiny batch)" if states < USABLE * 0.75 else "DOES NOT FIT"
    print(f"  {'':32s}  -> {verdict}")

meas_params, meas_tps = 76.4e6, 24694
meas_flops = 6 * meas_params * meas_tps
print(f"\n  MEASURED anchor: 76.4M @ {meas_tps:,} tok/s = {meas_flops/1e12:.1f} TFLOPS effective on the 4060")
proj_tps = meas_flops / (6 * N)
print(f"  Projected 500M throughput at the SAME effective TFLOPS: ~{proj_tps:,.0f} tok/s")
for D in (5e9, 10e9, 15e9, 20e9):
    days = D / proj_tps / 86400
    print(f"    {D/1e9:4.0f}B tokens -> {days:7.1f} days of continuous local training")

# ---------------------------------------------------------------- measured 2026-07-27
print("""
  !! MEASURED ON THE ACTUAL 4060 (2026-07-27) — the projection above is FANTASY
     for 500m. It assumes the model reaches the same TFLOPS as the prototype.
     It does not. Every 500m config paged:

       config                          peak GB    tok/s   effective
       proto-75m ctx1024 mb=1             1.93   16,522   7.6 TFLOPS
       proto-75m ctx1024 mb=4             4.37   24,910  11.4 TFLOPS  <- best
       proto-75m ctx1024 mb=8             7.57   19,563   9.0 TFLOPS  (degrading)
       500m      ctx512  mb=1             8.37      795   2.4 TFLOPS  (paging)
       500m      ctx1024 mb=1             9.50      182   0.5 TFLOPS  (paging)
       500m      ctx2048 mb=1 +gradckpt   8.47       64   0.2 TFLOPS  (paging)

     500m + AdamW needs 8.24 GB of weights+grads+moments alone, versus ~7.45 GB
     free. Gradient checkpointing does NOT rescue it — that trades activation
     memory, and the optimizer state is what overflows.

  !! WINDOWS DOES NOT RAISE OutOfMemoryError. The WDDM driver silently pages VRAM
     to system RAM, so an over-budget job LOOKS LIKE A HANG rather than failing.
     A local run that seems frozen is usually this. Judge fit by tok/s, never by
     "it started". (Cost us an hour of a run stuck at step 0.)

  !! Anything else on the GPU counts. A browser + Steam + a running game held
     ~3.5-4.9 GB, leaving under 4 GB for training. Close them, or expect paging.

  => LOCAL 500m TRAINING AND SMOKE-TESTING ARE BOTH OFF THE TABLE.
     Validate the 500m forward/backward WITHOUT an optimizer (4.29 GB, fits) to
     check the architecture, use proto-75m at mb=4 for pipeline work, and do the
     real 500m smoke test on a rented 4090 (~$0.44/hr, a few minutes ~ $1).

  => FOR THE CLOUD RUN: efficiency rose from 7.6 -> 11.4 TFLOPS going mb=1 -> mb=4,
     i.e. this model is launch-overhead-bound at small batch. On an 80 GB H100 use a
     LARGE micro-batch (start ~32-64 at ctx 2048) — the 35%% MFU assumed below is
     conservative and a big batch is the cheapest way to beat it.""")

# ---------------------------------------------------------------- cloud
print("\n" + "=" * 78)
print("CLOUD — time & cost for the 500M flagship  (FLOPs = 6*N*D)")
print("=" * 78)
# (name, dense bf16 TFLOPS, assumed MFU for a ~500M model, $/hr)
# Prices are the rates actually quoted for this run (2026-07-26). MFU is the honest
# unknown here — 0.35 is a conservative real-world figure for a model this small
# (kernel-launch and memory overhead dominate more than at 7B+). Re-measure on the
# smoke run and edit these numbers rather than trusting them.
#
# H200 note: SAME GH100 compute die as H100 (989 TFLOPS bf16 dense). Its advantage
# is 141GB HBM3e @ 4.8TB/s vs 80GB @ 3.35TB/s — which buys ~nothing for a 501M model
# that fits in ~8GB and is compute-bound. It is NOT a 1.5x faster H100.
GPUS = [
    ("RTX 4090 24GB",    165.2, 0.35, 0.44),   # smoke-test reference only
    ("RTX PRO 6000 96GB", 503.0, 0.35, 1.99),
    ("H100 80GB SXM",    989.0, 0.35, 2.99),
    ("H200 SXM 141GB",   989.0, 0.37, 4.39),   # +2% MFU from bandwidth, at best
]
TOKEN_BUDGETS = [10e9, 20e9, 50e9, 100e9]

print(f"{'GPU':17s} {'eff TFLOPS':>11s} {'tok/s':>10s} " + "".join(f"{int(d/1e9):>6d}B" for d in TOKEN_BUDGETS))
print("-" * 78)
rows = {}
for name, peak, mfu, price in GPUS:
    eff = peak * mfu
    tps = eff * 1e12 / (6 * N)
    cells, costs = [], []
    for D in TOKEN_BUDGETS:
        hrs = D / tps / 3600
        cells.append(f"{hrs:6.0f}h")
        costs.append(hrs * price)
    rows[name] = (costs, price)
    print(f"{name:17s} {eff:8.0f}    {tps:10,.0f} " + "".join(cells))

print(f"\n{'GPU':19s} {'$/hr':>6s} " + "".join(f"{int(d/1e9):>8d}B" for d in TOKEN_BUDGETS)
      + f" {'$/B tok':>9s}   <- COST (USD)")
print("-" * 78)
best = min(rows.items(), key=lambda kv: kv[1][0][0] / (TOKEN_BUDGETS[0] / 1e9))
for name, (costs, price) in rows.items():
    per_b = costs[0] / (TOKEN_BUDGETS[0] / 1e9)
    flag = "  <- best $/token" if name == best[0] else ""
    print(f"{name:19s} {price:6.2f} " + "".join(f"${c:7.0f}" for c in costs)
          + f" {per_b:8.2f} " + flag)

print("\n  $/B-token is the number that matters — it is price/hr divided by throughput,")
print("  so a cheaper card that is proportionally slower is NOT cheaper per run.")

# ---------------------------------------------------------------- chinchilla
print("\n" + "=" * 78)
print("CHINCHILLA / DATA CONTEXT")
print("=" * 78)
print(f"  Chinchilla-optimal for {N/1e6:.0f}M params = 20 tok/param = {20*N/1e9:.1f}B tokens")
for D in TOKEN_BUDGETS:
    print(f"    {D/1e9:4.0f}B = {D/N:5.1f} tok/param  ({'under' if D < 20*N else 'OVER'}-trained vs Chinchilla)"
          f"   {'<- modern small-model practice (over-train for inference efficiency)' if D >= 20*N else ''}")

print("\n  NOTE: Chinchilla is COMPUTE-optimal, not QUALITY-optimal. Modern small models")
print("  deliberately over-train (SmolLM2-1.7B ~11T, Qwen2.5-1.5B ~18T) because inference")
print("  cost dominates for a model meant to run on a laptop/phone.")

# ---------------------------------------------------------------- data reqs
print("\n" + "=" * 78)
print("DATA REQUIREMENTS  (what must actually be downloaded + tokenized)")
print("=" * 78)
HAVE_WEB = 175_283_970
SYNTH_PRE = 4_476_293
CHARS_PER_TOK = 4.52
TOKENIZE_TPS = 1.5e6   # HF tokenizers, multi-core batch encode
print(f"  Already tokenized (FineWeb-Edu): {HAVE_WEB/1e6:.1f}M tokens  (data/processed/train.bin, 350 MB)")
print(f"  Synthetic pretrain banked       : {SYNTH_PRE/1e6:.2f}M tokens  (NOT yet tokenized into a .bin)")
print()
for D in TOKEN_BUDGETS:
    need = D - HAVE_WEB
    gb_bin = D * 2 / 1e9
    gb_txt = need * CHARS_PER_TOK / 1e9
    hrs_tok = need / TOKENIZE_TPS / 3600
    print(f"  {D/1e9:4.0f}B target: need +{need/1e9:5.2f}B more web tokens "
          f"| raw text ~{gb_txt:5.0f} GB | train.bin {gb_bin:5.1f} GB | tokenize ~{hrs_tok:4.1f} h")

print("\n  SYNTHETIC UPWEIGHT (the <=4x rule applies to the CURATED SYNTHETIC layer, not the web bulk):")
for mult in (1, 2, 3, 4):
    eff = SYNTH_PRE * mult
    print(f"    {mult}x -> {eff/1e6:6.2f}M effective synthetic tokens = "
          + ", ".join(f"{100*eff/D:.2f}% of {D/1e9:.0f}B" for D in (10e9, 20e9)))
