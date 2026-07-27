#!/usr/bin/env bash
# Sprocket — one-shot cloud bootstrap. Runs on the rented pod.
#
# Corpus -> pretrain -> SFT -> HF export -> push to the Hub -> terminate the pod.
#
# IDEMPOTENT BY DESIGN. Every stage resumes. If the pod is preempted, relaunch
# the identical command and it continues where it stopped — that is the entire
# point, because spot instances WILL die mid-run.
#
# Requires a NETWORK VOLUME mounted at /workspace. Container disk is ephemeral;
# a checkpoint written there is gone the instant the pod dies, which defeats
# resume and wastes the whole run.
#
# Env (set via runpodctl --env or the pod template):
#   REPO_URL      git repo to clone                       (required)
#   HF_TOKEN      HuggingFace write token, for the push   (required to publish)
#   HF_REPO       target model repo, e.g. you/sprocket-500m
#   TOKENS        pretrain token budget      (default 50000000000)
#   SUBSET        FineWeb-Edu subset         (default sample/100BT)
#   PRESET        model preset               (default 500m)
#   CTX           context length             (default 2048)
#   MICRO_BATCH   per-step micro batch       (default 32 — big on 80GB, see below)
#   GRAD_ACCUM    gradient accumulation      (default 4)
#   MAX_ITERS     pretrain steps             (default 0 = derive from TOKENS)
#   AUTO_STOP     terminate pod when done    (default 1)
set -euo pipefail

WORK=/workspace
REPO_URL="${REPO_URL:?set REPO_URL}"
TOKENS="${TOKENS:-50000000000}"
SUBSET="${SUBSET:-sample/100BT}"
PRESET="${PRESET:-500m}"
CTX="${CTX:-2048}"
MICRO_BATCH="${MICRO_BATCH:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_ITERS="${MAX_ITERS:-0}"
AUTO_STOP="${AUTO_STOP:-1}"
HF_REPO="${HF_REPO:-}"

MAX_HOURS="${MAX_HOURS:-200}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p "$WORK"; cd "$WORK"

# ---------------------------------------------------------------- billing guard
# THREE independent ways this pod stops charging you. An unattended run must not
# depend on the happy path: a crash on line 1 would otherwise bill $2.99/hr until
# someone notices, which on a weekend is ~$150.
terminate_pod() {
  local reason="$1"
  log "TERMINATING POD — $reason"
  [ -z "${RUNPOD_POD_ID:-}" ] && { log "  no RUNPOD_POD_ID; STOP THE POD MANUALLY"; return; }
  runpodctl remove pod "$RUNPOD_POD_ID" && return 0
  # Fallback if runpodctl is missing/broken in the image: hit the API directly.
  if [ -n "${RUNPOD_API_KEY:-}" ]; then
    log "  runpodctl failed; trying REST API"
    curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}" \
      >/dev/null && { log "  terminated via API"; return 0; }
  fi
  log "  !! COULD NOT SELF-TERMINATE — STOP THE POD MANUALLY, IT IS STILL BILLING"
}

on_exit() {
  local code=$?
  kill "${WATCHDOG_PID:-0}" 2>/dev/null || true   # don't outlive the script
  if [ "$code" -eq 0 ]; then
    log "pipeline finished cleanly"
  else
    log "!! PIPELINE FAILED (exit $code) — see the log above"
    log "!! state is on the volume; fix and relaunch to resume"
  fi
  [ "${AUTO_STOP:-1}" = "1" ] && terminate_pod "exit code $code"
}
trap on_exit EXIT

# Hard watchdog: fires even if the script wedges rather than crashing (a hung
# download, a deadlocked dataloader). Sized well above the expected run.
#
# CRITICAL: its stdio must be detached. A backgrounded job that inherits stdout
# holds the caller's pipe open, so `curl … | bash | tee train.log` would never
# see EOF and the container start command would hang forever AFTER training
# succeeded. Redirect to a file and disown.
WATCHDOG_LOG="$WORK/watchdog.log"
( sleep $(( MAX_HOURS * 3600 ))
  echo "[watchdog] MAX_HOURS=$MAX_HOURS exceeded"
  terminate_pod "watchdog timeout after ${MAX_HOURS}h"
) </dev/null >>"$WATCHDOG_LOG" 2>&1 &
WATCHDOG_PID=$!
disown "$WATCHDOG_PID" 2>/dev/null || true
log "billing guard armed: AUTO_STOP=${AUTO_STOP}, watchdog ${MAX_HOURS}h (pid $WATCHDOG_PID)"

if ! mountpoint -q "$WORK" 2>/dev/null && [ ! -d "$WORK/.persist" ]; then
  log "WARNING: $WORK may not be a network volume. If this pod is preempted,"
  log "         checkpoints will be LOST. Attach a network volume."
fi
mkdir -p "$WORK/.persist"

# ---------------------------------------------------------------- 1. code
if [ ! -d "$WORK/sprocket/.git" ]; then
  log "cloning $REPO_URL"
  git clone --depth 1 "$REPO_URL" "$WORK/sprocket"
else
  log "repo present; pulling"
  git -C "$WORK/sprocket" pull --ff-only || log "pull failed, continuing with local copy"
fi
cd "$WORK/sprocket"

log "installing deps"
pip install -q --no-input -r requirements.txt

python - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"  VRAM {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
PY

# Keep all mutable state on the volume, not the container disk.
mkdir -p "$WORK/data/processed" "$WORK/checkpoints"
ln -sfn "$WORK/data/processed" data/processed
ln -sfn "$WORK/checkpoints" checkpoints

# ---------------------------------------------------------------- 2a. PREFLIGHT
# Exercise EVERY dependency the run will eventually need, before spending a
# single GPU-hour. The first smoke run pretrained and SFT'd for 16 minutes and
# THEN died on a transformers/torch mismatch in the export step. On the real run
# that same failure lands on day 5. Anything env-shaped must fail here, in ~60s.
log "preflight: validating the whole toolchain before burning GPU time"
python - <<'PY'
import os, sys, tempfile
fail = []

def chk(name, fn):
    try:
        fn(); print(f"    ok   {name}")
    except Exception as e:
        print(f"    FAIL {name}: {type(e).__name__}: {e}")
        fail.append(name)

import torch
chk("cuda available", lambda: (_ for _ in ()).throw(RuntimeError("no CUDA"))
    if not torch.cuda.is_available() else None)

# The exact import that killed the first run.
def _tf():
    import transformers
    from transformers import LlamaConfig, LlamaForCausalLM  # noqa: F401
    print(f"         transformers {transformers.__version__}, torch {torch.__version__}")
chk("transformers LlamaForCausalLM import", _tf)

def _tok():
    from tokenizers import Tokenizer
    t = Tokenizer.from_file("config/tokenizer/tokenizer.json")
    assert t.get_vocab_size() == 32000, f"vocab {t.get_vocab_size()} != 32000"
    for s, i in [("<think>", 8), ("</think>", 9),
                 ("<|memory_read|>", 10), ("<|memory_write|>", 11)]:
        got = t.encode(s, add_special_tokens=False).ids
        assert got == [i], f"{s} -> {got}, expected [{i}]"
chk("tokenizer special ids (8/9/10/11, vocab 32000)", _tok)

def _sft():
    sys.path.insert(0, "src")
    from train.sft_data import ChatTemplate
    ChatTemplate("config/tokenizer/tokenizer.json").render(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
chk("SFT chat template renders", _sft)

def _corpus():
    assert os.path.exists("data/synthetic/sprocket_sft.jsonl"), "SFT corpus missing from repo"
    n = sum(1 for _ in open("data/synthetic/sprocket_sft.jsonl", encoding="utf-8"))
    assert n > 10000, f"only {n} SFT rows"
    print(f"         {n:,} SFT conversations present")
chk("SFT corpus shipped in the repo", _corpus)

# Writability of the eventual destination — a bad token should not surface on day 5.
if os.environ.get("HF_TOKEN") and os.environ.get("HF_REPO"):
    def _hf():
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(os.environ["HF_REPO"], repo_type="model",
                        private=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("preflight"); p = f.name
        api.upload_file(path_or_fileobj=p, path_in_repo=".preflight",
                        repo_id=os.environ["HF_REPO"], repo_type="model")
        os.unlink(p)
    chk("HuggingFace push (write access proven)", _hf)
else:
    print("    skip HuggingFace push check (HF_TOKEN/HF_REPO unset)")

if fail:
    print(f"\nPREFLIGHT FAILED: {fail}")
    sys.exit(1)
print("\n  preflight OK — safe to spend GPU time")
PY

# ---------------------------------------------------------------- 2. corpus
log "building corpus: target $TOKENS tokens from $SUBSET (resumable)"
python scripts/build_corpus.py \
  --target-tokens "$TOKENS" --subset "$SUBSET" \
  --out data/processed/train.bin --scratch "$WORK/_shards"

# ---------------------------------------------------------------- 3. pretrain
if [ "$MAX_ITERS" = "0" ]; then
  MAX_ITERS=$(python - <<PY
print(int($TOKENS / ($MICRO_BATCH * $GRAD_ACCUM * $CTX)))
PY
)
fi
log "pretraining $PRESET for $MAX_ITERS steps (ctx $CTX, mb $MICRO_BATCH x$GRAD_ACCUM)"
python -m src.train.train \
  --preset "$PRESET" --data data/processed/train.bin --loader memmap \
  --ctx "$CTX" --micro-batch "$MICRO_BATCH" --grad-accum "$GRAD_ACCUM" \
  --max-iters "$MAX_ITERS" --warmup $(( MAX_ITERS / 100 + 1 )) \
  --ckpt-minutes 10 --resume auto

# ---------------------------------------------------------------- 4. SFT
BASE="checkpoints/${PRESET}_final.pt"
[ -f "$BASE" ] || BASE="checkpoints/${PRESET}_latest.pt"
log "SFT from $BASE"
python -m src.train.sft \
  --base "$BASE" --preset "$PRESET" --ctx "$CTX" \
  --micro-batch 16 --grad-accum 2 --epochs 3 \
  --ckpt-minutes 10 --resume auto

# ---------------------------------------------------------------- 5. export
log "exporting to HuggingFace format (with logit-parity check)"
python scripts/export_hf.py "checkpoints/${PRESET}_sft_final.pt" "$WORK/hf-${PRESET}" \
  --dtype bfloat16

# ---------------------------------------------------------------- 6. publish
if [ -n "${HF_TOKEN:-}" ] && [ -n "$HF_REPO" ]; then
  log "pushing to https://huggingface.co/$HF_REPO"
  python - <<PY
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("$HF_REPO", repo_type="model", exist_ok=True)
api.upload_folder(folder_path="$WORK/hf-$PRESET", repo_id="$HF_REPO", repo_type="model")
print("  pushed -> https://huggingface.co/$HF_REPO")
PY
else
  log "HF_TOKEN or HF_REPO unset — skipping publish. Model is at $WORK/hf-$PRESET"
  log "  retrieve it with:  runpodctl send $WORK/hf-$PRESET"
fi

log "DONE."
log "  base ckpt : $WORK/checkpoints/${PRESET}_final.pt"
log "  sft ckpt  : $WORK/checkpoints/${PRESET}_sft_final.pt"
log "  hf model  : $WORK/hf-${PRESET}"

# Termination is handled by the EXIT trap above — it fires on success AND on
# failure, so there is no path where the script ends and the pod keeps billing.
