#!/usr/bin/env bash
# Sprocket - one-shot cloud bootstrap. Runs on the rented pod.
#
# Corpus -> pretrain -> SFT -> HF export -> push to the Hub -> terminate the pod.
#
# IDEMPOTENT BY DESIGN. Every stage resumes. If the pod is preempted, relaunch
# the identical command and it continues where it stopped - that is the entire
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
#   TOKENS        pretrain token budget      (default 20000000000)
#   SUBSET        FineWeb-Edu subset         (default sample/100BT)
#   PRESET        model preset               (default 500m)
#   CTX           context length             (default 2048)
#   MICRO_BATCH   per-step micro batch       (default 16 - MEASURED best safe)
#   GRAD_ACCUM    gradient accumulation      (default 4)
#   MAX_ITERS     pretrain steps             (default 0 = derive from TOKENS)
#   AUTO_STOP     terminate pod when done    (default 1)
#   GGUF          also emit a q4_k_m .gguf   (default 1) - this is what runs on a phone
set -euo pipefail

WORK=/workspace
REPO_URL="${REPO_URL:?set REPO_URL}"
TOKENS="${TOKENS:-20000000000}"
SUBSET="${SUBSET:-sample/100BT}"
PRESET="${PRESET:-500m}"
CTX="${CTX:-2048}"
MICRO_BATCH="${MICRO_BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_ITERS="${MAX_ITERS:-0}"
AUTO_STOP="${AUTO_STOP:-1}"
HF_REPO="${HF_REPO:-}"

MAX_HOURS="${MAX_HOURS:-200}"
# MEASURED on H100 80GB, 500m, torch 2.8 + compile + enable_gqa (11 configs):
#   ctx1024 mb48 130,584 tok/s  75.3GB   ctx2048 mb24 117,730 tok/s  75.3GB
#   ctx1024 mb32 127,464 tok/s  52.6GB   ctx2048 mb16 115,295 tok/s  52.6GB  <-- DEFAULT
#   ctx1024 mb24 125,689 tok/s  41.2GB   ctx2048 mb12 113,403 tok/s  41.2GB
# ctx2048 costs ~10% throughput vs ctx1024 and is worth it: a 1024-token chat
# model is barely usable. mb16 over mb24 trades 2% speed for 23GB of headroom,
# which matters across a 48h run where fragmentation can OOM a tight config.
# 20B tokens @ 115,295 tok/s = 48.2 h = $144 at $2.99/hr.
# torch.compile: MEASURED 1.70x on H100 (73,252 -> 126,226 tok/s) and it also cuts
# VRAM 62 -> 43 GB. Needs an image with torch >= 2.5 (use the 2.8 image) - leave
# COMPILE=0 only if compilation itself fails. Costs ~2 min of warmup, once.
COMPILE="${COMPILE:-1}"
COMPILE_FLAG=""
[ "$COMPILE" = "1" ] && COMPILE_FLAG="--compile"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
mkdir -p "$WORK"; cd "$WORK"

# ---------------------------------------------------------------- billing guard
# THREE independent ways this pod stops charging you. An unattended run must not
# depend on the happy path: a crash on line 1 would otherwise bill $2.99/hr until
# someone notices, which on a weekend is ~$150.
terminate_pod() {
  local reason="$1"
  log "TERMINATING POD - $reason"
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
  log "  !! COULD NOT SELF-TERMINATE - STOP THE POD MANUALLY, IT IS STILL BILLING"
}

on_exit() {
  local code=$?
  kill "${WATCHDOG_PID:-0}" 2>/dev/null || true   # don't outlive the script
  if [ "$code" -eq 0 ]; then
    log "pipeline finished cleanly"
  else
    log "!! PIPELINE FAILED (exit $code) - see the log above"
    log "!! state is on the volume; fix and relaunch to resume"
  fi
  [ "${AUTO_STOP:-1}" = "1" ] && terminate_pod "exit code $code"
}
trap on_exit EXIT

# Hard watchdog: fires even if the script wedges rather than crashing (a hung
# download, a deadlocked dataloader). Sized well above the expected run.
#
# CRITICAL: its stdio must be detached. A backgrounded job that inherits stdout
# holds the caller's pipe open, so `curl ... | bash | tee train.log` would never
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

# The corpus filename encodes the SUBSET. A smoke run on sample/10BT and a real
# run on sample/100BT must never share a file: build_corpus.py refuses to append
# a different subset onto an existing manifest, so a shared name means the real
# run dies instantly with "manifest is for subset ...". That is exactly how the
# first 20B launch was lost - 20 seconds after a clean preflight.
CORPUS="data/processed/train_$(echo "$SUBSET" | tr '/ ' '__').bin"
log "corpus file: $CORPUS"

# FRESH=1 archives previous checkpoints instead of resuming them. Needed when a
# smoke run has left <preset>_final.pt at iter 200 and <preset>_sft_latest.pt
# already past its step count - the SFT stage then reports "RESUMED at step
# 466/465" and silently does NOTHING, shipping the smoke model as the flagship.
#
# FIRES AT MOST ONCE PER VOLUME. FRESH is passed as a pod ENV VAR, and the whole
# point of this script is that you relaunch the identical command after a
# preemption. Without this guard, a spot kill 20 hours into the run would come
# back up, see FRESH=1 again, and archive the in-progress checkpoints - throwing
# away the entire run and silently restarting from zero. The marker lives on the
# volume, so it survives exactly as long as the checkpoints it protects.
FRESH_MARK="$WORK/.persist/fresh_done"
if [ "${FRESH:-0}" = "1" ] && [ ! -f "$FRESH_MARK" ]; then
  if ls "$WORK/checkpoints/${PRESET}"*.pt >/dev/null 2>&1; then
    ARCHIVE="$WORK/checkpoints/_archived_$(date -u +%Y%m%d_%H%M%S)"
    mkdir -p "$ARCHIVE"
    mv "$WORK/checkpoints/${PRESET}"*.pt "$ARCHIVE"/ 2>/dev/null || true
    log "FRESH: archived previous ${PRESET} checkpoints -> $ARCHIVE"
  fi
  rm -f "$WORK/data/processed/sft_packed.npz"
  touch "$FRESH_MARK"
  log "FRESH: cleared the packed-SFT cache; marker set, will not fire again"
elif [ "${FRESH:-0}" = "1" ]; then
  log "FRESH requested but already done on this volume - resuming normally"
fi

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

# Writability of the eventual destination - a bad token should not surface on day 5.
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
print("\n  preflight OK - safe to spend GPU time")
PY

# ---------------------------------------------------------------- 2. corpus
log "building corpus: target $TOKENS tokens from $SUBSET (resumable)"
python scripts/build_corpus.py \
  --target-tokens "$TOKENS" --subset "$SUBSET" \
  --out "$CORPUS" --scratch "$WORK/_shards"

# ---------------------------------------------------------------- 3. pretrain
if [ "$MAX_ITERS" = "0" ]; then
  MAX_ITERS=$(python - <<PY
print(int($TOKENS / ($MICRO_BATCH * $GRAD_ACCUM * $CTX)))
PY
)
fi
log "pretraining $PRESET for $MAX_ITERS steps (ctx $CTX, mb $MICRO_BATCH x$GRAD_ACCUM)"
python -m src.train.train \
  --preset "$PRESET" --data "$CORPUS" --loader memmap \
  --ctx "$CTX" --micro-batch "$MICRO_BATCH" --grad-accum "$GRAD_ACCUM" \
  --max-iters "$MAX_ITERS" --warmup $(( MAX_ITERS / 100 + 1 )) \
  --ckpt-minutes 10 --resume auto $COMPILE_FLAG

# ---------------------------------------------------------------- 4. SFT
# SFT_FRESH re-runs ONLY the SFT stage - to fit a NEW instruct corpus onto the
# base model - while keeping the expensive pretrain checkpoint untouched.
#
# Without it, re-running is a silent no-op: --resume auto finds the previous
# <preset>_sft_latest.pt already at/past its step count and reports
# "RESUMED at step 466/465" followed by "SFT complete", having trained on
# nothing. The old model then gets re-exported and pushed as if it were new.
# That is how you ship the wrong weights and never find out.
#
# Pass a TAG, not just 1 (e.g. SFT_FRESH=v2). The marker is per-tag, so a
# preemption mid-SFT resumes normally instead of restarting, but bumping the
# tag when the corpus changes again forces a clean re-fit.
if [ -n "${SFT_FRESH:-}" ] && [ "${SFT_FRESH}" != "0" ]; then
  SFT_MARK="$WORK/.persist/sft_fresh_${SFT_FRESH}"
  if [ ! -f "$SFT_MARK" ]; then
    if ls "$WORK/checkpoints/${PRESET}_sft"*.pt >/dev/null 2>&1; then
      SFT_ARCHIVE="$WORK/checkpoints/_sft_archived_$(date -u +%Y%m%d_%H%M%S)"
      mkdir -p "$SFT_ARCHIVE"
      mv "$WORK/checkpoints/${PRESET}_sft"*.pt "$SFT_ARCHIVE"/ 2>/dev/null || true
      log "SFT_FRESH=${SFT_FRESH}: archived old SFT checkpoints -> $SFT_ARCHIVE"
    fi
    # The packed cache is keyed to nothing - it would silently reuse the OLD
    # conversations even with a new jsonl on disk.
    rm -f "$WORK/data/processed/sft_packed.npz"
    touch "$SFT_MARK"
    log "SFT_FRESH=${SFT_FRESH}: dropped packed cache; base checkpoint kept"
  else
    log "SFT_FRESH=${SFT_FRESH} already applied - resuming SFT normally"
  fi
fi

BASE="checkpoints/${PRESET}_final.pt"
[ -f "$BASE" ] || BASE="checkpoints/${PRESET}_latest.pt"
log "SFT from $BASE"
python -m src.train.sft \
  --base "$BASE" --preset "$PRESET" --ctx "$CTX" \
  --micro-batch 16 --grad-accum 2 --epochs 3 \
  --ckpt-minutes 10 --resume auto $COMPILE_FLAG

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
  log "HF_TOKEN or HF_REPO unset - skipping publish. Model is at $WORK/hf-$PRESET"
  log "  retrieve it with:  runpodctl send $WORK/hf-$PRESET"
fi

# ---------------------------------------------------------------- 7. GGUF
# The .safetensors export runs under transformers. A PHONE runs GGUF. Build it on
# the pod (llama.cpp needs compiling) and do it AFTER the HF push, so a build
# failure can never cost us the model itself.
if [ "${GGUF:-1}" = "1" ]; then
  log "converting to GGUF q4_k_m (phone / Ollama / LM Studio)"
  (
    set -e
    cd "$WORK"
    [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
    cd llama.cpp
    pip install -q -r requirements/requirements-convert_hf_to_gguf.txt 2>/dev/null \
      || pip install -q sentencepiece protobuf 2>/dev/null || true
    python convert_hf_to_gguf.py "$WORK/hf-${PRESET}" \
      --outfile "$WORK/sprocket-${PRESET}-f16.gguf" --outtype f16
    if cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 \
       && cmake --build build --config Release -j --target llama-quantize >/dev/null 2>&1; then
      ./build/bin/llama-quantize "$WORK/sprocket-${PRESET}-f16.gguf" \
        "$WORK/sprocket-${PRESET}-q4_k_m.gguf" Q4_K_M
    else
      log "  quantize build failed; shipping f16 gguf only"
    fi
    ls -lh "$WORK"/sprocket-*.gguf
    if [ -n "${HF_TOKEN:-}" ] && [ -n "$HF_REPO" ]; then
      python - <<PY
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in sorted(glob.glob("$WORK/sprocket-*.gguf")):
    api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                    repo_id="$HF_REPO", repo_type="model")
    print("  uploaded", os.path.basename(f))
PY
    fi
  ) || log "! GGUF step failed - the HF model is already pushed and safe; convert later"
fi

# ---------------------------------------------------------------- 8. the log
# train.log is the ONLY record of throughput, the loss curve, the LR schedule
# and the corpus-build rates, and until this step it exists nowhere but the
# network volume - so deleting the volume to stop paying for it would silently
# destroy the entire training history. Push it to the Hub next to the model.
# Runs last and never fails the pipeline: the model is already safe by here.
if [ -n "${HF_TOKEN:-}" ] && [ -n "$HF_REPO" ]; then
  log "archiving train.log to the Hub"
  python - <<PY || log "  ! log upload failed (model is already pushed and safe)"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for src, dst in [("$WORK/train.log", "debug/train.log"),
                 ("$WORK/watchdog.log", "debug/watchdog.log")]:
    if os.path.exists(src) and os.path.getsize(src):
        api.upload_file(path_or_fileobj=src, path_in_repo=dst,
                        repo_id="$HF_REPO", repo_type="model")
        print(f"  archived {dst} ({os.path.getsize(src):,} bytes)")
PY
fi

log "DONE."
log "  base ckpt : $WORK/checkpoints/${PRESET}_final.pt"
log "  sft ckpt  : $WORK/checkpoints/${PRESET}_sft_final.pt"
log "  hf model  : $WORK/hf-${PRESET}"
log "  gguf      : $WORK/sprocket-${PRESET}-q4_k_m.gguf"

# Termination is handled by the EXIT trap above - it fires on success AND on
# failure, so there is no path where the script ends and the pod keeps billing.
