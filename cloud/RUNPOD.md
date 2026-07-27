# Sprocket on RunPod — from one command to a downloadable model

## LIVE CONFIG (provisioned 2026-07-27)

| Item | Value |
|---|---|
| Network volume | **`94biin05n5`** ("sprocket-vol", 50 GB, **US-NE-1**) |
| GPU for both smoke + real run | **H100 SXM 80GB @ $2.69/hr** |
| Repo | `github.com/kandivault-source/Sprocket-500M` (private) |

**Why US-NE-1:** H100 SXM is only bookable in 9 datacenters and only 5 of those
support network volumes (AP-JP-1, EU-FR-1, EUR-IS-3, EUR-NO-2, US-NE-1). US-NE-1
is the US one. A network volume **only attaches to pods in its own datacenter**,
so this choice is load-bearing — the volume and the GPU must match.

**Stock is "Low" everywhere for H100 SXM.** Budget for a retry loop on launch; if
it can't allocate, either wait or fall back to EU-FR-1 / EUR-NO-2 (which needs a
volume there too).

**Measured prices are below the originally quoted ones** — H100 SXM $2.69 (not
$2.99), RTX PRO 6000 $1.69, H200 SXM $3.59. Recomputed at $2.69: **20B = $130 ·
50B = $325 · 100B = $649.**

**Run the smoke test on the H100 in US-NE-1, not a cheap 4090 elsewhere.** It
costs ~$1.35 instead of ~$0.20, and in exchange it validates the exact GPU,
datacenter, and volume the real run will use — including whether "Low" stock is
actually obtainable. A 4090 smoke test in another DC proves much less.

---


The whole flow is: **one `runpodctl` command → pod boots → corpus → pretrain → SFT →
HuggingFace export → pushed to the Hub → pod terminates itself.**

If the pod is preempted, **run the identical command again**. Every stage resumes
from the network volume. That is the design, not a fallback.

---

## One-time setup (about 10 minutes)

**1. Push this repo to GitHub.** The pod clones it. Include `data/synthetic/*.jsonl`
(~37 MB — that's the SFT corpus). Do **not** include `data/processed/*.bin`; the pod
rebuilds those.

**2. Install and authenticate the CLI.**
```bash
brew install runpod/runpodctl/runpodctl     # or: https://github.com/runpod/runpodctl/releases
runpodctl config --apiKey YOUR_RUNPOD_KEY
```

**3. Create a NETWORK VOLUME** (RunPod console → Storage → Network Volume).
Pick the same datacenter you'll rent the GPU in.

| Token budget | `train.bin` | Volume to create |
|---|---|---|
| 20B | ~40 GB | **100 GB** |
| 50B | ~100 GB | **200 GB** |
| 100B | ~200 GB | **350 GB** |

This is the single most important setting. Container disk is **ephemeral** — a
checkpoint written there dies with the pod, and resume silently has nothing to
resume from. Note the volume ID.

**4. Get a HuggingFace write token** (huggingface.co → Settings → Access Tokens →
write). This is what makes the model land somewhere public and accessible.

> Set `HF_TOKEN` as a pod environment variable. Don't paste it into a script or
> commit it — anything in the repo is in your git history forever.

---

## The launch command

```bash
runpodctl create pod \
  --name sprocket-500m \
  --gpuType "NVIDIA H100 80GB HBM3" --gpuCount 1 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --networkVolumeId YOUR_VOLUME_ID --volumePath /workspace \
  --containerDiskSize 30 \
  --env REPO_URL=https://github.com/YOU/sprocket \
  --env HF_TOKEN=hf_xxx --env HF_REPO=YOU/sprocket-500m \
  --env TOKENS=50000000000 --env SUBSET=sample/100BT \
  --env PRESET=500m --env CTX=2048 --env MICRO_BATCH=32 \
  --args "bash -c 'curl -sL https://raw.githubusercontent.com/YOU/sprocket/main/cloud/bootstrap.sh | bash 2>&1 | tee -a /workspace/train.log'"
```

> **Verify flag names against your CLI version** — `runpodctl create pod --help`.
> RunPod has renamed flags between releases; the semantics below are what matter.

Then watch it:
```bash
runpodctl get pod                      # status
tail -f /workspace/train.log           # from the pod's web terminal
```

Or do the same thing in the web console: **Deploy → pick GPU → attach the network
volume → paste the env vars → put the `curl … | bash` line in "Container Start
Command."** Identical outcome; the CLI is just scriptable.

---

## What you get, and where

| Artifact | Path on the volume | What it's for |
|---|---|---|
| `checkpoints/500m_final.pt` | base model | resuming, further pretraining |
| `checkpoints/500m_sft_final.pt` | instruct model | the actual chat model |
| `hf-500m/` | HF format | **the accessible one** |
| `train.log` | full run log | loss curve, throughput, post-mortem |

Published to `https://huggingface.co/YOU/sprocket-500m`, after which anyone —
including you, on any machine — does:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("YOU/sprocket-500m")
t = AutoTokenizer.from_pretrained("YOU/sprocket-500m")
```

Didn't set `HF_TOKEN`? Pull it off the pod directly:
```bash
runpodctl send /workspace/hf-500m       # prints a one-time code
runpodctl receive <code>                # run this on your PC
```

**For phone/laptop deployment**, convert to GGUF — this is what makes it run in
~350 MB on an iPhone or in Ollama/LM Studio:
```bash
python llama.cpp/convert_hf_to_gguf.py /workspace/hf-500m --outtype f16
./llama.cpp/llama-quantize sprocket-500m-f16.gguf sprocket-500m-q4_k_m.gguf Q4_K_M
```
This works *only* because `export_hf.py` produces a genuine Llama-architecture
model — llama.cpp reads it with no custom code.

---

## Do the $5 smoke run first. Every time.

Never let a 5-day run be the first time the script executes end to end.

```bash
runpodctl create pod --name sprocket-smoke \
  --gpuType "NVIDIA GeForce RTX 4090" --gpuCount 1 \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --networkVolumeId YOUR_VOLUME_ID --volumePath /workspace \
  --env REPO_URL=https://github.com/YOU/sprocket \
  --env TOKENS=200000000 --env SUBSET=sample/10BT \
  --env PRESET=500m --env CTX=1024 --env MICRO_BATCH=8 --env MAX_ITERS=200 \
  --env AUTO_STOP=0 \
  --args "bash -c 'curl -sL https://raw.githubusercontent.com/YOU/sprocket/main/cloud/bootstrap.sh | bash 2>&1 | tee -a /workspace/smoke.log'"
```

~20 minutes, well under $1. It exercises the whole path: download → tokenize →
train → checkpoint → SFT → export. **Then test the part that actually matters:**

```bash
# kill the pod mid-training, recreate it with the same command, and confirm
# the log says "RESUMED ... at iter N" and the loss picks up where it left off
runpodctl remove pod <id>
```

If resume doesn't work, you find out for $1 instead of on day 3 of a $361 run.

---

## Cost, measured

| GPU | $/hr | 20B | 50B | 100B | $/B tok |
|---|---|---|---|---|---|
| RTX PRO 6000 96GB | 1.99 | 95h / $189 | 237h / $472 | 474h / $944 | 9.44 |
| **H100 80GB SXM** | **2.99** | **48h / $144** | **121h / $361** | **241h / $721** | **7.21** ← best |
| H200 SXM 141GB | 4.39 | 46h / $200 | 114h / $501 | 228h / $1,002 | 10.02 |

Plus corpus build: measured **2.86M tok/s** tokenizing locally, so 50B ≈ 5h ≈ **$15**
of H100 time. Cheaper to do it on a CPU-only pod against the same volume if the
datacenter offers one.

**H200 is a trap here** — same GH100 compute die as the H100 (989 TFLOPS bf16).
The premium buys HBM3e capacity/bandwidth that a 501M model, which uses ~8 GB and
is compute-bound, cannot exploit. ~6% faster for 47% more.

Re-run `py scripts/flagship_plan.py` after the smoke run with your measured MFU.

---

## Things that will actually bite you

**No network volume.** The #1 way to lose a run. Ephemeral disk + preemption =
start over. Check `/workspace` is the mount, not a container path.

**`MICRO_BATCH` too small on an 80 GB card.** Measured locally: efficiency rose
7.6 → 11.4 TFLOPS going mb 1 → 4, i.e. this model is launch-overhead-bound at
small batch. Start at 32 on an H100 and raise until VRAM is ~80% used. This is
the cheapest available speedup and it directly cuts the bill.

**Spot vs on-demand.** Spot is ~50-70% cheaper and resume makes it survivable —
that's why we built it. But a 5-day spot run in a busy datacenter can thrash.
If you're getting preempted more than a few times a day, switch to on-demand;
the restart overhead costs more than the discount saves.

**Forgetting to stop the pod.** `AUTO_STOP=1` handles the happy path. If the
script *crashes*, the pod keeps billing at $2.99/hr until you notice. Check
`runpodctl get pod` after any failure.

**Pushing secrets.** `HF_TOKEN` goes in `--env`, never in the repo.

**Cross-datacenter volume.** A network volume only attaches to pods in its own
datacenter. If your GPU choice isn't available there, you need a volume there too.
