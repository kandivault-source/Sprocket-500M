# Homebrew LLM — build a language model from scratch

A ground-up large language model: our own tokenizer/vocab, our own architecture,
trained on our own curated + synthetic corpus, run locally, and published to the world.

Built and owned by KandiVault LLC. Every layer is understood and controlled — no black boxes.

---

## Hardware

**Local "lab" machine (development + prototyping):**
- GPU: NVIDIA RTX 4060, 8 GB VRAM (Ada, compute 8.9 — bf16 + flash-attention capable)
- CPU: Intel i9-10850K (10c / 20t)
- RAM: 64 GB
- Disk: D: has ~1.36 TB free (project + datasets + checkpoints live here)

**Cloud "factory" (the big training run):**
- Rented GPU, 90+ GB VRAM (A100 80GB / H100 80GB / H200 141GB class), ~$1.5–3/hr
- Used ONLY once the pipeline is fully validated locally. Never debug on rented hours.

VRAM is the binding constraint locally (8 GB). It sets the local model ceiling at ~125M params.
The cloud GPU is what unlocks the 1B-parameter target.

---

## Locked decisions

- **Data strategy:** Hybrid — curated real web text (FineWeb-Edu, free) as the base,
  plus Haiku-generated synthetic data for coherence, instruction-following, and persona.
- **Architecture:** Decoder-only transformer, modern Llama-style
  (RoPE positions, RMSNorm, SwiGLU MLP, PyTorch SDPA/flash attention). Config-driven so the
  same code scales from a 50M local prototype to the 1B cloud run.
- **Full capability arc:** base (pretraining) -> instruct/chat (SFT) -> tool use -> in-character persona.
- **Persona:** a distinctive character voice (candidates below). Decided at the instruct stage,
  so it can be chosen/changed late with zero rework to pretraining.

## Open decisions

- Final persona character (see candidates).
- Exact local prototype size (50M vs 125M) and exact cloud target (nail down token budget).
- Vocab size (leaning 32k BPE).

---

## Capability glossary (what the stages actually mean)

- **Base model (pretraining):** learns language by predicting the next token over raw text.
  Result: continues text like the internet, but does not "answer" — ask it a question and it may
  just write more questions. This is the foundation; everything else builds on it.
- **Instruct / SFT (supervised fine-tuning):** further-train the base model on
  (instruction -> good response) pairs. This is what converts a text-continuer into an assistant
  that responds helpfully to requests.
- **Chat:** instruction-tuning structured as multi-turn conversations with user/assistant roles
  and a chat template. Chat = instruct with conversation format.
- **Tool use / function calling:** train the model to emit a structured call (e.g. JSON) when it
  needs an external capability (search, calculator, code), plus a runtime that executes the call
  and feeds the result back. Needs a consistent format + demonstration data. Gets reliable only at
  larger scale — ~1B is roughly the low end where basic tool use starts working.
- **Persona:** all assistant responses are written in-character during instruct-tuning. This is
  where Haiku earns its keep — generate the whole instruct set in the chosen voice.

---

## Persona candidates

| Character | Voice hook | Fit |
|---|---|---|
| Goblin tinkerer | Greedy, gleeful, calls you "boss", loves shiny things and deals; rationalizes tool use as "gadgets" | Strong for a tool-using assistant |
| Gremlin | Chaotic, fast-talking, breaks and fixes things | Playful, fits a coding helper |
| Swamp / hedge witch | Cryptic, folksy, brews-and-omens metaphors, calls you "dearie" | Atmospheric, memorable |
| Bridge troll | Gruff, riddling, demands mock "payment" before answering, secretly soft | Comedic |
| Library golem / archivist spirit | Dusty, formal, speaks in catalog references, oddly wise | Knowledgeable feel |

---

## Measured hardware limits (2026-07-22)

- **Usable VRAM ~7 GB** (8.59 total minus desktop). Windows silently spills overflow to
  system RAM instead of OOM-ing, which destroys throughput — so we hard-budget to ~7 GB.
- **125M training throughput: ~18,100 tok/s** at micro-batch 4 (5.5 GB). Batch 6+ spills.
- **Local training ceiling ~340M params** (fp32 AdamW = ~16 bytes/param). The 1B model's
  optimizer states alone are 17.7 GB => **1B and up are cloud-only.** Biggest fully-local
  model ≈ 300M.
- **Long context is not free.** KV cache for a 1B model: 88 KB/token => 128k ctx = 11.5 GB
  (exceeds the 4060 even for inference), 1M ctx = 90 GB (exceeds an 80GB A100). Attention
  compute is O(T^2): 1M tokens = ~15,000x the FLOPs of an 8k pass. Long context requires
  (a) train short + extend via RoPE scaling, and (b) sliding-window attention to bound the
  KV cache. Realistic ship target: 32–64k full attention, 128k stretch via sliding window.
- **Training precision bf16 is standard and lossless in practice** — not the same as
  post-training quantization. Model width (dim 512–2048) is separate from precision.

## Verified environment (Phase 0 — DONE 2026-07-22)

- **torch 2.6.0+cu124**, CUDA available, RTX 4060 detected, bf16 supported, 8.59 GB VRAM.
- Real bf16 forward+backward+optimizer step confirmed (test loss 345 -> 4.8).
- FlashAttention kernel is NOT compiled in the Windows wheel, BUT the default SDPA uses the
  memory-efficient (cuDNN/cutlass) backend -> attention memory is ~O(T), not O(T^2).
  Measured: T=1024 uses 0.03 GB (vs 1.03 GB math fallback); T=4096 still only 0.07 GB.
  => long contexts are fine on 8 GB locally; the Linux cloud box gets true FlashAttention.
- Verification scripts: `scripts/verify_gpu.py`, `scripts/check_attention.py`.

## Roadmap

0. **Environment** — CUDA PyTorch + verify a real GPU training step. (DONE)
1. **Corpus** — FineWeb-Edu sample: 168k docs / ~200M tokens in `data/raw/`. (DONE; more volume later)
2. **Tokenizer** — 32k byte-level BPE trained from scratch, 4.52 chars/token. (DONE — `config/tokenizer/`)
3. **Architecture** — config-driven Llama-style decoder. (DONE — `src/model/model.py`)
3b. **Dashboard** — live local build dashboard w/ interactive tokenizer + Simple/Technical toggle.
   (DONE — `dashboard/`, served by `scripts/server.py`)
4. **Tokenize corpus** — flat uint16 `.bin` for training. (IN PROGRESS — `data/processed/train.bin`)
5. **Training loop + 75M prototype** — bf16, grad accumulation, cosine LR, AdamW, resumable
   checkpoints, live loss curve streaming to the dashboard. (NEXT)
6. **Synthetic data (Haiku fleet)** — generate the instruct + persona corpus at scale.
7. **Cloud run** — rent the 90+GB GPU, launch the 1B base pretraining, then SFT + persona.
8. **Tool use** — add function-calling format + demonstration data + a runtime harness.
9. **Inference** — sampling script + local FastAPI server + GGUF export for LM Studio / Ollama.
10. **Release** — HuggingFace repo, model card, license.

## Strategy: local lab, cloud factory

Build and debug the *entire* pipeline on the 4060 with a small model that trains overnight.
Only when it runs end-to-end, bug-free, on identical code, do we rent the big GPU — so the
expensive cloud hours are spent training, not debugging.
