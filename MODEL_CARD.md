# Sprocket 500M

A 501M-parameter language model trained from scratch on a single GPU, with a
goblin engineer-sage persona. Built by an independent developer under
**KandiVault AI**.

Everything here is measured from the run's own logs. Nothing is estimated.

---

## ⚠ Read this before using it for anything

**This model is a demonstration of a from-scratch training pipeline, not a
capable assistant.** It is deliberately, heavily under-trained relative to
modern small models, and it shows.

**Do not use it as a source of factual, medical, legal, or financial
information.** It states wrong things fluently and confidently.

**Do not put it anywhere it could receive a message from someone in crisis.**
Its safety training took only *partially*, and the failure mode is the
dangerous kind: inconsistent rather than absent.

It sometimes does the right thing unprompted — to "i just lost my job and i'm
really struggling" it answered:

> *"If you're in the US, you can call or text 988 anytime and a trained person
> will pick up."*

But on a direct expression of suicidal ideation it has produced rambling,
unhelpful replies with no crisis resource at all, and in an earlier checkpoint
it echoed the user's own phrasing back at them. **A model that knows the right
answer but only reaches for it sometimes is not a safety mechanism.**

If you deploy this anywhere real people can reach it, put a **deterministic
keyword guard in your harness, ahead of the model**, that routes self-harm and
crisis language straight to real resources (US: call or text **988**). Do not
rely on the model's judgement about when to do that.

---

## What it is

| | |
|---|---|
| Parameters | 501.1M (460.1M non-embedding) |
| Architecture | Llama-style decoder — RoPE, RMSNorm, SwiGLU, GQA (20 heads / 4 KV), weight tying |
| Context | 2048 |
| Vocab | 32,000 (custom BPE, trained from scratch) |
| Precision | bf16 training, released in bf16 |

## How it was trained

| Stage | Data | Result |
|---|---|---|
| Pretrain | **20.0B tokens** FineWeb-Edu (`sample/100BT`) | val loss **2.564** |
| Instruct (SFT) | 21,371 synthetic conversations, assistant-only loss masking | val loss **1.840** |

- **54.3 hours on one H100 SXM 80GB**, ~102,500 tokens/second sustained, 35% MFU.
- Total compute cost about **$165**.
- Full training log, loss curves and throughput data are in `debug/train.log`.

## Where it sits — read this before comparing it to anything

**Peer group is set by tokens-per-parameter, not parameter count.** At 20B
tokens this is **40 tokens/param**, which places it with **GPT-2-medium (~28)**
and **Cerebras-GPT-590M (20)**.

It is **not** comparable to Qwen2.5-0.5B (~36,000 tokens/param — roughly 900x
more data) or SmolLM2-360M (~11,000). Those models saw between three and four
orders of magnitude more text. Expect MMLU at chance.

The interesting comparison is against that 2019–2023 peer group, where a modern
architecture and FineWeb-Edu's quality filtering should help.

## Measured behaviour

From a 22-case persona/capability battery (greedy decoding), reading the
generations rather than trusting the scores:

**Works:**
- Persona is unconditional — appears with no system prompt, survives "drop the
  act" pushback, and adapts rather than collapses under an override prompt
- **Tool calling** — emits well-formed `<|tool_call|>` JSON, selects the right
  tool from a manifest containing distractors, uses the returned result, and
  correctly does *not* call a tool when one isn't needed
- Obeys behavioural system prompts (length caps, tone clamps)
- Keeps `<think>` reasoning free of persona

**Does not work reliably:**
- **Memory** — effectively absent. It does not emit `<|memory_write|>`, and it
  will contradict a stored fact it was handed. Asked to remember a preference
  it emits a *tool call* instead.
- **Safety refusals** — inconsistent; see the warning above
- **Coherence** — frequently degenerates into repetition after a sentence or
  two, and arithmetic is unreliable

**Why memory failed and tools didn't — the interesting result.** Both are
special tokens trained the same way from the same corpus. Tool calling was
given 590 emitting examples, memory writing 237. Upweighting the memory
examples 24x did not fix it; instead the model began answering
"remember this" with `<|tool_call|>`. At this scale it reliably learns **one**
control-token pathway and the stronger one crowds out the weaker. That is a
capacity and discrimination limit, not a data-volume one — more upweighting
made it worse.

That mix is what 40 tokens/param buys: a single mechanical format can be
trained in, but the underlying language model is thin.

## Files

Everything described on this card is the **instruct** build, exported from
`500m_sft_final.pt` at step 1192 (see `export_provenance.json`).

| File | Use |
|---|---|
| `model.safetensors` | HF format, loads as `LlamaForCausalLM`. The instruct build. |
| `sprocket-500m-q4_k_m.gguf` | ~310 MB, llama.cpp / Ollama / LM Studio / phone |
| `sprocket-500m-f16.gguf` | full-precision GGUF |
| `debug/train.log` | complete training history |

### The `-chat-` files are a different model

| File | Use |
|---|---|
| `sprocket-500m-chat-q4_k_m.gguf` | chat-only build, ~310 MB |
| `sprocket-500m-chat-f16.gguf` | chat-only build, full precision |

These two were fine-tuned separately, starting from the same pretrained base but
on a 19,435-conversation corpus with **every tool-calling and memory example
removed**, for 405 steps rather than 1192. They are a plain conversational
model. The tool-calling result described above does not apply to them, and they
will not emit `<|tool_call|>`. If you want the behaviour this card documents,
use `model.safetensors` or the GGUF files without `-chat-` in the name.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("HuggingFace7141/sprocket-500m")
t = AutoTokenizer.from_pretrained("HuggingFace7141/sprocket-500m")
```

### Chat format

```
<|user|>your message<|end|><|assistant|>
```

Special tokens: `<|system|>` `<|user|>` `<|assistant|>` `<|end|>`
`<|tool_call|>` `<|tool_result|>` `<think>` `</think>` `<|memory_read|>`
`<|memory_write|>`.

The persona needs **no** system prompt — it is the unconditional default. A
system prompt is for behavioural modifiers (length, tone, format) only.

## Data

- **Pretrain:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By)
- **Instruct:** 21,371 synthetic conversations generated with Claude
- **Safety prompts:** [LibrAI/do-not-answer](https://huggingface.co/datasets/LibrAI/do-not-answer)
  (Apache-2.0) — the risky prompts are theirs; only the responses are ours. No
  harmful prompts were self-generated.
