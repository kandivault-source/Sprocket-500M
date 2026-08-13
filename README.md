# Sprocket 500M

A 501M-parameter language model built from scratch: custom tokenizer, original
architecture implementation, training and fine-tuning pipeline, and the cloud
orchestration that ran it unattended. Pretrained on 20.0B tokens across 54.3
hours on one rented H100, for about $165 in GPU time.

Weights, GGUF builds and a full behavioural writeup are published at
[HuggingFace7141/sprocket-500m](https://huggingface.co/HuggingFace7141/sprocket-500m).
The model card there is the honest account of what the model can and cannot do.
This README is about how it was built.

## What "from scratch" means here

Everything above the framework is original: the BPE tokenizer and its 32,000
token vocabulary, the transformer implementation, the training loop, the
supervised fine-tuning stage with assistant-only loss masking, the corpus
builder, the HuggingFace export with numerical parity checking, and the
single-command cloud bootstrap.

PyTorch supplies the autograd engine and the fused attention kernel. FineWeb-Edu
supplies the pretraining text. Nothing else is borrowed. `src/model/model.py` is
a little over 200 lines, and there is no line in it that is not understood.

## The constraint that shaped the project

Development happened on an RTX 4060 with 8 GB of VRAM, of which about 7 GB is
usable once the desktop has its share. Windows silently spills VRAM overflow into
system RAM rather than raising an out-of-memory error, so an oversized batch does
not fail, it just quietly destroys throughput. Everything was hard-budgeted to
7 GB to stay ahead of that.

fp32 AdamW costs roughly 16 bytes per parameter across weights, gradients and
optimizer moments. That puts the local training ceiling near 340M parameters, and
a 1B model's optimizer state alone at 17.7 GB. The 501M flagship was therefore
never trainable locally, which set the strategy: build and debug the entire
pipeline against a small model that trains overnight, and rent the large GPU only
once the same code has run end to end without intervention.

That prototype was a 76.4M model, trained locally to a validation loss of 4.7
from a 10.4 start. Its purpose was not to be good. Its purpose was to prove the
tokenizer, the architecture, the training loop, the checkpoint format and the
resume path before any of them cost money.

## Architecture

| | |
|---|---|
| Parameters | 501.1M total, 460.1M non-embedding |
| Layers / width | 26 layers, dim 1280 |
| Attention | 20 query heads, 4 KV heads (grouped-query) |
| Position | RoPE, real-valued split-half rotation |
| Normalisation | RMSNorm, pre-norm |
| Feed-forward | SwiGLU |
| Embeddings | Weight-tied input and output |
| Context | 2048 |
| Vocabulary | 32,000, byte-level BPE trained from scratch |
| Precision | bf16 |

One config dataclass drives every size from the 50M prototype to the flagship,
which is what made the local-then-cloud strategy possible: the code that ran on
the 4060 is the code that ran on the H100.

Grouped-query attention at 20:4 was chosen for inference rather than training.
For a model meant to run on a laptop or a phone, the KV cache is what constrains
long context, not the weights. Cutting KV heads 5x cuts that cache 5x.

## The run

Every figure below is generated from the run's own log by
`scripts/plot_figures.py`. Nothing is redrawn by hand.

![Pretraining loss](docs/figures/pretrain-loss.svg)

Pretraining cross-entropy against tokens seen, with the first 1B tokens omitted
because the collapse out of random initialisation (10.6 down to about 3.9 inside
the first few percent) compresses everything after it into a flat band. What is
left is the part that took 54 hours: a long grind from 3.1 to a final validation
loss of **2.564** at 152,580 steps, with a visible step down as the cosine
schedule anneals. Validation tracks train the whole way, which at 40 tokens per
parameter is what you would expect. The model never sees enough repeated data to
start memorising it.

![Instruct fine-tuning loss](docs/figures/sft-loss.svg)

Instruction fine-tuning over 1,192 steps, roughly 8 epochs of the synthetic
conversation set, with loss masked to assistant tokens only so the model is
never scored on predicting the user's turn. Final train 1.604, validation
**1.840**. The gap opening after about step 600 is the point where further
epochs stop buying generalisation.

## Throughput engineering

The first cloud smoke run sustained 63,470 tokens/second at 19% model FLOPs
utilisation. Two changes took that to 115,295 tok/s at 35% MFU, a 1.99x
improvement, which on a fixed token budget is close to halving the bill.

**Grouped-query attention was materialising its own inefficiency.** The original
code expanded K and V with `repeat_interleave` to match the query head count,
building tensors 5x larger than needed on every layer of every step, roughly
3.3 GB of redundant memory traffic per step. PyTorch 2.5 added `enable_gqa` to
scaled dot-product attention, which broadcasts KV heads internally instead. The
replacement was verified numerically rather than assumed correct: maximum
absolute difference 9.18e-06 and 100% argmax agreement against the old path.

**`torch.compile` was worth 1.70x on its own** (73,252 to 126,226 tok/s) and cut
VRAM from 62 GB to 43 GB at the same time. It cannot be measured on Windows,
because the inductor backend needs a C compiler, so this was found only by
running the sweep on the pod itself.

![Sustained throughput](docs/figures/throughput.svg)

Sustained tokens per second across the whole 20B run. The flat line is the
result: no thermal decay, no dataloader stalls, no drift over 54 hours. The only
movement is the compile warmup in the first few minutes.

A full sweep of 11 configurations settled the final choice at context 2048 with
micro-batch 16. Context 1024 is about 10% faster, but a 1024-token chat model is
barely usable, and over a 20B-token run that difference is around $14. Micro-batch
16 over 24 gives up 2% throughput for 23 GB of headroom, which matters across a
48-hour run where memory fragmentation can kill a tight configuration.

## Choosing the token budget

Cost is GPU-hours, not VRAM, so a bigger card buys speed rather than savings.
Quality against tokens is strongly sublinear: going from 10B to 50B tokens costs
5x for roughly 21% lower perplexity, with the knee around 15B to 20B. Perplexity
also saturates well before capability does. Past a point, real capability comes
from model size and instruction-data quality rather than more pretraining tokens.
20B was chosen at that knee.

The resulting model sits at 40 tokens per parameter. That number, not parameter
count, sets its peer group: GPT-2-medium is around 28 and Cerebras-GPT-590M is 20.
It is not comparable to Qwen2.5-0.5B at roughly 36,000 tokens per parameter. The
model card is explicit about this, because a 0.5B model invites exactly the wrong
comparison.

## The most interesting result

Tool calling worked. Memory did not. Both are special control tokens, trained the
same way, from the same corpus, in the same run.

Tool calling had 590 emitting examples and learned to produce well-formed JSON,
select the right tool from a manifest containing distractors, use the returned
result, and correctly stay silent when no tool was needed. Memory writing had 237
examples and never fired at all. Upweighting the memory examples 24x did not fix
it. Instead the model started answering "remember this" with a tool call.

At this scale the model reliably learns one control-token pathway, and the
stronger one crowds out the weaker. More upweighting made it worse, which is the
signature of a capacity and discrimination limit rather than a data-volume
problem. That is a result that only shows up if you actually read the generations
instead of trusting a pass-rate.

The follow-through is the part that mattered. If one control-token pathway was
already crowding out another, the next question was what the control tokens were
costing the ordinary conversation. So a second fine-tune ran from the same
pretrained base with the tool and memory examples removed entirely: 19,435
conversations over 405 steps. That build holds a conversation more consistently
than the instruct one, and it is what ships as the `-chat-` GGUF.

Both are published, because they are different tradeoffs against the same base
rather than one superseding the other. The instruct build can call tools. The
chat build talks. At 501M parameters the capability budget is real, and spending
part of it on a mechanical output format has a price that shows up somewhere
else.

## Running unattended for 54 hours

A spot instance will be preempted, and a script that needs supervision is a
script that will fail at 3am on day two. The orchestration in `cloud/bootstrap.sh`
is built around that.

**Every stage resumes.** A checkpoint carries model weights, optimizer state,
learning-rate schedule position, RNG states, token count and dashboard history, so
a resumed run continues as though it never stopped rather than silently restarting
the cosine schedule at peak learning rate. Checkpoints are written to a temporary
file and moved into place atomically, so preemption during a write cannot destroy
a good checkpoint. After a preemption the correct action is to relaunch the
identical command.

**A preflight exercises every dependency before spending a GPU-hour.** CUDA
availability, the exact `transformers` import that killed an earlier run, tokenizer
special-token IDs, chat-template rendering, corpus presence, and a real upload to
HuggingFace to prove write access. Anything environment-shaped fails in about 60
seconds instead of on day five.

**Three independent billing guards.** Normal completion, an EXIT trap that fires
on any failure, and a hard wall-clock watchdog for a wedge rather than a crash.
All three were tested against success, crash and preemption paths. The watchdog's
stdio has to be detached and disowned, because a backgrounded job inheriting
stdout holds the caller's pipe open, which made `curl | bash | tee` hang forever
after training had already succeeded, with the pod alive and billing.

## What only a real run could teach

Six failures were caught by cheap smoke runs, none of which were reproducible
locally:

1. `transformers` was unpinned, resolved to 5.x, and died on a `DTensor`
   ImportError during export, after pretraining and fine-tuning had both
   completed. On a real run that failure lands on the last day.
2. The corpus builder wrote once per document, which is 194,000 tiny writes to a
   network volume: 453K tokens/second against 2.86M/s on local disk. It now
   buffers 16M tokens.
3. A stale corpus manifest from a smoke run killed the first 20B launch 20 seconds
   after a clean preflight, and it self-terminated so cleanly that it looked
   launched from outside. Corpus filenames now encode the dataset subset.
4. A leftover smoke fine-tuning checkpoint made the SFT stage a silent no-op. It
   reported "RESUMED at step 466/465" followed by "SFT complete", having trained
   on nothing, and would have shipped the smoke model as the flagship.
5. The guard against that, in turn, needed its own guard. Archiving old
   checkpoints on a flag passed as a pod environment variable would re-fire on
   every preemption relaunch, throwing away 20 hours of progress. It is now
   marker-guarded on the network volume, so it can fire at most once.
6. llama.cpp identifies BPE pre-tokenizers by hashing their configuration against
   a hardcoded list, so GGUF conversion failed on an unrecognised tokenizer after
   the model had already trained and been pushed. The build now registers the
   hash explicitly rather than disabling the check, so a genuinely unknown
   tokenizer still fails loudly.

The theme is that the expensive failures were not in the model code. They were in
the seams: version resolution, filesystem behaviour under a network volume, and
resume logic that was too eager to believe it had already finished.

## Repository layout

```
src/model/model.py        transformer implementation
src/train/train.py        pretraining loop, resume, memmap and GPU data loaders
src/train/sft.py          supervised fine-tuning, assistant-only loss masking
src/tokenizer/            BPE tokenizer training
scripts/build_corpus.py   streaming FineWeb-Edu download and tokenization
scripts/export_hf.py      HuggingFace export with logit-parity verification
scripts/server.py         local dashboard server
scripts/plot_run.py       parses train.log into an interactive HTML report
scripts/plot_figures.py   the same charts as standalone SVG, for this README
cloud/bootstrap.sh        one-shot pod bootstrap: corpus, pretrain, SFT, export, publish
cloud/RUNPOD.md           cloud setup, measured costs, and the failure modes
eval/persona_battery.py   22-case behavioural battery
dashboard/                live training dashboard
logs/train.log            the complete run log, including the throughput sweep
logs/run_report.html      generated report: loss, throughput, LR, corpus build
docs/figures/             generated SVGs used above
```

`logs/run_report.html` is a single self-contained file with hover readouts and
the full data table, built from the same parser. It is the version to open when
the static figures are not enough.

`scripts/` also holds the planning tools that produced the numbers above:
`scaling_math.py`, `train_cost.py`, `quality_curve.py`, `memory_math.py` and
`tune_throughput.py`.

## The dashboard

`dashboard/` is a local build dashboard served by `scripts/server.py`, written to
explain the project to non-technical people via a Simple and Technical toggle. It
has an interactive tokenizer, a vocabulary browser, a 3D embedding map, a weights
inspector, a live loss curve streamed from the training loop, and a text
generation playground. `preview-screenshot.png` shows it during the 76.4M
prototype run, which is the phase it was built for.

## Data and reproduction

Pretraining used FineWeb-Edu (`sample/100BT`), which is downloaded and tokenized
at build time rather than stored here. See NOTICE for licensing of that and of the
safety prompt set.

**The synthetic instruction corpus is not distributed with this repository.** The
21,371 fine-tuning conversations were generated with Claude, and the generation
pipeline is here (`workflows/`, `scripts/harvest_round.py`,
`scripts/consolidate_instruct.py`, `scripts/audit_synth.py`) but its output is not.
The cloud bootstrap expects that corpus at `data/synthetic/sprocket_sft.jsonl` and
fails in preflight with a pointer if it is absent, rather than 20 GPU-hours later.
Pretraining and the throughput work reproduce without it.

## License

Apache 2.0. See LICENSE and NOTICE.
