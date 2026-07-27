"""export_hf.py — convert a native Sprocket checkpoint to HuggingFace Llama format.

WHY THIS EXISTS
    src/model/model.py is architecturally identical to Llama: RoPE + RMSNorm +
    SwiGLU + GQA + tied embeddings, no biases anywhere. So the checkpoint can be
    expressed as a `LlamaForCausalLM` with nothing but a key rename — and that
    unlocks the entire ecosystem for free:
        lm-evaluation-harness (`--model hf`)  ->  published-style benchmark numbers
        llama.cpp / GGUF                      ->  Q4/Q8 quantization, edge deploy
        Ollama, LM Studio, vLLM, TGI          ->  distribution
    Writing our own format instead would mean reimplementing all of that.

THE ROPE POINT (the usual conversion trap, and we are on the good side of it)
    Meta's original Llama rotates INTERLEAVED pairs, so HF's conversion script has
    to permute q/k. Our apply_rope() is SPLIT-HALF:
        out = cat([x1*cos - x2*sin,  x2*cos + x1*sin])
    HF's is  q*cos + rotate_half(q)*sin  with cos/sin duplicated to head_dim, where
    rotate_half(x) = cat([-x2, x1]):
        first  half: q1*cos + (-q2)*sin  = q1*cos - q2*sin   [matches]
        second half: q2*cos + ( q1)*sin  = q2*cos + q1*sin   [matches]
    Identical. NO PERMUTATION NEEDED — a straight rename is correct.
    If apply_rope() is ever changed to interleaved, this script must add the
    permutation or the export will be silently wrong (fluent but degraded output).

USAGE
    py scripts/export_hf.py checkpoints/proto-75m_final.pt out/hf-proto75m
    py scripts/export_hf.py <ckpt> <outdir> [--dtype float32|bfloat16] [--skip-verify]

The parity check is the point — it is not optional decoration. A wrong rename
produces a model that still emits plausible text, so silent corruption here is
entirely possible without it.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import GPT, ModelConfig  # noqa: E402

TOKENIZER_JSON = "config/tokenizer/tokenizer.json"

# id -> token, baked into the vocab before tokenizer training (ids 0-15 reserved).
SPECIALS = {
    0: "<|endoftext|>", 1: "<|pad|>", 2: "<|system|>", 3: "<|user|>",
    4: "<|assistant|>", 5: "<|end|>", 6: "<|tool_call|>", 7: "<|tool_result|>",
    8: "<think>", 9: "</think>", 10: "<|memory_read|>", 11: "<|memory_write|>",
}


def remap(sd: dict, n_layers: int) -> dict:
    """Native -> HF Llama key rename. Values are passed through untouched."""
    out = {
        "model.embed_tokens.weight": sd["tok_emb.weight"],
        "model.norm.weight": sd["norm.weight"],
        "lm_head.weight": sd["lm_head.weight"],
    }
    per_layer = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn.wq.weight": "self_attn.q_proj.weight",
        "attn.wk.weight": "self_attn.k_proj.weight",
        "attn.wv.weight": "self_attn.v_proj.weight",
        "attn.wo.weight": "self_attn.o_proj.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "ffn.w1.weight": "mlp.gate_proj.weight",   # gate
        "ffn.w3.weight": "mlp.up_proj.weight",     # up
        "ffn.w2.weight": "mlp.down_proj.weight",   # down
    }
    for i in range(n_layers):
        for src, dst in per_layer.items():
            out[f"model.layers.{i}.{dst}"] = sd[f"blocks.{i}.{src}"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("outdir")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--tol", type=float, default=2e-4,
                    help="max abs logit difference allowed in the parity check")
    a = ap.parse_args()

    from transformers import LlamaConfig, LlamaForCausalLM

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    cfg_d = ck["cfg"]
    cfg = ModelConfig(**cfg_d)

    n_layers = cfg.n_layers
    n_kv = cfg.n_kv_heads or cfg.n_heads
    intermediate = sd["blocks.0.ffn.w1.weight"].shape[0]   # read, never assume

    print("=" * 74)
    print(f"EXPORT  {a.ckpt}  ->  {a.outdir}")
    print("=" * 74)
    print(f"  dim={cfg.dim} layers={n_layers} heads={cfg.n_heads} kv_heads={n_kv} "
          f"ffn={intermediate} vocab={cfg.vocab_size} seq={cfg.max_seq_len}")

    hf_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.dim,
        intermediate_size=intermediate,
        num_hidden_layers=n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=n_kv,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope_theta,
        hidden_act="silu",
        tie_word_embeddings=True,     # lm_head.weight IS tok_emb.weight
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=0,
        eos_token_id=5,               # <|end|> terminates an assistant turn
        pad_token_id=1,
    )

    hf_sd = remap(sd, n_layers)
    hf = LlamaForCausalLM(hf_cfg)
    missing, unexpected = hf.load_state_dict(hf_sd, strict=False)
    # tie_word_embeddings means lm_head may be reported missing; it is tied, not absent.
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing or unexpected:
        sys.exit(f"  STATE DICT MISMATCH\n   missing={missing}\n   unexpected={unexpected}")
    print(f"  remapped {len(hf_sd)} tensors, 0 missing, 0 unexpected")

    # ---------------------------------------------------------------- parity
    if not a.skip_verify:
        native = GPT(cfg)
        native.load_state_dict(sd)
        native.eval()
        hf.eval()
        torch.manual_seed(0)
        ids = torch.randint(0, cfg.vocab_size, (2, min(128, cfg.max_seq_len)))
        with torch.no_grad():
            ln, _ = native(ids)
            lh = hf(ids).logits
        diff = (ln.float() - lh.float()).abs()
        mx, mean = diff.max().item(), diff.mean().item()
        # Compare argmax too: tiny numeric drift is fine, changed predictions are not.
        agree = (ln.argmax(-1) == lh.argmax(-1)).float().mean().item()
        print(f"  PARITY  max|diff|={mx:.2e}  mean|diff|={mean:.2e}  argmax agreement={agree:.4%}")
        if mx > a.tol or agree < 1.0:
            sys.exit(f"  FAILED — export does not match the native model "
                     f"(tol={a.tol:.0e}). Do NOT ship this checkpoint.")
        print("  PARITY OK — exported model is numerically identical to native.")

    # ---------------------------------------------------------------- write
    os.makedirs(a.outdir, exist_ok=True)
    hf = hf.to(getattr(torch, a.dtype))
    hf.save_pretrained(a.outdir, safe_serialization=True)

    if os.path.exists(TOKENIZER_JSON):
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(
            tokenizer_file=TOKENIZER_JSON,
            bos_token=SPECIALS[0], eos_token=SPECIALS[5], pad_token=SPECIALS[1],
            unk_token=SPECIALS[0],
            additional_special_tokens=[SPECIALS[i] for i in sorted(SPECIALS) if i not in (0, 1, 5)],
        )
        # GUARD: PreTrainedTokenizerFast silently APPENDS any special token that
        # isn't already in the vocab, which would push len(tok) past the model's
        # embedding rows and crash at inference (or, worse, emit OOV ids).
        # Caught exactly this in testing: <think> was absent, so vocab became 32004.
        if len(tok) != cfg.vocab_size:
            sys.exit(f"  TOKENIZER/MODEL VOCAB MISMATCH: tokenizer={len(tok)} "
                     f"model={cfg.vocab_size}.\n"
                     f"   A special token in SPECIALS is not in the trained vocab and "
                     f"was appended.\n   Run: py scripts/finalize_tokenizer.py --apply")
        for tid, s in SPECIALS.items():
            got = tok.convert_tokens_to_ids(s)
            if got != tid:
                sys.exit(f"  SPECIAL TOKEN ID MISMATCH: {s!r} is id {got}, expected {tid}")
        tok.save_pretrained(a.outdir)
        rt = tok("Oi. Sprocket. What're we buildin'?")["input_ids"]
        print(f"  tokenizer: vocab={len(tok)} (matches model), all {len(SPECIALS)} "
              f"special ids verified")
        print(f"  round-trip ok: {tok.decode(rt)!r}")
    else:
        print(f"  ! {TOKENIZER_JSON} not found — model exported WITHOUT a tokenizer")

    with open(os.path.join(a.outdir, "export_provenance.json"), "w", encoding="utf-8") as f:
        json.dump({"source_ckpt": a.ckpt, "source_iter": ck.get("it"),
                   "native_cfg": cfg_d, "dtype": a.dtype}, f, indent=2)

    print(f"\n  wrote {a.outdir}")
    print("  next:")
    print(f"    lm_eval --model hf --model_args pretrained={a.outdir},dtype=float32 \\")
    print("            --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande --batch_size 16")
    print(f"    python llama.cpp/convert_hf_to_gguf.py {a.outdir} --outtype f16")


if __name__ == "__main__":
    main()
