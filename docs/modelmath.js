/**
 * The size, memory and cost arithmetic behind the explorer page.
 *
 * These are the same formulas the project used to decide what to build and what
 * to spend, so they are kept in one place and checked rather than reimplemented
 * inline. scripts/verify_web_modelmath.py runs paramCount() against the real
 * PyTorch model for every preset and fails on any disagreement.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ModelMath = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  /** The shipped model, for reference and for the reset button. */
  var SPROCKET = {
    vocab: 32000, dim: 1280, layers: 26, heads: 20, kvHeads: 4,
    ctx: 2048, ffnMultiple: 256
  };

  /**
   * Exact parameter count for this architecture.
   *
   * Matches src/model/model.py: RMSNorm has a weight and no bias, every linear
   * is bias-free, the feed-forward hidden size is 8/3 of the width rounded up to
   * a multiple of 256, and the output projection shares the embedding matrix, so
   * the embedding is counted once.
   */
  function paramCount(c) {
    var headDim = Math.floor(c.dim / c.heads);
    var mult = c.ffnMultiple || 256;
    var hidden = Math.floor(8 / 3 * c.dim);
    hidden = mult * Math.ceil(hidden / mult);

    var embedding = c.vocab * c.dim;
    var attn =
      c.dim * (c.heads * headDim) +      // wq
      c.dim * (c.kvHeads * headDim) +    // wk
      c.dim * (c.kvHeads * headDim) +    // wv
      (c.heads * headDim) * c.dim;       // wo
    var ffn = 3 * c.dim * hidden;        // gate, up, down
    var norms = 2 * c.dim;               // one before attention, one before ffn

    var perLayer = attn + ffn + norms;
    var total = embedding + perLayer * c.layers + c.dim;  // + final norm

    return {
      total: total,
      embedding: embedding,
      nonEmbedding: total - embedding,
      perLayer: perLayer,
      attnPerLayer: attn,
      ffnPerLayer: ffn,
      hidden: hidden,
      headDim: headDim
    };
  }

  /**
   * Bytes needed to TRAIN, which is what decides whether a machine can run it.
   *
   * Weights, gradients, and Adam's two running averages, at 4 bytes each in
   * fp32. This is why a card that can happily run a model can still be unable to
   * train one a quarter the size.
   */
  function trainingBytes(params) { return params * 16; }

  /** Bytes to hold the weights alone for inference at 2 bytes each. */
  function inferenceBytes(params) { return params * 2; }

  /**
   * Bytes of key-value cache per token of context.
   *
   * This is what grouped-query attention shrinks, and at long context it, not
   * the weights, is what runs a machine out of memory.
   */
  function kvBytesPerToken(c) {
    var headDim = Math.floor(c.dim / c.heads);
    return 2 * c.layers * c.kvHeads * headDim * 2;  // key and value, 2 bytes each
  }

  /**
   * Hours to pretrain, scaled from the throughput actually measured on the pod.
   *
   * Work per token is proportional to parameter count, so a model twice the size
   * runs about half as fast on the same card. Anchoring to a measured number
   * beats a theoretical peak nobody reaches.
   */
  var MEASURED = { params: 501090560, tokPerSec: 115295, usdPerHour: 2.99 };

  function trainingHours(params, tokens) {
    var rate = MEASURED.tokPerSec * (MEASURED.params / params);
    return tokens / rate / 3600;
  }

  function trainingCost(params, tokens, usdPerHour) {
    return trainingHours(params, tokens) * (usdPerHour || MEASURED.usdPerHour);
  }

  /** Training text per parameter. Sets which models are fair comparisons. */
  function tokensPerParam(params, tokens) { return tokens / params; }

  return {
    SPROCKET: SPROCKET,
    MEASURED: MEASURED,
    paramCount: paramCount,
    trainingBytes: trainingBytes,
    inferenceBytes: inferenceBytes,
    kvBytesPerToken: kvBytesPerToken,
    trainingHours: trainingHours,
    trainingCost: trainingCost,
    tokensPerParam: tokensPerParam
  };
});
