/**
 * Sprocket's tokenizer, reimplemented for the browser.
 *
 * The model's real tokenizer is the HuggingFace `tokenizers` library reading
 * config/tokenizer/tokenizer.json. This is a byte-level BPE with no normalizer,
 * which is simple enough to reproduce exactly, so the explorer page can encode
 * text with nothing running behind it.
 *
 * "Exactly" is a claim that needs checking, so scripts/verify_web_tokenizer.py
 * runs both implementations over the same strings and fails on any divergence.
 *
 * Kept dependency-free and usable from both a browser and node, so the same code
 * that ships is the code the test exercises.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SprocketTokenizer = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  /**
   * One printable character per byte value.
   *
   * Raw bytes include spaces, newlines and control codes, none of which can be
   * stored in a vocabulary file or shown on screen as themselves. This is the
   * same mapping GPT-2 uses, which is why a leading space appears as U+0120.
   */
  function buildByteMaps() {
    var bs = [], i;
    for (i = 33; i <= 126; i++) bs.push(i);
    for (i = 161; i <= 172; i++) bs.push(i);
    for (i = 174; i <= 255; i++) bs.push(i);

    var cs = bs.slice(), n = 0;
    for (i = 0; i < 256; i++) {
      if (bs.indexOf(i) === -1) { bs.push(i); cs.push(256 + n); n++; }
    }

    var enc = {}, dec = {};
    for (i = 0; i < bs.length; i++) {
      var ch = String.fromCodePoint(cs[i]);
      enc[bs[i]] = ch;
      dec[ch] = bs[i];
    }
    return { enc: enc, dec: dec };
  }

  var BYTES = buildByteMaps();

  /**
   * Pre-tokenizer split. Merges are never allowed to cross these boundaries,
   * which is what stops the vocabulary learning entries that span a word break.
   */
  var SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  var ENCODER = new TextEncoder();
  var DECODER = new TextDecoder("utf-8", { fatal: false });

  function toByteString(s) {
    var bytes = ENCODER.encode(s), out = "";
    for (var i = 0; i < bytes.length; i++) out += BYTES.enc[bytes[i]];
    return out;
  }

  function fromByteString(s) {
    var arr = [];
    for (var ch of s) {
      var b = BYTES.dec[ch];
      if (b !== undefined) arr.push(b);
    }
    return DECODER.decode(new Uint8Array(arr));
  }

  function escapeRe(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function Tokenizer(data) {
    this.ids = data.tokens;
    this.vocab = new Map();
    for (var i = 0; i < this.ids.length; i++) this.vocab.set(this.ids[i], i);

    // Merge order is the whole algorithm: rank 0 was the most frequent pair in
    // the training sample and is always applied first.
    this.ranks = new Map();
    for (i = 0; i < data.merges.length; i++) this.ranks.set(data.merges[i], i);

    this.specials = data.specials || {};
    this.specialById = {};
    var keys = Object.keys(this.specials);
    for (i = 0; i < keys.length; i++) this.specialById[this.specials[keys[i]]] = keys[i];

    // Longest-first, so <|memory_write|> is never matched as a prefix of itself.
    keys.sort(function (a, b) { return b.length - a.length; });
    this.specialRe = keys.length
      ? new RegExp("(" + keys.map(escapeRe).join("|") + ")")
      : null;

    this.cache = new Map();
  }

  /** Apply the lowest-ranked available merge until none remain. */
  Tokenizer.prototype._bpe = function (piece) {
    var hit = this.cache.get(piece);
    if (hit) return hit;

    var word = Array.from(piece);
    if (word.length < 2) { this.cache.set(piece, word); return word; }

    for (;;) {
      var best = Infinity, at = -1;
      for (var i = 0; i < word.length - 1; i++) {
        var r = this.ranks.get(word[i] + " " + word[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at === -1) break;
      word = word.slice(0, at)
        .concat([word[at] + word[at + 1]])
        .concat(word.slice(at + 2));
    }

    this.cache.set(piece, word);
    return word;
  };

  /** Encode to [{ t, id, special }]. `t` is the byte-level form. */
  Tokenizer.prototype.encode = function (text) {
    var out = [];
    if (!text) return out;

    var parts = this.specialRe ? text.split(this.specialRe) : [text];
    for (var p = 0; p < parts.length; p++) {
      var part = parts[p];
      if (!part) continue;

      if (this.specials[part] !== undefined) {
        out.push({ t: part, id: this.specials[part], special: true });
        continue;
      }

      var matched = part.match(SPLIT);
      if (!matched) continue;

      for (var j = 0; j < matched.length; j++) {
        var pieces = this._bpe(toByteString(matched[j]));
        for (var k = 0; k < pieces.length; k++) {
          var id = this.vocab.get(pieces[k]);
          out.push({ t: pieces[k], id: id === undefined ? -1 : id, special: false });
        }
      }
    }
    return out;
  };

  Tokenizer.prototype.ids_of = function (text) {
    return this.encode(text).map(function (t) { return t.id; });
  };

  /** Readable form of a token, with reserved tokens passed through as-is. */
  Tokenizer.prototype.display = function (tok) {
    return tok.special ? tok.t : fromByteString(tok.t);
  };

  return {
    Tokenizer: Tokenizer,
    toByteString: toByteString,
    fromByteString: fromByteString
  };
});
