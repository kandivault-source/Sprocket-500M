"""Dashboard server: static files + live endpoints.

  GET /api/tokenize?text=...              -> real tokenizer output
  GET /api/metrics                        -> current training metrics
  GET /api/generate?prompt=...&tokens=N   -> run the REAL trained model and return its text

Run:  py scripts/server.py 8017
"""
import json
import os
import sys
import threading
from functools import partial
from urllib.parse import urlparse, parse_qs
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from tokenizers import Tokenizer
from model.model import GPT, ModelConfig

TOK = Tokenizer.from_file("config/tokenizer/tokenizer.json")
EOT = TOK.token_to_id("<|endoftext|>")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Load the trained prototype if a checkpoint exists.
MODEL = None
CKPT = "checkpoints/proto-75m_final.pt"
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    MODEL = GPT(ModelConfig(**ck["cfg"])).to(DEV).eval()
    MODEL.load_state_dict(ck["model"])
    print(f"loaded model: {MODEL.num_params()/1e6:.1f}M params on {DEV}", flush=True)
GEN_LOCK = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/tokenize":
            enc = TOK.encode(q.get("text", [""])[0])
            self._json({"pieces": [TOK.decode([i]) for i in enc.ids], "ids": enc.ids,
                        "n": len(enc.ids), "chars": len(q.get("text", [""])[0])})
            return
        if u.path == "/api/metrics":
            p = os.path.join("dashboard", "data", "metrics.json")
            self._json(json.load(open(p)) if os.path.exists(p) else {"status": "no run yet"})
            return
        if u.path == "/api/generate":
            if MODEL is None:
                self._json({"error": "no checkpoint"}); return
            prompt = q.get("prompt", [""])[0]
            ntok = max(1, min(int(q.get("tokens", ["100"])[0]), 200))
            temp = float(q.get("temp", ["0.8"])[0])
            ids = TOK.encode(prompt).ids if prompt else [EOT]
            with GEN_LOCK, torch.no_grad():
                out = MODEL.generate(torch.tensor([ids], device=DEV), ntok,
                                     temperature=temp, top_k=50)
            new_ids = out[0].tolist()[len(ids):]
            self._json({"prompt": prompt, "generated": TOK.decode([i for i in new_ids if i > 15]),
                        "n": len(new_ids)})
            return
        return super().do_GET()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8017
    handler = partial(Handler, directory="dashboard")
    print(f"dashboard: http://127.0.0.1:{port}/  (model {'loaded' if MODEL else 'none'})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
