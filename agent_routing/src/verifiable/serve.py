"""Small frozen HF advisor server. Bind to loopback; no paid external API needed."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import hashlib
from pathlib import Path

from .backend import HFBackend
from .protocol import KINDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-context", type=int, default=16384)
    args = p.parse_args()
    backend = HFBackend(args.model, args.checkpoint, args.max_context)
    from .runner import checkpoint_identity
    fingerprint = {"model": args.model, "checkpoint": checkpoint_identity(args.checkpoint or args.model),
                   "resolved_revision": getattr(backend.model.config, "_commit_hash", None),
                   "template_sha256": hashlib.sha256(Path(__file__).with_name("chat_template.jinja").read_bytes()).hexdigest()}

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, data):
            payload = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.send_json(200, {"status": "ready", "model": args.model,
                                 "checkpoint": args.checkpoint, "aliases": KINDS, "margent_advisor": fingerprint})

        def do_POST(self):
            try:
                if self.path != "/v1/chat/completions":
                    return self.send_json(404, {"error": "Unknown endpoint"})
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0 or length > 2_000_000:
                    return self.send_json(400, {"error": "Invalid request size"})
                request = json.loads(self.rfile.read(length))
                if request.get("model") not in KINDS:
                    return self.send_json(400, {"error": "Unknown frozen advisor alias"})
                result = backend.generate(request["messages"], max_tokens=int(request["max_tokens"]))
                self.send_json(200, {"choices": [{"message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": "length" if result["truncated"] else "stop"}], "usage": {
                    "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"]}, "margent_advisor": fingerprint})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})

    print(f"Frozen advisor ready at http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
