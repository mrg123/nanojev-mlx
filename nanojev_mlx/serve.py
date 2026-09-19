"""本地 HTTP 服务：与 NanoJev 原实现的 POST /api/evaluate 协议兼容。

    python -m nanojev_mlx.serve --checkpoint-dir checkpoints/NanoJev --port 8765

单线程 HTTPServer：MLX 的推理调用不做并发共享，串行处理可避免竞争。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .model import load_model
from .predict import run_prediction
from .text import reject_nonfinite, unique_object, validate_request

MAX_REQUEST_BYTES = 2_000_000
LIMITS = {"states": 32, "questions": 96, "paths": 256}


def server_class(engine, web_root: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, content, mime="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, code, data):
            self.send(code, json.dumps(data, ensure_ascii=False, allow_nan=False).encode())

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/health":
                self.send_json(
                    200,
                    {
                        "ready": True,
                        "runtime": "mlx",
                        "model_loaded_once": True,
                        "provider_calls": 0,
                    },
                )
                return
            if web_root is None:
                self.send_json(404, {"error": "此服务未挂载静态站点，请使用 --web-root"})
                return
            relative = unquote(route).lstrip("/") or "index.html"
            target = (web_root / relative).resolve()
            if not target.is_relative_to(web_root) or not target.is_file():
                self.send_json(404, {"error": "File not found"})
                return
            self.send(200, target.read_bytes(), mimetypes.guess_type(str(target))[0] or "application/octet-stream")

        def do_POST(self):
            if urlparse(self.path).path != "/api/evaluate":
                self.send_json(404, {"error": "Unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError(f"Request must contain 1..{MAX_REQUEST_BYTES} bytes")
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise ValueError("Cross-origin requests are disabled")
                payload = json.loads(
                    self.rfile.read(length), object_pairs_hook=unique_object, parse_constant=reject_nonfinite
                )
                states = validate_request(payload)
                questions = [q for s in states for q in s["questions"].values()]
                paths = sum(1 for q in questions if q["type"] == "boolean") + sum(
                    len(q["criteria"]) for q in questions if q["type"] != "boolean"
                )
                if len(states) > LIMITS["states"] or len(questions) > LIMITS["questions"] or paths > LIMITS["paths"]:
                    raise ValueError(
                        f"Local demo limit: {LIMITS['states']} states, "
                        f"{LIMITS['questions']} questions, {LIMITS['paths']} candidate paths per request"
                    )
                before = time.perf_counter()
                result = run_prediction(engine, payload)
                result["execution"]["server_evaluation_seconds"] = time.perf_counter() - before
                self.send_json(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception:
                self.send_json(
                    500,
                    {"error": "Local model inference failed; inspect the server process. No fallback was used."},
                )
                raise

        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--web-root", default=None, help="可选：同时挂载一个静态站点目录")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    engine = load_model(args.checkpoint_dir)

    web_root = None
    if args.web_root:
        web_root = Path(args.web_root).resolve()
        if not (web_root / "index.html").is_file():
            raise ValueError("web-root must contain index.html")

    server = HTTPServer((args.host, args.port), server_class(engine, web_root))
    print(
        json.dumps(
            {"url": f"http://{args.host}:{args.port}", "ready": True, "runtime": "mlx", "provider_calls": 0}
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
