"""Demo V1's local server: one process, no dependencies, no build step.

Serves the interactive page and runs the governed pipeline on request. Binds
127.0.0.1 by default: the demo is a local tool, and a demo that listens on
every interface is a demo somebody else can run code through.

The API is three endpoints and nothing else:

    GET  /            the page
    GET  /api/cases   the case list and the backend status
    POST /api/run     {"demo_id": ..., "mode": ...} -> the marked run

A run that asks for real inference without a configured backend returns the
reason, not a fixture result. That is enforced in the engine, not here.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from demo.v1_pipeline import cases_payload, run_case

PAGE = Path(__file__).resolve().parent / "v1_page.html"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8741


def run_request(request: dict) -> tuple[int, dict]:
    """The /api/run contract, as a pure function.

    Extracted from the handler so the API's validation and error shapes are
    testable without a socket: the demo's contract is what a caller can rely
    on, and a 502 from some intermediary must not be able to fake it.
    """
    if not isinstance(request, dict):
        return 400, {"error": "bad request", "message": "expected a JSON object"}
    demo_id = request.get("demo_id")
    mode = request.get("mode", "fixture")
    if not isinstance(demo_id, str) or not demo_id.strip():
        return 400, {"error": "bad request", "message": "expected a demo_id"}
    if mode not in ("fixture", "live", "herb"):
        return 400, {"error": "bad request",
                     "message": "mode must be one of fixture, live, herb"}
    try:
        return 200, run_case(demo_id, mode=mode)
    except KeyError as exc:
        return 404, {"error": "unknown case", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the page shows what happened
        return 500, {"error": type(exc).__name__, "message": str(exc)}


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "PhytoForgeDemo/1.0"

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _page(self) -> None:
        if not PAGE.is_file():
            self._json({"error": "page missing", "message": str(PAGE)}, status=500)
            return
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._page()
        elif path == "/api/cases":
            self._json(cases_payload())
        else:
            self._json({"error": "not found", "message": path}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        path = urlparse(self.path).path
        if path != "/api/run":
            self._json({"error": "not found", "message": path}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            self._json({"error": "bad request", "message": "expected a JSON body"},
                       status=400)
            return
        status, payload = run_request(request)
        self._json(payload, status=status)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - base signature
        # One line per request, to stderr, so the server's own console stays readable.
        print(f"[demo] {self.address_string()} {format % args}", flush=True)


def main(argv: list[str] | None = None) -> int:
    # Runs are serialised inside the engine (demo.v1_pipeline._RUN_LOCK): the
    # server is threaded, but a run mutates process-wide backend configuration
    # for the duration of a case that declares one, so two concurrent runs must
    # not interleave. A visitor waits for the previous run instead of silently
    # borrowing its backend — the marking stays true under concurrency.
    import argparse

    parser = argparse.ArgumentParser(description="PhytoForge Demo V1")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo V1 on http://{args.host}:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
