from __future__ import annotations

import argparse
import json
import os
import time
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, unquote, urlparse

from agenteconomy.web.dashboard import (
    DEFAULT_OUTPUT_ROOT,
    latest_run,
    iter_run_dirs,
    read_stage_events,
    safe_run_dir,
    summarize_record_file,
    summarize_run_dir,
    _run_priority,
    _record_files,
    _safe_int,
)


WEB_ROOT = Path(__file__).resolve().parent


class DashboardHandler(SimpleHTTPRequestHandler):
    output_root = DEFAULT_OUTPUT_ROOT
    started_at = time.time()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.log_date_time_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(WEB_ROOT / "templates" / "dashboard.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/api/health":
            self._json(
                {
                    "ok": True,
                    "output_root": str(self.output_root),
                    "output_exists": self.output_root.exists(),
                    "uptime_seconds": round(time.time() - self.started_at, 3),
                }
            )
            return
        if parsed.path == "/api/runs":
            params = parse_qs(parsed.query)
            limit = max(1, min(_safe_int((params.get("limit") or ["50"])[0], 50), 200))
            items = [summarize_run_dir(run, self.output_root) for run in iter_run_dirs(self.output_root)]
            items.sort(key=_run_priority, reverse=True)
            self._json({"runs": items[:limit], "count": len(items)})
            return
        if parsed.path == "/api/latest":
            run = latest_run(self.output_root)
            if run is None:
                self._json({"run": None, "records": [], "events": [], "stage": {}})
                return
            records = [summarize_record_file(path) for path in _record_files(run)]
            events, stage = read_stage_events(run)
            self._json(
                {
                    "run": summarize_run_dir(run, self.output_root),
                    "records": records,
                    "events": events,
                    "stage": stage,
                }
            )
            return
        if parsed.path == "/api/run":
            params = parse_qs(parsed.query)
            run_id = unquote((params.get("path") or [""])[0])
            run = safe_run_dir(self.output_root, run_id)
            if run is None:
                self._json({"error": "run not found"}, status=HTTPStatus.NOT_FOUND)
                return
            records = [summarize_record_file(path) for path in _record_files(run)]
            events, stage = read_stage_events(run)
            self._json(
                {
                    "run": summarize_run_dir(run, self.output_root),
                    "records": records,
                    "events": events,
                    "stage": stage,
                }
            )
            return
        return super().do_GET()

    def _json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "file not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dependency-free AgentEconomy web dashboard")
    parser.add_argument("--host", default=os.getenv("AGENTECONOMY_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENTECONOMY_WEB_PORT", "7860")))
    parser.add_argument("--output-root", default=os.getenv("AGENTECONOMY_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    args = parser.parse_args()

    handler = DashboardHandler
    handler.output_root = Path(args.output_root).resolve()
    handler.started_at = time.time()
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving AgentEconomy dashboard on http://{args.host}:{args.port}", flush=True)
    print(f"Output root: {handler.output_root}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
