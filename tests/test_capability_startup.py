"""backend/capability.py：启动拜访 18000 的客户端行为。"""
from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.capability import (
    CapabilityUnavailable,
    describe_active,
    fetch_capability_snapshot,
)


def _payload() -> dict:
    return {
        "ok": True,
        "registry": {
            "schema_version": 1,
            "active": {"arm": "right_arm", "hand_id": "qiangnao-1-right"},
            "hands": [{"id": "qiangnao-1-right", "name": "强脑-1-右"}],
            "capabilities": [],
            "calibrations": [],
        },
        "calibrations": [{
            "arm": "right_arm",
            "hand_id": "qiangnao-1-right",
            "status": "ready",
        }],
        "meta": {},
    }


class _Server:
    def __init__(self, body: bytes):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):   # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(outer.body)

            def log_message(self, *_):
                pass

        self.body = body
        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FetchSnapshotTests(unittest.TestCase):
    def test_valid_payload_roundtrip(self):
        server = _Server(json.dumps(_payload()).encode())
        self.addCleanup(server.close)
        payload = fetch_capability_snapshot(server.url, attempts=1)
        self.assertEqual(payload["registry"]["active"]["hand_id"],
                         "qiangnao-1-right")

    def test_unreachable_raises_with_hint(self):
        with self.assertRaisesRegex(CapabilityUnavailable, "capability.sh"):
            fetch_capability_snapshot(
                "http://127.0.0.1:1", attempts=1, timeout_s=0.5)

    def test_not_ok_or_malformed_payload_raises(self):
        for body in (json.dumps({"ok": False}).encode(),
                     json.dumps({"ok": True, "registry": []}).encode()):
            server = _Server(body)
            self.addCleanup(server.close)
            with self.assertRaisesRegex(CapabilityUnavailable, "返回异常"):
                fetch_capability_snapshot(server.url, attempts=1)

    def test_bad_json_raises_unavailable(self):
        server = _Server(b"<html>oops</html>")
        self.addCleanup(server.close)
        with self.assertRaises(CapabilityUnavailable):
            fetch_capability_snapshot(server.url, attempts=1)


class DescribeActiveTests(unittest.TestCase):
    def test_active_combo_line(self):
        line = describe_active(_payload())
        self.assertIn("右臂", line)
        self.assertIn("强脑-1-右", line)
        self.assertIn("ready", line)

    def test_no_active_combo(self):
        payload = _payload()
        payload["registry"]["active"] = None
        self.assertIn("未设置激活组合", describe_active(payload))


if __name__ == "__main__":
    unittest.main()
