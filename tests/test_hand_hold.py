from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend import app as app_module
from backend.hand_hold import HandHoldController, HandHoldError


class HandHoldControllerTest(unittest.TestCase):
    def _controller(self, fake):
        controller = HandHoldController(lambda: "https://127.0.0.1:18089")
        patcher = patch("backend.hand_hold._request_json", side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(controller.stop)
        return controller

    def test_start_holds_and_stop_terminates(self):
        calls = []
        first_command = threading.Event()

        def fake(url, payload=None, timeout=3.0):
            calls.append((url, payload))
            if url.endswith("/api/command"):
                first_command.set()
            return {"ok": True}

        controller = self._controller(fake)
        state = controller.start("brainco_revo2", "right")
        self.assertTrue(state["running"])
        self.assertTrue(first_command.wait(2.0))

        self.assertEqual(calls[0][0], "https://127.0.0.1:18089/api/connect")
        self.assertEqual(calls[0][1], {"device_id": "brainco_revo2"})
        command = next(
            payload for url, payload in calls if url.endswith("/api/command")
        )
        self.assertEqual(command["side"], "right")
        self.assertEqual(command["positions"], [0.0] * 6)
        self.assertTrue(command["continuous"])

        deadline = time.time() + 2.0
        while time.time() < deadline and controller.status()["sent_count"] == 0:
            time.sleep(0.02)
        self.assertGreaterEqual(controller.status()["sent_count"], 1)

        stopped = controller.stop()
        self.assertFalse(stopped["running"])
        self.assertFalse(controller.status()["running"])

    def test_start_idempotent_same_target_conflict_other(self):
        def fake(url, payload=None, timeout=3.0):
            return {"ok": True}

        controller = self._controller(fake)
        controller.start("brainco_revo2", "right")
        again = controller.start("brainco_revo2", "right")
        self.assertTrue(again["running"])
        with self.assertRaises(HandHoldError):
            controller.start("inspire_dfx", "right")

    def test_command_error_recorded_not_fatal(self):
        def fake(url, payload=None, timeout=3.0):
            if url.endswith("/api/command"):
                raise HandHoldError("18089: 手动控制正在控制灵巧手")
            return {"ok": True}

        controller = self._controller(fake)
        controller.start("brainco_revo2", "right")
        deadline = time.time() + 2.0
        while time.time() < deadline and controller.status()["error_count"] == 0:
            time.sleep(0.02)
        status = controller.status()
        self.assertTrue(status["running"])
        self.assertGreaterEqual(status["error_count"], 1)
        self.assertIn("手动控制", status["last_error"])

    def test_connect_failure_raises_and_stays_stopped(self):
        def fake(url, payload=None, timeout=3.0):
            raise HandHoldError("18089 不可达: connection refused")

        controller = self._controller(fake)
        with self.assertRaises(HandHoldError):
            controller.start("brainco_revo2", "right")
        self.assertFalse(controller.status()["running"])

    def test_input_validation(self):
        controller = HandHoldController(lambda: "https://127.0.0.1:18089")
        with self.assertRaises(ValueError):
            controller.start("", "right")
        with self.assertRaises(ValueError):
            controller.start("brainco_revo2", "up")


class HandHoldApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)
        self.old_hold = app_module.hand_hold

    def tearDown(self):
        app_module.hand_hold = self.old_hold

    def test_start_derives_device_and_side_from_hand_id(self):
        recorded = {}

        class FakeHold:
            def start(self, device_id, side):
                recorded["args"] = (device_id, side)
                return {"running": True, "device_id": device_id, "side": side}

        app_module.hand_hold = FakeHold()
        response = self.client.post(
            "/api/mount/hand-hold/start", json={"hand_id": "qiangnao-1-right"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(recorded["args"], ("brainco_revo2", "right"))
        self.assertTrue(response.json()["hold"]["running"])

    def test_start_unknown_hand_id_404(self):
        response = self.client.post(
            "/api/mount/hand-hold/start", json={"hand_id": "no-such-hand"}
        )
        self.assertEqual(response.status_code, 404)

    def test_start_requires_device(self):
        response = self.client.post("/api/mount/hand-hold/start", json={})
        self.assertEqual(response.status_code, 400)

    def test_start_rejects_bad_side(self):
        response = self.client.post(
            "/api/mount/hand-hold/start",
            json={"device_id": "brainco_revo2", "side": "up"},
        )
        self.assertEqual(response.status_code, 400)

    def test_start_conflict_maps_to_409(self):
        class FakeHold:
            def start(self, device_id, side):
                raise HandHoldError("18089: 视觉控制正在控制灵巧手")

        app_module.hand_hold = FakeHold()
        response = self.client.post(
            "/api/mount/hand-hold/start",
            json={"device_id": "brainco_revo2", "side": "right"},
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("视觉控制", response.json()["error"])

    def test_status_and_stop_passthrough(self):
        class FakeHold:
            def status(self):
                return {"running": False, "sent_count": 3}

            def stop(self):
                return {"running": False}

        app_module.hand_hold = FakeHold()
        status = self.client.get("/api/mount/hand-hold")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["hold"]["sent_count"], 3)
        stop = self.client.post("/api/mount/hand-hold/stop")
        self.assertEqual(stop.status_code, 200)
        self.assertFalse(stop.json()["hold"]["running"])


if __name__ == "__main__":
    unittest.main()
