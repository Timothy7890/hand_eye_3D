from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import app as app_module


def _stream(width: int, height: int) -> dict:
    return {
        "width": width,
        "height": height,
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 1.0,
            "cy": 1.0,
        },
        "distortion": {
            "model": "brown_conrady",
            "coefficient_order": [
                "k1",
                "k2",
                "p1",
                "p2",
                "k3",
                "k4",
                "k5",
                "k6",
            ],
            "coefficients": [0.0] * 8,
        },
    }


def _write_calibration(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "device": {"serial": "TEST"},
        "color": _stream(4, 3),
        "depth": _stream(4, 3),
        "depth_to_color": {
            "rotation_row_major": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
            "translation_unit": "mm",
        },
        "depth_scale": {"value": 1.0, "unit": "mm_per_raw_unit"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class _FakeCamera:
    def __init__(self):
        self.sequence = 0

    @staticmethod
    def info() -> dict:
        return {
            "source": "orbbec",
            "serial": "TEST",
            "recording_supported": True,
        }

    def wait_record_frame(self, _after_sequence: int, timeout_s: float = 2.0) -> dict:
        del timeout_s
        sequence = self.sequence
        self.sequence += 1
        return {
            "sequence": sequence,
            "timestamp_ns": 1000 + sequence,
            "color_bgr": np.full((3, 4, 3), 20 + sequence, np.uint8),
            "depth_z16": np.full((3, 4), 900 + sequence, np.uint16),
            "depth_scale_mm": 1.0,
        }


class _FakePoseProvider:
    source = "h2"
    available = True
    base_link = "torso_link"
    wrist_link = "wrist_yaw_link"

    def __init__(self):
        self.read_count = 0

    def read_arm_q(self) -> np.ndarray:
        q = np.arange(7, dtype=float) + self.read_count * 0.01
        self.read_count += 1
        return q


class ReplayEpisodeRecorderTest(unittest.TestCase):
    def setUp(self):
        self.old_values = {
            "camera": app_module.camera,
            "pose_provider": app_module.pose_provider,
            "offline_backend": app_module.offline_backend,
            "record_task_dir": app_module.record_task_dir,
            "rgbd_calib_path": app_module.rgbd_calib_path,
            "save_path": app_module.save_path,
            "arm_side": app_module.arm_side,
            "arm_factory": app_module.arm_factory,
            "arm_controller": app_module.arm_controller,
        }
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        calibration_path = root / "rgbd.json"
        _write_calibration(calibration_path)
        self.camera = _FakeCamera()
        self.pose_provider = _FakePoseProvider()
        app_module.camera = self.camera
        app_module.pose_provider = self.pose_provider
        app_module.offline_backend = None
        app_module.record_task_dir = root / "task"
        app_module.rgbd_calib_path = calibration_path
        app_module.save_path = root / "results"
        app_module.arm_side = "right"
        app_module.arm_factory = None
        app_module.arm_controller = None
        app_module.init_state()

    def tearDown(self):
        for name, value in self.old_values.items():
            setattr(app_module, name, value)
        self.tempdir.cleanup()

    def _payload(self, episode: str) -> dict:
        assert app_module.record_task_dir is not None
        return json.loads(
            (app_module.record_task_dir / episode / "data.json").read_text(
                encoding="utf-8"
            )
        )

    def test_metadata_and_durable_capture_id_idempotency(self):
        stability = {
            "stable": True,
            "window_s": 0.8,
            "max_error_rad": 0.009,
            "samples": 20,
        }
        request = {
            "frame_count": 3,
            "run_id": "robot-07.run-20260903",
            "waypoint_id": "sample:head-01",
            "capture_id": "robot-07.run-20260903:sample-head-01",
            "target_q_rad": [0.1] * 7,
            "stability": stability,
        }
        first = asyncio.run(app_module.api_record_episode(request))

        self.assertTrue(first["ok"])
        self.assertFalse(first["idempotent_replay"])
        self.assertEqual(self.camera.sequence, 3)
        payload = self._payload(first["episode"])
        info = payload["info"]
        self.assertEqual(info["run_id"], request["run_id"])
        self.assertEqual(info["waypoint_id"], request["waypoint_id"])
        self.assertEqual(info["capture_id"], request["capture_id"])
        self.assertEqual(info["target_q_rad"], request["target_q_rad"])
        self.assertEqual(info["stability"], stability)
        self.assertEqual(info["measured_q_rad"], (np.arange(7) + 0.02).tolist())
        self.assertEqual(info["measured_q_summary"]["sample_count"], 3)
        np.testing.assert_allclose(
            info["measured_q_summary"]["range_rad"], [0.02] * 7
        )

        duplicate = asyncio.run(
            app_module.api_record_episode(
                {"frame_count": 3, "capture_id": request["capture_id"]}
            )
        )
        self.assertTrue(duplicate["idempotent_replay"])
        self.assertEqual(duplicate["episode"], first["episode"])
        self.assertEqual(self.camera.sequence, 3)
        assert app_module.record_task_dir is not None
        self.assertEqual(
            len(list(app_module.record_task_dir.glob("episode_*/data.json"))), 1
        )

    def test_left_arm_episode_uses_left_schema(self):
        app_module.arm_side = "left"

        result = asyncio.run(app_module.api_record_episode({"frame_count": 3}))
        payload = self._payload(result["episode"])
        info = payload["info"]

        self.assertEqual(info["arm"], "left")
        self.assertEqual(info["arm_state_key"], "left_arm")
        self.assertEqual(
            info["left_arm_joint_order"],
            [
                "left_shoulder_pitch",
                "left_shoulder_roll",
                "left_shoulder_yaw",
                "left_elbow",
                "left_wrist_roll",
                "left_wrist_pitch",
                "left_wrist_yaw",
            ],
        )
        self.assertNotIn("right_arm_joint_order", info)
        for row in payload["data"]:
            self.assertIn("left_arm", row["states"])
            self.assertNotIn("right_arm", row["states"])

    def test_request_validation_and_capture_only_status(self):
        invalid_id = asyncio.run(
            app_module.api_record_episode(
                {"frame_count": 3, "capture_id": "../unsafe"}
            )
        )
        self.assertEqual(invalid_id.status_code, 400)
        invalid_vector = asyncio.run(
            app_module.api_record_episode(
                {"frame_count": 3, "target_q_rad": [0.0] * 6}
            )
        )
        self.assertEqual(invalid_vector.status_code, 400)

        legacy = asyncio.run(app_module.api_record_episode())
        self.assertTrue(legacy["ok"])
        self.assertEqual(legacy["frame_count"], 5)

        status = asyncio.run(app_module.api_arm_status())
        self.assertEqual(
            status,
            {
                "enabled": False,
                "armed": False,
                "publishing": False,
                "arm": "right",
                "mode": "capture_only",
            },
        )


if __name__ == "__main__":
    unittest.main()
