from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.offline import (
    EpisodeValidationError,
    OfflineEpisodeBackend,
    PointCloudStaleError,
)


class _FakeRobotModel:
    def __init__(self, _config):
        pass

    @staticmethod
    def base_link(_chain):
        return "torso_link"

    @staticmethod
    def end_link(_chain):
        return "right_wrist_yaw_link"

    @staticmethod
    def joint_names(_chain):
        return list(RIGHT_ARM_TEST_JOINTS)

    @staticmethod
    def forward_kinematics(_joint_values, only_links):
        return {link: np.eye(4) for link in only_links}


RIGHT_ARM_TEST_JOINTS = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


def _stream(width: int, height: int, *, fx: float, fy: float, cx: float, cy: float):
    return {
        "width": width,
        "height": height,
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
        "distortion": {
            "model": "brown_conrady",
            "coefficient_order": ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
            "coefficients": [0.0] * 8,
        },
    }


def _write_calibration(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "device": {"serial": "TEST"},
        "color": _stream(4, 3, fx=100.0, fy=100.0, cx=1.0, cy=1.0),
        "depth": _stream(4, 3, fx=100.0, fy=100.0, cx=1.0, cy=1.0),
        "depth_to_color": {
            "rotation_row_major": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
            "translation_unit": "mm",
        },
        "depth_scale": {"value": 1.0, "unit": "mm_per_raw_unit"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_episode(
    task_dir: Path,
    name: str,
    depths: list[int],
    *,
    dtype=np.uint16,
    camera_serial: str = "TEST",
    arm: str = "right",
) -> Path:
    root = task_dir / name
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rows = []
    for index, depth_mm in enumerate(depths):
        image = np.full((3, 4, 3), 20 + index, dtype=np.uint8)
        cv2.imwrite(str(root / "rgb" / f"{index:06d}_head_rgb.jpg"), image)
        np.save(
            root / "depth" / f"{index:06d}_head_depth.npy",
            np.full((3, 4), depth_mm, dtype=dtype),
            allow_pickle=False,
        )
        rows.append(
            {
                "idx": index,
                "colors": {"head_rgb": f"rgb/{index:06d}_head_rgb.jpg"},
                "depths": {"head_depth": f"depth/{index:06d}_head_depth.npy"},
                "states": {f"{arm}_arm": {"qpos": [index * 0.01] * 7}},
                "timestamps": {"sample_timestamp_ns": 1000 + index},
                "rgbd": {
                    "color_shape": [3, 4],
                    "depth_shape": [3, 4],
                    "color_format": "jpeg",
                    "depth_format": "depth_z16",
                    "depth_dtype": "uint16",
                },
            }
        )
    payload = {
        "info": {
            "version": "1.0.0",
            "kind": "hand_eye_calibration",
            "frame_count": len(rows),
            "robot": "H2",
            "camera_serial": camera_serial,
            f"{arm}_arm_joint_order": [
                name.replace("right_", f"{arm}_", 1)
                for name in RIGHT_ARM_TEST_JOINTS
            ],
        },
        "data": rows,
    }
    (root / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


class OfflineEpisodeBackendTest(unittest.TestCase):
    def setUp(self):
        self.robot_model_patch = patch("backend.offline.RobotModel", _FakeRobotModel)
        self.robot_config_patch = patch(
            "backend.offline.load_robot_config", return_value={}
        )
        self.robot_model_patch.start()
        self.robot_config_patch.start()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.calibration_path = self.root / "rgbd.json"
        _write_calibration(self.calibration_path)

    def tearDown(self):
        self.tempdir.cleanup()
        self.robot_config_patch.stop()
        self.robot_model_patch.stop()

    def test_parse_preview_align_backproject_median_q_and_fk(self):
        _write_episode(
            self.root, "episode_0001", [980, 1000, 1020, 1010, 990]
        )
        backend = OfflineEpisodeBackend(self.root, self.calibration_path)

        episodes = backend.scan()
        self.assertEqual([item["name"] for item in episodes], ["episode_0001"])
        self.assertTrue(episodes[0]["valid"])
        self.assertEqual(episodes[0]["representative_frame"], 2)
        self.assertGreater(len(backend.preview_jpeg("episode_0001")), 0)

        result = backend.pick("episode_0001", 2, 1)
        np.testing.assert_allclose(result["p_camera"], [0.01, 0.0, 1.0], atol=1e-8)
        np.testing.assert_allclose(result["right_arm_q_median"], [0.02] * 7)
        self.assertEqual(result["valid_depth_frames"], 5)
        self.assertEqual(result["depth_spread_mm"], 40.0)
        self.assertEqual(result["base_link"], "torso_link")
        self.assertEqual(result["wrist_link"], "right_wrist_yaw_link")
        T = np.asarray(result["T_base_wrist"])
        self.assertEqual(T.shape, (4, 4))
        np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-10)

    def test_mount_rgb_candidate_maps_circle_center_to_cloud_vertex(self):
        _write_episode(
            self.root, "episode_0008", [980, 1000, 1020, 1010, 990]
        )
        backend = OfflineEpisodeBackend(self.root, self.calibration_path)
        detected = [
            {
                "id": "marker-red-01",
                "color": "red",
                "center": [2.0, 1.0],
                "radius_px": 3.0,
                "confidence": 0.95,
                "color_confidence": 0.96,
                "circularity": 0.9,
                "source": "auto",
                "flags": [],
            }
        ]

        with patch(
            "backend.offline.detect_mount_markers_bgr",
            return_value=detected,
        ):
            result = backend.detect_mount_candidates("episode_0008", stride=1)

        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["counts"], {"red": 1, "green": 0})
        candidate = result["candidates"][0]
        self.assertEqual(candidate["candidate_id"], "marker-red-01")
        self.assertEqual(candidate["pixel"], [2, 1])
        np.testing.assert_allclose(candidate["p_camera"], [0.01, 0.0, 1.0])
        cloud = backend.point_cloud("episode_0008", 1)
        self.assertEqual(
            cloud.pixels[candidate["vertex_index"]].tolist(),
            [2, 1],
        )

    def test_serial_mismatch_is_visible_warning_not_validation_error(self):
        _write_episode(
            self.root,
            "episode_0006",
            [980, 1000, 1020, 1010, 990],
            camera_serial="OTHER-CAMERA",
        )
        backend = OfflineEpisodeBackend(self.root, self.calibration_path)

        episodes = backend.scan()
        self.assertTrue(episodes[0]["valid"])
        self.assertEqual(episodes[0]["calibration_serial"], "TEST")
        self.assertIn("OTHER-CAMERA", episodes[0]["warnings"][0])
        self.assertIn("TEST", episodes[0]["warnings"][0])

        pick = backend.pick("episode_0006", 2, 1)
        self.assertTrue(pick["warnings"])
        self.assertEqual(pick["p_camera"][2], 1.0)

    def test_left_arm_episode_backend_uses_left_dataset_schema(self):
        _write_episode(
            self.root,
            "episode_0007",
            [980, 1000, 1020, 1010, 990],
            arm="left",
        )
        backend = OfflineEpisodeBackend(
            self.root, self.calibration_path, arm="left"
        )

        result = backend.pick("episode_0007", 2, 1)
        self.assertEqual(backend.arm, "left")
        self.assertIn("left_arm_q_median", result)
        self.assertNotIn("right_arm_q_median", result)

    def test_rejects_unstable_depth(self):
        _write_episode(
            self.root, "episode_0002", [900, 950, 1000, 1050, 1100]
        )
        backend = OfflineEpisodeBackend(self.root, self.calibration_path)
        with self.assertRaisesRegex(EpisodeValidationError, "跳动 200mm"):
            backend.pick("episode_0002", 2, 1)
        with self.assertRaisesRegex(EpisodeValidationError, "没有可用于点云"):
            backend.point_cloud("episode_0002")

    def test_confirm_many_aligns_each_depth_once_and_keeps_valid_markers(self):
        _write_episode(
            self.root, "episode_0004", [980, 1000, 1020, 1010, 990]
        )
        backend = OfflineEpisodeBackend(self.root, self.calibration_path)
        real_aligner = backend.aligner

        class CountingAligner:
            def __init__(self):
                self.calls = 0

            def align(self, depth):
                self.calls += 1
                return real_aligner.align(depth)

        counting = CountingAligner()
        backend.aligner = counting
        result = backend.confirm_markers(
            "episode_0004",
            [
                {
                    "id": "marker-red",
                    "color": "red",
                    "center": [2, 1],
                    "radius_px": 12,
                    "source": "edited",
                },
                {
                    "id": "marker-blue",
                    "color": "blue",
                    "center": [99, 99],
                    "radius_px": 12,
                    "source": "edited",
                },
            ],
        )
        self.assertEqual(counting.calls, 5)
        self.assertEqual(result["confirmed_count"], 1)
        self.assertEqual(result["error_count"], 1)
        observation = result["observations"][0]
        self.assertEqual(observation["schema_version"], 2)
        self.assertEqual(observation["marker_id"], "marker-red")
        self.assertEqual(observation["color"], "red")
        self.assertEqual(observation["pose_id"], "episode_0004")
        np.testing.assert_allclose(observation["p_camera"], [0.01, 0.0, 1.0])
        self.assertIn("像素越界", result["errors"][0]["error"])

        with self.assertRaisesRegex(EpisodeValidationError, "canonical color"):
            backend.confirm_markers(
                "episode_0004",
                [
                    {
                        "id": "red-a",
                        "color": "red",
                        "center": [1, 1],
                        "radius_px": 10,
                        "source": "edited",
                    },
                    {
                        "id": "red-b",
                        "color": "red",
                        "center": [2, 1],
                        "radius_px": 10,
                        "source": "edited",
                    },
                ],
            )
        self.assertEqual(counting.calls, 5)

    def test_point_cloud_is_deterministic_and_confirms_vertex_indices(self):
        _write_episode(
            self.root, "episode_0005", [980, 1000, 1020, 1010, 990]
        )
        backend = OfflineEpisodeBackend(self.root, self.calibration_path)

        cloud = backend.point_cloud("episode_0005", stride=2)
        self.assertEqual(cloud.stride, 2)
        self.assertEqual(cloud.points.shape, (4, 3))
        np.testing.assert_array_equal(
            cloud.pixels,
            [[0, 0], [2, 0], [0, 2], [2, 2]],
        )
        np.testing.assert_allclose(
            cloud.points,
            [
                [-0.01, -0.01, 1.0],
                [0.01, -0.01, 1.0],
                [-0.01, 0.01, 1.0],
                [0.01, 0.01, 1.0],
            ],
            atol=1e-7,
        )
        np.testing.assert_array_equal(cloud.valid_depth_frames, [5, 5, 5, 5])
        np.testing.assert_allclose(cloud.depth_spread_mm, [40.0] * 4)

        ply, same_cloud = backend.point_cloud_ply("episode_0005", stride=2)
        self.assertEqual(same_cloud.cloud_id, cloud.cloud_id)
        self.assertIn(b"format binary_little_endian 1.0", ply)
        self.assertIn(b"element vertex 4", ply)

        result = backend.confirm_points(
            "episode_0005",
            cloud.cloud_id,
            2,
            [{"id": "marker-red", "color": "red", "vertex_index": 1}],
        )
        self.assertEqual(result["confirmed_count"], 1)
        observation = result["observations"][0]
        self.assertEqual(observation["schema_version"], 2)
        self.assertEqual(observation["source"], "point_cloud")
        self.assertEqual(observation["pixel"], [2, 0])
        self.assertEqual(observation["provenance"]["mode"], "offline_point_cloud")
        np.testing.assert_allclose(observation["p_camera"], [0.01, -0.01, 1.0])

        with self.assertRaises(PointCloudStaleError):
            backend.confirm_points(
                "episode_0005",
                "stale-cloud-id",
                2,
                [{"id": "marker-red", "color": "red", "vertex_index": 1}],
            )
        with self.assertRaisesRegex(EpisodeValidationError, "越界"):
            backend.confirm_points(
                "episode_0005",
                cloud.cloud_id,
                2,
                [{"id": "marker-red", "color": "red", "vertex_index": 99}],
            )

    def test_scan_reports_non_uint16_depth(self):
        _write_episode(
            self.root, "episode_0003", [1000, 1000, 1000], dtype=np.float32
        )
        backend = OfflineEpisodeBackend(self.root, self.calibration_path)
        episodes = backend.scan()
        self.assertFalse(episodes[0]["valid"])
        self.assertIn("原始 uint16", episodes[0]["error"])


class SampleProvenanceTest(unittest.TestCase):
    def test_preserves_provenance_and_rejects_duplicate_episode(self):
        from backend import app as app_module

        old_save_path = app_module.save_path
        old_offline_backend = app_module.offline_backend
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                app_module.save_path = Path(tempdir)
                app_module.offline_backend = SimpleNamespace(
                    base_link="torso_link",
                    wrist_link="right_wrist_yaw_link",
                    calibration=SimpleNamespace(
                        color_shape=(1080, 1920),
                        serial="TEST",
                    ),
                )
                app_module.init_state()
                status = asyncio.run(app_module.api_status())
                self.assertEqual(status["mode"], "offline")
                self.assertEqual(status["base_link"], "torso_link")
                self.assertEqual(status["wrist_link"], "right_wrist_yaw_link")
                self.assertEqual(status["camera"]["serial"], "TEST")
                body = {
                    "p_camera": [0.0, 0.0, 1.0],
                    "T_base_wrist": np.eye(4).tolist(),
                    "pixel": [10, 20],
                    "episode": "episode_0042",
                    "provenance": {"mode": "offline_teleop_episode"},
                }
                first = asyncio.run(app_module.api_add_sample(body))
                self.assertTrue(first["ok"])
                saved = json.loads(
                    (Path(tempdir) / "samples" / "0000.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved["episode"], "episode_0042")
                self.assertEqual(saved["provenance"], body["provenance"])
                self.assertEqual(saved["pose_source"], "offline_teleop_episode")
                self.assertEqual(saved["camera"]["source"], "offline_teleop_episode")

                duplicate = asyncio.run(app_module.api_add_sample(body))
                self.assertEqual(duplicate.status_code, 409)
                payload = json.loads(duplicate.body)
                self.assertEqual(payload["duplicate_episode"], "episode_0042")
                self.assertEqual(payload["existing_index"], 0)
            finally:
                app_module.save_path = old_save_path
                app_module.offline_backend = old_offline_backend


class OrphanedOfflineSamplesTest(unittest.TestCase):
    def test_deleted_episode_samples_are_excluded_with_live_episode_backend(self):
        from backend import app as app_module

        old_save_path = app_module.save_path
        old_offline_backend = app_module.offline_backend
        old_episode_backend = app_module.episode_backend
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                app_module.save_path = Path(tempdir)
                app_module.offline_backend = None
                app_module.episode_backend = SimpleNamespace(
                    episode_names=lambda: {"episode_0001"}
                )
                app_module.init_state()
                samples_dir = Path(tempdir) / "samples"
                (samples_dir / "0000.json").write_text(
                    json.dumps({"index": 0, "episode": "episode_0001"}),
                    encoding="utf-8",
                )
                (samples_dir / "0001.json").write_text(
                    json.dumps({"index": 1, "episode": "episode_0002"}),
                    encoding="utf-8",
                )

                loaded = app_module._load_samples()
                self.assertEqual([sample["index"] for sample in loaded], [0])
            finally:
                app_module.save_path = old_save_path
                app_module.offline_backend = old_offline_backend
                app_module.episode_backend = old_episode_backend


class BatchSampleApiTest(unittest.TestCase):
    @staticmethod
    def _observation(marker_id: str, color: str, episode: str) -> dict:
        return {
            "schema_version": 2,
            "id": marker_id,
            "marker_id": marker_id,
            "color": color,
            "episode": episode,
            "pose_id": episode,
            "center": [20.0, 30.0],
            "radius_px": 14.0,
            "source": "edited",
            "p_camera": [0.0, 0.0, 1.0],
            "T_base_wrist": np.eye(4).tolist(),
            "provenance": {"mode": "offline_teleop_episode", "episode": episode},
        }

    def test_batch_allows_colors_and_rejects_duplicate_key_atomically(self):
        from backend import app as app_module

        old_save_path = app_module.save_path
        old_offline_backend = app_module.offline_backend
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                app_module.save_path = Path(tempdir)
                app_module.offline_backend = None
                app_module.init_state()
                episode = "episode_0100"
                body = {
                    "episode": episode,
                    "observations": [
                        self._observation("marker-red", "red", episode),
                        self._observation("marker-blue", "blue", episode),
                    ],
                }
                saved = asyncio.run(app_module.api_add_samples_batch(body))
                self.assertTrue(saved["ok"])
                self.assertEqual(saved["indices"], [0, 1])
                records = app_module._load_samples()
                self.assertEqual([record["schema_version"] for record in records], [2, 2])
                self.assertEqual(
                    {(record["episode"], record["marker_id"]) for record in records},
                    {(episode, "marker-red"), (episode, "marker-blue")},
                )
                app_module.offline_backend = SimpleNamespace(
                    scan=lambda: [{"name": episode, "valid": True}]
                )
                listing = asyncio.run(app_module.api_offline_episodes())
                self.assertEqual(
                    listing["episodes"][0]["imported_marker_ids"],
                    ["marker-blue", "marker-red"],
                )
                self.assertEqual(
                    listing["episodes"][0]["imported_marker_count"], 2
                )
                app_module.offline_backend = None

                duplicate = asyncio.run(
                    app_module.api_add_samples_batch(
                        {
                            "episode": episode,
                            "observations": [
                                self._observation("marker-red", "red", episode)
                            ],
                        }
                    )
                )
                self.assertEqual(duplicate.status_code, 409)
                self.assertEqual(len(app_module._load_samples()), 2)

                invalid_episode = "episode_0101"
                invalid = asyncio.run(
                    app_module.api_add_samples_batch(
                        {
                            "episode": invalid_episode,
                            "observations": [
                                self._observation("red-a", "red", invalid_episode),
                                self._observation("red-b", "red", invalid_episode),
                            ],
                        }
                    )
                )
                self.assertEqual(invalid.status_code, 400)
                self.assertEqual(len(app_module._load_samples()), 2)
            finally:
                app_module.save_path = old_save_path
                app_module.offline_backend = old_offline_backend


class LiveEpisodeRecorderTest(unittest.TestCase):
    def test_records_sdk_frames_in_offline_compatible_layout(self):
        from backend import app as app_module

        class FakeCamera:
            source = "orbbec"

            def __init__(self):
                self.sequence = 0

            @staticmethod
            def info():
                return {
                    "source": "orbbec",
                    "serial": "TEST",
                    "recording_supported": True,
                }

            def wait_record_frame(self, _after_sequence, timeout_s=2.0):
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

        old_values = {
            "camera": app_module.camera,
            "pose_provider": app_module.pose_provider,
            "offline_backend": app_module.offline_backend,
            "record_task_dir": app_module.record_task_dir,
            "rgbd_calib_path": app_module.rgbd_calib_path,
            "save_path": app_module.save_path,
        }
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            calibration_path = root / "rgbd.json"
            _write_calibration(calibration_path)
            try:
                app_module.camera = FakeCamera()
                app_module.pose_provider = SimpleNamespace(
                    source="h2",
                    available=True,
                    base_link="torso_link",
                    wrist_link="right_wrist_yaw_link",
                    read_arm_q=lambda: np.arange(7, dtype=float),
                )
                app_module.offline_backend = None
                app_module.record_task_dir = root / "task"
                app_module.rgbd_calib_path = calibration_path
                app_module.save_path = root / "results"
                app_module.init_state()

                result = app_module._record_episode(5)
                self.assertEqual(result["episode"], "episode_0000")
                episode_root = root / "task" / "episode_0000"
                payload = json.loads(
                    (episode_root / "data.json").read_text(encoding="utf-8")
                )
                self.assertEqual(payload["info"]["kind"], "hand_eye_calibration")
                self.assertEqual(payload["info"]["frame_count"], 5)
                self.assertEqual(len(payload["data"]), 5)
                self.assertEqual(
                    payload["data"][0]["states"]["right_arm"]["qpos"],
                    np.arange(7, dtype=float).tolist(),
                )
                depth = np.load(
                    episode_root / payload["data"][0]["depths"]["head_depth"],
                    allow_pickle=False,
                )
                self.assertEqual(depth.dtype, np.uint16)
                self.assertEqual(depth.shape, (3, 4))
                self.assertTrue(
                    (episode_root / payload["data"][0]["colors"]["head_rgb"]).is_file()
                )
            finally:
                for name, value in old_values.items():
                    setattr(app_module, name, value)


if __name__ == "__main__":
    unittest.main()
