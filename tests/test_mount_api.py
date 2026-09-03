from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from fastapi.testclient import TestClient

from backend import app as app_module
from backend.hands import get_hand_model
from backend.offline import OfflineEpisodeBackend
from backend.solver import rpy_to_rot
from test_offline import _FakeRobotModel, _write_calibration, _write_episode

HAND_ID = "yinshi-1-right"
MOUNT_POINT_IDS = [
    "palm-red-01",
    "palm-red-02",
    "palm-red-03",
    "back-green-01",
    "back-green-02",
]
PROFILE_POINT_IDS = [
    *[f"palm-red-{index:02d}" for index in range(1, 9)],
    *[f"back-green-{index:02d}" for index in range(1, 9)],
]


def _make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


class MountApiTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.old_save_path = app_module.save_path
        self.old_offline = app_module.offline_backend
        self.old_episode = app_module.episode_backend
        self.old_mount_calib = app_module.mount_calib_path
        self.old_mount_profile_dir = app_module.mount_profile_dir
        self.old_pose_provider = app_module.pose_provider
        self.old_arm_side = app_module.arm_side
        self.old_camera = app_module.camera
        app_module.save_path = self.root / "session"
        app_module.offline_backend = None
        app_module.episode_backend = None
        app_module.mount_calib_path = None
        app_module.mount_profile_dir = self.root / "mount_model_profiles"
        app_module.arm_side = "right"
        app_module.pose_provider = SimpleNamespace(
            source="mock",
            available=False,
            base_link="torso_link",
            wrist_link="right_wrist_yaw_link",
        )
        app_module.camera = SimpleNamespace(
            info=lambda: {
                "source": "mock",
                "serial": "CP0T263000BE",
                "name": "Orbbec Gemini 335",
                "width": 1920,
                "height": 1080,
            }
        )
        self.capability = {
            "ok": True,
            "available": True,
            "active": {"arm": "right_arm", "hand_id": HAND_ID},
            "arm": "right",
            "hand_id": HAND_ID,
            "hand_name": "因时-右-1",
        }
        self.capability_patch = patch.object(
            app_module,
            "get_capability_hint",
            new=AsyncMock(side_effect=lambda: dict(self.capability)),
        )
        self.capability_patch.start()
        app_module.init_state()
        self.client = TestClient(app_module.app)

    def tearDown(self):
        self.capability_patch.stop()
        app_module.save_path = self.old_save_path
        app_module.offline_backend = self.old_offline
        app_module.episode_backend = self.old_episode
        app_module.mount_calib_path = self.old_mount_calib
        app_module.mount_profile_dir = self.old_mount_profile_dir
        app_module.pose_provider = self.old_pose_provider
        app_module.arm_side = self.old_arm_side
        app_module.camera = self.old_camera
        self.tempdir.cleanup()

    # ---------- 手型号目录 ----------

    def test_hands_catalog_and_model_endpoint(self):
        listed = self.client.get("/api/hands").json()
        self.assertTrue(listed["ok"])
        self.assertEqual(listed["count"], 4)
        ids = {hand["hand_id"] for hand in listed["hands"]}
        self.assertIn(HAND_ID, ids)
        self.assertIn("qiangnao-1-left", ids)

        model = self.client.get(f"/api/hands/{HAND_ID}/model").json()
        self.assertTrue(model["ok"])
        self.assertEqual(model["base_link"], "R_hand_base_link")
        self.assertEqual(len(model["actuated_joints"]), 6)
        tips = [p for p in model["feature_points"] if p["source"] == "tip_link"]
        self.assertEqual(len(tips), 5)
        self.assertGreater(len(model["links"]), 10)
        mesh_url = model["links"][0]["visuals"][0]["mesh_url"]
        response = self.client.get(mesh_url)
        self.assertEqual(response.status_code, 200)

        missing = self.client.get("/api/hands/no-such-hand/model")
        self.assertEqual(missing.status_code, 404)

    def _model_profile_body(self, name: str = "生产手点位") -> dict:
        metadata = get_hand_model(HAND_ID).metadata()
        link = metadata["links"][0]["link"]
        return {
            "schema_version": 1,
            "name": name,
            "hand_id": HAND_ID,
            "points": [
                {
                    "point_id": point_id,
                    "label": f"模型点 {index + 1}",
                    "link": link,
                    "p_local": [index * 0.001, 0.01, -0.02],
                    "p_hand": [index * 0.001, 0.01, -0.02],
                }
                for index, point_id in enumerate(PROFILE_POINT_IDS)
            ],
        }

    def test_model_point_profile_crud_and_same_name_overwrite(self):
        self.assertEqual(
            self.client.get(
                f"/api/mount/model-point-profiles?hand_id={HAND_ID}"
            ).json()["profiles"],
            [],
        )

        created = self.client.post(
            "/api/mount/model-point-profiles",
            json=self._model_profile_body(),
        )
        self.assertEqual(created.status_code, 200, created.text)
        created_payload = created.json()
        self.assertTrue(created_payload["created"])
        profile = created_payload["profile"]
        profile_id = profile["profile_id"]
        self.assertEqual(len(profile_id), 64)
        self.assertEqual(
            [point["point_id"] for point in profile["points"]],
            PROFILE_POINT_IDS,
        )
        self.assertTrue(
            (app_module.mount_profile_dir / f"{profile_id}.json").is_file()
        )

        loaded = self.client.get(
            f"/api/mount/model-point-profiles/{profile_id}"
        )
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.json()["profile"], profile)

        replacement = self._model_profile_body()
        replacement["points"][0]["p_hand"] = [0.2, 0.1, 0.3]
        overwritten = self.client.post(
            "/api/mount/model-point-profiles", json=replacement
        )
        self.assertEqual(overwritten.status_code, 200, overwritten.text)
        overwritten_payload = overwritten.json()
        self.assertFalse(overwritten_payload["created"])
        self.assertEqual(overwritten_payload["profile"]["profile_id"], profile_id)
        self.assertEqual(
            overwritten_payload["profile"]["created_at"], profile["created_at"]
        )
        self.assertEqual(
            overwritten_payload["profile"]["points"][0]["p_hand"],
            [0.2, 0.1, 0.3],
        )

        listing = self.client.get(
            f"/api/mount/model-point-profiles?hand_id={HAND_ID}"
        ).json()
        self.assertEqual(listing["count"], 1)
        self.assertEqual(listing["profiles"][0]["profile_id"], profile_id)

        deleted = self.client.delete(
            f"/api/mount/model-point-profiles/{profile_id}"
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(
            self.client.get(
                f"/api/mount/model-point-profiles/{profile_id}"
            ).status_code,
            404,
        )

    def test_model_point_profile_rejects_invalid_content(self):
        partial = self._model_profile_body("未完成草稿")
        partial["points"] = partial["points"][:1]
        partial_response = self.client.post(
            "/api/mount/model-point-profiles", json=partial
        )
        self.assertEqual(partial_response.status_code, 200)
        partial_profile = partial_response.json()["profile"]
        self.assertEqual(partial_profile["point_count"], 1)
        self.assertFalse(partial_profile["complete"])

        empty = self._model_profile_body("空草稿")
        empty["points"] = []
        self.assertEqual(
            self.client.post(
                "/api/mount/model-point-profiles", json=empty
            ).status_code,
            400,
        )

        duplicate = self._model_profile_body()
        duplicate["points"][-1]["point_id"] = duplicate["points"][0]["point_id"]
        duplicate_response = self.client.post(
            "/api/mount/model-point-profiles", json=duplicate
        )
        self.assertEqual(duplicate_response.status_code, 400)
        self.assertIn("重复", duplicate_response.json()["error"])

        bad_link = self._model_profile_body()
        bad_link["points"][0]["link"] = "../../not-a-link"
        self.assertEqual(
            self.client.post(
                "/api/mount/model-point-profiles", json=bad_link
            ).status_code,
            400,
        )

        bad_vector = self._model_profile_body()
        bad_vector["points"][0]["p_hand"] = ["nan", 0.0, 0.0]
        self.assertEqual(
            self.client.post(
                "/api/mount/model-point-profiles", json=bad_vector
            ).status_code,
            400,
        )

        unknown_hand = self._model_profile_body()
        unknown_hand["hand_id"] = "no-such-hand"
        self.assertEqual(
            self.client.post(
                "/api/mount/model-point-profiles", json=unknown_hand
            ).status_code,
            404,
        )

    def test_model_point_profile_list_tolerates_bad_files(self):
        app_module.mount_profile_dir.mkdir(parents=True, exist_ok=True)
        (app_module.mount_profile_dir / "manual-name.json").write_text(
            "{}", encoding="utf-8"
        )
        (app_module.mount_profile_dir / f"{'a' * 64}.json").write_text(
            "{broken", encoding="utf-8"
        )

        response = self.client.get(
            f"/api/mount/model-point-profiles?hand_id={HAND_ID}"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["profiles"], [])
        self.assertEqual(payload["invalid_count"], 2)

    # ---------- 离线配对确认 ----------

    def test_confirm_mount_points_with_live_episode_backend(self):
        task_dir = self.root / "task"
        task_dir.mkdir()
        calib_path = self.root / "rgbd.json"
        _write_calibration(calib_path)
        _write_episode(task_dir, "episode_0001", [980, 1000, 1020, 1010, 990])
        with patch("backend.offline.RobotModel", _FakeRobotModel), patch(
            "backend.offline.load_robot_config", return_value={}
        ):
            app_module.episode_backend = OfflineEpisodeBackend(task_dir, calib_path)
        cloud = app_module.episode_backend.point_cloud("episode_0001", 2)

        p_hand = [0.013, -0.027, 0.006]
        body = {
            "episode": "episode_0001",
            "cloud_id": cloud.cloud_id,
            "stride": 2,
            "hand_id": HAND_ID,
            "selections": [
                {
                    "point_id": "palm-red-01",
                    "label": "手心自由点 1",
                    "link": "R_hand_base_link",
                    "p_hand": p_hand,
                    "vertex_index": 1,
                }
            ],
        }
        result = self.client.post("/api/offline/confirm-mount-points", json=body).json()
        self.assertTrue(result["ok"], result)
        observation = result["observations"][0]
        self.assertEqual(observation["schema_version"], 3)
        self.assertEqual(observation["hand_id"], HAND_ID)
        self.assertEqual(observation["point_id"], "palm-red-01")
        np.testing.assert_allclose(observation["p_hand"], p_hand)
        np.testing.assert_allclose(observation["p_camera"], [0.01, -0.01, 1.0])
        self.assertEqual(
            observation["provenance"]["mode"], "offline_point_cloud_mount"
        )

        # 非零手关节要求被拒绝
        bent = dict(body, hand_joints=[0.3] * 6)
        response = self.client.post("/api/offline/confirm-mount-points", json=bent)
        self.assertEqual(response.status_code, 400)
        self.assertIn("归零", response.json()["error"])

        # 未登记手型号
        unknown = dict(body, hand_id="no-such-hand")
        self.assertEqual(
            self.client.post("/api/offline/confirm-mount-points", json=unknown).status_code,
            404,
        )

    # ---------- 样本保存与解算 ----------

    def _synthetic_batch(self):
        """已知安装变换的合成观测：3 个 pose x 5 个指尖。"""
        model = get_hand_model(HAND_ID)
        tips = [p for p in model.feature_points() if p["source"] == "tip_link"]
        R_mount = rpy_to_rot(0.04, -0.1, 1.55)
        t_mount = np.array([0.012, -0.03, 0.08])
        T_mount = _make_T(R_mount, t_mount)
        T_cam2base = _make_T(np.eye(3), [0.0, 0.0, -0.5])
        poses = [
            _make_T(rpy_to_rot(0.0, 0.0, 0.0), [0.30, 0.10, 0.25]),
            _make_T(rpy_to_rot(0.2, 0.1, -0.3), [0.32, 0.05, 0.30]),
            _make_T(rpy_to_rot(-0.15, 0.25, 0.4), [0.28, 0.15, 0.22]),
        ]
        observations = []
        for pose_index, T_wrist in enumerate(poses):
            for point_id, tip in zip(MOUNT_POINT_IDS, tips):
                p_hand = np.array(tip["p_hand"])
                p_base = (T_wrist @ T_mount @ np.array([*p_hand, 1.0]))[:3]
                p_camera = T_cam2base[:3, :3].T @ (p_base - T_cam2base[:3, 3])
                observations.append(
                    {
                        "schema_version": 3,
                        "point_id": point_id,
                        "hand_id": HAND_ID,
                        "pose_id": f"episode_{pose_index:04d}",
                        "p_hand": p_hand.tolist(),
                        "p_camera": p_camera.tolist(),
                        "T_base_wrist": T_wrist.tolist(),
                        "base_link": "torso_link",
                        "wrist_link": "right_wrist_yaw_link",
                        "arm": "right",
                    }
                )
        return observations, T_mount, T_cam2base

    def test_batch_save_solve_and_merged_output(self):
        observations, T_mount, T_cam2base = self._synthetic_batch()
        depths = [obs["p_camera"][2] for obs in observations]
        self.assertTrue(all(0.3 <= z <= 1.5 for z in depths), depths)

        saved = self.client.post(
            "/api/mount/samples/batch", json={"observations": observations}
        ).json()
        self.assertTrue(saved["ok"], saved)
        self.assertEqual(saved["saved_count"], 15)

        # 同一 pose 的同一模型点不能重复保存
        duplicate = self.client.post(
            "/api/mount/samples/batch", json={"observations": observations[:1]}
        )
        self.assertEqual(duplicate.status_code, 409)

        # 混入其他手型号被拒绝
        other = dict(observations[0], hand_id="qiangnao-1-right", pose_id="episode_9999")
        mixed = self.client.post(
            "/api/mount/samples/batch", json={"observations": [other]}
        )
        self.assertEqual(mixed.status_code, 409)

        # 没有相机标定时解算应失败
        no_calib = self.client.post("/api/mount/solve", json={})
        self.assertEqual(no_calib.status_code, 400)

        calib = {
            "eye": "left",
            "base_link": "torso_link",
            "tip_link": "right_wrist_yaw_link",
            "T_cam2base": T_cam2base.tolist(),
            "R_cam2base": T_cam2base[:3, :3].tolist(),
            "t_cam2base_m": T_cam2base[:3, 3].tolist(),
            "p_tool_wrist_m": [0.0, 0.0, 0.0],
        }
        intrinsics_path = self.root / "camera_intrinsics.json"
        intrinsics_path.write_text(
            json.dumps(
                {
                    "serial": "CP0T263000BE",
                    "name": "Any RGB-D Camera",
                    "width": 1920,
                    "height": 1080,
                }
            ),
            encoding="utf-8",
        )
        calib["intrinsics_file"] = str(intrinsics_path)
        external_calib = self.root / "handeye_result_left.json"
        external_calib.write_text(
            json.dumps(calib), encoding="utf-8"
        )
        app_module.mount_calib_path = external_calib

        result = self.client.post("/api/mount/solve", json={}).json()
        self.assertTrue(result["ok"], result)
        solved = np.asarray(result["T_wrist2hand"])
        np.testing.assert_allclose(solved, T_mount, atol=1e-9)
        self.assertLess(result["residual_mm"]["rms"], 1e-6)
        self.assertTrue(result["leave_one_pose_out"]["feasible"])
        self.assertEqual(result["hand_id"], HAND_ID)
        self.assertEqual(result["calib_used"], str(external_calib))
        self.assertEqual(result["calib_camera"]["serial"], "CP0T263000BE")
        self.assertEqual(result["active_combo"]["hand_id"], HAND_ID)
        self.assertEqual(len(result["per_pose_overlay_T_camera_hand"]), 3)
        self.assertEqual(len(result["tcp_points_wrist_m"]), 5)

        # 叠加变换核验：T_camera_hand @ p_hand 应等于观测的 p_camera
        overlay = np.asarray(
            result["per_pose_overlay_T_camera_hand"]["episode_0000"]
        )
        first = observations[0]
        predicted = (overlay @ np.array([*first["p_hand"], 1.0]))[:3]
        np.testing.assert_allclose(predicted, first["p_camera"], atol=1e-9)

        merged_path = Path(result["merged_calib"])
        self.assertTrue(merged_path.is_file())
        merged = json.loads(merged_path.read_text())
        np.testing.assert_allclose(merged["T_wrist2hand"], T_mount, atol=1e-9)
        self.assertEqual(merged["hand_id"], HAND_ID)
        np.testing.assert_allclose(
            merged["R_cam2base"], T_cam2base[:3, :3], atol=1e-12
        )
        self.assertEqual(merged["mount_calib_used"], str(external_calib))

    def test_batch_rejects_bad_depth_and_schema(self):
        observations, _, _ = self._synthetic_batch()
        shallow = dict(observations[0])
        shallow["p_camera"] = [0.0, 0.0, 0.1]
        response = self.client.post(
            "/api/mount/samples/batch", json={"observations": [shallow]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("深度", response.json()["error"])

        wrong_version = dict(observations[0], schema_version=2)
        response = self.client.post(
            "/api/mount/samples/batch", json={"observations": [wrong_version]}
        )
        self.assertEqual(response.status_code, 400)

        wrong_point_id = dict(observations[0], point_id="tip:index")
        response = self.client.post(
            "/api/mount/samples/batch", json={"observations": [wrong_point_id]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("palm-red-01..08", response.json()["error"])

        wrong_arm = dict(observations[0], arm="left")
        response = self.client.post(
            "/api/mount/samples/batch", json={"observations": [wrong_arm]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("当前为 right", response.json()["error"])

        self.capability["hand_id"] = "qiangnao-1-right"
        self.capability["active"] = {
            "arm": "right_arm",
            "hand_id": "qiangnao-1-right",
        }
        response = self.client.post(
            "/api/mount/samples/batch", json={"observations": [observations[0]]}
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("激活手型号", response.json()["error"])

    def test_capability_center_unavailable_only_warns(self):
        observations, _, _ = self._synthetic_batch()
        self.capability.clear()
        self.capability.update(
            {"ok": True, "available": False, "error": "connection refused"}
        )
        response = self.client.post(
            "/api/mount/samples/batch", json={"observations": observations[:1]}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("能力中心不可用", payload["warnings"][0])

    def test_available_capability_rejects_arm_hand_side_mismatch(self):
        observations, _, _ = self._synthetic_batch()
        observation = dict(observations[0], hand_id="yinshi-1-left")
        self.capability["hand_id"] = "yinshi-1-left"
        self.capability["active"] = {
            "arm": "right_arm",
            "hand_id": "yinshi-1-left",
        }
        response = self.client.post(
            "/api/mount/samples/batch", json={"observations": [observation]}
        )
        self.assertEqual(response.status_code, 409)
        self.assertIn("属于 left 臂", response.json()["error"])

    def test_mount_solve_rejects_wrong_camera_serial(self):
        observations, _, T_cam2base = self._synthetic_batch()
        saved = self.client.post(
            "/api/mount/samples/batch", json={"observations": observations}
        )
        self.assertEqual(saved.status_code, 200)

        intrinsics_path = self.root / "wrong_intrinsics.json"
        intrinsics_path.write_text(
            json.dumps(
                {
                    "serial": "OTHER-CAMERA",
                    "name": "Orbbec Gemini 335",
                    "width": 1920,
                    "height": 1080,
                }
            ),
            encoding="utf-8",
        )
        calib_path = self.root / "wrong_camera_result.json"
        calib_path.write_text(
            json.dumps(
                {
                    "base_link": "torso_link",
                    "T_cam2base": T_cam2base.tolist(),
                    "R_cam2base": T_cam2base[:3, :3].tolist(),
                    "t_cam2base_m": T_cam2base[:3, 3].tolist(),
                    "intrinsics_file": str(intrinsics_path),
                }
            ),
            encoding="utf-8",
        )
        app_module.mount_calib_path = calib_path
        response = self.client.post("/api/mount/solve", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("相机序列号不一致", response.json()["error"])

        intrinsics_path.write_text(
            json.dumps(
                {
                    "serial": "CP0T263000BE",
                    "name": "Different Camera Model",
                    "width": 1280,
                    "height": 720,
                }
            ),
            encoding="utf-8",
        )
        response = self.client.post("/api/mount/solve", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("RGB width 不一致", response.json()["error"])

    def test_mount_solve_uses_current_session_handeye_result(self):
        observations, T_mount, T_cam2base = self._synthetic_batch()
        saved = self.client.post(
            "/api/mount/samples/batch", json={"observations": observations}
        )
        self.assertEqual(saved.status_code, 200)
        current_result = {
            "base_link": "torso_link",
            "T_cam2base": T_cam2base.tolist(),
            "R_cam2base": T_cam2base[:3, :3].tolist(),
            "t_cam2base_m": T_cam2base[:3, 3].tolist(),
            "camera": {
                "serial": "CP0T263000BE",
                "name": "Session RGB-D",
                "width": 1920,
                "height": 1080,
            },
        }
        result_path = app_module.save_path / "handeye3d_result.json"
        result_path.write_text(json.dumps(current_result), encoding="utf-8")

        result = self.client.post("/api/mount/solve", json={}).json()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["calib_used"], str(result_path))
        np.testing.assert_allclose(result["T_wrist2hand"], T_mount, atol=1e-9)

    def test_delete_and_clear(self):
        observations, _, _ = self._synthetic_batch()
        self.client.post(
            "/api/mount/samples/batch", json={"observations": observations[:2]}
        )
        listing = self.client.get("/api/mount/samples").json()
        self.assertEqual(listing["count"], 2)
        self.assertEqual(
            self.client.delete("/api/mount/samples/0").json()["count"], 1
        )
        self.assertEqual(self.client.post("/api/mount/clear").json()["count"], 0)


if __name__ == "__main__":
    unittest.main()
