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

from backend.solver import (
    leave_one_pose_out_multi,
    make_T,
    rpy_to_rot,
    solve_multi_marker,
)


def _synthetic_observations(noise_m: float = 0.0):
    rng = np.random.default_rng(20260817)
    R_camera = rpy_to_rot(0.18, -0.24, 0.31)
    t_camera = np.array([0.22, -0.08, 0.47])
    offsets = {
        "red": np.array([0.055, -0.014, 0.105]),
        "blue": np.array([-0.038, 0.022, 0.091]),
        "gold": np.array([0.012, 0.043, 0.126]),
    }
    wrist_rpy = [
        (-0.35, -0.20, 0.10),
        (0.28, -0.10, -0.32),
        (0.12, 0.34, 0.22),
        (-0.24, 0.18, -0.18),
        (0.40, 0.12, 0.30),
        (-0.18, -0.38, 0.26),
    ]
    visibility = [
        ("red", "blue"),
        ("red", "gold"),
        ("blue", "gold"),
        ("red", "blue", "gold"),
        ("red", "blue", "gold"),
        ("red", "blue", "gold"),
    ]
    p_camera = []
    transforms = []
    markers = []
    poses = []
    for pose_index, angles in enumerate(wrist_rpy):
        R_wrist = rpy_to_rot(*angles)
        t_wrist = np.array(
            [
                0.42 + 0.045 * pose_index,
                -0.20 + 0.035 * ((pose_index * 2) % 5),
                0.76 + 0.025 * ((pose_index * 3) % 4),
            ]
        )
        T_wrist = make_T(R_wrist, t_wrist)
        for marker in visibility[pose_index]:
            point_base = R_wrist @ offsets[marker] + t_wrist
            point_camera = R_camera.T @ (point_base - t_camera)
            if noise_m:
                point_camera = point_camera + rng.normal(scale=noise_m, size=3)
            p_camera.append(point_camera)
            transforms.append(T_wrist)
            markers.append(marker)
            poses.append(f"pose-{pose_index}")
    return (
        np.asarray(p_camera),
        np.asarray(transforms),
        markers,
        poses,
        R_camera,
        t_camera,
        offsets,
    )


class MultiMarkerSolverTest(unittest.TestCase):
    def test_exact_recovery_with_missing_markers(self):
        p_camera, transforms, markers, poses, R_true, t_true, offsets = (
            _synthetic_observations()
        )
        result = solve_multi_marker(p_camera, transforms, markers, poses)
        np.testing.assert_allclose(result["R_cam2base"], R_true, atol=2e-6)
        np.testing.assert_allclose(result["t_cam2base_m"], t_true, atol=2e-6)
        for marker, expected in offsets.items():
            np.testing.assert_allclose(
                result["p_tool_wrist_m_by_marker"][marker], expected, atol=2e-6
            )
        self.assertEqual(result["p_tool_reference"], "red")
        np.testing.assert_allclose(
            result["p_tool_wrist_m"], offsets["red"], atol=2e-6
        )
        self.assertLess(result["residual_mm"]["max"], 0.01)
        self.assertEqual(result["pose_count"], 6)
        self.assertEqual(result["marker_count"], 3)

    def test_noisy_recovery_and_leave_one_pose_out(self):
        p_camera, transforms, markers, poses, R_true, t_true, offsets = (
            _synthetic_observations(noise_m=0.0007)
        )
        result = solve_multi_marker(p_camera, transforms, markers, poses)
        np.testing.assert_allclose(result["R_cam2base"], R_true, atol=0.012)
        np.testing.assert_allclose(result["t_cam2base_m"], t_true, atol=0.012)
        for marker, expected in offsets.items():
            np.testing.assert_allclose(
                result["p_tool_wrist_m_by_marker"][marker], expected, atol=0.012
            )
        self.assertLess(result["residual_mm"]["rms"], 2.0)
        validation = leave_one_pose_out_multi(
            p_camera, transforms, markers, poses
        )
        self.assertTrue(validation["feasible"], validation)
        self.assertEqual(len(validation["folds"]), 6)

    def test_rejects_insufficient_marker_coverage(self):
        p_camera, transforms, markers, poses, *_ = _synthetic_observations()
        bad_markers = list(markers)
        gold_indices = [i for i, marker in enumerate(bad_markers) if marker == "gold"]
        keep = np.ones(len(bad_markers), dtype=bool)
        keep[gold_indices[1:]] = False
        with self.assertRaisesRegex(ValueError, "marker gold.*至少需要 2"):
            solve_multi_marker(
                p_camera[keep],
                transforms[keep],
                np.asarray(bad_markers, dtype=object)[keep],
                np.asarray(poses, dtype=object)[keep],
            )


class MultiMarkerSolveApiTest(unittest.TestCase):
    def test_auto_selects_v2_and_rejects_mixed_versions(self):
        from backend import app as app_module

        p_camera, transforms, markers, poses, *_ = _synthetic_observations()
        old_save_path = app_module.save_path
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                app_module.save_path = Path(tempdir)
                app_module.init_state()
                for index, (point, transform, marker, pose) in enumerate(
                    zip(p_camera, transforms, markers, poses)
                ):
                    record = {
                        "schema_version": 2,
                        "index": index,
                        "episode": pose,
                        "pose_id": pose,
                        "marker_id": f"marker-{marker}",
                        "color": marker,
                        "p_camera": point.tolist(),
                        "T_base_wrist": transform.tolist(),
                    }
                    (Path(tempdir) / "samples" / f"{index:04d}.json").write_text(
                        json.dumps(record), encoding="utf-8"
                    )
                solved = asyncio.run(app_module.api_solve())
                self.assertTrue(solved["ok"])
                self.assertEqual(solved["schema_version"], 2)
                self.assertEqual(solved["mode"], "multi_marker_tool_offset_joint")
                self.assertTrue(
                    (Path(tempdir) / "handeye3d_result.json").is_file()
                )

                legacy_index = len(markers)
                legacy = {
                    "schema_version": 1,
                    "index": legacy_index,
                    "p_camera": [0.0, 0.0, 1.0],
                    "T_base_wrist": np.eye(4).tolist(),
                }
                (Path(tempdir) / "samples" / f"{legacy_index:04d}.json").write_text(
                    json.dumps(legacy), encoding="utf-8"
                )
                mixed = asyncio.run(app_module.api_solve())
                self.assertEqual(mixed.status_code, 400)
                self.assertIn("不能混合解算", json.loads(mixed.body)["error"])
            finally:
                app_module.save_path = old_save_path


if __name__ == "__main__":
    unittest.main()
