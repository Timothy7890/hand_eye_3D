from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.solver import (
    leave_one_pose_out_mount,
    rpy_to_rot,
    solve_hand_mount,
)

# 五个近似指尖的模型点（手基座系，米），刻意不共面
HAND_POINTS = {
    "tip:thumb": [0.03, -0.04, 0.09],
    "tip:index": [0.02, 0.01, 0.16],
    "tip:middle": [0.0, 0.015, 0.17],
    "tip:ring": [-0.02, 0.012, 0.16],
    "tip:pinky": [-0.038, 0.008, 0.135],
}

ORDERED_MARKER_POINTS = {
    **{
        f"palm-red-{index + 1:02d}": [
            -0.03 + 0.02 * (index % 4),
            -0.025 + 0.05 * (index // 4),
            -0.006,
        ]
        for index in range(8)
    },
    **{
        f"back-green-{index + 1:02d}": [
            -0.03 + 0.02 * (index % 4),
            -0.025 + 0.05 * (index // 4),
            0.012,
        ]
        for index in range(8)
    },
}

R_TRUE = rpy_to_rot(0.05, -0.12, 1.6)
T_TRUE = np.array([0.011, -0.032, 0.078])


def _observations(poses: int, noise_mm: float = 0.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    p_hand, p_wrist, point_ids, pose_ids = [], [], [], []
    for pose in range(poses):
        for point_id, coords in HAND_POINTS.items():
            hand = np.asarray(coords)
            wrist = R_TRUE @ hand + T_TRUE
            if noise_mm:
                wrist = wrist + rng.normal(0.0, noise_mm / 1000.0, 3)
            p_hand.append(hand)
            p_wrist.append(wrist)
            point_ids.append(point_id)
            pose_ids.append(f"episode_{pose:04d}")
    return (
        np.asarray(p_hand),
        np.asarray(p_wrist),
        point_ids,
        pose_ids,
    )


class SolveHandMountTest(unittest.TestCase):
    def test_recovers_mount_from_ordered_palm_and_back_markers(self):
        point_ids = list(ORDERED_MARKER_POINTS)
        p_hand = np.asarray(list(ORDERED_MARKER_POINTS.values()))
        p_wrist = (R_TRUE @ p_hand.T).T + T_TRUE

        result = solve_hand_mount(
            p_hand,
            p_wrist,
            point_ids,
            ["episode_0000"] * len(point_ids),
        )

        self.assertEqual(result["point_count"], 16)
        np.testing.assert_allclose(result["R_wrist2hand"], R_TRUE, atol=1e-9)
        np.testing.assert_allclose(result["t_wrist2hand_m"], T_TRUE, atol=1e-9)

    def test_recovers_ground_truth_without_noise(self):
        result = solve_hand_mount(*_observations(3))
        T = np.asarray(result["T_wrist2hand"])
        np.testing.assert_allclose(T[:3, :3], R_TRUE, atol=1e-9)
        np.testing.assert_allclose(T[:3, 3], T_TRUE, atol=1e-9)
        self.assertLess(result["residual_mm"]["rms"], 1e-6)
        self.assertEqual(result["point_count"], 5)
        self.assertEqual(result["pose_count"], 3)
        self.assertEqual(result["mode"], "hand_mount_known_points")

    def test_noise_degrades_gracefully_and_stats_are_grouped(self):
        result = solve_hand_mount(*_observations(4, noise_mm=2.0))
        T = np.asarray(result["T_wrist2hand"])
        self.assertLess(np.linalg.norm(T[:3, 3] - T_TRUE) * 1000.0, 5.0)
        self.assertLess(result["residual_mm"]["rms"], 8.0)
        self.assertEqual(set(result["residual_by_point_mm"]), set(HAND_POINTS))
        self.assertEqual(len(result["residual_by_pose_mm"]), 4)
        for stats in result["residual_by_pose_mm"].values():
            self.assertEqual(stats["count"], 5)

    def test_single_pose_is_solvable(self):
        result = solve_hand_mount(*_observations(1))
        np.testing.assert_allclose(
            np.asarray(result["T_wrist2hand"])[:3, 3], T_TRUE, atol=1e-9
        )

    def test_rejects_fewer_than_three_unique_points(self):
        p_hand, p_wrist, point_ids, pose_ids = _observations(3)
        keep = [i for i, name in enumerate(point_ids) if "thumb" in name or "index" in name]
        with self.assertRaisesRegex(ValueError, "3 个不同的模型点"):
            solve_hand_mount(
                p_hand[keep],
                p_wrist[keep],
                [point_ids[i] for i in keep],
                [pose_ids[i] for i in keep],
            )

    def test_rejects_collinear_points(self):
        line = {f"p{i}": [0.0, 0.0, 0.05 * i] for i in range(4)}
        p_hand = np.asarray(list(line.values()))
        p_wrist = (R_TRUE @ p_hand.T).T + T_TRUE
        with self.assertRaisesRegex(ValueError, "共线"):
            solve_hand_mount(p_hand, p_wrist, list(line), ["ep0"] * 4)

    def test_rejects_duplicate_pose_point_pair(self):
        p_hand, p_wrist, point_ids, pose_ids = _observations(1)
        p_hand = np.vstack([p_hand, p_hand[:1]])
        p_wrist = np.vstack([p_wrist, p_wrist[:1]])
        point_ids = point_ids + [point_ids[0]]
        pose_ids = pose_ids + [pose_ids[0]]
        with self.assertRaisesRegex(ValueError, "只能有一个 observation"):
            solve_hand_mount(p_hand, p_wrist, point_ids, pose_ids)


class LeaveOnePoseOutMountTest(unittest.TestCase):
    def test_feasible_with_multiple_poses(self):
        report = leave_one_pose_out_mount(*_observations(3, noise_mm=1.0))
        self.assertTrue(report["feasible"])
        self.assertEqual(len(report["folds"]), 3)
        self.assertLess(report["stats_mm"]["rms"], 8.0)

    def test_single_pose_reports_infeasible(self):
        report = leave_one_pose_out_mount(*_observations(1))
        self.assertFalse(report["feasible"])
        self.assertTrue(report["coverage_diagnostics"])


if __name__ == "__main__":
    unittest.main()
