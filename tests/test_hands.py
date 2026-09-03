from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.hands import (
    HandModel,
    load_hand_catalog,
)

EXPECTED_HANDS = {
    "yinshi-1-right": "R_hand_base_link",
    "yinshi-1-left": "L_hand_base_link",
    "qiangnao-1-right": "base_link",
    "qiangnao-1-left": "base_link",
}


class HandCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_hand_catalog()
        cls.models = {
            hand_id: HandModel(spec) for hand_id, spec in cls.catalog.items()
        }

    def test_catalog_lists_four_vendored_hands(self):
        self.assertEqual(set(self.catalog), set(EXPECTED_HANDS))
        for spec in self.catalog.values():
            self.assertTrue(spec.urdf_path.is_file())
            self.assertTrue(
                spec.urdf_path.resolve().is_relative_to(PROJECT_ROOT.resolve())
            )
            self.assertEqual(len(spec.actuated_joints), 6)

    def test_base_link_is_urdf_root(self):
        for hand_id, expected_base in EXPECTED_HANDS.items():
            with self.subTest(hand=hand_id):
                self.assertEqual(self.models[hand_id].base_link, expected_base)

    def test_five_tip_feature_points_at_zero_pose(self):
        for hand_id, model in self.models.items():
            with self.subTest(hand=hand_id):
                points = model.feature_points()
                tips = [p for p in points if p["source"] == "tip_link"]
                self.assertEqual(len(tips), 5)
                labels = {p["label"] for p in tips}
                self.assertEqual(
                    labels,
                    {"拇指指尖", "食指指尖", "中指指尖", "无名指指尖", "小指指尖"},
                )
                coords = np.array([p["p_hand"] for p in tips])
                self.assertTrue(np.all(np.isfinite(coords)))
                # 指尖离手基座应有厘米级距离，且五指不重合
                self.assertGreater(np.linalg.norm(coords, axis=1).min(), 0.02)
                spread = coords - coords.mean(axis=0)
                self.assertGreater(np.linalg.svd(spread, compute_uv=False)[1], 1e-4)

    def test_mimic_joints_expand_from_actuated(self):
        model = self.models["yinshi-1-right"]
        expanded = model.expand_joints(
            model.coerce_joints({"R_index_proximal_joint": 0.5})
        )
        self.assertAlmostEqual(expanded["R_index_intermediate_joint"], 0.5)
        # 拇指两级 mimic：intermediate=1.6x，distal=2.4x
        expanded = model.expand_joints(
            model.coerce_joints({"R_thumb_proximal_pitch_joint": 0.2})
        )
        self.assertAlmostEqual(expanded["R_thumb_intermediate_joint"], 0.32)
        self.assertAlmostEqual(expanded["R_thumb_distal_joint"], 0.48)

    def test_nonzero_joint_moves_tip(self):
        model = self.models["qiangnao-1-right"]
        zero = {
            p["id"]: np.array(p["p_hand"]) for p in model.feature_points()
        }
        bent = {
            p["id"]: np.array(p["p_hand"])
            for p in model.feature_points({"right_index_proximal_joint": 0.8})
        }
        index_id = next(k for k in zero if "index" in k)
        pinky_id = next(k for k in zero if "pinky" in k)
        self.assertGreater(np.linalg.norm(bent[index_id] - zero[index_id]), 0.01)
        np.testing.assert_allclose(bent[pinky_id], zero[pinky_id], atol=1e-12)

    def test_visuals_payload_meshes_exist_and_transforms_homogeneous(self):
        for hand_id, model in self.models.items():
            with self.subTest(hand=hand_id):
                payload = model.visuals_payload()
                self.assertGreater(len(payload), 10)
                for entry in payload:
                    T = np.asarray(entry["T_hand_link"])
                    self.assertEqual(T.shape, (4, 4))
                    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)
                    for visual in entry["visuals"]:
                        url = visual["mesh_url"]
                        self.assertTrue(url.startswith("/assets/hands/"))
                        self.assertTrue((PROJECT_ROOT / url.lstrip("/")).is_file())

    def test_coerce_joints_validation(self):
        model = self.models["yinshi-1-right"]
        with self.assertRaisesRegex(ValueError, "6 个值"):
            model.coerce_joints([0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "未知手关节"):
            model.coerce_joints({"nonexistent_joint": 0.1})
        with self.assertRaisesRegex(ValueError, "非法值"):
            model.coerce_joints([float("nan")] * 6)

    def test_metadata_is_json_ready(self):
        import json

        meta = self.models["yinshi-1-left"].metadata()
        encoded = json.dumps(meta)
        self.assertIn("feature_points", meta)
        self.assertIn("links", meta)
        self.assertGreater(len(encoded), 1000)


if __name__ == "__main__":
    unittest.main()
