from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.paths import H2_ROBOT_CONFIG_PATH
from backend.robotics import RobotModel, load_robot_config


class H2RoboticsRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_robot_config(H2_ROBOT_CONFIG_PATH)
        cls.model = RobotModel(cls.config)

    def test_config_urdf_and_arm_chains_are_vendored(self):
        root = PROJECT_ROOT.resolve()
        self.assertEqual(
            H2_ROBOT_CONFIG_PATH.resolve(),
            root / "config" / "robots" / "h2.yaml",
        )
        self.assertTrue(H2_ROBOT_CONFIG_PATH.is_file())
        self.assertTrue(self.config.urdf_path.is_file())
        self.assertTrue(self.config.urdf_path.resolve().is_relative_to(root))
        self.assertEqual(set(self.config.chains), {"left_arm", "right_arm"})
        self.assertEqual(len(self.config.chains["left_arm"].joints), 7)
        self.assertEqual(len(self.config.chains["right_arm"].joints), 7)

    def test_model_loads_and_zero_pose_fk_is_homogeneous(self):
        targets = {
            "torso_link",
            self.model.end_link("left_arm"),
            self.model.end_link("right_arm"),
        }
        zero_joints = {
            name: 0.0
            for chain_id in ("left_arm", "right_arm")
            for name in self.model.joint_names(chain_id)
        }
        transforms = self.model.forward_kinematics(
            zero_joints,
            only_links=targets,
        )

        self.assertEqual(set(transforms).intersection(targets), targets)
        for link in targets:
            with self.subTest(link=link):
                transform = transforms[link]
                self.assertEqual(transform.shape, (4, 4))
                self.assertTrue(np.all(np.isfinite(transform)))
                np.testing.assert_allclose(
                    transform[3],
                    [0.0, 0.0, 0.0, 1.0],
                    atol=1e-12,
                )

    def test_arm_and_pose_provider_import_without_ik_replay(self):
        code = f"""
import sys
from pathlib import Path

root = Path({str(PROJECT_ROOT)!r}).resolve()
sys.path.insert(0, str(root))
from backend.arm import _load_arm_model
from backend.robot import H2PoseProvider

model, chain = _load_arm_model("right")
assert chain == "right_arm"
assert len(model.joint_names(chain)) == 7
assert Path(model.urdf_path).resolve().is_relative_to(root)
assert H2PoseProvider.__module__ == "backend.robot"
assert not any(name == "IK_replay" or name.startswith("IK_replay.") for name in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-I", "-c", code],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
