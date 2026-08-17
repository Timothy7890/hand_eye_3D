"""Shared paths for the vendored hand_eye_3D runtime."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RGBD_CALIB_PATH = (
    PROJECT_ROOT / "config" / "camera" / "orbbec_rgbd_calibration.json"
)
H2_ROBOT_CONFIG_PATH = PROJECT_ROOT / "config" / "robots" / "h2.yaml"
