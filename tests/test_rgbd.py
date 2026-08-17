from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rgbd import RGBDCalibration, SoftwareDepthAligner, decode_rgbd_parts


def _calibration(
    *,
    color_shape=(3, 4),
    depth_shape=(3, 4),
    color_fx=1.0,
    color_cx=0.0,
    translation=None,
    depth_scale_mm=1.0,
):
    return RGBDCalibration(
        path=Path("/tmp/test-rgbd-calibration.json"),
        serial="TEST",
        color_shape=color_shape,
        depth_shape=depth_shape,
        color_matrix=np.array(
            [[color_fx, 0.0, color_cx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        depth_matrix=np.eye(3, dtype=np.float64),
        color_distortion=np.zeros(8, dtype=np.float64),
        depth_distortion=np.zeros(8, dtype=np.float64),
        depth_to_color_rotation=np.eye(3, dtype=np.float64),
        depth_to_color_translation_mm=(
            np.zeros(3, dtype=np.float64)
            if translation is None
            else np.asarray(translation, dtype=np.float64)
        ),
        depth_scale_mm=depth_scale_mm,
    )


def _stream(width: int, height: int) -> dict:
    return {
        "width": width,
        "height": height,
        "fps": 30,
        "format": "Y16",
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": 100.0,
            "fy": 101.0,
            "cx": width / 2,
            "cy": height / 2,
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


def _write_calibration(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "device": {"serial": "TEST"},
        "color": _stream(4, 3),
        "depth": _stream(4, 3),
        "depth_to_color": {
            "rotation_row_major": np.eye(3).tolist(),
            "translation": [1.0, 2.0, 3.0],
            "translation_unit": "mm",
        },
        "depth_scale": {"value": 1.0, "unit": "mm_per_raw_unit"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metadata() -> dict:
    return {
        "data_format": "rgbd",
        "color_format": "jpeg",
        "depth_format": "depth_z16",
        "depth_dtype": "uint16",
        "color_shape": [3, 4],
        "depth_shape": [3, 4],
    }


def _parts(metadata: dict, jpeg: bytes = b"jpeg", depth: bytes | None = None):
    if depth is None:
        depth = np.arange(12, dtype=np.uint16).tobytes()
    return [json.dumps(metadata).encode("utf-8"), jpeg, depth]


class CalibrationSchemaTest(unittest.TestCase):
    def test_parses_exported_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded = RGBDCalibration.from_file(
                _write_calibration(Path(directory) / "calibration.json")
            )

        self.assertEqual(loaded.serial, "TEST")
        self.assertEqual(loaded.color_shape, (3, 4))
        self.assertEqual(loaded.depth_shape, (3, 4))
        np.testing.assert_array_equal(
            loaded.depth_to_color_translation_mm,
            np.array([1.0, 2.0, 3.0]),
        )

    def test_rejects_unsupported_units(self):
        cases = (
            (("depth_to_color", "translation_unit"), "m", "translation_unit"),
            (("depth_scale", "unit"), "m_per_raw_unit", "depth_scale.unit"),
        )
        for keys, value, message in cases:
            with self.subTest(field=".".join(keys)):
                with tempfile.TemporaryDirectory() as directory:
                    path = _write_calibration(Path(directory) / "calibration.json")
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload[keys[0]][keys[1]] = value
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        RGBDCalibration.from_file(path)


class SoftwareDepthAlignerTest(unittest.TestCase):
    def test_identity_preserves_depth(self):
        raw = np.array(
            [[100, 200, 300, 400], [500, 0, 700, 800], [900, 1000, 1100, 1200]],
            dtype=np.uint16,
        )
        aligned = SoftwareDepthAligner(_calibration()).align(raw)
        np.testing.assert_array_equal(aligned, raw.astype(np.float32))

    def test_z_buffer_keeps_nearest_depth(self):
        aligner = SoftwareDepthAligner(
            _calibration(color_shape=(1, 1), depth_shape=(1, 2), color_fx=0.1)
        )
        aligned = aligner.align(np.array([[500, 100]], dtype=np.uint16))
        self.assertEqual(float(aligned[0, 0]), 100.0)

    def test_rejects_wrong_shape_and_dtype(self):
        aligner = SoftwareDepthAligner(_calibration())
        with self.assertRaisesRegex(ValueError, "shape"):
            aligner.align(np.zeros((2, 2), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "dtype"):
            aligner.align(np.zeros((3, 4), dtype=np.float32))

    def test_applies_depth_scale(self):
        aligner = SoftwareDepthAligner(
            _calibration(
                color_shape=(1, 2),
                depth_shape=(1, 2),
                depth_scale_mm=2.5,
            )
        )
        aligned = aligner.align(np.array([[0, 10]], dtype=np.uint16))
        self.assertEqual(float(aligned[0, 1]), 25.0)

    def test_translation_shifts_projection(self):
        aligner = SoftwareDepthAligner(
            _calibration(
                color_shape=(1, 3),
                depth_shape=(1, 1),
                translation=[1000.0, 0.0, 0.0],
            )
        )
        aligned = aligner.align(np.array([[1000]], dtype=np.uint16))
        np.testing.assert_array_equal(
            aligned,
            np.array([[0.0, 1000.0, 0.0]], dtype=np.float32),
        )


class RGBDProtocolTest(unittest.TestCase):
    def test_decodes_valid_message(self):
        bgr = np.zeros((3, 4, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", bgr)
        self.assertTrue(ok)
        depth = np.arange(12, dtype=np.uint16).reshape(3, 4)
        metadata, jpeg, decoded_depth = decode_rgbd_parts(
            _parts(_metadata(), encoded.tobytes(), depth.tobytes()),
            _calibration(),
            verify_jpeg_shape=True,
        )
        self.assertEqual(metadata, _metadata())
        self.assertEqual(jpeg, encoded.tobytes())
        np.testing.assert_array_equal(decoded_depth, depth)

    def test_strictly_rejects_shapes(self):
        cases = (
            ("malformed", {"color_shape": [3]}, "color_shape"),
            ("color profile", {"color_shape": [2, 4]}, "color shape"),
            ("depth profile", {"depth_shape": [2, 4]}, "depth shape"),
        )
        for name, changes, message in cases:
            with self.subTest(name=name):
                metadata = _metadata()
                metadata.update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    decode_rgbd_parts(_parts(metadata), _calibration())

        wrong_shape = np.zeros((2, 4, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", wrong_shape)
        self.assertTrue(ok)
        with self.assertRaisesRegex(ValueError, "JPEG shape"):
            decode_rgbd_parts(
                _parts(_metadata(), encoded.tobytes()),
                _calibration(),
                verify_jpeg_shape=True,
            )

    def test_strictly_rejects_profiles(self):
        cases = (
            ("data_format", "color", "metadata"),
            ("color_format", "png", "color_format"),
            ("depth_format", "float32", "depth_format"),
            ("depth_dtype", "float32", "depth_dtype"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                metadata = _metadata()
                metadata[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    decode_rgbd_parts(_parts(metadata), _calibration())

    def test_strictly_rejects_payloads(self):
        with self.assertRaisesRegex(ValueError, "3 段"):
            decode_rgbd_parts(_parts(_metadata())[:2], _calibration())
        with self.assertRaisesRegex(ValueError, "payload"):
            decode_rgbd_parts(_parts(_metadata(), depth=b"\x00\x01"), _calibration())


class ExporterSafetyTest(unittest.TestCase):
    def test_help_does_not_import_orbbec_sdk(self):
        script = PROJECT_ROOT / "tools" / "export_orbbec_rgbd_calibration.py"
        code = f"""
import importlib.abc
import runpy
import sys

class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyorbbecsdk":
            raise AssertionError("help imported pyorbbecsdk")
        return None

sys.meta_path.insert(0, BlockSDK())
sys.argv = [{str(script)!r}, "--help"]
runpy.run_path({str(script)!r}, run_name="__main__")
"""
        result = subprocess.run(
            [sys.executable, "-B", "-I", "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Orbbec SDK", result.stdout)


if __name__ == "__main__":
    unittest.main()
