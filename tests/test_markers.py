from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.markers import CANONICAL_COLORS, MARKER_CATALOG, detect_markers_jpeg


class SyntheticMarkerDetectorTest(unittest.TestCase):
    def test_detects_eight_mm_scale_circle_in_full_hd_image(self):
        image = np.full((1080, 1920, 3), 245, dtype=np.uint8)
        cv2.circle(
            image,
            (960, 540),
            6,
            next(item["lab_bgr"] for item in MARKER_CATALOG if item["color"] == "red"),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94]
        )
        self.assertTrue(ok)
        detected = detect_markers_jpeg(encoded.tobytes())
        red = next(item for item in detected if item["color"] == "red")
        np.testing.assert_allclose(red["center"], [960, 540], atol=2.0)
        self.assertGreaterEqual(red["radius_px"], 4.0)

    def test_detects_nine_colored_circles_deterministically(self):
        image = np.full((600, 900, 3), 245, dtype=np.uint8)
        expected_centers = {}
        for index, item in enumerate(MARKER_CATALOG):
            center = (100 + (index % 5) * 170, 150 + (index // 5) * 280)
            expected_centers[item["color"]] = center
            cv2.circle(image, center, 38, item["lab_bgr"], thickness=-1, lineType=cv2.LINE_AA)
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94]
        )
        self.assertTrue(ok)

        first = detect_markers_jpeg(encoded.tobytes())
        second = detect_markers_jpeg(encoded.tobytes())
        self.assertEqual(first, second)
        self.assertEqual([item["color"] for item in first], list(CANONICAL_COLORS))
        self.assertEqual(len({item["id"] for item in first}), 9)
        for candidate in first:
            expected = expected_centers[candidate["color"]]
            np.testing.assert_allclose(candidate["center"], expected, atol=3.0)
            self.assertGreater(candidate["radius_px"], 30.0)
            self.assertGreater(candidate["circularity"], 0.75)
            self.assertEqual(candidate["source"], "auto")

        ambiguity_colors = {"gray", "gold", "brown"}
        for candidate in first:
            if candidate["color"] in ambiguity_colors:
                self.assertLessEqual(candidate["color_confidence"], 0.78)
                self.assertIn("ambiguity_prone_color", candidate["flags"])

    def test_detects_visible_hand_subset_and_rejects_distant_lights(self):
        image = np.full((1080, 1920, 3), 235, dtype=np.uint8)
        bgr_by_color = {item["color"]: item["lab_bgr"] for item in MARKER_CATALOG}
        hand_markers = {
            "orange": (800, 870),
            "purple": (825, 830),
            "gray": (850, 790),
            "gold": (875, 750),
        }
        for color, center in hand_markers.items():
            cv2.circle(
                image, center, 9, bgr_by_color[color], thickness=-1, lineType=cv2.LINE_AA
            )
        for color, center in {
            "red": (300, 200),
            "blue": (400, 200),
            "green": (500, 200),
        }.items():
            cv2.circle(
                image, center, 9, bgr_by_color[color], thickness=-1, lineType=cv2.LINE_AA
            )

        detected = detect_markers_jpeg(
            cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tobytes()
        )
        self.assertEqual(
            [item["color"] for item in detected],
            ["orange", "purple", "gray", "gold"],
        )
        for item in detected:
            np.testing.assert_allclose(
                item["center"], hand_markers[item["color"]], atol=2.0
            )


if __name__ == "__main__":
    unittest.main()
