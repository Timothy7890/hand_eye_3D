"""九色圆形标记目录与无需训练的 OpenCV 检测器。"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


MARKER_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "color": "red",
        "label_zh": "红色",
        "display_color": "#E53935",
        "hsv_ranges": (((0, 90, 65), (7, 255, 255)), ((168, 90, 65), (179, 255, 255))),
        "lab_bgr": (54, 80, 200),
    },
    {
        "color": "blue",
        "label_zh": "蓝色",
        "display_color": "#1E88E5",
        "hsv_ranges": (((94, 70, 45), (132, 255, 255)),),
        "lab_bgr": (210, 95, 45),
    },
    {
        "color": "orange",
        "label_zh": "橙色",
        "display_color": "#FB8C00",
        "hsv_ranges": (((8, 95, 75), (23, 255, 255)),),
        "lab_bgr": (25, 130, 235),
    },
    {
        "color": "green",
        "label_zh": "绿色",
        "display_color": "#43A047",
        "hsv_ranges": (((35, 55, 40), (91, 255, 255)),),
        "lab_bgr": (75, 165, 65),
    },
    {
        "color": "pink",
        "label_zh": "粉色",
        "display_color": "#EC407A",
        "hsv_ranges": (((156, 55, 80), (174, 255, 255)),),
        "lab_bgr": (150, 95, 235),
    },
    {
        "color": "purple",
        "label_zh": "紫色",
        "display_color": "#8E24AA",
        "hsv_ranges": (((133, 55, 45), (155, 255, 255)),),
        "lab_bgr": (150, 55, 135),
    },
    {
        "color": "gray",
        "label_zh": "灰色",
        "display_color": "#80868B",
        "hsv_ranges": (((0, 0, 55), (179, 55, 220)),),
        "lab_bgr": (128, 128, 128),
        "ambiguous": True,
    },
    {
        "color": "gold",
        "label_zh": "金色",
        "display_color": "#D4AF37",
        "hsv_ranges": (((20, 65, 65), (35, 255, 255)),),
        "lab_bgr": (45, 165, 205),
        "ambiguous": True,
    },
    {
        "color": "brown",
        "label_zh": "棕色",
        "display_color": "#795548",
        "hsv_ranges": (((2, 45, 30), (20, 220, 175)),),
        "lab_bgr": (55, 85, 125),
        "ambiguous": True,
    },
)

CANONICAL_COLORS = tuple(item["color"] for item in MARKER_CATALOG)
_CATALOG_BY_COLOR = {item["color"]: item for item in MARKER_CATALOG}


def marker_catalog_public() -> list[dict[str, Any]]:
    """返回前后端共享目录，并明确静态 OpenCV 阈值（不依赖训练模型）。"""
    result = []
    for item in MARKER_CATALOG:
        result.append(
            {
                "color": item["color"],
                "label_zh": item["label_zh"],
                "display_color": item["display_color"],
                "detection": {
                    "engine": "opencv_circle_first_hsv_lab",
                    "training_required": False,
                    "hsv_ranges": [
                        {"lower": list(lower), "upper": list(upper)}
                        for lower, upper in item["hsv_ranges"]
                    ],
                    "lab_reference_bgr": list(item["lab_bgr"]),
                    "ambiguity_prone": bool(item.get("ambiguous")),
                },
            }
        )
    return result


def canonical_color(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("marker color 必须是字符串")
    color = value.strip().lower()
    if color not in _CATALOG_BY_COLOR:
        raise ValueError(
            f"未知 marker color {value!r}，必须是: {', '.join(CANONICAL_COLORS)}"
        )
    return color


def _reference_lab(item: dict[str, Any]) -> np.ndarray:
    bgr = np.asarray(item["lab_bgr"], dtype=np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


_LAB_REFERENCES = {
    item["color"]: _reference_lab(item) for item in MARKER_CATALOG
}


def _interval_distance(value: float, lower: float, upper: float) -> float:
    if lower <= value <= upper:
        return 0.0
    return min(abs(value - lower), abs(value - upper))


def _hue_interval_distance(value: float, lower: float, upper: float) -> float:
    return min(
        _interval_distance(value, lower, upper),
        _interval_distance(value + 180.0, lower, upper),
        _interval_distance(value - 180.0, lower, upper),
    )


def _hsv_range_distance(
    hsv_value: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]
) -> float:
    h, s, v = (float(value) for value in hsv_value)
    dh = _hue_interval_distance(h, float(lower[0]), float(upper[0])) / 18.0
    ds = _interval_distance(s, float(lower[1]), float(upper[1])) / 80.0
    dv = _interval_distance(v, float(lower[2]), float(upper[2])) / 100.0
    return math.sqrt(dh * dh + ds * ds + dv * dv)


def _classify_circle_color(
    hsv_value: np.ndarray, lab_value: np.ndarray
) -> tuple[str, float, float]:
    """圆心确定后分类；HSV 主判色相，Lab 只用于同分消歧。"""
    ranked: list[tuple[float, float, int, str]] = []
    for index, item in enumerate(MARKER_CATALOG):
        hsv_distance = min(
            _hsv_range_distance(hsv_value, lower, upper)
            for lower, upper in item["hsv_ranges"]
        )
        lab_distance = float(
            np.linalg.norm(lab_value - _LAB_REFERENCES[item["color"]])
        )
        ranked.append((hsv_distance, lab_distance, index, item["color"]))
    ranked.sort()
    best_hsv, best_lab, _, color = ranked[0]
    second_hsv = ranked[1][0]
    membership = math.exp(-1.8 * best_hsv)
    lab_quality = max(0.0, 1.0 - best_lab / 150.0)
    margin = float(np.clip((second_hsv - best_hsv) / 0.75, 0.0, 1.0))
    confidence = 0.62 * membership + 0.20 * lab_quality + 0.18 * margin
    return color, float(np.clip(confidence, 0.0, 1.0)), second_hsv - best_hsv


def _circle_keypoints(
    gray: np.ndarray, min_area: float, max_area: float
) -> list[cv2.KeyPoint]:
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 245
    params.thresholdStep = 8
    params.minDistBetweenBlobs = 4
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = 0.55
    params.filterByConvexity = True
    params.minConvexity = 0.68
    params.filterByInertia = True
    params.minInertiaRatio = 0.42
    params.filterByColor = False
    return list(cv2.SimpleBlobDetector_create(params).detect(gray))


def _disk_values(
    array: np.ndarray, cx: float, cy: float, inner: float, outer: float
) -> np.ndarray:
    height, width = array.shape[:2]
    x0 = max(0, int(math.floor(cx - outer)))
    x1 = min(width, int(math.ceil(cx + outer + 1)))
    y0 = max(0, int(math.floor(cy - outer)))
    y1 = min(height, int(math.ceil(cy + outer + 1)))
    if x0 >= x1 or y0 >= y1:
        return np.empty((0, array.shape[2]), dtype=array.dtype)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    selection = (distance_sq >= inner * inner) & (distance_sq <= outer * outer)
    return array[y0:y1, x0:x1][selection]


def detect_markers_bgr(
    image: np.ndarray,
    *,
    allowed_colors: set[str] | None = None,
    keep_color_duplicates: bool = False,
) -> list[dict[str, Any]]:
    """先检测贴在浅色手部上的圆形，再对圆心区域分类；允许只看见部分颜色。"""
    bgr = np.asarray(image)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("marker 检测输入必须是 uint8 BGR 图像")
    height, width = bgr.shape[:2]
    if height < 16 or width < 16:
        raise ValueError("marker 检测图像尺寸过小")

    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    image_area = float(height * width)
    # 8 mm sticker 在约 0.7 m、fx≈1037 px 时半径只有约 6 px。面积阈值不能按
    # 大目标比例设置，否则 Full-HD 图像会把真实 sticker 全部过滤掉。
    min_area = max(12.0, image_area * 0.000004)
    max_area = image_area * 0.02
    min_radius = max(2.0, min(height, width) * 0.0015)
    max_radius = min(height, width) * 0.08
    raw: list[dict[str, Any]] = []
    for keypoint in _circle_keypoints(gray, min_area, max_area):
        cx, cy = (float(value) for value in keypoint.pt)
        radius = float(keypoint.size) * 0.5
        if radius < min_radius or radius > max_radius:
            continue
        core_hsv = _disk_values(hsv, cx, cy, 0.0, max(2.0, radius * 0.62))
        core_lab = _disk_values(lab, cx, cy, 0.0, max(2.0, radius * 0.62))
        annulus_hsv = _disk_values(hsv, cx, cy, radius * 1.30, radius * 2.20)
        annulus_lab = _disk_values(lab, cx, cy, radius * 1.30, radius * 2.20)
        if min(map(len, (core_hsv, core_lab, annulus_hsv, annulus_lab))) == 0:
            continue
        center_hsv = np.median(core_hsv.astype(float), axis=0)
        center_lab = np.median(core_lab.astype(float), axis=0)
        support_hsv = np.median(annulus_hsv.astype(float), axis=0)
        support_lab = np.median(annulus_lab.astype(float), axis=0)
        local_contrast = float(np.linalg.norm(center_lab - support_lab))
        # 大多数圆贴在白色手部上；指尖圆可能紧邻黑色关节，因此高饱和圆也允许进入
        # 候选，随后再依靠同一只手上的紧凑颜色簇排除远处指示灯。
        white_support = support_hsv[2] >= 140.0 and support_hsv[1] <= 105.0
        chromatic_circle = (
            center_hsv[1] >= 90.0
            and center_hsv[2] >= 65.0
            and local_contrast >= 25.0
        )
        if (
            center_hsv[2] < 38.0
            or local_contrast < 16.0
            or not (white_support or chromatic_circle)
        ):
            continue
        color, color_confidence, color_margin = _classify_circle_color(
            center_hsv, center_lab
        )
        if allowed_colors is not None and color not in allowed_colors:
            continue
        item = _CATALOG_BY_COLOR[color]
        flags: list[str] = []
        if color_margin < 0.20:
            flags.append("ambiguous_hsv_color")
            color_confidence *= 0.82
        if item.get("ambiguous"):
            flags.append("ambiguity_prone_color")
            color_confidence = min(color_confidence, 0.78)
        contrast_confidence = float(np.clip(local_contrast / 85.0, 0.0, 1.0))
        shape_confidence = 0.84
        confidence = float(
            np.clip(
                0.55 * color_confidence
                + 0.28 * shape_confidence
                + 0.17 * contrast_confidence,
                0.0,
                1.0,
            )
        )
        if item.get("ambiguous"):
            confidence = min(confidence, 0.84)
        raw.append(
            {
                "color": color,
                "center": [cx, cy],
                "radius_px": radius,
                "confidence": confidence,
                "color_confidence": float(color_confidence),
                "circularity": shape_confidence,
                "source": "auto",
                "flags": flags,
                "_color_index": CANONICAL_COLORS.index(color),
            }
        )

    # 同一物体可能命中相邻色域。按置信度确定性保留，空间重叠候选去重。
    raw.sort(
        key=lambda item: (
            -item["confidence"],
            item["_color_index"],
            round(item["center"][1], 4),
            round(item["center"][0], 4),
            -item["radius_px"],
        )
    )
    spatially_distinct: list[dict[str, Any]] = []
    for candidate in raw:
        center = np.asarray(candidate["center"])
        overlaps = False
        for accepted in spatially_distinct:
            distance = float(np.linalg.norm(center - np.asarray(accepted["center"])))
            if distance < 0.55 * (candidate["radius_px"] + accepted["radius_px"]):
                overlaps = True
                break
        if overlaps:
            continue
        spatially_distinct.append(candidate)

    # 手心和手背各只会露出一部分颜色。若存在至少 3 色的紧凑簇，只保留最强簇，
    # 避免远处控制柜指示灯被当作当前手上的 marker；不要求同帧出现九色。
    cluster_radius = max(90.0, min(height, width) * 0.22)
    best_cluster: list[dict[str, Any]] | None = None
    best_cluster_score: tuple[float, float, float] | None = None
    for anchor in spatially_distinct:
        anchor_center = np.asarray(anchor["center"])
        cluster = [
            item
            for item in spatially_distinct
            if float(np.linalg.norm(np.asarray(item["center"]) - anchor_center))
            <= cluster_radius
        ]
        unique_colors = len({item["color"] for item in cluster})
        distances = [
            float(np.linalg.norm(np.asarray(item["center"]) - anchor_center))
            for item in cluster
        ]
        compactness = 1.0 - float(np.mean(distances)) / cluster_radius
        vertical_prior = float(anchor["center"][1]) / float(height)
        score = (
            unique_colors + 1.5 * vertical_prior + 0.5 * compactness,
            sum(float(item["confidence"]) for item in cluster),
            -float(anchor["center"][0]),
        )
        if best_cluster_score is None or score > best_cluster_score:
            best_cluster_score = score
            best_cluster = cluster
    if (
        best_cluster_score is not None
        and best_cluster is not None
        and len({item["color"] for item in best_cluster}) >= 3
    ):
        spatially_distinct = best_cluster or []
        typical_radius = float(
            np.percentile(
                [float(item["radius_px"]) for item in spatially_distinct], 75
            )
        )
        spatially_distinct = [
            item
            for item in spatially_distinct
            if float(item["radius_px"]) >= max(min_radius, typical_radius * 0.55)
        ]
        # 再按较短距离连通，防止宽松手部区域中的腕部螺钉成为某个颜色的赢家。
        link_radius = max(70.0, min(height, width) * 0.11)
        pending = set(range(len(spatially_distinct)))
        components: list[list[dict[str, Any]]] = []
        while pending:
            stack = [pending.pop()]
            indices: list[int] = []
            while stack:
                index = stack.pop()
                indices.append(index)
                center = np.asarray(spatially_distinct[index]["center"])
                linked = [
                    other
                    for other in pending
                    if float(
                        np.linalg.norm(
                            np.asarray(spatially_distinct[other]["center"]) - center
                        )
                    )
                    <= link_radius
                ]
                for other in linked:
                    pending.remove(other)
                    stack.append(other)
            components.append([spatially_distinct[index] for index in indices])
        strongest = max(
            components,
            key=lambda component: (
                len({item["color"] for item in component})
                + 0.8
                * float(np.mean([item["center"][1] for item in component]))
                / float(height),
                sum(float(item["confidence"]) for item in component),
            ),
            default=[],
        )
        if len({item["color"] for item in strongest}) >= 3:
            spatially_distinct = strongest

    deduplicated: list[dict[str, Any]] = []
    seen_colors: set[str] = set()
    for candidate in sorted(
        spatially_distinct,
        key=lambda item: (
            -item["confidence"],
            item["_color_index"],
            round(item["center"][1], 4),
            round(item["center"][0], 4),
        ),
    ):
        if not keep_color_duplicates and candidate["color"] in seen_colors:
            continue
        seen_colors.add(candidate["color"])
        deduplicated.append(candidate)

    deduplicated.sort(
        key=lambda item: (
            item["_color_index"],
            round(item["center"][1], 4),
            round(item["center"][0], 4),
        )
    )
    result = []
    color_counts: dict[str, int] = {}
    for item in deduplicated:
        item = dict(item)
        item.pop("_color_index", None)
        color_counts[item["color"]] = color_counts.get(item["color"], 0) + 1
        item["id"] = (
            f"marker-{item['color']}-{color_counts[item['color']]:02d}"
            if keep_color_duplicates
            else f"marker-{item['color']}"
        )
        item["center"] = [round(value, 3) for value in item["center"]]
        item["radius_px"] = round(item["radius_px"], 3)
        item["confidence"] = round(item["confidence"], 6)
        item["color_confidence"] = round(item["color_confidence"], 6)
        item["circularity"] = round(item["circularity"], 6)
        result.append(item)
    return result


def detect_mount_markers_bgr(image: np.ndarray) -> list[dict[str, Any]]:
    """用 HSV 连通域检测手安装贴纸，并为每种颜色保留最强空间簇。"""
    bgr = np.asarray(image)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("安装 marker 检测输入必须是 uint8 BGR 图像")
    height, width = bgr.shape[:2]
    if height < 16 or width < 16:
        raise ValueError("安装 marker 检测图像尺寸过小")
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    masks = {
        "red": (
            ((hue <= 15) | (hue >= 165))
            & (saturation >= 70)
            & (value >= 50)
        ),
        "green": (
            (hue >= 35)
            & (hue <= 95)
            & (saturation >= 45)
            & (value >= 35)
        ),
    }
    image_area = float(height * width)
    min_area = max(30.0, image_area * 0.000025)
    max_area = image_area * 0.015
    min_radius = max(3.0, min(height, width) * 0.0025)
    max_radius = min(height, width) * 0.06
    link_radius = max(70.0, min(height, width) * 0.18)
    result: list[dict[str, Any]] = []

    for color in ("red", "green"):
        mask = masks[color].astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: list[dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if not min_area <= area <= max_area or perimeter <= 0:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)
            if not min_radius <= radius <= max_radius:
                continue
            circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
            fill_ratio = float(area / max(math.pi * radius * radius, 1e-9))
            if circularity < 0.50 or fill_ratio < 0.45:
                continue
            confidence = float(
                np.clip(0.55 * circularity + 0.45 * fill_ratio, 0.0, 1.0)
            )
            candidates.append(
                {
                    "color": color,
                    "center": [float(cx), float(cy)],
                    "radius_px": radius,
                    "confidence": confidence,
                    "color_confidence": confidence,
                    "circularity": circularity,
                    "source": "auto_hsv_component",
                    "flags": [],
                    "_quality": area * confidence,
                }
            )

        pending = set(range(len(candidates)))
        components: list[list[dict[str, Any]]] = []
        while pending:
            stack = [pending.pop()]
            indices: list[int] = []
            while stack:
                index = stack.pop()
                indices.append(index)
                center = np.asarray(candidates[index]["center"])
                linked = [
                    other
                    for other in pending
                    if float(
                        np.linalg.norm(
                            np.asarray(candidates[other]["center"]) - center
                        )
                    )
                    <= link_radius
                ]
                for other in linked:
                    pending.remove(other)
                    stack.append(other)
            components.append([candidates[index] for index in indices])
        strongest = max(
            components,
            key=lambda component: (
                sum(
                    item["_quality"]
                    for item in sorted(
                        component,
                        key=lambda item: item["_quality"],
                        reverse=True,
                    )[:8]
                ),
                min(len(component), 8),
            ),
            default=[],
        )
        strongest = sorted(
            strongest,
            key=lambda item: item["_quality"],
            reverse=True,
        )[:8]
        strongest.sort(
            key=lambda item: (
                round(item["center"][1], 4),
                round(item["center"][0], 4),
            )
        )
        for index, item in enumerate(strongest, start=1):
            item = dict(item)
            item.pop("_quality", None)
            item["id"] = f"marker-{color}-{index:02d}"
            item["center"] = [round(value, 3) for value in item["center"]]
            item["radius_px"] = round(item["radius_px"], 3)
            item["confidence"] = round(item["confidence"], 6)
            item["color_confidence"] = round(item["color_confidence"], 6)
            item["circularity"] = round(item["circularity"], 6)
            result.append(item)
    grouped = {
        color: [item for item in result if item["color"] == color]
        for color in ("red", "green")
    }
    scores = {
        color: sum(
            math.pi
            * float(item["radius_px"]) ** 2
            * float(item["confidence"])
            for item in items
        )
        for color, items in grouped.items()
    }
    dominant = max(scores, key=scores.get)
    other = "green" if dominant == "red" else "red"
    if (
        len(grouped[dominant]) >= 5
        and scores[dominant] > max(scores[other] * 4.0, 1.0)
    ):
        result = grouped[dominant]
    return result


def detect_markers_jpeg(data: bytes) -> list[dict[str, Any]]:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("代表 RGB 图片不是有效 JPEG")
    return detect_markers_bgr(image)
