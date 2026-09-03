"""离线遥操作 episode 读取、RGB-D 对齐、彩色点云与 H2 FK。"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .markers import (
    CANONICAL_COLORS,
    canonical_color,
    detect_markers_bgr,
    detect_mount_markers_bgr,
)
from .paths import DEFAULT_RGBD_CALIB_PATH, H2_ROBOT_CONFIG_PATH
from .rgbd import RGBDCalibration, SoftwareDepthAligner
from .robotics import RobotModel, load_robot_config
RIGHT_ARM_DATASET_JOINTS = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]
ARM_DATASET_JOINTS = {
    "right": RIGHT_ARM_DATASET_JOINTS,
    "left": [name.replace("right_", "left_", 1) for name in RIGHT_ARM_DATASET_JOINTS],
}
DEPTH_VALID_MIN_MM = 60.0
DEPTH_VALID_MAX_MM = 15000.0
DEPTH_MAX_SPREAD_MM = 80.0
POINT_CLOUD_PICK_MIN_MM = 300.0
POINT_CLOUD_PICK_MAX_MM = 1500.0
class EpisodeValidationError(ValueError):
    """遥操作 episode 内容不完整或与生产配置不兼容。"""


class PointCloudStaleError(EpisodeValidationError):
    """前端提交的点云已经不是当前 episode 对应的版本。"""


@dataclass(frozen=True)
class OfflineFrame:
    index: int
    color_path: Path
    depth_path: Path
    arm_q: np.ndarray
    timestamps: dict[str, Any]


@dataclass(frozen=True)
class OfflineEpisode:
    name: str
    root: Path
    data_path: Path
    info: dict[str, Any]
    frames: tuple[OfflineFrame, ...]
    warnings: tuple[str, ...] = ()

    @property
    def representative(self) -> OfflineFrame:
        return self.frames[len(self.frames) // 2]


@dataclass(frozen=True)
class OfflinePointCloud:
    cloud_id: str
    stride: int
    points: np.ndarray
    colors: np.ndarray
    pixels: np.ndarray
    valid_depth_frames: np.ndarray
    depth_spread_mm: np.ndarray

    def to_binary_ply(self) -> bytes:
        records = np.empty(
            len(self.points),
            dtype=np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ],
                align=False,
            ),
        )
        records["x"], records["y"], records["z"] = self.points.T
        records["red"], records["green"], records["blue"] = self.colors.T
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"comment cloud_id {self.cloud_id}\n"
            "comment coordinate_system camera_x_right_y_down_z_forward\n"
            f"comment sampling_stride {self.stride}\n"
            f"element vertex {len(records)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        return header + records.tobytes()


class OfflineEpisodeBackend:
    """一个 robot-style task 目录对应的只读离线数据后端。"""

    def __init__(
        self,
        task_dir: str | Path,
        rgbd_calib_path: str | Path,
        arm: str = "right",
    ):
        self.task_dir = Path(task_dir).expanduser().resolve()
        self.rgbd_calib_path = Path(rgbd_calib_path).expanduser().resolve()
        if not self.task_dir.is_dir():
            raise ValueError(f"遥操作任务目录不存在或不是目录: {self.task_dir}")
        if arm not in ARM_DATASET_JOINTS:
            raise ValueError(f"不支持的手臂 {arm!r}，必须是 right 或 left")
        self.arm = arm
        self.arm_state_key = f"{arm}_arm"
        self.dataset_joint_order = ARM_DATASET_JOINTS[arm]

        self.calibration = RGBDCalibration.from_file(self.rgbd_calib_path)
        self.aligner = SoftwareDepthAligner(self.calibration)
        config = load_robot_config(H2_ROBOT_CONFIG_PATH)
        self.robot_model = RobotModel(config)
        self.chain = self.arm_state_key
        self.base_link = self.robot_model.base_link(self.chain)
        self.wrist_link = self.robot_model.end_link(self.chain)
        self.joint_names = self.robot_model.joint_names(self.chain)
        if len(self.joint_names) != 7:
            raise ValueError(
                f"H2 {self.chain} 配置应有 7 个关节，实际为 {len(self.joint_names)}"
            )
        self._aligned_depth_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._point_cloud_cache: dict[tuple[Any, ...], OfflinePointCloud] = {}

    def _candidate_paths(self) -> list[Path]:
        return sorted(self.task_dir.glob("episode_*/data.json"))

    def episode_names(self) -> set[str]:
        """当前磁盘上仍存在 data.json 的 episode 名称。"""
        return {path.parent.name for path in self._candidate_paths()}

    def episode_camera_serials(self) -> dict[str, str | None]:
        """返回当前 episode 名称对应的采集相机，防止重用编号时匹配到旧样本。"""
        serials: dict[str, str | None] = {}
        for data_path in self._candidate_paths():
            try:
                payload = self._read_json(data_path)
            except EpisodeValidationError:
                continue
            info = payload.get("info")
            if isinstance(info, dict) and info.get("kind") == "hand_eye_calibration":
                serial = info.get("camera_serial")
                serials[data_path.parent.name] = (
                    serial.strip() if isinstance(serial, str) and serial.strip() else None
                )
        return serials

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EpisodeValidationError(f"episode 元数据不存在: {path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise EpisodeValidationError(f"无法读取 episode 元数据 {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise EpisodeValidationError(f"episode 元数据根节点必须是 JSON object: {path}")
        return payload

    def scan(self) -> list[dict[str, Any]]:
        """扫描 hand_eye_calibration episode；坏 episode 会保留并标注错误。"""
        episodes: list[dict[str, Any]] = []
        for data_path in self._candidate_paths():
            name = data_path.parent.name
            try:
                payload = self._read_json(data_path)
                info = payload.get("info")
                if not isinstance(info, dict) or info.get("kind") != "hand_eye_calibration":
                    continue
                episode = self._parse_payload(name, data_path, payload)
                episodes.append(self._summary(episode))
            except EpisodeValidationError as exc:
                episodes.append({"name": name, "valid": False, "error": str(exc)})
        return episodes

    def load(self, name: str) -> OfflineEpisode:
        if not isinstance(name, str) or not name.startswith("episode_"):
            raise EpisodeValidationError("episode 名称必须形如 episode_0001")
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise EpisodeValidationError("episode 名称不合法")
        data_path = self.task_dir / name / "data.json"
        payload = self._read_json(data_path)
        info = payload.get("info")
        if not isinstance(info, dict) or info.get("kind") != "hand_eye_calibration":
            raise EpisodeValidationError(
                f"{name} 不是 hand_eye_calibration episode"
            )
        return self._parse_payload(name, data_path, payload)

    def _resolve_asset(self, episode_root: Path, value: Any, label: str) -> Path:
        if not isinstance(value, str) or not value:
            raise EpisodeValidationError(f"{label} 路径缺失")
        path = (episode_root / value).resolve()
        try:
            path.relative_to(episode_root)
        except ValueError as exc:
            raise EpisodeValidationError(f"{label} 路径越出 episode 目录: {value}") from exc
        if not path.is_file():
            raise EpisodeValidationError(f"{label} 文件不存在: {path}")
        return path

    @staticmethod
    def _asset_value(mapping: Any, preferred: str, label: str) -> Any:
        if not isinstance(mapping, dict) or not mapping:
            raise EpisodeValidationError(f"{label} 字段缺失")
        if preferred in mapping:
            return mapping[preferred]
        if len(mapping) == 1:
            return next(iter(mapping.values()))
        raise EpisodeValidationError(f"{label} 中找不到 {preferred!r}")

    def _parse_payload(
        self, name: str, data_path: Path, payload: dict[str, Any]
    ) -> OfflineEpisode:
        info = payload["info"]
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            raise EpisodeValidationError(f"{name} 没有有效 data 帧")
        try:
            declared_count = int(info["frame_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EpisodeValidationError(f"{name} 的 info.frame_count 不合法") from exc
        if declared_count != len(rows):
            raise EpisodeValidationError(
                f"{name} 声明 {declared_count} 帧，但 data 中有 {len(rows)} 帧"
            )

        order = info.get(f"{self.arm_state_key}_joint_order")
        if order != self.dataset_joint_order:
            raise EpisodeValidationError(
                f"{name} 的{self.arm}臂关节顺序不符合 H2 生产约定"
            )
        episode_warnings: list[str] = []
        serial = info.get("camera_serial")
        if serial and self.calibration.serial and str(serial) != self.calibration.serial:
            episode_warnings.append(
                f"{name} 相机序列号 {serial} 与 RGB-D 标定 {self.calibration.serial} 不一致"
            )

        episode_root = data_path.parent.resolve()
        frames: list[OfflineFrame] = []
        for position, row in enumerate(rows):
            if not isinstance(row, dict):
                raise EpisodeValidationError(f"{name} 第 {position} 帧必须是 JSON object")
            try:
                index = int(row.get("idx", position))
                q = np.asarray(
                    row["states"][self.arm_state_key]["qpos"], dtype=float
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise EpisodeValidationError(
                    f"{name} 第 {position} 帧缺少{self.arm}臂 qpos"
                ) from exc
            if q.shape != (7,) or not np.all(np.isfinite(q)):
                raise EpisodeValidationError(
                    f"{name} 第 {position} 帧{self.arm}臂 qpos 必须是 7 个有限数值"
                )

            color_path = self._resolve_asset(
                episode_root,
                self._asset_value(row.get("colors"), "head_rgb", "colors"),
                f"{name} 第 {position} 帧 RGB",
            )
            if color_path.suffix.lower() not in {".jpg", ".jpeg"}:
                raise EpisodeValidationError(
                    f"{name} 第 {position} 帧 RGB 必须是 JPEG"
                )
            depth_path = self._resolve_asset(
                episode_root,
                self._asset_value(row.get("depths"), "head_depth", "depths"),
                f"{name} 第 {position} 帧深度",
            )
            try:
                depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise EpisodeValidationError(
                    f"{name} 第 {position} 帧深度 NPY 无法读取: {exc}"
                ) from exc
            if depth.dtype != np.uint16:
                raise EpisodeValidationError(
                    f"{name} 第 {position} 帧深度必须是原始 uint16，实际为 {depth.dtype}"
                )
            if depth.shape != self.calibration.depth_shape:
                raise EpisodeValidationError(
                    f"{name} 第 {position} 帧深度尺寸 {depth.shape} "
                    f"与 RGB-D 标定 {self.calibration.depth_shape} 不一致"
                )

            rgbd = row.get("rgbd")
            if isinstance(rgbd, dict):
                if rgbd.get("depth_format") not in (None, "depth_z16"):
                    raise EpisodeValidationError(
                        f"{name} 第 {position} 帧不是原始 depth_z16"
                    )
                if rgbd.get("depth_dtype") not in (None, "uint16"):
                    raise EpisodeValidationError(
                        f"{name} 第 {position} 帧 depth_dtype 不是 uint16"
                    )
                color_shape = tuple(rgbd.get("color_shape", ()))
                depth_shape = tuple(rgbd.get("depth_shape", ()))
                if color_shape and color_shape != self.calibration.color_shape:
                    raise EpisodeValidationError(
                        f"{name} 第 {position} 帧 RGB 尺寸 {color_shape} "
                        f"与标定 {self.calibration.color_shape} 不一致"
                    )
                if depth_shape and depth_shape != self.calibration.depth_shape:
                    raise EpisodeValidationError(
                        f"{name} 第 {position} 帧深度元数据尺寸 {depth_shape} "
                        f"与标定 {self.calibration.depth_shape} 不一致"
                    )
            timestamps = row.get("timestamps")
            frames.append(
                OfflineFrame(
                    index=index,
                    color_path=color_path,
                    depth_path=depth_path,
                    arm_q=q.copy(),
                    timestamps=dict(timestamps) if isinstance(timestamps, dict) else {},
                )
            )
        return OfflineEpisode(
            name=name,
            root=episode_root,
            data_path=data_path.resolve(),
            info=dict(info),
            frames=tuple(frames),
            warnings=tuple(episode_warnings),
        )

    def _summary(self, episode: OfflineEpisode) -> dict[str, Any]:
        return {
            "name": episode.name,
            "valid": True,
            "frame_count": len(episode.frames),
            "representative_frame": episode.representative.index,
            "preview_frame_idx": episode.representative.index,
            "camera_serial": episode.info.get("camera_serial"),
            "calibration_serial": self.calibration.serial,
            "color_shape": list(self.calibration.color_shape),
            "preview_url": f"/api/offline/episodes/{episode.name}/preview",
            "warnings": list(episode.warnings),
        }

    def preview_jpeg(self, name: str) -> bytes:
        episode = self.load(name)
        path = episode.representative.color_path
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise EpisodeValidationError(f"无法读取代表 RGB 图片 {path}: {exc}") from exc
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise EpisodeValidationError(f"代表 RGB 图片不是有效 JPEG: {path}")
        if decoded.shape[:2] != self.calibration.color_shape:
            raise EpisodeValidationError(
                f"代表 RGB 图片尺寸 {decoded.shape[:2]} "
                f"与标定 {self.calibration.color_shape} 不一致"
            )
        return data

    def detect_markers(self, name: str) -> dict[str, Any]:
        """在代表 JPEG 上检测当前手心或手背可见的圆形 marker 子集。"""
        episode = self.load(name)
        path = episode.representative.color_path
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        except (OSError, ValueError) as exc:
            raise EpisodeValidationError(f"无法读取代表 RGB 图片 {path}: {exc}") from exc
        if image is None:
            raise EpisodeValidationError(f"代表 RGB 图片不是有效 JPEG: {path}")
        if image.shape[:2] != self.calibration.color_shape:
            raise EpisodeValidationError(
                f"代表 RGB 图片尺寸 {image.shape[:2]} "
                f"与标定 {self.calibration.color_shape} 不一致"
            )
        candidates = detect_markers_bgr(image)
        detected_colors = {candidate["color"] for candidate in candidates}
        detection_warnings = [
            f"{candidate['color']} 颜色容易受光照/材质影响，请人工确认"
            for candidate in candidates
            if "ambiguity_prone_color" in candidate["flags"]
        ]
        return {
            "ok": True,
            "episode": episode.name,
            "representative_frame": episode.representative.index,
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "candidates": candidates,
            "markers": candidates,
            "missing_colors": [
                color for color in CANONICAL_COLORS if color not in detected_colors
            ],
            "warnings": [*episode.warnings, *detection_warnings],
        }

    def detect_mount_candidates(
        self,
        name: str,
        stride: int = 2,
        edited_markers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """检测重复红/绿圆，并映射到当前稳定点云中的最近顶点。"""
        episode = self.load(name)
        image = cv2.imread(
            str(episode.representative.color_path), cv2.IMREAD_COLOR
        )
        if image is None or image.shape[:2] != self.calibration.color_shape:
            raise EpisodeValidationError(
                f"代表 RGB 图片无法用于安装圆点检测: {episode.representative.color_path}"
            )
        if edited_markers is None:
            detected = detect_mount_markers_bgr(image)
        else:
            if not isinstance(edited_markers, list):
                raise EpisodeValidationError("markers 必须是数组")
            detected = [
                self._validated_marker(marker, position)
                for position, marker in enumerate(edited_markers)
            ]
            if any(marker["color"] not in {"red", "green"} for marker in detected):
                raise EpisodeValidationError("安装圆点颜色只能是 red 或 green")
            ids = [marker["id"] for marker in detected]
            if len(set(ids)) != len(ids):
                raise EpisodeValidationError("安装圆点 id 必须唯一")
            detected = [
                {
                    **marker,
                    "center": np.asarray(marker["center"], dtype=float).tolist(),
                    "radius_px": float(marker["radius_px"]),
                }
                for marker in detected
            ]
        cloud = self.point_cloud(name, stride)
        cloud_pixels = cloud.pixels.astype(np.float64, copy=False)
        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        used_vertices: set[int] = set()
        for marker in detected:
            center = np.asarray(marker["center"], dtype=np.float64)
            search_radius = max(
                4.0, float(marker["radius_px"]) * 1.5, float(stride) * 1.5
            )
            delta = cloud_pixels - center
            nearby = np.flatnonzero(
                (np.abs(delta[:, 0]) <= search_radius)
                & (np.abs(delta[:, 1]) <= search_radius)
            )
            if not len(nearby):
                rejected.append(
                    {
                        **marker,
                        "error": "圆心附近没有稳定深度点",
                    }
                )
                continue
            distances_sq = np.sum(delta[nearby] ** 2, axis=1)
            nearest = int(nearby[int(np.argmin(distances_sq))])
            if nearest in used_vertices:
                rejected.append(
                    {
                        **marker,
                        "error": "与另一个圆映射到同一点云顶点",
                    }
                )
                continue
            if float(np.min(distances_sq)) > search_radius * search_radius:
                rejected.append(
                    {
                        **marker,
                        "error": "圆心附近没有足够接近的稳定深度点",
                    }
                )
                continue
            used_vertices.add(nearest)
            point = cloud.points[nearest].astype(float)
            candidates.append(
                {
                    **marker,
                    "candidate_id": marker["id"],
                    "vertex_index": nearest,
                    "pixel": cloud.pixels[nearest].astype(int).tolist(),
                    "p_camera": point.tolist(),
                    "depth_mm": float(point[2] * 1000.0),
                    "valid_depth_frames": int(cloud.valid_depth_frames[nearest]),
                    "depth_spread_mm": float(cloud.depth_spread_mm[nearest]),
                }
            )
        counts = {
            color: sum(item["color"] == color for item in candidates)
            for color in ("red", "green")
        }
        warnings_out = list(episode.warnings)
        for color, count in counts.items():
            if count > 8:
                warnings_out.append(
                    f"检测到 {count} 个{color}候选，超过预期 8 个，请人工排除误检"
                )
        if rejected:
            warnings_out.append(
                f"{len(rejected)} 个 RGB 圆心因缺少稳定深度未映射到点云"
            )
        return {
            "ok": True,
            "episode": episode.name,
            "representative_frame": episode.representative.index,
            "preview_url": f"/api/offline/episodes/{episode.name}/preview",
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "cloud_id": cloud.cloud_id,
            "point_cloud_stride": cloud.stride,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "counts": counts,
            "rejected": rejected,
            "rejected_count": len(rejected),
            "warnings": warnings_out,
        }

    def _wrist_pose(self, q: np.ndarray) -> np.ndarray:
        joint_values = dict(zip(self.joint_names, q.tolist()))
        transforms = self.robot_model.forward_kinematics(
            joint_values, only_links=[self.base_link, self.wrist_link]
        )
        try:
            return np.linalg.inv(transforms[self.base_link]) @ transforms[self.wrist_link]
        except KeyError as exc:
            raise EpisodeValidationError(f"H2 FK 结果缺少 link: {exc}") from exc

    def _depth_cache_key(self, episode: OfflineEpisode) -> tuple[Any, ...]:
        return (
            episode.name,
            tuple(
                (
                    str(frame.depth_path),
                    frame.depth_path.stat().st_mtime_ns,
                    frame.depth_path.stat().st_size,
                )
                for frame in episode.frames
            ),
        )

    def _aligned_depth_stack(self, episode: OfflineEpisode) -> np.ndarray:
        """每帧只做一次软件对齐，供同一批 marker 共享；同一 episode 复用缓存。"""
        key = self._depth_cache_key(episode)
        cached = self._aligned_depth_cache.get(key)
        if cached is not None:
            return cached
        aligned = []
        for frame in episode.frames:
            raw_depth = np.load(frame.depth_path, allow_pickle=False)
            aligned.append(np.asarray(self.aligner.align(raw_depth), dtype=np.float32))
        stack = np.stack(aligned, axis=0)
        self._aligned_depth_cache = {key: stack}
        return stack

    def _point_cloud_cache_key(
        self, episode: OfflineEpisode, stride: int
    ) -> tuple[Any, ...]:
        representative = episode.representative.color_path
        calibration_stat = self.rgbd_calib_path.stat()
        return (
            self._depth_cache_key(episode),
            str(representative),
            representative.stat().st_mtime_ns,
            representative.stat().st_size,
            str(self.rgbd_calib_path),
            calibration_stat.st_mtime_ns,
            calibration_stat.st_size,
            stride,
        )

    def point_cloud(self, name: str, stride: int = 2) -> OfflinePointCloud:
        """生成稳定的彩色相机系点云，顶点顺序按采样后的像素行优先排列。"""
        if isinstance(stride, bool) or not isinstance(stride, int) or not 1 <= stride <= 8:
            raise EpisodeValidationError("point cloud stride 必须是 1~8 的整数")
        episode = self.load(name)
        key = self._point_cloud_cache_key(episode, stride)
        cached = self._point_cloud_cache.get(key)
        if cached is not None:
            return cached

        stack = self._aligned_depth_stack(episode)
        valid = (
            np.isfinite(stack)
            & (stack > DEPTH_VALID_MIN_MM)
            & (stack < DEPTH_VALID_MAX_MM)
        )
        valid_count = np.sum(valid, axis=0)
        required = max(3, len(stack) // 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_mm = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
        minimum_mm = np.min(np.where(valid, stack, np.inf), axis=0)
        maximum_mm = np.max(np.where(valid, stack, -np.inf), axis=0)
        spread_mm = maximum_mm - minimum_mm
        stable = (
            (valid_count >= required)
            & np.isfinite(median_mm)
            & np.isfinite(spread_mm)
            & (spread_mm <= DEPTH_MAX_SPREAD_MM)
            & (median_mm >= POINT_CLOUD_PICK_MIN_MM)
            & (median_mm <= POINT_CLOUD_PICK_MAX_MM)
        )

        color_h, color_w = self.calibration.color_shape
        sampled_v = np.arange(0, color_h, stride, dtype=np.int32)
        sampled_u = np.arange(0, color_w, stride, dtype=np.int32)
        grid_u, grid_v = np.meshgrid(sampled_u, sampled_v)
        sampled_stable = stable[grid_v, grid_u]
        pixels = np.column_stack(
            (grid_u[sampled_stable], grid_v[sampled_stable])
        ).astype(np.int32, copy=False)
        if not len(pixels):
            raise EpisodeValidationError("该 episode 没有可用于点云的稳定深度像素")

        u = pixels[:, 0]
        v = pixels[:, 1]
        z = median_mm[v, u].astype(np.float32) / 1000.0
        fx, fy, cx, cy = self.calibration.color_intrinsics
        points = np.column_stack(
            ((u - cx) * z / fx, (v - cy) * z / fy, z)
        ).astype(np.float32, copy=False)

        image = cv2.imread(str(episode.representative.color_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != self.calibration.color_shape:
            raise EpisodeValidationError(
                f"代表 RGB 图片无法用于点云着色: {episode.representative.color_path}"
            )
        colors = image[v, u, ::-1].astype(np.uint8, copy=True)
        cloud_id = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:24]
        cloud = OfflinePointCloud(
            cloud_id=cloud_id,
            stride=stride,
            points=points,
            colors=colors,
            pixels=pixels,
            valid_depth_frames=valid_count[v, u].astype(np.uint8, copy=False),
            depth_spread_mm=spread_mm[v, u].astype(np.float32, copy=False),
        )
        self._point_cloud_cache = {key: cloud}
        return cloud

    def point_cloud_ply(
        self, name: str, stride: int = 2
    ) -> tuple[bytes, OfflinePointCloud]:
        cloud = self.point_cloud(name, stride)
        return cloud.to_binary_ply(), cloud

    def depth_overlay_png(self, name: str) -> bytes:
        """把五帧 RGB 对齐深度的逐像素中值渲染成透明伪彩 PNG。"""
        episode = self.load(name)
        stack = self._aligned_depth_stack(episode)
        valid = (
            np.isfinite(stack)
            & (stack > DEPTH_VALID_MIN_MM)
            & (stack < DEPTH_VALID_MAX_MM)
        )
        required = max(3, len(stack) // 2)
        stable = np.sum(valid, axis=0) >= required
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_mm = np.nanmedian(np.where(valid, stack, np.nan), axis=0)

        values = median_mm[stable]
        if values.size == 0:
            raise EpisodeValidationError("该 episode 的五帧深度没有可叠加的稳定像素")
        near_mm, far_mm = np.percentile(values, [2.0, 98.0])
        if far_mm - near_mm < 1.0:
            far_mm = near_mm + 1.0
        normalized = np.clip(
            (median_mm - near_mm) / (far_mm - near_mm), 0.0, 1.0
        )
        color = cv2.applyColorMap(
            np.asarray(np.nan_to_num(normalized) * 255.0, dtype=np.uint8),
            cv2.COLORMAP_TURBO,
        )
        alpha = np.where(stable, 255, 0).astype(np.uint8)
        bgra = np.dstack((color, alpha))
        ok, encoded = cv2.imencode(".png", bgra)
        if not ok:
            raise EpisodeValidationError("深度叠加 PNG 编码失败")
        return encoded.tobytes()

    @staticmethod
    def _validated_marker(marker: Any, position: int) -> dict[str, Any]:
        if not isinstance(marker, dict):
            raise EpisodeValidationError(f"第 {position} 个 marker 必须是 JSON object")
        marker_id = marker.get("id", marker.get("marker_id"))
        if not isinstance(marker_id, str) or not marker_id.strip():
            raise EpisodeValidationError(f"第 {position} 个 marker 缺少非空 id")
        try:
            color = canonical_color(marker.get("color"))
        except ValueError as exc:
            raise EpisodeValidationError(str(exc)) from exc
        try:
            center = np.asarray(marker["center"], dtype=float).reshape(2)
            radius_px = float(marker["radius_px"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EpisodeValidationError(
                f"marker {marker_id} 的 center/radius_px 不合法"
            ) from exc
        if not np.all(np.isfinite(center)) or not np.isfinite(radius_px) or radius_px <= 0:
            raise EpisodeValidationError(
                f"marker {marker_id} 的 center/radius_px 必须是有限正数"
            )
        source = marker.get("source", "edited")
        if not isinstance(source, str) or not source.strip():
            raise EpisodeValidationError(f"marker {marker_id} 的 source 必须是非空字符串")
        result = {
            "id": marker_id.strip(),
            "color": color,
            "center": center,
            "radius_px": radius_px,
            "source": source.strip(),
        }
        for key in ("confidence", "color_confidence", "circularity", "flags"):
            if key in marker:
                result[key] = marker[key]
        return result

    def pick_many(self, name: str, edited_markers: list[dict[str, Any]]) -> dict[str, Any]:
        """一次对齐 burst 深度并为多个编辑后的 marker 生成确认 observation。"""
        episode = self.load(name)
        if not isinstance(edited_markers, list) or not edited_markers:
            raise EpisodeValidationError("markers 必须是非空数组")
        markers = [
            self._validated_marker(marker, position)
            for position, marker in enumerate(edited_markers)
        ]
        ids = [marker["id"] for marker in markers]
        colors = [marker["color"] for marker in markers]
        if len(set(ids)) != len(ids):
            raise EpisodeValidationError("同一 episode 中 marker id 必须唯一")
        if len(set(colors)) != len(colors):
            raise EpisodeValidationError("同一 episode 中每种 canonical color 只能确认一个 marker")

        color_h, color_w = self.calibration.color_shape
        depth_stack = self._aligned_depth_stack(episode)
        q_median = np.median(
            np.stack([frame.arm_q for frame in episode.frames]), axis=0
        )
        T_base_wrist = self._wrist_pose(q_median)
        fx, fy, cx, cy = self.calibration.color_intrinsics
        representative = episode.representative
        observations: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for marker in markers:
            u = int(round(float(marker["center"][0])))
            v = int(round(float(marker["center"][1])))
            error_prefix = {"id": marker["id"], "color": marker["color"]}
            if not (0 <= u < color_w and 0 <= v < color_h):
                errors.append(
                    {
                        **error_prefix,
                        "error": f"像素越界 ({u},{v})，RGB 图像为 {color_w}x{color_h}",
                    }
                )
                continue

            vals = depth_stack[:, v, u]
            valid = vals[
                np.isfinite(vals)
                & (vals > DEPTH_VALID_MIN_MM)
                & (vals < DEPTH_VALID_MAX_MM)
            ]
            required = max(3, len(vals) // 2)
            if valid.size < required:
                errors.append(
                    {
                        **error_prefix,
                        "error": f"该像素没有稳定深度（{valid.size}/{len(vals)} 帧有效）。"
                        "细小/深色/反光表面双目常测不到，请编辑中心后重试",
                    }
                )
                continue
            spread = float(np.max(valid) - np.min(valid))
            if spread > DEPTH_MAX_SPREAD_MM:
                errors.append(
                    {
                        **error_prefix,
                        "error": f"该像素深度在多帧间跳动 {spread:.0f}mm"
                        "（边缘闪烁），请编辑中心后重试",
                    }
                )
                continue

            z_mm = float(np.median(valid))
            z = z_mm / 1000.0
            p_camera = np.array(
                [(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float
            )
            observation = {
                "schema_version": 2,
                "id": marker["id"],
                "marker_id": marker["id"],
                "color": marker["color"],
                "episode": episode.name,
                "pose_id": episode.name,
                "center": [float(value) for value in marker["center"]],
                "pixel": [u, v],
                "radius_px": float(marker["radius_px"]),
                "source": marker["source"],
                "p_camera": p_camera.tolist(),
                "T_base_wrist": T_base_wrist.tolist(),
                "depth_mm": z_mm,
                "valid_ratio": float(valid.size / len(vals)),
                "valid_depth_frames": int(valid.size),
                "depth_frame_count": len(vals),
                "burst_frames_used": len(vals),
                "depth_spread_mm": spread,
                f"{self.arm_state_key}_q_median": q_median.tolist(),
                "qpos_median_rad": q_median.tolist(),
                "base_link": self.base_link,
                "wrist_link": self.wrist_link,
                "camera_serial": episode.info.get("camera_serial"),
                "provenance": {
                    "mode": "offline_teleop_episode",
                    "episode": episode.name,
                    "task_dir": str(self.task_dir),
                    "data_json": str(episode.data_path),
                    "rgbd_calibration": str(self.rgbd_calib_path),
                    "camera_serial": episode.info.get("camera_serial"),
                    "frame_indices": [frame.index for frame in episode.frames],
                    "representative_frame": representative.index,
                    "preview_frame_idx": representative.index,
                    "representative_rgb": str(representative.color_path),
                    "timestamps": representative.timestamps,
                    "pixel": [u, v],
                    "burst_frames_used": len(vals),
                    "valid_depth_frames": int(valid.size),
                    "depth_mm": z_mm,
                    "depth_spread_mm": spread,
                    "qpos_median_rad": q_median.tolist(),
                },
            }
            for key in ("confidence", "color_confidence", "circularity", "flags"):
                if key in marker:
                    observation[key] = marker[key]
            observations.append(observation)

        return {
            "ok": bool(observations),
            "episode": episode.name,
            "pose_id": episode.name,
            "observations": observations,
            "markers": observations,
            "errors": errors,
            "confirmed_count": len(observations),
            "error_count": len(errors),
            "T_base_wrist": T_base_wrist.tolist(),
            f"{self.arm_state_key}_q_median": q_median.tolist(),
            "burst_frames_used": len(episode.frames),
            "warnings": list(episode.warnings),
        }

    def confirm_points(
        self,
        name: str,
        cloud_id: str,
        stride: int,
        selections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """按可信点云顶点索引生成多 marker observation。"""
        if not isinstance(cloud_id, str) or not cloud_id.strip():
            raise EpisodeValidationError("cloud_id 必须是非空字符串")
        if not isinstance(selections, list) or not selections:
            raise EpisodeValidationError("selections 必须是非空数组")
        episode = self.load(name)
        cloud = self.point_cloud(name, stride)
        if cloud.cloud_id != cloud_id.strip():
            raise PointCloudStaleError("点云已经变化，请重新加载后再选点")

        normalized: list[dict[str, Any]] = []
        for position, selection in enumerate(selections):
            if not isinstance(selection, dict):
                raise EpisodeValidationError(
                    f"第 {position} 个 selection 必须是 JSON object"
                )
            marker_id = selection.get("id", selection.get("marker_id"))
            if not isinstance(marker_id, str) or not marker_id.strip():
                raise EpisodeValidationError(
                    f"第 {position} 个 selection 缺少非空 marker id"
                )
            try:
                color = canonical_color(selection.get("color"))
                vertex_index = int(selection["vertex_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EpisodeValidationError(
                    f"第 {position} 个 selection 的颜色或 vertex_index 不合法"
                ) from exc
            if vertex_index < 0 or vertex_index >= len(cloud.points):
                raise EpisodeValidationError(
                    f"第 {position} 个 selection 的 vertex_index {vertex_index} 越界，"
                    f"点云共有 {len(cloud.points)} 个顶点"
                )
            normalized.append(
                {
                    "marker_id": marker_id.strip(),
                    "color": color,
                    "vertex_index": vertex_index,
                }
            )

        marker_ids = [item["marker_id"] for item in normalized]
        colors = [item["color"] for item in normalized]
        if len(set(marker_ids)) != len(marker_ids):
            raise EpisodeValidationError("同一 episode 中 marker id 必须唯一")
        if len(set(colors)) != len(colors):
            raise EpisodeValidationError(
                "同一 episode 中每种 canonical color 只能确认一个点"
            )

        q_median = np.median(
            np.stack([frame.arm_q for frame in episode.frames]), axis=0
        )
        T_base_wrist = self._wrist_pose(q_median)
        representative = episode.representative
        observations = []
        for selection in normalized:
            index = selection["vertex_index"]
            point = cloud.points[index].astype(float)
            pixel = cloud.pixels[index].astype(int)
            valid_frames = int(cloud.valid_depth_frames[index])
            spread = float(cloud.depth_spread_mm[index])
            depth_mm = float(point[2] * 1000.0)
            marker_id = selection["marker_id"]
            observation = {
                "schema_version": 2,
                "id": marker_id,
                "marker_id": marker_id,
                "color": selection["color"],
                "episode": episode.name,
                "pose_id": episode.name,
                "source": "point_cloud",
                "vertex_index": index,
                "cloud_id": cloud.cloud_id,
                "point_cloud_stride": cloud.stride,
                "pixel": pixel.tolist(),
                "center": pixel.astype(float).tolist(),
                "p_camera": point.tolist(),
                "T_base_wrist": T_base_wrist.tolist(),
                "depth_mm": depth_mm,
                "valid_ratio": float(valid_frames / len(episode.frames)),
                "valid_depth_frames": valid_frames,
                "depth_frame_count": len(episode.frames),
                "burst_frames_used": len(episode.frames),
                "depth_spread_mm": spread,
                f"{self.arm_state_key}_q_median": q_median.tolist(),
                "qpos_median_rad": q_median.tolist(),
                "base_link": self.base_link,
                "wrist_link": self.wrist_link,
                "camera_serial": episode.info.get("camera_serial"),
                "provenance": {
                    "mode": "offline_point_cloud",
                    "episode": episode.name,
                    "task_dir": str(self.task_dir),
                    "data_json": str(episode.data_path),
                    "rgbd_calibration": str(self.rgbd_calib_path),
                    "camera_serial": episode.info.get("camera_serial"),
                    "frame_indices": [frame.index for frame in episode.frames],
                    "representative_frame": representative.index,
                    "representative_rgb": str(representative.color_path),
                    "timestamps": representative.timestamps,
                    "cloud_id": cloud.cloud_id,
                    "point_cloud_stride": cloud.stride,
                    "vertex_index": index,
                    "pixel": pixel.tolist(),
                    "burst_frames_used": len(episode.frames),
                    "valid_depth_frames": valid_frames,
                    "depth_mm": depth_mm,
                    "depth_spread_mm": spread,
                    "qpos_median_rad": q_median.tolist(),
                },
            }
            observations.append(observation)

        return {
            "ok": True,
            "episode": episode.name,
            "pose_id": episode.name,
            "cloud_id": cloud.cloud_id,
            "point_cloud_stride": cloud.stride,
            "observations": observations,
            "markers": observations,
            "confirmed_count": len(observations),
            "error_count": 0,
            "errors": [],
            "T_base_wrist": T_base_wrist.tolist(),
            f"{self.arm_state_key}_q_median": q_median.tolist(),
            "burst_frames_used": len(episode.frames),
            "warnings": list(episode.warnings),
        }

    def confirm_mount_points(
        self,
        name: str,
        cloud_id: str,
        stride: int,
        hand: dict[str, Any],
        selections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """生成自由模型点与相机点的手安装 observation（schema v3）。"""
        if not isinstance(cloud_id, str) or not cloud_id.strip():
            raise EpisodeValidationError("cloud_id 必须是非空字符串")
        if not isinstance(selections, list) or not selections:
            raise EpisodeValidationError("selections 必须是非空数组")
        episode = self.load(name)
        cloud = self.point_cloud(name, stride)
        if cloud.cloud_id != cloud_id.strip():
            raise PointCloudStaleError("点云已经变化，请重新加载后再选点")

        normalized: list[dict[str, Any]] = []
        for position, selection in enumerate(selections):
            if not isinstance(selection, dict):
                raise EpisodeValidationError(
                    f"第 {position} 个 selection 必须是 JSON object"
                )
            point_id = selection.get("point_id", selection.get("id"))
            if not isinstance(point_id, str) or not point_id.strip():
                raise EpisodeValidationError(
                    f"第 {position} 个 selection 缺少非空 point_id"
                )
            try:
                p_hand = np.asarray(selection["p_hand"], dtype=float).reshape(3)
                p_local = np.asarray(
                    selection.get("p_local", p_hand), dtype=float
                ).reshape(3)
                vertex_index = int(selection["vertex_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EpisodeValidationError(
                    f"第 {position} 个 selection 的模型点或 vertex_index 不合法"
                ) from exc
            if not np.all(np.isfinite(p_hand)) or not np.all(np.isfinite(p_local)):
                raise EpisodeValidationError(
                    f"第 {position} 个 selection 的模型点包含非法值"
                )
            if vertex_index < 0 or vertex_index >= len(cloud.points):
                raise EpisodeValidationError(
                    f"第 {position} 个 selection 的 vertex_index {vertex_index} 越界，"
                    f"点云共有 {len(cloud.points)} 个顶点"
                )
            normalized.append(
                {
                    "point_id": point_id.strip(),
                    "label": str(selection.get("label") or point_id.strip()),
                    "link": selection.get("link"),
                    "p_local": p_local.tolist(),
                    "p_hand": p_hand.tolist(),
                    "vertex_index": vertex_index,
                }
            )
        point_ids = [item["point_id"] for item in normalized]
        if len(set(point_ids)) != len(point_ids):
            raise EpisodeValidationError("同一姿态中每个模型点只能确认一次")

        q_median = np.median(
            np.stack([frame.arm_q for frame in episode.frames]), axis=0
        )
        T_base_wrist = self._wrist_pose(q_median)
        representative = episode.representative
        observations = []
        for selection in normalized:
            index = selection["vertex_index"]
            point = cloud.points[index].astype(float)
            pixel = cloud.pixels[index].astype(int)
            valid_frames = int(cloud.valid_depth_frames[index])
            spread = float(cloud.depth_spread_mm[index])
            observations.append(
                {
                    "schema_version": 3,
                    "point_id": selection["point_id"],
                    "label": selection["label"],
                    "link": selection["link"],
                    "hand_id": hand["hand_id"],
                    "hand_base_link": hand["base_link"],
                    "hand_joints": hand["joints"],
                    "p_local": selection["p_local"],
                    "p_hand": selection["p_hand"],
                    "episode": episode.name,
                    "pose_id": episode.name,
                    "arm": self.arm,
                    "source": "point_cloud",
                    "vertex_index": index,
                    "cloud_id": cloud.cloud_id,
                    "point_cloud_stride": cloud.stride,
                    "pixel": pixel.tolist(),
                    "p_camera": point.tolist(),
                    "T_base_wrist": T_base_wrist.tolist(),
                    "depth_mm": float(point[2] * 1000.0),
                    "valid_depth_frames": valid_frames,
                    "depth_frame_count": len(episode.frames),
                    "depth_spread_mm": spread,
                    "qpos_median_rad": q_median.tolist(),
                    f"{self.arm_state_key}_q_median": q_median.tolist(),
                    "base_link": self.base_link,
                    "wrist_link": self.wrist_link,
                    "camera_serial": episode.info.get("camera_serial"),
                    "provenance": {
                        "mode": "offline_point_cloud_mount",
                        "episode": episode.name,
                        "task_dir": str(self.task_dir),
                        "data_json": str(episode.data_path),
                        "rgbd_calibration": str(self.rgbd_calib_path),
                        "camera_serial": episode.info.get("camera_serial"),
                        "frame_indices": [frame.index for frame in episode.frames],
                        "representative_frame": representative.index,
                        "representative_rgb": str(representative.color_path),
                        "cloud_id": cloud.cloud_id,
                        "point_cloud_stride": cloud.stride,
                        "vertex_index": index,
                        "pixel": pixel.tolist(),
                        "qpos_median_rad": q_median.tolist(),
                    },
                }
            )

        return {
            "ok": True,
            "episode": episode.name,
            "pose_id": episode.name,
            "arm": self.arm,
            "cloud_id": cloud.cloud_id,
            "point_cloud_stride": cloud.stride,
            "observations": observations,
            "confirmed_count": len(observations),
            "T_base_wrist": T_base_wrist.tolist(),
            "qpos_median_rad": q_median.tolist(),
            "burst_frames_used": len(episode.frames),
            "warnings": list(episode.warnings),
        }

    def confirm_markers(
        self, name: str, edited_markers: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self.pick_many(name, edited_markers)

    def pick(self, name: str, u: int, v: int) -> dict[str, Any]:
        episode = self.load(name)
        color_h, color_w = self.calibration.color_shape
        if not (0 <= u < color_w and 0 <= v < color_h):
            raise EpisodeValidationError(
                f"像素越界 ({u},{v})，RGB 图像为 {color_w}x{color_h}"
            )

        vals = np.asarray(self._aligned_depth_stack(episode)[:, v, u], dtype=float)
        valid = vals[
            np.isfinite(vals)
            & (vals > DEPTH_VALID_MIN_MM)
            & (vals < DEPTH_VALID_MAX_MM)
        ]
        required = max(3, len(vals) // 2)
        if valid.size < required:
            raise EpisodeValidationError(
                f"该像素没有稳定深度（{valid.size}/{len(vals)} 帧有效）。"
                "细小/深色/反光表面双目常测不到，请稍微挪一点再点"
            )
        spread = float(np.max(valid) - np.min(valid))
        if spread > DEPTH_MAX_SPREAD_MM:
            raise EpisodeValidationError(
                f"该像素深度在多帧间跳动 {spread:.0f}mm（边缘闪烁），稍微挪一点再点"
            )

        z_mm = float(np.median(valid))
        z = z_mm / 1000.0
        fx, fy, cx, cy = self.calibration.color_intrinsics
        p_camera = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)
        q_median = np.median(
            np.stack([frame.arm_q for frame in episode.frames]), axis=0
        )
        T_base_wrist = self._wrist_pose(q_median)
        representative = episode.representative
        return {
            "ok": True,
            "episode": episode.name,
            "pixel": [u, v],
            "p_camera": p_camera.tolist(),
            "depth_mm": z_mm,
            "valid_ratio": float(valid.size / len(vals)),
            "valid_depth_frames": int(valid.size),
            "depth_frame_count": len(vals),
            "burst_frames_used": len(vals),
            "depth_spread_mm": spread,
            f"{self.arm_state_key}_q_median": q_median.tolist(),
            "qpos_median_rad": q_median.tolist(),
            "T_base_wrist": T_base_wrist.tolist(),
            "base_link": self.base_link,
            "wrist_link": self.wrist_link,
            "camera_serial": episode.info.get("camera_serial"),
            "warnings": list(episode.warnings),
            "provenance": {
                "mode": "offline_teleop_episode",
                "episode": episode.name,
                "task_dir": str(self.task_dir),
                "data_json": str(episode.data_path),
                "rgbd_calibration": str(self.rgbd_calib_path),
                "camera_serial": episode.info.get("camera_serial"),
                "frame_indices": [frame.index for frame in episode.frames],
                "representative_frame": representative.index,
                "preview_frame_idx": representative.index,
                "representative_rgb": str(representative.color_path),
                "timestamps": representative.timestamps,
                "pixel": [u, v],
                "burst_frames_used": len(vals),
                "valid_depth_frames": int(valid.size),
                "depth_mm": z_mm,
                "depth_spread_mm": spread,
                "qpos_median_rad": q_median.tolist(),
            },
        }
