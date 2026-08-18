"""FastAPI app：彩色流/离线点云 + P_camera + 手腕位姿配对 + 联合解算。

每个样本 = P_camera（点击反投影）+ T_base^wrist（自动读取或手填 xyz+rpy）。
解算联合估计 T_base^camera 和指尖偏移 p_tool（腕系），不需要事先量偏移。
样本落盘为 <save_path>/samples/NNNN.json，重启不丢。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .camera import CameraBase, MockCamera
from .offline import (
    DEFAULT_RGBD_CALIB_PATH,
    EpisodeValidationError,
    OfflineEpisodeBackend,
    PointCloudStaleError,
    RIGHT_ARM_DATASET_JOINTS,
)
from .markers import CANONICAL_COLORS, canonical_color, marker_catalog_public
from .robot import ManualPoseProvider, PoseProvider
from .rgbd import RGBDCalibration
from .solver import (
    MIN_SAMPLES_PIVOT,
    MIN_SAMPLES_TOOL,
    MIN_SAMPLES_TOOL_ONLY,
    leave_one_pose_out_multi,
    leave_one_out_pivot,
    leave_one_out_tool,
    make_T,
    rpy_to_rot,
    solve_pivot,
    solve_multi_marker,
    solve_tool_fixed_cam,
    solve_with_tool_offset,
)

# --------------- 注入的全局状态 ---------------

camera: CameraBase = MockCamera()
pose_provider: PoseProvider = ManualPoseProvider()
arm_factory = None      # run_server 传 --arm-control 时注入（工厂，点「获取控制」才创建）
arm_controller = None   # 当前接管中的 H2ArmController（None = 未接管）
arm_lock = threading.Lock()
save_path: Path = Path("./handeye3d_data")
offline_backend: OfflineEpisodeBackend | None = None
teleop_task_dir: Path | None = None
record_task_dir: Path | None = None
rgbd_calib_path: Path = DEFAULT_RGBD_CALIB_PATH
samples_lock = threading.Lock()
record_lock = threading.Lock()

app = FastAPI(title="Hand-Eye 3D (point + wrist-pose) Calibration")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def init_state() -> None:
    (save_path / "samples").mkdir(parents=True, exist_ok=True)
    (save_path / "pivot_samples").mkdir(parents=True, exist_ok=True)
    if record_task_dir is not None:
        record_task_dir.mkdir(parents=True, exist_ok=True)


def _samples_dir() -> Path:
    return save_path / "samples"


def _pivot_dir() -> Path:
    return save_path / "pivot_samples"


def _load_samples() -> list[dict]:
    items = []
    for f in sorted(_samples_dir().glob("*.json")):
        try:
            items.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return items


def _imported_episode(sample: dict):
    value = sample.get("episode")
    source = sample.get("provenance")
    if value is None and isinstance(source, dict):
        value = source.get("episode")
    return value


def _sample_schema_version(sample: dict) -> int:
    value = sample.get("schema_version", 1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _next_index() -> int:
    used = [int(p.stem) for p in _samples_dir().glob("*.json") if p.stem.isdigit()]
    return (max(used) + 1) if used else 0


def _active_pose_source() -> str:
    return "offline_teleop_episode" if offline_backend is not None else pose_provider.source


def _active_base_link() -> str:
    return offline_backend.base_link if offline_backend is not None else pose_provider.base_link


def _active_wrist_link() -> str:
    return offline_backend.wrist_link if offline_backend is not None else pose_provider.wrist_link


def _active_camera_info() -> dict:
    if offline_backend is None:
        return camera.info()
    color_h, color_w = offline_backend.calibration.color_shape
    return {
        "source": "offline_teleop_episode",
        "serial": offline_backend.calibration.serial,
        "name": "Recorded Orbbec RGB-D",
        "width": color_w,
        "height": color_h,
    }


def _recorded_episode_count() -> int:
    if record_task_dir is None or not record_task_dir.is_dir():
        return 0
    return sum(1 for path in record_task_dir.glob("episode_*/data.json") if path.is_file())


# --------------- 状态 / 相机 ---------------


@app.get("/api/status")
async def api_status():
    return {
        "mode": "offline" if offline_backend is not None else "live",
        "camera": _active_camera_info(),
        "pose_source": _active_pose_source(),
        "pose_auto": False if offline_backend is not None else pose_provider.available,
        "base_link": _active_base_link(),
        "wrist_link": _active_wrist_link(),
        "save_path": str(save_path),
        "sample_count": len(_load_samples()),
        "min_samples": MIN_SAMPLES_TOOL,
        "teleop_task_dir": str(teleop_task_dir) if teleop_task_dir else None,
        "rgbd_calib": str(rgbd_calib_path),
        "recording": {
            "enabled": (
                offline_backend is None
                and record_task_dir is not None
                and bool(_active_camera_info().get("recording_supported"))
                and callable(getattr(pose_provider, "read_arm_q", None))
            ),
            "task_dir": str(record_task_dir) if record_task_dir else None,
            "episode_count": _recorded_episode_count(),
            "frame_count": 5,
        },
        "offline": {
            "enabled": offline_backend is not None,
            "teleop_task_dir": str(teleop_task_dir) if teleop_task_dir else None,
            "rgbd_calib": str(rgbd_calib_path),
            "serial_mismatch_policy": "warning",
        },
    }


def _next_record_episode_name() -> str:
    assert record_task_dir is not None
    used = []
    for path in record_task_dir.glob("episode_*"):
        if not path.is_dir():
            continue
        try:
            used.append(int(path.name.removeprefix("episode_")))
        except ValueError:
            continue
    return f"episode_{((max(used) + 1) if used else 0):04d}"


def _record_episode(frame_count: int) -> dict:
    if offline_backend is not None:
        raise RuntimeError("离线处理模式不能继续拍摄")
    if record_task_dir is None:
        raise RuntimeError("未配置离线数据保存目录")
    read_arm_q = getattr(pose_provider, "read_arm_q", None)
    if not callable(read_arm_q):
        raise RuntimeError("离线拍摄需要 --pose-source h2，以同步保存右臂关节角")
    calibration = RGBDCalibration.from_file(rgbd_calib_path)
    camera_info = _active_camera_info()
    if not camera_info.get("recording_supported"):
        raise RuntimeError("当前相机来源不支持原始 RGB-D 落盘；请使用 --camera-source orbbec")
    if (
        calibration.serial
        and camera_info.get("serial")
        and str(camera_info["serial"]) != calibration.serial
    ):
        raise RuntimeError(
            f"相机序列号 {camera_info['serial']} 与 RGB-D 标定 {calibration.serial} 不一致"
        )

    episode_name = _next_record_episode_name()
    final_root = record_task_dir / episode_name
    temp_root = record_task_dir / f".{episode_name}-{uuid.uuid4().hex}.tmp"
    rgb_dir = temp_root / "rgb"
    depth_dir = temp_root / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir()

    rows = []
    sequence = -1
    try:
        for index in range(frame_count):
            frame = camera.wait_record_frame(sequence, timeout_s=2.0)
            sequence = int(frame["sequence"])
            color = np.asarray(frame["color_bgr"], dtype=np.uint8)
            depth = np.asarray(frame["depth_z16"])
            q = np.asarray(read_arm_q(), dtype=float).reshape(-1)
            if color.shape[:2] != calibration.color_shape or color.ndim != 3:
                raise RuntimeError(
                    f"SDK 彩色帧尺寸 {color.shape[:2]} 与标定 "
                    f"{calibration.color_shape} 不一致"
                )
            if depth.dtype != np.uint16 or depth.shape != calibration.depth_shape:
                raise RuntimeError(
                    f"SDK 原始深度应为 uint16 {calibration.depth_shape}，"
                    f"实际为 {depth.dtype} {depth.shape}"
                )
            if q.shape != (7,) or not np.all(np.isfinite(q)):
                raise RuntimeError(f"H2 右臂关节角应为 7 个有限数值，实际 shape={q.shape}")
            depth_scale_mm = float(frame["depth_scale_mm"])
            if not np.isclose(
                depth_scale_mm, calibration.depth_scale_mm, rtol=0.0, atol=1e-6
            ):
                raise RuntimeError(
                    f"SDK depth scale {depth_scale_mm} mm 与标定 "
                    f"{calibration.depth_scale_mm} mm 不一致"
                )

            rgb_rel = Path("rgb") / f"{index:06d}_head_rgb.jpg"
            depth_rel = Path("depth") / f"{index:06d}_head_depth.npy"
            if not cv2.imwrite(
                str(temp_root / rgb_rel), color, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"无法写入彩色帧 {rgb_rel}")
            np.save(temp_root / depth_rel, depth, allow_pickle=False)
            rows.append(
                {
                    "idx": index,
                    "colors": {"head_rgb": rgb_rel.as_posix()},
                    "depths": {"head_depth": depth_rel.as_posix()},
                    "states": {"right_arm": {"qpos": q.tolist()}},
                    "timestamps": {
                        "sample_timestamp_ns": int(frame["timestamp_ns"]),
                        "joint_read_timestamp_ns": time.time_ns(),
                    },
                    "rgbd": {
                        "color_shape": list(color.shape[:2]),
                        "depth_shape": list(depth.shape),
                        "color_format": "jpeg",
                        "depth_format": "depth_z16",
                        "depth_dtype": "uint16",
                        "depth_scale_mm": depth_scale_mm,
                        "alignment": "raw_depth",
                    },
                }
            )

        payload = {
            "info": {
                "version": "1.0.0",
                "kind": "hand_eye_calibration",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "frame_count": len(rows),
                "robot": "H2",
                "camera_serial": camera_info.get("serial"),
                "camera_source": camera_info.get("source"),
                "right_arm_joint_order": list(RIGHT_ARM_DATASET_JOINTS),
            },
            "data": rows,
        }
        (temp_root / "data.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temp_root.replace(final_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return {
        "ok": True,
        "episode": episode_name,
        "frame_count": len(rows),
        "path": str(final_root),
    }


@app.post("/api/record/episode")
async def api_record_episode(body: dict | None = None):
    body = body or {}
    try:
        frame_count = int(body.get("frame_count", 5))
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "frame_count 必须是整数"}, status_code=400)
    if not 3 <= frame_count <= 30:
        return JSONResponse(
            {"ok": False, "error": "frame_count 必须在 3～30 之间"}, status_code=400
        )
    if not record_lock.acquire(blocking=False):
        return JSONResponse({"ok": False, "error": "正在拍摄上一组数据"}, status_code=409)
    try:
        result = await asyncio.to_thread(_record_episode, frame_count)
        return result
    except (OSError, RuntimeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    finally:
        record_lock.release()


@app.get("/api/markers/colors")
async def api_marker_colors():
    colors = marker_catalog_public()
    return {"ok": True, "colors": colors, "count": len(colors)}


@app.get("/api/stream")
async def api_stream():
    """彩色相机 MJPEG 预览流。"""
    if offline_backend is not None:
        return JSONResponse(
            {"ok": False, "error": "离线模式没有实时相机流"},
            status_code=409,
        )

    def gen():
        while True:
            data = camera.get_jpeg()
            if data is None:
                time.sleep(0.2)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                   + data + b"\r\n")
            time.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/frame.jpg")
async def api_live_frame():
    """单帧 RGB，供不稳定支持 MJPEG 的浏览器轮询。"""
    if offline_backend is not None:
        return JSONResponse(
            {"ok": False, "error": "离线模式没有实时相机流"}, status_code=409
        )
    data = await asyncio.to_thread(camera.get_jpeg)
    if data is None:
        return JSONResponse({"ok": False, "error": "相机还没有 RGB 帧"}, status_code=503)
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


def _encode_live_depth_overlay(depth_mm: np.ndarray) -> bytes:
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"实时深度必须是二维图，实际 shape={depth.shape}")
    valid = np.isfinite(depth) & (depth >= 300.0) & (depth <= 2000.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (2000.0 - depth[valid]) * (255.0 / 1700.0), 0.0, 255.0
    ).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    overlay = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
    overlay[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("实时深度伪彩 PNG 编码失败")
    return encoded.tobytes()


@app.get("/api/depth-overlay.png")
async def api_live_depth_overlay():
    """单帧 SDK 对齐深度伪彩，和 RGB 单帧轮询配合使用。"""
    if offline_backend is not None:
        return JSONResponse(
            {"ok": False, "error": "离线模式请使用 episode 深度叠加"},
            status_code=409,
        )
    snapshot_reader = getattr(camera, "depth_preview_snapshot", camera.depth_snapshot)
    snapshot = await asyncio.to_thread(snapshot_reader)
    if snapshot is None:
        return JSONResponse({"ok": False, "error": "相机还没有深度帧"}, status_code=503)
    depth_mm, _intrinsics = snapshot
    data = await asyncio.to_thread(_encode_live_depth_overlay, depth_mm)
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/api/depth-stream")
async def api_depth_stream():
    """SDK 对齐到彩色分辨率后的实时深度伪彩 PNG 流。"""
    if offline_backend is not None:
        return JSONResponse(
            {"ok": False, "error": "离线模式请使用 episode 深度叠加"},
            status_code=409,
        )

    def gen():
        snapshot_reader = getattr(
            camera, "depth_preview_snapshot", camera.depth_snapshot
        )
        while True:
            snapshot = snapshot_reader()
            if snapshot is None:
                time.sleep(0.1)
                continue
            depth_mm, _intrinsics = snapshot
            try:
                data = _encode_live_depth_overlay(depth_mm)
            except (RuntimeError, ValueError):
                time.sleep(0.1)
                continue
            yield (
                b"--frame\r\nContent-Type: image/png\r\n"
                b"Content-Length: "
                + str(len(data)).encode()
                + b"\r\n\r\n"
                + data
                + b"\r\n"
            )
            time.sleep(0.1)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/api/pick")
async def api_pick(body: dict):
    """点击像素反投影。Body: {"u": int, "v": int}，返回彩色相机系坐标（米）。"""
    try:
        u, v = int(body["u"]), int(body["v"])
    except (KeyError, TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "需要整数 u、v"}, status_code=400)
    result = await asyncio.to_thread(camera.pick, u, v)
    status = 200 if result.get("ok") else 502
    return JSONResponse(result, status_code=status)


@app.get("/api/offline/episodes")
async def api_offline_episodes():
    if offline_backend is None:
        return JSONResponse(
            {"ok": False, "error": "未配置离线遥操作任务目录，请用 --teleop-task-dir 启动"},
            status_code=409,
        )
    scanned = await asyncio.to_thread(offline_backend.scan)
    saved_samples = _load_samples()
    imported = {_imported_episode(item) for item in saved_samples}
    episodes = []
    errors = []
    for item in scanned:
        if item.get("valid"):
            episode_samples = [
                sample
                for sample in saved_samples
                if _imported_episode(sample) == item["name"]
            ]
            imported_marker_ids = sorted(
                {
                    sample["marker_id"]
                    for sample in episode_samples
                    if _sample_schema_version(sample) == 2
                    and isinstance(sample.get("marker_id"), str)
                }
            )
            item["imported_marker_ids"] = imported_marker_ids
            item["imported_marker_count"] = len(imported_marker_ids)
            item["already_imported"] = item["name"] in imported
            episodes.append(item)
        else:
            errors.append(item)
    return {
        "ok": True,
        "episodes": episodes,
        "count": len(episodes),
        "invalid_episodes": errors,
    }


@app.get("/api/offline/episodes/{name}/preview")
async def api_offline_preview(name: str):
    if offline_backend is None:
        return JSONResponse(
            {"ok": False, "error": "未配置离线遥操作任务目录，请用 --teleop-task-dir 启动"},
            status_code=409,
        )
    try:
        jpeg = await asyncio.to_thread(offline_backend.preview_jpeg, name)
    except EpisodeValidationError as exc:
        status = 404 if "元数据不存在" in str(exc) else 422
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/offline/episodes/{name}/depth-overlay")
async def api_offline_depth_overlay(name: str):
    if offline_backend is None:
        return JSONResponse(
            {"ok": False, "error": "未配置离线遥操作任务目录，请用 --teleop-task-dir 启动"},
            status_code=409,
        )
    try:
        png = await asyncio.to_thread(offline_backend.depth_overlay_png, name)
    except EpisodeValidationError as exc:
        status = 404 if "元数据不存在" in str(exc) else 422
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"深度叠加生成失败: {exc}"}, status_code=422
        )
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/offline/episodes/{name}/point-cloud.ply")
async def api_offline_point_cloud(name: str, stride: int = 2):
    if offline_backend is None:
        return JSONResponse(
            {"ok": False, "error": "未配置离线遥操作任务目录，请用 --teleop-task-dir 启动"},
            status_code=409,
        )
    try:
        ply, cloud = await asyncio.to_thread(
            offline_backend.point_cloud_ply, name, stride
        )
    except EpisodeValidationError as exc:
        status = 404 if "元数据不存在" in str(exc) else 422
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"点云生成失败: {exc}"}, status_code=422
        )
    return Response(
        content=ply,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache",
            "Content-Disposition": f'inline; filename="{name}.ply"',
            "X-Point-Cloud-Id": cloud.cloud_id,
            "X-Point-Count": str(len(cloud.points)),
            "X-Point-Cloud-Stride": str(cloud.stride),
            "Access-Control-Expose-Headers": (
                "X-Point-Cloud-Id, X-Point-Count, X-Point-Cloud-Stride"
            ),
        },
    )


@app.post("/api/offline/pick")
async def api_offline_pick(body: dict):
    if offline_backend is None:
        return JSONResponse(
            {"ok": False, "error": "未配置离线遥操作任务目录，请用 --teleop-task-dir 启动"},
            status_code=409,
        )
    try:
        name = body["episode"]
        if not isinstance(name, str):
            raise ValueError("episode 必须是字符串")
        u, v = int(body["u"]), int(body["v"])
    except (KeyError, TypeError, ValueError) as exc:
        message = str(exc) if str(exc) else "需要 episode、整数 u 和整数 v"
        return JSONResponse({"ok": False, "error": message}, status_code=400)
    try:
        result = await asyncio.to_thread(offline_backend.pick, name, u, v)
    except EpisodeValidationError as exc:
        status = 404 if "元数据不存在" in str(exc) else 422
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"离线 episode 处理失败: {exc}"}, status_code=422
        )
    return result


@app.post("/api/offline/confirm-points")
async def api_offline_confirm_points(body: dict):
    if offline_backend is None:
        return JSONResponse(
            {"ok": False, "error": "未配置离线遥操作任务目录，请用 --teleop-task-dir 启动"},
            status_code=409,
        )
    try:
        name = body["episode"]
        cloud_id = body["cloud_id"]
        stride = body.get("stride", 2)
        selections = body["selections"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("episode 必须是非空字符串")
        if not isinstance(cloud_id, str) or not cloud_id.strip():
            raise ValueError("cloud_id 必须是非空字符串")
        if isinstance(stride, bool):
            raise ValueError("stride 必须是整数")
        stride = int(stride)
        if not isinstance(selections, list):
            raise ValueError("selections 必须是数组")
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    try:
        result = await asyncio.to_thread(
            offline_backend.confirm_points,
            name.strip(),
            cloud_id.strip(),
            stride,
            selections,
        )
    except PointCloudStaleError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    except EpisodeValidationError as exc:
        status = 404 if "元数据不存在" in str(exc) else 422
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"点云选点确认失败: {exc}"}, status_code=422
        )
    return result


@app.post("/api/offline/detect-markers")
async def api_offline_detect_markers(body: dict):
    if offline_backend is None:
        return JSONResponse(
            {"ok": False, "error": "未配置离线遥操作任务目录，请用 --teleop-task-dir 启动"},
            status_code=409,
        )
    try:
        name = body["episode"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("episode 必须是非空字符串")
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    try:
        result = await asyncio.to_thread(offline_backend.detect_markers, name.strip())
    except EpisodeValidationError as exc:
        status = 404 if "元数据不存在" in str(exc) else 422
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"marker 检测失败: {exc}"}, status_code=422
        )
    return result


@app.post("/api/offline/confirm-markers")
async def api_offline_confirm_markers(body: dict):
    if offline_backend is None:
        return JSONResponse(
            {"ok": False, "error": "未配置离线遥操作任务目录，请用 --teleop-task-dir 启动"},
            status_code=409,
        )
    try:
        name = body["episode"]
        markers = body["markers"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("episode 必须是非空字符串")
        if not isinstance(markers, list):
            raise ValueError("markers 必须是数组")
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    try:
        result = await asyncio.to_thread(
            offline_backend.confirm_markers, name.strip(), markers
        )
    except EpisodeValidationError as exc:
        status = 404 if "元数据不存在" in str(exc) else 422
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"离线 episode 处理失败: {exc}"}, status_code=422
        )
    return result


@app.get("/api/wrist_pose")
async def api_wrist_pose():
    """自动读取当前手腕位姿（pose_provider 可用时）。"""
    if not pose_provider.available:
        return JSONResponse(
            {"ok": False, "error": f"pose source '{pose_provider.source}' 不支持自动读取，请手填"},
            status_code=409,
        )
    try:
        T = await asyncio.to_thread(pose_provider.read_pose)
        return {"ok": True, "T_base_wrist": np.asarray(T, dtype=float).reshape(4, 4).tolist(),
                "source": pose_provider.source}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


# --------------- 手臂点动（可选，--arm-control 时启用） ---------------


def _arm_absent():
    if arm_factory is None:
        return JSONResponse(
            {"ok": False, "error": "未启用手臂控制，启动时加 --arm-control"}, status_code=409)
    return JSONResponse(
        {"ok": False, "error": "尚未接管手臂，请先点「获取控制」"}, status_code=409)


@app.get("/api/arm/status")
async def api_arm_status():
    if arm_factory is None and arm_controller is None:
        return {"enabled": False}
    if arm_controller is None:
        return {"enabled": True, "armed": False}
    st = arm_controller.status()
    st["enabled"] = True
    st["armed"] = True
    return st


@app.post("/api/arm/engage")
def api_arm_engage():
    """获取控制：创建控制器、发布 rt/arm_sdk、在当前姿态刚性保持。真机会被接管！

    同步 def：跑在线程池里，创建控制器（DDS 握手，可能几秒）不会卡住事件循环。
    """
    global arm_controller
    if arm_factory is None:
        return _arm_absent()
    with arm_lock:
        if arm_controller is not None:
            return {"ok": True, "armed": True, "message": "已处于接管状态"}
        try:
            controller = arm_factory()
            controller.start()
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"接管失败: {exc}"}, status_code=502)
        arm_controller = controller
    print("[handeye3d] 已接管手臂，开始发布 rt/arm_sdk")
    print(f"[handeye3d] 重力前馈: {controller.describe_gravity()}")
    return {"ok": True, "armed": True, **controller.status()}


@app.post("/api/arm/disarm")
def api_arm_disarm():
    """归还控制：权重渐出、交还本体控制器。调用前请扶住手臂。"""
    global arm_controller
    with arm_lock:
        if arm_controller is None:
            return {"ok": True, "armed": False, "message": "本来就未接管"}
        controller, arm_controller = arm_controller, None
    controller.shutdown()
    print("[handeye3d] 已归还手臂控制权")
    return {"ok": True, "armed": False, "message": "已归还，控制权交还本体控制器"}


@app.post("/api/arm/enable_jog")
async def api_arm_enable_jog():
    if arm_controller is None:
        return _arm_absent()
    arm_controller.enable_jog()
    return {"ok": True, **arm_controller.status()}


@app.post("/api/arm/disable_jog")
async def api_arm_disable_jog():
    if arm_controller is None:
        return _arm_absent()
    arm_controller.disable_jog()
    return {"ok": True, **arm_controller.status()}


@app.post("/api/arm/stop")
async def api_arm_stop():
    """冻结在当前指令位并刚性保持（也用于退出卸力）。"""
    if arm_controller is None:
        return _arm_absent()
    arm_controller.stop()
    return {"ok": True, **arm_controller.status()}


@app.post("/api/arm/hand_move")
async def api_arm_hand_move():
    """卸力拖动模式：kp=0 只留阻尼，手臂会下坠，必须有人扶住！"""
    if arm_controller is None:
        return _arm_absent()
    ok = arm_controller.enter_hand_move()
    if not ok:
        return JSONResponse(
            {"ok": False, "error": "点动开启时不能进入卸力模式，请先停止点动"}, status_code=409)
    return {"ok": True, **arm_controller.status()}


@app.post("/api/arm/nudge")
async def api_arm_nudge(body: dict):
    """单关节步进。Body: {"index": int, "delta": float}（弧度）。"""
    if arm_controller is None:
        return _arm_absent()
    try:
        index = int(body["index"])
        delta = float(body["delta"])
    except (KeyError, TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "需要 index(int) 和 delta(float)"},
                            status_code=400)
    try:
        accepted = arm_controller.nudge(index, delta)
    except (IndexError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    if not accepted:
        return JSONResponse({"ok": False, "error": "点动未开启（或处于卸力模式）"},
                            status_code=409)
    return {"ok": True, **arm_controller.status()}


# --------------- 样本管理 ---------------


def _parse_wrist_pose(body: dict) -> np.ndarray:
    """接受 {"T_base_wrist": 4x4} 或 {"wrist_xyz": [3], "wrist_rpy": [3]}（弧度）。"""
    if "T_base_wrist" in body:
        T = np.asarray(body["T_base_wrist"], dtype=float).reshape(4, 4)
    elif "wrist_xyz" in body and "wrist_rpy" in body:
        xyz = [float(v) for v in body["wrist_xyz"]]
        rpy = [float(v) for v in body["wrist_rpy"]]
        T = make_T(rpy_to_rot(*rpy), xyz)
    else:
        raise ValueError("需要 T_base_wrist（4x4）或 wrist_xyz + wrist_rpy")
    if not np.all(np.isfinite(T)):
        raise ValueError("手腕位姿包含非法值")
    return T


@app.get("/api/samples")
async def api_samples():
    items = _load_samples()
    return {"samples": items, "count": len(items)}


DEPTH_MIN_M = 0.30   # 双目最近测距标称 0.25m，但 0.3m 内实测有系统偏差
                     # （20260726_235153 会话：<0.3m 的样本残差 9~17mm，>0.3m 的 4~7mm）
DEPTH_MAX_M = 1.5    # 标定时指尖不该离相机超过这个距离，超了就是点到背景/飞点


@app.post("/api/samples")
async def api_add_sample(body: dict):
    """保存一个样本。Body: {"p_camera": [3], "T_base_wrist": 4x4 或 wrist_xyz+wrist_rpy, "pixel": [u,v]?}"""
    try:
        p_cam = np.asarray(body["p_camera"], dtype=float).reshape(3)
        T_wrist = _parse_wrist_pose(body)
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    if not np.all(np.isfinite(p_cam)):
        return JSONResponse({"ok": False, "error": "p_camera 包含非法值"}, status_code=400)
    if not (DEPTH_MIN_M <= float(p_cam[2]) <= DEPTH_MAX_M):
        return JSONResponse(
            {"ok": False, "error": f"深度 {p_cam[2]:.2f}m 超出 {DEPTH_MIN_M}~{DEPTH_MAX_M}m，"
                                   "像是点到背景（边缘飞点）或离相机太近——往手指内侧一点重新点击"},
            status_code=400)

    episode = body.get("episode")
    provenance = body.get("provenance")
    if episode is not None and (not isinstance(episode, str) or not episode.strip()):
        return JSONResponse(
            {"ok": False, "error": "episode 必须是非空字符串"}, status_code=400)
    if episode is not None:
        episode = episode.strip()
    if provenance is not None and not isinstance(provenance, dict):
        return JSONResponse(
            {"ok": False, "error": "provenance 必须是 JSON object"}, status_code=400)

    with samples_lock:
        if episode is not None:
            duplicate = next(
                (sample for sample in _load_samples()
                 if _imported_episode(sample) == episode),
                None,
            )
            if duplicate is not None:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": f"episode {episode} 已导入为样本 {duplicate.get('index')}",
                        "duplicate_episode": episode,
                        "existing_index": duplicate.get("index"),
                    },
                    status_code=409,
                )
        index = _next_index()
        record = {
            "schema_version": 1,
            "index": index,
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "p_camera": p_cam.tolist(),
            "T_base_wrist": T_wrist.tolist(),
            "pixel": body.get("pixel"),
            "pose_source": _active_pose_source(),
            "camera": {
                k: _active_camera_info().get(k) for k in ("serial", "source")
            },
        }
        if episode is not None:
            record["episode"] = episode
        if provenance is not None:
            record["provenance"] = provenance
        (_samples_dir() / f"{index:04d}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False))
    return {"ok": True, "index": index, "count": len(_load_samples())}


def _validated_v2_observation(
    observation: dict, episode: str, position: int
) -> dict:
    if not isinstance(observation, dict):
        raise ValueError(f"第 {position} 个 observation 必须是 JSON object")
    if observation.get("schema_version") != 2:
        raise ValueError(f"第 {position} 个 observation 的 schema_version 必须是 2")
    marker_id = observation.get("marker_id", observation.get("id"))
    if not isinstance(marker_id, str) or not marker_id.strip():
        raise ValueError(f"第 {position} 个 observation 缺少非空 marker_id")
    marker_id = marker_id.strip()
    color = canonical_color(observation.get("color"))
    observation_episode = observation.get("episode", episode)
    if observation_episode != episode:
        raise ValueError(
            f"第 {position} 个 observation 的 episode {observation_episode!r} "
            f"与批次 {episode!r} 不一致"
        )
    pose_id = observation.get("pose_id")
    if not isinstance(pose_id, (str, int)) or not str(pose_id).strip():
        raise ValueError(f"第 {position} 个 observation 缺少非空 pose_id")
    try:
        p_cam = np.asarray(observation["p_camera"], dtype=float).reshape(3)
        T_wrist = _parse_wrist_pose(observation)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"第 {position} 个 observation 坐标/位姿不合法: {exc}") from exc
    if not np.all(np.isfinite(p_cam)):
        raise ValueError(f"第 {position} 个 observation 的 p_camera 包含非法值")
    if not (DEPTH_MIN_M <= float(p_cam[2]) <= DEPTH_MAX_M):
        raise ValueError(
            f"第 {position} 个 observation 深度 {p_cam[2]:.2f}m 超出 "
            f"{DEPTH_MIN_M}~{DEPTH_MAX_M}m"
        )
    provenance = observation.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError(f"第 {position} 个 observation 的 provenance 必须是 JSON object")

    record = dict(observation)
    record.update(
        {
            "schema_version": 2,
            "episode": episode,
            "pose_id": str(pose_id).strip(),
            "marker_id": marker_id,
            "id": marker_id,
            "color": color,
            "p_camera": p_cam.tolist(),
            "T_base_wrist": T_wrist.tolist(),
        }
    )
    return record


@app.post("/api/samples/batch")
async def api_add_samples_batch(body: dict):
    try:
        episode = body["episode"]
        observations = body["observations"]
        replace_existing = body.get("replace_existing", False)
        if not isinstance(episode, str) or not episode.strip():
            raise ValueError("episode 必须是非空字符串")
        episode = episode.strip()
        if not isinstance(observations, list) or not observations:
            raise ValueError("observations 必须是非空数组")
        if not isinstance(replace_existing, bool):
            raise ValueError("replace_existing 必须是布尔值")
        records = [
            _validated_v2_observation(observation, episode, position)
            for position, observation in enumerate(observations)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        message = str(exc) or "需要 episode 和 observations"
        return JSONResponse({"ok": False, "error": message}, status_code=400)

    marker_ids = [record["marker_id"] for record in records]
    colors = [record["color"] for record in records]
    if len(set(marker_ids)) != len(marker_ids):
        return JSONResponse(
            {"ok": False, "error": "同一批次 marker_id 必须唯一"}, status_code=400
        )
    if len(set(colors)) != len(colors):
        return JSONResponse(
            {
                "ok": False,
                "error": "同一 episode 中每种 canonical color 只能保存一个 marker",
            },
            status_code=400,
        )

    with samples_lock:
        existing = _load_samples()
        existing_v2 = [
            sample
            for sample in existing
            if _sample_schema_version(sample) == 2
            and _imported_episode(sample) == episode
        ]
        existing_keys = {
            (episode, sample.get("marker_id")): sample
            for sample in existing_v2
            if isinstance(sample.get("marker_id"), str)
        }
        existing_colors = {
            sample.get("color"): sample
            for sample in existing_v2
            if sample.get("color") in CANONICAL_COLORS
        }
        if not replace_existing:
            for record in records:
                duplicate = existing_keys.get((episode, record["marker_id"]))
                if duplicate is not None:
                    return JSONResponse(
                        {
                            "ok": False,
                            "error": f"episode {episode} 的 marker "
                            f"{record['marker_id']} 已导入为样本 {duplicate.get('index')}",
                            "duplicate_key": [episode, record["marker_id"]],
                            "existing_index": duplicate.get("index"),
                        },
                        status_code=409,
                    )
                same_color = existing_colors.get(record["color"])
                if same_color is not None:
                    return JSONResponse(
                        {
                            "ok": False,
                            "error": f"episode {episode} 已保存 {record['color']} marker "
                            f"为样本 {same_color.get('index')}",
                            "duplicate_color": record["color"],
                            "existing_index": same_color.get("index"),
                        },
                        status_code=409,
                    )

        next_index = _next_index()
        indices: list[int] = []
        updated_count = 0
        for record in records:
            previous = existing_colors.get(record["color"]) if replace_existing else None
            if previous is not None:
                indices.append(int(previous["index"]))
                updated_count += 1
            else:
                indices.append(next_index)
                next_index += 1
        now = datetime.now().isoformat(timespec="seconds")
        token = uuid.uuid4().hex
        staged: list[tuple[Path, Path]] = []
        installed: list[Path] = []
        previous_contents: dict[Path, str | None] = {}
        try:
            for index, record in zip(indices, records):
                saved = dict(record)
                saved.update(
                    {
                        "index": index,
                        "datetime": now,
                        "pose_source": _active_pose_source(),
                        "camera": {
                            key: _active_camera_info().get(key)
                            for key in ("serial", "source")
                        },
                    }
                )
                final_path = _samples_dir() / f"{index:04d}.json"
                temp_path = _samples_dir() / f".batch-{token}-{index:04d}.tmp"
                previous_contents[final_path] = (
                    final_path.read_text(encoding="utf-8") if final_path.exists() else None
                )
                temp_path.write_text(
                    json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                staged.append((temp_path, final_path))
            for temp_path, final_path in staged:
                temp_path.replace(final_path)
                installed.append(final_path)
        except OSError as exc:
            for temp_path, _ in staged:
                temp_path.unlink(missing_ok=True)
            for final_path in installed:
                previous = previous_contents.get(final_path)
                if previous is None:
                    final_path.unlink(missing_ok=True)
                else:
                    final_path.write_text(previous, encoding="utf-8")
            return JSONResponse(
                {"ok": False, "error": f"批量保存失败，未保留部分结果: {exc}"},
                status_code=500,
            )
    return {
        "ok": True,
        "episode": episode,
        "indices": indices,
        "saved_count": len(indices),
        "updated_count": updated_count,
        "count": len(_load_samples()),
    }


@app.delete("/api/samples/{index}")
async def api_delete_sample(index: int):
    f = _samples_dir() / f"{index:04d}.json"
    if not f.exists():
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    f.unlink()
    return {"ok": True, "count": len(_load_samples())}


def _find_latest_calib() -> Path | None:
    """找最新一份手眼标定结果：先看本会话目录，再翻数据根目录下各时间戳会话。"""
    candidates = [save_path / "handeye3d_result.json"]
    parent = save_path.parent
    if parent.is_dir():
        candidates += list(parent.glob("*/handeye3d_result.json"))
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


# --------------- 只解指尖偏移（固定相机外参，点击指尖尖端采样） ---------------


@app.post("/api/solve_tool")
async def api_solve_tool(body: dict | None = None):
    """固定 T_base^camera，只解 p_tool。样本 = 点击指尖尖端 + 腕位姿（与联合解共用）。

    Body 可选: {"calib_path": "..."}，默认自动用最新一份 handeye3d_result.json。
    """
    body = body or {}
    calib_path = Path(body["calib_path"]) if body.get("calib_path") else _find_latest_calib()
    if calib_path is None or not calib_path.is_file():
        return JSONResponse(
            {"ok": False, "error": "找不到已有的手眼标定结果（handeye3d_result.json），"
                                   "请先做一次联合解算或指定 calib_path"},
            status_code=400)
    try:
        calib = json.loads(calib_path.read_text())
        R = np.asarray(calib["R_cam2base"], dtype=float).reshape(3, 3)
        t = np.asarray(calib["t_cam2base_m"], dtype=float).reshape(3)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": f"标定文件无法解析: {exc}"}, status_code=400)

    samples = _load_samples()
    if len(samples) < MIN_SAMPLES_TOOL_ONLY:
        return JSONResponse(
            {"ok": False, "error": f"至少 {MIN_SAMPLES_TOOL_ONLY} 个样本，当前 {len(samples)} 个"},
            status_code=400)
    p_cam = np.array([s["p_camera"] for s in samples])
    T_wrist = np.array([s["T_base_wrist"] for s in samples])
    indices = [s["index"] for s in samples]

    # 自动剔除离群样本（飞点/采样时手臂在动）：反复"解算→踢掉最差的"，
    # 直到最差残差可接受或只剩下限个样本。被剔除的会如实报告。
    keep = np.arange(len(samples))
    dropped: list[dict] = []
    try:
        while True:
            result = await asyncio.to_thread(
                solve_tool_fixed_cam, p_cam[keep], T_wrist[keep], R, t)
            errs = np.asarray(result["residual_mm"]["per_sample"], dtype=float)
            worst = int(np.argmax(errs))
            median = float(np.median(errs))
            if len(keep) <= MIN_SAMPLES_TOOL_ONLY or \
                    errs[worst] <= max(30.0, 5.0 * median):
                break
            dropped.append({"index": indices[keep[worst]],
                            "residual_mm": round(float(errs[worst]), 1)})
            keep = np.delete(keep, worst)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    result["ok"] = True
    result["sample_indices"] = [indices[i] for i in keep]
    result["dropped_samples"] = dropped
    result["solved_at"] = datetime.now().isoformat(timespec="seconds")
    result["calib_used"] = str(calib_path)
    result["base_link"] = _active_base_link()
    result["wrist_link"] = _active_wrist_link()

    old = np.asarray(calib.get("p_tool_wrist_m", []), dtype=float)
    new = np.asarray(result["p_tool_wrist_m"], dtype=float)
    if old.shape == (3,):
        result["delta_vs_calib_mm"] = ((new - old) * 1000.0).tolist()
        result["delta_vs_calib_norm_mm"] = float(np.linalg.norm(new - old) * 1000.0)

    # 生成替换了 p_tool 的完整标定文件，可直接给 reach_server --calib 用
    merged = dict(calib)
    merged["p_tool_wrist_m"] = result["p_tool_wrist_m"]
    merged["p_tool_source"] = "tool_only_fixed_cam"
    merged["tool_solved_at"] = result["solved_at"]
    merged_path = save_path / "handeye3d_result_tool.json"
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    result["merged_calib"] = str(merged_path)

    out = save_path / "tool_result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    result["saved_to"] = str(out)
    return result


# --------------- 指尖尖点标定（pivot：多姿态触同一固定点，只用 FK） ---------------


def _load_pivot_samples() -> list[dict]:
    items = []
    for f in sorted(_pivot_dir().glob("*.json")):
        try:
            items.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return items


@app.get("/api/pivot/samples")
async def api_pivot_samples():
    items = _load_pivot_samples()
    return {"samples": items, "count": len(items), "min_samples": MIN_SAMPLES_PIVOT}


@app.post("/api/pivot/samples")
async def api_pivot_add(body: dict | None = None):
    """记录一个尖点样本 = 当前手腕位姿（指尖此刻顶着那个固定点）。

    默认自动读 DDS 位姿；也接受手填 {"T_base_wrist": 4x4} / {"wrist_xyz","wrist_rpy"}。
    """
    body = body or {}
    try:
        if "T_base_wrist" in body or "wrist_xyz" in body:
            T = _parse_wrist_pose(body)
        else:
            if not pose_provider.available:
                return JSONResponse(
                    {"ok": False, "error": f"pose source '{pose_provider.source}' 不支持自动读取，请手填"},
                    status_code=409)
            T = np.asarray(await asyncio.to_thread(pose_provider.read_pose),
                           dtype=float).reshape(4, 4)
    except (ValueError, TypeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"读取手腕位姿失败: {exc}"}, status_code=503)

    used = [int(p.stem) for p in _pivot_dir().glob("*.json") if p.stem.isdigit()]
    index = (max(used) + 1) if used else 0
    record = {
        "index": index,
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "T_base_wrist": T.tolist(),
        "pose_source": pose_provider.source,
    }
    (_pivot_dir() / f"{index:04d}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False))
    return {"ok": True, "index": index, "count": len(_load_pivot_samples())}


@app.delete("/api/pivot/samples/{index}")
async def api_pivot_delete(index: int):
    f = _pivot_dir() / f"{index:04d}.json"
    if not f.exists():
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    f.unlink()
    return {"ok": True, "count": len(_load_pivot_samples())}


@app.post("/api/pivot/clear")
async def api_pivot_clear():
    for f in _pivot_dir().glob("*.json"):
        f.unlink()
    return {"ok": True, "count": 0}


@app.post("/api/pivot/solve")
async def api_pivot_solve():
    samples = _load_pivot_samples()
    if len(samples) < MIN_SAMPLES_PIVOT:
        return JSONResponse(
            {"ok": False, "error": f"尖点标定至少 {MIN_SAMPLES_PIVOT} 个姿态，当前 {len(samples)} 个"},
            status_code=400)
    T_wrist = np.array([s["T_base_wrist"] for s in samples])
    try:
        result = await asyncio.to_thread(solve_pivot, T_wrist)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    loo = await asyncio.to_thread(leave_one_out_pivot, T_wrist)
    result["leave_one_out_mm"] = loo
    finite = [e for e in loo if np.isfinite(e)]
    if finite:
        result["leave_one_out_stats_mm"] = {
            "mean": float(np.mean(finite)), "max": float(np.max(finite)),
        }
    result["ok"] = True
    result["sample_indices"] = [s["index"] for s in samples]
    result["solved_at"] = datetime.now().isoformat(timespec="seconds")
    result["base_link"] = _active_base_link()
    result["wrist_link"] = _active_wrist_link()

    # 与现有手眼标定的 p_tool 对比（若有），并生成一份"替换了 p_tool 的完整
    # 标定文件"（handeye3d_result_pivot.json），可直接给 reach_server --calib 用
    handeye = _find_latest_calib()
    if handeye is not None:
        try:
            he = json.loads(handeye.read_text())
            old = np.asarray(he.get("p_tool_wrist_m", []), dtype=float)
            new = np.asarray(result["p_tool_wrist_m"], dtype=float)
            if old.shape == (3,):
                result["delta_vs_handeye_mm"] = ((new - old) * 1000.0).tolist()
                result["delta_vs_handeye_norm_mm"] = float(np.linalg.norm(new - old) * 1000.0)
            he["p_tool_wrist_m"] = result["p_tool_wrist_m"]
            he["p_tool_source"] = "pivot"
            he["pivot_solved_at"] = result["solved_at"]
            merged = save_path / "handeye3d_result_pivot.json"
            merged.write_text(json.dumps(he, indent=2, ensure_ascii=False))
            result["merged_calib"] = str(merged)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    out = save_path / "pivot_result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    result["saved_to"] = str(out)
    return result


# --------------- 解算 ---------------


@app.post("/api/solve")
async def api_solve():
    samples = _load_samples()
    versions = {_sample_schema_version(sample) for sample in samples}
    if not samples:
        return JSONResponse(
            {"ok": False, "error": "没有可用于联合解的样本"},
            status_code=400)
    if not versions.issubset({1, 2}):
        return JSONResponse(
            {
                "ok": False,
                "error": f"样本包含不支持的 schema_version: {sorted(versions)}",
            },
            status_code=400,
        )
    if versions == {1, 2}:
        return JSONResponse(
            {
                "ok": False,
                "error": "样本同时包含 legacy v1 和 multi-marker v2，不能混合解算；"
                "请分开保存到不同标定会话",
            },
            status_code=400,
        )

    try:
        p_cam = np.array([s["p_camera"] for s in samples])
        T_wrist = np.array([s["T_base_wrist"] for s in samples])
        if versions == {2}:
            marker_labels = [canonical_color(s["color"]) for s in samples]
            pose_ids = [s["pose_id"] for s in samples]
            result = await asyncio.to_thread(
                solve_multi_marker, p_cam, T_wrist, marker_labels, pose_ids
            )
            result["leave_one_pose_out"] = await asyncio.to_thread(
                leave_one_pose_out_multi,
                p_cam,
                T_wrist,
                marker_labels,
                pose_ids,
            )
            for residual, sample in zip(
                result["per_observation_residuals"], samples
            ):
                residual["sample_index"] = sample.get("index")
                residual["marker_id"] = sample.get("marker_id")
                residual["color"] = sample.get("color")
        else:
            if len(samples) < MIN_SAMPLES_TOOL:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": f"联合解至少 {MIN_SAMPLES_TOOL} 个样本，"
                        f"当前 {len(samples)} 个",
                    },
                    status_code=400,
                )
            result = await asyncio.to_thread(
                solve_with_tool_offset, p_cam, T_wrist
            )
            loo = await asyncio.to_thread(leave_one_out_tool, p_cam, T_wrist)
            result["leave_one_out_mm"] = loo
            finite = [e for e in loo if np.isfinite(e)]
            if finite:
                result["leave_one_out_stats_mm"] = {
                    "mean": float(np.mean(finite)), "max": float(np.max(finite)),
                }
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    result["ok"] = True
    result["sample_indices"] = [s["index"] for s in samples]
    result["schema_version"] = 2 if versions == {2} else 1
    result["solved_at"] = datetime.now().isoformat(timespec="seconds")
    result["base_link"] = _active_base_link()
    result["wrist_link"] = _active_wrist_link()
    result["camera"] = _active_camera_info()

    out = save_path / "handeye3d_result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    result["saved_to"] = str(out)
    return result
