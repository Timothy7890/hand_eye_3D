"""手安装标定（T_wrist2hand）API：手型号目录、模型点配对样本与解算。

两步解的第二步：T_cam2base 用已有手眼标定结果固定，这里只解
腕 → 手基座 的 6 自由度安装变换。样本 schema_version=3，独立存放在
<save_path>/mount_samples/，不与 v1/v2 标记样本混用。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .hand_hold import HOLD_SIDES, VENDOR_DEVICE_IDS, HandHoldError
from .hands import HandCatalogError, get_hand_model, hand_catalog
from .offline import EpisodeValidationError, PointCloudStaleError
from .paths import PROJECT_ROOT
from .solver import (
    MIN_MOUNT_POINTS,
    leave_one_pose_out_mount,
    solve_hand_mount,
    solve_rigid_transform,
)

router = APIRouter()

MOUNT_DEPTH_MIN_M = 0.30
MOUNT_DEPTH_MAX_M = 1.5
MOUNT_POINT_ID_RE = re.compile(r"^(?:palm-red|back-green)-(?:0[1-8])$")
MOUNT_PROFILE_POINT_IDS = tuple(
    [f"palm-red-{index:02d}" for index in range(1, 9)]
    + [f"back-green-{index:02d}" for index in range(1, 9)]
)
MOUNT_PROFILE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
MOUNT_PROFILE_NAME_MAX_LENGTH = 128
MOUNT_PROFILE_LABEL_MAX_LENGTH = 128
mount_profile_lock = threading.Lock()


def _state():
    """app 模块的注入状态（save_path / episode backend 等），运行时读取。"""
    from . import app as app_state

    return app_state


def _mount_dir() -> Path:
    directory = _state().save_path / "mount_samples"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _mount_profile_dir() -> Path:
    directory = _state().mount_profile_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _mount_profile_id(hand_id: str, name: str) -> str:
    identity = json.dumps(
        [hand_id, name], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def _validated_mount_profile_id(profile_id: str) -> str:
    if not isinstance(profile_id, str) or not MOUNT_PROFILE_ID_RE.fullmatch(profile_id):
        raise ValueError("profile_id 必须是服务端生成的 64 位十六进制 ID")
    return profile_id


def _validated_profile_vector(value: Any, field: str, position: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"第 {position} 个点的 {field} 必须是 3 个有限数")
    if any(
        isinstance(component, bool) or not isinstance(component, (int, float))
        for component in value
    ):
        raise ValueError(f"第 {position} 个点的 {field} 必须是 3 个有限数")
    normalized = [float(component) for component in value]
    if not all(np.isfinite(component) for component in normalized):
        raise ValueError(f"第 {position} 个点的 {field} 必须是 3 个有限数")
    return normalized


def _validated_mount_profile(body: dict) -> tuple[str, str, list[dict[str, Any]]]:
    if not isinstance(body, dict):
        raise ValueError("请求体必须是 JSON object")
    if body.get("schema_version", 1) != 1:
        raise ValueError("schema_version 必须是 1")

    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name 必须是非空字符串")
    name = name.strip()
    if len(name) > MOUNT_PROFILE_NAME_MAX_LENGTH:
        raise ValueError(f"name 最长 {MOUNT_PROFILE_NAME_MAX_LENGTH} 个字符")

    hand_id = body.get("hand_id")
    if not isinstance(hand_id, str) or not hand_id.strip():
        raise ValueError("hand_id 必须是非空字符串")
    hand_id = hand_id.strip()
    model = get_hand_model(hand_id)
    metadata = model.metadata()
    valid_links = {
        item.get("link")
        for item in metadata.get("links", [])
        if isinstance(item, dict) and isinstance(item.get("link"), str)
    }

    points = body.get("points")
    if not isinstance(points, list) or not 1 <= len(points) <= len(MOUNT_PROFILE_POINT_IDS):
        raise ValueError("points 必须包含 1–16 个模型点")
    normalized_by_id: dict[str, dict[str, Any]] = {}
    for position, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"第 {position} 个点必须是 JSON object")
        point_id = _validate_mount_point_id(point.get("point_id"), position)
        if point_id in normalized_by_id:
            raise ValueError(f"point_id {point_id} 重复")
        label = point.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"第 {position} 个点的 label 必须是非空字符串")
        label = label.strip()
        if len(label) > MOUNT_PROFILE_LABEL_MAX_LENGTH:
            raise ValueError(
                f"第 {position} 个点的 label 最长 "
                f"{MOUNT_PROFILE_LABEL_MAX_LENGTH} 个字符"
            )
        link = point.get("link")
        if not isinstance(link, str) or link not in valid_links:
            raise ValueError(
                f"第 {position} 个点的 link {link!r} 不属于手型号 "
                f"{hand_id} 的零位模型 metadata links"
            )
        normalized_by_id[point_id] = {
            "point_id": point_id,
            "label": label,
            "link": link,
            "p_local": _validated_profile_vector(
                point.get("p_local"), "p_local", position
            ),
            "p_hand": _validated_profile_vector(
                point.get("p_hand"), "p_hand", position
            ),
        }
    normalized = [
        normalized_by_id[point_id]
        for point_id in MOUNT_PROFILE_POINT_IDS
        if point_id in normalized_by_id
    ]
    return name, hand_id, normalized


def _read_mount_profile(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("方案文件根节点必须是 JSON object")
    try:
        name, hand_id, points = _validated_mount_profile(payload)
    except (HandCatalogError, KeyError) as exc:
        message = str(exc.args[0]) if isinstance(exc, KeyError) else str(exc)
        raise ValueError(message) from exc
    profile_id = payload.get("profile_id")
    if profile_id != _mount_profile_id(hand_id, name):
        raise ValueError("profile_id 与 hand_id/name 不一致")
    for field in ("created_at", "updated_at"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"{field} 必须是非空字符串")
    return {
        "schema_version": 1,
        "profile_id": profile_id,
        "name": name,
        "hand_id": hand_id,
        "points": points,
        "point_count": len(points),
        "complete": len(points) == len(MOUNT_PROFILE_POINT_IDS),
        "created_at": payload["created_at"],
        "updated_at": payload["updated_at"],
    }


def _write_mount_profile_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.parent / f".{path.stem}-{uuid.uuid4().hex}.tmp"
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_mount_samples() -> list[dict]:
    items = []
    for f in sorted(_mount_dir().glob("*.json")):
        try:
            items.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return items


def _next_mount_index() -> int:
    used = [int(p.stem) for p in _mount_dir().glob("*.json") if p.stem.isdigit()]
    return (max(used) + 1) if used else 0


def _camera_metadata(calib: dict, calib_path: Path) -> dict[str, Any]:
    """读取外参同批次的 RGB 内参元数据，用于确认物理相机与分辨率。"""
    for key in ("camera", "camera_intrinsics", "intrinsics"):
        value = calib.get(key)
        if isinstance(value, dict):
            return value
    for key in ("intrinsics_file", "intrinsics_yaml"):
        value = calib.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = calib_path.parent / path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _normalized_camera_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    shape = result.get("color_shape")
    if (
        isinstance(shape, (list, tuple))
        and len(shape) == 2
        and "width" not in result
        and "height" not in result
    ):
        result["height"], result["width"] = shape
    return result


def _load_mount_calibration(
    calib_path: Path,
    *,
    require_camera_identity: bool = True,
) -> tuple[dict, np.ndarray, np.ndarray, dict[str, Any]]:
    """加载外参，并按相机序列号及 RGB 分辨率核验其物理来源。"""
    state = _state()
    try:
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        R_cam = np.asarray(calib["R_cam2base"], dtype=float).reshape(3, 3)
        t_cam = np.asarray(calib["t_cam2base_m"], dtype=float).reshape(3)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"标定文件无法解析: {exc}") from exc
    if not np.all(np.isfinite(R_cam)) or not np.all(np.isfinite(t_cam)):
        raise ValueError("相机外参包含 NaN 或无穷值")
    if not np.allclose(R_cam.T @ R_cam, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(R_cam), 1.0, atol=1e-5
    ):
        raise ValueError("R_cam2base 不是合法旋转矩阵")

    T_value = calib.get("T_cam2base")
    if T_value is not None:
        try:
            T_cam = np.asarray(T_value, dtype=float).reshape(4, 4)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"T_cam2base 不是 4x4 矩阵: {exc}") from exc
        if not np.all(np.isfinite(T_cam)) or not np.allclose(
            T_cam[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8
        ):
            raise ValueError("T_cam2base 非法")
        if not np.allclose(T_cam[:3, :3], R_cam, atol=1e-8) or not np.allclose(
            T_cam[:3, 3], t_cam, atol=1e-8
        ):
            raise ValueError("T_cam2base 与 R_cam2base/t_cam2base_m 不一致")

    expected_base = state._active_base_link()
    calib_base = str(calib.get("base_link") or expected_base)
    if calib_base != expected_base:
        raise ValueError(
            f"相机外参基座为 {calib_base}，当前 episode/FK 基座为 {expected_base}"
        )

    metadata = _normalized_camera_metadata(_camera_metadata(calib, calib_path))
    active_camera = state._active_camera_info()
    expected_serial = str(active_camera.get("serial") or "").strip()
    calib_serial = str(metadata.get("serial") or "").strip()
    if require_camera_identity:
        missing = [
            key for key in ("serial", "width", "height")
            if metadata.get(key) in (None, "")
        ]
        if missing:
            raise ValueError(
                f"相机外参缺少身份字段: {', '.join(missing)}"
            )
        active_missing = [
            key
            for key in ("serial", "width", "height")
            if active_camera.get(key) in (None, "")
        ]
        if active_missing:
            raise ValueError(
                f"当前相机/episode 缺少身份字段: {', '.join(active_missing)}"
            )
    if expected_serial and calib_serial and calib_serial != expected_serial:
        raise ValueError(
            f"相机序列号不一致：外参 {calib_serial}，episode/RGB-D {expected_serial}"
        )
    for key in ("width", "height"):
        expected = active_camera.get(key)
        actual = metadata.get(key)
        if expected is not None and actual is not None and int(actual) != int(expected):
            raise ValueError(
                f"RGB {key} 不一致：外参 {actual}，episode/RGB-D {expected}"
            )
    identity = {
        "path": str(calib_path),
        "base_link": calib_base,
        "serial": calib_serial or expected_serial or None,
        "name": metadata.get("name") or active_camera.get("name"),
        "width": metadata.get("width") or active_camera.get("width"),
        "height": metadata.get("height") or active_camera.get("height"),
    }
    return calib, R_cam, t_cam, identity


def _mount_calibration_catalog() -> tuple[list[dict[str, Any]], str | None]:
    """列出 2D/3D 项目中的相机外参，并标记与当前 episode 是否兼容。"""
    state = _state()
    candidates: set[Path] = set()
    if state.mount_calib_path is not None:
        candidates.add(Path(state.mount_calib_path).expanduser().resolve())
    latest = state._find_latest_calib()
    if latest is not None:
        candidates.add(Path(latest).expanduser().resolve())
    for path in state.save_path.parent.glob("*/handeye3d_result.json"):
        candidates.add(path.resolve())
    handeye_2d_data = PROJECT_ROOT.parent / "hand_eye_2D" / "handeye_data"
    if handeye_2d_data.is_dir():
        for path in handeye_2d_data.glob("*/handeye_result*.json"):
            candidates.add(path.resolve())

    entries: list[dict[str, Any]] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        metadata = _normalized_camera_metadata(
            _camera_metadata(payload, path) if isinstance(payload, dict) else {}
        )
        try:
            _, _, _, identity = _load_mount_calibration(path)
            compatible = True
            error = None
        except ValueError as exc:
            compatible = False
            error = str(exc)
            identity = {
                "serial": metadata.get("serial"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "base_link": payload.get("base_link") if isinstance(payload, dict) else None,
            }
        stat = path.stat()
        source = "hand_eye_2D" if "hand_eye_2D" in path.parts else "hand_eye_3D"
        entries.append(
            {
                "path": str(path),
                "filename": path.name,
                "session": path.parent.name,
                "source": source,
                "eye": payload.get("eye") if isinstance(payload, dict) else None,
                "method": payload.get("method") if isinstance(payload, dict) else None,
                "solved_at": (
                    payload.get("solved_at") if isinstance(payload, dict) else None
                ),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(
                    timespec="seconds"
                ),
                "serial": identity.get("serial"),
                "width": identity.get("width"),
                "height": identity.get("height"),
                "base_link": identity.get("base_link"),
                "compatible": compatible,
                "error": error,
            }
        )
    entries.sort(
        key=lambda item: (bool(item["compatible"]), item["modified_at"]),
        reverse=True,
    )
    configured = (
        str(Path(state.mount_calib_path).expanduser().resolve())
        if state.mount_calib_path is not None
        else None
    )
    recommended = next(
        (
            item["path"]
            for item in entries
            if item["compatible"] and item["path"] == configured
        ),
        None,
    )
    if recommended is None:
        recommended = next(
            (item["path"] for item in entries if item["compatible"]),
            None,
        )
    return entries, recommended


def _validate_mount_point_id(point_id: Any, position: int) -> str:
    if not isinstance(point_id, str) or not MOUNT_POINT_ID_RE.fullmatch(point_id.strip()):
        raise ValueError(
            f"第 {position} 个 point_id 必须是 palm-red-01..08 "
            "或 back-green-01..08"
        )
    return point_id.strip()


async def _require_active_combo(
    hand_id: str,
) -> tuple[dict | None, str | None, JSONResponse | None]:
    state = _state()
    hint = await state.get_capability_hint()
    if not hint.get("available"):
        warning = (
            "18000 能力中心不可用，未校验激活臂/手组合："
            f"{hint.get('error') or '未配置激活组合'}"
        )
        return hint, warning, None
    active_arm = state._active_arm()
    if hint.get("arm") != active_arm:
        return hint, None, JSONResponse(
            {
                "ok": False,
                "error": f"18000 激活臂为 {hint.get('arm')}，当前 episode 为 {active_arm}",
            },
            status_code=409,
        )
    try:
        hand_side = get_hand_model(hand_id).spec.side
    except (HandCatalogError, KeyError):
        hand_side = None
    if hand_side is not None and hand_side != active_arm:
        return hint, None, JSONResponse(
            {
                "ok": False,
                "error": f"手型号 {hand_id} 属于 {hand_side} 臂，"
                f"当前 episode 为 {active_arm} 臂",
            },
            status_code=409,
        )
    if hint.get("hand_id") != hand_id:
        return hint, None, JSONResponse(
            {
                "ok": False,
                "error": f"18000 激活手型号为 {hint.get('hand_id')}，当前选择为 {hand_id}",
            },
            status_code=409,
        )
    return hint, None, None


def _active_combo_payload(capability: dict | None) -> dict[str, Any] | None:
    if not capability or not capability.get("available"):
        return None
    active = capability.get("active")
    active = active if isinstance(active, dict) else {}
    return {
        "arm": active.get("arm") or capability.get("arm"),
        "hand_id": capability.get("hand_id"),
    }


# --------------- 手型号目录与模型 ---------------


@router.get("/api/mount/calibrations")
async def api_mount_calibrations():
    calibrations, recommended = await asyncio.to_thread(
        _mount_calibration_catalog
    )
    return {
        "calibrations": calibrations,
        "count": len(calibrations),
        "recommended_path": recommended,
    }


@router.get("/api/hands")
async def api_hands():
    try:
        catalog = hand_catalog()
    except HandCatalogError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    hands = [
        {
            "hand_id": spec.hand_id,
            "label": spec.label,
            "vendor": spec.vendor,
            "side": spec.side,
            "actuated_joints": list(spec.actuated_joints),
        }
        for spec in catalog.values()
    ]
    return {"ok": True, "hands": hands, "count": len(hands)}


@router.get("/api/hands/{hand_id}/model")
async def api_hand_model(hand_id: str, joints: str | None = None):
    """手模型元数据：link 变换、mesh 摆放参数、预置特征点。默认零位。"""
    try:
        model = get_hand_model(hand_id)
    except HandCatalogError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    except KeyError as exc:
        return JSONResponse({"ok": False, "error": str(exc.args[0])}, status_code=404)
    joint_values = None
    if joints:
        try:
            joint_values = [float(v) for v in joints.split(",")]
        except ValueError:
            return JSONResponse(
                {"ok": False, "error": "joints 必须是逗号分隔的数值"}, status_code=400
            )
    try:
        payload = await asyncio.to_thread(model.metadata, joint_values)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    payload["ok"] = True
    return payload


# --------------- 模型点命名方案（独立于 episode） ---------------


@router.get("/api/mount/model-point-profiles")
async def api_mount_model_point_profiles(hand_id: str | None = None):
    if hand_id is not None:
        hand_id = hand_id.strip()
        if not hand_id:
            return JSONResponse(
                {"ok": False, "error": "hand_id 必须是非空字符串"},
                status_code=400,
            )
        try:
            get_hand_model(hand_id)
        except HandCatalogError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
        except KeyError as exc:
            return JSONResponse(
                {"ok": False, "error": str(exc.args[0])}, status_code=404
            )

    profiles: list[dict[str, Any]] = []
    invalid_profiles: list[dict[str, str]] = []
    try:
        with mount_profile_lock:
            paths = sorted(_mount_profile_dir().glob("*.json"))
            for path in paths:
                if not MOUNT_PROFILE_ID_RE.fullmatch(path.stem):
                    invalid_profiles.append(
                        {"file": path.name, "error": "文件名不是服务端生成的 profile_id"}
                    )
                    continue
                try:
                    profile = _read_mount_profile(path)
                    if profile.get("profile_id") != path.stem:
                        raise ValueError("文件内容 profile_id 与文件名不一致")
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    invalid_profiles.append({"file": path.name, "error": str(exc)})
                    continue
                if hand_id is None or profile.get("hand_id") == hand_id:
                    profiles.append(profile)
    except OSError as exc:
        return JSONResponse(
            {"ok": False, "error": f"读取模型点方案目录失败: {exc}"},
            status_code=500,
        )
    return {
        "ok": True,
        "profiles": profiles,
        "count": len(profiles),
        "invalid_profiles": invalid_profiles,
        "invalid_count": len(invalid_profiles),
    }


@router.get("/api/mount/model-point-profiles/{profile_id}")
async def api_mount_model_point_profile(profile_id: str):
    try:
        profile_id = _validated_mount_profile_id(profile_id)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    try:
        path = _mount_profile_dir() / f"{profile_id}.json"
        with mount_profile_lock:
            if not path.is_file():
                return JSONResponse(
                    {"ok": False, "error": "model point profile not found"},
                    status_code=404,
                )
            profile = _read_mount_profile(path)
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"模型点方案文件损坏: {exc}"},
            status_code=422,
        )
    except OSError as exc:
        return JSONResponse(
            {"ok": False, "error": f"读取模型点方案失败: {exc}"},
            status_code=500,
        )
    if profile.get("profile_id") != profile_id:
        return JSONResponse(
            {"ok": False, "error": "模型点方案 profile_id 与文件名不一致"},
            status_code=422,
        )
    return {"ok": True, "profile": profile}


@router.post("/api/mount/model-point-profiles")
async def api_save_mount_model_point_profile(body: dict):
    try:
        name, hand_id, points = _validated_mount_profile(body)
    except HandCatalogError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    except KeyError as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc.args[0])}, status_code=404
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    profile_id = _mount_profile_id(hand_id, name)
    now = datetime.now().isoformat(timespec="seconds")
    try:
        path = _mount_profile_dir() / f"{profile_id}.json"
        with mount_profile_lock:
            created = not path.is_file()
            created_at = now
            if not created:
                try:
                    previous = _read_mount_profile(path)
                    previous_created_at = previous.get("created_at")
                    if isinstance(previous_created_at, str) and previous_created_at:
                        created_at = previous_created_at
                except (OSError, json.JSONDecodeError, ValueError):
                    pass
            profile = {
                "schema_version": 1,
                "profile_id": profile_id,
                "name": name,
                "hand_id": hand_id,
                "points": points,
                "point_count": len(points),
                "complete": len(points) == len(MOUNT_PROFILE_POINT_IDS),
                "created_at": created_at,
                "updated_at": now,
            }
            _write_mount_profile_atomic(path, profile)
    except OSError as exc:
        return JSONResponse(
            {"ok": False, "error": f"保存模型点方案失败: {exc}"},
            status_code=500,
        )
    return {"ok": True, "profile": profile, "created": created}


@router.delete("/api/mount/model-point-profiles/{profile_id}")
async def api_delete_mount_model_point_profile(profile_id: str):
    try:
        profile_id = _validated_mount_profile_id(profile_id)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    try:
        path = _mount_profile_dir() / f"{profile_id}.json"
        with mount_profile_lock:
            if not path.is_file():
                return JSONResponse(
                    {"ok": False, "error": "model point profile not found"},
                    status_code=404,
                )
            path.unlink()
    except OSError as exc:
        return JSONResponse(
            {"ok": False, "error": f"删除模型点方案失败: {exc}"},
            status_code=500,
        )
    return {"ok": True, "profile_id": profile_id}


# --------------- 离线点云配对确认 ---------------


@router.post("/api/offline/detect-mount-candidates")
async def api_detect_mount_candidates(body: dict):
    state = _state()
    backend = state._available_episode_backend()
    if backend is None:
        return JSONResponse(
            {"ok": False, "error": "未配置可读取的 episode 任务目录"},
            status_code=409,
        )
    try:
        episode = body["episode"]
        stride = body.get("stride", 2)
        markers = body.get("markers")
        if not isinstance(episode, str) or not episode.strip():
            raise ValueError("episode 必须是非空字符串")
        if isinstance(stride, bool):
            raise ValueError("stride 必须是整数")
        stride = int(stride)
        if markers is not None and not isinstance(markers, list):
            raise ValueError("markers 必须是数组")
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    try:
        return await asyncio.to_thread(
            backend.detect_mount_candidates,
            episode.strip(),
            stride,
            markers,
        )
    except EpisodeValidationError as exc:
        status = 404 if "元数据不存在" in str(exc) else 422
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"安装圆点检测失败: {exc}"},
            status_code=422,
        )


@router.post("/api/offline/confirm-mount-points")
async def api_confirm_mount_points(body: dict):
    state = _state()
    backend = state._available_episode_backend()
    if backend is None:
        return JSONResponse(
            {"ok": False, "error": "未配置可读取的 episode 任务目录"},
            status_code=409,
        )
    try:
        episode = body["episode"]
        cloud_id = body["cloud_id"]
        stride = body.get("stride", 2)
        hand_id = body["hand_id"]
        selections = body["selections"]
        hand_joints = body.get("hand_joints")
        if not isinstance(episode, str) or not episode.strip():
            raise ValueError("episode 必须是非空字符串")
        if not isinstance(hand_id, str) or not hand_id.strip():
            raise ValueError("hand_id 必须是非空字符串")
        if isinstance(stride, bool):
            raise ValueError("stride 必须是整数")
        stride = int(stride)
        if not isinstance(selections, list) or not selections:
            raise ValueError("selections 必须是非空数组")
        for position, selection in enumerate(selections):
            if not isinstance(selection, dict):
                raise ValueError(f"第 {position} 个 selection 必须是 JSON object")
            selection["point_id"] = _validate_mount_point_id(
                selection.get("point_id", selection.get("id")), position
            )
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    try:
        model = get_hand_model(hand_id.strip())
        joint_values = model.coerce_joints(hand_joints)
    except HandCatalogError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
    except KeyError as exc:
        return JSONResponse({"ok": False, "error": str(exc.args[0])}, status_code=404)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    capability, capability_warning, combo_error = await _require_active_combo(
        model.spec.hand_id
    )
    if combo_error is not None:
        return combo_error
    if any(abs(v) > 1e-9 for v in joint_values.values()):
        # 本期标定流程要求手归零：模型点坐标默认按零位 FK 提供
        return JSONResponse(
            {"ok": False, "error": "手安装标定要求灵巧手 6 关节归零后采样"},
            status_code=400,
        )

    hand_info = {
        "hand_id": model.spec.hand_id,
        "base_link": model.base_link,
        "joints": joint_values,
    }
    try:
        result = await asyncio.to_thread(
            backend.confirm_mount_points,
            episode.strip(),
            str(cloud_id).strip(),
            stride,
            hand_info,
            selections,
        )
    except PointCloudStaleError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    except EpisodeValidationError as exc:
        status = 404 if "元数据不存在" in str(exc) else 422
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=status)
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"手安装选点确认失败: {exc}"}, status_code=422
        )
    active_combo = _active_combo_payload(capability)
    for observation in result.get("observations", []):
        if active_combo is not None:
            observation["active_combo"] = active_combo
        provenance = observation.setdefault("provenance", {})
        if active_combo is not None:
            provenance["active_combo"] = active_combo
    result["active_combo"] = active_combo
    if capability_warning:
        result.setdefault("warnings", []).append(capability_warning)
    return result


# --------------- 18089 手保持零位 ---------------


@router.get("/api/mount/hand-hold")
async def api_mount_hand_hold_status():
    return {"ok": True, "hold": _state().hand_hold.status()}


@router.post("/api/mount/hand-hold/start")
async def api_mount_hand_hold_start(body: dict | None = None):
    """开启保持。Body: {hand_id} 或显式 {device_id, side}。"""
    state = _state()
    body = body or {}
    device_id = body.get("device_id")
    side = body.get("side")
    hand_id = body.get("hand_id")
    if hand_id is not None:
        if not isinstance(hand_id, str) or not hand_id.strip():
            return JSONResponse(
                {"ok": False, "error": "hand_id 必须是非空字符串"}, status_code=400
            )
        try:
            spec = get_hand_model(hand_id.strip()).spec
        except HandCatalogError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
        except KeyError as exc:
            return JSONResponse(
                {"ok": False, "error": str(exc.args[0])}, status_code=404
            )
        device_id = device_id or VENDOR_DEVICE_IDS.get(spec.vendor)
        side = side or spec.side
        if device_id is None:
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"手型号 {hand_id} 的厂商 {spec.vendor!r} 没有对应的 "
                    "18089 设备映射，请显式传 device_id",
                },
                status_code=400,
            )
    if not isinstance(device_id, str) or not device_id.strip():
        return JSONResponse(
            {"ok": False, "error": "需要 hand_id 或 device_id 以确定 18089 设备"},
            status_code=400,
        )
    if side not in HOLD_SIDES:
        return JSONResponse(
            {"ok": False, "error": "side 必须是 left 或 right"}, status_code=400
        )
    try:
        hold = await asyncio.to_thread(state.hand_hold.start, device_id.strip(), side)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except HandHoldError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    return {"ok": True, "hold": hold}


@router.post("/api/mount/hand-hold/stop")
async def api_mount_hand_hold_stop():
    hold = await asyncio.to_thread(_state().hand_hold.stop)
    return {"ok": True, "hold": hold}


# --------------- 样本管理 ---------------


def _validated_mount_observation(observation: dict, position: int) -> dict:
    state = _state()
    if not isinstance(observation, dict):
        raise ValueError(f"第 {position} 个 observation 必须是 JSON object")
    if observation.get("schema_version") != 3:
        raise ValueError(f"第 {position} 个 observation 的 schema_version 必须是 3")
    for key in ("hand_id", "pose_id"):
        value = observation.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"第 {position} 个 observation 缺少非空 {key}")
    point_id = _validate_mount_point_id(observation.get("point_id"), position)
    try:
        p_hand = np.asarray(observation["p_hand"], dtype=float).reshape(3)
        p_camera = np.asarray(observation["p_camera"], dtype=float).reshape(3)
        T_wrist = state._parse_wrist_pose(observation)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"第 {position} 个 observation 坐标/位姿不合法: {exc}") from exc
    if not np.all(np.isfinite(p_hand)) or not np.all(np.isfinite(p_camera)):
        raise ValueError(f"第 {position} 个 observation 坐标包含非法值")
    if not (MOUNT_DEPTH_MIN_M <= float(p_camera[2]) <= MOUNT_DEPTH_MAX_M):
        raise ValueError(
            f"第 {position} 个 observation 深度 {p_camera[2]:.2f}m 超出 "
            f"{MOUNT_DEPTH_MIN_M}~{MOUNT_DEPTH_MAX_M}m"
        )
    active_arm = state._active_arm()
    observation_arm = observation.get("arm")
    if observation_arm is not None and observation_arm != active_arm:
        raise ValueError(
            f"第 {position} 个 observation 属于 {observation_arm} 臂，当前为 {active_arm} 臂"
        )
    expected_base = state._active_base_link()
    observation_base = observation.get("base_link")
    if observation_base is not None and observation_base != expected_base:
        raise ValueError(
            f"第 {position} 个 observation 基座为 {observation_base}，当前为 {expected_base}"
        )
    expected_wrist = state._active_wrist_link()
    observation_wrist = observation.get("wrist_link")
    if observation_wrist is not None and observation_wrist != expected_wrist:
        raise ValueError(
            f"第 {position} 个 observation 腕 link 为 {observation_wrist}，"
            f"当前为 {expected_wrist}"
        )
    expected_serial = str(state._active_camera_info().get("serial") or "").strip()
    observation_serial = str(observation.get("camera_serial") or "").strip()
    if expected_serial and observation_serial and observation_serial != expected_serial:
        raise ValueError(
            f"第 {position} 个 observation 相机为 {observation_serial}，"
            f"当前为 {expected_serial}"
        )
    record = dict(observation)
    record.update(
        {
            "schema_version": 3,
            "point_id": point_id,
            "hand_id": str(observation["hand_id"]).strip(),
            "pose_id": str(observation["pose_id"]).strip(),
            "p_hand": p_hand.tolist(),
            "p_camera": p_camera.tolist(),
            "T_base_wrist": T_wrist.tolist(),
        }
    )
    return record


@router.get("/api/mount/samples")
async def api_mount_samples():
    items = _load_mount_samples()
    return {
        "samples": items,
        "count": len(items),
        "min_points": MIN_MOUNT_POINTS,
    }


@router.post("/api/mount/samples/batch")
async def api_mount_samples_batch(body: dict):
    state = _state()
    try:
        observations = body["observations"]
        if not isinstance(observations, list) or not observations:
            raise ValueError("observations 必须是非空数组")
        records = [
            _validated_mount_observation(observation, position)
            for position, observation in enumerate(observations)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc) or "需要 observations"}, status_code=400
        )

    hand_ids = {record["hand_id"] for record in records}
    if len(hand_ids) != 1:
        return JSONResponse(
            {"ok": False, "error": f"同一批次只能属于一个手型号，收到: {sorted(hand_ids)}"},
            status_code=400,
        )
    hand_id = next(iter(hand_ids))
    _, capability_warning, combo_error = await _require_active_combo(hand_id)
    if combo_error is not None:
        return combo_error

    with state.samples_lock:
        existing = _load_mount_samples()
        existing_keys = {
            (item.get("pose_id"), item.get("point_id")): item.get("index")
            for item in existing
        }
        existing_hands = {item.get("hand_id") for item in existing}
        if existing_hands and hand_ids - existing_hands:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "当前会话已有其他手型号的安装样本，"
                    f"已有: {sorted(existing_hands)}，新增: {sorted(hand_ids)}；"
                    "请换一个标定会话目录",
                },
                status_code=409,
            )
        for record in records:
            key = (record["pose_id"], record["point_id"])
            if key in existing_keys:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": f"pose {key[0]} 的模型点 {key[1]} 已保存为样本 "
                        f"{existing_keys[key]}",
                        "existing_index": existing_keys[key],
                    },
                    status_code=409,
                )
        first_index = _next_mount_index()
        indices = list(range(first_index, first_index + len(records)))
        now = datetime.now().isoformat(timespec="seconds")
        for index, record in zip(indices, records):
            saved = dict(record)
            saved["index"] = index
            saved["datetime"] = now
            (_mount_dir() / f"{index:04d}.json").write_text(
                json.dumps(saved, indent=2, ensure_ascii=False)
            )
    response = {
        "ok": True,
        "indices": indices,
        "saved_count": len(indices),
        "count": len(_load_mount_samples()),
    }
    response["warnings"] = [capability_warning] if capability_warning else []
    return response


@router.delete("/api/mount/samples/{index}")
async def api_mount_sample_delete(index: int):
    f = _mount_dir() / f"{index:04d}.json"
    if not f.exists():
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    f.unlink()
    return {"ok": True, "count": len(_load_mount_samples())}


@router.post("/api/mount/clear")
async def api_mount_clear():
    for f in _mount_dir().glob("*.json"):
        f.unlink()
    return {"ok": True, "count": 0}


# --------------- 解算 ---------------


@router.get("/api/mount/result")
async def api_mount_result():
    state = _state()
    path = state.save_path / "mount_result.json"
    if not path.is_file():
        return {"result": None, "stale": False}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"安装标定结果无法读取: {exc}"},
            status_code=500,
        )
    if not isinstance(result, dict):
        return JSONResponse(
            {"ok": False, "error": "安装标定结果格式不合法"},
            status_code=500,
        )
    result["saved_to"] = str(path)
    merged = state.save_path / "handeye3d_result_mount.json"
    if merged.is_file():
        result["merged_calib"] = str(merged)
    current_indices = {
        sample.get("index") for sample in _load_mount_samples()
    }
    result_indices = set(result.get("sample_indices") or [])
    stale = current_indices != result_indices
    return {"result": result, "stale": stale}


def _diagnostic_stats(values_mm: list[float]) -> dict[str, float | int]:
    values = np.asarray(values_mm, dtype=float)
    if not len(values):
        return {"count": 0, "rms": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "count": int(len(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _mount_point_display(point_id: str) -> tuple[str, str]:
    number = int(point_id.rsplit("-", 1)[-1])
    if point_id.startswith("palm-red-"):
        return f"红{number}", "red"
    return f"绿{number}", "green"


def _best_same_color_swap(
    samples: list[dict[str, Any]],
    observed_wrist: list[np.ndarray],
) -> dict[str, Any]:
    point_by_id: dict[str, np.ndarray] = {}
    for sample in samples:
        point_by_id.setdefault(
            sample["point_id"], np.asarray(sample["p_hand"], dtype=float)
        )

    def solve_mapping(mapping: dict[str, str]) -> float:
        source = np.stack(
            [point_by_id[mapping[sample["point_id"]]] for sample in samples]
        )
        target = np.stack(observed_wrist)
        return float(solve_rigid_transform(source, target)["residual_mm"]["rms"])

    identity = {point_id: point_id for point_id in point_by_id}
    current_rms = solve_mapping(identity)
    best_rms = current_rms
    best_pair: list[str] | None = None
    tested = 0
    for prefix in ("palm-red-", "back-green-"):
        point_ids = sorted(
            point_id for point_id in point_by_id if point_id.startswith(prefix)
        )
        for left_index, left in enumerate(point_ids):
            for right in point_ids[left_index + 1 :]:
                tested += 1
                mapping = dict(identity)
                mapping[left], mapping[right] = right, left
                rms = solve_mapping(mapping)
                if rms < best_rms:
                    best_rms = rms
                    best_pair = [left, right]

    improvement = current_rms - best_rms
    significant = improvement >= max(0.5, current_rms * 0.08)
    if best_pair is None or improvement < 1e-6:
        verdict = "consistent"
        label = "未发现编号互换收益"
        message = f"已检查 {tested} 种同色两点互换，当前顺序误差最低。"
    elif significant:
        verdict = "suspect"
        labels = [_mount_point_display(point_id)[0] for point_id in best_pair]
        label = f"建议复核 {labels[0]} / {labels[1]}"
        message = (
            f"互换后 RMS 可由 {current_rms:.2f} 降至 {best_rms:.2f} mm，"
            "存在点序错误的可能。"
        )
    else:
        verdict = "consistent"
        label = "当前编号基本一致"
        message = (
            f"最佳互换仅改善 {improvement:.2f} mm，"
            "不足以说明点序有误。"
        )
    return {
        "verdict": verdict,
        "label": label,
        "message": message,
        "tested_swap_count": tested,
        "current_rms_mm": current_rms,
        "best_swap_rms_mm": best_rms,
        "improvement_mm": improvement,
        "best_swap": best_pair,
    }


@router.get("/api/mount/diagnostics")
async def api_mount_diagnostics():
    """返回 7015 使用的手安装对应关系、误差向量与点序诊断。"""
    state = _state()
    result_path = state.save_path / "mount_result.json"
    if not result_path.is_file():
        return {"available": False, "reason": "尚未生成安装标定结果"}
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        samples = _load_mount_samples()
        calib_path = Path(result["calib_used"]).expanduser().resolve()
        _, R_cam, t_cam, calib_identity = _load_mount_calibration(
            calib_path, require_camera_identity=False
        )
        T_mount = np.asarray(result["T_wrist2hand"], dtype=float).reshape(4, 4)
    except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"安装标定诊断数据无法读取: {exc}"},
            status_code=500,
        )
    if not samples:
        return {"available": False, "reason": "安装样本已被清空"}

    R_mount = T_mount[:3, :3]
    t_mount = T_mount[:3, 3]
    observations: list[dict[str, Any]] = []
    observed_wrist: list[np.ndarray] = []
    by_pose: dict[str, list[dict[str, Any]]] = {}
    by_point: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        p_camera = np.asarray(sample["p_camera"], dtype=float)
        p_base = R_cam @ p_camera + t_cam
        p_wrist = np.linalg.solve(
            np.asarray(sample["T_base_wrist"], dtype=float),
            np.array([*p_base, 1.0]),
        )[:3]
        p_hand = np.asarray(sample["p_hand"], dtype=float)
        predicted_wrist = R_mount @ p_hand + t_mount
        observed_hand = R_mount.T @ (p_wrist - t_mount)
        residual_vector_mm = (observed_hand - p_hand) * 1000.0
        residual_mm = float(np.linalg.norm(residual_vector_mm))
        short_label, color = _mount_point_display(sample["point_id"])
        observation = {
            "sample_index": sample.get("index"),
            "pose_id": sample["pose_id"],
            "point_id": sample["point_id"],
            "label": sample.get("label") or sample["point_id"],
            "short_label": short_label,
            "color": color,
            "model_point_hand_m": p_hand.tolist(),
            "observed_point_hand_m": observed_hand.tolist(),
            "predicted_point_wrist_m": predicted_wrist.tolist(),
            "observed_point_wrist_m": p_wrist.tolist(),
            "residual_vector_hand_mm": residual_vector_mm.tolist(),
            "residual_mm": residual_mm,
            "pixel": sample.get("pixel"),
        }
        observations.append(observation)
        observed_wrist.append(p_wrist)
        by_pose.setdefault(sample["pose_id"], []).append(observation)
        by_point.setdefault(sample["point_id"], []).append(observation)

    poses = []
    for pose_id, pose_observations in sorted(by_pose.items()):
        stats = _diagnostic_stats(
            [observation["residual_mm"] for observation in pose_observations]
        )
        poses.append(
            {
                "pose_id": pose_id,
                **stats,
                "observations": pose_observations,
            }
        )
    point_stats = []
    for point_id, point_observations in sorted(by_point.items()):
        stats = _diagnostic_stats(
            [observation["residual_mm"] for observation in point_observations]
        )
        short_label, color = _mount_point_display(point_id)
        point_stats.append(
            {
                "point_id": point_id,
                "short_label": short_label,
                "color": color,
                **stats,
            }
        )
    color_stats = {
        color: _diagnostic_stats(
            [
                observation["residual_mm"]
                for observation in observations
                if observation["color"] == color
            ]
        )
        for color in ("red", "green")
    }

    current_indices = {sample.get("index") for sample in samples}
    result_indices = set(result.get("sample_indices") or [])
    return {
        "ok": True,
        "available": True,
        "stale": current_indices != result_indices,
        "hand_id": result.get("hand_id"),
        "hand_label": result.get("hand_label"),
        "solved_at": result.get("solved_at"),
        "calib_camera": calib_identity,
        "summary": {
            "sample_count": len(observations),
            "pose_count": len(poses),
            "point_count": len(point_stats),
            **_diagnostic_stats(
                [observation["residual_mm"] for observation in observations]
            ),
            "by_color": color_stats,
            "leave_one_pose_out": result.get("leave_one_pose_out"),
        },
        "order_check": await asyncio.to_thread(
            _best_same_color_swap, samples, observed_wrist
        ),
        "poses": poses,
        "point_stats": point_stats,
        "T_wrist2hand": result["T_wrist2hand"],
        "t_wrist2hand_m": result.get("t_wrist2hand_m"),
        "rpy_deg": result.get("rpy_deg"),
    }


@router.post("/api/mount/solve")
async def api_mount_solve(body: dict | None = None):
    """固定 T_cam2base，解 T_wrist2hand。Body 可选: {"calib_path": "..."}。"""
    state = _state()
    body = body or {}
    if body.get("calib_path"):
        calib_path = Path(body["calib_path"]).expanduser().resolve()
    elif state.mount_calib_path is not None:
        calib_path = state.mount_calib_path
    else:
        calib_path = state._find_latest_calib()
    if calib_path is None or not calib_path.is_file():
        return JSONResponse(
            {
                "ok": False,
                "error": "找不到手安装使用的相机外参，"
                "请通过 --mount-calib 或 calib_path 指定",
            },
            status_code=400,
        )
    try:
        calib, R_cam, t_cam, calib_identity = _load_mount_calibration(calib_path)
    except ValueError as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc)}, status_code=400
        )

    samples = _load_mount_samples()
    if not samples:
        return JSONResponse(
            {"ok": False, "error": "没有手安装标定样本"}, status_code=400
        )
    try:
        samples = [
            _validated_mount_observation(sample, position)
            for position, sample in enumerate(samples)
        ]
    except ValueError as exc:
        return JSONResponse(
            {"ok": False, "error": f"已有安装样本与当前会话不一致: {exc}"},
            status_code=409,
        )
    hand_ids = {sample.get("hand_id") for sample in samples}
    if len(hand_ids) != 1:
        return JSONResponse(
            {"ok": False, "error": f"样本包含多个手型号: {sorted(hand_ids)}"},
            status_code=400,
        )
    hand_id = next(iter(hand_ids))
    capability, capability_warning, combo_error = await _require_active_combo(hand_id)
    if combo_error is not None:
        return combo_error
    try:
        model = get_hand_model(hand_id)
    except (HandCatalogError, KeyError) as exc:
        return JSONResponse(
            {"ok": False, "error": f"样本引用的手型号不可用: {exc}"}, status_code=400
        )

    p_hand = np.array([sample["p_hand"] for sample in samples], dtype=float)
    p_camera = np.array([sample["p_camera"] for sample in samples], dtype=float)
    T_wrist = np.array([sample["T_base_wrist"] for sample in samples], dtype=float)
    point_ids = [sample["point_id"] for sample in samples]
    pose_ids = [sample["pose_id"] for sample in samples]

    # 相机点 → 基座系 → 腕系
    p_base = (R_cam @ p_camera.T).T + t_cam
    p_wrist = np.stack(
        [
            np.linalg.solve(
                T_wrist[i],
                np.array([*p_base[i], 1.0]),
            )[:3]
            for i in range(len(samples))
        ]
    )

    try:
        result = await asyncio.to_thread(
            solve_hand_mount, p_hand, p_wrist, point_ids, pose_ids
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    result["leave_one_pose_out"] = await asyncio.to_thread(
        leave_one_pose_out_mount, p_hand, p_wrist, point_ids, pose_ids
    )

    T_mount = np.asarray(result["T_wrist2hand"], dtype=float)
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam
    T_cam2base[:3, 3] = t_cam
    T_base2cam = np.linalg.inv(T_cam2base)

    # 每个 pose 的手模型 → 相机系变换，前端用来把手模型叠加回点云核对
    overlay = {}
    for pose in result["pose_ids"]:
        index = pose_ids.index(pose)
        T_camera_hand = T_base2cam @ T_wrist[index] @ T_mount
        overlay[pose] = T_camera_hand.tolist()
    result["per_pose_overlay_T_camera_hand"] = overlay

    # 指尖等预置点在腕系下的坐标（派生 TCP，供 IK_replay 等消费端使用）
    tcp_points = []
    for feature in model.feature_points():
        p = T_mount @ np.array([*feature["p_hand"], 1.0])
        tcp_points.append(
            {
                "id": feature["id"],
                "label": feature["label"],
                "link": feature["link"],
                "p_wrist_m": [float(v) for v in p[:3]],
            }
        )
    result["tcp_points_wrist_m"] = tcp_points

    result["ok"] = True
    result["hand_id"] = hand_id
    result["hand_label"] = model.spec.label
    result["hand_base_link"] = model.base_link
    result["sample_indices"] = [sample.get("index") for sample in samples]
    result["solved_at"] = datetime.now().isoformat(timespec="seconds")
    result["calib_used"] = str(calib_path)
    result["calib_camera"] = calib_identity
    result["active_combo"] = _active_combo_payload(capability)
    result["warnings"] = [capability_warning] if capability_warning else []
    result["base_link"] = samples[0].get("base_link")
    result["wrist_link"] = samples[0].get("wrist_link")
    result["arm"] = samples[0].get("arm")

    out = state.save_path / "mount_result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    result["saved_to"] = str(out)

    # 合并版标定文件：原相机外参 + 手安装变换，可整体归档给消费端
    merged = dict(calib)
    merged.pop("saved_to", None)
    merged["T_wrist2hand"] = result["T_wrist2hand"]
    merged["hand_id"] = hand_id
    merged["hand_label"] = model.spec.label
    merged["hand_base_link"] = model.base_link
    merged["tcp_points_wrist_m"] = tcp_points
    merged["per_pose_overlay_T_camera_hand"] = result[
        "per_pose_overlay_T_camera_hand"
    ]
    merged["mount_residual_mm"] = result["residual_mm"]
    merged["mount_solved_at"] = result["solved_at"]
    merged["mount_calib_used"] = str(calib_path)
    merged["mount_calib_camera"] = calib_identity
    merged["active_combo"] = result["active_combo"]
    merged_path = state.save_path / "handeye3d_result_mount.json"
    merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    result["merged_calib"] = str(merged_path)
    return result
