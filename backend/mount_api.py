"""手安装标定（T_wrist2hand）API：手型号目录、模型点配对样本与解算。

两步解的第二步：T_cam2base 用已有手眼标定结果固定，这里只解
腕 → 手基座 的 6 自由度安装变换。样本 schema_version=3，独立存放在
<save_path>/mount_samples/，不与 v1/v2 标记样本混用。
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .hands import HandCatalogError, get_hand_model, hand_catalog
from .offline import EpisodeValidationError, PointCloudStaleError
from .solver import (
    MIN_MOUNT_POINTS,
    leave_one_pose_out_mount,
    solve_hand_mount,
)

router = APIRouter()

MOUNT_DEPTH_MIN_M = 0.30
MOUNT_DEPTH_MAX_M = 1.5
MOUNT_POINT_ID_RE = re.compile(r"^(?:palm-red|back-green)-(?:0[1-8])$")


def _state():
    """app 模块的注入状态（save_path / episode backend 等），运行时读取。"""
    from . import app as app_state

    return app_state


def _mount_dir() -> Path:
    directory = _state().save_path / "mount_samples"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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


# --------------- 离线点云配对确认 ---------------


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
