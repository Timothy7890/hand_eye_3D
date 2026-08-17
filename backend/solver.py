"""3D-3D 手眼求解。

两种模式:
1. solve_rigid_transform: 已知点对 (P_camera_i, P_base_i)，Kabsch 闭式解 T_base^camera。
2. solve_with_tool_offset: 标记点在基座系的坐标未知，只知道每次采样时的
   手腕位姿 T_base^wrist_i；把指尖偏移 p_tool（腕系下，常量）和 T_base^camera
   联合解出。约束: R @ P_cam_i + t = R_w_i @ p_tool + t_w_i。
   用交替最小二乘：固定 p_tool 是 Kabsch，固定 (R,t) 是线性最小二乘，
   两步都各自全局最优，从 p_tool=0 出发单调下降收敛。

单位一律米。
"""

from __future__ import annotations

import math

import numpy as np

MIN_SAMPLES = 3
MIN_SAMPLES_TOOL = 5  # 联合解 9 个未知量，5 对(15 方程)起步，建议 >= 10
MIN_WRIST_ROT_DEG = 15.0  # 手腕姿态变化不足时 p_tool 与 t 不可分


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF 固定轴 RPY → 旋转矩阵（R = Rz(yaw) Ry(pitch) Rx(roll)）。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def rot_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """URDF 固定轴 RPY 约定（R = Rz(yaw) Ry(pitch) Rx(roll)），与 hand_eye 项目一致。"""
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(pitch)) < 1e-8:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def _collinearity(points: np.ndarray) -> float:
    """点集第二奇异值与第一奇异值之比，接近 0 说明近似共线（解不稳定）。"""
    centered = points - points.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    if s[0] < 1e-12:
        return 0.0
    return float(s[1] / s[0])


def solve_rigid_transform(
    p_camera: np.ndarray, p_base: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    """Kabsch 算法求 R, t 使 ||R @ p_camera + t - p_base|| 最小。

    p_camera, p_base: (N, 3)，米。返回包含 T、残差统计的 dict。
    """
    p_camera = np.asarray(p_camera, dtype=float).reshape(-1, 3)
    p_base = np.asarray(p_base, dtype=float).reshape(-1, 3)
    n = len(p_camera)
    if len(p_base) != n:
        raise ValueError(f"点数不一致: camera {n} vs base {len(p_base)}")
    if n < MIN_SAMPLES:
        raise ValueError(f"至少需要 {MIN_SAMPLES} 对点，当前 {n} 对")
    if not np.all(np.isfinite(p_camera)) or not np.all(np.isfinite(p_base)):
        raise ValueError("点坐标包含 NaN 或无穷值")
    if weights is None:
        normalized_weights = np.full(n, 1.0 / n)
    else:
        raw_weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(raw_weights) != n:
            raise ValueError(f"权重数量不一致: {len(raw_weights)} vs {n}")
        if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights <= 0):
            raise ValueError("权重必须是有限正数")
        normalized_weights = raw_weights / raw_weights.sum()

    collinearity = _collinearity(p_camera)
    if collinearity < 1e-6:
        raise ValueError("采样点几乎共线，无法唯一确定旋转，请把点在空间中撒开")

    centroid_cam = np.sum(p_camera * normalized_weights[:, None], axis=0)
    centroid_base = np.sum(p_base * normalized_weights[:, None], axis=0)
    q_cam = p_camera - centroid_cam
    q_base = p_base - centroid_base

    H = q_cam.T @ (q_base * normalized_weights[:, None])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])  # 修正镜像解
    R = Vt.T @ D @ U.T
    t = centroid_base - R @ centroid_cam

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # 逐点残差（毫米）
    predicted = (R @ p_camera.T).T + t
    errors_mm = np.linalg.norm(predicted - p_base, axis=1) * 1000.0
    rpy = rot_to_rpy(R)

    return {
        "num_samples": n,
        "T_cam2base": T.tolist(),
        "R_cam2base": R.tolist(),
        "t_cam2base_m": t.tolist(),
        "rpy_rad": list(rpy),
        "rpy_deg": [math.degrees(a) for a in rpy],
        "residual_mm": {
            "per_sample": errors_mm.tolist(),
            "rms": float(np.sqrt((errors_mm ** 2).mean())),
            "mean": float(errors_mm.mean()),
            "max": float(errors_mm.max()),
        },
        "collinearity": collinearity,
    }


def solve_with_tool_offset(p_camera: np.ndarray, T_wrist: np.ndarray,
                           max_iters: int = 200, tol: float = 1e-10) -> dict:
    """联合估计 T_base^camera 和指尖偏移 p_tool（腕系）。

    p_camera: (N,3) 相机系坐标（米）
    T_wrist:  (N,4,4) 每次采样时的手腕位姿 T_base^wrist
    """
    p_camera = np.asarray(p_camera, dtype=float).reshape(-1, 3)
    T_wrist = np.asarray(T_wrist, dtype=float).reshape(-1, 4, 4)
    n = len(p_camera)
    if len(T_wrist) != n:
        raise ValueError(f"点数不一致: camera {n} vs wrist {len(T_wrist)}")
    if n < MIN_SAMPLES_TOOL:
        raise ValueError(f"联合解至少需要 {MIN_SAMPLES_TOOL} 对样本，当前 {n} 对")

    R_w = T_wrist[:, :3, :3]
    t_w = T_wrist[:, :3, 3]

    # 可辨识性检查：手腕姿态必须有足够变化，否则 p_tool 与 t 耦合不可分
    max_rot = max(geodesic_deg(R_w[0], R_w[i]) for i in range(1, n))
    if max_rot < MIN_WRIST_ROT_DEG:
        raise ValueError(
            f"手腕姿态变化只有 {max_rot:.1f}°（需要 ≥ {MIN_WRIST_ROT_DEG}°），"
            "请让手腕朝向也充分变化，否则指尖偏移无法解出")

    p_tool = np.zeros(3)
    prev_cost = np.inf
    iterations = 0
    for iterations in range(1, max_iters + 1):
        # 步骤 1: 固定 p_tool，P_base_i = R_w_i @ p_tool + t_w_i，Kabsch 解 (R, t)
        p_base = (R_w @ p_tool) + t_w
        base = solve_rigid_transform(p_camera, p_base)
        R = np.array(base["R_cam2base"])
        t = np.array(base["t_cam2base_m"])

        # 步骤 2: 固定 (R, t)，线性最小二乘解 p_tool:
        #   R_w_i @ p_tool = (R @ p_cam_i + t) - t_w_i
        target = (R @ p_camera.T).T + t - t_w          # (N,3)
        A = R_w.reshape(-1, 3)                          # (3N,3)
        b = target.reshape(-1)
        p_tool_new, *_ = np.linalg.lstsq(A, b, rcond=None)

        residual = (R_w @ p_tool_new) + t_w - target
        cost = float((residual ** 2).sum())
        if abs(prev_cost - cost) < tol:
            p_tool = p_tool_new
            break
        p_tool = p_tool_new
        prev_cost = cost

    # 最终一轮解算 + 残差
    p_base = (R_w @ p_tool) + t_w
    result = solve_rigid_transform(p_camera, p_base)
    result["mode"] = "tool_offset_joint"
    result["p_tool_wrist_m"] = p_tool.tolist()
    result["iterations"] = iterations
    result["wrist_rotation_spread_deg"] = max_rot
    return result


MIN_MULTI_OBSERVATIONS = 6
MIN_MULTI_POSES = 3
MIN_POSES_PER_MARKER = 2


def _residual_stats(values_mm: np.ndarray) -> dict:
    values = np.asarray(values_mm, dtype=float).reshape(-1)
    return {
        "count": int(values.size),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _rotation_spread_deg(rotations: np.ndarray) -> float:
    return max(
        (
            geodesic_deg(rotations[i], rotations[j])
            for i in range(len(rotations))
            for j in range(i + 1, len(rotations))
        ),
        default=0.0,
    )


def _validate_wrist_transforms(T_wrist: np.ndarray) -> None:
    if not np.all(np.isfinite(T_wrist)):
        raise ValueError("T_base_wrist 包含 NaN 或无穷值")
    for index, transform in enumerate(T_wrist):
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=2e-3
        ):
            raise ValueError(f"第 {index} 个 T_base_wrist 的旋转矩阵不合法")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(f"第 {index} 个 T_base_wrist 的末行不合法")


def _multi_inputs(
    p_camera: np.ndarray,
    T_wrist: np.ndarray,
    marker_ids,
    pose_ids,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    p_camera = np.asarray(p_camera, dtype=float).reshape(-1, 3)
    T_wrist = np.asarray(T_wrist, dtype=float).reshape(-1, 4, 4)
    marker_ids_array = np.asarray(marker_ids, dtype=object).reshape(-1)
    pose_ids_array = np.asarray(pose_ids, dtype=object).reshape(-1)
    n = len(p_camera)
    if len(T_wrist) != n or len(marker_ids_array) != n or len(pose_ids_array) != n:
        raise ValueError(
            "多 marker 输入数量不一致: "
            f"camera={n}, wrist={len(T_wrist)}, marker={len(marker_ids_array)}, "
            f"pose={len(pose_ids_array)}"
        )
    if not np.all(np.isfinite(p_camera)):
        raise ValueError("p_camera 包含 NaN 或无穷值")
    _validate_wrist_transforms(T_wrist)

    markers = []
    poses = []
    for index, value in enumerate(marker_ids_array):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"第 {index} 个 marker_id 必须是非空字符串")
        markers.append(value.strip())
    for index, value in enumerate(pose_ids_array):
        if not isinstance(value, (str, int)) or str(value).strip() == "":
            raise ValueError(f"第 {index} 个 pose_id 必须是非空字符串或整数")
        poses.append(str(value))
    marker_ids_array = np.asarray(markers, dtype=object)
    pose_ids_array = np.asarray(poses, dtype=object)
    marker_names = sorted(set(markers))
    pose_names = sorted(set(poses))
    required = max(MIN_MULTI_OBSERVATIONS, len(marker_names) + 2)
    if n < required:
        raise ValueError(
            f"{len(marker_names)} 种 marker 的联合解至少需要 {required} 个 observation，"
            f"当前 {n} 个"
        )
    if len(pose_names) < MIN_MULTI_POSES:
        raise ValueError(
            f"多 marker 联合解至少需要 {MIN_MULTI_POSES} 个不同 pose，当前 {len(pose_names)} 个"
        )
    duplicates = [
        f"{pose}/{marker}"
        for pose, marker in sorted(zip(poses, markers))
        if sum(
            other_pose == pose and other_marker == marker
            for other_pose, other_marker in zip(poses, markers)
        )
        > 1
    ]
    if duplicates:
        raise ValueError(f"同一 pose/marker 只能有一个 observation: {duplicates[0]}")
    for marker in marker_names:
        marker_pose_count = len(set(pose_ids_array[marker_ids_array == marker]))
        if marker_pose_count < MIN_POSES_PER_MARKER:
            raise ValueError(
                f"marker {marker} 只覆盖 {marker_pose_count} 个 pose，至少需要 "
                f"{MIN_POSES_PER_MARKER} 个"
            )

    collinearity = _collinearity(p_camera)
    if collinearity < 1e-6:
        raise ValueError("相机点退化（几乎共线或重合），请增加空间分布不同的 observation")
    spread = _rotation_spread_deg(T_wrist[:, :3, :3])
    if spread < MIN_WRIST_ROT_DEG:
        raise ValueError(
            f"手腕姿态变化只有 {spread:.1f}°（需要 ≥ {MIN_WRIST_ROT_DEG}°），"
            "请增加朝向变化，否则 marker 腕系偏移与相机平移不可分"
        )
    return (
        p_camera,
        T_wrist,
        marker_ids_array,
        pose_ids_array,
        marker_names,
        pose_names,
    )


def solve_multi_marker(
    p_camera: np.ndarray,
    T_wrist: np.ndarray,
    marker_ids,
    pose_ids,
    max_iters: int = 500,
    tol: float = 1e-16,
) -> dict:
    """共享相机外参、每种 marker 独立腕系偏移的鲁棒交替最小二乘。"""
    (
        p_camera,
        T_wrist,
        marker_ids,
        pose_ids,
        marker_names,
        pose_names,
    ) = _multi_inputs(p_camera, T_wrist, marker_ids, pose_ids)
    n = len(p_camera)
    R_w = T_wrist[:, :3, :3]
    t_w = T_wrist[:, :3, 3]
    marker_indices = {
        marker: np.flatnonzero(marker_ids == marker) for marker in marker_names
    }
    offsets = {marker: np.zeros(3, dtype=float) for marker in marker_names}
    weights = np.ones(n, dtype=float)
    previous_cost = np.inf
    converged = False
    iterations = 0

    for iterations in range(1, max_iters + 1):
        p_base = np.stack(
            [R_w[i] @ offsets[str(marker_ids[i])] + t_w[i] for i in range(n)]
        )
        rigid = solve_rigid_transform(p_camera, p_base, weights=weights)
        R = np.asarray(rigid["R_cam2base"], dtype=float)
        t = np.asarray(rigid["t_cam2base_m"], dtype=float)
        transformed = (R @ p_camera.T).T + t

        new_offsets: dict[str, np.ndarray] = {}
        for marker, indices in marker_indices.items():
            target = transformed[indices] - t_w[indices]
            A = R_w[indices].reshape(-1, 3)
            b = target.reshape(-1)
            repeated_weights = np.repeat(np.sqrt(weights[indices]), 3)
            estimate, *_ = np.linalg.lstsq(
                A * repeated_weights[:, None], b * repeated_weights, rcond=None
            )
            new_offsets[marker] = estimate

        target_base = np.stack(
            [
                R_w[i] @ new_offsets[str(marker_ids[i])] + t_w[i]
                for i in range(n)
            ]
        )
        residual_m = np.linalg.norm(transformed - target_base, axis=1)
        median = float(np.median(residual_m))
        mad = float(np.median(np.abs(residual_m - median)))
        robust_scale = max(1.4826 * mad, 1e-6)
        huber_cutoff = max(median + 2.5 * robust_scale, 2e-6)
        new_weights = np.minimum(1.0, huber_cutoff / np.maximum(residual_m, 1e-12))
        cost = float(np.average(residual_m ** 2, weights=new_weights))
        offset_delta = max(
            np.linalg.norm(new_offsets[name] - offsets[name]) for name in marker_names
        )
        if abs(previous_cost - cost) < tol and offset_delta < math.sqrt(tol):
            offsets = new_offsets
            weights = new_weights
            converged = True
            break
        offsets = new_offsets
        weights = new_weights
        previous_cost = cost

    p_base = np.stack(
        [R_w[i] @ offsets[str(marker_ids[i])] + t_w[i] for i in range(n)]
    )
    rigid = solve_rigid_transform(p_camera, p_base, weights=weights)
    R = np.asarray(rigid["R_cam2base"], dtype=float)
    t = np.asarray(rigid["t_cam2base_m"], dtype=float)
    predicted = (R @ p_camera.T).T + t
    residual_mm = np.linalg.norm(predicted - p_base, axis=1) * 1000.0

    by_marker = {}
    for marker, indices in marker_indices.items():
        stats = _residual_stats(residual_mm[indices])
        stats["per_observation"] = residual_mm[indices].tolist()
        stats["pose_count"] = len(set(pose_ids[indices].tolist()))
        by_marker[marker] = stats
    observation_residuals = [
        {
            "observation_index": index,
            "pose_id": str(pose_ids[index]),
            "marker_id": str(marker_ids[index]),
            "residual_mm": float(residual_mm[index]),
            "robust_weight": float(weights[index]),
        }
        for index in range(n)
    ]
    primary = "red" if "red" in offsets else marker_names[0]
    result = dict(rigid)
    result.update(
        {
            "mode": "multi_marker_tool_offset_joint",
            "num_samples": n,
            "observation_count": n,
            "pose_count": len(pose_names),
            "marker_count": len(marker_names),
            "marker_ids": marker_names,
            "pose_ids": pose_names,
            "p_tool_wrist_m_by_marker": {
                marker: offsets[marker].tolist() for marker in marker_names
            },
            "p_tool_wrist_m": offsets[primary].tolist(),
            "p_tool_reference": primary,
            "p_tool_reference_marker": primary,
            "wrist_rotation_spread_deg": _rotation_spread_deg(R_w),
            "per_observation_residuals": observation_residuals,
            "per_marker_residual_stats_mm": by_marker,
            "residual_by_marker_mm": by_marker,
            "residual_mm": {
                "per_sample": residual_mm.tolist(),
                **_residual_stats(residual_mm),
            },
            "iterations": iterations,
            "convergence": {
                "converged": converged,
                "iterations": iterations,
                "max_iterations": max_iters,
                "final_cost_m2": float(np.average((residual_mm / 1000.0) ** 2, weights=weights)),
                "robust_method": "Huber IRLS without observation dropping",
            },
        }
    )
    return result


def leave_one_pose_out_multi(
    p_camera: np.ndarray,
    T_wrist: np.ndarray,
    marker_ids,
    pose_ids,
) -> dict:
    """按整个位姿留一；覆盖不够时返回诊断，不偷偷减少 marker 集。"""
    try:
        (
            p_camera,
            T_wrist,
            marker_ids,
            pose_ids,
            marker_names,
            pose_names,
        ) = _multi_inputs(p_camera, T_wrist, marker_ids, pose_ids)
    except ValueError as exc:
        return {"feasible": False, "coverage_diagnostics": [str(exc)]}

    diagnostics = []
    if len(pose_names) < MIN_MULTI_POSES + 1:
        diagnostics.append(
            f"留一 pose 验证至少需要 {MIN_MULTI_POSES + 1} 个 pose，当前 {len(pose_names)} 个"
        )
    for marker in marker_names:
        count = len(set(pose_ids[marker_ids == marker]))
        if count < MIN_POSES_PER_MARKER + 1:
            diagnostics.append(
                f"marker {marker} 需覆盖至少 {MIN_POSES_PER_MARKER + 1} 个 pose "
                f"才能按 pose 留一，当前 {count} 个"
            )
    if diagnostics:
        return {"feasible": False, "coverage_diagnostics": diagnostics}

    folds = []
    all_errors = []
    for pose in pose_names:
        train = pose_ids != pose
        test = ~train
        try:
            fitted = solve_multi_marker(
                p_camera[train],
                T_wrist[train],
                marker_ids[train],
                pose_ids[train],
            )
        except ValueError as exc:
            diagnostics.append(f"留出 pose {pose} 后不可解: {exc}")
            continue
        R = np.asarray(fitted["R_cam2base"], dtype=float)
        t = np.asarray(fitted["t_cam2base_m"], dtype=float)
        offsets = {
            marker: np.asarray(value, dtype=float)
            for marker, value in fitted["p_tool_wrist_m_by_marker"].items()
        }
        test_indices = np.flatnonzero(test)
        errors = []
        for index in test_indices:
            marker = str(marker_ids[index])
            if marker not in offsets:
                diagnostics.append(f"留出 pose {pose} 后缺少 marker {marker} 的偏移")
                continue
            camera_base = R @ p_camera[index] + t
            wrist_base = (
                T_wrist[index, :3, :3] @ offsets[marker]
                + T_wrist[index, :3, 3]
            )
            errors.append(float(np.linalg.norm(camera_base - wrist_base) * 1000.0))
        if errors:
            all_errors.extend(errors)
            folds.append(
                {
                    "pose_id": pose,
                    "observation_count": len(errors),
                    "residual_mm": _residual_stats(np.asarray(errors)),
                    "per_observation": errors,
                }
            )
    if diagnostics or len(folds) != len(pose_names):
        return {
            "feasible": False,
            "completed_fold_count": len(folds),
            "required_fold_count": len(pose_names),
            "coverage_diagnostics": diagnostics,
        }
    return {
        "feasible": True,
        "folds": folds,
        "stats_mm": _residual_stats(np.asarray(all_errors)),
        "coverage_diagnostics": [],
    }


MIN_SAMPLES_TOOL_ONLY = 3


def solve_tool_fixed_cam(p_camera: np.ndarray, T_wrist: np.ndarray,
                         R_cam2base: np.ndarray, t_cam2base: np.ndarray) -> dict:
    """固定相机外参，只解指尖偏移 p_tool（腕系）。

    已有可信的 T_base^camera 时用这个：每个样本给 3 个方程、只有 3 个未知量，
    线性最小二乘一步解出，样本少也稳。约束:
        R_w_i @ p_tool + t_w_i = R @ p_cam_i + t
    与联合解不同，这里 p_tool 单独可辨识，姿态跨度不是硬性要求——但跨度越大，
    解出的 p_tool 对姿态变化越鲁棒（能暴露"换姿态就偏"的问题），仍建议转开。
    """
    p_camera = np.asarray(p_camera, dtype=float).reshape(-1, 3)
    T_wrist = np.asarray(T_wrist, dtype=float).reshape(-1, 4, 4)
    n = len(p_camera)
    if len(T_wrist) != n:
        raise ValueError(f"点数不一致: camera {n} vs wrist {len(T_wrist)}")
    if n < MIN_SAMPLES_TOOL_ONLY:
        raise ValueError(f"至少需要 {MIN_SAMPLES_TOOL_ONLY} 个样本，当前 {n} 个")
    R = np.asarray(R_cam2base, dtype=float).reshape(3, 3)
    t = np.asarray(t_cam2base, dtype=float).reshape(3)

    R_w = T_wrist[:, :3, :3]
    t_w = T_wrist[:, :3, 3]
    target = (R @ p_camera.T).T + t - t_w        # (N,3) = R_w_i @ p_tool 应等于的值
    A = R_w.reshape(-1, 3)
    b = target.reshape(-1)
    p_tool, *_ = np.linalg.lstsq(A, b, rcond=None)

    errors_mm = np.linalg.norm((R_w @ p_tool) - target, axis=1) * 1000.0
    max_rot = max((geodesic_deg(R_w[0], R_w[i]) for i in range(1, n)), default=0.0)
    return {
        "mode": "tool_only_fixed_cam",
        "num_samples": n,
        "p_tool_wrist_m": p_tool.tolist(),
        "wrist_rotation_spread_deg": max_rot,
        "residual_mm": {
            "per_sample": errors_mm.tolist(),
            "rms": float(np.sqrt((errors_mm ** 2).mean())),
            "mean": float(errors_mm.mean()),
            "max": float(errors_mm.max()),
        },
    }


MIN_SAMPLES_PIVOT = 4
MIN_PIVOT_ROT_DEG = 25.0   # 姿态转开才能把 p_tool 的横向分量解出来


def solve_pivot(T_wrist: np.ndarray) -> dict:
    """尖点标定（pivot calibration）：指尖从多个姿态触碰同一个固定点。

    约束: R_w_i @ p_tool + t_w_i = q（固定点，基座系，未知）。
    线性最小二乘 [R_i | -I] [p_tool; q] = -t_i，一次解出 p_tool 和 q。
    不需要相机——只用手腕 FK 位姿。姿态转得越开（务必包含反手/大角度
    roll），p_tool 越可信；残差直接反映"指尖并没真正钉在同一点上"的程度。
    """
    T_wrist = np.asarray(T_wrist, dtype=float).reshape(-1, 4, 4)
    n = len(T_wrist)
    if n < MIN_SAMPLES_PIVOT:
        raise ValueError(f"尖点标定至少需要 {MIN_SAMPLES_PIVOT} 个姿态，当前 {n} 个")
    R_w = T_wrist[:, :3, :3]
    t_w = T_wrist[:, :3, 3]

    max_rot = max(geodesic_deg(R_w[0], R_w[i]) for i in range(1, n))
    if max_rot < MIN_PIVOT_ROT_DEG:
        raise ValueError(
            f"手腕姿态变化只有 {max_rot:.1f}°（需要 ≥ {MIN_PIVOT_ROT_DEG}°），"
            "请把手腕转开（含反手大角度 roll）再采样，否则指尖偏移解不出来")

    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    for i in range(n):
        A[3 * i:3 * i + 3, :3] = R_w[i]
        A[3 * i:3 * i + 3, 3:] = -np.eye(3)
        b[3 * i:3 * i + 3] = -t_w[i]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    p_tool, q = x[:3], x[3:]

    errors_mm = np.linalg.norm((R_w @ p_tool) + t_w - q, axis=1) * 1000.0
    return {
        "mode": "pivot",
        "num_samples": n,
        "p_tool_wrist_m": p_tool.tolist(),
        "pivot_point_base_m": q.tolist(),
        "wrist_rotation_spread_deg": max_rot,
        "residual_mm": {
            "per_sample": errors_mm.tolist(),
            "rms": float(np.sqrt((errors_mm ** 2).mean())),
            "mean": float(errors_mm.mean()),
            "max": float(errors_mm.max()),
        },
    }


def leave_one_out_pivot(T_wrist: np.ndarray) -> list[float]:
    """尖点标定的留一验证：剔除样本 i 解算，再看该姿态下预测指尖与固定点差多少（毫米）。"""
    T_wrist = np.asarray(T_wrist, dtype=float).reshape(-1, 4, 4)
    n = len(T_wrist)
    if n < MIN_SAMPLES_PIVOT + 1:
        return []
    errors = []
    for i in range(n):
        mask = np.arange(n) != i
        try:
            res = solve_pivot(T_wrist[mask])
        except ValueError:
            errors.append(float("nan"))
            continue
        p = np.array(res["p_tool_wrist_m"])
        q = np.array(res["pivot_point_base_m"])
        pred = T_wrist[i, :3, :3] @ p + T_wrist[i, :3, 3]
        errors.append(float(np.linalg.norm(pred - q) * 1000.0))
    return errors


def leave_one_out_tool(p_camera: np.ndarray, T_wrist: np.ndarray) -> list[float]:
    """联合解的留一交叉验证（毫米）。"""
    p_camera = np.asarray(p_camera, dtype=float).reshape(-1, 3)
    T_wrist = np.asarray(T_wrist, dtype=float).reshape(-1, 4, 4)
    n = len(p_camera)
    if n < MIN_SAMPLES_TOOL + 1:
        return []
    errors = []
    for i in range(n):
        mask = np.arange(n) != i
        try:
            res = solve_with_tool_offset(p_camera[mask], T_wrist[mask])
        except ValueError:
            errors.append(float("nan"))
            continue
        R = np.array(res["R_cam2base"])
        t = np.array(res["t_cam2base_m"])
        p_tool = np.array(res["p_tool_wrist_m"])
        pred_base = R @ p_camera[i] + t
        true_base = T_wrist[i, :3, :3] @ p_tool + T_wrist[i, :3, 3]
        errors.append(float(np.linalg.norm(pred_base - true_base) * 1000.0))
    return errors


def leave_one_out(p_camera: np.ndarray, p_base: np.ndarray) -> list[float]:
    """留一交叉验证：每次剔除一个点解算，再用该点评估预测误差（毫米）。

    比拟合残差更诚实地反映真实精度。点数 < MIN_SAMPLES+1 时返回空列表。
    """
    p_camera = np.asarray(p_camera, dtype=float).reshape(-1, 3)
    p_base = np.asarray(p_base, dtype=float).reshape(-1, 3)
    n = len(p_camera)
    if n < MIN_SAMPLES + 1:
        return []
    errors = []
    for i in range(n):
        mask = np.arange(n) != i
        try:
            res = solve_rigid_transform(p_camera[mask], p_base[mask])
        except ValueError:
            errors.append(float("nan"))
            continue
        R = np.array(res["R_cam2base"])
        t = np.array(res["t_cam2base_m"])
        pred = R @ p_camera[i] + t
        errors.append(float(np.linalg.norm(pred - p_base[i]) * 1000.0))
    return errors
