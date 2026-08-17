#!/usr/bin/env python3
"""重力前馈自检：与 pinocchio RNEA 对拍 + 50Hz 控制环耗时 + 典型姿态力矩表。

    python tools/check_gravity.py [--payload-kg 0.2]

pinocchio 缺失时只跑数值差分对拍（势能梯度的独立实现），仍能验证解析式正确性。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.gravity import G_ACCEL, ArmGravityModel  # noqa: E402
from backend.paths import H2_ROBOT_CONFIG_PATH  # noqa: E402
from backend.robotics import RobotModel, load_robot_config  # noqa: E402


def potential_energy(model, gm, q) -> float:
    """独立实现：全身 FK + Σ m g h，用来做数值差分对拍。"""
    values = dict(zip(gm.joint_names, [float(v) for v in q]))
    transforms = model.forward_kinematics(values)
    total = 0.0
    for link, body in gm.bodies.items():
        p = transforms[link][:3, :3] @ body.com + transforms[link][:3, 3]
        total += body.mass * G_ACCEL * float(p[2])
    return total


def numeric_torque(model, gm, q, eps=1e-6) -> np.ndarray:
    out = np.zeros(len(q))
    for i in range(len(q)):
        qp, qm = np.array(q, dtype=float), np.array(q, dtype=float)
        qp[i] += eps
        qm[i] -= eps
        out[i] = (potential_energy(model, gm, qp) - potential_energy(model, gm, qm)) / (2 * eps)
    return out


def pinocchio_torque(urdf_path, gm, q):
    import pinocchio as pin

    full = pin.buildModelFromUrdf(str(urdf_path))
    keep = [full.getJointId(n) for n in gm.joint_names]
    lock = [jid for jid in range(1, full.njoints) if jid not in keep]
    reduced = pin.buildReducedModel(full, lock, pin.neutral(full))
    data = reduced.createData()
    order = [reduced.getJointId(n) for n in gm.joint_names]
    q_pin = pin.neutral(reduced)
    for name, value in zip(gm.joint_names, q):
        idx = reduced.joints[reduced.getJointId(name)].idx_q
        q_pin[idx] = float(value)
    tau = pin.rnea(reduced, data, q_pin, np.zeros(reduced.nv), np.zeros(reduced.nv))
    out = np.zeros(len(gm.joint_names))
    for i, name in enumerate(gm.joint_names):
        out[i] = tau[reduced.joints[reduced.getJointId(name)].idx_v]
    return out, order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="right_arm")
    ap.add_argument("--payload-kg", type=float, default=0.0)
    args = ap.parse_args()

    model = RobotModel(load_robot_config(H2_ROBOT_CONFIG_PATH))
    gm = ArmGravityModel(model, args.chain, payload_kg=args.payload_kg)

    print("== 模型 ==")
    for key, value in gm.describe().items():
        print(f"  {key}: {value}")
    print(f"  joints: {gm.joint_names}")

    rng = np.random.default_rng(0)
    lower, upper = model.joint_limits(args.chain)
    print("\n== 与数值差分（势能梯度，独立实现）对拍 ==")
    worst = 0.0
    for _ in range(20):
        q = rng.uniform(np.maximum(lower, -2.0), np.minimum(upper, 2.0))
        err = float(np.max(np.abs(gm.torque(q) - numeric_torque(model, gm, q))))
        worst = max(worst, err)
    print(f"  20 组随机姿态最大偏差: {worst:.3e} Nm  -> {'OK' if worst < 1e-4 else '不一致!'}")

    print("\n== 与 pinocchio RNEA 对拍 ==")
    try:
        worst = 0.0
        for _ in range(10):
            q = rng.uniform(np.maximum(lower, -2.0), np.minimum(upper, 2.0))
            tau_pin, _ = pinocchio_torque(model.urdf_path, gm, q)
            err = float(np.max(np.abs(gm.torque(q) - tau_pin)))
            worst = max(worst, err)
        print(f"  10 组随机姿态最大偏差: {worst:.3e} Nm  -> {'OK' if worst < 1e-3 else '不一致!'}")
    except Exception as exc:
        print(f"  跳过（{type(exc).__name__}: {exc}）")

    print("\n== 50Hz 控制环耗时 ==")
    q = np.zeros(len(gm.joint_names))
    gm.torque(q)
    t0 = time.perf_counter()
    for _ in range(2000):
        gm.torque(q)
    per_call_ms = (time.perf_counter() - t0) / 2000 * 1e3
    print(f"  单次 {per_call_ms:.3f} ms，占 20ms 周期的 {per_call_ms / 20 * 100:.2f}%")

    print("\n== 典型姿态的重力力矩 (Nm) ==")
    poses = {
        "零位（手臂自然下垂）": np.zeros(7),
        "肩前抬 90°": np.array([-1.57, 0, 0, 0, 0, 0, 0]),
        "前平举 + 肘 45°": np.array([-1.57, 0, 0, 0.785, 0, 0, 0]),
        "抬到胸前（拨开关姿态附近）": np.array([-1.0, -0.2, 0.0, 0.9, 0.0, 0.0, 0.0]),
    }
    names = [n.replace("right_", "").replace("_joint", "") for n in gm.joint_names]
    print("  " + " ".join(f"{n:>14}" for n in names))
    for label, pose in poses.items():
        tau = gm.torque(pose)
        print("  " + " ".join(f"{v:>14.2f}" for v in tau) + f"   <- {label}")

    print("\n== 姿态倾斜（IMU 修正）敏感度 ==")
    from backend.gravity import gravity_dir_from_quaternion

    q = poses["抬到胸前（拨开关姿态附近）"]
    base = gm.torque(q)
    for pitch_deg in (-5.0, 5.0, 10.0):
        half = np.deg2rad(pitch_deg) / 2
        quat = (np.cos(half), 0.0, np.sin(half), 0.0)   # 绕 y 轴（俯仰）
        tau = gm.torque(q, g_dir=gravity_dir_from_quaternion(quat))
        print(f"  躯干俯仰 {pitch_deg:+.0f}°: 力矩最大变化 {np.max(np.abs(tau - base)):.2f} Nm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
