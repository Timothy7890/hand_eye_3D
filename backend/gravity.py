"""手臂重力前馈：从 URDF 的 inertial 直接算出"托住自重需要多少力矩"τ_g(q)。

为什么需要它
------------
H2 的关节是纯 PD（kp/kd）+ 前馈力矩，低层没有积分项。纯 PD 想产生托举力矩
只能靠"位置偏差 × kp"，所以手臂必然停在低于目标若干度的地方（下垂）。
把 τ_g 当前馈直接喂进去，PD 只需要负责"消除误差"而不必"扛住重力"，
稳态偏差可以从几度降到零点几度。官方 VR 遥操（xr_teleoperate）用
pinocchio 的 RNEA 做同一件事，这里用解析式重力项，结果与 RNEA 一致
（零速度零加速度时 RNEA 退化为纯重力项），但不引入 pinocchio 依赖。

算法
----
势能 U(q) = Σ_j m_j · g_vec · p_cj(q)（对该关节下游的所有连杆求和），
对转动关节 i 有 ∂p_cj/∂q_i = z_i × (p_cj − p_i)，于是

    τ_i = ∂U/∂q_i = −Σ_{j ∈ subtree(i)} m_j · g_vec · (z_i × (p_cj − p_i))

一次 FK 就能拿到全部 z_i / p_i / p_cj，7 个关节的力矩一次算完，
没有数值差分，50Hz 控制环里跑毫无压力。

坐标系
------
一切在 URDF 根系（pelvis）下计算。重力方向默认取 (0,0,-1)，即假设躯干直立；
躯干前倾/后仰时可以把 IMU 测到的重力方向传进 torque(g_dir=...) 做修正。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

G_ACCEL = 9.81


@dataclass(frozen=True)
class BodyInertial:
    """连杆的质量与质心（质心在连杆自身坐标系下）。"""

    link: str
    mass: float
    com: np.ndarray


def parse_inertials(urdf_path) -> dict[str, BodyInertial]:
    """读 URDF 里每个 link 的 <inertial>（没有 inertial 或质量为 0 的跳过）。"""
    root = ET.parse(str(urdf_path)).getroot()
    out: dict[str, BodyInertial] = {}
    for link_el in root.findall("link"):
        inertial_el = link_el.find("inertial")
        if inertial_el is None:
            continue
        mass_el = inertial_el.find("mass")
        mass = float(mass_el.attrib.get("value", 0.0)) if mass_el is not None else 0.0
        if mass <= 0.0:
            continue
        origin_el = inertial_el.find("origin")
        xyz = (origin_el.attrib.get("xyz", "0 0 0")
               if origin_el is not None else "0 0 0")
        com = np.array([float(v) for v in xyz.split()], dtype=float)
        name = link_el.attrib["name"]
        out[name] = BodyInertial(link=name, mass=mass, com=com)
    return out


def parse_effort_limits(urdf_path, joint_names) -> np.ndarray:
    """读 URDF <limit effort=...>，用作前馈力矩的安全上限（缺省给 10Nm 保守值）。"""
    root = ET.parse(str(urdf_path)).getroot()
    table: dict[str, float] = {}
    for joint_el in root.findall("joint"):
        limit_el = joint_el.find("limit")
        if limit_el is None:
            continue
        try:
            table[joint_el.attrib["name"]] = float(limit_el.attrib.get("effort", 0.0))
        except ValueError:
            continue
    return np.array([table.get(name) or 10.0 for name in joint_names], dtype=float)


class ArmGravityModel:
    """某条链（如 right_arm）的重力力矩模型。

    payload_kg: 模型之外的额外负载（比如换装的因时灵巧手比 URDF 里的官方手重）。
    默认挂在链末端最深的那个有质量的连杆的质心上——语义就是"这只手比模型重 N 公斤"。
    """

    def __init__(self, model, chain_id: str, *,
                 payload_kg: float = 0.0, payload_link: str | None = None):
        self.model = model
        self.chain_id = chain_id
        self.joint_names = list(model.joint_names(chain_id))
        self.base_link = model.base_link(chain_id)

        inertials = parse_inertials(model.urdf_path)

        # 根系 → 链基座（腰/腿为零位时的静态变换，只算一次）
        self._T_root_base = model.forward_kinematics({})[self.base_link].copy()

        self._chain_joints = [model.joints[n] for n in self.joint_names]
        subtrees = [self._collect_subtree(j.child) for j in self._chain_joints]

        # 只有"关节下游"的质量会在该关节上产生重力力矩：躯干本体、头、另一条
        # 手臂虽然也在 base_link 之下，但不受这 7 个关节驱动，一律不参与计算
        moving = {link for links in subtrees for link in links} & set(inertials)

        self.payload_kg = float(payload_kg)
        self.payload_link = payload_link or self._deepest_massive_link(inertials)
        self.bodies: dict[str, BodyInertial] = {}
        for link in sorted(moving):
            body = inertials[link]
            if link == self.payload_link and self.payload_kg > 0.0:
                body = BodyInertial(link=link, mass=body.mass + self.payload_kg,
                                    com=body.com)
            self.bodies[link] = body

        self._downstream = [[ln for ln in links if ln in self.bodies] for links in subtrees]

        # FK 只展开"通往这些连杆"的分支，并预存静态量，免得每周期重解析 URDF
        needed = set(self.bodies) | {j.child for j in self._chain_joints}
        self._joint_static: dict[str, tuple] = {}
        self._expand: dict[str, list[str]] = {}
        for link in self._collect_subtree(self.base_link):
            for joint in model.parent_to_joints.get(link, []):
                if not (set(self._collect_subtree(joint.child)) & needed):
                    continue
                axis = np.asarray(joint.axis, dtype=float)
                norm = float(np.linalg.norm(axis))
                self._joint_static[joint.name] = (
                    joint,
                    _origin_transform(joint),
                    axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0]),
                )
                self._expand.setdefault(link, []).append(joint.name)

    # ---- URDF 树遍历 ----

    def _collect_subtree(self, root_link: str) -> list[str]:
        out: list[str] = []
        stack = [root_link]
        while stack:
            link = stack.pop()
            out.append(link)
            for joint in self.model.parent_to_joints.get(link, []):
                stack.append(joint.child)
        return out

    def _deepest_massive_link(self, inertials: dict[str, BodyInertial]) -> str | None:
        """链末端往下最深的有质量连杆——手掌通常就挂在这里。"""
        end_link = self.model.end_link(self.chain_id)
        best, best_depth = None, -1
        stack = [(end_link, 0)]
        while stack:
            link, depth = stack.pop()
            if link in inertials and depth > best_depth:
                best, best_depth = link, depth
            for joint in self.model.parent_to_joints.get(link, []):
                stack.append((joint.child, depth + 1))
        return best

    # ---- 正运动学（只走这条链，不遍历全身） ----

    def _fk(self, q: np.ndarray) -> dict[str, np.ndarray]:
        values = dict(zip(self.joint_names, [float(v) for v in q]))
        transforms = {self.base_link: self._T_root_base}
        stack = [self.base_link]
        while stack:
            link = stack.pop()
            parent_T = transforms[link]
            for joint_name in self._expand.get(link, []):
                joint, origin, axis = self._joint_static[joint_name]
                T = parent_T @ origin
                if joint.joint_type in ("revolute", "continuous"):
                    value = values.get(joint.name, 0.0)
                    if value != 0.0:
                        T = T @ _axis_rotation(axis, value)
                elif joint.joint_type == "prismatic":
                    T = T.copy()
                    T[:3, 3] += T[:3, :3] @ (axis * values.get(joint.name, 0.0))
                transforms[joint.child] = T
                stack.append(joint.child)
        return transforms

    # ---- 重力力矩 ----

    def torque(self, q, g_dir=None) -> np.ndarray:
        """返回该链 7 个关节托住自重所需的力矩 (Nm)。

        q: 关节角（按 chain 的关节顺序）。用【指令角】而不是实测角来算，
           这样前馈不会被实测噪声/下垂污染（官方遥操也是用解算出的目标角）。
        g_dir: 重力单位方向在 URDF 根系下的分量，默认 (0,0,-1)（躯干直立）。
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != len(self.joint_names):
            raise ValueError(f"需要 {len(self.joint_names)} 个关节角，收到 {q.size}")
        if g_dir is None:
            g_vec = np.array([0.0, 0.0, -G_ACCEL])
        else:
            g_dir = np.asarray(g_dir, dtype=float).reshape(3)
            norm = float(np.linalg.norm(g_dir))
            g_vec = g_dir / norm * G_ACCEL if norm > 1e-9 else np.array([0.0, 0.0, -G_ACCEL])

        transforms = self._fk(q)
        coms = {link: transforms[link][:3, :3] @ body.com + transforms[link][:3, 3]
                for link, body in self.bodies.items()}

        tau = np.zeros(len(self.joint_names))
        for i, joint in enumerate(self._chain_joints):
            T = transforms[joint.child]
            axis = self._joint_static[joint.name][2]
            z_i = T[:3, :3] @ axis          # 关节轴在根系下的方向
            p_i = T[:3, 3]                  # 关节原点在根系下的位置
            total = 0.0
            if joint.joint_type == "prismatic":
                for link in self._downstream[i]:
                    total += self.bodies[link].mass * float(g_vec @ z_i)
            else:
                for link in self._downstream[i]:
                    lever = np.cross(z_i, coms[link] - p_i)
                    total += self.bodies[link].mass * float(g_vec @ lever)
            tau[i] = -total
        return tau

    # ---- 自检信息 ----

    def describe(self) -> dict:
        moving = sorted({link for group in self._downstream for link in group})
        return {
            "chain": self.chain_id,
            "base_link": self.base_link,
            "moving_links": {ln: round(self.bodies[ln].mass, 4) for ln in moving},
            "moving_mass_kg": round(sum(self.bodies[ln].mass for ln in moving), 4),
            "payload_kg": self.payload_kg,
            "payload_link": self.payload_link,
        }


def _origin_transform(joint) -> np.ndarray:
    """URDF <origin xyz rpy> → 4x4。"""
    rpy = np.asarray(joint.rpy, dtype=float)
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(joint.xyz, dtype=float)
    return T


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """罗德里格斯公式，绕单位轴转 angle。"""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + s * K + (1.0 - c) * (K @ K)
    return T


def gravity_dir_from_quaternion(quat) -> np.ndarray:
    """IMU 四元数 (w,x,y,z，躯干在世界系下的姿态) → 重力方向在根系下的分量。

    直立时返回 (0,0,-1)；躯干前倾/后仰时给出倾斜后的真实重力方向。
    """
    w, x, y, z = [float(v) for v in quat]
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R.T @ np.array([0.0, 0.0, -1.0])
