"""可插拔的"手腕位姿"读取器。

联合解模式下，机器人侧每次采样需要提供手腕位姿 T_base^wrist（4x4，米），
指尖相对手腕的偏移 p_tool 由求解器和 T_base^camera 一起解出，无需事先测量。

来源:
  - manual : 不自动读取，位姿由操作员在网页里手填 xyz+rpy（默认）
  - http   : GET JSON 端点 {"T": 4x4} 或 {"xyz": [...], "rpy": [...]}（弧度）
  - h2     : DDS 订阅 rt/lowstate（只读，绝不发指令）+ 项目内 H2 URDF FK
  - mock   : 随时间变化的假位姿，联调 UI 用

H2 说明
-------
只订阅 rt/lowstate 读关节角，不发布 rt/arm_sdk / rt/lowcmd，
和其他控制程序并存不会引起机器人抽搐。
FK 只用手臂 7 关节（腰/腿按 0 处理），因此 base_link 默认 torso_link，
这样腰部姿态不影响结果；P_base 均为该 link 坐标系下的值。
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.request

import numpy as np

from .paths import H2_ROBOT_CONFIG_PATH
from .robotics import RobotModel, load_robot_config
from .solver import make_T, rpy_to_rot

# rt/lowstate motor_state 里右臂 7 关节的下标（与 h2.yaml 的 right_arm joints 顺序一致，
# 来源: eai_teleoperate_studio/tools/h2_official_arm_sdk_control.py 的 H2JointIndex）
H2_RIGHT_ARM_MOTOR_INDICES = [22, 23, 24, 25, 26, 27, 28]
H2_LEFT_ARM_MOTOR_INDICES = [15, 16, 17, 18, 19, 20, 21]
H2_WAIST_MOTOR_INDICES = [12, 13, 14]
H2_WAIST_JOINT_NAMES = ["waist_yaw", "waist_roll", "waist_pitch"]


def read_torso_state(low_state) -> dict:
    """从一帧 rt/lowstate 里取躯干姿态：腰三关节 + IMU 姿态。

    手臂 IK 全部在 torso_link 系下解算，隐含假设"躯干不动"。本体控制器
    在运动模式下会为了平衡而动腰/踝，手抬起来时躯干可能后仰几度——
    那样即使手臂关节角完全到位，指尖在世界系里也偏了。这个函数负责
    把"躯干到底动了多少"如实读出来，供执行前后对比。
    """
    imu = getattr(low_state, "imu_state", None)
    quat = [float(v) for v in getattr(imu, "quaternion", [1.0, 0.0, 0.0, 0.0])]
    rpy = [float(v) for v in getattr(imu, "rpy", [0.0, 0.0, 0.0])]
    return {
        "waist_rad": [float(low_state.motor_state[i].q) for i in H2_WAIST_MOTOR_INDICES],
        "waist_names": list(H2_WAIST_JOINT_NAMES),
        "imu_quat": quat,
        "imu_rpy": rpy,
    }


class PoseProvider:
    """接口: read_pose() 返回 T_base^wrist (4,4)，失败时抛异常。"""

    source: str = "base"
    available: bool = True
    base_link: str = "?"
    wrist_link: str = "?"

    def read_pose(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ManualPoseProvider(PoseProvider):
    source = "manual"
    available = False

    def read_pose(self) -> np.ndarray:
        raise RuntimeError("manual 模式没有自动读取，请在界面里填手腕位姿")


class MockPoseProvider(PoseProvider):
    source = "mock"
    base_link = "mock_base"
    wrist_link = "mock_wrist"

    def read_pose(self) -> np.ndarray:
        a = (time.monotonic() * 0.3) % (2 * math.pi)
        R = rpy_to_rot(0.4 * math.sin(a), 0.3 * math.cos(a), a * 0.5)
        return make_T(R, [0.3 + 0.1 * math.cos(a), -0.2, 0.1 + 0.1 * math.sin(a)])


class HttpPoseProvider(PoseProvider):
    """GET JSON 端点，机器人侧 sidecar 做 FK 后发布即可。"""

    source = "http"

    def __init__(self, url: str, timeout: float = 2.0):
        self.url = url
        self.timeout = float(timeout)

    def read_pose(self) -> np.ndarray:
        req = urllib.request.Request(self.url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        if "T" in data:
            T = np.asarray(data["T"], dtype=float).reshape(4, 4)
        elif "xyz" in data and "rpy" in data:
            T = make_T(rpy_to_rot(*[float(v) for v in data["rpy"]]),
                       [float(v) for v in data["xyz"]])
        else:
            raise ValueError(f"端点需返回 {{'T': 4x4}} 或 {{'xyz','rpy'}}，实际: {data!r}")
        if not np.all(np.isfinite(T)):
            raise ValueError("位姿包含非法值")
        return T


class H2PoseProvider(PoseProvider):
    """H2 真机：订阅 rt/lowstate（只读）→ 项目内 URDF FK → T_torso^wrist。

    依赖:
      - unitree_sdk2py（DDS）
      - 本项目 config/robots/h2.yaml 与 assets/robots/h2/robot.urdf
    """

    source = "h2"

    def __init__(self, network_interface: str | None = None,
                 arm: str = "right", base_link: str | None = None,
                 lowstate_timeout: float = 5.0, q_reader=None):
        """q_reader: 可选的无参函数，返回该手臂 7 关节角。
        提供时（如共用 H2ArmController 的订阅）不再自建 DDS 订阅。"""
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        from .dds import ensure_dds_initialized

        cfg = load_robot_config(H2_ROBOT_CONFIG_PATH)
        self._model = RobotModel(cfg)
        self._chain = f"{arm}_arm"
        if self._chain not in self._model.chain_ids:
            raise ValueError(f"h2.yaml 中没有链 {self._chain!r}（可选: {self._model.chain_ids}）")
        self._joint_names = self._model.joint_names(self._chain)
        self._motor_indices = (H2_RIGHT_ARM_MOTOR_INDICES if arm == "right"
                               else H2_LEFT_ARM_MOTOR_INDICES)
        self.base_link = base_link or self._model.base_link(self._chain)
        self.wrist_link = self._model.end_link(self._chain)

        self._q_reader = q_reader
        self._lock = threading.Lock()
        self._low_state = None
        if q_reader is None:
            # DDS 初始化 + 订阅（只读，不创建任何 publisher）
            ensure_dds_initialized(network_interface)
            self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self._subscriber.Init(self._on_low_state, 10)

            deadline = time.monotonic() + lowstate_timeout
            while self._low_state is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"{lowstate_timeout:.0f}s 内没收到 rt/lowstate（网卡对吗？机器人开机了吗？）")
                time.sleep(0.05)

    def _on_low_state(self, msg) -> None:
        with self._lock:
            self._low_state = msg

    def read_arm_q(self) -> np.ndarray:
        if self._q_reader is not None:
            return np.asarray(self._q_reader(), dtype=float).reshape(-1)
        with self._lock:
            state = self._low_state
        if state is None:
            raise RuntimeError("还没收到 rt/lowstate")
        return np.asarray([state.motor_state[i].q for i in self._motor_indices], dtype=float)

    def read_torso_state(self) -> dict:
        """腰关节 + IMU 姿态（只读）。没有自建订阅时返回 None。"""
        with self._lock:
            state = self._low_state
        return read_torso_state(state) if state is not None else None

    def read_motor_q(self, indices) -> list | None:
        """按全身电机序号读任意电机角度（rad，只读）。还没收到帧返回 None。"""
        with self._lock:
            state = self._low_state
        if state is None:
            return None
        return [float(state.motor_state[int(i)].q) for i in indices]

    def read_pose(self) -> np.ndarray:
        q = self.read_arm_q()
        joint_values = dict(zip(self._joint_names, q.tolist()))
        transforms = self._model.forward_kinematics(joint_values)
        for link in (self.base_link, self.wrist_link):
            if link not in transforms:
                raise RuntimeError(f"FK 结果里没有 link {link!r}")
        # forward_kinematics 以 URDF 根为参考，换算成 base_link 系
        T_root_base = transforms[self.base_link]
        T_root_wrist = transforms[self.wrist_link]
        return np.linalg.inv(T_root_base) @ T_root_wrist


def make_pose_provider(source: str, *, http_url: str | None = None,
                       network_interface: str | None = None,
                       arm: str = "right",
                       base_link: str | None = None,
                       q_reader=None) -> PoseProvider:
    if source == "manual":
        return ManualPoseProvider()
    if source == "mock":
        return MockPoseProvider()
    if source == "http":
        if not http_url:
            raise ValueError("pose source 'http' 需要 --pose-url")
        return HttpPoseProvider(http_url)
    if source == "h2":
        return H2PoseProvider(network_interface=network_interface,
                              arm=arm, base_link=base_link, q_reader=q_reader)
    raise ValueError(f"未知 pose source: {source!r}（可选 manual/http/h2/mock）")
