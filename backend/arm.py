"""H2 手臂点动控制器（真机运动！），走官方 rt/arm_sdk 混合通道。

安全模型（与 hand_eye/backend/arm.py 一致）:
- 官方 arm_sdk 需要持续流式发令，后台线程以 50Hz 发送位置保持目标；
- 发出的目标 (cmd_q) 只会以受限速度向期望目标 (desired_q) 滑动，
  且始终钳制在 URDF 关节限位内 → 界面狂点也只会平滑慢速运动；
- 启动时先读当前实测姿态并从它开始保持，权重 1 秒内 0→1 渐入，不会跳变；
- 点动默认锁定，enable_jog() 后才接受目标；
- 卸力模式：被控手臂 kp=0、kd=小阻尼，人可以拖动（手臂会下坠，必须扶住），
  恢复点动时从人放置的位置重新抓取保持；
- 退出时权重 1 秒渐出交还本体控制器——退出前请扶住手臂。

只点动一条手臂（--arm），另一条手臂全程保持在启动时的实测姿态。

重力前馈
--------
关节低层只有 kp/kd 两项，没有积分，纯 PD 想托住手臂自重只能靠"位置偏差×kp"，
所以手臂必然停在低于目标几度的地方（这正是之前"抬不到位、够不着开关"的根因）。
每周期按当前指令角算出托举力矩 τ_g(q) 直接前馈进去，PD 就只负责消误差而不用
扛重力，稳态偏差从几度降到零点几度。算法与官方 VR 遥操（xr_teleoperate 的
pinocchio RNEA）一致，已逐点对拍到 1e-15 Nm，见 tools/check_gravity.py。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

from .dds import ensure_dds_initialized
from .gravity import ArmGravityModel, gravity_dir_from_quaternion, parse_effort_limits
from .robot import (
    H2_LEFT_ARM_MOTOR_INDICES,
    H2_RIGHT_ARM_MOTOR_INDICES,
    IK_REPLAY_ROOT,
    read_torso_state,
)

CONTROL_DT = 0.02          # 50Hz，官方示例节拍
WEIGHT_RAMP_S = 1.0        # 权重渐入/渐出时长
ARM_SDK_TOPIC = "rt/arm_sdk"
LOWSTATE_TOPIC = "rt/lowstate"
WEIGHT_MOTOR_INDEX = 31
DEFAULT_KP = 80.0
DEFAULT_KD = 1.5
# 腕部电机小得多（URDF effort 10Nm vs 肩 130Nm），跟大关节同一档刚度会发抖
DEFAULT_KP_WRIST = 50.0
DEFAULT_KD_WRIST = 2.0
PUSH_TAU_LIMIT = 20.0      # 主动出力（拨开关）的上限
EFFORT_MARGIN = 0.6        # 前馈总量不超过 URDF 额定力矩的这个比例

# 注意：曾试过经由 rt/arm_sdk 控制腰 yaw（12 号电机），真机验证固件直接
# 忽略——H2 的 arm_sdk 混合通道只覆盖双臂 15~28 + 权重 31（官方例程同）。
# 要转腰/对准柜面请走高层 loco 的 SetVelocity 原地转身，见 reach 的 /turn。


def _load_arm_model(arm: str):
    """加载 IK_replay 的 h2 URDF 模型，返回 (model, chain_id)。"""
    if str(IK_REPLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(IK_REPLAY_ROOT))
    from core.robot_config import load_robot_config
    from core.robot_model import RobotModel

    model = RobotModel(load_robot_config(IK_REPLAY_ROOT / "config" / "robots" / "h2.yaml"))
    return model, f"{arm}_arm"


class H2ArmController:
    """位置保持 + 限速点动 + 卸力拖动，发布 rt/arm_sdk（真机运动）。"""

    def __init__(self, arm: str = "right", network_interface: str | None = None,
                 max_speed_rad_s: float = 0.2, hand_move_kd: float = 2.0,
                 kp: float = DEFAULT_KP, kd: float = DEFAULT_KD,
                 kp_wrist: float | None = None, kd_wrist: float | None = None,
                 grav_alpha: float = 1.0, payload_kg: float = 0.0,
                 grav_in_float: bool = False, use_imu_gravity: bool = False,
                 lowstate_timeout: float = 5.0):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.arm = arm
        model, chain = _load_arm_model(arm)
        self.joint_names = model.joint_names(chain)
        lower, upper = model.joint_limits(chain)
        self.limits = np.stack([lower, upper], axis=1)
        self.n = len(self.joint_names)
        self.max_speed = float(max_speed_rad_s)
        self._speed_ceiling = self.max_speed   # 启动参数是天花板，段级只能往下调
        self.hand_move_kd = float(hand_move_kd)
        self.kp = float(kp)
        self.kd = float(kd)
        self.kp_wrist = float(kp_wrist if kp_wrist is not None else DEFAULT_KP_WRIST)
        self.kd_wrist = float(kd_wrist if kd_wrist is not None else DEFAULT_KD_WRIST)
        is_wrist = np.array(["wrist" in n for n in self.joint_names])
        self.kp_vec = np.where(is_wrist, self.kp_wrist, self.kp)
        self.kd_vec = np.where(is_wrist, self.kd_wrist, self.kd)

        # ---- 重力前馈 ----
        self.grav_alpha = float(np.clip(grav_alpha, 0.0, 1.2))
        self.payload_kg = float(payload_kg)
        self.grav_in_float = bool(grav_in_float)
        self.use_imu_gravity = bool(use_imu_gravity)
        self._grav_model = (ArmGravityModel(model, chain, payload_kg=self.payload_kg)
                            if self.grav_alpha > 0 else None)
        self._tau_cap = EFFORT_MARGIN * parse_effort_limits(model.urdf_path, self.joint_names)

        self._jog_indices = (H2_RIGHT_ARM_MOTOR_INDICES if arm == "right"
                             else H2_LEFT_ARM_MOTOR_INDICES)
        self._other_indices = (H2_LEFT_ARM_MOTOR_INDICES if arm == "right"
                               else H2_RIGHT_ARM_MOTOR_INDICES)

        ensure_dds_initialized(network_interface)
        self._crc = CRC()
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._publisher = ChannelPublisher(ARM_SDK_TOPIC, LowCmd_)
        self._publisher.Init()
        self._state_lock = threading.Lock()
        self._low_state = None
        self._subscriber = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        self._subscriber.Init(self._on_low_state, 10)

        deadline = time.monotonic() + lowstate_timeout
        while self._low_state is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{lowstate_timeout:.0f}s 内没收到 {LOWSTATE_TOPIC}")
            time.sleep(0.05)

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._engaged = False
        self._jog_enabled = False
        self._float = False
        self._weight = 0.0

        q0 = self._clamp(self._read_motors(self._jog_indices))
        self._cmd_q = q0.copy()
        self._desired_q = q0.copy()
        self._tau_push = np.zeros(self.n)   # 主动出力（按压/拨动），外部设定
        self._tau_grav = np.zeros(self.n)   # 重力前馈，每周期按指令角重算
        self._other_hold_q = self._read_motors(self._other_indices)
        # 另一条手臂全程定姿保持，重力力矩是常数，开机算一次即可；
        # 不给它前馈的话，接管瞬间它会比本体控制器托着时又往下掉几度
        self._other_tau = np.zeros(self.n)
        if self._grav_model is not None:
            other_chain = "left_arm" if arm == "right" else "right_arm"
            other_grav = ArmGravityModel(model, other_chain, payload_kg=self.payload_kg)
            self._other_tau = np.clip(self.grav_alpha * other_grav.torque(self._other_hold_q),
                                      -self._tau_cap, self._tau_cap)
        self._thread = threading.Thread(target=self._loop, name="h2-arm-jog", daemon=True)

    # ---- DDS 读 ----

    def _on_low_state(self, msg) -> None:
        with self._state_lock:
            self._low_state = msg

    def _read_motors(self, indices) -> np.ndarray:
        with self._state_lock:
            state = self._low_state
        if state is None:
            raise RuntimeError("还没收到 rt/lowstate")
        return np.asarray([state.motor_state[i].q for i in indices], dtype=float)

    def read_measured(self) -> np.ndarray:
        return self._read_motors(self._jog_indices)

    def read_torso_state(self) -> dict | None:
        """腰三关节 + IMU 姿态，用于"手臂到位了但躯干动了吗"的诊断。"""
        with self._state_lock:
            state = self._low_state
        return read_torso_state(state) if state is not None else None

    def _gravity_dir(self) -> np.ndarray | None:
        """重力方向（根系）。默认按躯干直立处理；开了 IMU 修正才用实测姿态。"""
        if not self.use_imu_gravity:
            return None
        with self._state_lock:
            state = self._low_state
        if state is None:
            return None
        try:
            return gravity_dir_from_quaternion(state.imu_state.quaternion)
        except Exception:
            return None

    # ---- 生命周期 ----

    def start(self) -> None:
        """立即在当前实测姿态开始保持（点动仍锁定）。"""
        self._engaged = True
        self._thread.start()

    def shutdown(self) -> None:
        """权重渐出后停止发布。调用前请扶住手臂。"""
        self._stop_evt.set()
        self._thread.join(WEIGHT_RAMP_S + 1.0)

    # ---- 控制循环 ----

    def _clamp(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        return np.minimum(np.maximum(q, self.limits[:, 0]), self.limits[:, 1])

    def _write_command(self, jog_q: np.ndarray, float_mode: bool, weight: float,
                       tau_ff: np.ndarray) -> None:
        cmd = self._low_cmd
        cmd.motor_cmd[WEIGHT_MOTOR_INDEX].q = float(weight)
        for i, idx in enumerate(self._jog_indices):
            m = cmd.motor_cmd[idx]
            m.tau = float(tau_ff[i])
            m.q = float(jog_q[i])
            m.dq = 0.0
            m.kp = 0.0 if float_mode else float(self.kp_vec[i])
            m.kd = self.hand_move_kd if float_mode else float(self.kd_vec[i])
        for i, idx in enumerate(self._other_indices):
            m = cmd.motor_cmd[idx]
            m.tau = float(self._other_tau[i] * weight)
            m.q = float(self._other_hold_q[i])
            m.dq = 0.0
            m.kp = float(self.kp_vec[i])
            m.kd = float(self.kd_vec[i])
        cmd.crc = self._crc.Crc(cmd)
        self._publisher.Write(cmd)

    def _compute_tau(self, cmd_q: np.ndarray, tau_push: np.ndarray,
                     float_mode: bool, weight: float) -> np.ndarray:
        """总前馈 = 重力托举 + 主动出力，再按电机额定力矩兜底钳位。

        保持/点动模式下重力项按【指令角】算而不是实测角：实测角本身就含下垂，
        拿它算前馈等于承认下垂，越补越低；指令角是我们想要它待的地方，托举
        力矩就该按那里算。卸力模式下没有位置目标（kp=0），调用方会传入
        【实测角】——托举的是手臂当前实际所在的姿态，人拖到哪补到哪。
        权重渐入期间同步缩放，避免接管瞬间力矩阶跃。
        """
        tau_grav = np.zeros(self.n)
        if self._grav_model is not None and (not float_mode or self.grav_in_float):
            try:
                tau_grav = self.grav_alpha * self._grav_model.torque(
                    cmd_q, g_dir=self._gravity_dir())
            except Exception:
                tau_grav = np.zeros(self.n)
        self._tau_grav = tau_grav
        tau = tau_grav + (np.zeros(self.n) if float_mode else tau_push)
        return np.clip(tau * float(weight), -self._tau_cap, self._tau_cap)

    def _loop(self) -> None:
        next_t = time.perf_counter()
        while True:
            stopping = self._stop_evt.is_set()
            with self._lock:
                float_mode = self._float
                if stopping:
                    self._weight = max(0.0, self._weight - CONTROL_DT / WEIGHT_RAMP_S)
                else:
                    self._weight = min(1.0, self._weight + CONTROL_DT / WEIGHT_RAMP_S)
                weight = self._weight
                if not float_mode:
                    # 矢量同步限速：按最饱和的关节整体等比减速，方向不变，
                    # 所有关节同时到达 → 关节空间直线不会被扭成"先平移后抬升"
                    step = self.max_speed * CONTROL_DT
                    delta = self._desired_q - self._cmd_q
                    worst = float(np.max(np.abs(delta)))
                    if worst > step:
                        delta = delta * (step / worst)
                    self._cmd_q = self._cmd_q + delta
                cmd_q = self._cmd_q.copy()
                tau_push = self._tau_push.copy()
            # 重力前馈用哪个角度算，两种模式不同（见 _compute_tau 注释）：
            # 保持/点动 → 指令角（抗下垂）；卸力 → 实测角（cmd_q 冻结在入口
            # 姿态，人一拖远前馈就全错，手臂会被错误力矩推得乱扭）
            q_grav = cmd_q
            if float_mode:
                measured = self._safe_measured()
                if measured is not None:
                    q_grav = measured
            tau_ff = self._compute_tau(q_grav, tau_push, float_mode, weight)
            try:
                self._write_command(cmd_q, float_mode, weight, tau_ff)
            except Exception:
                pass
            if stopping and weight <= 0.0:
                break
            next_t += CONTROL_DT
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    # ---- 控制操作（与 hand_eye 的 ArmController 同名同语义） ----

    def _safe_measured(self) -> np.ndarray | None:
        try:
            return self._clamp(self.read_measured())
        except Exception:
            return None

    def describe_gravity(self) -> dict:
        """启动时打印用：参与计算的连杆质量、当前姿态需要多少托举力矩。"""
        if self._grav_model is None:
            return {"enabled": False}
        info = dict(self._grav_model.describe())
        info["enabled"] = True
        info["alpha"] = self.grav_alpha
        try:
            tau = self.grav_alpha * self._grav_model.torque(self.read_measured())
            info["tau_now_nm"] = [round(float(v), 2) for v in tau]
        except Exception:
            pass
        info["tau_cap_nm"] = [round(float(v), 1) for v in self._tau_cap]
        return info

    def enter_hand_move(self) -> bool:
        """卸力拖动，仅点动关闭时允许。

        grav_in_float=False（默认）时手臂会下坠，必须有人扶住；
        开了之后重力前馈在卸力态继续给，手臂近似"失重"，推到哪停哪，
        录路点会轻松很多——代价是补过头时手臂会缓慢上飘，需要先验证 alpha。
        """
        with self._lock:
            if self._jog_enabled:
                return False
            self._float = True
        return True

    def enable_jog(self) -> None:
        measured = self._safe_measured() if self._float else None
        with self._lock:
            if self._float and measured is not None:
                self._cmd_q = measured
            self._float = False
            self._desired_q = self._cmd_q.copy()
            self._jog_enabled = True
            self._tau_push[:] = 0.0

    def disable_jog(self) -> None:
        with self._lock:
            self._desired_q = self._cmd_q.copy()
            self._jog_enabled = False
            self._tau_push[:] = 0.0

    def stop(self) -> None:
        """冻结 + 刚性保持（也用于退出卸力模式）。"""
        measured = self._safe_measured() if self._float else None
        with self._lock:
            if self._float and measured is not None:
                self._cmd_q = measured
            self._float = False
            self._desired_q = self._cmd_q.copy()
            self._jog_enabled = False
            self._tau_push[:] = 0.0

    def set_max_speed(self, v: float) -> None:
        """段级速度档：普通段慢而稳，快拨段提速；不会超过启动参数的天花板。"""
        with self._lock:
            self.max_speed = float(np.clip(v, 0.05, self._speed_ceiling))

    def set_tau_ff(self, tau) -> bool:
        """设置主动出力（Nm/关节，任一关节超 ±20 时整体等比缩小，保持力方向）。
        位置指令照常，出力叠加在重力前馈之上；点动关闭、卸力、急停都会自动清零。
        用于贴着表面按压/拨动时主动出力。"""
        with self._lock:
            if not self._jog_enabled:
                return False
            tau = np.asarray(tau, dtype=float).reshape(-1)
            if tau.size != self.n:
                raise ValueError(f"需要 {self.n} 个关节力矩，收到 {tau.size}")
            if not np.all(np.isfinite(tau)):
                raise ValueError("力矩包含非法值")
            worst = float(np.max(np.abs(tau)))
            if worst > PUSH_TAU_LIMIT:
                tau = tau * (PUSH_TAU_LIMIT / worst)
            self._tau_push = tau
            return True

    def set_target(self, q_desired) -> bool:
        with self._lock:
            if not self._jog_enabled:
                return False
            q = np.asarray(q_desired, dtype=float).reshape(-1)
            if q.size != self.n:
                raise ValueError(f"需要 {self.n} 个关节目标，收到 {q.size}")
            if not np.all(np.isfinite(q)):
                raise ValueError("目标包含非法值")
            self._desired_q = self._clamp(q)
            return True

    def nudge(self, index: int, delta: float) -> bool:
        with self._lock:
            if not self._jog_enabled:
                return False
            if not (0 <= index < self.n):
                raise IndexError(f"关节下标 {index} 越界")
            q = self._desired_q.copy()
            q[index] += float(delta)
            self._desired_q = self._clamp(q)
            return True

    def status(self) -> dict:
        with self._lock:
            cmd = self._cmd_q.copy()
            desired = self._desired_q.copy()
            jog = self._jog_enabled
            floating = self._float
            weight = self._weight
            push = self._tau_push.copy()
        try:
            measured = self.read_measured().tolist()
        except Exception:
            measured = None
        return {
            "arm": self.arm,
            "engaged": self._engaged,
            "jog_enabled": jog,
            "float": floating,
            "weight": weight,
            "joint_names": self.joint_names,
            "measured_rad": measured,
            "cmd_rad": cmd.tolist(),
            "desired_rad": desired.tolist(),
            "limits_rad": self.limits.tolist(),
            "max_speed_rad_s": self.max_speed,
            "kp": self.kp,
            "kd": self.kd,
            "kp_wrist": self.kp_wrist,
            "kd_wrist": self.kd_wrist,
            "grav_alpha": self.grav_alpha,
            "payload_kg": self.payload_kg,
            "grav_in_float": self.grav_in_float,
            "use_imu_gravity": self.use_imu_gravity,
            "tau_grav_nm": np.asarray(self._tau_grav).tolist(),
            "tau_push_nm": push.tolist(),
        }
