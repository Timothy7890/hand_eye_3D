"""RGB-D 相机封装：ZMQ 生产输入、Orbbec SDK 调试输入和 mock。

坐标系约定（与 video_tools 的彩色点云一致）：
深度对齐到彩色相机后，pick() 用彩色内参反投影，
因此返回的 P_camera 在【彩色相机坐标系】下（X 右、Y 下、Z 前，米）。
后续标定得到的 T_base^camera 也就是彩色相机的位姿。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np

DEPTH_HISTORY = 8  # 保留最近 N 帧对齐深度做时域中值滤波


class CameraBase:
    source = "base"

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def get_jpeg(self) -> bytes | None:
        raise NotImplementedError

    def pick(self, u: int, v: int, win: int = 5) -> dict:
        raise NotImplementedError

    def depth_snapshot(self) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        """返回 (多帧中值深度图 mm, 彩色内参 fx/fy/cx/cy)，无数据时 None。"""
        return None

    def info(self) -> dict:
        return {"source": self.source}


class MockCamera(CameraBase):
    """棋盘纹理 + 固定深度 1m，联调 UI 用。"""

    source = "mock"
    width, height = 1280, 720

    def get_jpeg(self) -> bytes | None:
        img = np.full((self.height, self.width, 3), 40, np.uint8)
        s = 80
        for y in range(0, self.height, s):
            for x in range(0, self.width, s):
                if (x // s + y // s) % 2 == 0:
                    img[y:y + s, x:x + s] = 70
        cv2.putText(img, f"MOCK {time.strftime('%H:%M:%S')}", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else None

    def pick(self, u: int, v: int, win: int = 5) -> dict:
        fx = fy = 900.0
        z = 1.0
        return {
            "ok": True,
            "p_camera": [(u - self.width / 2) * z / fx, (v - self.height / 2) * z / fy, z],
            "depth_mm": 1000.0, "valid_ratio": 1.0, "pixel": [u, v],
        }

    def depth_snapshot(self):
        # 合成场景：0.9m 处一面墙，中间凸出一个 0.6m 的方块，联调障碍扫描用
        depth = np.full((self.height, self.width), 900.0, np.float32)
        depth[200:500, 500:800] = 600.0
        return depth, (900.0, 900.0, self.width / 2, self.height / 2)

    def info(self) -> dict:
        return {"source": self.source, "width": self.width, "height": self.height,
                "serial": "MOCK", "name": "Mock Camera"}


class OrbbecRGBDCamera(CameraBase):
    """后台线程持续取 彩色帧 + 对齐到彩色的深度帧。

    serial=None 时用 SDK 找到的第一台设备。
    """

    source = "orbbec"

    def __init__(self, serial: str | None = None):
        import pyorbbecsdk as ob
        self._ob = ob
        self.serial = serial
        self.name = ""
        self.width = 0
        self.height = 0
        self.intrinsics = None  # (fx, fy, cx, cy) 彩色相机内参
        self._lock = threading.Lock()
        self._color_bgr = None
        self._depth_hist: deque[np.ndarray] = deque(maxlen=DEPTH_HISTORY)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    # ---- 生命周期 ----

    def start(self) -> None:
        ob = self._ob
        ctx = ob.Context()
        try:
            ctx.set_logger_level(ob.OBLogLevel.ERROR)
        except Exception:
            pass
        devices = ctx.query_devices()
        if devices.get_count() == 0:
            raise RuntimeError("SDK 未找到任何 Orbbec 设备")
        device = None
        serials = []
        for i in range(devices.get_count()):
            d = devices.get_device_by_index(i)
            sn = d.get_device_info().get_serial_number()
            serials.append(sn)
            if self.serial is None or sn == self.serial:
                device = d
                self.serial = sn
                break
        if device is None:
            raise RuntimeError(f"序列号 {self.serial} 不在设备列表中（已发现: {serials}）")
        self.name = device.get_device_info().get_name()

        self._pipeline = ob.Pipeline(device)
        config = ob.Config()
        color_profiles = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = self._pick_best_color_profile(ob, color_profiles)
        config.enable_stream(color_profile)
        depth_profiles = self._pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        try:
            depth_profile = depth_profiles.get_video_stream_profile(0, 0, ob.OBFormat.Y16, 0)
        except Exception:
            depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        self._pipeline.enable_frame_sync()

        self.width = color_profile.get_width()
        self.height = color_profile.get_height()
        intr = color_profile.get_intrinsic()
        self.intrinsics = (intr.fx, intr.fy, intr.cx, intr.cy)

        self._align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        self._config = config
        self._pipeline.start(config)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        # 看门狗：pipeline 偶尔会"启动成功但永远不出帧"（上个进程没释放干净），
        # 白屏难排查，这里等首帧，超时先重启管道一次，再不行直接报错退出。
        if not self._wait_first_frame(6.0):
            print("[camera] 启动后 6s 没有帧，重启 pipeline 重试…")
            try:
                self._pipeline.stop()
            except Exception:
                pass
            time.sleep(1.0)
            self._pipeline.start(config)
            if not self._wait_first_frame(6.0):
                self._stop_evt.set()
                raise RuntimeError(
                    "相机 pipeline 不出帧。通常是设备被别的进程占用或处于坏状态，"
                    "请关掉其他用相机的程序；仍不行可用 SDK reboot 相机或重插 USB。")

    def _wait_first_frame(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._color_bgr is not None and self._depth_hist:
                    return True
            time.sleep(0.2)
        return False

    @staticmethod
    def _pick_best_color_profile(ob, profiles):
        """遍历所有可解码的彩色档位，按 (像素数, 格式优先级, 帧率) 选最优。

        高分辨率彩色通常只有 MJPG 压缩格式（裸 RGB 受 USB 带宽限制），
        所以 RGB/MJPG/YUYV/NV12 都接受，_loop 里按格式解码。
        """
        fmt_pref = {ob.OBFormat.RGB: 3, ob.OBFormat.MJPG: 2,
                    ob.OBFormat.NV12: 1, ob.OBFormat.YUYV: 1}
        best = None
        best_key = (-1, -1, -1)
        available = []
        try:
            for i in range(profiles.get_count()):
                p = profiles.get_stream_profile_by_index(i)
                try:
                    vp = p.as_video_stream_profile()
                except Exception:
                    continue
                fmt = vp.get_format()
                fps = vp.get_fps()
                available.append(f"{vp.get_width()}x{vp.get_height()}@{fps} {fmt}")
                if fmt not in fmt_pref or fps > 30:
                    continue
                key = (vp.get_width() * vp.get_height(), fmt_pref[fmt], fps)
                if key > best_key:
                    best_key = key
                    best = vp
        except Exception as e:
            print(f"[camera] 枚举彩色档位失败: {e}")
            best = None
        print(f"[camera] 可用彩色档位: {available}")
        if best is not None:
            print(f"[camera] 选用: {best.get_width()}x{best.get_height()}"
                  f"@{best.get_fps()} {best.get_format()}")
            return best
        print("[camera] 未找到可解码彩色档位，回退默认档")
        try:
            return profiles.get_video_stream_profile(0, 0, ob.OBFormat.RGB, 0)
        except Exception:
            return profiles.get_default_video_stream_profile()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(3.0)
        try:
            self._pipeline.stop()
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                frames = self._pipeline.wait_for_frames(200)
                if frames is None:
                    continue
                frames = self._align.process(frames)
                if not frames:
                    continue
                frames = frames.as_frame_set()
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if color is None or depth is None:
                    continue

                bgr = self._decode_color(color)
                if bgr is None:
                    continue

                dw, dh = depth.get_width(), depth.get_height()
                scale = depth.get_depth_scale()
                depth_mm = np.frombuffer(depth.get_data(), np.uint16).reshape(dh, dw)
                depth_mm = depth_mm.astype(np.float32) * scale

                with self._lock:
                    self._color_bgr = bgr
                    self._depth_hist.append(depth_mm)
                    self.error = None
            except Exception as e:
                self.error = str(e)
                time.sleep(0.2)

    def _decode_color(self, color) -> np.ndarray | None:
        """按格式把彩色帧解码成 BGR。"""
        ob = self._ob
        fmt = color.get_format()
        w, h = color.get_width(), color.get_height()
        data = np.frombuffer(color.get_data(), np.uint8)
        if fmt == ob.OBFormat.RGB:
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt == ob.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == ob.OBFormat.YUYV:
            return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
        if fmt == ob.OBFormat.NV12:
            return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
        return None

    # ---- 数据接口 ----

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            bgr = None if self._color_bgr is None else self._color_bgr.copy()
        if bgr is None:
            return None
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else None

    def pick(self, u: int, v: int, win: int = 5) -> dict:
        """严格取像素 (u,v) 本身的深度（只做多帧时域中值，不做空间外扩）。

        以前用 5×5 空间窗口，点指尖尖端时窗口跨轮廓、被背景"多数票"带走
        （指尖本身常因黑色/细小/反光在深度图里是空洞，窗口里能投票的全是
        背景）。现在就用你点中的那一个像素：8 帧里有效帧不足直接报错，
        绝不拿邻近/背景值顶替。返回彩色相机坐标系下坐标（米）。
        """
        if self.intrinsics is None:
            return {"ok": False, "error": "相机未就绪"}
        with self._lock:
            hist = list(self._depth_hist)
        if not hist:
            return {"ok": False, "error": "还没有深度帧"}

        h, w = hist[0].shape
        if not (0 <= u < w and 0 <= v < h):
            return {"ok": False, "error": f"像素越界 ({u},{v})，深度图 {w}x{h}"}

        vals = np.array([d[v, u] for d in hist], dtype=float)
        valid = vals[(vals > 60) & (vals < 15000)]
        if valid.size < max(3, len(vals) // 2):
            return {"ok": False,
                    "error": f"该像素没有稳定深度（{valid.size}/{len(vals)} 帧有效）。"
                             "细小/深色/反光表面双目常测不到——可在指尖贴一小块"
                             "哑光贴纸，或点指腹等有深度的位置"}
        spread = float(np.max(valid) - np.min(valid))
        z_mm = float(np.median(valid))
        if spread > 80.0:
            return {"ok": False,
                    "error": f"该像素深度在多帧间跳动 {spread:.0f}mm（边缘闪烁），"
                             "稍微挪一点再点"}
        z = z_mm / 1000.0
        fx, fy, cx, cy = self.intrinsics
        p = [(u - cx) * z / fx, (v - cy) * z / fy, z]
        return {
            "ok": True,
            "p_camera": p,
            "depth_mm": z_mm,
            "valid_ratio": float(valid.size / len(vals)),
            "pixel": [u, v],
        }

    def depth_snapshot(self):
        if self.intrinsics is None:
            return None
        with self._lock:
            hist = list(self._depth_hist)
        if not hist:
            return None
        depth = np.median(np.stack(hist), axis=0).astype(np.float32)
        return depth, self.intrinsics

    def info(self) -> dict:
        fx, fy, cx, cy = self.intrinsics or (0, 0, 0, 0)
        return {
            "source": self.source, "serial": self.serial, "name": self.name,
            "width": self.width, "height": self.height,
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "error": self.error,
        }


def make_camera(
    source: str,
    serial: str | None = None,
    *,
    host: str = "127.0.0.1",
    calibration_path: str | Path | None = None,
    camera_name: str = "head_rgbd_camera",
    request_port: int = 60000,
    stream_port: int | None = None,
    stale_after_s: float = 2.0,
    startup_timeout_s: float = 15.0,
) -> CameraBase:
    if source == "mock":
        return MockCamera()
    if source == "zmq":
        if calibration_path is None:
            raise ValueError("ZMQ 相机必须提供 calibration_path")
        ik_replay_root = Path("/home/robot/yx/project/IK_replay")
        root = str(ik_replay_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from camera_sources.zmq_rgbd import ZmqRGBDCamera

        return ZmqRGBDCamera(
            host=host,
            calibration_path=calibration_path,
            camera_name=camera_name,
            request_port=request_port,
            stream_port=stream_port,
            stale_after_s=stale_after_s,
            startup_timeout_s=startup_timeout_s,
        )
    if source == "orbbec":
        return OrbbecRGBDCamera(serial=serial)
    raise ValueError(f"未知 camera source: {source!r}（可选 zmq/orbbec/mock）")
