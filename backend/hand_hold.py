"""18089 灵巧手保持零位：手安装标定采样期间周期性下发全零位置。

手安装标定要求灵巧手 6 关节零位（模型点按零位 FK 提供），但手指电机不加
持时可被外力扳动。开启保持后，后端周期向 18089 hand_web 服务发送全零
positions（0 = 张开 = URDF 零位，见 hand_web poses.json），让手主动回到并
停在零位。18089 把所有 HTTP 指令统一视作 manual 源，与其视觉控制等占用
互斥：被占用时它返回 409，本模块只透传错误、绝不抢占。
"""

from __future__ import annotations

import json
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

HOLD_INTERVAL_S = 0.3
HOLD_DURATION_MS = 300
HOLD_POSITIONS = (0.0,) * 6
HOLD_SIDES = ("left", "right")
# hands.yaml 的 vendor → 18089 设备目录 id
VENDOR_DEVICE_IDS = {"brainco": "brainco_revo2", "inspire": "inspire_dfx"}


class HandHoldError(RuntimeError):
    """18089 交互失败（不可达、被其他控制源占用、参数被拒）。"""


def _request_json(
    url: str, payload: dict | None = None, timeout: float = 3.0
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    context = (
        ssl._create_unverified_context()  # 18089 使用本机自签名证书
        if url.lower().startswith("https://")
        else None
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=context
        ) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8", errors="replace"))
            message = str(detail.get("error") or f"HTTP {exc.code}")
        except (ValueError, OSError):
            message = f"HTTP {exc.code}"
        raise HandHoldError(f"18089: {message}") from exc
    except OSError as exc:
        raise HandHoldError(f"18089 不可达: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HandHoldError(f"18089 返回非 JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise HandHoldError("18089 返回格式不是 JSON object")
    if body.get("ok") is False:
        raise HandHoldError(f"18089: {body.get('error') or '未知错误'}")
    return body


class HandHoldController:
    """单实例保持线程：start 幂等（同设备同侧），stop 后线程退出。

    连接参数（通信方式、端口等）沿用 18089 config.json 里各设备的默认值，
    这里只传 device_id。命令循环里的失败不中断保持（可能是瞬时占用），
    最近一次错误通过 status() 暴露给页面。
    """

    def __init__(self, url_getter: Callable[[], str]) -> None:
        self._url_getter = url_getter
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state: dict[str, Any] = {"running": False}

    def _url(self, path: str) -> str:
        return str(self._url_getter()).rstrip("/") + path

    def start(self, device_id: str, side: str) -> dict[str, Any]:
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id 必须是非空字符串")
        if side not in HOLD_SIDES:
            raise ValueError(f"side 必须是 left/right，收到 {side!r}")
        device_id = device_id.strip()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if (
                    self._state.get("device_id") == device_id
                    and self._state.get("side") == side
                ):
                    return self._snapshot_locked()
                raise HandHoldError(
                    f"正在保持 {self._state.get('device_id')}/{self._state.get('side')}，"
                    "请先停止再切换设备或侧"
                )
            # 已连接同设备时 18089 会复用现有通道
            _request_json(self._url("/api/connect"), {"device_id": device_id})
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._state = {
                "running": True,
                "device_id": device_id,
                "side": side,
                "started_at": time.time(),
                "sent_count": 0,
                "error_count": 0,
                "last_error": None,
                "last_ok_at": None,
                "interval_s": HOLD_INTERVAL_S,
            }
            self._thread = threading.Thread(
                target=self._run,
                args=(side, stop_event),
                name="hand-hold-18089",
                daemon=True,
            )
            self._thread.start()
            return self._snapshot_locked()

    def _run(self, side: str, stop_event: threading.Event) -> None:
        payload = {
            "side": side,
            "positions": list(HOLD_POSITIONS),
            "duration_ms": HOLD_DURATION_MS,
            "continuous": True,
        }
        while not stop_event.is_set():
            try:
                _request_json(self._url("/api/command"), payload)
                with self._lock:
                    if not stop_event.is_set():
                        self._state["sent_count"] += 1
                        self._state["last_ok_at"] = time.time()
                        self._state["last_error"] = None
            except HandHoldError as exc:
                with self._lock:
                    if not stop_event.is_set():
                        self._state["error_count"] += 1
                        self._state["last_error"] = str(exc)
            stop_event.wait(HOLD_INTERVAL_S)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
            self._thread = None
            self._state = {**self._state, "running": False}
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            return self._snapshot_locked()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        state = dict(self._state)
        state["running"] = bool(
            state.get("running") and self._thread is not None and self._thread.is_alive()
        )
        return state
