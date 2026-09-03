"""启动拜访 18000 能力中心：HTTP 拉取能力注册表快照（硬依赖）。

约定（与 IK_replay 各服务一致）：
· run_server.py 启动时调用 fetch_capability_snapshot()，拿不到就退出——
  本服务必须在 18000 可达时才启动。
· 自动拉起 18000 是启动脚本（start.sh → IK_replay/capability.sh）的职责；
  进程内只确认可达，不做拉起。
· 快照语义：进程生命周期内以启动时刻的快照为准，改 18000 配置后重启生效。
  确认安装样本时的臂/手组合校验仍走 app.get_capability_hint() 每次现查，
  与快照互不代替。
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

DEFAULT_CAPABILITY_URL = "http://127.0.0.1:18000"
REGISTRY_ENDPOINT = "/api/capability/registry"
ARM_LABELS = {"right_arm": "右臂", "left_arm": "左臂"}


class CapabilityUnavailable(RuntimeError):
    """18000 不可达或返回异常。"""


def fetch_capability_snapshot(
    base_url: str | None = None,
    *,
    timeout_s: float = 3.0,
    attempts: int = 3,
) -> dict[str, Any]:
    """GET /api/capability/registry，返回完整 payload。

    只做结构性检查（ok=true 且 registry 是 object）；内容合法性由 18000
    服务端保存时校验。网络失败重试 attempts 次。
    """
    base = (base_url or DEFAULT_CAPABILITY_URL).rstrip("/")
    url = base + REGISTRY_ENDPOINT
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        if attempt:
            time.sleep(0.5)
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(
                    response.read().decode("utf-8", errors="replace"))
        except (OSError, ValueError) as exc:
            last_error = exc
            continue
        if (not isinstance(payload, dict) or payload.get("ok") is not True
                or not isinstance(payload.get("registry"), dict)):
            raise CapabilityUnavailable(
                f"18000 返回异常（{url}）: {str(payload)[:200]}")
        return payload
    raise CapabilityUnavailable(
        f"访问不到 18000 能力中心（{url}）: {last_error}。"
        "请先运行 IK_replay/capability.sh（./start.sh 会自动拉起）。")


def describe_active(payload: dict[str, Any]) -> str:
    """启动日志一行描述：激活组合 + 该组合标定登记状态。"""
    registry = payload.get("registry") or {}
    active = registry.get("active")
    if not active:
        return "18000 未设置激活组合（臂+手型号）"
    hand = next(
        (item for item in registry.get("hands") or []
         if item.get("id") == active.get("hand_id")),
        {},
    )
    status = next(
        (item.get("status") for item in payload.get("calibrations") or []
         if item.get("arm") == active.get("arm")
         and item.get("hand_id") == active.get("hand_id")),
        "missing",
    )
    return (f"激活组合: {ARM_LABELS.get(active.get('arm'), active.get('arm'))}"
            f" + {hand.get('name') or active.get('hand_id')}（标定 {status}）")
