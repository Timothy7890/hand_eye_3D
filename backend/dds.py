"""DDS ChannelFactory 全局初始化（进程内只能调一次，arm 和 pose 共用）。"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_initialized = False


def ensure_dds_initialized(network_interface: str | None = None) -> None:
    global _initialized
    with _lock:
        if _initialized:
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        if network_interface:
            ChannelFactoryInitialize(0, network_interface)
        else:
            ChannelFactoryInitialize(0)
        _initialized = True
