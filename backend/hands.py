"""灵巧手模型目录与零位 FK：手安装标定（T_wrist2hand）的模型侧。

config/hands.yaml 登记每个手型号的 URDF 与 6 个驱动关节。灵巧手 6 关节
归零后是几何完全已知的刚体：本模块给出任意关节角（默认全零）下每个
link 在手基座系（URDF 根 link）的变换、tip 特征点坐标，以及供前端
Three.js 直接摆放 mesh 的可视化数据——前端不需要解析 URDF。

URDF 里的 mimic 联动关节（拇指/各指远端）在 FK 前自动展开，因此
非零关节角也能得到正确姿态；本期标定流程只用零位。
"""

from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .paths import PROJECT_ROOT
from .robotics.robot_config import ChainConfig, RobotConfig, project_relative
from .robotics.robot_model import RobotModel
from .robotics.types import Pose

HANDS_CONFIG_PATH = PROJECT_ROOT / "config" / "hands.yaml"

# tip link 名称到中文标签的手指映射（因时 R_/L_ 前缀与强脑 right_/left_ 前缀通用）
_FINGER_LABELS = {
    "thumb": "拇指",
    "index": "食指",
    "middle": "中指",
    "ring": "无名指",
    "pinky": "小指",
}


class HandCatalogError(ValueError):
    """hands.yaml 配置或手 URDF 不合法。"""


@dataclass(frozen=True)
class HandSpec:
    hand_id: str
    label: str
    vendor: str
    side: str
    urdf_path: Path
    actuated_joints: list[str]
    extra_feature_points: list[dict[str, Any]] = field(default_factory=list)


def load_hand_catalog(path: str | Path = HANDS_CONFIG_PATH) -> dict[str, HandSpec]:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise HandCatalogError(f"无法读取手型号目录 {path}: {exc}") from exc
    hands_raw = raw.get("hands")
    if not isinstance(hands_raw, dict) or not hands_raw:
        raise HandCatalogError(f"{path} 必须包含非空的 hands 映射")

    catalog: dict[str, HandSpec] = {}
    for hand_id, entry in hands_raw.items():
        hand_id = str(hand_id).strip()
        if not hand_id or not isinstance(entry, dict):
            raise HandCatalogError(f"手型号 {hand_id!r} 配置必须是映射")
        side = str(entry.get("side", "")).strip()
        if side not in {"left", "right"}:
            raise HandCatalogError(f"手型号 {hand_id} 的 side 必须是 left 或 right")
        urdf_path = Path(str(entry.get("urdf_path", "")))
        if not urdf_path.is_absolute():
            urdf_path = PROJECT_ROOT / urdf_path
        if not urdf_path.is_file():
            raise HandCatalogError(f"手型号 {hand_id} 的 URDF 不存在: {urdf_path}")
        joints = [str(j) for j in entry.get("actuated_joints") or []]
        if len(joints) != len(set(joints)) or not joints:
            raise HandCatalogError(f"手型号 {hand_id} 的 actuated_joints 必须非空且不重复")
        extra = entry.get("extra_feature_points") or []
        if not isinstance(extra, list):
            raise HandCatalogError(f"手型号 {hand_id} 的 extra_feature_points 必须是数组")
        catalog[hand_id] = HandSpec(
            hand_id=hand_id,
            label=str(entry.get("label") or hand_id),
            vendor=str(entry.get("vendor") or ""),
            side=side,
            urdf_path=urdf_path.resolve(),
            actuated_joints=joints,
            extra_feature_points=[dict(item) for item in extra],
        )
    return catalog


def _parse_root_and_mimics(
    urdf_path: Path,
) -> tuple[str, dict[str, tuple[str, float, float]]]:
    """轻量预扫 URDF：找根 link 与 mimic 关系（RobotModel 不解析 mimic）。"""
    root_el = ET.parse(urdf_path).getroot()
    links = [el.attrib["name"] for el in root_el.findall("link")]
    children = set()
    mimics: dict[str, tuple[str, float, float]] = {}
    for joint_el in root_el.findall("joint"):
        child_el = joint_el.find("child")
        if child_el is not None:
            children.add(child_el.attrib["link"])
        mimic_el = joint_el.find("mimic")
        if mimic_el is not None:
            mimics[joint_el.attrib["name"]] = (
                mimic_el.attrib["joint"],
                float(mimic_el.attrib.get("multiplier", 1.0)),
                float(mimic_el.attrib.get("offset", 0.0)),
            )
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise HandCatalogError(f"手 URDF {urdf_path} 应有唯一根 link，实际: {roots}")
    return roots[0], mimics


class HandModel:
    """单个手型号的 URDF 模型：零位/给定关节角下的 link 变换与特征点。"""

    def __init__(self, spec: HandSpec):
        self.spec = spec
        self.base_link, self.mimics = _parse_root_and_mimics(spec.urdf_path)
        config = RobotConfig(
            name=spec.hand_id,
            display_name=spec.label,
            urdf_path=spec.urdf_path,
            mesh_root=spec.urdf_path.parent,
            preview_links=[],
            chains={
                "hand": ChainConfig(
                    name="hand",
                    display_name=spec.label,
                    subtitle=spec.hand_id,
                    panel_side=spec.side,
                    base_link=self.base_link,
                    end_link=self.base_link,
                    joints=list(spec.actuated_joints),
                )
            },
            tcp_offsets={"hand": Pose(xyz=[0.0, 0.0, 0.0])},
            initial_joints={"hand": {name: 0.0 for name in spec.actuated_joints}},
        )
        self.model = RobotModel(config)
        for name, (source, _, _) in self.mimics.items():
            if source not in self.model.joints:
                raise HandCatalogError(
                    f"{spec.hand_id} 的 mimic 关节 {name} 引用了不存在的 {source}"
                )
        overlap = set(spec.actuated_joints) & set(self.mimics)
        if overlap:
            raise HandCatalogError(
                f"{spec.hand_id} 的 actuated_joints 不应包含 mimic 关节: {sorted(overlap)}"
            )

    # ---------- 关节 ----------

    def coerce_joints(self, joints: Any) -> dict[str, float]:
        """接受 None / 6 元数组 / {关节名: 角度}，返回驱动关节完整字典（弧度）。"""
        names = self.spec.actuated_joints
        if joints is None:
            return {name: 0.0 for name in names}
        if isinstance(joints, dict):
            unknown = set(joints) - set(names)
            if unknown:
                raise ValueError(f"未知手关节: {sorted(unknown)}")
            values = {name: float(joints.get(name, 0.0)) for name in names}
        else:
            seq = [float(v) for v in joints]
            if len(seq) != len(names):
                raise ValueError(f"手关节应为 {len(names)} 个值，收到 {len(seq)} 个")
            values = dict(zip(names, seq))
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("手关节角包含非法值")
        return values

    def expand_joints(self, actuated: dict[str, float]) -> dict[str, float]:
        """驱动关节 → 含 mimic 联动的完整关节字典。"""
        values = dict(actuated)
        # mimic 链最深两级（拇指 distal→proximal_pitch），迭代到收敛即可
        for _ in range(4):
            changed = False
            for name, (source, multiplier, offset) in self.mimics.items():
                if source in values:
                    new = values[source] * multiplier + offset
                    if values.get(name) != new:
                        values[name] = new
                        changed = True
            if not changed:
                break
        return values

    # ---------- FK 与特征点 ----------

    def link_transforms(self, joints: Any = None) -> dict[str, np.ndarray]:
        """所有 link 在手基座系（URDF 根 link）的 4x4 变换。"""
        values = self.expand_joints(self.coerce_joints(joints))
        return self.model.forward_kinematics(values)

    def tip_links(self) -> list[str]:
        """叶子且名字含 tip 的 link，按 URDF 出现顺序。"""
        return [
            name
            for name in self.model.links
            if "tip" in name.lower() and name not in self.model.parent_to_joints
        ]

    @staticmethod
    def _finger_label(link_name: str) -> str:
        lowered = link_name.lower()
        for key, label in _FINGER_LABELS.items():
            if key in lowered:
                return f"{label}指尖"
        return link_name

    def feature_points(self, joints: Any = None) -> list[dict[str, Any]]:
        """预置特征点：各 tip link 原点 + 配置里的自定义点（手基座系，米）。"""
        transforms = self.link_transforms(joints)
        points: list[dict[str, Any]] = []
        for link in self.tip_links():
            points.append(
                {
                    "id": f"tip:{link}",
                    "label": self._finger_label(link),
                    "link": link,
                    "p_local": [0.0, 0.0, 0.0],
                    "p_hand": [float(v) for v in transforms[link][:3, 3]],
                    "source": "tip_link",
                }
            )
        for position, item in enumerate(self.spec.extra_feature_points):
            link = str(item.get("link") or self.base_link)
            if link not in transforms:
                raise HandCatalogError(
                    f"{self.spec.hand_id} 的 extra_feature_points[{position}] "
                    f"引用了不存在的 link {link!r}"
                )
            p_local = np.asarray(item.get("xyz", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            p_hand = transforms[link] @ np.array([*p_local, 1.0])
            points.append(
                {
                    "id": str(item.get("id") or f"extra-{position}"),
                    "label": str(item.get("label") or item.get("id") or f"自定义点{position}"),
                    "link": link,
                    "p_local": p_local.tolist(),
                    "p_hand": [float(v) for v in p_hand[:3]],
                    "source": "config",
                }
            )
        return points

    # ---------- 前端可视化数据 ----------

    def _mesh_url(self, filename: str) -> str | None:
        candidate = (self.spec.urdf_path.parent / filename).resolve()
        if not candidate.is_file():
            return None
        return f"/{project_relative(candidate)}"

    def visuals_payload(self, joints: Any = None) -> list[dict[str, Any]]:
        """每个 link 的 4x4 变换与 mesh 摆放参数，前端直接套用无需解析 URDF。"""
        transforms = self.link_transforms(joints)
        payload: list[dict[str, Any]] = []
        for name, link in self.model.links.items():
            visuals = []
            for visual in link.visuals:
                if not visual.filename:
                    continue
                url = self._mesh_url(visual.filename)
                if url is None:
                    continue
                visuals.append(
                    {
                        "mesh_url": url,
                        "xyz": visual.xyz.tolist(),
                        "rpy": visual.rpy.tolist(),
                        "scale": visual.scale.tolist(),
                        "color": visual.color,
                    }
                )
            if visuals and name in transforms:
                payload.append(
                    {
                        "link": name,
                        "T_hand_link": transforms[name].tolist(),
                        "visuals": visuals,
                    }
                )
        return payload

    def metadata(self, joints: Any = None) -> dict[str, Any]:
        actuated = self.coerce_joints(joints)
        return {
            "hand_id": self.spec.hand_id,
            "label": self.spec.label,
            "vendor": self.spec.vendor,
            "side": self.spec.side,
            "base_link": self.base_link,
            "urdf_path": project_relative(self.spec.urdf_path),
            "actuated_joints": list(self.spec.actuated_joints),
            "joint_values": actuated,
            "mimic_joints": {
                name: {"source": source, "multiplier": multiplier, "offset": offset}
                for name, (source, multiplier, offset) in self.mimics.items()
            },
            "feature_points": self.feature_points(joints),
            "links": self.visuals_payload(joints),
        }


_catalog_lock = threading.Lock()
_catalog_cache: dict[str, HandSpec] | None = None
_model_cache: dict[str, HandModel] = {}


def hand_catalog() -> dict[str, HandSpec]:
    global _catalog_cache
    with _catalog_lock:
        if _catalog_cache is None:
            _catalog_cache = load_hand_catalog()
        return _catalog_cache


def get_hand_model(hand_id: str) -> HandModel:
    catalog = hand_catalog()
    if hand_id not in catalog:
        raise KeyError(f"未登记的手型号 {hand_id!r}，可用: {sorted(catalog)}")
    with _catalog_lock:
        model = _model_cache.get(hand_id)
        if model is None:
            model = HandModel(catalog[hand_id])
            _model_cache[hand_id] = model
        return model


def reset_catalog_cache() -> None:
    """测试用：清空目录与模型缓存。"""
    global _catalog_cache
    with _catalog_lock:
        _catalog_cache = None
        _model_cache.clear()
