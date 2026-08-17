"""Vendored robot configuration and URDF kinematics runtime."""

from .robot_config import load_robot_config
from .robot_model import RobotModel

__all__ = ["load_robot_config", "RobotModel"]
