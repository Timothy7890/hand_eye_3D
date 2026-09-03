#!/usr/bin/env python3
"""3D-3D 手眼标定（眼在手外，联合估计指尖偏移）采集/解算服务入口。

示例
----
本地联调（无相机无机器人）:
    python run_server.py --camera-source mock --pose-source mock

H2 真机（ZMQ RGB-D + DDS 只读 rt/lowstate + 项目内 H2 FK，右臂）:
    python run_server.py --camera-source zmq --camera-host 127.0.0.1 \
        --pose-source h2 --network-interface eth0

手腕位姿手填 / sidecar:
    python run_server.py --camera-source zmq                          # manual 手填
    python run_server.py --pose-source http --pose-url http://127.0.0.1:18091/pose
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.paths import DEFAULT_RGBD_CALIB_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Hand-Eye 3D point-pair calibration server")
    parser.add_argument(
        "--save-path",
        default=str(PROJECT_ROOT / "handeye3d_data"),
        help="采样与解算结果目录（默认保存在本项目 handeye3d_data 下）",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8132)
    parser.add_argument("--no-timestamp-dir", action="store_true",
                        help="直接存到 --save-path，不建时间戳子目录")

    parser.add_argument(
        "--camera-source",
        choices=["zmq", "orbbec", "mock"],
        default="zmq",
        help="相机来源；默认只读 teleimager ZMQ，orbbec 仅用于显式 SDK 调试",
    )
    parser.add_argument("--camera-serial", default=None,
                        help="仅 --camera-source orbbec 使用的相机序列号")
    parser.add_argument("--camera-host", default="127.0.0.1",
                        help="teleimager ZMQ 服务主机")
    parser.add_argument("--camera-request-port", type=int, default=60000,
                        help="teleimager 配置请求端口")
    parser.add_argument("--camera-port", type=int, default=None,
                        help="RGB-D ZMQ 端口；默认从配置服务查询")
    parser.add_argument("--camera-name", default="head_rgbd_camera",
                        help="teleimager RGB-D stream 名称")
    parser.add_argument("--camera-stale-after", type=float, default=2.0,
                        help="超过多少秒未收到帧视为过期")
    parser.add_argument("--camera-startup-timeout", type=float, default=15.0,
                        help="启动等待首个合法 RGB-D 帧的秒数")
    parser.add_argument("--teleop-task-dir", default=None,
                        help="离线 robot-style 任务目录；设置后不占用 Orbbec 相机")
    parser.add_argument(
        "--record-task-dir",
        default=str(PROJECT_ROOT / "teleop_data" / "biaoding"),
        help="Orbbec SDK 实时模式“拍摄当前姿态”的离线 episode 保存目录",
    )
    parser.add_argument(
        "--rgbd-calib",
        default=str(DEFAULT_RGBD_CALIB_PATH),
        help="ZMQ 实时或离线原始深度到 RGB 的生产标定 JSON",
    )
    parser.add_argument(
        "--mount-calib",
        default=None,
        help="手安装标定固定使用的相机外参 JSON；未指定时使用当前会话结果",
    )
    parser.add_argument(
        "--mount-profile-dir",
        default=str(PROJECT_ROOT / "handeye3d_data" / "mount_model_profiles"),
        help="手安装模型点命名方案目录（默认固定在项目 handeye3d_data 下）",
    )

    parser.add_argument("--hand-service-url", default="https://127.0.0.1:18089",
                        help="18089 灵巧手控制服务地址（手安装标定「保持零位」用）")
    parser.add_argument("--capability-url", default="http://127.0.0.1:18000",
                        help="18000 能力中心地址（启动拜访，必须可达；"
                             "start.sh 会自动拉起）")
    parser.add_argument("--pose-source", choices=["manual", "http", "h2", "mock"],
                        default="manual", help="手腕位姿来源（默认 manual 手填）")
    parser.add_argument("--pose-url", help="http 模式的 JSON 端点，返回 {\"T\": 4x4} 或 {\"xyz\",\"rpy\"}")
    parser.add_argument("--network-interface", help="h2 模式的 DDS 网卡，如 eth0")
    parser.add_argument("--arm", choices=["right", "left"], default="right",
                        help="h2 模式用哪条手臂（默认 right）")
    parser.add_argument("--base-link", default=None,
                        help="h2 模式的基座 link（默认取 h2.yaml 的 torso_link）")

    parser.add_argument("--arm-control", action="store_true",
                        help="启用手臂点动/卸力控制（发布 rt/arm_sdk，真机会动！"
                             "确保没有其他程序在控制手臂）")
    parser.add_argument("--arm-max-speed", type=float, default=0.2,
                        help="点动最大关节速度 rad/s（默认 0.2）")
    parser.add_argument("--arm-grav-ff", type=float, default=1.0,
                        help="重力前馈系数（默认 1.0 = 完整补偿）。给 0 关闭")
    parser.add_argument("--arm-payload-kg", type=float, default=0.0,
                        help="URDF 之外的额外手部负载 kg（换装更重的灵巧手时填差值）")
    parser.add_argument("--arm-grav-in-float", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="卸力拖动时也给重力前馈（按实测角实时算）：手臂近似失重，"
                             "摆位省力、松手不下坠（默认开）。用 --no-arm-grav-in-float 关闭")
    args = parser.parse_args()

    from backend import app as app_module
    from backend.camera import make_camera
    from backend.capability import (
        CapabilityUnavailable,
        describe_active,
        fetch_capability_snapshot,
    )
    from backend.offline import OfflineEpisodeBackend
    from backend.robot import make_pose_provider

    # 启动拜访 18000（硬依赖，先于相机/机器人）：拿不到注册表快照就不启动
    try:
        capability_snapshot = fetch_capability_snapshot(args.capability_url)
    except CapabilityUnavailable as exc:
        print(f"[handeye3d] 启动拜访 18000 失败：{exc}")
        return 1
    print(f"[handeye3d] 18000 {describe_active(capability_snapshot)}")

    session_dir = Path(args.save_path)
    if not args.no_timestamp_dir:
        session_dir = session_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

    offline_backend = None
    episode_backend = None
    live_record_task_dir = None
    if args.teleop_task_dir:
        offline_backend = OfflineEpisodeBackend(
            args.teleop_task_dir, args.rgbd_calib, arm=args.arm
        )
        camera = make_camera("mock")
        print("[handeye3d] mode = offline（不打开、不占用 Orbbec 相机）")
        print(f"[handeye3d] teleop_task_dir = {offline_backend.task_dir}")
        print(f"[handeye3d] rgbd_calib = {offline_backend.rgbd_calib_path}")
    else:
        live_record_task_dir = Path(args.record_task_dir).expanduser().resolve()
        live_record_task_dir.mkdir(parents=True, exist_ok=True)
        episode_backend = OfflineEpisodeBackend(
            live_record_task_dir, args.rgbd_calib, arm=args.arm
        )
        camera = make_camera(
            args.camera_source,
            serial=args.camera_serial,
            host=args.camera_host,
            calibration_path=args.rgbd_calib,
            camera_name=args.camera_name,
            request_port=args.camera_request_port,
            stream_port=args.camera_port,
            stale_after_s=args.camera_stale_after,
            startup_timeout_s=args.camera_startup_timeout,
        )
        if args.camera_source == "zmq":
            print(
                f"[handeye3d] camera = zmq "
                f"({args.camera_host}/{args.camera_name}, calib={args.rgbd_calib})"
            )
        else:
            print(f"[handeye3d] camera = {args.camera_source} "
                  f"(serial={args.camera_serial or 'auto'})")
    camera.start()
    print(f"[handeye3d] camera info: {camera.info()}")

    arm_factory = None
    if args.arm_control and offline_backend is None:
        from backend.arm import H2ArmController

        def arm_factory():
            return H2ArmController(
                arm=args.arm, network_interface=args.network_interface,
                max_speed_rad_s=args.arm_max_speed,
                grav_alpha=args.arm_grav_ff, payload_kg=args.arm_payload_kg,
                grav_in_float=args.arm_grav_in_float,
            )

        print("[handeye3d] 手臂控制可用：在网页里点「获取控制」后才开始发布 rt/arm_sdk。")
        print("[handeye3d] !!! 获取控制前请确认没有其他程序（遥操作等）在控制手臂。")
    elif args.arm_control:
        print("[handeye3d] 离线模式忽略 --arm-control，不连接或控制机器人。")

    # 位姿读取自建只读订阅（绝不发指令），与是否接管手臂无关
    pose_source = "manual" if offline_backend is not None else args.pose_source
    pose_provider = make_pose_provider(
        pose_source, http_url=args.pose_url,
        network_interface=args.network_interface,
        arm=args.arm, base_link=args.base_link,
    )
    print(f"[handeye3d] pose_source = {pose_provider.source} (auto={pose_provider.available}, "
          f"base={pose_provider.base_link}, wrist={pose_provider.wrist_link})")

    app_module.camera = camera
    app_module.pose_provider = pose_provider
    app_module.arm_side = args.arm
    app_module.arm_factory = arm_factory
    app_module.save_path = session_dir
    app_module.offline_backend = offline_backend
    app_module.episode_backend = episode_backend
    app_module.teleop_task_dir = (
        offline_backend.task_dir if offline_backend is not None else None
    )
    app_module.record_task_dir = (
        None
        if offline_backend is not None
        else live_record_task_dir
    )
    app_module.rgbd_calib_path = Path(args.rgbd_calib).expanduser().resolve()
    app_module.mount_calib_path = (
        Path(args.mount_calib).expanduser().resolve()
        if args.mount_calib
        else None
    )
    app_module.mount_profile_dir = (
        Path(args.mount_profile_dir).expanduser().resolve()
    )
    app_module.hand_service_url = args.hand_service_url
    app_module.capability_url = args.capability_url.rstrip("/")
    app_module.capability_snapshot = capability_snapshot
    app_module.init_state()

    print(f"[handeye3d] save_path = {session_dir}")
    if app_module.record_task_dir is not None:
        print(f"[handeye3d] record_task_dir = {app_module.record_task_dir}")
    if app_module.mount_calib_path is not None:
        print(f"[handeye3d] mount_calib = {app_module.mount_calib_path}")
    print(f"[handeye3d] mount_profile_dir = {app_module.mount_profile_dir}")
    print(f"[handeye3d] serving on http://{args.host}:{args.port}")

    import uvicorn
    try:
        uvicorn.run(app_module.app, host=args.host, port=args.port)
    finally:
        if app_module.hand_hold.status().get("running"):
            print("[handeye3d] 停止灵巧手保持零位 ...")
            app_module.hand_hold.stop()
        if app_module.arm_controller is not None:
            print("[handeye3d] 手臂权重渐出、交还本体控制器（请扶住手臂）...")
            app_module.arm_controller.shutdown()
        camera.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
