#!/usr/bin/env bash
# 一键启动 hand_eye_3D 前后端。
#
#   ./start.sh              # 默认：H2 真机 + ZMQ RGB-D + 手臂控制可用（网页点「获取控制」后才接管真机）
#   ./start.sh --no-arm     # 不启用手臂控制（只读 rt/lowstate，绝不发布，可与其他控制程序并存）
#   ./start.sh <其他参数>    # 其余参数原样传给 run_server.py（如 --arm-grav-in-float）
#
# Ctrl+C 退出：后端会先把手臂权重渐出、交还本体控制器（此时请扶住手臂），再退出。
set -u

cd "$(dirname "$0")"

PY="${PYTHON:-$HOME/miniconda3/envs/fastapi/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python)"
fi
IFACE="${NETWORK_INTERFACE:-enp86s0}"
CAMERA_HOST="${CAMERA_HOST:-127.0.0.1}"
RGBD_CALIB="${RGBD_CALIB:-$PWD/config/camera/orbbec_rgbd_calibration.json}"

ARM_ARGS=(--arm-control)
EXTRA=()
OFFLINE=0
for a in "$@"; do
  if [ "$a" = "--no-arm" ]; then
    ARM_ARGS=()
  else
    EXTRA+=("$a")
    case "$a" in
      --teleop-task-dir|--teleop-task-dir=*) OFFLINE=1 ;;
    esac
  fi
done
if [ "$OFFLINE" -eq 1 ]; then
  ARM_ARGS=()
  echo "离线遥操作数据模式：不打开相机、不连接或控制机器人。"
fi

if [ ${#ARM_ARGS[@]} -gt 0 ]; then
  echo "手臂控制可用：网页里点「获取控制」后才发布 rt/arm_sdk（获取前请确认没有其他控制程序）。"
  echo "（完全不需要动手臂就用 ./start.sh --no-arm）"
fi

BACK_PID=""
FRONT_IMAGE_PID=""
FRONT_CLOUD_PID=""
cleanup() {
  echo ""
  echo "[start] 正在退出（手臂权重渐出，请扶住手臂）..."
  [ -n "$FRONT_IMAGE_PID" ] && kill "$FRONT_IMAGE_PID" 2>/dev/null
  [ -n "$FRONT_CLOUD_PID" ] && kill "$FRONT_CLOUD_PID" 2>/dev/null
  [ -n "$BACK_PID" ] && kill -INT "$BACK_PID" 2>/dev/null
  [ -n "$BACK_PID" ] && wait "$BACK_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

"$PY" run_server.py \
  --camera-source zmq --camera-host "$CAMERA_HOST" --rgbd-calib "$RGBD_CALIB" \
  --pose-source h2 --network-interface "$IFACE" \
  "${ARM_ARGS[@]}" "${EXTRA[@]}" &
BACK_PID=$!

sleep 1
if ! kill -0 "$BACK_PID" 2>/dev/null; then
  echo "[start] 后端启动失败，见上方报错" >&2
  exit 1
fi

(cd frontend && npm run dev --silent) &
FRONT_IMAGE_PID=$!
(cd frontend && npm run dev:pointcloud --silent) &
FRONT_CLOUD_PID=$!

if hostname -I >/dev/null 2>&1; then
  IP="$(hostname -I | awk '{print $1}')"
else
  IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"
fi
echo "[start] 后端 http://${IP}:8132"
echo "[start] 图像选点 http://${IP}:7012   点云选点 http://${IP}:7013   （Ctrl+C 一起退出）"

wait "$FRONT_IMAGE_PID" "$FRONT_CLOUD_PID" "$BACK_PID"
cleanup
