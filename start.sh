#!/usr/bin/env bash
# 一键启动 hand_eye_3D 前后端。
#
#   ./start.sh              # 默认：7012 采集、7013 选点、7015 查看安装标定诊断
#   ./start.sh --no-arm     # 不启用手臂控制（只读 rt/lowstate，绝不发布，可与其他控制程序并存）
#   ./start.sh --teleop-task-dir /path/to/task  # 兼容的纯离线处理模式
#   ./start.sh <其他参数>    # 其余参数原样传给 run_server.py（如 --arm-grav-in-float）
#
# Ctrl+C 退出：后端会先把手臂权重渐出、交还本体控制器（此时请扶住手臂），再退出。
set -u

cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  PY_CANDIDATES=(
    "$HOME/miniconda3/envs/fastapi/bin/python"
    "$HOME/anaconda3/envs/fastapi/bin/python"
    "/opt/anaconda3/envs/fastapi/bin/python"
    "/opt/miniconda3/envs/fastapi/bin/python"
  )
  if [ -n "${CONDA_PREFIX:-}" ]; then
    PY_CANDIDATES+=(
      "$CONDA_PREFIX/bin/python"
      "$CONDA_PREFIX/envs/fastapi/bin/python"
    )
  fi
  for candidate in "${PY_CANDIDATES[@]}"; do
    if [ -x "$candidate" ] &&
       "$candidate" -c "import cv2, fastapi, numpy, uvicorn, yaml" >/dev/null 2>&1; then
      PY="$candidate"
      break
    fi
  done
fi
if [ -z "$PY" ] || [ ! -x "$PY" ] ||
   ! "$PY" -c "import cv2, fastapi, numpy, uvicorn, yaml" >/dev/null 2>&1; then
  echo "[start] 找不到包含 numpy/cv2/fastapi/uvicorn/yaml 的 Python 环境。" >&2
  echo "请设置 PYTHON=/path/to/conda/env/bin/python 后重试。" >&2
  exit 1
fi
echo "[start] Python: $PY"
IFACE="${NETWORK_INTERFACE:-enp86s0}"
CAMERA_HOST="${CAMERA_HOST:-127.0.0.1}"
if [ -n "${RGBD_CALIB:-}" ]; then
  RGBD_CALIB="$RGBD_CALIB"
elif [ -f "$PWD/config/camera/orbbec_rgbd_calibration.json" ]; then
  RGBD_CALIB="$PWD/config/camera/orbbec_rgbd_calibration.json"
else
  RGBD_CALIB="/home/robot/yx/project/IK_replay/config/camera/orbbec_rgbd_calibration.json"
fi

# ---- 18000 能力中心：启动前拜访；未运行则自动拉起（后端拿不到快照会拒绝启动） ----
CAPABILITY_URL="${CAPABILITY_URL:-http://127.0.0.1:18000}"
CAPABILITY_SH="${CAPABILITY_SH:-/home/robot/yx/project/IK_replay/capability.sh}"
if curl -sf --max-time 2 "$CAPABILITY_URL/api/capability/registry" >/dev/null 2>&1; then
  echo "[start] 18000 能力中心可达: $CAPABILITY_URL"
else
  echo "[start] 18000 能力中心未运行，自动拉起: $CAPABILITY_SH"
  if [ ! -f "$CAPABILITY_SH" ] || ! bash "$CAPABILITY_SH"; then
    echo "[start] 18000 能力中心拉起失败，中止（所有项目启动前都要拜访它）。" >&2
    exit 1
  fi
fi

ARM_ARGS=(--arm-control)
EXTRA=()
OFFLINE=0
SAVE_PATH_GIVEN=0
for a in "$@"; do
  if [ "$a" = "--no-arm" ]; then
    ARM_ARGS=()
  else
    EXTRA+=("$a")
    case "$a" in
      --teleop-task-dir|--teleop-task-dir=*) OFFLINE=1 ;;
      --save-path|--save-path=*) SAVE_PATH_GIVEN=1 ;;
    esac
  fi
done
if [ "$OFFLINE" -eq 1 ]; then
  ARM_ARGS=()
  echo "离线遥操作数据模式：不打开相机、不连接或控制机器人。"
else
  echo "统一实时模式：7012 连接相机并采集；7013 读取同一目录中已落盘的 episode。"
  echo "在 7012 按 C 完成采集后，到 7013 点击刷新即可加载。"
fi

VITE_BIN="$PWD/frontend/node_modules/.bin/vite"
if [ ! -x "$VITE_BIN" ]; then
  echo "[start] 前端依赖未安装（缺少 frontend/node_modules/.bin/vite），正在 npm install ..."
  if ! (cd frontend && npm install); then
    echo "[start] npm install 失败。请手动执行: cd frontend && npm install" >&2
    exit 1
  fi
fi
if [ ! -x "$VITE_BIN" ]; then
  echo "[start] 仍找不到 $VITE_BIN，无法启动 7012/7013/7015。" >&2
  exit 1
fi

for port in 8132 7012 7013 7015; do
  if ! "$PY" -c \
    "import socket, sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0', int(sys.argv[1]))); s.close()" \
    "$port" >/dev/null 2>&1; then
    echo "[start] 端口 $port 已被占用。请先结束旧的标定进程再重试。" >&2
    exit 1
  fi
done

if [ ${#ARM_ARGS[@]} -gt 0 ]; then
  echo "手臂控制可用：网页里点「获取控制」后才发布 rt/arm_sdk（获取前请确认没有其他控制程序）。"
  echo "（完全不需要动手臂就用 ./start.sh --no-arm）"
fi

BACK_PID=""
FRONT_IMAGE_PID=""
FRONT_CLOUD_PID=""
FRONT_MOUNT_DIAG_PID=""
cleanup() {
  echo ""
  echo "[start] 正在退出（手臂权重渐出，请扶住手臂）..."
  [ -n "$FRONT_IMAGE_PID" ] && kill "$FRONT_IMAGE_PID" 2>/dev/null
  [ -n "$FRONT_CLOUD_PID" ] && kill "$FRONT_CLOUD_PID" 2>/dev/null
  [ -n "$FRONT_MOUNT_DIAG_PID" ] && kill "$FRONT_MOUNT_DIAG_PID" 2>/dev/null
  [ -n "$BACK_PID" ] && kill -INT "$BACK_PID" 2>/dev/null
  [ -n "$BACK_PID" ] && wait "$BACK_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

SERVER_ARGS=(
  --camera-source zmq
  --camera-host "$CAMERA_HOST"
  --rgbd-calib "$RGBD_CALIB"
  --pose-source h2
  --network-interface "$IFACE"
  --capability-url "$CAPABILITY_URL"
)
if [ "$SAVE_PATH_GIVEN" -eq 0 ]; then
  PERSISTENT_SAVE_PATH="${CALIB_SAVE_PATH:-$PWD/handeye3d_data/biaoding}"
  SERVER_ARGS+=(--save-path "$PERSISTENT_SAVE_PATH" --no-timestamp-dir)
  echo "[start] 标定进度固定保存到: $PERSISTENT_SAVE_PATH"
fi
if [ ${#ARM_ARGS[@]} -gt 0 ]; then
  SERVER_ARGS+=("${ARM_ARGS[@]}")
fi
if [ ${#EXTRA[@]} -gt 0 ]; then
  SERVER_ARGS+=("${EXTRA[@]}")
fi

"$PY" run_server.py "${SERVER_ARGS[@]}" &
BACK_PID=$!

sleep 1
if ! kill -0 "$BACK_PID" 2>/dev/null; then
  echo "[start] 后端启动失败，见上方报错" >&2
  exit 1
fi

(cd frontend && exec "$VITE_BIN") &
FRONT_IMAGE_PID=$!
(cd frontend && exec "$VITE_BIN" --config vite.pointcloud.config.js) &
FRONT_CLOUD_PID=$!
(cd frontend && exec "$VITE_BIN" --config vite.mount-diagnostics.config.js) &
FRONT_MOUNT_DIAG_PID=$!

sleep 1
if ! kill -0 "$FRONT_IMAGE_PID" 2>/dev/null ||
   ! kill -0 "$FRONT_CLOUD_PID" 2>/dev/null ||
   ! kill -0 "$FRONT_MOUNT_DIAG_PID" 2>/dev/null; then
  echo "[start] 前端 Vite 启动失败，7012/7013/7015 将无法访问。" >&2
  cleanup
fi

echo "[start] 后端/前端均监听 0.0.0.0，本机所有网卡 IP 都可访问："
if hostname -I >/dev/null 2>&1; then
  for IP in $(hostname -I); do
    echo "         后端 http://${IP}:8132"
    echo "         图像选点 http://${IP}:7012   点云选点 http://${IP}:7013"
    echo "         安装诊断 http://${IP}:7015"
  done
else
  IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"
  echo "         后端 http://${IP}:8132"
  echo "         图像选点 http://${IP}:7012   点云选点 http://${IP}:7013"
  echo "         安装诊断 http://${IP}:7015"
fi
echo "[start] Ctrl+C 一起退出"

wait "$FRONT_IMAGE_PID" "$FRONT_CLOUD_PID" "$FRONT_MOUNT_DIAG_PID" "$BACK_PID"
cleanup
