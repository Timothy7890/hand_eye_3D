#!/usr/bin/env bash
# 启动离线多颜色手眼标定：7012 图像编辑 + 7013 点云选点。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="${TELEOP_TASK_DIR:-$ROOT/teleop_data/biaoding}"
SAVE_PATH="${CALIB_SAVE_PATH:-$ROOT/handeye3d_data/biaoding}"
RGBD_CALIB="${RGBD_CALIB:-$ROOT/config/camera/orbbec_rgbd_calibration.json}"
MOUNT_CALIB="${MOUNT_CALIB:-$SAVE_PATH/handeye3d_result.json}"

if [ ! -d "$TASK_DIR" ]; then
  echo "[calibration] 遥操作任务目录不存在: $TASK_DIR" >&2
  echo "可通过 TELEOP_TASK_DIR=/path/to/task 指定。" >&2
  exit 1
fi

if [ ! -f "$RGBD_CALIB" ]; then
  echo "[calibration] RGB-D 标定文件不存在: $RGBD_CALIB" >&2
  echo "请先运行 tools/export_orbbec_rgbd_calibration.py，或设置 RGBD_CALIB。" >&2
  exit 1
fi

echo "[calibration] 数据目录: $TASK_DIR"
echo "[calibration] RGB-D 标定: $RGBD_CALIB"
echo "[calibration] 保存目录: $SAVE_PATH"

ARGS=(
  --teleop-task-dir "$TASK_DIR"
  --rgbd-calib "$RGBD_CALIB"
  --save-path "$SAVE_PATH"
  --no-timestamp-dir
)
if [ -f "$MOUNT_CALIB" ]; then
  echo "[calibration] 手安装相机外参: $MOUNT_CALIB"
  ARGS+=(--mount-calib "$MOUNT_CALIB")
else
  echo "[calibration] 尚无手安装相机外参，将只启用原有采集/手眼解算。" >&2
  echo "[calibration] 可通过 MOUNT_CALIB=/path/to/result.json 指定。" >&2
fi

exec "$ROOT/start.sh" "${ARGS[@]}" "$@"
