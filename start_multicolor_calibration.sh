#!/usr/bin/env bash
# 启动离线多颜色手眼标定：7012 图像编辑 + 7013 点云选点。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="${TELEOP_TASK_DIR:-$ROOT/teleop_data/biaoding}"
SAVE_PATH="${CALIB_SAVE_PATH:-$ROOT/handeye3d_data/biaoding}"
RGBD_CALIB="${RGBD_CALIB:-$ROOT/config/camera/orbbec_rgbd_calibration.json}"

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
echo "[calibration] 标定文件: $RGBD_CALIB"
echo "[calibration] 保存目录: $SAVE_PATH"

exec "$ROOT/start.sh" \
  --teleop-task-dir "$TASK_DIR" \
  --rgbd-calib "$RGBD_CALIB" \
  --save-path "$SAVE_PATH" \
  --no-timestamp-dir \
  "$@"
