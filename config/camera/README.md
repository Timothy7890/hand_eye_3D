# Orbbec RGB-D 标定

`orbbec_rgbd_calibration.json` 与具体相机设备、彩色/深度流分辨率及格式绑定，仅用于匹配的本地设备配置。更换设备或流配置后，请使用本项目的 `tools/export_orbbec_rgbd_calibration.py` 重新生成。

运行时若相机序列号与标定文件不一致，只会发出警告；图像 shape 和像素格式仍会严格校验，不匹配时拒绝使用标定。

例如，导出彩色 `1920x1080@8`、深度 `1280x800@6` 的标定：

```bash
python tools/export_orbbec_rgbd_calibration.py \
  --color-width 1920 --color-height 1080 --color-fps 8 \
  --depth-width 1280 --depth-height 800 --depth-fps 6
```
