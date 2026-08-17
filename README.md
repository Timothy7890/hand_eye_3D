# Hand-Eye 3D — 眼在手外标定（联合估计指尖偏移，免棋盘格）

利用深度相机能直接给三维坐标的特点做手眼标定：同一个物理标记点
（灵巧手指尖 / 手背贴纸），相机侧点击画面得 \(P_{camera}\)，机器人侧
只需提供**手腕位姿** \(T_{base}^{wrist}\)（DDS 自动读取）。求解器把
相机外参和指尖偏移一起解出来，**不需要事先测量指尖装在哪**：

\[ R\,P_{camera,i} + t \;=\; R_{w,i}\,p_{tool} + t_{w,i} \]

未知量：\(T_{base}^{camera}\)（6 维）+ \(p_{tool}\)（腕系下指尖偏移，3 维）。
交替最小二乘（固定 \(p_{tool}\) 是 Kabsch 闭式解，固定 \(T\) 是线性 LS），
单调收敛。已用仿真验证：14 样本 + 2mm 噪声下旋转误差 0.1°、平移 1.9mm、
\(p_{tool}\) 误差 1.7mm。

与 `../hand_eye`（棋盘格 + `cv2.calibrateHandEye`）互补：本方法不用打印
标定板、上手快，精度约 3–8mm；追求更高精度用那套。

## 坐标系约定

- \(P_{camera}\)：**彩色相机坐标系**（生产模式使用本地导出的 RGB-D 内外参，
  将 ZMQ 原始深度软件对齐到彩色后再用彩色内参反投影；X 右、Y 下、Z 前，米）。
  与 video_tools 的彩色点云同系，
  解出的 T 可直接用于点云。
- \(T_{base}^{wrist}\)：H2 模式下 base = `torso_link`、wrist = `right_wrist_yaw_link`
  （取自 IK_replay 的 h2.yaml），FK 只用手臂 7 关节。

## 目录

```
backend/
  solver.py    Kabsch + 联合解(交替 LS) + 留一验证 + 退化检测
  camera.py    teleimager ZMQ RGB-D（生产）+ Orbbec SDK（显式调试）+ mock
  robot.py     手腕位姿 Provider：manual / http / h2(DDS+FK) / mock
  app.py       FastAPI：预览、点击反投影、样本管理、解算
run_server.py  入口（后端 8132）
frontend/      Vue3 + Vite 界面（端口 7012）
```

## 环境

```bash
pip install -r backend/requirements.txt   # 包含 pyzmq；pyorbbecsdk2 仅供显式 SDK 调试
cd frontend && npm install
```

H2 模式额外需要 `unitree_sdk2py` + `cyclonedds`（目前这台机器只有
`unifolm-wma` 环境装了）。三种方案任选：装进你的环境 / 直接用
unifolm-wma 环境跑本服务 / 在 unifolm-wma 里跑一个 pose sidecar 走
`--pose-source http`。

## 启动

```bash
# H2 真机（推荐）：teleimager ZMQ RGB-D + DDS 只读 rt/lowstate + IK_replay FK
python run_server.py --camera-source zmq --camera-host 127.0.0.1 \
    --pose-source h2 --network-interface eth0

# H2 真机 + 网页点动/卸力（发布 rt/arm_sdk，真机会动！确保没有其他控制程序）
python run_server.py --camera-source zmq --camera-host 127.0.0.1 \
    --pose-source h2 --network-interface eth0 --arm-control

# 手腕位姿手填（任何机器人可用）
python run_server.py --camera-source zmq

# 仅限明确需要本机 SDK 调试时使用（会直接打开并占用相机）
python run_server.py --camera-source orbbec --camera-serial CP0BB53000FS

# 纯联调
python run_server.py --camera-source mock --pose-source mock

# 前端
cd frontend && npm run dev     # http://<IP>:7012
```

> 默认 H2 模式**只订阅** `rt/lowstate`，绝不发布 `rt/arm_sdk`/`rt/lowcmd`，
> 与现有控制程序并存不会引起抢占/抽搐。摆位姿用你现有的控制方式。
> 默认相机模式也只订阅 teleimager ZMQ，不会启动 SDK 或直接打开 USB 相机。

### 导入 eai-teleop-studio 离线采集数据

本项目可以直接读取 eai-teleop-studio 的手眼标定任务目录：

```bash
./start.sh \
  --teleop-task-dir /home/robot/yx/project/calib/hand_eye_3D/teleop_data/biaoding \
  --rgbd-calib /home/robot/yx/project/IK_replay/config/camera/orbbec_rgbd_calibration.json
```

离线模式不会打开 Orbbec、不会连接 DDS，也不会控制机器人。后端会复用
IK_replay 的 H2 URDF/FK 和 `SoftwareDepthAligner`：

1. 网页左侧选择一个 `episode_*`，显示该点位 5 帧中的代表 RGB。检测器先寻找
   手部附近的 8 mm 圆形，再按圆心区域的 HSV/Lab 颜色分类。
2. 每个姿态只确认手心或手背当前实际可见的颜色，不要求同一帧凑齐九色。在 SVG
   叠加层中逐个检查圆心和颜色；可以拖动圆心、修改颜色、删除误检或补充漏检。
   灰色、金色和棕色受光照影响较大，必须重点检查。
3. 点击「确认全部标记」。后端将全部 raw `uint16` 深度只配准一次，分别取各圆心
   的 5 帧深度中值并反投影为彩色相机系 `p_camera`。任何无稳定深度的圆都会明确
   报错，修正圆心后重新确认。
4. 后端对 5 帧右臂实测关节角取中值，通过 H2 FK 计算
   `T_torso_link←right_wrist_yaw_link`。
5. 点击「整批保存观测」，再处理下一个 episode。至少需要 3 个不同姿态，每种参与
   解算的颜色必须覆盖至少 2 个姿态；建议采集 12–20 个手腕朝向充分分散的姿态，
   然后点击「解算」。

多标记求解共享同一个 `T_base←camera`，并为每种颜色独立估计固定的腕系位置
`p_tool_wrist_m_by_marker[color]`。它允许部分圆被遮挡。相比每个姿态只用一个点，
正确标注的多点能提高约束数量和抗随机深度噪声能力；但错误颜色、错误圆心和反光
深度属于系统误差，不会因为点数多而自动消失。

转换后的 `samples/*.json`、来源信息和最终 `handeye3d_result.json` 默认保存在
本项目的 `handeye3d_data/<时间戳>/` 下。列表会显示每个 episode 已导入的颜色
数量，同一 episode 的同一种颜色不能重复保存。RGB-D 相机序列号、分辨率、深度
类型或右臂关节顺序与 JSON 标定不一致时，导入会直接拒绝，不会使用错误几何继续
解算。旧的单点样本仍可单独解算，但不能与多标记样本混在同一次解算中。

### --arm-control 手臂点动（可选）

加 `--arm-control` 后，网页顶部多出「手臂控制」卡片。**启动时不接管**——
点「获取控制」才开始发布 `rt/arm_sdk` 并在当前姿态刚性保持，点「归还控制」
权重渐出交还本体控制器（与 IK_replay 的接管/释放语义一致）。摆位姿方式等价于老 hand_eye：

- **开启点动**：7 个关节各有 ±按钮，步长 0.5°/1°/2°/5°/10° 可选；
  目标以限速（`--arm-max-speed`，默认 0.2 rad/s）平滑逼近，且钳制在 URDF 限位内。
- **卸力拖动**：被控手臂 kp=0 只留小阻尼，人手直接拖到目标位姿；
  **手臂会因重力下坠，进入前必须扶住**。摆好后点「保持当前位置」即刚性锁定。
- 另一条手臂全程保持在启动瞬间的实测姿态；启动/退出时权重 1s 渐入/渐出，无跳变。

**安全须知**：接管后会发布 `rt/arm_sdk`。宇树机器人不允许两个程序同时控制身体，
点「获取控制」前务必停掉遥操作等一切在控制手臂的程序，否则会抽搐。
归还控制 / 退出服务前请扶住手臂（权重渐出后手臂交还本体控制器）。

## 操作流程

1. 灵巧手保持**固定手势**（整个标定期间不许变，p_tool 是常量的前提），
   在手背/指节贴一块哑光标记。
2. 机械臂移到新位姿并停稳（位置撒满任务空间，**手腕朝向也要充分变化**，
   姿态跨度 < 15° 时求解器会拒绝解算——朝向不变的话 p_tool 和 t 分不开）。
3. 网页点击标记点 → 得 \(P_{camera}\)（8 帧 × 5×5 窗口中值，自动拒绝飞点）；
   h2/http 模式会在同一时刻自动抓取手腕位姿，manual 模式手填 xyz+rpy。
4. 「保存这个样本」。重复 12–20 次。
5. 「解算」→ 输出 4×4 矩阵、RPY、p_tool、拟合 RMS、留一交叉验证；
   存到 `<save_path>/handeye3d_result.json`。

验收参考：拟合 RMS < 8mm、留一均值 < 10mm。超标常见原因：采样时手臂没
停稳、点到深度飞点、灵巧手手势中途变了。删掉可疑样本重解即可。

## 指尖尖点标定（pivot，网页第 4 卡片）

只标 p_tool、不动相机外参时用这个——比联合解更准，因为它不吃相机深度噪声：

1. 找一个固定的尖角参照物（桌角、螺丝尖）。
2. 「卸力拖动」把**指尖顶在该点上** → 「保持当前位置」→ 「采样当前姿态」。
3. 换手腕姿态（**务必包含反手大角度 roll**，即拨开关那个姿态）重新顶到同一点，
   重复 6 次以上，姿态跨度 < 25° 会拒绝解算。
4. 「解算」→ 线性最小二乘同时解出 p_tool（腕系）和固定点（基座系），
   残差直接反映"各姿态下指尖没钉在同一点"的程度。

结果存 `<save_path>/pivot_result.json`；若同目录已有 `handeye3d_result.json`，
会另存一份替换了 p_tool 的完整标定文件 `handeye3d_result_pivot.json`，
可直接给 `reach_server --calib` 用（原 `--tool-out-mm 10` 的补偿此时应给 0，
因为 pivot 标的就是真指尖）。

原理：每个姿态满足 \(R_i\,p_{tool} + t_i = q\)（q 为固定点），堆叠成
\([R_i \mid -I]\,[p_{tool}; q] = -t_i\) 一次解出。姿态转得越开，
p_tool 的横向分量越可辨识——正是反手 roll 下误差翻倍问题的对症解法。

## 结果怎么用

```python
import json, numpy as np
r = json.load(open("handeye3d_result.json"))
T = np.array(r["T_cam2base"])              # torso_link <- 彩色相机
p_base = (T @ np.append(p_camera, 1.0))[:3]
p_tool = np.array(r["p_tool_wrist_m"])     # 顺带解出的指尖在腕系的位置
```
