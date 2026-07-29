# Franka Global/Local 视觉 Waypoint 操作指南

本指南对应以下脚本：

- `vision_guided_waypoint_commander.py`
- `run_global_vision_stage.py`
- `run_local_vision_stage.py`
- `global_vision_force_adapter.py`
- `local_vision_force_adapter.py`

当前流程仅用于高位 waypoint 和抓取姿态观察验证，不会闭合夹爪、下降抓取、接触零件、执行插入或启动力控。

## 阅读标签

本文所有操作按以下标签区分：

- **【必要｜宿主机】**：每次正式运行都必须在 Ubuntu 宿主机执行。
- **【必要｜Docker】**：每次正式运行都必须在 `multipanda-container` 中执行。
- **【必要｜Franka Desk】**：每次真机运行都必须在 Franka 管理界面确认。
- **【首次/变更后检查】**：首次部署、修改代码/配置、重新编译、换机器人或故障恢复后执行；正常重复运行无需每次全部执行。
- **【故障检查】**：只有出现对应异常时执行。
- **【可选观察】**：不影响主流程，可用于观察和记录。

### 推荐终端分工

| 终端 | 运行位置 | 用途 | 是否必要 |
|---|---|---|---|
| H1 | 宿主机 `jingyun` | 启动/进入 Docker，必要时复制 Commander | 必要 |
| D1 | Docker `developer` | Franka + MoveIt | 必要 |
| D2 | Docker `developer` | `vision_guided_waypoint_commander.py` | 必要 |
| H2 | 宿主机 `jingyun` | Global stage，之后切换为 Local stage | 必要 |
| H3 | 宿主机 `jingyun` | `rqt_image_view` | 可选观察 |
| H4/D3 | 与被检查节点相同 ROS 网络 | `topic echo`、TF、控制器等诊断 | 首次/故障检查 |

识别当前所在环境：

```bash
whoami
```

- 输出 `jingyun`：宿主机。
- 输出 `developer`：Docker 容器。

宿主机路径 `/home/jingyun/Xinyu/...` 默认不能直接在 Docker 中使用；
Docker 路径 `/home/developer/multipanda_ws/...` 也不能直接用宿主机 `ls` 查看。

### 每次正式运行的最小必要流程

以下是问题排除完成后的最短正式流程；其余章节中的 `topic echo`、`rqt`、包路径和参数查询均不是每次必做：

1. **【Franka Desk】** 解除制动、激活 FCI，操作员和急停就位。
2. **【宿主机 H1】** 启动 Docker；宿主机与 Docker 设置相同 `ROS_DOMAIN_ID` 和 `ROS_LOCALHOST_ONLY=0`。
3. **【Docker D1】** source ROS/工作空间，启动真机 Franka + MoveIt，明确 `use_fake_hardware:=false`。
4. **【Docker D2】** source ROS/工作空间，以 `dry_run:=false`、低速参数启动 Commander；如果本次属于首次或变更后运行，先做 dry-run。
5. **【Docker】** 等待 Commander 输出 `MoveIt SUCCESS and live TF arrival verified`；它会自动连续核对实时 `panda_hand`，独立 `tf2_echo` 仍建议保留观察。
6. **【宿主机 H2】** 启动 Global stage，保持相机和机械臂静止，等待 D2 输出 `Global snapshot complete`。
7. **【宿主机 H2】** D2 提示等待 Local 服务后，停止 Global；确认相机驱动退出，再启动 Local stage。
8. **【Docker D2】** Commander 自动切换 Base 三孔和六种零件的 Local 目标并执行高位 waypoint。
9. **【所有终端】** 流程结束后按顺序停止 Commander、Local stage、MoveIt/控制器，并在 Franka Desk 恢复安全状态。

可选的 H3 `rqt_image_view` 可以从 Global 启动后一直保持打开；阶段切换时暂时无画面属于正常现象，它不会独占相机。

---

## 0. 安全边界与流程概览

执行顺序如下：

```text
启动 Franka + MoveIt
        ↓
Commander 移动到固定 Global 观察位
        ↓
启动 Global stage，获取七类零件的稳定粗定位
        ↓
Commander 移动到 Base 上方
        ↓
停止 Global stage（同时关闭 RealSense）
        ↓
启动 Local stage（重新启动 RealSense）
        ↓
Commander 自动检测三个 Base 孔
        ↓
Commander 按顺序移动到各零件上方并自动切换 Local target_part
        ↓
到达 table_z_m + 0.20 m，夹爪保持向下并旋转到 Local 抓取 yaw
```

真机运行前必须满足：

- 急停可用，操作员始终在机器人旁。
- 工作台、相机线缆和夹爪周围没有障碍物。
- 未验证过的代码、位姿或控制链先运行一次 `dry_run:=true`。
- 第一次真实 waypoint 测试使用低速度和低加速度。
- 不得把本流程当作已经验证的抓取或装配流程。

### dry-run 到底是不是每次都必须

`dry_run:=true` 是**安全预检查模式**，不是视觉算法或正式流程的必要运行阶段。它只阻止 Commander 向 MoveIt 发送运动目标，仍会等待视觉话题和参数服务。

满足以下全部条件时，可以跳过 dry-run，直接以低速 `dry_run:=false` 从“移动到固定 Global 观察位”开始：

1. 操作员在机器人旁，急停可用，并正确管理制动和 FCI。
2. 固定 Global 位姿、桌面高度、工作区和碰撞环境已经在当前真机验证。
3. 当前版本 Commander、MoveIt 配置或位姿参数自上次验证后没有改变。
4. 控制器、实时关节状态和 TF 均来自真机，而不是 fake/mock hardware。
5. MoveIt 目标执行后，实时 `panda_link0 <- panda_hand` 已被验证确实到位。

仅“有人管理制动”并不足以跳过其他检查。制动管理不能发现错误目标、错误坐标系、假硬件、陈旧 joint state 或 MoveIt 执行假成功。

> **当前已知阻断项（2026-07-13）**：现场曾出现 MoveIt 返回 `SUCCESS`/Commander 打印 `motion completed`，但实时末端位置仍为 `(0.306421, 0.001104, 0.590808)`，未到目标 `(0.357125, 0.000101, 0.696016)`。在控制器执行链修复并完成到位复验前，不得直接继续非 dry-run 自动流程。还曾出现 RealSense `Depth stream start failure`；深度流未恢复稳定时不得启动 Global/Local 正式流程。

---

## 1. 【必要｜Franka Desk + 宿主机】连接 Franka 并进入 Docker

**【必要｜Franka Desk】** 打开 Franka 管理网页：

```text
http://192.168.4.100
```

解除制动并激活 FCI。机器人指示灯应进入允许外部控制的状态。

**【必要｜宿主机】** 启动并进入项目 Docker：

```bash
docker start multipanda-container
docker exec -it multipanda-container bash
```

**【必要｜Docker】** 进入容器中的 Franka 工作空间后：

```bash
source /opt/ros/humble/setup.bash
cd /home/developer/multipanda_ws
source install/setup.bash

export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
```

**【替代拓扑】** 如果明确不使用容器，则在每个 Franka/MoveIt 终端中执行：

```bash
source /opt/ros/humble/setup.bash
source <Franka工作空间>/install/setup.bash
```

不要使用当前机器上不存在的路径：

```text
~/Xinyu/ais_hiwi/install/setup.bash
```

**【首次/变更后检查｜Docker】** 确认 MoveIt 包可见：

```bash
ros2 pkg prefix franka_moveit_config
ros2 pkg prefix moveit_ros_move_group
```

`franka_moveit_config` 不得显示 `Package not found`。

如果 MoveIt 在容器中、视觉节点在宿主机运行，必须保证两侧：

- 使用相同的 `ROS_DOMAIN_ID`。
- DDS 网络可以互相发现。
- 时间同步正常。
- RealSense 只由运行 stage runner 的一侧访问。

### 当前机器固定的宿主机/容器分工

`/home/jingyun/Xinyu/...` 是宿主机路径；容器内的用户和工作目录是
`developer:/home/developer/multipanda_ws`。除非创建容器时显式做了 bind
mount，否则容器内不能 `cd /home/jingyun/Xinyu/...`。

推荐分工：

```text
Docker 容器：Franka、MoveIt、vision_guided_waypoint_commander.py
宿主机：     RealSense、Global stage、Local stage、rqt_image_view
```

**【首次/网络故障检查｜宿主机】** 检查容器网络：

```bash
docker inspect -f '{{.HostConfig.NetworkMode}}' multipanda-container
```

推荐输出为 `host`。**【必要｜宿主机和 Docker】** 两侧每个相关终端都设置相同的 ROS domain：

```bash
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
```

**【代码变更后必要｜宿主机】** 把最新 Commander 复制到容器工作空间：

```bash
docker cp \
  "/home/jingyun/Xinyu/ais_hiwi/jiarui/NEU FRANKA/vision_guided_waypoint_commander.py" \
  multipanda-container:/home/developer/multipanda_ws/vision_guided_waypoint_commander.py

docker cp \
  "/home/jingyun/Xinyu/ais_hiwi/cam_calibration/outputs/03_hand_eye/hand_eye_result.npz" \
  multipanda-container:/home/developer/multipanda_ws/hand_eye_result.npz
```

每次宿主机上的 Commander 有修改后，都需要重新执行一次 `docker cp`；
hand-eye 标定结果改变后也必须重新复制 NPZ。或者重新创建容器时把该目录配置为只读/读写 bind mount。

---

## 2. 【必要｜Docker D1】启动 Franka 和 MoveIt

按照根目录 README 中已经验证的方式启动：

```bash
ros2 launch franka_moveit_config my_moveit.launch.py \
  robot_ip:=192.168.4.100 \
  load_gripper:=true \
  use_fake_hardware:=false \
  fake_sensor_commands:=false
```

**【首次/变更后检查｜Docker】** 确认 MoveIt action：

```bash
ros2 action list -t | grep move_action
ros2 action info /move_action
```

**【首次/故障检查｜Docker】** 确认 MoveIt 节点和控制器：

```bash
ros2 node list | grep move_group
ros2 control list_controllers
```

`panda_arm_controller` 和 `joint_state_broadcaster` 应为 `active`。

**【首次/故障检查｜Docker】** 确认 Franka 实时 TF：

```bash
ros2 run tf2_ros tf2_echo panda_link0 panda_hand -p 9
```

移动机械臂时，该 TF 必须持续更新。

---

## 3. 【运行前必要确认】固定 Global 观察位

Commander 默认会在工作流开始时移动到以下绝对位姿：

```text
frame: panda_link0
link:  panda_hand

position (m):
x = 0.357124617
y = 0.000100923
z = 0.696015951

quaternion xyzw:
x = 0.998979036635
y = 0.002766559556
z = 0.036441954906
w = 0.026555873828
```

它来自以下 `T_panda_link0_panda_hand`：

```text
 0.997328660  0.003591974  0.072956435  0.357124617
 0.007462966 -0.998574263 -0.052855885  0.000100923
 0.072662561  0.053259160 -0.995933539  0.696015951
 0.000000000  0.000000000  0.000000000  1.000000000
```

默认参数：

```bash
-p move_to_global_observation_on_start:=true
```

如果已经人工移动到该位置，可以禁止自动移动：

```bash
-p move_to_global_observation_on_start:=false
```

注意：该固定位姿只用于 MoveIt 观察 waypoint。视觉坐标转换仍会对每一帧按照图像时间戳查询实时 `panda_link0 <- panda_hand` TF，绝不会用此固定位姿替代实时 TF。

---

## 4. 【运行前必要确认】设置桌面高度

Commander 所有 waypoint 的 Z 都按桌面平面计算：

```text
target_z = table_z_m + waypoint_height_m
```

检测结果中的物体 Z 会记录在日志中，但不参与 waypoint 高度计算。

如果桌面在 `panda_link0` 中为 `z=0`：

```bash
-p table_z_m:=0.0
```

如果实测桌面为 `z=0.025 m`：

```bash
-p table_z_m:=0.025
```

以下两处必须使用相同的 `table_z_m`：

1. `vision_config.yaml` 中 Global 节点的 `table_z_m`。
2. `vision_guided_waypoint_commander.py` 的 ROS 参数 `table_z_m`。

当前高度规则：

```text
Base Local 观察位：       table_z_m + 0.45 m
零件 Local 观察位：       table_z_m + 0.40/0.45 m
精检测抓取姿态 waypoint：table_z_m + 0.20 m
装配孔上方 waypoint：     table_z_m + 0.20 m
```

---

## 5. 【首次/变更后检查｜Docker D2】可选 dry-run

以下命令在 Docker 容器内执行。先从宿主机进入另一个容器终端：

```bash
docker exec -it multipanda-container bash
```

然后在容器中执行：

```bash
source /opt/ros/humble/setup.bash
cd /home/developer/multipanda_ws
source install/setup.bash

export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
```

确认脚本已经复制成功：

```bash
ls -l /home/developer/multipanda_ws/vision_guided_waypoint_commander.py
```

首次部署、代码/位姿/控制配置变更后建议使用：

```bash
python3 vision_guided_waypoint_commander.py \
  --ros-args \
  -p dry_run:=true \
  -p table_z_m:=0.0 \
  -p gripper_axis_mode:=y_parallel
```

dry-run 会打印固定 Global 观察位，但不发送 MoveIt 运动：

```text
MOVE fixed Global observation pose
dry_run=true: no robot motion sent
```

dry-run 只禁止机械臂运动，不会跳过：

- Global 七类结果等待。
- Local 参数服务等待。
- Local `target_part` 自动切换。
- Local 稳定检测结果等待。

由于 Local 检测依赖相机实际移动到目标上方，一次完全不运动的 dry-run 通常无法自动获得全部 Local 结果。dry-run 适合检查 Global 话题、坐标、参数服务、目标高度公式和姿态转换日志，但不能代替完整真机 waypoint 验证。

### 5.1 【正式运行必要｜Docker D2】直接启动低速真机流程

如果已满足第 0 节列出的跳过 dry-run 条件，可以不执行上面的 dry-run，直接运行：

```bash
source /opt/ros/humble/setup.bash
cd /home/developer/multipanda_ws
source install/setup.bash

export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0

python3 vision_guided_waypoint_commander.py \
  --ros-args \
  -p dry_run:=false \
  -p table_z_m:=0.0 \
  -p velocity_scale:=0.08 \
  -p acceleration_scale:=0.04 \
  -p gripper_axis_mode:=y_parallel \
  -p arrival_position_tolerance_m:=0.025 \
  -p arrival_orientation_tolerance_deg:=15.0 \
  -p arrival_tf_max_age_sec:=0.5 \
  -p arrival_verification_samples:=3
```

Commander 首先发送固定 Global 观察位目标。MoveIt 返回成功后会打印：

```text
MoveIt returned SUCCESS; verifying live TF panda_link0 <- panda_hand before continuing
```

此时尚未确认到位，也不会进入下一阶段。只有连续三次实时 TF 均通过位置和姿态误差检查后才会打印：

```text
MoveIt SUCCESS and live TF arrival verified
```

只有第二条日志出现后 Commander 才会继续。默认位置误差上限为 0.025 m，姿态误差上限为 15°，TF 年龄不超过 0.5 秒，并且连续采样的 TF 时间戳必须递增。任一 TF 不可用、陈旧、停止更新或误差超限都会打印 `SAFETY ABORT`、保存实际 TF 并终止整个流程。姿态误差比较的是已经包含 Local 抓取 yaw 的命令姿态与真实 hand 姿态，因此正常的夹爪旋转不会被当成误差。

仍可在 Docker 的另一个终端独立观察：

```bash
source /opt/ros/humble/setup.bash
source /home/developer/multipanda_ws/install/setup.bash
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0

ros2 run tf2_ros tf2_echo panda_link0 panda_hand -p 9
```

平移必须接近：

```text
[0.357124617, 0.000100923, 0.696015951]
```

若独立观察与 Commander 验证结果不一致，立即停止 Commander，不得启动后续自动流程。操作员管理制动不能替代这一到位确认。

---

## 6. 【必要｜宿主机 H2】启动 Global stage

以下命令在宿主机执行，不是在 `developer` 容器 shell 中执行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0

cd "/home/jingyun/Xinyu/ais_hiwi/jiarui/NEU FRANKA"

python3 run_global_vision_stage.py
```

该命令是持续运行节点，不会自行退出。看到 `RealSense Node Is Up!` 只代表驱动完成启动；看到 `Verified 27 fixed global RealSense parameters` 代表固定参数验证完成；只有 Docker D2 的 Commander 打印 `Global snapshot complete`，才代表七类 Global 结果已收齐。

如果出现以下任一情况，Global 未完成，必须先排障：

```text
Depth stream start failure
realsense2_camera_node has stopped unexpectedly
彩色/原始深度/aligned depth 任一路不发布
```

Global stage 会：

1. 启动 `realsense2_camera`。
2. 等待相机参数服务。
3. 启动 Global detector。
4. 强制写入并读回验证 Global 相机参数。
5. 使用 `exposure=260`、`white_balance=3600`。
6. 等待 RGB-D warmup。
7. 等待图像稳定。
8. 执行与 Global notebook 对齐的单帧检测。
9. 等待坐标 consensus。
10. 发布七类零件及可用的 Base 孔。

Commander 必须收到：

```text
Base
Shaft_large
Shaft_small
Screw
Gear_compound_handle
Gear_large_handle
Gear_small_handle
```

**【首次/故障检查｜宿主机】** 检查话题：

```bash
ros2 topic echo /vision/global/Base/pose --once
ros2 topic echo /vision/global/Shaft_large/pose --once
```

必须满足：

```text
header.frame_id: panda_link0
```

---

## 7. 【可选观察｜宿主机 H3】查看 Global 检测窗口

使用 ROS 图像订阅不会抢占 RealSense：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0

ros2 run rqt_image_view rqt_image_view \
  /vision/global/debug/publication_gate
```

也可以查看 notebook 单帧标注：

```bash
ros2 run rqt_image_view rqt_image_view \
  /vision/global/debug/annotated
```

检查：

- 工作区和 mask 是否正确。
- 七类零件是否正确标注。
- Base 是否正确。
- 稳定后是否显示 `PUBLISHED`。
- 轻微抖动时是否显示 `BLOCKED: mild image motion; consensus retained`。
- 持续或明显运动时是否显示 `BLOCKED: sustained/large motion; consensus reset`。
- TF 不可用时是否停止发布。

不要同时运行 notebook 的 `pyrealsense2.pipeline()`、`realsense-viewer` 或另一套 `realsense2_camera`，这些程序会直接争抢相机设备。

---

## 8. 【必要｜Docker D2 + 宿主机 H2】Global 完成与阶段切换

收齐七类结果后，Commander 打印：

```text
Global snapshot complete
```

非 dry-run 时，Commander 随后使用 Base 的 Global XY 移动到：

```text
z = table_z_m + 0.45 m
```

而不是：

```text
z = Base detected z + 0.45 m
```

接着打印：

```text
Waiting for Local vision parameter service.
Stop the Global stage and start run_local_vision_stage.py now.
```

此时在宿主机 H2 按 `Ctrl+C`。`run_global_vision_stage.py` 会同时关闭：

- Global detector。
- RealSense driver。

等待进程完全退出后再启动 Local，避免两套 RealSense driver 同时打开同一相机。

---

## 9. 【必要｜宿主机 H2】启动 Local stage

确认 Global stage 已完全退出，然后在同一宿主机终端执行：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0

cd "/home/jingyun/Xinyu/ais_hiwi/jiarui/NEU FRANKA"

python3 run_local_vision_stage.py \
  --target-part Base \
  --base-hole-target hole_large
```

建议初始目标使用 `Base/hole_large`。随后 Commander 会自动通过 `/local_vision_force_adapter/set_parameters` 切换目标，无需手动修改 notebook，也无需反复重启 Local。

Local stage 会：

1. 重新启动 RealSense。
2. 启动 Local detector。
3. 写入并验证 Local 相机参数。
4. 使用 `exposure=220`、`white_balance=4600`。
5. 启用 spatial 和 temporal filter。
6. 关闭 hole-filling filter。
7. 重新执行 warmup。
8. 等待机械臂和相机稳定。
9. 只执行当前 `target_part` 对应的 notebook 检测分支。
10. 默认保持 `detection_enabled=false`，不执行目标检测。
11. 只有 Commander 通过实时 TF 确认已到对应 Global 观察位后，才启用当前目标检测。
12. 丢弃启用前排队的 RGB-D 帧，再从新的相机帧重新收集 consensus。
13. 发布当前目标的稳定 Local pose。

---

## 10. 【自动执行，无需人工切换】Local 检测顺序

Commander 首先记录 Base 三个语义孔：

```text
target_part=Base, base_hole_target=hole_large
target_part=Base, base_hole_target=hole_small
target_part=Base, base_hole_target=hole_screw
```

然后依次执行：

```text
Shaft_large
Shaft_small
Screw
Gear_compound_handle
Gear_large_handle
Gear_small_handle
```

每个零件的顺序都是：

```text
关闭 Local 检测并清空上一目标 consensus
        ↓
使用 hand-eye 标定把目标 Global XY 换算为“相机光心对准目标”的 panda_hand 观察位
        ↓
连续三次实时 TF 确认到达该 Global 观察位
        ↓
自动设置 Local target_part 并启用 detection_enabled
        ↓
丢弃启用前排队的旧 RGB-D 帧
        ↓
从到位后的新帧执行目标对应的 notebook 检测路径
        ↓
等待新的稳定 Local PoseStamped
        ↓
立即再次关闭 Local 检测
        ↓
保持高位移动到 Local XY，并用实时 TF 验证
        ↓
在高位选择180°等价的最短抓取 yaw、旋转并验证
        ↓
使用 Pilz LIN 只沿 Z 下降到 table_z_m + 0.20 m
        ↓
实时 TF 验证后，使用 Pilz LIN 只沿 Z 返回安全高度
```

因此不需要手动编辑 Local notebook 中的目标变量，也不需要手动执行 `ros2 param set`。

这个顺序对 `Shaft_large`、`Shaft_small`、三种 Gear、Screw 和 Base 孔统一生效。Local 在机械臂移动期间不会继续把上一目标的画面作为下一目标 consensus；即使相似零件同时出现在画面内，发布结果还必须通过与对应 Global 坐标不超过 5 cm 的外层检查。该门控不修改 notebook 的 mask、contour、候选排序或抓取点算法。

相机安装在夹爪侧边，因此 Local 观察位不能简单令 `panda_hand.x/y = Global.x/y`。Commander 从 `hand_eye_result.npz` 读取标定的 `T_gripper_camera`，按当前 tool-down 姿态计算 `R_base_gripper * t_gripper_camera`，再反解使相机光心落在冻结 Global XY 上方的 hand XY。该补偿只用于 Base 和各零件的 Local 观察位；Local 检测完成后的抓取中点仍直接作为 hand 的抓取 XY，不施加相机补偿。默认 `Shaft_small` 仅在执行 Local 检测时使用 `table_z_m + 0.35 m`，其他 Shaft/Screw 的检测高度为 `+0.40 m`，从而缩小小轴画面范围；Shaft_small 检测完成后的高位 XY 对准和 yaw 旋转仍返回 `+0.40 m`。这些都属于检测外层的相机位姿调整，没有修改 notebook 算法。

最新版 notebook 的 `shaft_small` 与 `shaft_large` 使用相同的白色分割、面积范围和矩形度门槛。`shaft_small` 另外使用与相机对中配套的中心方形 ROI、允许透视变形的近正方形长宽比（不超过 `1.65`）和图像边界过滤，以排除 Base 的明显长方形银色区域；方形 ROI 内存在多个合格候选时固定选择面积第二大，只有一个候选时保留该候选。检测结果的 `shaft_rejected_candidates` 会记录每个淘汰轮廓的几何指标及原因。当长宽比不超过 `1.25` 时，算法把正方形的两组正交边视为等价抓取轴，并选择最接近上一帧有效 yaw 的一组，避免 `minAreaRect` 宽高互换造成约 90° 跳变；`shaft_orientation_ambiguous` 和 `shaft_orientation_source` 可用于核对该分支。顶层 `depth_valid` 与实际 `grasp_midpoint.depth_valid` 保持一致。`shaft_large` 仍保持最大合格候选规则。可通过 `/vision/local/debug/shaft_target_roi_mask`、`shaft_candidate_mask` 和 `shaft_rejected_mask` 查看三层筛选结果。

到位日志应包含：

```text
CAMERA-CENTRED Shaft_small Local observation pose
desired camera XY=(...)
commanded hand XY=(...)
calibrated base offset=(...)
```

七类 Global 坐标一旦收齐，Commander 会输出 `Global snapshot complete and locked for this workflow`，立即把这七个位置写入 JSON 并冻结。即使 Global stage 在人工按 `Ctrl+C` 前继续发布，新消息也不会覆盖本次任务用于 Local 观察位的坐标。因此 `global_snapshot.poses` 与机械臂之后实际采用的 Global 目标完全一致。

对容易混淆的同族零件还有两项语义保护：`Shaft_small/Shaft_large` 的 Local 结果必须比另一根 Shaft 的 Global 坐标更接近自己的 Global 坐标至少 5 mm；三种 Gear 同样处理。如果后来检测的同族 Local 坐标与已经接受的另一个同族结果相距小于 2 cm，也会视为重复识别并拒绝。上次 `Shaft_small` 与 `Shaft_large` Local 结果只相距约 4.5 mm，因此会被这一规则拦截。

### 10.1 【正式运行必要】Local 坐标记录与运动安全门

Commander 不修改 notebook 检测策略。每次收到 Local 候选后，它在运动命令外层执行：

1. 立即把候选的时间戳、位置和四元数写入 JSON。
2. 检查坐标是否位于允许的绝对工作空间内。
3. 零件 Local XY 与同名 Global XY 比较；Base 三孔与 Global Base XY 比较。
4. 异常候选标记为 `accepted: false`，不发送 MoveIt 目标，并继续等待同一目标的下一个候选。
5. 只有通过两层检查的候选才写入 `accepted_local_poses` 并用于 waypoint。

默认安全参数按“`panda_link0` 位于工作区后边缘中点，桌面与 `panda_link0` 的 `z=0` 平面重合”设置为：

```text
x: 0.00 ... 0.80 m
y: -0.40 ... 0.40 m
z: 0.00 ... 0.85 m（实际运动目标；检测到的物体 z 不用于运动）
Local 零件与对应 Global 零件的 XY 距离: <= 0.05 m
Local 孔与 Global Base 的 XY 距离: <= 0.15 m
同族目标相对其他 Global 零件的最小距离优势: 0.005 m
不同同族 Local 结果的最小间距: 0.02 m
```

这表示一个 80×80 cm 的矩形目标区域，并禁止运动目标低于 `panda_link0` 的 `z=0` 平面。如果 link0 并不位于工作区后边缘中点，必须根据工作区在 `panda_link0` 中的实际偏移修改 XY 上下限。Docker D2 中可显式写为：

```bash
-p workspace_x_min_m:=0.00 \
-p workspace_x_max_m:=0.80 \
-p workspace_y_min_m:=-0.40 \
-p workspace_y_max_m:=0.40 \
-p workspace_z_min_m:=0.00 \
-p workspace_z_max_m:=0.85 \
-p max_local_global_xy_delta_m:=0.05 \
-p max_hole_base_xy_delta_m:=0.15 \
-p similar_global_association_margin_m:=0.005 \
-p min_similar_local_separation_m:=0.02 \
-p arrival_position_tolerance_m:=0.025 \
-p arrival_orientation_tolerance_deg:=15.0 \
-p arrival_tf_max_age_sec:=0.5 \
-p arrival_verification_samples:=3
```

不要把边界放宽到机械臂的全部理论可达范围。边界应包住零件工作区并留少量检测误差，而不是包住相机画面中的所有背景。

每次运行会在 Commander 当前目录下生成：

```text
local_pose_records/local_pose_records_<UTC时间>.json
local_pose_records/latest.json
```

文件在每个候选后立即更新；急停、`Ctrl+C` 或后续步骤失败时，之前的数据仍保留。流程完整结束时 `status` 为 `completed`；中断为 `interrupted`；异常终止为 `aborted`。在 Docker 中查看：

```bash
cd /home/developer/multipanda_ws
python3 -m json.tool local_pose_records/latest.json
```

记录中的关键字段：

- `global_snapshot.poses`：本次流程实际使用的七类 Global 坐标。
- `local_candidates`：所有收到的 Local 候选和接受/拒绝原因。
- `accepted_local_poses`：最终允许用于运动的 Local 坐标。
- `motion_attempts[].target`：每一次发送或 dry-run 记录的运动目标。
- `motion_attempts[].arrival_verification.samples[].actual`：MoveIt 返回成功后的实时 `panda_link0 <- panda_hand`。
- `position_error_m`、`orientation_error_deg`：目标与真实 TF 的误差。
- `tf_age_sec`、`tf_timestamp_increased`：TF 是否新鲜且持续更新。
- `status: verified_arrival`：该次运动确认真实到位；其他状态均不允许进入下一步。

复制到宿主机保存：

```bash
docker cp \
  multipanda-container:/home/developer/multipanda_ws/local_pose_records/latest.json \
  "/home/jingyun/Xinyu/ais_hiwi/jiarui/NEU FRANKA/latest_local_pose_record.json"
```

注意：当前安全门检查的是检测坐标和每个 MoveIt **目标端点**。它不能数学保证 OMPL 规划出的整条关节轨迹中所有机械臂连杆都不越过某条空间边界。若现场需要这种等级的限制，还必须在 MoveIt PlanningScene 中加入工作台、围栏和禁入区碰撞体；这不能用放宽视觉阈值代替。

---

## 11. 【可选观察｜宿主机 H3】查看 Local 检测窗口

查看包含发布门控状态的画面：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0

ros2 run rqt_image_view rqt_image_view \
  /vision/local/debug/publication_gate
```

查看原始单帧 notebook 精检测标注：

```bash
ros2 run rqt_image_view rqt_image_view \
  /vision/local/debug/annotated_result
```

查看所有实际发布的中间图像：

```bash
ros2 topic list | grep '/vision/.*/debug'
```

重点检查：

- 目标 ROI。
- 候选 contour。
- 左抓取点。
- 右抓取点。
- 抓取中点。
- 抓取方向线。
- depth source 和 fallback source。
- `BLOCKED: mild image motion; consensus retained`（暂停但保留有效历史）。
- `BLOCKED: sustained/large motion; consensus reset`（真实移动保护）。
- `BLOCKED: collecting consensus`。
- `PUBLISHED`。

Global 停止、Local 尚未启动期间，rqt 窗口暂时不更新是正常现象。rqt 只是 ROS 图像订阅者，不会占用相机。

---

## 12. 【正式运行必要｜Docker D2】低速真实 waypoint 流程

本命令与第 5.1 节相同。如果 D2 已经运行了一份 Commander，不要再启动第二份；若此前运行的是 dry-run，应先按 `Ctrl+C` 停止，再重新从固定 Global 观察位开始：

```bash
python3 vision_guided_waypoint_commander.py \
  --ros-args \
  -p dry_run:=false \
  -p table_z_m:=0.0 \
  -p velocity_scale:=0.08 \
  -p acceleration_scale:=0.04 \
  -p gripper_axis_mode:=y_parallel \
  -p arrival_position_tolerance_m:=0.025 \
  -p arrival_orientation_tolerance_deg:=15.0 \
  -p arrival_tf_max_age_sec:=0.5 \
  -p arrival_verification_samples:=3
```

Commander 会先移动到固定 Global 观察位，然后等待 Global stage。

当前脚本只执行高位 waypoint：

- 不下降到物体。
- 不闭合夹爪。
- 不抓取。
- 不插入。
- 不执行力控。

精检测后目标高度固定为：

```text
z = table_z_m + 0.20 m
```

精检测结果不会再作为单个 OMPL“横移+下降+旋转”目标发送。高位 XY 对准和高位 yaw 对准分别验证；下降与返回使用 `pilz_industrial_motion_planner/LIN`，目标 XY 和姿态保持不变。如果 Pilz LIN 不可用或规划失败，流程直接中止，不会回退到 OMPL。

日志会显示：

```text
symmetric gripper shortest-yaw selection
notebook line yaw=... deg
current hand yaw=... deg
selected equivalent yaw=... deg
rotation delta=... deg
planner=Pilz LIN
```

---

## 13. 【首次/工具变更后检查】判断 Panda 夹爪轴方向

Local adapter 发布 notebook 左右抓取点形成的平面 yaw。Commander 将该 yaw 转成 `roll=pi` 的 tool-down 四元数。

观察：

- Local debug 左抓取点和右抓取点。
- 两点连线方向。
- Panda 夹爪实际方向。

当前真机验证发现原 `x_parallel` 稳定相差 90°，因此默认使用：

```bash
-p gripper_axis_mode:=y_parallel
```

只有在更换夹爪、末端 frame 或工具定义后再次出现 90° 偏差，才退回检查：

```bash
-p gripper_axis_mode:=x_parallel
```

不要通过修改 notebook 阈值、mask 或抓取点定义来修正机械臂工具轴约定。

---

## 14. 【辅助检查，非每次必要】常用检查命令

### 使用 move_to_link0_pose.py 检查已记录的 XY

该脚本必须在 Docker 的 Franka/MoveIt 工作空间中运行。宿主机版本更新
后先复制：

```bash
docker cp \
  "/home/jingyun/Xinyu/ais_hiwi/move_to_link0_pose.py" \
  multipanda-container:/home/developer/multipanda_ws/move_to_link0_pose.py
```

进入 Docker 并 source：

```bash
docker exec -it \
  --workdir /home/developer/multipanda_ws \
  multipanda-container bash

source /opt/ros/humble/setup.bash
source /home/developer/multipanda_ws/install/setup.bash
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0
```

只检查视觉 XY 时应使用安全高位，例如：

```bash
python3 move_to_link0_pose.py \
  --ros-args \
  -p target_x:=0.487869523 \
  -p target_y:=-0.096566113 \
  -p target_z:=0.40 \
  -p target_frame:=panda_link0 \
  -p target_q_x:=1.0 \
  -p target_q_y:=0.0 \
  -p target_q_z:=0.0 \
  -p target_q_w:=0.0 \
  -p use_linear:=false \
  -p velocity_scale:=0.05 \
  -p acceleration_scale:=0.02
```

不要直接把 Local PoseStamped 的平面 yaw 四元数当作完整
`panda_hand` 姿态。检查 XY 时保留上述 tool-down 四元数，并先使用
`target_z:=0.40`。不得直接使用检测结果中的低 z 值做首次测试。

### 检查相机节点和服务

**运行位置：宿主机。**

```bash
ros2 node list | grep camera
ros2 service list | grep '/camera/camera/.*parameters'
```

### 检查相机固定参数

**运行位置：宿主机，且对应 Global/Local stage 正在运行。**

```bash
ros2 param get /camera/camera rgb_camera.exposure
ros2 param get /camera/camera rgb_camera.white_balance
ros2 param get /camera/camera spatial_filter.enable
ros2 param get /camera/camera temporal_filter.enable
ros2 param get /camera/camera hole_filling_filter.enable
```

### 检查 Local 自动目标

**运行位置：宿主机。**

```bash
ros2 param get /local_vision_force_adapter target_part
ros2 param get /local_vision_force_adapter base_hole_target
```

### 检查姿态话题

**运行位置：宿主机；若 Docker 与宿主机 DDS 通信正常，也可在 Docker 中检查。**

```bash
ros2 topic echo /vision/global/Shaft_large/pose --once
ros2 topic echo /vision/local/Shaft_large/pose --once
ros2 topic echo /vision/local/hole_large/pose --once
```

### 检查 TF

**运行位置：Docker（Franka/MoveIt 所在环境）。**

```bash
ros2 run tf2_ros tf2_echo panda_link0 panda_hand -p 9
```

---

## 15. 【故障检查】常见问题排查

### Commander 一直等待七类 Global 结果

检查：

```bash
ros2 topic list | grep /vision/global
```

使用 Global debug 确认是否有零件未检测、置信度过低、图像仍在运动或 consensus 尚未完成。不要为了让流程继续而随意修改 notebook 阈值。

### Commander 一直等待 Local 参数服务

检查：

```bash
ros2 node list | grep local_vision_force_adapter
ros2 service list | grep local_vision_force_adapter/set_parameters
```

确认 Global stage 已完全退出，再启动 Local stage。

### Local 切换后没有结果

检查：

- 机械臂是否已经到达目标零件上方。
- 相机画面是否稳定。
- 当前 `target_part` 是否正确。
- 目标是否位于该零件 notebook 检测 ROI 中。
- depth 是否有效。
- debug 是否显示 `collecting consensus`。

### rqt 没有画面

先检查：

```bash
ros2 topic list | grep debug
ros2 topic hz /vision/local/debug/publication_gate
```

如果 stage 已停止，对应话题没有新帧是正常的。

### MoveIt 拒绝目标

检查：

- `/move_action` 是否存在。
- `panda_arm_controller` 是否 active。
- 目标 XY 是否在机械臂工作空间。
- `table_z_m + height` 是否安全。
- 固定 Global 位姿是否可达。
- 当前姿态约束是否导致无解。

### MoveIt 显示成功但真机没有移动

这不是正常完成。立即停止 Commander，对比目标与实时 TF：

```bash
ros2 run tf2_ros tf2_echo panda_link0 panda_hand -p 9
ros2 control list_controllers
ros2 control list_hardware_components
ros2 action info /panda_arm_controller/follow_joint_trajectory
ros2 topic hz /franka/joint_states
```

确认 `panda_arm_controller` 和硬件为 `active`、关节状态来自真机、启动参数显式为 `use_fake_hardware:=false`。在该问题修复前，不得因为有人看管制动就继续自动流程。

### RealSense 有 RGB 但没有 aligned depth

先看相机终端是否出现：

```text
Depth stream start failure
Hardware Error
```

再分别检查：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/depth/image_rect_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
```

三路必须持续稳定发布。出现硬件错误时停止 stage，确保没有第二个相机进程，重新插拔并直连 USB 3.x；必要时用 `initial_reset:=true` 单独测试相机。不要通过删除 depth 同步、使用旧帧或修改 notebook 算法绕过。

### TF 不可用

视觉节点在图像时间戳对应的 TF 不可用时不会发布伪造结果。检查：

```bash
ros2 run tf2_ros tf2_echo panda_link0 panda_hand -p 9
```

不得通过使用固定 Franka 位姿或旧 TF 绕过该保护。

---

## 16. 【首次/变更后检查】真机验证清单

逐项确认：

- [ ] `franka_moveit_config` 可被 ROS2 找到。
- [ ] `/move_action` action server 可用。
- [ ] Franka 控制器为 active。
- [ ] 确认使用真机硬件，`use_fake_hardware=false`。
- [ ] `panda_link0 <- panda_hand` TF 持续更新。
- [ ] 固定 Global 观察位与记录矩阵一致。
- [ ] MoveIt 返回成功后，实时 TF 确认末端实际到位。
- [ ] RGB、原始 depth、aligned depth 均以配置帧率稳定发布，无硬件错误。
- [ ] Global 相机 exposure 为 260、white balance 为 3600。
- [ ] Local 相机 exposure 为 220、white balance 为 4600。
- [ ] `frame_id` 全部为 `panda_link0`。
- [ ] `table_z_m` 与实际桌面高度一致。
- [ ] 所有 waypoint Z 都由 `table_z_m` 计算。
- [ ] Global/Local XY 位于安全工作范围。
- [ ] Local 抓取中点位于实际可抓位置。
- [ ] 夹爪工具轴保持向下。
- [ ] Local yaw 与左右抓取点连线符合预期。
- [ ] `x_parallel/y_parallel` 已在真机观察验证。
- [ ] 切换目标后不会复用上一目标的 consensus。
- [ ] 机械臂运动时视觉发布被阻止。
- [ ] TF 失败时不会发布新结果。
- [ ] rqt 图像窗口持续更新且没有抢占相机。
- [ ] 当前测试没有执行下降、闭爪、抓取、插入或力控。

---

## 17. 【参考信息】已完成的软件验证

- `vision_guided_waypoint_commander.py` 通过 `py_compile`。
- ROS2 Humble 环境下模块导入通过。
- Local 参数切换使用标准 `SetParameters` 服务。
- 固定 Global 观察位旋转矩阵与四元数互相还原，最大旋转误差约 `4.27e-10`。
- Local 90° 平面 yaw 可转换为合法、归一化的 tool-down 四元数。
- Global/Local detector 的 notebook 对齐测试此前均已通过。

以上软件验证不等同于真机碰撞安全、夹爪轴方向或实际抓取装配验证。





## simple run

0. 运行前物理准备
必须确认：
急停已经解除。
Franka Desk 中机器人已经解锁并恢复错误。
工作区内没有杂物和人员。
相机 USB 已稳定连接。
机械臂可以从 Ready 位安全移动到固定 Global 位。
panda_link0 的 z=0 确实是桌面平面。
80×80 cm 工作区对应：
x = 0.00 ～ 0.80 m
y = -0.40 ～ 0.40 m
z = 0.00 ～ 0.85 m

1. 宿主机：同步最新版 Commander 到 Docker
docker cp \
  "/home/jingyun/Xinyu/ais_hiwi/jiarui/NEU FRANKA/vision_guided_waypoint_commander.py" \
  multipanda-container:/home/developer/multipanda_ws/vision_guided_waypoint_commander.py

docker cp \
  "/home/jingyun/Xinyu/ais_hiwi/cam_calibration/outputs/03_hand_eye/hand_eye_result.npz" \
  multipanda-container:/home/developer/multipanda_ws/hand_eye_result.npz

确认文件存在
docker exec multipanda-container \
  ls -l /home/developer/multipanda_ws/vision_guided_waypoint_commander.py

2. 清除上一轮残留进程 
宿主机
pkill -INT -f run_global_vision_stage.py
pkill -INT -f run_local_vision_stage.py

宿主机停止 Docker 中旧 Commander
docker exec multipanda-container \
  pkill -INT -f vision_guided_waypoint_commander.py

检查视觉节点：
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=1
export ROS_LOCALHOST_ONLY=0

ros2 node list | grep -E 'camera/camera|global_vision_force_adapter|local_vision_force_adapter'



3. Docker 终端 D1：启动 Franka 和 MoveIt
        docker start multipanda-container
        docker exec -it multipanda-container bash
        source install/setup.bash

        ros2 launch franka_moveit_config my_moveit.launch.py robot_ip:=192.168.4.100 load_gripper:=true

4. Docker 终端 D2：启动低速 Commander
        docker exec -it \
                --workdir /home/developer/multipanda_ws \
                multipanda-container \
                bash



        cd /home/developer/multipanda_ws

        source /opt/ros/humble/setup.bash
        source install/setup.bash

        export ROS_DOMAIN_ID=1
        export ROS_LOCALHOST_ONLY=0

        python3 vision_guided_waypoint_commander.py \
                --ros-args \
                -p dry_run:=false \
                -p move_to_global_observation_on_start:=true \
                -p global_wait_timeout_sec:=900.0 \
                -p global_optional_hole_collection_sec:=2.0 \
                -p local_wait_timeout_sec:=180.0 \
                -p local_node_wait_timeout_sec:=900.0 \
                -p table_z_m:=0.0 \
                -p base_local_observation_height_m:=0.55 \
                -p velocity_scale:=0.08 \
                -p acceleration_scale:=0.04 \
                -p gripper_axis_mode:=y_parallel \
                -p screw_fixed_grasp_yaw_deg:=0.0 \
                -p camera_centered_local_observation:=true \
                -p camera_aim_verification_tolerance_m:=0.02 \
                -p base_local_camera_target:=hole_screw \
                -p handeye_path:=/home/developer/multipanda_ws/hand_eye_result.npz \
                -p workspace_x_min_m:=0.00 \
                -p workspace_x_max_m:=0.80 \
                -p workspace_y_min_m:=-0.40 \
                -p workspace_y_max_m:=0.40 \
                -p workspace_z_min_m:=0.00 \
                -p workspace_z_max_m:=0.85 \
                -p max_local_global_xy_delta_m:=0.05 \
                -p max_hole_base_xy_delta_m:=0.15 \
                -p similar_global_association_margin_m:=0.005 \
                -p min_similar_local_separation_m:=0.02 \
                -p arrival_position_tolerance_m:=0.025 \
                -p arrival_orientation_tolerance_deg:=15.0 \
                -p arrival_tf_max_age_sec:=0.5 \
                -p arrival_verification_samples:=3 \
                -p execute_trajectory_action:=/execute_trajectory \
                -p linear_plan_attempts:=2 \
                -p linear_min_trajectory_points:=2 \
                -p linear_min_joint_displacement_rad:=0.001 \
                -p linear_min_duration_sec:=0.05 \
                -p linear_start_xy_tolerance_m:=0.01 \
                -p linear_start_orientation_tolerance_deg:=5.0 \
                -p linear_min_z_displacement_m:=0.02

机械臂移动到固定位置：p = (0.357124617, 0.000100923, 0.696015951)

5. 宿主机终端 H2：启动 Global

        source /opt/ros/humble/setup.bash

        export ROS_DOMAIN_ID=1
        export ROS_LOCALHOST_ONLY=0

        cd "/home/jingyun/Xinyu/ais_hiwi/jiarui/NEU FRANKA"

        python3 run_global_vision_stage.py

- 查看检测窗口 
        source /opt/ros/humble/setup.bash

        export ROS_DOMAIN_ID=1
        export ROS_LOCALHOST_ONLY=0

        ros2 run rqt_image_view rqt_image_view

        （/vision/global/debug/publication_gate 
        这个窗口会显示实时检测结果和发布状态：
        PUBLISHED: N：当前帧有 N 个稳定结果完成发布。
        BLOCKED: mild image motion; consensus retained：轻微抖动，暂停当前帧但保留历史。
        BLOCKED: sustained/large motion; consensus reset：持续或明显移动，清空历史。
        BLOCKED: image-time TF unavailable：图像时刻对应 Franka TF 不可用。
        PUBLISHED: 0：当前帧没有结果通过检测和稳定性门控。）

6. 切换 Global → Local

（当docker D2显示 wait for local 在global终端Ctrl+C，在宿主机检查：ros2 node list | grep -E 'camera/camera global_vision_force_adapter'）

7. 宿主机 H2：启动 Local   刚刚开启global的terminal
        
        source /opt/ros/humble/setup.bash
        export ROS_DOMAIN_ID=1
        export ROS_LOCALHOST_ONLY=0

        cd "/home/jingyun/Xinyu/ais_hiwi/jiarui/NEU FRANKA"

        python3 run_local_vision_stage.py \
                --target-part Base \
                --base-hole-target hole_large

- 查看 Local 窗口

        source /opt/ros/humble/setup.bash
        export ROS_DOMAIN_ID=1
        export ROS_LOCALHOST_ONLY
        export ROS_DOMAIN_ID=1
        export ROS_LOCALHOST_ONLY=0

        cd "/home/jingyun/Xinyu/ais_hiwi/jiarui/NEU FRANKA"

        python3 run_global_vision_stage.py

- 查看检测窗口 
        source /opt/ros/humble/setup.bash

        export ROS_DOMAIN_ID=1
        export ROS_LOCALHOST_ONLY=0

        ros2 run rqt_image_view rqt_image_view

        （/vision/global/debug/publ=0

        ros2 run rqt_image_view rqt_image_view \
                /vision/local/debug/publication_gate


Local 阶段的正常运动顺序：
        关闭检测
        → 移动到该零件冻结的 Global XY 上方
        → 实时 TF 验证到位
        → 启用对应 Local 检测
        → 等待新的稳定 Local 坐标
        → 检查 Local 与对应 Global XY 相差不超过 5 cm
        → 关闭检测
        → 高位移动到 Local XY
        → TF 验证
        → 高位选择最短等价 yaw 并旋转
        → TF 验证
        → Pilz LIN 仅沿 Z 下降到 z=0.20 m
        → TF 验证
        → Pilz LIN 仅沿 Z 返回高位

正常日志包括：
        Local detection enabled after verified Global arrival
        high Local XY alignment
        export ROS_DOMAIN_ID=1
        export ROS_LOCALHOST_ONLY=0

        cd "/home/jingyun/Xinyu/ais_hiwi/jiarui/NEU FRANKA"

        python3 run_global_vision_stage.py

- 查看检测窗口 
        source /opt/ros/humble/setup.bash

        export ROS_DOMAIN_ID=1
        export ROS_LOCALHOST_ONLY=0

        ros2 run rqt_image_view rqt_image_view

        （/vision/global/debug/publ
        symmetric gripper shortest-yaw selection
        high grasp-yaw alignment
        planner=Pilz LIN
        linear Z descent
        MoveIt SUCCESS and live TF arrival verified
        linear Z retreat

关键 Pilz LIN 现在不再使用 MoveGroup 的“规划并立即执行”。Commander
会先用 `plan_only=true` 获取轨迹，检查轨迹至少包含 2 个点、持续时间
不小于 0.05 秒、首尾最大关节位移不小于 0.001 rad，并检查下降前实时
TF 的 XY 和姿态已经与目标高位一致。只有轨迹通过验证后才发送到
`/execute_trajectory` 单独执行，最后仍用实时
`panda_link0 <- panda_hand` 验证实际到位。无效轨迹最多重新规划一次，
不会自动执行，也不会回退到 OMPL。

正式启动 Commander 时可显式保留以下默认安全参数：

```bash
-p execute_trajectory_action:=/execute_trajectory \
-p linear_plan_attempts:=2 \
-p linear_min_trajectory_points:=2 \
-p linear_min_joint_displacement_rad:=0.001 \
-p linear_min_duration_sec:=0.05 \
-p linear_start_xy_tolerance_m:=0.01 \
-p linear_start_orientation_tolerance_deg:=5.0 \
-p linear_min_z_displacement_m:=0.02
```

启动 MoveIt 后应能看到以下 action：

```bash
ros2 action list -t | grep -E 'move_action|execute_trajectory'
```

预期至少包括：

```text
/move_action [moveit_msgs/action/MoveGroup]
/execute_trajectory [moveit_msgs/action/ExecuteTrajectory]
```

8. 全流程完成标志
Docker D2 最终应输出：
Waypoint workflow completed. No gripper closing, contact descent,
force-control insertion or gear placement was executed.

Ctrl+C 结束 Local 宿主机终端


9. 查看本轮记录

        cd /home/developer/multipanda_ws

        python3 -m json.tool \
                local_pose_records/latest.json | less


==========

        位置信息： timeout 15 ros2 topic echo /vision/global/Base/pose --once

        查看global检测结果：
        cd /home/developer/multipanda_ws
        python3 -m json.tool local_pose_records/latest.json


- Local 检测
        cd "/home/jingyun/Xinyu/ais_hiwi/jiarui/NEU FRANKA"

        python3 run_local_vision_stage.py \
                --target-part Base \
                --base-hole-target hole_large


        可视化检查:     source /opt/ros/humble/setup.bash

                        export ROS_DOMAIN_ID=1
                        export ROS_LOCALHOST_ONLY=0

                        ros2 run rqt_image_view rqt_image_view
                        （/vision/local/debug/publication_gate）



- 自检测  需确保关闭：宿主机
        source /opt/ros/humble/setup.bash

        export ROS_DOMAIN_ID=1
        export ROS_LOCALHOST_ONLY=0

        ros2 node list | grep -E '^/camera/camera$|^/global_vision_force_adapter$|^/local_vision_force_adapter$|^/vision_guided_waypoint_commander$'

        pgrep -af 'run_global_vision_stage|run_local_vision_stage|global_rgb_part_detection|local_rgbd_grasp_refinement|realsense2_camera_node|rs_launch.py'
        如果仍有输出 则pkill -INT -f global_rgb_part_detection_node.py
        pkill -INT -f local_rgbd_grasp_refinement_node.py
        pkill -INT -f realsense2_camera_node
        pkill -INT -f rs_launch.py

        docker exec multipanda-container \
        pgrep -af 'vision_guided_waypoint_commander|move_to_link0_pose'
        如果仍有输出 则
        docker exec multipanda-container \
        pkill -INT -f vision_guided_waypoint_commander.py

        docker exec multipanda-container \
        pkill -INT -f move_to_link0_pose.py

        =========或者 单独执行

        ros2 node list | grep '^/camera/camera$'
        ros2 node list | grep '^/local_vision_force_adapter$'
