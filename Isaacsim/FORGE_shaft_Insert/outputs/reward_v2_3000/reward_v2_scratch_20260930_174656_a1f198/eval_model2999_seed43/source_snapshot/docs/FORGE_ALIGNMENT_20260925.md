# FORGE 论文对齐与真机标定清单

依据：用户提供的 `forge_paper.pdf`（正文 II–IV 节）及
`forge_supplementary.pdf`（II–III 节、Table I 的 **8 mm Peg** 列）。
[作者项目页](https://noseworm.github.io/forge/)及
[补充材料](https://noseworm.github.io/forge/static/forge_supplementary.pdf)。

这是在现有 Shaft_large / Base CAD 上移植 FORGE 方法，不是论文 8 mm 插销的几何复现。
论文中的数值、本文的工程选择、尚未对齐的项目分别标注，不能据此宣称完成 sim-to-real 验证。

## 入口与原任务的关系

- 新插入训练任务：`FORGE-Shaft-Insert-Paper-v0`。
- `FORGE-Shaft-Insert-v0`、Position-Only、Camera 仍为旧的 XY 对齐/诊断任务。
- 原任务只在 XY 误差小于 0.25 mm 时成功，不能当作插入成功。
- 原有 `franka_diff_ik.py` 和相机标定未修改。
- 新旧任务动作和观测不兼容，需要重新训练，不能续训旧 checkpoint。

配置集中在 `forge_shaft_insert/forge_paper_env_cfg.py` 的 `ForgePaperParametersCfg`；
控制、观测、奖励在 `mdp/forge_paper.py`；训练网络在 `agents/rsl_rl_ppo_cfg.py`。

## 奖励逐项对齐

每个策略步实际返回的奖励为：

```text
R = K(50, 2, d) + K(100, 0, d)
    + I_place + I_inserted
    - 0.2 * max(0, ||F_contact||₂ - F_threshold)
    - |p_success - I_inserted|
K(a,b,d) = 1 / (exp(-a*d) + b + exp(a*d))
```

| 项目 | 原实现 | 新实现 / 依据 |
|---|---|---|
| 接近奖励 | XY 误差进度，裁剪 ±1 | 粗/细 logistic 关键点奖励；补充材料 Table I |
| 关键点 | 无 | 轴线上 0、20、40 mm 三点，平均距离；点数/位置为本任务工程选择 |
| 放置奖励 | 无 | 每步满足放置条件 +1；正文式 (2) |
| 成功奖励 | XY 对齐 +20 并结束 | 真正向下插入 +1；正文式 (2) |
| 过大接触力 | 无 | 超出阈值的每 N 惩罚 0.2；正文式 (3)、Table I |
| 成功预测 | 无 | `-|p-y|`；正文式 (7) |
| 动作平方惩罚 | -0.001 | 新任务移除；论文总奖励未给出该项 |

`I_place` 采用横向距离 ≤2.5 mm、轴底不高于孔口 2.5 mm、未过深穿透、轴倾角 ≤5°。
论文给出 Place Dist. 数值及“居中放置”的描述，具体高度/倾角门限组合属于本任务实现。
成功要求横向距离 ≤0.5 mm、轴底深度位于目标 ±1 mm、轴倾角 ≤5°。
其中横向容差按本 CAD 的半径间隙设置，倾角与深度双侧检查为工程防误判措施。

**几何适配：** 实际 CAD 网格截面显示插入段直径约25 mm、孔径约26 mm、
孔深约50 mm，轴肩从轴底上方47 mm开始。因此 `insertion_depth=0.047 m`，成功范围46–48 mm，
XY容差0.5 mm。论文的24 mm只适用于其8 mm插销，未照搬。该值是根据CAD轴肩推导的装配目标，
仍应由真机实际止挡/成功装配姿态确认。轴根在轴底，`shaft_bottom_offset=0`。
孔口与轴端都有约2 mm的圆角过渡；接触验证仍需考虑实际碰撞近似。
Z目标范围扩至±75 mm，为47 mm深度、孔估计误差和阻抗压入余量留空间；
若仍使用论文±50 mm，部分带Z估计偏差的目标会落在可达动作范围之外。

Isaac Lab RewardManager 会乘 `step_dt`；配置将各项权重除以 `1/15 s`，
所以运行时显示权重 15、15、15、-3、-15，实际每步奖励仍是上式。
奖励使用无噪接触力；只对策略观测加力噪声。
力阈值是训练条件和软惩罚，不是能保证瞬时力永不超限的安全控制器。

## 控制、观测、随机化

| 项目 | 当前新任务值 | 来源 / 对齐程度 |
|---|---|---|
| 策略频率 | 15 Hz | 正文 IV-A |
| 单回合时长 | 10 s = 150 步 | 补充 Table I |
| 仿真/控制更新 | 240 Hz | 工程选择；论文真机控制为 1000 Hz，不等同 |
| 策略动作 | x、y、z、yaw、成功预测 | 正文 II-B / III-C |
| 位置动作 | 孔估计位置 + 相对目标，xy ±5 cm、z ±7.5 cm | xy 对齐正文 IV-B；z 为长轴适配，不再累计增量 |
| 目标限幅 λ | 每回合 U[16,25] mm | Table I；逐轴相对当前 TCP 限幅 |
| 平移阻抗 Kp | 每回合 U[400,800] N/m | Table I |
| 平移阻尼 Kd | 2√Kp | 正文式 (6) |
| 力死区 | 每回合每轴 U[0,5] N | 正文 III-B；`sign(F) max(|F|-DZ,0)` |
| 力阈值 Fth | 每回合 U[5,10] N，并输入 actor | Table I / 正文 III-A |
| 旋转 Kp | 30 N·m/rad | 工程默认，论文未给相应数值 |
| yaw 动作范围 | ±45° | 工程默认；不能与论文初始 yaw 随机化混淆 |
| 零件摩擦 | 轴、底座静/动摩擦 [0.5,1.0]，每回合重新分配 | Table I；PhysX 使用64个材料桶，dynamic≤static |
| 孔位置误差 | Gaussian σ=2.5 mm，回合内保持不变 | 补充 II / Table I |
| TCP 位置误差 | Gaussian σ=0.25 mm，每步采样 | 补充 II / Table I |
| 力观测误差 | 每轴 Gaussian σ=1 N，每步采样 | 补充 II / Table I |
| 线速度 | 带噪位置的有限差分 | 补充 II；重置首帧速度置0 |
| actor | 相对孔的虚拟 TCP 位置、姿态、线/角速度、力、Fth、上一动作，22维 | 不读取轴真实姿态或控制增益 |
| critic | 无噪 TCP/轴状态、力、Kp、λ、死区、Fth、动作，40维 | 非对称 actor-critic；尚未包含材料/质量特权输入 |
| 网络 | recurrent PPO，1层128维 LSTM，MLP [128,128] | recurrent PPO 对齐正文；具体层数/宽度为工程默认 |
| PPO 优化器 | 32步 rollout，lr=3e-4，γ=.99，GAE=.95，clip=.2，5 epochs/4 minibatches | 延用项目工程参数；论文未公开这些具体数值 |
| 成功时结束 | 训练不因真实/预测成功而提前结束 | 正文 III-C |
| 异常结束 | 超出工作区或明显过深 | 工程例外；其余回合保持150步 |

控制器实际输出 `Jᵀ wrench + gravity compensation`，关闭机械臂关节位置 PD，
保留夹爪驱动，轴仍是靠接触/摩擦夹持的独立刚体。
控制点是名义轴底的**固定虚拟 TCP**，由夹持几何标定，不跟随仿真真实轴滑动。
这使相对孔的动作范围适配长轴，但与论文的 EE 控制点定义仍有差异，部署时必须使用相同变换。

## 尚未完全对齐的部分

1. **几何与初态：** 保留实际 Shaft_large、Base 和物理夹持。通过离线 IK 把初始 TCP
   移至 `(0.627,0,0.13) m`，轴底约在孔上方19 mm；孔每回合在 XY ±2 cm 范围随机化，
   回合内底座保持固定。这匹配相对 XY 搜索范围，但不等同于论文全工作空间固定件位置、
   hand z/yaw、held x/z 的联合随机化。没有把论文 8 mm 插销的夹持偏移硬套到35 mm外包络的轴上。
2. **接触力：** 使用轴对 Base 的过滤接触力，排除夹爪夹紧力。这是插入力代理，
   不等同 Panda 关节力矩估计的末端总外力；当前也不包含其他机器人/工作台接触或完整切向摩擦力。
   真机部署前应比较相同坐标系、滤波及偏置下的力信号。
3. **随机质量：** 正文提到质量随机化，但补充表没有范围；目前保留 CAD 质量，
   未将人为范围冒充论文设置。需要称重和质心信息后再设定标称值与范围。
4. **部分观测噪声：** 补充材料只量化位置和力噪声；未擅设姿态和角速度噪声。
5. **成功预测部署：** 已训练第五动作及其奖励；未对真机启用自动结束/自动提高力阈值。
   阈值 pterm 需要用验证数据标定。论文示例从7.5 N开始每次增加5 N，用于需约15 N的
   snap-fit；不能直接作为当前轴插入任务的自动加力规则。
6. **后端：** Isaac Lab 2.3.2 / Isaac Sim 5.1 与论文 IsaacGym/Factory 不同，
   求解器、碰撞几何及时间离散误差仍需接触验证。没有将随机策略测试当作学会插入。

## 真机数据：优先级与用途

只需要优先提供容易获得且影响最大的项目，难测量的可先使用上述论文设置。

| 优先级 | 希望获得的数据 | 如何提供 / 用来改什么 |
|---|---|---|
| P0 | 轴底到轴肩距离、插入段直径、孔径、孔深、倒角、成功装配轴底深度 | CAD或卡尺读数，mm；决定几何、`insertion_depth`、容差，避免学错成功状态 |
| P0 | 实际 TCP 定义、hand→TCP、TCP→轴底的夹持变换 | xyz(m)+四元数(wxyz)或4×4矩阵；修改 `control_point_in_hand` 与轴底偏移 |
| P0 | 真机控制接口和参数 | Cartesian impedance/其他模式、Kp/Kd及单位、控制频率、策略频率、目标限幅、力矩限幅；对应控制映射 |
| P0 | 同步的一段空载与持轴静止力日志 | 时间戳、坐标系、Fx/Fy/Fz、TCP姿态、是否去重力、滤波；估计零偏/噪声并验证力传感器定义 |
| P1 | 10–30次重复孔定位结果及参考孔位置 | 同一位置的视觉估计xyz、标定参考、时间戳；区分回合固定偏差和逐帧抖动，修改 `hole_noise_std` |
| P1 | 几次已成功的插入轨迹 | 同步目标/实测TCP、关节角/速度、外力、夹爪宽度、成功时间；校准延迟、力分布、成功判据与初态 |
| P1 | 轴质量、工具/夹爪附加载荷和质心 | 电子秤g、质心大致位置mm；修正质量、惯量与力估计中的重力补偿 |
| P1 | 夹爪设定及夹持重复性 | 开口、指尖材料、力/电流设置、几次夹持后轴相对TCP偏差；建立可行的夹持随机化范围 |
| P2 | 小幅自由空间目标变化的跟踪日志 | 目标与实际TCP及时间戳；拟合跟踪滞后/动作延迟/死区，而非仅照搬名义Kp |
| P2 | 已有的滑移或摩擦测量 | 材料组合、法向力与切向力；没有时先用论文[0.5,1.0] |

建议日志 CSV 字段：`timestamp_s, target_x/y/z, tcp_x/y/z, tcp_qw/qx/qy/qz,
Fx/Fy/Fz, q1..q7, dq1..dq7, gripper_width, success_label`；并附坐标系、单位、
采样频率和力滤波/补偿说明。若只提供三项，优先：**装配几何、TCP/夹持变换、控制与力日志**。

## 使用与验证

从 `IsaacLab_2_3_2/` 运行：

```bash
./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/paper_smoke_test.py --headless --num_envs 4 --mode hold --steps 160
./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/paper_smoke_test.py --headless --num_envs 4 --mode directed --steps 145
./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/paper_smoke_test.py --headless --num_envs 32 --mode random --steps 160
./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/train.py --task FORGE-Shaft-Insert-Paper-v0 --headless --num_envs 128
```

`hold`/`directed` 诊断关闭孔误差、相对XY随机化与力死区，隔离控制问题；
`random` 使用完整训练随机化。零动作在新任务中表示目标在估计孔口，**不表示保持当前位置**。
实际验证记录：

- 纯张量检查通过：logistic峰值/单调性/极端距离有限性、超力阈值、带符号死区。
- 4环境保持160步：完整10秒回合，4次timeout，0次异常结束，最大TCP偏移4.06 mm。
- CAD深度修正后4环境受控下插145步：几何成功标签累计56个环境步，0次异常结束，
  轴底到达约47 mm。该脚本使用无噪真值目标，不是学得的策略，不能作为RL成功率。
- 此固定下压测试的峰值力41.48 N，且轴相对名义TCP出现厘米级位移。
  它验证了接触/成功奖励链路，也暴露了夹持与过度加压问题；不能宣称已实现温和无滑移插入。
  需要用训练后的策略及真机夹持/力日志评估。
- 已验证成功判据拒绝仅在孔上方、横向偏离、过深状态，并接受正确深度的居中状态。
- 上述物理诊断使用原±50 mm Z动作范围；最终配置扩到±75 mm以容纳长轴及估计误差。


- 32环境随机动作160步：观测/奖励全部有限，29次timeout、3次工作区终止，
  峰值力47.03 N；随机动作导致的掉落/滑移不代表训练策略性能。该次测试同样使用±50 mm Z范围。
- 最终±75 mm Z配置：8环境 recurrent PPO 实际完成2次更新、512个样本，损失有限，
  保存了 `model_0.pt` / `model_1.pt`。这仅证明训练链路可运行，模型未训练到收敛，插入奖励仍为0。
- 完整日志已保存到 `outputs/paper_validation/`；其中
  [hold4](../outputs/paper_validation/hold4.log)、
  [directed4](../outputs/paper_validation/directed4.log)、
  [random32](../outputs/paper_validation/random32.log)、
  [ppo2](../outputs/paper_validation/ppo2.log)、
  [geometry](../outputs/paper_validation/geometry.log)。
- 训练检查点位于 `IsaacLab_2_3_2/logs/rsl_rl/forge_shaft_insert_paper/2026-09-25_13-31-44_alignment_smoke/`。
  不应把这些仅2次更新的检查点拿到真机执行。

当前成功奖励与论文一样依据零件相对位姿，不额外证明轴仍被稳定夹持。
已观测到的滑移需要在后续训练评估中单独统计；在夹持模型与真机外力测量未校准前，
不能把几何成功率等同于无滑移、安全装配成功率。
