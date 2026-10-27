# FORGE shaft_large/base：两版第 200 轮策略实机交接包

## 策略身份

| 子目录 | 训练方式 | checkpoint |
|---|---|---|
| `zero/` | 零随机化训练 | 第 200 轮最终模型，训练 reward 标签 404.08377 |
| `randomized/` | 完整随机化训练 | 第 200 轮最终模型，训练 reward 标签 341.10693 |

这是本轮 FORGE 自有 CAD 扰动实验使用的两个模型。权重文件、来源绝对路径及 SHA256 见 `manifest.json`；reward 文件名不代表两者实机成功率。

每个目录含原始完整 `checkpoint_ep200.pth`、可独立运行的 `policy_actor.ts`、原训练 `training_params/`、参考序列与导出验证记录。TorchScript 包含 actor、LSTM、训练观测均值/方差；不需要 critic、优化器或 Isaac Sim 来推理。`.pth` 用于追溯/原 RL-Games 加载。

## 使用

推理与验证仅依赖 PyTorch。导出环境 PyTorch 2.7.0+cu128；CPU 推理无需 GPU，建议在相同主次版本上校验。重新导出还需 rl-games 1.6.1、numpy 1.26.0、PyYAML 6.0.2。完整版本在 manifest 中。

```bash
cd /home/masais/Xinyu/Isaacsim/real_test/forge_ep200_transfer
/home/masais/miniconda3/envs/env_jingyun/bin/python verify_package.py
```

实机主程序中调用示例（`observation_24` 必须由实际传感接口按下文构造）：

```python
from inference import Policy
policy = Policy('/path/to/forge_ep200_transfer/randomized', seed=0)
policy.reset()  # 新一轮装配时清零网络记忆
normalized_action = policy.step(observation_24, mode='stochastic')
# normalized_action 需经过 FORGE EMA、目标解码、目标限幅及底层控制器。
```

`stochastic` 与此前评估相同，输出 clamp(mu+sigma*epsilon,-1,1)。`mean` 使用 epsilon=0，是可显式选择的确定性推理方式，但改变评估协议，不能直接沿用随机动作评估的成功率。零随机化训练不等于确定性动作策略。切换模型或开始新回合须清零 LSTM；不要每个控制步清零。

## 观测接口：float32 [batch,24]

先按原定义构造物理量，然后交给策略内的归一化；不要外部重复归一化。米、秒、弧度、牛顿。

| Python 切片 | 字段 | 语义 |
|---|---|---|
| 0:3 | fingertip_pos_rel_fixed | 夹爪双指中点位置减孔口参考点；同一世界/机器人映射坐标系中的向量，不是直接转到孔朝向局部坐标 |
| 3:7 | fingertip_quat | 原四元数顺序 wxyz；FORGE 观测将 w、z 分量设为 0，然后施加符号增广，不能直接替换为未经处理的法兰四元数 |
| 7:10 | ee_linvel | 原环境位置差分速度，不是任意机器人 SDK 速度字段 |
| 10:13 | ee_angvel | 原环境四元数差分角速度，X/Y 分量置 0 |
| 13:16 | ft_force | 腕部力 XYZ，按原世界坐标/参考变换、平滑后使用；不是直接送原始工具坐标传感器读数 |
| 16:17 | force_threshold | 接触惩罚阈值；零随机化值 7.5 N，随机化训练范围 5–10 N；不是硬件急停力阈值 |
| 17:24 | prev_actions | 上一步 EMA 后的七维动作，其中索引 3、4 置 0；不是上一步原始网络输出 |

孔口参考为 base root 经姿态变换后的局部 `[0,0,height+base_height]`，CAD height=0.047 m，具体 base_height 见配置。工具参考是双指中点，不是法兰原点。shaft 质量 0.0175 kg、长度 0.083 m。

速度差分、力平滑和 action decimation 必须按源代码调用时序对齐：仿真物理步长 1/120 s，decimation=8，策略约 15 Hz；不能将物理速率更新直接替换为 15 Hz 更新而认为滤波等价。力平滑系数 0.25，源代码见 `source_reference/forge/forge_env.py::_compute_intermediate_values`。观测四元数处理比较特殊，应保留原实现顺序，不能在置零后擅自重新归一化。实机不需要人为注入训练噪声，但需要选择并固定符合接口的四元数符号约定。

## 动作接口：float32 [batch,7]

TorchScript 输出已裁剪到 [-1,1]，尚未做环境 EMA 或目标解码。

1. EMA：a_filtered=alpha*a+(1-alpha)*a_previous；零随机化 alpha=0.0625，随机化训练 alpha 范围 0.025–0.1。`prev_actions` 来自该滤波状态。
2. 前三维乘 `[0.05,0.05,0.05]` m 后加孔口参考点，得到绝对位置目标；不是每步累计位移。随后相对当前指尖位置进行位置阈值限幅，标称各轴 0.020 m。
3. 索引 3、4 在姿态目标计算中忽略；索引 5 映射到 yaw：−π+(3π/2)*(a[5]+1)/2，再与 roll=π 的变换组合。随后按原 Euler wrap 和阈值限幅；标称旋转阈值 0.097 rad。不能直接把该 yaw 当成机器人法兰 yaw。
4. 索引 6 是成功预测，映射 `(a[6]+1)/2`；不是夹爪开合。原流程维持夹爪闭合，没有“插入 25 mm 后松爪”的逻辑。
5. 解码后的末端目标交给原任务空间控制流程。增益、死区、雅可比与关节力矩映射见 `source_reference/factory/factory_control.py`、`factory_env.py` 和 `forge/forge_env.py`。只有策略网络并不等于完整机器人控制器。

回合 reset 的 EMA 初始动作不是全零：位置部分由当前双指中点相对孔口位置除以 0.05 构造，yaw 由原逆映射构造，成功预测初始化为 −1。详见 `forge_env.py::_reset_idx`；LSTM 清零和动作滤波状态初始化是两件事。

## 实机接口边界

本包已验证网络导出等价性，尚未连接机器人或进行实机测试。机器人型号、SDK、法兰到双指中点标定、base/孔口定位、力传感坐标与偏置/载荷补偿、实际控制模式需由实机程序对齐后才能发出运动命令。现有仿真将 shaft 重力关闭；实机不能直接复制该假设，需要检查负载及重力补偿。这里没有加入未经验证的硬件驱动或自动启动运动脚本。

`source_reference/` 是接口追溯用的原源码，不是直接可运行的机器人驱动。`training_params/` 中绝对路径属于训练机，不能视为目标机已安装路径。实验结果附件仅用于了解仿真表现，不代表实机成功率。

## 验证

两个模型各与原 RL-Games 网络比较 batch=1/4、各 80 步的连续推理，包含部分环境记忆 reset、归一化裁剪、mu/sigma、随机采样裁剪动作和 h/c 状态。当前 CPU 数值最大绝对差为 0。`verify_package.py` 可在目标机检查文件指纹并复演各模型 80 步参考序列；它不会连接硬件。
