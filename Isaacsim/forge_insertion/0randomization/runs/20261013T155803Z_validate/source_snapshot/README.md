# 自有 CAD FORGE：零环境随机化消融

参照实验：`../experiments/forge_cad_baseline/logs/20261013T115201_931054Z_train`。
本目录独立注册 `Isaac-Forge-ShaftLarge-ZeroRand-Direct-v0`，日志和模型只写入本目录 `runs/`。
未启动完整训练。原训练、官方源代码和原 CAD 任务均不修改。

## 启动

```bash
cd /home/masais/Xinyu/Isaacsim/forge_insertion/0randomization
./run.sh train
```

每次启动创建新的带时间戳目录，不加载 checkpoint。从该终端直接读取进度。
只查看命令：`./run.sh train --dry-run`。
配置审计：`./run.sh audit`。物理短验证（4 个环境，非训练）：`./run.sh validate`。
启动器校验 `reference.json` 中官方实现、原 CAD 和 PPO 的 SHA256；发生变化会拒绝运行，避免静默偏离参照实验。

## 与参照训练相同

- Isaac Lab 2.3.2，同一 Python/Isaac Sim 环境；shaft_large 插 base，CAD 文件逐字复制。
- 128 环境，seed 0，200 epochs，从零训练；horizon 128，minibatch 512，mini_epochs 4。
- 同一官方 PPO YAML：学习率 1e-4，gamma 0.995，tau 0.95，LSTM/MLP、critic、归一化、动作探索均保留。
- 仿真 120 Hz、decimation 8、10 s 回合；控制器、奖励公式及权重、成功判据、抓取/复位流程不变。
- 重力和固定约束不变：base 固定于世界，shaft 的 disable_gravity=True。不会额外贴桌或加入松爪流程。

## 关闭的环境随机化及固定值

| 项目 | 零随机化处理 |
|---|---|
| base 位置及朝向扰动 | 0；保留标称 root=(0.6,0,0.05)，yaw=0 |
| 夹爪初始位置/姿态扰动 | 0；保留原标称姿态和高度 |
| shaft 在夹爪内位置扰动 | 0 |
| 孔位置、指尖位置/旋转、力传感观测噪声 | 0 |
| shaft 质量扰动 | 加性范围 [0,0]；质量恒为 0.0175 kg |
| base 静摩擦 | [0.25,1.25] 固定为中点 0.75；动摩擦仍为 0.25 |
| shaft/机器人摩擦、恢复系数 | 原本就是固定值，保留 |
| 比例增益、动作位置/旋转阈值扰动 | 0；使用原 default 值 |
| EMA | [0.025,0.1] 固定为中点 0.0625 |
| 接触惩罚力阈值 | [5,10] 固定为中点 7.5 N；奖励公式/权重不变 |
| 死区 | [0,default] 固定为中点 [2.5,2.5,2.5,0.5,0.5,0.5]；关闭 interval 随机事件 |
| 四元数随机正负号增广 | 固定 +1；物理旋转不变 |

“零随机化”指关闭环境域随机化、reset 扰动和观测增广，并非把所有物理参数设成零。
有标称值的参数保留标称值；仅给随机范围的参数使用中点，这些固定值是明确的消融选择，并非官方另外提供的零随机化 preset。
PPO 随机动作、网络初始化和 minibatch shuffle 保留；不能据此承诺 GPU 仿真逐位确定性。

`forge_zero/env.py` 只覆盖 reset 后两个硬编码随机量（死区和四元数符号），其余通过配置关闭；不复制/修改控制器、奖励或 PPO。
官方代码仍可能生成最后乘以零的随机数，但它们不产生环境扰动。

## 验证与结果解释

`verify.py` 对所有配置差异实施精确白名单，并与参照实验保存的 agent.yaml 比较 PPO。
物理验证执行两次 reset 和各 35 步 hold（超过 2 秒 interval），检查有限观测、抓取宽度、标称 base 位置、质量、控制参数、无观测噪声和重复 reset 在官方 IK 的 1 mm 位置容差内一致性。
结果位于 `runs/*_validate/passed.json`；失败会记录 failure.txt 并返回非零状态。
完整训练完成后仍需单独评估；本版本的固定场景成功率不能代表随机化场景的泛化成功率，也不能沿用原模型成功率。

验证说明：最初使用的 0.1 mm reset 检查阈值失败，测得差异约 0.151 mm。官方 IK 本身使用 1 mm 位置容差；最终验证按该容差检查，未改动 IK 或物理设置。无环境随机化并不表示接触求解后的状态逐位相同。
