# 当前进度与第四步训练准备

## 对照最初四步方案

| 步骤 | 当前状态 | 范围与缺口 |
|---|---|---|
| 1 明确 P 与 G | 完成 | P=回合固定 XY 孔位估计偏置；G=真实孔几何间隙。其他设置继承完整随机化 FORGE |
| 2 改几何并验证装配 | 资产预检完成，完整装配验证未完成 | 四档间隙、两种 P 共八项通过配置/reset/抓取/离散碰撞预检；没有验证完整连续插入轨迹，也没有实测加工尺寸 |
| 3 现有两模型工况摸底 | 未完成 | 尚未运行“误差幅值 × 间隙”的两模型网格评估；不能说前三步都完成，也没有据此确定最佳训练范围 |
| 4 四组正式训练 | 代码准备完成，未启动 | 四组×三个 seed；新混合几何加载尚未运行，用户启动后先过集成检查 |

四档间隙 0.25、0.5、1、2 mm 是当前明确指定的候选。直接启动第四步表示按此候选范围开展训练，不意味着第三步已完成。若要严格遵循原顺序，应先补第三步摸底；该步骤不在此次训练入口中伪装为已完成。

## 四组和固定预算

| 组 | XY 估计偏置 | 几何集合 |
|---|---|---|
| P0G0 | 0 | 1 mm |
| P1G0 | XY 各 σ=1 mm 高斯 | 1 mm |
| P0G1 | 0 | 0.25 / 0.5 / 1 / 2 mm，等权 |
| P1G1 | XY 各 σ=1 mm 高斯 | 同一等权集合 |

Z 孔位偏置原 σ=1 mm、初始位姿、质量、摩擦、控制随机化、指尖及力观测噪声、接触阈值、四元数增广均保留。P0G0 不是之前的零随机化模型。

每组 seed=0/1/2、128 环境、200 轮，从零训练；共 12 次训练。沿用原 FORGE RL-Games PPO YAML、网络、horizon、更新次数、奖励、控制器和成功判据。不存在跨模型加载已有权重。seed 顺序内依次运行四组，以减少整组训练时间跨度造成的环境差异。实际训练时长尚无此混合资产流程的实测值，不用评估耗时推算训练时长。

## G1 实现与必要的基础设施差异

用 Isaac Lab 的 MultiUsdFileCfg，以固定顺序将四个 USD 均匀分配至 128 个环境，每档 32 个。环境几何在整次训练期间固定，P0G1/P1G1 的分配相同；训练 batch 汇集不同几何，属于环境级等权几何分布训练，不是每回合重新抽几何。概率、间隙集合和名义 1 mm 不根据训练成绩调整。

四组统一通过同一个多资产生成器加载，G0 只含名义 1 mm 文件。四组统一 `scene.replicate_physics=False`，以防不同碰撞形状被复用成同一种。此为显式记录的场景构建差异，物理时间步、材料参数、控制及 PPO 不变。与历史完整随机化模型对比时应注明此差异，以本次同条件重训 P1G0 作为主要对照。

任务元数据中的孔径保持名义值，当前 FORGE 源码没有消费 `fixed_asset_cfg.diameter` 的控制/奖励逻辑；真实 clearance 来自每个 USD 碰撞网格，策略不会额外收到几何标签。

## 启动前自动关卡

`check_mixed.py` 对四组逐一启动 4 环境，检查配置差异白名单、场景中真实加载的孔径序列、reset 偏置与有限值，并进行短程保持。任何组失败均终止队列，不进行 PPO 更新。它不替代连续插入测试或第三步策略网格评估。

已执行的检查：Python 语法检查、训练队列 dry-run（12 个配置）、锁定源文件/资产指纹。未执行的检查：新多资产配置运行时关卡、正式训练。本轮遵从“先不要开启”。

## Terminal 指令

仅预览（不启动 Isaac Sim 或训练）：

```bash
cd /home/masais/Xinyu/Isaacsim/forge_insertion/Pose_Uncertainty_Physical_Clearance
/home/masais/miniconda3/envs/env_jingyun/bin/python train_queue.py
```

建议先运行混合几何集成检查（会启动仿真，但不训练）：

```bash
cd /home/masais/Xinyu/Isaacsim/forge_insertion/Pose_Uncertainty_Physical_Clearance
/home/masais/miniconda3/envs/env_jingyun/bin/python -u train_queue.py --execute --preflight-only
```

明确决定按当前四档范围训练后，后台开启完整队列（仍会重新执行必过关卡）：

```bash
cd /home/masais/Xinyu/Isaacsim/forge_insertion/Pose_Uncertainty_Physical_Clearance
nohup /home/masais/miniconda3/envs/env_jingyun/bin/python -u train_queue.py --execute > training_queue.log 2>&1 &
```

查看总进度：

```bash
tail -f /home/masais/Xinyu/Isaacsim/forge_insertion/Pose_Uncertainty_Physical_Clearance/training_queue.log
```

每次训练的详细 epoch 输出在 `training/<组名>/seed_<编号>/console.log`，例如：

```bash
tail -f /home/masais/Xinyu/Isaacsim/forge_insertion/Pose_Uncertainty_Physical_Clearance/training/P0G0/seed_0/console.log
```

结果在各自独立目录，包含官方生成的 params、TensorBoard 和 nn 权重。队列校验文件指纹并跳过已完整完成的训练；中途中断的训练不会静默覆盖或自动假装续训，需要先保留/重命名其目录，再决定从零重跑或另行配置 checkpoint 恢复。`training_manifest.json` 固定当前源代码和资产；修改后不可混用已有结果。留意先前扰动评估可能仍占用 GPU，避免同时启动长训练影响吞吐。
