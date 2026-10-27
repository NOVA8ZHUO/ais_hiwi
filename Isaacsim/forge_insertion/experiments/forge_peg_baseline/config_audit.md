# FORGE 配置对照审查（2026-10-12）

## 实验定义

本次是 **Isaac Lab v2.3.2 官方代码 Baseline**，不是严格论文参数复现。使用未修改的检出 `37ddf626871758333d6ed89cf64ad702aef127d0`；任务 `Isaac-Forge-PegInsert-Direct-v0`，官方注册 `rl_games_cfg_entry_point`。正式训练只固定 seed=0，不覆盖官方环境数量、随机化或 PPO 预算。

证据：用户提供的 `/home/masais/Downloads/forge_paper.pdf`（正文 III、IV-B）、`forge_supplementary.pdf`（Table I、II、III、V）；本地官方源码及其继承调用。上游入口：https://github.com/isaac-sim/IsaacLab/tree/v2.3.2/source/isaaclab_tasks/isaaclab_tasks/direct/forge 。

下表源码路径相对于 `IsaacLab_2_3_2/source/isaaclab_tasks/isaaclab_tasks/direct/`。MATCH 仅指对应项目，不表示整个实验等同论文；数值近似但不相等标为 DIFFERENCE。

| 参数 | 论文设置 | 当前官方代码设置 | 是否一致 | 源码位置 |
|---|---|---|---|---|
| 仿真框架 | IsaacGym / Factory | Isaac Sim 5.1 / Isaac Lab 2.3.2 | DIFFERENCE | factory/factory_env.py |
| Peg 几何 | 8 mm peg | 官方 factory_peg_8mm.usd，直径7.986 mm，孔8.1 mm | MATCH（名义尺寸；网格逐点未验证） | factory/factory_tasks_cfg.py:Peg8mm, Hole8mm |
| Fixed 初始 xyz | [0.55,0.65],[-0.05,0.05],[0,0.1] m | 中心[0.6,0,0.05]，各±0.05 m | MATCH | factory/factory_tasks_cfg.py:PegInsert; factory/factory_env.py |
| Hand 相对 xyz | xy ±2 cm；z 3.7–5.7 cm | [0,0,0.047] ±[0.02,0.02,0.01] m | MATCH | factory/factory_tasks_cfg.py:PegInsert |
| Hand yaw | ±45° | ±0.785 rad | DIFFERENCE（舍入近似） | factory/factory_tasks_cfg.py:PegInsert |
| Held 相对初始位置 | x ±3 mm，y 0，z 14–20 mm | 噪声[0.003,0,0.003] m，叠加 fingerpad/grasp 坐标变换 | UNVERIFIED（需逐帧核对 z 坐标定义） | factory/factory_env.py:get_handheld_asset_relative_pose, randomize_initial_state |
| Fixed 位置估计噪声 | Gaussian σ=2.5 mm | Gaussian σ=1 mm，各轴；每回合采样 | DIFFERENCE | factory/factory_env_cfg.py:ObsRandCfg; factory/factory_env.py:648 |
| EE 位置噪声 | σ=0.25 mm | σ=0.25 mm | MATCH | forge/forge_env_cfg.py:ForgeObsRandCfg; forge/forge_env.py |
| 力观测噪声 | σ=1 N | σ=1 N；叠加到平滑的腕部力 | MATCH（噪声标准差） | forge/forge_env.py:94–114 |
| 力滤波 | 未找到数值 | smoothing factor=0.25 | UNVERIFIED | forge/forge_env_cfg.py |
| 平移控制增益 | [400,800] | 565 × multiplier，multiplier∈[1/1.41,1.41]，范围约[400.71,796.65]；非均匀采样 | DIFFERENCE（范围近似） | forge/forge_env_cfg.py:ForgeCtrlCfg; forge/forge_utils.py:get_random_prop_gains |
| 平移动作裁剪 λ | [1.6,2.5] cm | 2 cm × multiplier∈[1/1.25,1.25]，即[1.6,2.5] cm | MATCH（范围；论文分布细节未确认） | factory/factory_env_cfg.py:CtrlCfg; forge/forge_env.py:_reset_idx |
| 动作目标空间边界 | 固定物体周围各轴最多5 cm | pos_action_bounds=[0.05,0.05,0.05] | MATCH | factory/factory_env_cfg.py:CtrlCfg |
| 动作维数 | 4D 相对位姿 + 成功预测 | 7D 输出，包含 xyz、旋转和成功预测；旋转受控制器约束 | DIFFERENCE（接口维数） | forge/forge_env_cfg.py:ForgeEnvCfg; forge/forge_env.py:_apply_action |
| Dead zone 幅值 | [0,5] N | 平移各轴[0,5] N；旋转[0,1] | MATCH（平移范围） | forge/forge_env_cfg.py; forge/forge_events.py |
| Dead zone 更新 | 正文描述每回合采样 | reset 采样，另每2 s重新采样 | DIFFERENCE | forge/forge_env.py:_reset_idx; forge/forge_env_cfg.py:EventCfg |
| Force threshold | [5,10] N | [5,10] N，每回合随机 | MATCH | forge/forge_tasks_cfg.py:ForgeTask; forge/forge_env.py:_reset_idx |
| Peg 部件摩擦 | [0.5,1.0] | held静/动0.75；fixed静[0.25,1.25]、动0.25，startup采样128 buckets | DIFFERENCE | forge/forge_env_cfg.py:EventCfg |
| 质量随机化 | 正文提及，未找到范围 | peg标称0.019 kg，reset加Uniform(-0.005,0.005) kg | UNVERIFIED | forge/forge_env_cfg.py:EventCfg; factory/factory_tasks_cfg.py:Peg8mm |
| Coarse / Fine kernel | (50,2) / (100,0) | 同值 | MATCH | factory/factory_tasks_cfg.py:PegInsert |
| 完整 keypoint reward | 补充公式仅列 coarse+fine | 另含 baseline kernel(5,4) | DIFFERENCE | factory/factory_env.py:_get_factory_rew_dict |
| 接触力惩罚 β | 0.2 | 0.2 | MATCH | forge/forge_tasks_cfg.py:ForgePegInsert |
| 完整奖励其他项 | 成功预测、成功/接近奖励见正文；非所有系数明确 | action gradient 0.1、asset-relative penalty 0.001、success/engaged bonus等 | UNVERIFIED（不宣称完整 reward 一致） | forge/forge_tasks_cfg.py; forge/forge_env.py; factory/factory_env.py |
| 成功预测惩罚 | -abs(pred-label) | 同公式，但成功比例达到0.25后启用 | DIFFERENCE（启用时机未见论文明确描述） | forge/forge_env.py:_get_rewards |
| Episode / policy frequency | 150 steps / 10 s，15 Hz | dt=1/120，decimation=8，10 s，150 steps | MATCH（配置；实际终止循环边界另需观测） | factory/factory_env_cfg.py; forge/forge_env_cfg.py |
| 插入成功深度 | 距孔底1 mm以内，即孔顶下24 mm | 0.025×0.04=0.001 m，另xy<0.0025 m | MATCH（深度） | factory/factory_env.py:_get_curr_successes |
| RL 算法 | recurrent PPO，asymmetric actor-critic | RL-Games PPO + privileged central value + LSTM | MATCH（算法类别） | forge/agents/rl_games_ppo_cfg.yaml |
| 网络细节 | 所提供PDF未找到全部参数 | 两层1024 LSTM，MLP[512,128,64]，ELU | UNVERIFIED | forge/agents/rl_games_ppo_cfg.yaml |
| PPO 超参数 | 所提供PDF未找到完整表 | lr=1e-4，gamma=.995，tau=.95，clip=.2，entropy=0，horizon=128，minibatch=512，mini_epochs=4 | UNVERIFIED | forge/agents/rl_games_ppo_cfg.yaml |
| 官方训练预算 | PDF未找到对应epoch预算 | 128 envs，max_epochs=200；每epoch名义16384 transitions | UNVERIFIED（与论文比较） | factory/factory_env_cfg.py; forge/agents/rl_games_ppo_cfg.yaml |
| 归一化 | 未找到完整细节 | normalize_input/value/advantage=True，mixed_precision=True | UNVERIFIED | forge/agents/rl_games_ppo_cfg.yaml |
| 测试协议 | 正文3个训练seed；补充Fig.3每seed 128次评估 | 当前先训练seed=0；辅助评估128回合，随机策略，固定10 s期限 | DIFFERENCE（单seed基线，不是论文汇总结果） | 本项目 scripts/rollout.py |

## 处理原则

已发现的差异全部保留，不改官方奖励、随机化、控制器或预算。论文确实有多项不一致，因此不能回答“全部对齐”。若之后需要严格论文参数分支，应作为独立实验命名并单独核查，而不能覆盖本次官方代码 Baseline。
