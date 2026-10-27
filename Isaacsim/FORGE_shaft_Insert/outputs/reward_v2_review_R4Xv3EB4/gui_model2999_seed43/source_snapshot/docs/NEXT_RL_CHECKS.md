> 最新任务已改为 **25毫米插入即松爪**，无精密对中/倾角门槛，默认20秒、actor29/critic53。当前定义、命令和结果见 [INSERTION_25MM.md](INSERTION_25MM.md)；下文旧判据与失败结果保留作历史记录。

> 当前控制修复与验收以 [CONTROLLER_REPAIR.md](CONTROLLER_REPAIR.md) 为准。下文旧测试结果和旧配置保留作历史记录；不要将旧成功率当作当前版本结果。

# 下一步：场景检查、可视化短训练、200轮pilot

## 先回答成功率和论文设置

此前的hold测试没有执行插入，不能作为插入成功率测试；random32日志成功步为0，两次短PPO也没有inserted奖励，因此没有已验证的成功插入案例。旧smoke统计在自动reset之后读取，可能漏掉结束瞬间，且没有逐回合记录，不能严谨地给出“训练策略成功率=0/N”。新统计从reset前读取，今后直接用CSV的完整回合数量作分母。

论文正文II-B式(2)、III-A、III-C式(7)列出关键点、place/success奖励、超力惩罚、成功预测误差；补充材料Table I列出核函数和超力系数，没有列出独立失败、时间、动作抖动或掉落惩罚。当前没有额外添加这些项，是为了保留该奖励结构，不代表这些惩罚绝对没用。加入前应验证是否确有相应问题，并单独做对照，否则可能改变论文任务目标或让策略靠不动来规避惩罚。

论文III-C Early Termination明确说：**“During training, episodes are executed for the maximum length.”** 随后区分部署时按成功预测阈值提前停止。因此当前训练成功后继续运行有直接论文依据；持续给成功奖励也使保持成功状态有回报。当前工程额外增加了越界/夹宽失败终止，并非论文所有回合都严格完整执行的复刻。部署成功预测提前停止还未实现，不能把“训练不提前结束”理解成“真机也必须一直执行”。

原文：[正文PDF](/home/masais/Downloads/forge_paper.pdf)，[补充材料PDF](/home/masais/Downloads/forge_supplementary.pdf)。

## 新加入的指标定义

训练和测试都使用 `ForgeMetricsEnv`，在自动reset**之前**采集最后一帧；手动reset未完成回合不会记成失败。每回合一行CSV，只有完成回合进入成功率分母；最终还没结束的回合不进入分母。

| CSV字段 | 定义 |
|---|---|
| `success_any` | 该回合至少一次满足现有几何成功条件 |
| `success_final` | 结束前最后控制步满足成功条件 |
| `success_stable` | 回合中曾连续成功≥0.5 s；15 Hz下至少8步，即0.533 s；不是新termination |
| `longest_success_s` | 最长连续成功时长 |
| `final_success_streak_s` | 结束时连续成功时长，可区别成功后又退出 |
| `force_p95_N` | 本回合所有15 Hz控制步上，无额外观测噪声的轴–Base合力模长的95百分位，包含最后一步 |
| `force_peak_N` | 上述样本的最大值；不是240 Hz接触尖峰峰值 |
| `action_saturation` | 允许执行下插的步内，前4维运动动作中abs(a)≥0.99的分量比例；排除第5维成功预测和等待阶段 |
| `active_action_steps` | 动作饱和率对应的有效控制步数；0时饱和率记录0，解释时需同时看此列 |
| `steps/duration_s` | 实际已执行控制步及其时长；不采用随机初始化的框架回合计数 |
| 三项termination列 | 最后一步各终止条件是否触发；可以重叠 |

接触力不是指尖夹紧力，也不是完整六维末端力；控制频率采样可能漏掉两个控制步之间的尖峰。P95包含无接触步，接触很短的回合P95可能为0，应一起看峰值。
饱和率测的是策略命令接近裁剪边界，不是位置目标λ裁剪比例或关节力矩饱和比例。
这些成功标签仍采用原来的几何定义，没有额外要求轴必须被夹住；0.5 s是诊断默认值，不是论文参数。

TensorBoard新增 `Metrics_cumulative/*`：自本次进程启动以来完成回合的累计平均值和完成回合数。`force_p95_N`曲线为**各回合P95的平均**，不是把所有接触样本混合后的P95。曲线随训练累计可能反应慢；请从CSV抽取最近100回合看近期趋势。续训新进程重新开始累计，CSV也保存在新的run目录，不跨运行混合。
训练轨迹带探索噪声，训练成功率不是独立确定性评估成功率。确定性播放可通过 `env.task.metrics_csv_path=...` 单独输出评估CSV。

## 还差什么才适合正式长训

PPO、奖励、随机化、reset、观测、动作和termination已经配置，可以运行。还缺的是验证证据，而不是必须再添加一串奖励：

1. 无误差零残差下插是否能到达目标，接触和夹持是否合理；若无误差基准也明显失败，先定位，不直接跑数小时。
2. 带误差时的失败形态是否符合预期：轴是否还受夹爪控制，是否发生不合理穿透/飞出、真实Base是否保持固定。
3. 可视化短训练是否数值稳定、动作是否异常；20轮不是收敛验收。
4. 用新CSV核对成功率、连续成功时长与接触力，而非只看reward。

精确的真机20 N到仿真驱动力映射尚未标定，这是sim-to-real待校准项，不是无法开始pilot的技术阻塞。

## 顺序执行命令

在同一个终端先设置路径。不要同时启动多个GUI或大规模仿真；上一条进程结束后再运行下一条。以下CSV路径会被覆盖，重测需改文件名以保留旧数据。

```bash
cd /home/masais/Xinyu/Isaacsim
FORGE_PY=/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python
```

### 第一步：4环境可视化场景

下面没有`--headless`，会打开GUI。

```bash
# 1A 保持：看夹紧后持续施力、Base固定、自然偏移；本项不预期插入成功
"$FORGE_PY" FORGE_shaft_Insert/scripts/paper_smoke_test.py --mode hold --num_envs 4 --steps 160 --metrics_csv FORGE_shaft_Insert/outputs/scene_checks/hold.csv

# 1B 零误差下插：先隔离抓取/孔误差和力死区；仍有Kp/摩擦随机化
"$FORGE_PY" FORGE_shaft_Insert/scripts/paper_smoke_test.py --mode directed --nominal --num_envs 4 --steps 160 --metrics_csv FORGE_shaft_Insert/outputs/scene_checks/nominal.csv

# 1C 带误差下插：保留用户全部误差，零残差；它不是已训练策略
"$FORGE_PY" FORGE_shaft_Insert/scripts/paper_smoke_test.py --mode directed --num_envs 4 --steps 160 --metrics_csv FORGE_shaft_Insert/outputs/scene_checks/errors.csv

# 检查完成回合数与三种成功率，其他CSV同理
"$FORGE_PY" FORGE_shaft_Insert/scripts/summarize_episode_metrics.py FORGE_shaft_Insert/outputs/scene_checks/nominal.csv
```

160步覆盖一次正常150步结束，并留10步验证自动reset。若失败提前reset，完成回合数可能多于4。窗口快慢取决于机器实际速度，不保证按现实10秒运行完。
用零残差检查时，预测动作固定−1，不影响机械运动；持续夹紧与训练相同。hold额外关闭力死区并暂停下插，不能用它替代真实下插检查。
`RESIDUAL_TEST_PASS`仅表明接口断言通过；必须查看`EPISODE_METRICS`和画面判断下插。

### 第二步：4环境可视化训练20轮

```bash
"$FORGE_PY" FORGE_shaft_Insert/scripts/train.py --task FORGE-Shaft-Insert-Paper-v0 --num_envs 4 --max_iterations 20 --run_name visual_debug_metrics agent.save_interval=10 agent.algorithm.num_mini_batches=2
```

环境数4、rollout32、2个mini-batch，保持当前PPO起点。观察抖动、越界、脱手和数值异常。探索期随机动作较大，不能因为早期失败就判定PPO无效；但明显几何/控制错误需要先修。
运行目录为 `logs/rsl_rl/forge_shaft_insert_residual/<时间>_visual_debug_metrics/`，其中自动包含 `episode_metrics.csv`、TensorBoard与checkpoint。CSV每个完成回合立即刷新。

### 第三步：128环境headless、200轮pilot

确认场景合理后再执行。本命令从新模型开始，不自动续训4环境debug模型。

```bash
"$FORGE_PY" FORGE_shaft_Insert/scripts/train.py --task FORGE-Shaft-Insert-Paper-v0 --headless --num_envs 128 --max_iterations 200 --run_name pilot_metrics agent.num_steps_per_env=32 agent.algorithm.num_mini_batches=4
```

约819,200新样本。按此前本机128环境测速，预算2–3小时；新增统计有额外开销，正式耗时以本次日志为准。GUI两阶段暂预算10–25分钟与15–30分钟，尚无GUI实测。

独立终端查看曲线：

```bash
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/tensorboard --logdir /home/masais/Xinyu/Isaacsim/logs/rsl_rl/forge_shaft_insert_residual
```

训练后用实际检查点路径替换占位符播放，输出与训练分开的CSV：

```bash
"$FORGE_PY" FORGE_shaft_Insert/scripts/play.py --task FORGE-Shaft-Insert-Paper-v0 --num_envs 4 --checkpoint /实际绝对路径/model_实际编号.pt env.task.metrics_csv_path=FORGE_shaft_Insert/outputs/scene_checks/policy_eval.csv
```

播放本身不会以100回合自动退出；可在另一个终端运行汇总脚本查看完成数，至少100回合后停止播放器，未完成回合不会计入。不同种子重复评估时另存CSV，避免覆盖。
先看稳定成功率与终局成功率，再看力P95/峰值与运动饱和率；不要只凭累计reward决定是否上真机。

## 本次新增验证结果

- 无抓取/孔误差零残差下插：4环境×160步，完成4回合；曾几何成功3、终局几何成功3、曾连续成功≥0.5秒2、终局成功且无失败1。
- 带完整用户误差零残差下插：4环境×160步，完成4回合；曾几何成功1、终局几何成功1、曾连续成功≥0.5秒1、终局成功且无失败1。
- 无误差基准的3个几何成功回合中有2个同时触发夹宽失败；不能把75%几何成功率称为稳定装配成功率。4个样本也不足以估计泛化成功率。
- CPU回归检查通过：成功连续性、终止样本、P95、第五维动作排除、无效动作阶段排除和手动reset不计失败。
- 4环境2轮PPO集成检查通过，验证训练CSV与TensorBoard指标链路；不是20轮GUI训练，也不是200轮pilot。
- 本次场景验证均为headless，GUI三阶段命令需按前文运行；没有替用户启动数小时pilot。

原始日志和CSV见 [metrics_validation](../outputs/metrics_validation/)。


### 20轮GUI训练后的配置修正

本机RSL-RL循环PPO在每mini-batch只有1个环境时，使用无维度参数的squeeze误删环境轴，导致概率比广播错误。4环境/4组不适用；之前的该项建议已更正为**4环境/2组**，128环境/4组保持不变。训练入口已检查整除及每组至少2环境，不修改安装库。
`2026-09-28_13-28-16_visual_debug_metrics`受此问题影响，不建议续训该检查点或用它判断PPO能否收敛。它的环境轨迹统计仍有效：17个完成回合，0成功、16次超时、1次越界、0夹宽失败；策略更新受到错误概率比影响。
