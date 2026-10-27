# 从model_19新增200轮pilot

## 启动

```bash
cd /home/masais/Xinyu/Isaacsim
PYTHONUNBUFFERED=1 /home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/run_pilot200.py
```

在有GPU的本机终端运行，保持终端会话打开。脚本不启动GUI。准备本流程时仅运行了CPU检查和`--dry-run`，未启动长训练。

## 固定协议

- 从`2026-09-28_15-34-34_release25_gui_20260928_152608/model_19.pt`恢复策略和优化器，新增200次PPO更新，不是从零训练总计200轮。
- 训练128环境，rollout32，每轮4096样本，4个mini-batch，每组1024 rollout样本（循环网络实际按序列分组）；新增总计819200环境样本。训练seed42。
- 只改变训练规模（原4环境/2mini-batch→128环境/4mini-batch）和保存频率；物理、控制、奖励、成功/释放判据与随机化不变。脚本启动前校验任务包源码和初始检查点与基准一致。
- 每10轮保存一次。当前RSL-RL从存储iter=19开始循环，所以新增约52、102、200轮分别对应`model_70.pt`、`model_120.pt`、`model_218.pt`。报告标注真实新增轮次，不把model_200误认为新增200轮。
- 完成200轮训练后再串行评估三个检查点。每次32环境×每环境前4完整回合=128回合，seed42，关闭动作探索，保留环境随机化，完整回合起始，done时清空LSTM。与model_19基准20.3125%使用相同协议。
- 不运行训练/评估并行，不自动训练到1000轮。

## 输出与决定

训练目录：`logs/rsl_rl/forge_shaft_insert_release25/<时间>_pilot200_from_model19_.../`。

编排目录：`FORGE_shaft_Insert/outputs/pilot200/pilot200_from_model19_.../`。其中保存`pilot_plan.json`、训练/评估完整输出、三个评估目录、`comparison.json`和`COMPARISON.md`。

对照表包含独立成功率、严格落座率、超时、失败终局深度2–25 mm的浅插数量、每回合P95接触力均值、接触峰值、夹持阶段最大倾角、物理步TCP最大速度、每回合平均动作饱和率。浅插数量是终局分类，不等同于已证明的机械卡死原因。

任务的成功仍为目标孔区域内深度≥25 mm即松爪，精密对中/倾角仅用于独立落座诊断。比较最终成功率和物理指标后再决定是否继续。脚本不会凭reward上涨或某个单一阈值自动授权后续训练；小幅成功率差异可能是抽样波动，各评估目录保存置信区间。

预算约3–4小时（训练暂估2–3小时，三次评估约40–60分钟），训练以128环境实际前10–20轮速度修正；不是耗时保证。

## 中断与检查

子进程失败时整个流程停止并在`pilot_plan.json`记录错误，保留日志/检查点，不会跳过失败继续报喜。Ctrl+C停止当前子进程，不会自动进入下一阶段。

如果训练已完整结束但评估中断，可使用新的编排目录重新执行三次评估，避免重跑训练：

```bash
PYTHONUNBUFFERED=1 /home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/run_pilot200.py \
  --training-run /实际完整路径/已完成的pilot训练目录
```

这会重新评估三个检查点，不覆盖旧评估目录。未完成的训练不要用该选项伪装为完成。

只查看命令、不启动训练：

```bash
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/run_pilot200.py --dry-run
```

验证：`scripts/test_pilot200.py`检查续训编号、128/4训练与32×4评估命令、固定配额轨迹统计排除额外回合；dry-run已通过。完整200轮流程待用户执行。
