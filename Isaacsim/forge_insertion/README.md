# FORGE Peg-in-Hole 官方代码 Baseline

此目录提供独立实验入口，调用本地官方 `Isaac-Forge-PegInsert-Direct-v0` 与注册的 RL-Games PPO 配置。不复制改写训练算法、奖励、控制器或随机化。

默认源码路径：相邻的 `IsaacLab_2_3_2`。`IsaacLab` 当前是 3.0 beta，不能默认视作同一版本。可用 `--lab /absolute/path` 指定其他检出；独立 smoke/evaluate 脚本面向 2.3.2 API。

已验证可用环境为 `env_jingyun`，无需激活或安装；推荐使用 `./run.sh`（默认指向该环境）。也可激活它后执行下列 Python 命令：

```bash
cd /home/masais/Xinyu/Isaacsim/forge_insertion
python scripts/run.py inspect
python scripts/run.py smoke
python scripts/run.py short
# 前两阶段通过后执行完整训练；不覆盖官方环境数量或训练预算
python scripts/run.py train
# 将路径替换为实际生成的 checkpoint
python scripts/run.py play --checkpoint /absolute/path/model.pth --gui
python scripts/run.py evaluate --checkpoint /absolute/path/model.pth --episodes 128
```

所有命令支持 `--python /path/to/env/bin/python`、`--seed 0` 与 `--dry-run`。不自动安装、升级或删除环境。正式训练默认 headless，使用官方配置。短训练仅将环境数设为 4、迭代数设为 2；不构成最终实验。短训练的 checkpoint 是否生成及能否加载需要实际验证。

每次运行保存至 `experiments/forge_peg_baseline/logs/<UTC时间>_<阶段>/`：控制台日志、命令、返回码、源码 hash 和 FORGE/Factory 源码快照。官方训练保存 TensorBoard、checkpoint 和参数文件；脚本同时归档 checkpoint 与参数到对应目录。参数 YAML 是官方入口导出的配置，部分运行时派生值（如 actor 数）可能在导出后设置，应结合环境配置核对。

`play` 使用官方回放入口，通常持续运行，Ctrl+C 退出。`evaluate` 是独立辅助评估：沿用官方随机策略设置、官方几何成功判据；记录每回合最终成功、曾经成功、回报、首次成功时间、控制频率采样的平滑腕部力和结束标志。在自动 reset 前读取成功和力，避免将下个回合状态计入上个回合。它不代表已经确认的论文评估协议，也不测量物理子步峰值接触力。

`smoke` 使用零动作完成有限个回合，并检查观测、奖励和力是否有限。它不证明策略性能。评估和 smoke 不改变原始奖励返回值。

当前状态见 `experiments/forge_peg_baseline/results.md` 与 `pipeline_status.json`。已完成环境检查、论文/源码参数审查和四回合 smoke；后续阶段以实际运行记录为准。论文和官方代码有多项差异（位置估计噪声、摩擦随机化、dead zone 更新等），详见 `config_audit.md`，本项目保留官方实现。参考 README 的环境删除及安装命令没有执行。

可复用启动入口：

```bash
./run.sh inspect
./run.sh train --dry-run
# 正式训练正在运行时不要重复启动
# ./run.sh train
```

`train_pipeline.py` 在完整预算训练后自动加载最终 checkpoint，完成128回合辅助评估，并更新 `pipeline_status.json`、`results.md`。它不修改训练超参数；失败时保存错误并停止。该文件由后台启动流程运行，避免同时开启重复训练。

监控：

```bash
cat experiments/forge_peg_baseline/pipeline_status.json
tail -f experiments/forge_peg_baseline/pipeline.log
/home/masais/miniconda3/envs/env_jingyun/bin/python -m tensorboard.main --logdir experiments/forge_peg_baseline/logs --host 127.0.0.1
```


自有CAD独立任务已加入：`Isaac-Forge-ShaftLarge-CAD-Direct-v0`。用 `--variant cad` 选择，所有产物保存至 `experiments/forge_cad_baseline`。资产定义、17.5g标称质量与官方随机化、坐标适配及验收步骤见 [CAD任务说明](forge_cad/README.md)。原官方8mm任务仍为默认选项。

```bash
# 读取CAD训练状态（不是原8mm模型日志）
tail -f /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_cad_baseline/pipeline.log
```
