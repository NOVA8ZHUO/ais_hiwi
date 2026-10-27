# 停训保存与复现说明

最新完整权重：latest/model_5140.pt。保留最佳基准：best820/model_820.pt。
两者均保存完整检查点（actor/critic、优化器、iter以及state_dict中的归一化状态），并附带params、source_snapshot、原始experiment_manifest和权重SHA256。
原训练进程已停止。5140之后未写盘的更新无法恢复。原日志及全部历史检查点仍位于原训练目录，见stop_record.json。manifest为训练来源证据，不重写为新的训练成功记录；停止状态单独记录。

best820/eval_seed42、43、44保留每环境前4回合评估摘要和明细，原基准241/384成功（62.76%），240/384严格落座。对应IsaacLab 2.3.2、原Xinyu_isaacsim Python环境；当前包版本见pip_freeze.txt。硬件、PhysX/GPU运行差异可能影响数值，不保证逐回合或位级完全一致。

复现820评估示例（从工作区运行，将输出目录改为新的未存在目录，三个seed分别执行）：

```bash
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/evaluate_policy.py --headless --num_envs 32 --episodes_per_env 4 --seed 42 --checkpoint CHECKPOINT --output NEW_OUTPUT
```

CHECKPOINT使用本目录best820/model_820.pt的绝对路径，seed依次42/43/44。评估脚本会核对当前任务源码与历史快照；源码已改时应使用隔离副本恢复source_snapshot及对应运行依赖，不能绕过哈希检查。不要在正在运行的其他实验目录上直接覆盖历史代码。

每个模型source_snapshot/assets包含资产配置。其外部引用仍需要原工作区和Isaac安装资源；此备份不是完全自包含的系统镜像。恢复训练需对应配置、权重和优化器；RSL检查点不含仿真接触状态或完整随机数状态，不能逐步重放原训练。
