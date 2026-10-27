# model_19关闭探索评估基准

检查点：`/home/masais/Xinyu/Isaacsim/logs/rsl_rl/forge_shaft_insert_release25/2026-09-28_15-34-34_release25_gui_20260928_152608/model_19.pt`

SHA256：`071c17f0d60e4de4b22eec5d13c0e266233421e5de43786b101ad98962c7da7a`

固定策略均值动作，不训练，归一化冻结，LSTM在done时重置。任务包源码与训练快照一致；模型状态和检查点校验不变。32环境、seed42、每环境前4个完整回合，共128回合，均从完整episode开始。保留抓取误差X±8/Z±5 mm、孔估计XYZ±10 mm、刚度/摩擦/力死区及观测噪声。关闭的是策略探索，不是环境随机化。

## 结果

| 指标 | 结果 |
|---|---:|
| 完整回合 | 128 |
| 曾达到25 mm | 26/128 |
| 最终25 mm成功 | 26/128（20.3%） |
| 连续成功≥0.5秒 | 26/128 |
| 最终严格落座（46–48 mm/0.5 mm/5°） | 26/128 |
| 最终仅深度落座（46–48 mm） | 26/128 |
| 超时 | 102 |
| 越界 | 0 |
| 夹宽失败 | 0 |
| 释放观察完成 | 26 |
| 平均每回合P95接触力 | 3.761 N |
| 回合采样最大接触力 | 11.222 N |
| 平均每回合动作饱和率 | 0.03% |
| 平均回合用时（仿真时间） | 18.57 s |
| 评估墙钟时间（不含启动） | 12.98 min |

最终成功率的近似Wilson 95%区间为14.3%–28.1%，仅表示本次有限样本的统计不确定性，不涵盖种子/仿真到真机分布差异。

## 如何解释与后续比较

这是固定model_19关闭探索后的基准，和训练期间3/9（33.3%）的统计口径不同。不能把两者差异直接解释为退步，也不能仅凭本次存在成功就断言全部来自学习；要证明学习增益还需相同协议下的未训练策略或零残差对照。

200轮模型应复用本脚本、seed42、32环境、每环境4回合、相同控制/随机化/释放判据。seed和协议固定不意味着每个策略会遇到逐回合完全相同随机误差：自动reset时刻不同会改变随机数消耗。若中途改变环境/奖励/控制，必须另立版本，不能直接合并基准。

`evaluation_episodes.csv`是正式128回合分母；`episode_metrics.csv`包含快环境额外完成的回合，自动生成的通用`experiment_summary.json`按原始CSV统计，不是这份固定配额评估的权威结果。应读取`evaluation_summary.json`。原始源码、配置、命令、日志和逐步轨迹均在本目录归档。

## 复现命令

```bash
cd /home/masais/Xinyu/Isaacsim
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python FORGE_shaft_Insert/scripts/evaluate_policy.py \
  --headless --num_envs 32 --episodes_per_env 4 --seed 42 \
  --checkpoint logs/rsl_rl/forge_shaft_insert_release25/2026-09-28_15-34-34_release25_gui_20260928_152608/model_19.pt \
  --output FORGE_shaft_Insert/outputs/policy_baseline/model19_seed42_32x4_repeat
```

输出目录必须不存在，脚本拒绝覆盖已有评估。

成功回合平均用时12.95秒；终局深度范围46.984–46.999毫米。

未成功回合终局深度范围-2.296–20.007毫米。
