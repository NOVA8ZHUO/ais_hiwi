# 正式训练后的自动评估与结果记录

本目录是独立的后续队列。等待既有四组×3 seed训练全部完成，校验每个第200轮权重，再依次进行12个小规模运行检查、统一网格评估和统计报告。不会重启现有训练。失败会停止并保留日志，重新启动时校验并复用已完成任务。

所有组保留完整FORGE背景随机化。测试采用直径间隙0.25、0.5、1、2 mm，XY孔位估计偏差幅值0、0.5、1、2、3、4 mm；同一绝对误差用于全部间隙。0 mm仅关闭XY偏差，保留Z误差。每格每个模型使用5个独立评估seed（600–604）×128回合，正式任务总计1440个、184320回合。另有12×4回合运行检查，不计入结论。

按评估seed预生成独立因素样本，策略动作使用单独随机流；四组所有训练seed共享配对评估样本。控制器、奖励和原FORGE成功判据保持不变。真实几何装配补充判据尚未认证，报告不将原指标解释成无穿透或实机成功证明。

## 自动生成的结果

完整评估通过后生成 `RESULTS.md`：四组成功率表、四种边际收益、交互量I及配对95%区间、逐格解释、深度/横向误差/力/曾成功未保持等辅助指标、12次训练的checkpoint与SHA256、训练日志轮次与最后best reward记录。附成功率、收益和交互热力图。训练回报与评估成功率分开呈现。

`summary.json`保留逐训练seed成功率和统计值；`results/`保存回合数据；`models.json`锁定最终模型；`integrity.json`记录完整性检查。实际结果生成前不填入预期数值。

不确定性采用5000次配对交叉分层bootstrap，共同重采样训练seed及共享评估seed/环境。仅3个训练seed、未进行逐格多重比较校正，结论需体现这些限制。负交互不等于联合组更差。

## 指令

使用Isaac Sim对应环境启动或恢复后续队列（现有训练须已启动）：

```bash
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/Pose_Uncertainty_Physical_Clearance/confirmation/start.py
```

一个终端查看训练与评估进度、评估剩余时间：

```bash
watch -n 10 /home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/Pose_Uncertainty_Physical_Clearance/confirmation/progress.py
```

详细执行日志为本目录 `queue.log`，单任务日志为对应 `console.log`。评估ETA在正式任务启动后按最近任务的实测速度估计，不将训练等待时间混入评估耗时。
