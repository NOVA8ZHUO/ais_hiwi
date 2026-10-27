# 第三步：固定现有模型的工况摸底

不训练；固定原零随机化与完整随机化两个第 200 轮 checkpoint。四档直径间隙 0.25/0.5/1/2 mm × 五档 XY 孔位估计偏差幅值 0/0.5/1/2/4 mm。每格每模型 seeds 400–404，各 128 回合：200 任务、25,600 回合。

XY 偏置方向由独立样本带中原 hole_observation 高斯 XY 向量归一化得到，每回合固定，给定半径；方向在全部间隙、误差档位和模型之间配对。Z 仍为原 σ=1 mm 高斯偏置。全部其他 17 项扰动保持完整随机化，策略仍为原随机动作采样，使用配对策略样本。0 mm 只表示没有 XY 孔位偏置。

每个任务固定一种几何，不需要混合几何 spawner；所有格子保留当前训练基线的仿真设置、奖励和成功判据。策略看不到额外 clearance 标签。名义 diameter 元数据保持基线不变，真实几何由 USD 决定。

先做两个 4 环境 smoke：零模型/1mm间隙/0偏差，以及随机化模型/0.25mm间隙/4mm偏差。检查有限值、完整回合、实际 XY 偏差半径、样本及 checkpoint 指纹。通过后自动进入全部网格。原 paired adapter 和官方任务源文件不修改。

输出：status.json、pipeline.log、每任务 console.log / initial_state.json / evaluation.json / done.json；完成后 RESULTS.md、summary.json 和（matplotlib 可用时）success_heatmaps.png。初始 ETA 约 5–6 小时是独占 GPU 的粗估；与既有评估共享 GPU 时会变慢，运行中按近期任务时间更新。失败即停止，重启可校验后复用已完成任务。

限制：尚未做四组重训消融；本实验探索适用工况，不能归因于某项训练随机化。原成功判据横向容差固定，成功率需结合几何和深度理解，不能代替连续插入与实机装配验证。训练高斯分布无硬边界，因此 4 mm 不标为严格分布外。初筛只有一个训练 seed/模型，没有跨训练 seed 推断。

```bash
cd /home/masais/Xinyu/Isaacsim/forge_insertion/Pose_Uncertainty_Physical_Clearance/condition_mapping
bash start.sh
bash progress.sh
```
