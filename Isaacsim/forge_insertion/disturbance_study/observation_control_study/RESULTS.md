# 观测与控制单项补充实验：最终结果

仅在全部 480 个任务完成并通过配对完整性检查后生成。实验方案见 [EXPERIMENT.md](EXPERIMENT.md)。

所有差值单位均为百分点。初筛与确认分开报告，不合并后再选择结论。95% 区间为配对分层 bootstrap；确认阶段九项主要比较另列 Bonferroni 校正区间（近似家族置信度 95%）。

## 初筛：seeds [200, 201, 202, 203, 204]

| 条件 | 零随机化成功率 % | 随机化成功率 % | 提升 pp [95% CI] |
|---|---:|---:|---|
| zero | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| full | 24.53 | 98.44 | 73.91 [67.19, 79.53] |
| group_on_observation | 87.50 | 100.00 | 12.50 [9.38, 15.94] |
| group_off_observation | 25.00 | 99.22 | 74.22 [70.47, 77.81] |
| group_on_control | 84.06 | 100.00 | 15.94 [11.09, 21.56] |
| group_off_control | 21.88 | 99.53 | 77.66 [73.44, 81.72] |
| factor_on_hole_observation | 93.28 | 100.00 | 6.72 [4.69, 9.06] |
| factor_off_hole_observation | 24.38 | 98.91 | 74.53 [68.44, 79.84] |
| factor_on_tip_position_noise | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| factor_off_tip_position_noise | 23.12 | 98.59 | 75.47 [69.53, 80.47] |
| factor_on_tip_rotation_noise | 94.22 | 100.00 | 5.78 [3.91, 7.97] |
| factor_off_tip_rotation_noise | 22.81 | 99.06 | 76.25 [72.03, 80.47] |
| factor_on_force_noise | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| factor_off_force_noise | 24.53 | 98.91 | 74.38 [70.00, 78.59] |
| factor_on_gains | 99.38 | 100.00 | 0.62 [0.00, 1.41] |
| factor_off_gains | 24.38 | 99.22 | 74.84 [70.62, 78.91] |
| factor_on_ema | 99.84 | 100.00 | 0.16 [0.00, 0.78] |
| factor_off_ema | 24.06 | 99.22 | 75.16 [70.00, 80.00] |
| factor_on_position_threshold | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| factor_off_position_threshold | 22.97 | 98.75 | 75.78 [70.78, 80.47] |
| factor_on_rotation_threshold | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| factor_off_rotation_threshold | 23.91 | 99.06 | 75.16 [69.69, 79.54] |
| factor_on_dead_zone | 83.28 | 100.00 | 16.72 [13.44, 20.16] |
| factor_off_dead_zone | 21.25 | 99.06 | 77.81 [72.81, 82.19] |

### 单项敏感性及抗扰改善

D=无扰动成功率−单独开启成功率；R=单独关闭成功率−完整随机化成功率；T=(D+R)/2。主要改善量 ΔD=D零−D随机，正值表示随机化模型对该扰动的开启损失较小。T 不是失败贡献占比。

| 单项 | D零 | D随机 | R零 | R随机 | ΔD [95% CI] | ΔD 校正区间（仅确认） |
|---|---:|---:|---:|---:|---|---|
| 孔位置观测误差 | 6.72 | 0.00 | -0.16 | 0.47 | 6.72 [4.53, 9.06] | — |
| 指尖位置观测噪声 | 0.00 | 0.00 | -1.41 | 0.16 | 0.00 [0.00, 0.00] | — |
| 指尖旋转观测噪声 | 5.78 | 0.00 | -1.72 | 0.62 | 5.78 [3.91, 7.81] | — |
| 力传感噪声 | 0.00 | 0.00 | 0.00 | 0.47 | 0.00 [0.00, 0.00] | — |
| 控制增益 | 0.62 | 0.00 | -0.16 | 0.78 | 0.62 [0.00, 1.41] | — |
| EMA 平滑系数 | 0.16 | 0.00 | -0.47 | 0.78 | 0.16 [0.00, 0.62] | — |
| 位置动作阈值 | 0.00 | 0.00 | -1.56 | 0.31 | 0.00 [0.00, 0.00] | — |
| 旋转动作阈值 | 0.00 | 0.00 | -0.62 | 0.62 | 0.00 [0.00, 0.00] | — |
| 死区阈值 | 16.72 | 0.00 | -3.28 | 0.62 | 16.72 [13.28, 20.16] | — |

### 辅助指标

| 条件 / 模型 | 曾成功未保持（回合） | 最终深度 mm | 最终横向误差 mm | 平均腕力 N | 回合峰值力 P95 N |
|---|---:|---:|---:|---:|---:|
| zero / zero | 0 | 46.72 | 0.31 | 4.46 | 12.44 |
| zero / randomized | 0 | 46.75 | 0.35 | 2.94 | 14.03 |
| full / zero | 19 | 21.49 | 2.99 | 6.21 | 18.47 |
| full / randomized | 3 | 46.48 | 0.44 | 4.00 | 18.18 |
| group_on_observation / zero | 6 | 43.64 | 0.41 | 4.52 | 15.89 |
| group_on_observation / randomized | 0 | 46.77 | 0.35 | 3.08 | 14.65 |
| group_off_observation / zero | 4 | 20.31 | 3.37 | 6.11 | 17.88 |
| group_off_observation / randomized | 1 | 46.55 | 0.43 | 3.85 | 18.34 |
| group_on_control / zero | 1 | 42.40 | 0.50 | 4.54 | 18.06 |
| group_on_control / randomized | 0 | 46.76 | 0.41 | 3.17 | 17.38 |
| group_off_control / zero | 24 | 21.19 | 2.84 | 6.04 | 16.42 |
| group_off_control / randomized | 0 | 46.62 | 0.40 | 3.84 | 16.63 |
| factor_on_hole_observation / zero | 2 | 44.79 | 0.40 | 4.54 | 14.81 |
| factor_on_hole_observation / randomized | 0 | 46.76 | 0.35 | 3.00 | 14.34 |
| factor_off_hole_observation / zero | 16 | 21.59 | 2.80 | 6.20 | 18.23 |
| factor_off_hole_observation / randomized | 4 | 46.65 | 0.43 | 3.89 | 17.87 |
| factor_on_tip_position_noise / zero | 0 | 46.71 | 0.33 | 4.48 | 12.89 |
| factor_on_tip_position_noise / randomized | 0 | 46.76 | 0.34 | 2.97 | 13.97 |
| factor_off_tip_position_noise / zero | 23 | 21.03 | 3.21 | 6.20 | 18.54 |
| factor_off_tip_position_noise / randomized | 4 | 46.54 | 0.44 | 3.96 | 17.87 |
| factor_on_tip_rotation_noise / zero | 2 | 45.52 | 0.32 | 4.47 | 15.84 |
| factor_on_tip_rotation_noise / randomized | 0 | 46.76 | 0.35 | 2.96 | 14.30 |
| factor_off_tip_rotation_noise / zero | 4 | 19.48 | 3.50 | 6.19 | 17.98 |
| factor_off_tip_rotation_noise / randomized | 3 | 46.57 | 0.43 | 4.00 | 18.36 |
| factor_on_force_noise / zero | 0 | 46.71 | 0.32 | 4.49 | 12.87 |
| factor_on_force_noise / randomized | 0 | 46.76 | 0.36 | 2.97 | 14.32 |
| factor_off_force_noise / zero | 15 | 21.40 | 3.12 | 6.23 | 18.65 |
| factor_off_force_noise / randomized | 0 | 46.47 | 0.46 | 3.98 | 17.98 |
| factor_on_gains / zero | 0 | 46.55 | 0.35 | 4.50 | 15.47 |
| factor_on_gains / randomized | 0 | 46.76 | 0.37 | 3.00 | 14.68 |
| factor_off_gains / zero | 25 | 21.30 | 3.10 | 6.19 | 17.94 |
| factor_off_gains / randomized | 1 | 46.52 | 0.43 | 3.99 | 18.14 |
| factor_on_ema / zero | 1 | 46.70 | 0.29 | 4.42 | 17.67 |
| factor_on_ema / randomized | 0 | 46.76 | 0.35 | 2.94 | 15.76 |
| factor_off_ema / zero | 16 | 21.59 | 3.01 | 6.21 | 18.04 |
| factor_off_ema / randomized | 2 | 46.53 | 0.47 | 4.05 | 16.93 |
| factor_on_position_threshold / zero | 0 | 46.72 | 0.30 | 4.46 | 12.37 |
| factor_on_position_threshold / randomized | 0 | 46.76 | 0.34 | 2.94 | 13.84 |
| factor_off_position_threshold / zero | 18 | 20.82 | 3.02 | 6.24 | 18.08 |
| factor_off_position_threshold / randomized | 3 | 46.45 | 0.45 | 4.00 | 18.21 |
| factor_on_rotation_threshold / zero | 0 | 46.72 | 0.30 | 4.46 | 12.56 |
| factor_on_rotation_threshold / randomized | 0 | 46.75 | 0.35 | 2.95 | 14.04 |
| factor_off_rotation_threshold / zero | 23 | 21.19 | 3.18 | 6.23 | 18.66 |
| factor_off_rotation_threshold / randomized | 0 | 46.41 | 0.46 | 4.00 | 17.84 |
| factor_on_dead_zone / zero | 0 | 42.06 | 0.61 | 4.64 | 16.89 |
| factor_on_dead_zone / randomized | 0 | 46.76 | 0.43 | 3.18 | 15.21 |
| factor_off_dead_zone / zero | 21 | 20.52 | 3.08 | 6.11 | 18.28 |
| factor_off_dead_zone / randomized | 1 | 46.52 | 0.42 | 3.83 | 17.84 |
## 独立确认：seeds [300, 301, 302, 303, 304]

| 条件 | 零随机化成功率 % | 随机化成功率 % | 提升 pp [95% CI] |
|---|---:|---:|---|
| zero | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| full | 25.16 | 99.06 | 73.91 [70.00, 77.50] |
| group_on_observation | 87.81 | 100.00 | 12.19 [9.22, 15.31] |
| group_off_observation | 25.31 | 99.69 | 74.38 [69.69, 78.91] |
| group_on_control | 82.19 | 100.00 | 17.81 [14.84, 20.94] |
| group_off_control | 20.31 | 99.84 | 79.53 [75.62, 83.44] |
| factor_on_hole_observation | 94.69 | 99.84 | 5.16 [3.28, 7.34] |
| factor_off_hole_observation | 24.22 | 99.69 | 75.47 [71.88, 79.06] |
| factor_on_tip_position_noise | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| factor_off_tip_position_noise | 23.59 | 99.69 | 76.09 [72.19, 79.85] |
| factor_on_tip_rotation_noise | 93.59 | 100.00 | 6.41 [4.06, 9.22] |
| factor_off_tip_rotation_noise | 24.38 | 99.38 | 75.00 [70.00, 80.31] |
| factor_on_force_noise | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| factor_off_force_noise | 24.53 | 99.06 | 74.53 [70.94, 78.13] |
| factor_on_gains | 99.38 | 100.00 | 0.62 [0.00, 1.56] |
| factor_off_gains | 24.22 | 99.53 | 75.31 [71.09, 79.38] |
| factor_on_ema | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| factor_off_ema | 25.47 | 99.38 | 73.91 [69.53, 78.44] |
| factor_on_position_threshold | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| factor_off_position_threshold | 24.69 | 99.53 | 74.84 [70.47, 78.91] |
| factor_on_rotation_threshold | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| factor_off_rotation_threshold | 25.47 | 99.38 | 73.91 [70.00, 77.81] |
| factor_on_dead_zone | 85.00 | 100.00 | 15.00 [11.72, 18.29] |
| factor_off_dead_zone | 25.00 | 99.38 | 74.38 [70.78, 78.12] |

### 单项敏感性及抗扰改善

D=无扰动成功率−单独开启成功率；R=单独关闭成功率−完整随机化成功率；T=(D+R)/2。主要改善量 ΔD=D零−D随机，正值表示随机化模型对该扰动的开启损失较小。T 不是失败贡献占比。

| 单项 | D零 | D随机 | R零 | R随机 | ΔD [95% CI] | ΔD 校正区间（仅确认） |
|---|---:|---:|---:|---:|---|---|
| 孔位置观测误差 | 5.31 | 0.16 | -0.94 | 0.62 | 5.16 [3.12, 7.34] | [2.50, 8.28] |
| 指尖位置观测噪声 | 0.00 | 0.00 | -1.56 | 0.62 | 0.00 [0.00, 0.00] | [0.00, 0.00] |
| 指尖旋转观测噪声 | 6.41 | 0.00 | -0.78 | 0.31 | 6.41 [4.06, 9.22] | [3.28, 10.62] |
| 力传感噪声 | 0.00 | 0.00 | -0.62 | 0.00 | 0.00 [0.00, 0.00] | [0.00, 0.00] |
| 控制增益 | 0.62 | 0.00 | -0.94 | 0.47 | 0.62 [0.00, 1.56] | [0.00, 2.03] |
| EMA 平滑系数 | 0.00 | 0.00 | 0.31 | 0.31 | 0.00 [0.00, 0.00] | [0.00, 0.00] |
| 位置动作阈值 | 0.00 | 0.00 | -0.47 | 0.47 | 0.00 [0.00, 0.00] | [0.00, 0.00] |
| 旋转动作阈值 | 0.00 | 0.00 | 0.31 | 0.31 | 0.00 [0.00, 0.00] | [0.00, 0.00] |
| 死区阈值 | 15.00 | 0.00 | -0.16 | 0.31 | 15.00 [11.88, 18.44] | [10.78, 20.00] |

### 辅助指标

| 条件 / 模型 | 曾成功未保持（回合） | 最终深度 mm | 最终横向误差 mm | 平均腕力 N | 回合峰值力 P95 N |
|---|---:|---:|---:|---:|---:|
| zero / zero | 0 | 46.72 | 0.31 | 4.47 | 12.50 |
| zero / randomized | 0 | 46.75 | 0.35 | 2.94 | 14.41 |
| full / zero | 17 | 21.13 | 3.01 | 6.30 | 18.44 |
| full / randomized | 0 | 46.47 | 0.46 | 3.92 | 17.66 |
| group_on_observation / zero | 3 | 43.89 | 0.36 | 4.48 | 15.94 |
| group_on_observation / randomized | 0 | 46.77 | 0.35 | 3.08 | 14.73 |
| group_off_observation / zero | 7 | 20.36 | 3.40 | 6.04 | 18.40 |
| group_off_observation / randomized | 0 | 46.62 | 0.42 | 3.75 | 17.68 |
| group_on_control / zero | 1 | 41.85 | 0.50 | 4.50 | 17.68 |
| group_on_control / randomized | 0 | 46.76 | 0.42 | 3.13 | 17.02 |
| group_off_control / zero | 15 | 20.73 | 2.90 | 6.02 | 16.53 |
| group_off_control / randomized | 0 | 46.71 | 0.38 | 3.79 | 17.08 |
| factor_on_hole_observation / zero | 0 | 45.19 | 0.41 | 4.55 | 14.54 |
| factor_on_hole_observation / randomized | 0 | 46.68 | 0.36 | 3.01 | 14.29 |
| factor_off_hole_observation / zero | 19 | 21.21 | 2.91 | 6.25 | 18.74 |
| factor_off_hole_observation / randomized | 0 | 46.70 | 0.42 | 3.83 | 18.06 |
| factor_on_tip_position_noise / zero | 0 | 46.71 | 0.33 | 4.49 | 12.79 |
| factor_on_tip_position_noise / randomized | 0 | 46.76 | 0.34 | 2.98 | 14.19 |
| factor_off_tip_position_noise / zero | 21 | 20.68 | 3.02 | 6.26 | 18.33 |
| factor_off_tip_position_noise / randomized | 0 | 46.61 | 0.45 | 3.89 | 18.15 |
| factor_on_tip_rotation_noise / zero | 3 | 45.32 | 0.32 | 4.45 | 15.63 |
| factor_on_tip_rotation_noise / randomized | 0 | 46.75 | 0.34 | 2.97 | 14.06 |
| factor_off_tip_rotation_noise / zero | 8 | 19.67 | 3.52 | 6.22 | 18.34 |
| factor_off_tip_rotation_noise / randomized | 0 | 46.53 | 0.46 | 3.90 | 17.98 |
| factor_on_force_noise / zero | 0 | 46.71 | 0.31 | 4.49 | 13.04 |
| factor_on_force_noise / randomized | 0 | 46.76 | 0.35 | 2.97 | 13.99 |
| factor_off_force_noise / zero | 16 | 20.98 | 2.97 | 6.28 | 18.38 |
| factor_off_force_noise / randomized | 0 | 46.37 | 0.49 | 3.90 | 18.08 |
| factor_on_gains / zero | 0 | 46.54 | 0.35 | 4.49 | 15.29 |
| factor_on_gains / randomized | 0 | 46.76 | 0.36 | 2.99 | 14.48 |
| factor_off_gains / zero | 27 | 21.72 | 2.75 | 6.03 | 17.22 |
| factor_off_gains / randomized | 0 | 46.53 | 0.45 | 3.93 | 17.84 |
| factor_on_ema / zero | 0 | 46.74 | 0.29 | 4.38 | 17.66 |
| factor_on_ema / randomized | 0 | 46.76 | 0.34 | 2.92 | 15.59 |
| factor_off_ema / zero | 19 | 21.62 | 2.88 | 6.22 | 17.28 |
| factor_off_ema / randomized | 0 | 46.45 | 0.48 | 3.95 | 16.41 |
| factor_on_position_threshold / zero | 0 | 46.72 | 0.30 | 4.46 | 11.88 |
| factor_on_position_threshold / randomized | 0 | 46.75 | 0.35 | 2.95 | 13.99 |
| factor_off_position_threshold / zero | 20 | 21.13 | 3.03 | 6.32 | 18.16 |
| factor_off_position_threshold / randomized | 0 | 46.59 | 0.46 | 3.91 | 18.69 |
| factor_on_rotation_threshold / zero | 0 | 46.72 | 0.31 | 4.47 | 12.40 |
| factor_on_rotation_threshold / randomized | 0 | 46.75 | 0.34 | 2.95 | 14.03 |
| factor_off_rotation_threshold / zero | 17 | 20.91 | 3.14 | 6.33 | 18.48 |
| factor_off_rotation_threshold / randomized | 0 | 46.46 | 0.47 | 3.91 | 18.20 |
| factor_on_dead_zone / zero | 0 | 42.91 | 0.52 | 4.54 | 16.63 |
| factor_on_dead_zone / randomized | 0 | 46.76 | 0.42 | 3.15 | 15.22 |
| factor_off_dead_zone / zero | 13 | 21.24 | 3.11 | 6.14 | 18.41 |
| factor_off_dead_zone / randomized | 0 | 46.54 | 0.43 | 3.79 | 18.40 |

## 根据独立确认集得到的结论

- 观测扰动：按零随机化模型 T 点估计排序为 指尖旋转观测噪声、孔位置观测误差、力传感噪声、指尖位置观测噪声。排序不等于已证明各项之间存在显著差异。
- 控制参数：按零随机化模型 T 点估计排序为 死区阈值、EMA 平滑系数、旋转动作阈值、控制增益、位置动作阈值。排序不等于已证明各项之间存在显著差异。
- 孔位置观测误差：支持随机化模型的开启损失更小。
- 指尖位置观测噪声：尚不能确认开启损失有差异；不代表两者等效。
- 指尖旋转观测噪声：支持随机化模型的开启损失更小。
- 力传感噪声：尚不能确认开启损失有差异；不代表两者等效。
- 控制增益：尚不能确认开启损失有差异；不代表两者等效。
- EMA 平滑系数：尚不能确认开启损失有差异；不代表两者等效。
- 位置动作阈值：尚不能确认开启损失有差异；不代表两者等效。
- 旋转动作阈值：尚不能确认开启损失有差异；不代表两者等效。
- 死区阈值：支持随机化模型的开启损失更小。

仅比较两个固定 checkpoint，在当前扰动分布和固定值下解释。ON/OFF 背景不同，不能将各项分数相加归因为全部失败，也不能据此证明哪种训练随机化最有价值。全成功/全失败时经验 bootstrap 可能退化为零宽区间，不表示总体确定性；有限样本、五个评估 seed 和单个训练 seed 仍限制推断。均值腕力下降不代表峰值下降。
