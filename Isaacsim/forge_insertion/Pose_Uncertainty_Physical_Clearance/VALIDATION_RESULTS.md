# 正式训练前验证结果

本报告只包含资产/配置/reset/碰撞验证，不是训练或策略成功率。

| 条件 | 状态 | 说明 |
|---|---|---|
| P1_clearance_1mm | passed | [详细报告](validation/P1_clearance_1mm/validation.json) |
| P0_clearance_1mm | passed | [详细报告](validation/P0_clearance_1mm/validation.json) |
| P1_clearance_0.5mm | passed | [详细报告](validation/P1_clearance_0.5mm/validation.json) |
| P0_clearance_0.5mm | passed | [详细报告](validation/P0_clearance_0.5mm/validation.json) |
| P1_clearance_0.25mm | passed | [详细报告](validation/P1_clearance_0.25mm/validation.json) |
| P0_clearance_0.25mm | passed | [详细报告](validation/P0_clearance_0.25mm/validation.json) |
| P1_clearance_2mm | passed | [详细报告](validation/P1_clearance_2mm/validation.json) |
| P0_clearance_2mm | passed | [详细报告](validation/P0_clearance_2mm/validation.json) |

全部运行时检查通过；训练分布和实测间隙仍须在正式训练方案中固定。

其他随机化均继承完整随机化 CAD 基线。G1 的回合级几何采样和四组正式训练尚未实现/启动。

## 近壁碰撞探针（mm）

| 条件 | 自由位置偏移 | 自由位置最大位移 | 穿墙位置偏移 | 最小排斥位移 |
|---|---:|---:|---:|---:|
| P1_clearance_1mm | 0.25000 | 0.00003 | 1.00000 | 0.56033 |
| P0_clearance_1mm | 0.25000 | 0.00003 | 1.00000 | 0.56033 |
| P1_clearance_0.5mm | 0.12500 | 0.00004 | 0.75000 | 0.56753 |
| P0_clearance_0.5mm | 0.12500 | 0.00004 | 0.75000 | 0.56753 |
| P1_clearance_0.25mm | 0.06250 | 0.00004 | 0.62500 | 0.54257 |
| P0_clearance_0.25mm | 0.06250 | 0.00004 | 0.62500 | 0.54257 |
| P1_clearance_2mm | 0.50000 | 0.00006 | 1.50000 | 1.08524 |
| P0_clearance_2mm | 0.50000 | 0.00006 | 1.50000 | 1.08524 |

探针只覆盖离散初始姿态，不代表连续插入策略成功率，也不能确定 SDF 误差上限。未将旧模型成功率转移到新几何。
