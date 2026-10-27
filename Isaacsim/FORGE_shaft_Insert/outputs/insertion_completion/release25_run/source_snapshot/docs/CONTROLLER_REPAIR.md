# 本轮控制修复与验收（2026-09-28）

当前版本修复了明显的控制振荡，并实现接触试压、锁XY/yaw后持续下插；尚未通过稳定插入验收，不建议直接扩大到200轮PPO。修改属于工程适配，不能统称为论文参数。

## 越界的准确含义

`mdp/forge_paper.py::failed_workspace`计算**真实轴底相对真实固定孔口**的位置差e，任一条件触发失败：

- `sqrt(e.x²+e.y²)>0.100 m`。
- `e.z < −0.057 m`（47 mm目标深度再向下10 mm）。
- `e.z > +0.100 m`。

不是关节超限、TCP目标超限或孔估计偏差超限。阈值本轮未放宽。逐回合CSV保留完整回合统计，TensorBoard显示的终止均值不等于整场累计占比。

旧两轮GUI完整统计分别为1/17、41/47回合越界；第一轮另有4环境/4mini-batch的循环PPO广播问题，因此不能将差异归因于单一奖励修改。后一次确定性回放11/11为XY越界，实际TCP最高0.659 m/s，参考仅0.020 m/s，Y制动力已接近−10 N。说明控制跟踪故障是真实因素，不能仅解释为策略主动逃避惩罚。

## 当前设置

| 项目 | 原设置 | 当前设置 |
|---|---|---|
| 姿态控制 | Kp30，显式Kd10.954 | Kp30，独立Kd0.5 Nm·s/rad |
| 参考速度/加速度 | 20 mm/s、80 mm/s² | 12 mm/s、40 mm/s² |
| 平移力限制 | XYZ共享10 N | XY合力6 N，Z独立±8 N |
| 死区 | XYZ均0–5 N | XY弹簧项保留；阻尼和Z不经过死区 |
| 速度制动 | 参考限速 | 实际XY>25 mm/s停止弹簧驱动并制动；向下>15 mm/s仅制动 |
| 残差 | XYZ各±25 mm | 搜索XY±20 mm、Z±2 mm |
| 下插 | Z位置目标 | 试压/插入请求2–4 N、零动作3 N；首次接触后的搜索1.5 N |
| 下压力过渡 | 无 | 10 N/s，向下接近8 mm/s减小驱动，接触8–10 N卸载 |
| 夹持 | 每指−20 N，K/D均0 | −20 N持续、K0、D200 N·s/m，无位置同步 |
| reset | 标称轴约0.94°倾斜，等待0.5 s | 标称竖直，等待1 s；夹宽检查延迟1.5 s |
| 观测 | actor24/critic48 | actor28/critic52，增加4维阶段one-hot |

上述速度阈值是软制动，不是机器人实际速度的硬约束。命令力上限也不是碰撞峰值保证。每指20 N的模拟输入与真机API的20 N是否同一物理口径仍需标定。

四阶段：SETTLE等待夹持→SEARCH搜索→PROBE接触试压→INSERT持续插入。接触连续两帧≥1 N触发试压并锁定当时的TCP XY/yaw；向下进展≥3 mm且侧向力<2 N连续两帧确认进入。无接触进入也可由估计深度>16 mm和进展确认。试压0.8秒无进展返回搜索，冷却0.4秒。INSERT中横向残差不再改变目标，Z动作调节下压力。制动和过力卸载可以暂时产生向上力。阶段判断不是成功判据，不使用真实轴位姿作为在线纠偏输入。

七个机械臂关节均参与控制，没有锁定最后几个关节。hand的roll/pitch目标仍固定，策略没有主动pitch修正动作；歪在手里的shaft不会因为hand姿态环而自动恢复。

## 实测结果和限制

固定摩擦0.8的90步保持/yaw对照，数据在`outputs/controller_repair/probe_final/`：

| 对照 | 旧Kd峰值TCP速度 | 新Kd峰值TCP速度 | 新Kd最大TCP位移 |
|---|---:|---:|---:|
| 保持 | 586.7 mm/s | 4.29 mm/s | 1.61 mm |
| yaw转动 | 530.8 mm/s | 7.44 mm/s | 1.49 mm |

旧yaw组3次reset，新yaw组0次。它支持过大显式姿态阻尼导致不稳定的判断，不证明任何姿态/负载下都稳定。

最新无抓取/孔估计误差零残差：4环境160步，4完整回合全部超时，0越界、0夹宽失败，曾成功/稳定成功/终局成功均0/4。实际物理步TCP峰值10.7–14.0 mm/s；2环境进入INSERT，最深41.7 mm，未达到46–48 mm。采样接触峰值3.62 N，完成回合平均P95为2.43 N。详见`damped_grip_nominal.csv`及`damped_grip_analysis.json`。

reset后1秒倾角0.095–0.110°，基本竖直；实际轴底−虚拟TCP的Y误差约−0.57至−0.63 mm，已经接近/超过0.5 mm几何容差。全回合最大相对夹持漂移7.29 mm。**不能保证夹紧后无滑移**。不对称、x±8/z±5 mm抓取误差、孔估计xyz±10 mm仍保留，base不动。固定TCP做过一次离线均值修正，但改变夹爪阻尼后须再次标定，不应视为最终标定。

## 如何继续

先复查固定TCP与夹持中心动态：同步记录两指位置差、真实轴底−虚拟TCP、轴倾角、物理步速度、指令力与实际接触力。保持时漂移则先查夹持模型；仅yaw时偏移则查姿态耦合；参考平稳而实际速度突增则查阻尼、关节力矩饱和和制动。接触后轴底与TCP分离则不能靠移动TCP来推断轴已进入。

当前有序任务是：标定上述残差和夹爪动态→无误差零残差多回合验收→加入随机误差→处理提前失败收益与调整episode时长→20轮GUI学习检查→128环境pilot。奖励和成功/失败阈值本轮保持不变，提前失败回报漏洞尚未单独处理。4回合零成功不能靠宣称PPO会收敛来跳过。

可视化零残差（新目录名以免覆盖本轮数据）：

```bash
cd /home/masais/Xinyu/Isaacsim
export FORGE_PY=/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python
"$FORGE_PY" FORGE_shaft_Insert/scripts/paper_smoke_test.py --mode directed --nominal --num_envs 4 --steps 160 --metrics_csv FORGE_shaft_Insert/outputs/manual_stage_review/nominal.csv
```

之后若仅检查GUI训练链路，从新模型开始；旧24维checkpoint不兼容：

```bash
"$FORGE_PY" FORGE_shaft_Insert/scripts/train.py --task FORGE-Shaft-Insert-Paper-v0 --num_envs 4 --max_iterations 20 --run_name staged_gui_review agent.algorithm.num_mini_batches=2
```

这不是插入性能验收。新训练目录`logs/rsl_rl/forge_shaft_insert_staged/`保存源码快照、配置、TB指标、逐回合与控制轨迹。

最有用的真机数据：夹持稳定后的hand→轴底变换及重复性、左右手指位置/速度与20 N命令的对应关系、缓慢侧向加载时夹持中心位移、轴相对hand转角、下压时腕部六维力与轴运动。有限阻尼200是工程起点，不能代替这些测量。
