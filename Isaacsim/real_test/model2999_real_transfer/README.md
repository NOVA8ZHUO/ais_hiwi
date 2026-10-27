# model2999 真机迁移交付包

此包固定使用新奖励v2的model2999（不是3998）。仿真三seed成功318/384=82.81%，完整松爪严格落座310/384=80.73%。这是策略推理与接口适配资料包，不含已接通具体机器人驱动的程序，不会发送任何机器人命令。

## 最先使用

整目录复制到真机电脑。只运行策略不需要Isaac Sim或RSL-RL：使用Python（原环境3.11）、PyTorch（导出验证2.7.0，CPU可用）。按目标电脑安装对应CPU/CUDA构建；requirements-inference.txt只列推理依赖，不要照搬整个仿真pip freeze。

```bash
cd model2999_real_transfer
python3 verify_package.py
python3 runtime/policy.py
```

第二条运行的是全零合成观测的加载测试，输出不可直接用于机器人运动。真正接口调用：把runtime加入Python路径，`from policy import Policy`，`p=Policy('cpu')`；每个回合开始`p.reset()`，每15Hz控制步调用一次`p.step(obs29)`，返回5维裁剪后的均值动作。观测不要提前归一化，导出模型已经包含归一化。不能每步重置LSTM，也不能在同一步调用两次推进隐藏状态。

## 文件

- policy_actor.ts：独立TorchScript actor，含固定观测归一化。forward(obs[batch,29], h[1,batch,128], c[1,batch,128])返回raw_mean、clipped_action、h_next、c_next。
- runtime/policy.py：纯PyTorch单机器人推理包装，不含机器人驱动。
- model_2999.pt：原完整权重/优化器检查点，仅加载本可信来源文件；原检查点需要对应网络结构，推理优先用导出文件。
- params/：历史实际env/agent配置；source_snapshot/：训练时任务、控制、奖励、资产配置与脚本。
- OBSERVATION_ACTION.md：逐维接口合同和真机待对齐项。
- inference_validation.json：与原RSL策略400步序列、batch1/4、部分reset测试，最大输出差0。
- reference_test_vectors.pt与runtime/test_vectors.py：可在目标电脑验证导出模型的序列输出。
- evaluation_baseline/：seed42/43/44各128回合基准数据。
- versions.json/source_environment_pip_freeze.txt：来源依赖记录；不是目标电脑全部安装要求。
- SHA256SUMS.json/verify_package.py：迁移完整性检查。

## 真机接口还需完成

1. 标定hand→虚拟轴底TCP、真实夹持变换、孔口坐标以及机器人base与训练world的关系；不能把仿真坐标直接用于实机。
2. 实现观测、15Hz策略调度、循环状态reset、上一动作和阶段状态管理。底层仿真240Hz，真机控制周期与阻抗响应需对齐。
3. source_snapshot/forge_shaft_insert/mdp/forge_paper.py中的process_actions/apply_actions是动作到控制的参考，insertion_supervisor.py为阶段逻辑。原文件依赖Isaac，不能原样作为机器人驱动；需要移植同等状态更新、限速/力变化率/阻尼/限力语义。
4. 真机力传感器/外力估计与训练的shaft-base过滤接触净力不是同一个信号，需做重力补偿、零偏、符号和坐标匹配。
5. 成功25mm和松爪在仿真中依赖真实轴底几何。真机必须提供可信进深/入孔判断或人工确认，不能把预测动作当自动松爪信号。
6. 与机器人驱动对齐断连停止、独立运动/力限制和reset起始姿态。此包没有真机验证结果，不提供假定通用的硬件限值。

## 复现边界

所有推理文件使用相对路径可迁移。历史env.yaml和源码中保留原始绝对路径和外部USD引用以便溯源；完整重建Isaac仿真还需安装对应IsaacLab/Isaac Sim和外部资产，这不是离线仿真安装镜像。真机推理不需要这些USD。不要通过修改历史快照来记录新的真机标定，另建real_config文件。
