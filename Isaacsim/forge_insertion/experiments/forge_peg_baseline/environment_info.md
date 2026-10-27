# 实际运行环境（2026-10-12）

- 使用已有环境 `/home/masais/miniconda3/envs/env_jingyun/bin/python`；未安装、升级或删除任何环境。
- Python 3.11.16；Isaac Sim 5.1.0.0；PyTorch 2.7.0+cu128；CUDA runtime 12.8；RL-Games 1.6.1；Gymnasium 1.2.0；TensorBoard 2.21.0。
- 通过子进程 PYTHONPATH 显式选中本地 Isaac Lab v2.3.2，framework 0.54.2，commit `37ddf626871758333d6ed89cf64ad702aef127d0`，detached tag v2.3.2；原始源码无修改。
- PhysX 扩展缓存版本 `omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353`；此为 Omniverse 扩展版本，不冒充底层 SDK 的语义版本。
- GPU RTX 4090，24,564 MiB；驱动580.178.04；驱动显示CUDA 13.0不等于PyTorch运行时12.8。
- 启动前GPU已有两个其他项目仿真进程，各约3721 MiB；未停止或修改它们。共享GPU可能显著降低训练速度。
- 仿真实测：dt=1/120 s，decimation=8，policy step=1/15 s；Franka与官方8 mm Peg/Hole已成功加载。
- 当前 smoke 完成4回合，无非有限观测、奖励或力错误，exit=0。零动作测试不代表策略性能。

原始环境探测JSON：`logs/20261012T123257_592272Z_inspect/environment_info.json`。

运行中可见的非致命警告包括旧Gym维护提示、部分Kit接口弃用、CPU powersave和render_interval<decimation；未为消除警告修改物理或训练参数。
