# CAD实验运行环境

沿用已验证的 `/home/masais/miniconda3/envs/env_jingyun/bin/python`，通过PYTHONPATH选择本地IsaacLab_2_3_2，commit 37ddf626871758333d6ed89cf64ad702aef127d0。
Isaac Sim 5.1，官方Isaac Lab v2.3.2、RL-Games 1.6.1、PyTorch 2.7.0+cu128。没有安装或升级依赖。
GPU RTX 4090。启动正式训练时仍有其他GPU作业；不修改其他项目。
详细版本探测见 ../forge_peg_baseline/environment_info.md；本次每阶段日志保存真实AppLauncher和任务解析路径。
