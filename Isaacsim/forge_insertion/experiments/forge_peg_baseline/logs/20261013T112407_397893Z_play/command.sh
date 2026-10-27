#!/bin/bash
set -euo pipefail
cd /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261013T112407_397893Z_play
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/IsaacLab_2_3_2/scripts/reinforcement_learning/rl_games/play.py --task Isaac-Forge-PegInsert-Direct-v0 --seed 0 --num_envs 1 --real-time 'env.viewer.eye=[1.3,1.0,0.9]' 'env.viewer.lookat=[0.5,0.0,0.15]' --checkpoint /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261012T124421_296458Z_train/logs/rl_games/Forge/test/nn/last_Forge_ep_200_rew__336.2282_.pth
