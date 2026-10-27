#!/bin/bash
set -euo pipefail
cd /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261012T124038_597456Z_evaluate
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/rollout.py --task Isaac-Forge-PegInsert-Direct-v0 --seed 0 --headless --episodes 4 --num_envs 4 --checkpoint /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261012T123556_559314Z_short/logs/rl_games/Forge/test/nn/last_Forge_ep_2_rew__-9.759542_.pth
