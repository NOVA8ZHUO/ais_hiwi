#!/bin/bash
set -euo pipefail
cd /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261012T203505_391561Z_evaluate
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/rollout.py --task Isaac-Forge-PegInsert-Direct-v0 --seed 0 --headless --episodes 128 --checkpoint /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261012T124421_296458Z_train/logs/rl_games/Forge/test/nn/last_Forge_ep_200_rew__336.2282_.pth
