#!/bin/bash
set -euo pipefail
cd /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_cad_baseline/logs/20261013T114638_517058Z_evaluate
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/rollout.py --task Isaac-Forge-ShaftLarge-CAD-Direct-v0 --seed 0 --headless --episodes 4 --num_envs 4 --checkpoint /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261012T124421_296458Z_train/logs/rl_games/Forge/test/nn/last_Forge_ep_200_rew__336.2282_.pth
