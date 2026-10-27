#!/bin/bash
set -euo pipefail
cd /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_cad_baseline/logs/20261013T114541_642774Z_short
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/registered_entry.py /home/masais/Xinyu/Isaacsim/IsaacLab_2_3_2/scripts/reinforcement_learning/rl_games/train.py --task Isaac-Forge-ShaftLarge-CAD-Direct-v0 --seed 0 --headless --num_envs 4 --max_iterations 2
