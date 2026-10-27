#!/bin/bash
set -euo pipefail
cd /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_cad_baseline/logs/20261013T115201_931054Z_train
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/registered_entry.py /home/masais/Xinyu/Isaacsim/IsaacLab_2_3_2/scripts/reinforcement_learning/rl_games/train.py --task Isaac-Forge-ShaftLarge-CAD-Direct-v0 --seed 0 --headless
