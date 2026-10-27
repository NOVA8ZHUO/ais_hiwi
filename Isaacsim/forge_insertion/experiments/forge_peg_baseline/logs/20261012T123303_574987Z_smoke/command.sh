#!/bin/bash
set -euo pipefail
cd /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261012T123303_574987Z_smoke
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/rollout.py --task Isaac-Forge-PegInsert-Direct-v0 --seed 0 --headless --num_envs 4 --episodes 4
