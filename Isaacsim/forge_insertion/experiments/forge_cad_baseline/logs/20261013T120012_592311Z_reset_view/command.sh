#!/bin/bash
set -euo pipefail
cd /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_cad_baseline/logs/20261013T120012_592311Z_reset_view
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/view_reset.py --task Isaac-Forge-ShaftLarge-CAD-Direct-v0 --seed 0 --num_envs 1
