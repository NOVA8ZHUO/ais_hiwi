#!/bin/bash
set -euo pipefail
cd /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_cad_baseline/logs/20261013T114102_314229Z_validate
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/validate_cad.py --task Isaac-Forge-ShaftLarge-CAD-Direct-v0 --seed 0 --headless --num_envs 4
