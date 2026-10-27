#!/usr/bin/env bash
set -e
cd /home/masais/Xinyu/Isaacsim
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python /home/masais/Xinyu/Isaacsim/FORGE_shaft_Insert/scripts/evaluate_policy.py --checkpoint /home/masais/Xinyu/Isaacsim/logs/rsl_rl/forge_shaft_insert_coupled_release25/2026-09-29_15-40-45_coupled1000_20260929_154037_d798fa/model_1017.pt --num_envs 4 --episodes_per_env 4 --seed 43 --output /home/masais/Xinyu/Isaacsim/FORGE_shaft_Insert/outputs/coupled1000/coupled1000_20260929_154037_d798fa/gui_final_seed43
