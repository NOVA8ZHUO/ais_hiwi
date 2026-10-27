#!/bin/bash
set -euo pipefail
cd /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_cad_baseline/logs/20261013T160114_110218Z_evaluate
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/rollout.py --task Isaac-Forge-ShaftLarge-CAD-Direct-v0 --seed 0 --headless --episodes 128 --checkpoint /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_cad_baseline/logs/20261013T115201_931054Z_train/logs/rl_games/Forge/test/nn/last_Forge_ep_200_rew__341.10693_.pth
