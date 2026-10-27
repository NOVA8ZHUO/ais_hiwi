#!/bin/bash
# Successful verified commands are appended by run.py.
set -euo pipefail

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py validate --variant cad

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py smoke --variant cad

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py short --variant cad

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py evaluate --variant cad --checkpoint /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261012T124421_296458Z_train/logs/rl_games/Forge/test/nn/last_Forge_ep_200_rew__336.2282_.pth --num_envs 4 --episodes 4

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py evaluate --variant cad --checkpoint forge_insertion/experiments/forge_cad_baseline/logs/20261013T114541_642774Z_short/logs/rl_games/Forge/test/nn/last_Forge_ep_2_rew__24.709248_.pth --num_envs 4 --episodes 4

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py reset_view --variant cad --gui --num_envs 1

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py train --variant cad

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py evaluate --variant cad --checkpoint /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_cad_baseline/logs/20261013T115201_931054Z_train/logs/rl_games/Forge/test/nn/last_Forge_ep_200_rew__341.10693_.pth --episodes 128
