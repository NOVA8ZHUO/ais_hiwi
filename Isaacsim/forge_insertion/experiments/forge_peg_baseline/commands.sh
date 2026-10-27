#!/bin/bash
# Only commands from processes that exited with status zero are appended here.
# No simulator/training command has been executed at initial delivery.
set -euo pipefail

# Process exited zero; consult manifest/logs for stage validation.
/usr/bin/python3 /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py inspect --python /home/masais/miniconda3/envs/env_jingyun/bin/python

# Process exited zero; consult manifest/logs for stage validation.
/usr/bin/python3 /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py smoke --python /home/masais/miniconda3/envs/env_jingyun/bin/python

# Process exited zero; consult manifest/logs for stage validation.
/usr/bin/python3 /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py short --python /home/masais/miniconda3/envs/env_jingyun/bin/python

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py evaluate --checkpoint forge_insertion/experiments/forge_peg_baseline/logs/20261012T123556_559314Z_short/logs/rl_games/Forge/test/nn/last_Forge_ep_2_rew__-9.759542_.pth --episodes 4 --num_envs 4

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py train

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py evaluate --checkpoint /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261012T124421_296458Z_train/logs/rl_games/Forge/test/nn/last_Forge_ep_200_rew__336.2282_.pth --episodes 128

# Process exited zero; consult manifest/logs for stage validation.
/home/masais/miniconda3/envs/env_jingyun/bin/python /home/masais/Xinyu/Isaacsim/forge_insertion/scripts/run.py play --gui --num_envs 1 --checkpoint /home/masais/Xinyu/Isaacsim/forge_insertion/experiments/forge_peg_baseline/logs/20261012T124421_296458Z_train/logs/rl_games/Forge/test/nn/last_Forge_ep_200_rew__336.2282_.pth
