#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec watch -n 5 /home/masais/miniconda3/envs/env_jingyun/bin/python "$ROOT/training_progress.py"
