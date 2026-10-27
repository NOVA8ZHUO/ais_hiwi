#!/usr/bin/env bash
set -euo pipefail
ZERO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/masais/miniconda3/envs/env_jingyun/bin/python "$ZERO_ROOT/launch.py" "$@"
