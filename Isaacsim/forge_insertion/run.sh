#!/usr/bin/env bash
set -euo pipefail
FORGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FORGE_PYTHON="${FORGE_PYTHON:-/home/masais/miniconda3/envs/env_jingyun/bin/python}"
exec "$FORGE_PYTHON" "$FORGE_ROOT/scripts/run.py" "$@"
