#!/usr/bin/env bash
set -euo pipefail
STUDY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/masais/miniconda3/envs/env_jingyun/bin/python "$STUDY_ROOT/progress.py" "$@"
