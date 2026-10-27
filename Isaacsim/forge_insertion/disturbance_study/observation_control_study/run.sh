#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/masais/miniconda3/envs/env_jingyun/bin/python -u "$HERE/experiment.py" run
