#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--once" ]]; then
 exec /home/masais/miniconda3/envs/env_jingyun/bin/python "$HERE/experiment.py" progress
fi
exec watch -n 5 /home/masais/miniconda3/envs/env_jingyun/bin/python "$HERE/experiment.py" progress
