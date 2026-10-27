#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/masais/miniconda3/envs/env_jingyun/bin/python "$ROOT/start.py"
