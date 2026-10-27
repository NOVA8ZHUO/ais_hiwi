"""Read-only inventory; no simulator imports or package installation."""
import argparse
import importlib.metadata as m
import json
import platform
from pathlib import Path
import subprocess
import sys
p = argparse.ArgumentParser()
p.add_argument('--lab', type=Path, required=True)
a = p.parse_args()
data = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': {}}
for name in ['isaacsim', 'isaaclab', 'torch', 'rl-games', 'gymnasium', 'tensorboard']:
    try:
        data['packages'][name] = m.version(name)
    except m.PackageNotFoundError:
        data['packages'][name] = 'NOT INSTALLED'
for name, cmd in [('git_commit', ['git', '-C', str(a.lab), 'rev-parse', 'HEAD']),
                  ('git_branch', ['git', '-C', str(a.lab), 'branch', '--show-current']),
                  ('gpu', ['nvidia-smi'])]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data[name] = {'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        data[name] = str(e)
try:
    import torch
    data['torch_runtime'] = {'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}
except ImportError as e:
    data['torch_runtime'] = str(e)
data['physx'] = 'UNVERIFIED: inspect Kit extension versions after simulator startup'
Path('environment_info.json').write_text(json.dumps(data, indent=2))
print(json.dumps(data, indent=2))
