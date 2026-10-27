#!/usr/bin/env python3
"""Launch the locally installed official FORGE baseline without editing Isaac Lab."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASK = 'Isaac-Forge-PegInsert-Direct-v0'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('stage', choices=['inspect', 'smoke', 'short', 'train', 'play', 'evaluate', 'validate'])
p.add_argument('--variant', choices=['official', 'cad'], default='official')
p.add_argument('--lab', type=Path, default=Path(os.environ.get('ISAACLAB_PATH', ROOT.parent / 'IsaacLab_2_3_2')))
p.add_argument('--python', default=sys.executable, help='Python executable in an existing Isaac Lab environment')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--checkpoint', type=Path)
p.add_argument('--episodes', type=int, default=128)
p.add_argument('--num_envs', type=int, help='Playback or auxiliary evaluation only; training keeps official defaults')
p.add_argument('--gui', action='store_true')
p.add_argument('--dry-run', action='store_true')
a = p.parse_args()
lab = a.lab.resolve()
if a.variant == 'cad':
    TASK = 'Isaac-Forge-ShaftLarge-CAD-Direct-v0'
if a.stage == 'validate' and a.variant != 'cad':
    p.error('validate is specific to the CAD asset task')
if a.num_envs is not None and (a.stage not in ['evaluate', 'play'] or a.num_envs < 1):
    p.error('--num_envs is only allowed for playback or auxiliary evaluation and must be positive')
if a.seed < 0 or a.episodes < 1:
    p.error('seed must be nonnegative and episodes must be positive')
if not (lab / 'source/isaaclab_tasks/isaaclab_tasks/direct/forge').is_dir():
    p.error(f'FORGE source not found: {lab}')
if a.stage in ['play', 'evaluate'] and (a.checkpoint is None or not a.checkpoint.is_file()):
    p.error('play/evaluate requires an existing --checkpoint')
python = shutil.which(a.python)
if python is None:
    p.error(f'Python not found: {a.python}')
base = ROOT / 'experiments' / ('forge_cad_baseline' if a.variant == 'cad' else 'forge_peg_baseline')
if a.stage == 'validate':
    script = ROOT / 'scripts/validate_cad.py'
elif a.stage == 'inspect':
    script = ROOT / 'scripts/inspect_env.py'
elif a.stage in ['smoke', 'evaluate']:
    script = ROOT / 'scripts/rollout.py'
else:
    script = lab / 'scripts/reinforcement_learning/rl_games' / ('play.py' if a.stage == 'play' else 'train.py')
cmd = [python, str(script)]
if a.variant == 'cad' and a.stage in ['train', 'short', 'play']:
    cmd = [python, str(ROOT / 'scripts/registered_entry.py'), str(script)]
if a.stage == 'inspect':
    cmd += ['--lab', str(lab)]
else:
    cmd += ['--task', TASK, '--seed', str(a.seed)]
    if not a.gui:
        cmd += ['--headless']
    if a.stage in ['smoke', 'short', 'validate']:
        cmd += ['--num_envs', '4']
    if a.stage == 'short':
        cmd += ['--max_iterations', '2']
    if a.stage == 'smoke':
        cmd += ['--episodes', '4']
    if a.stage == 'evaluate':
        cmd += ['--episodes', str(a.episodes)]
        if a.num_envs is not None:
            cmd += ['--num_envs', str(a.num_envs)]
    if a.stage == 'play':
        if a.num_envs is not None:
            cmd += ['--num_envs', str(a.num_envs)]
        if a.gui:
            cmd += ['--real-time', 'env.viewer.eye=[1.3,1.0,0.9]', 'env.viewer.lookat=[0.5,0.0,0.15]']
    if a.checkpoint:
        cmd += ['--checkpoint', str(a.checkpoint.resolve())]
print(shlex.join(cmd), flush=True)
if a.dry_run:
    sys.exit(0)
run = base / 'logs' / (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + a.stage)
run.mkdir(parents=True)
source = lab / 'source/isaaclab_tasks/isaaclab_tasks/direct'
hashes = {}
for component in ['forge', 'factory']:
    for f in (source / component).rglob('*'):
        if f.suffix in ['.py', '.yaml']:
            rel = f.relative_to(source)
            target = run / 'source_snapshot' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            hashes[str(rel)] = hashlib.sha256(f.read_bytes()).hexdigest()
meta = {'stage': a.stage, 'variant': a.variant, 'command': cmd, 'cwd': str(run), 'lab': str(lab), 'source_sha256': hashes,
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'running', 'launcher_pid': os.getpid(),
        'entrypoint_sha256': hashlib.sha256(script.read_bytes()).hexdigest()}
for name, args in [('commit', ['rev-parse', 'HEAD']), ('git_status', ['status', '--porcelain'])]:
    result = subprocess.run(['git', '-C', str(lab), *args], capture_output=True, text=True)
    meta[name] = result.stdout if result.returncode == 0 else result.stderr
(run / 'manifest.json').write_text(json.dumps(meta, indent=2))
(run / 'command.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(run)) + '\n' + shlex.join(cmd) + '\n')
if a.variant == 'cad':
    for folder in ['forge_cad', 'assets/cad', 'scripts']:
        shutil.copytree(ROOT / folder, run / 'project_snapshot' / folder, ignore=shutil.ignore_patterns('__pycache__'))
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
# Select this checkout explicitly; avoid mixing editable installations from another checkout.
paths = [str(ROOT)] + [str(lab / 'source' / x) for x in ['isaaclab', 'isaaclab_assets', 'isaaclab_tasks', 'isaaclab_rl']]
env['PYTHONPATH'] = os.pathsep.join(paths + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
code = 130
try:
    with (run / 'console.log').open('w') as log:
        child = subprocess.Popen(cmd, cwd=run, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        meta['child_pid'] = child.pid
        (run / 'manifest.json').write_text(json.dumps(meta, indent=2))
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
finally:
    # SimulationApp.close may terminate Python before a raised exception sets a nonzero exit.
    if a.stage == 'validate':
        report = run / 'validation.json'
        if not report.exists() or json.loads(report.read_text()).get('status') != 'passed':
            code = code or 1
    if a.stage in ['smoke', 'evaluate'] and (not (run / 'evaluation.json').exists() or (run / 'failure.txt').exists()):
        code = code or 1
    meta.update(exit_code=code, status='succeeded' if code == 0 else 'failed', finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    (run / 'manifest.json').write_text(json.dumps(meta, indent=2))
    for pattern, dest in [('**/params/*.yaml', 'resolved_configs'), ('**/nn/*.pth', 'checkpoints')]:
        for f in run.glob(pattern):
            target = base / dest / run.name / f.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
    for name in ['evaluation.json', 'failure.txt', 'validation.json']:
        f = run / name
        if f.exists():
            target = base / 'evaluation' / run.name / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
    if code == 0:
        with (base / 'commands.sh').open('a') as f:
            f.write('\n# Process exited zero; consult manifest/logs for stage validation.\n' + shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]) + '\n')
print(f'Artifacts: {run}')
sys.exit(code)
