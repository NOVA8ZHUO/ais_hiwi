"""Independent output directories; full budget is fixed to the reference training."""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1]/'IsaacLab_2_3_2'
p = argparse.ArgumentParser()
p.add_argument('stage', choices=['audit', 'validate', 'train'])
p.add_argument('--dry-run', action='store_true')
a = p.parse_args()
ref = json.loads((ROOT/'reference.json').read_text())
for name, expected in ref['sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise SystemExit('Reference changed; review before running: '+name)
paths = [ROOT]+[LAB/'source'/x for x in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']]
env = os.environ.copy()
env['PYTHONPATH'] = os.pathsep.join(map(str, paths))
env['PYTHONUNBUFFERED'] = '1'
if a.stage == 'train':
    cmd = [sys.executable, str(ROOT/'entry.py'), str(LAB/'scripts/reinforcement_learning/rl_games/train.py'),
           '--task', 'Isaac-Forge-ShaftLarge-ZeroRand-Direct-v0', '--headless', '--seed', '0',
           '--num_envs', '128', '--max_iterations', '200']
else:
    cmd = [sys.executable, str(ROOT/'verify.py'), '--headless', '--mode', a.stage]
print(' '.join(cmd), flush=True)
if a.dry_run: sys.exit(0)
run = ROOT/'runs'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+a.stage)
run.mkdir(parents=True)
for folder in ['forge_zero', 'assets']:
    shutil.copytree(ROOT/folder, run/'source_snapshot'/folder, ignore=shutil.ignore_patterns('__pycache__'))
for name in ['launch.py','entry.py','verify.py','reference.json','README.md']:
    shutil.copy2(ROOT/name,run/'source_snapshot'/name)
meta = {'command':cmd, 'reference_run':ref['reference_run'], 'status':'running'}
(run/'manifest.json').write_text(json.dumps(meta,indent=2))
with (run/'console.log').open('w') as log:
    child = subprocess.Popen(cmd, cwd=run, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in child.stdout:
            print(line,end='',flush=True); log.write(line); log.flush()
        code = child.wait()
    except KeyboardInterrupt:
        child.terminate(); child.wait(); code = 130
if a.stage != 'train' and not (run/'passed.json').exists(): code = code or 1
if a.stage == 'train' and not list(run.rglob('*ep_200*.pth')): code = code or 1
meta.update(status='passed' if code==0 else 'failed', exit_code=code)
(run/'manifest.json').write_text(json.dumps(meta,indent=2))
print('Artifacts:', run)
sys.exit(code)
