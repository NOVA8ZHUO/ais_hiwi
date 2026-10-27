#!/usr/bin/env python3
"""Run the official full budget then evaluate its final checkpoint; never retune."""
import datetime as dt
import json
import os
import fcntl
from pathlib import Path
import subprocess
import sys
import traceback
import argparse
import hashlib

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['official','cad'], default='official')
args = parser.parse_args()
BASE = ROOT / 'experiments' / ('forge_cad_baseline' if args.variant == 'cad' else 'forge_peg_baseline')
BASE.mkdir(parents=True, exist_ok=True)
STATUS = BASE / 'pipeline_status.json'
lock_file = (BASE / 'pipeline.lock').open('w')
fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
state = {'pid': os.getpid(), 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'stage': 'starting', 'variant': args.variant}

def save(**values):
    state.update(values)
    state['updated_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATUS)

def run(stage, *args):
    before = set((BASE / 'logs').glob(f'*_{stage}'))
    save(stage=stage, status='running')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run.py'), stage, '--variant', state['variant'], *args])
    created = set((BASE / 'logs').glob(f'*_{stage}')) - before
    if len(created) != 1:
        raise RuntimeError(f'Expected one {stage} run, found {len(created)}')
    directory = created.pop()
    save(**{stage + '_directory': str(directory), stage + '_exit_code': result.returncode})
    if result.returncode:
        raise RuntimeError(f'{stage} failed; inspect {directory}/console.log')
    return directory

try:
    if args.variant == 'cad':
        gate = json.loads((BASE / 'preflight_results.json').read_text())
        for required in ['validate', 'smoke', 'short', 'checkpoint_reload']:
            if gate[required]['exit_code'] != 0:
                raise RuntimeError(f'Preflight gate failed: {required}')
        if hashlib.sha256((ROOT / 'forge_cad/config.py').read_bytes()).hexdigest() != gate['config_sha256']:
            raise RuntimeError('CAD config changed after validation')
        for asset in gate['asset_hashes'].values():
            if hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest() != asset['sha256']:
                raise RuntimeError('CAD asset changed after validation')
    training = run('train')
    subprocess.run([sys.executable, str(ROOT / 'scripts/export_metrics.py'), str(training)], check=True)
    # RL-Games saves last_* at max_epochs; prefer it over best-reward weights.
    weights = list(training.glob('**/nn/last_*.pth'))
    if not weights:
        raise RuntimeError('No final last_* checkpoint; refusing to label an incomplete run complete')
    checkpoint = max(weights, key=lambda path: path.stat().st_mtime_ns)
    import yaml
    agent_file = training / 'logs/rl_games/Forge/test/params/agent.yaml'
    config = yaml.safe_load(agent_file.read_text())
    budget = config['params']['config']['max_epochs']
    if f'_ep_{budget}_' not in checkpoint.name:
        raise RuntimeError(f'Final checkpoint does not confirm official budget {budget}: {checkpoint.name}')
    save(checkpoint=str(checkpoint))
    evaluation = run('evaluate', '--checkpoint', str(checkpoint), '--episodes', '128')
    report = json.loads((evaluation / 'evaluation.json').read_text())
    if report['episodes'] != 128:
        raise RuntimeError('Incomplete evaluation')
    save(status='complete', stage='complete', evaluation_summary={k:v for k,v in report.items() if k != 'rows'})
    (BASE / 'results.md').write_text(
        ('# 官方 FORGE + 自有 CAD 结果\n\n' if args.variant == 'cad' else '# 官方代码 Baseline 结果\n\n')
        + f'正式训练目录：`{training}`\n\n最终模型：`{checkpoint}`\n\n'
        f'辅助评估128回合：最终成功率 {report["success_rate_final"]:.4f}；'
        f'曾经成功率 {report["success_rate_any"]:.4f}；平均回报 {report["mean_episode_reward"]:.4f}。\n\n'
        '本结果仅为本任务seed=0的结果；不是论文三seed汇总或真机性能。'
        '配置差异详见config_audit.md。TensorBoard原始事件位于训练目录中。\n')
except BaseException:
    save(status='failed', error=traceback.format_exc())
    raise
