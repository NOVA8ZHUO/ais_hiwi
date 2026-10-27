"""Portable experiment evidence: immutable source snapshot and lossless scalar export."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

_active = None
_streams = None
_console = None


class Tee:
    def __init__(self, original, file):
        self.original, self.file = original, file

    def write(self, text):
        self.original.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


def begin(log_dir, argv=None):
    global _active, _streams, _console
    _active = Path(log_dir)
    _active.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    snapshot = _active/'source_snapshot'
    manifest = {}
    for folder in ('forge_shaft_insert', 'scripts', 'docs'):
        for source in (root/folder).rglob('*'):
            if source.suffix not in ('.py', '.md', '.yaml', '.json') or '__pycache__' in source.parts:
                continue
            relative = source.relative_to(root)
            target = snapshot/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (_active/'experiment_manifest.json').write_text(json.dumps({
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'argv': list(argv if argv is not None else sys.argv), 'python_executable': sys.executable, 'python_version': sys.version, 'command_environment_note': os.environ.get('FORGE_EXPERIMENT_NOTE', ''),
        'source_sha256': manifest, 'status': 'running',
        'console_scope': 'Python stdout/stderr from environment creation onward; native Kit output may be absent',
    }, indent=2)+'\n')
    _console = (_active/'training_console.log').open('w')
    _streams = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(sys.stdout, _console), Tee(sys.stderr, _console)


def export_run(path):
    """Export ALL scalar events; historical source code is never fabricated."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    path = Path(path)
    events = EventAccumulator(str(path), size_guidance={'scalars': 0}, purge_orphaned_data=False).Reload()
    final = {}
    with (path/'metrics_scalars.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['tag', 'iteration', 'wall_time', 'value'])
        for tag in events.Tags().get('scalars', []):
            values = events.Scalars(tag)
            for value in values:
                writer.writerow([tag, value.step, value.wall_time, value.value])
            if values:
                final[tag] = values[-1].value
    episode_file = path/'episode_metrics.csv'
    rows = list(csv.DictReader(episode_file.open())) if episode_file.exists() else []
    n = len(rows)
    summary = {'run': str(path.resolve()), 'completed_episodes': n, 'final_scalars': final,
               'source_snapshot_available': (path/'source_snapshot').is_dir(),
               'terminations': {key: sum(int(row.get(key, 0)) for row in rows)
                                for key in ('time_out', 'outside_workspace', 'invalid_grasp_width')},
               'success_rates': {key: sum(int(row[key]) for row in rows)/n if n else None
                                 for key in ('success_any', 'success_final', 'success_stable')}}
    (path/'experiment_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def finish(status):
    global _active, _console, _streams
    if _active is None:
        return
    try:
        export_run(_active)
        path = _active/'experiment_manifest.json'
        metadata = json.loads(path.read_text())
        metadata.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(metadata, indent=2)+'\n')
    finally:
        if _streams:
            sys.stdout, sys.stderr = _streams
        if _console:
            _console.close()
        _active = _console = _streams = None
