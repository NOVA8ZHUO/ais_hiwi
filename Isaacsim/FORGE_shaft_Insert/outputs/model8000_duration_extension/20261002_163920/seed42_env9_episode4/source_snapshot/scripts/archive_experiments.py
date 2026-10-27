"""Backfill scalar exports and summaries without inventing historical source snapshots."""
import argparse
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('runs_root', type=Path)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('archive', Path(__file__).parents[1]/'forge_shaft_insert/experiment_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)
summary = []
for run in sorted(args.runs_root.iterdir()):
    if run.is_dir() and list(run.glob('events.out.tfevents.*')):
        summary.append(archive.export_run(run))
(args.runs_root/'experiment_index.json').write_text(json.dumps(summary, indent=2)+'\n')
print(f'Archived {len(summary)} runs; all scalar events exported, original files retained.')
