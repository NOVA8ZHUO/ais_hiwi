#!/usr/bin/env python3
"""Export recorded TensorBoard scalars; never infer evaluation success from reward."""
import argparse
import csv
import json
import math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
a = p.parse_args()
rows = []
series = {}
for directory in a.run.glob('**/summaries'):
    events = EventAccumulator(str(directory), size_guidance={'scalars': 0})
    events.Reload()
    for tag in events.Tags()['scalars']:
        values = events.Scalars(tag)
        series[tag] = values
        for x in values:
            rows.append({'source': str(directory.relative_to(a.run)), 'tag': tag,
                         'step': x.step, 'wall_time': x.wall_time, 'value': x.value})
if not rows:
    raise SystemExit('No scalar events found')
output = a.run / 'metrics'
output.mkdir(exist_ok=True)
with (output / 'scalars.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
invalid = [x for x in rows if not math.isfinite(x['value'])]
(output / 'summary.json').write_text(json.dumps({'scalar_count': len(rows), 'non_finite_count': len(invalid),
    'last_values': {k: v[-1].value for k, v in series.items() if v}}, indent=2))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = [['rewards/iter', 'rewards/step'], ['losses/a_loss', 'losses/c_loss', 'losses/cval_loss'],
              ['successes/iter', 'logs_rew_curr_success/iter']]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    for ax, names, title in zip(axes, groups, ['Training reward', 'Training losses', 'Training success signals (not evaluation)']):
        for tag in names:
            if tag in series:
                v = series[tag]
                ax.plot([x.step for x in v], [x.value for x in v], marker='.', label=tag)
        ax.set_title(title)
        ax.set_xlabel('TensorBoard step (tag-specific)')
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No scalar data recorded', transform=ax.transAxes, ha='center')
    fig.savefig(output / 'training_curves.png', dpi=150)
    plt.close(fig)
except ImportError:
    print('matplotlib unavailable; CSV and JSON exported without installing packages')
print(f'Exported {len(rows)} scalars; non-finite={len(invalid)}; {output}')
if invalid:
    raise SystemExit(1)
