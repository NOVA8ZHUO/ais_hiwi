"""CPU regression checks: terminal success, quantiles, streak breaks and resets."""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('metrics', Path(__file__).parents[1]/'forge_shaft_insert/episode_metrics.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
m = module.EpisodeMetrics(2, 10, .1, 'cpu')
ids = torch.tensor([0, 1])
for step in range(6):
    # First env succeeds for five consecutive samples, including terminal sample.
    success = torch.tensor([step >= 1, step in (0, 2, 4)])
    m.sample(ids, success, torch.tensor([float(step), 10.]),
             torch.tensor([[1., 0., 0., 0., 1.], [0., 0., 0., 0., 1.]]), torch.tensor([True, False]))
rows = m.finish(ids, {'time_out': torch.tensor([True, True])})
assert rows[0]['success_final'] == rows[0]['success_stable'] == 1
assert rows[1]['success_any'] == 1 and rows[1]['success_final'] == rows[1]['success_stable'] == 0
assert abs(rows[0]['force_p95_N'] - 4.75) < 1e-6
assert rows[0]['action_saturation'] == .25 and rows[1]['action_saturation'] == 0
assert m.summary()['success_final'] == .5 and m.summary()['completed_episodes'] == 2
assert (m.steps == 0).all() and torch.isnan(m.forces).all()
m.sample(torch.tensor([0]), torch.tensor([True]), torch.tensor([3.]), torch.ones(1, 5), torch.tensor([True]))
m.reset(torch.tensor([0]))
assert m.completed == 2 and not m.ever.any()
print('EPISODE_METRICS_TEST_PASS')
