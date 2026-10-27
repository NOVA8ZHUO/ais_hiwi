"""CPU-only regression checks; no SimulationApp is needed."""
import importlib.util
from pathlib import Path
import torch

path = Path(__file__).resolve().parents[1] / 'forge_shaft_insert/mdp/forge_math.py'
spec = importlib.util.spec_from_file_location('forge_math', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

distance = torch.tensor([0.0, 0.001, 0.01, 0.1, 100.0])
coarse = module.logistic_kernel(distance, 50.0, 2.0)
fine = module.logistic_kernel(distance, 100.0, 0.0)
assert coarse[0] == 0.25 and fine[0] == 0.5
assert torch.isfinite(coarse).all() and torch.isfinite(fine).all()
assert torch.all(coarse[:-1] >= coarse[1:])
assert torch.all(fine[:-1] >= fine[1:])
forces = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]])
assert torch.allclose(module.force_excess(forces, torch.tensor([5.0, 7.5, 5.0])), torch.tensor([0.0, 2.5, 0.0]))
assert torch.equal(module.dead_zone(torch.tensor([-7., -2., 0., 3., 8.]), torch.tensor(5.)), torch.tensor([-2., 0., 0., 0., 3.]))
print('FORGE reward kernel, force threshold and controller dead-zone checks passed.')
