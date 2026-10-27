"""Check target reversals, vector limits and the soft tilt penalty without Isaac."""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('math_terms', Path(__file__).parents[1]/'forge_shaft_insert/mdp/forge_math.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = torch.zeros(4, 3)
v = torch.zeros_like(p)
dt = 1/240
for step in range(1200):
    goal = torch.full_like(p, .025 if (step//80)%2 else -.025)
    next_p, next_v = m.bounded_reference(p, v, goal, dt, .02, .08)
    assert (next_v.norm(dim=-1) <= .020001).all()
    assert ((next_v-v).norm(dim=-1)/dt <= .08001).all()
    assert torch.allclose(next_p-p, next_v*dt, atol=1e-8)
    p, v = next_p, next_v
angles = torch.deg2rad(torch.tensor([0., 2., 3., 10., -10., 90.]))
cost = .25*m.pitch_cost(angles, torch.deg2rad(torch.tensor(3.)), torch.deg2rad(torch.tensor(7.)))
assert torch.allclose(cost, torch.tensor([0., 0., 0., .25, .25, 2.]), atol=1e-6)
print('GENTLE_REFERENCE_TEST_PASS: reversal acceleration/speed limits; pitch deadband/sign/cap')
