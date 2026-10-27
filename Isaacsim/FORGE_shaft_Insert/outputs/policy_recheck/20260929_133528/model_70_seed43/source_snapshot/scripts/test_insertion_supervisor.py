"""Stage transitions: contact dwell, progress, lock persistence and stalled recovery."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

spec = importlib.util.spec_from_file_location('supervisor', Path(__file__).parents[1]/'forge_shaft_insert/mdp/insertion_supervisor.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cfg = SimpleNamespace(probe_contact_N=1., free_entry_depth=.016, probe_progress=.003,
                      entry_lateral_force_N=2., probe_timeout_s=.8, probe_cooldown_s=.4)
s = m.InsertionSupervisor(2, 'cpu')
ready=torch.ones(2,dtype=torch.bool)
p=torch.tensor([[.01, .02, 0.],[-.01,.02,0.]])
hole=torch.zeros_like(p)
force=torch.tensor([[0.,0.,2.],[0.,0.,2.]])
yaw=torch.zeros(2,1)
for _ in range(2): s.update(ready,p,hole,force,yaw,1/15,cfg)
assert (s.mode == 2).all()
lock=s.locked_xy.clone()
p[0,2] = -.004
for _ in range(2): s.update(ready,p,hole,force,yaw,1/15,cfg)
assert s.mode[0] == 3
p[0,:2] += .01
for _ in range(15): s.update(ready,p,hole,force,yaw,1/15,cfg)
assert s.mode[0] == 3 and torch.equal(s.locked_xy[0],lock[0])
assert s.mode[1] == 1  # Stalled rim probe released.
s.reset(torch.tensor([0]))
assert s.mode[0] == 0 and s.mode[1] == 1
print('INSERTION_SUPERVISOR_TEST_PASS')
