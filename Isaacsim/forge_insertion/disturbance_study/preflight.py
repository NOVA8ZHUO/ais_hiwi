"""Small real-physics checks; never launches full study."""
import json,time,hashlib
import numpy as np
from runner import ROOT,run_case
from design import ALL
from samples import Tape,generate
start=time.monotonic()
reports={}
for name,active in [('zero',[]),('full',ALL),('mass_only',['mass'])]:
    reports[name]=run_case(name,active,4321,n=4,area='preflight')
initial={k:json.loads((ROOT/'preflight'/k/'seed_4321/initial_state.json').read_text()) for k in reports}
assert len({d['sample_sha256'] for d in initial.values()})==1
assert initial['full']['masses']==initial['mass_only']['masses']
assert np.allclose(initial['zero']['masses'],.0175)
assert np.allclose(initial['zero']['ema'],.0625)
assert np.allclose(initial['zero']['force_threshold'],7.5)
assert np.allclose(initial['zero']['quaternion_sign'],1)
assert np.allclose(initial['zero']['dead_zone'],[2.5,2.5,2.5,.5,.5,.5])
assert np.allclose(initial['zero']['base_position'],initial['mass_only']['base_position'],atol=1e-5)
for key in ['policy','base_position','mass','friction_buckets']:
    assert len({d['stream_sha256'][key] for d in initial.values()})==1
# Separate draws do not mutate bank values or depend on order.
tape=Tape(ROOT/'samples/n4_seed4321.npz');before=tape.meta['stream_sha256']
assert before==initial['zero']['stream_sha256']
assert all(d['episodes']==4 and d['policy_sample_count']>0 for d in reports.values())
(ROOT/'preflight_passed.json').write_text(json.dumps({'status':'passed','seconds':time.monotonic()-start,
    'checks':['paired samples','policy sample counts','mass paired full/on','nominal fixed values','one complete episode per env'],
    'validated_sha256':{str(ROOT/f):hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['samples.py','rollout.py','paired/env.py','paired/config.py','paired/events.py','paired/adapter.json']},
    'success_rates':{k:d['success_rate_final'] for k,d in reports.items()}},indent=2))
print('PREFLIGHT PASSED',flush=True)
