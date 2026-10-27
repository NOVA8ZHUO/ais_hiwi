"""Validate both checkpoints on identical small paired evaluation; exclude from study."""
import json
import numpy as np
from runner import ROOT,run_case
from design import ALL
for model in ['zero','randomized']:
    run_case('full',ALL,4321,n=4,area='two_model_preflight_'+model,model=model)
initial={m:json.loads((ROOT/('two_model_preflight_'+m)/'full/seed_4321/initial_state.json').read_text()) for m in ['zero','randomized']}
assert initial['zero']['sample_sha256']==initial['randomized']['sample_sha256']
assert initial['zero']['stream_sha256']==initial['randomized']['stream_sha256']
for key in ['base_position','base_quaternion','masses','friction','gains','ema','force_threshold','dead_zone','quaternion_sign']:
    np.testing.assert_allclose(initial['zero'][key],initial['randomized'][key],atol=1e-6,rtol=0)
(ROOT/'two_model_preflight_passed.json').write_text(json.dumps({'status':'passed','checks':['both epoch-200 checkpoints loaded','same environmental and policy sample tapes','same applied reset perturbations'],'episodes_per_model':4,'checkpoint_sha256':{m:json.loads((ROOT/('two_model_preflight_'+m)/'full/seed_4321/done.json').read_text())['identity']['checkpoint_sha256'] for m in ['zero','randomized']}},indent=2))
print('TWO MODEL PREFLIGHT PASSED',flush=True)
