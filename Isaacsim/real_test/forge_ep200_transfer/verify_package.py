"""Verify files and replay exported reference sequences, using CPU PyTorch only."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
for name,expected in json.loads((ROOT/'SHA256SUMS.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
with torch.inference_mode():
    for model in ['zero','randomized']:
        actor=torch.jit.load(str(ROOT/model/'policy_actor.ts'),map_location='cpu').eval()
        vectors=torch.load(ROOT/model/'reference_vectors.pt',map_location='cpu',weights_only=True)
        h=torch.zeros(2,1,1024);c=h.clone()
        for v in vectors:
            h[:,v['reset']]=0;c[:,v['reset']]=0
            action,mu,sigma,h,c=actor(v['observation'],h,c,v['epsilon'])
            for result,key in [(action,'action'),(mu,'mu'),(sigma,'sigma'),(h,'h'),(c,'c')]:
                torch.testing.assert_close(result,v[key],rtol=2e-5,atol=2e-5)
        print(model+': checksum and 80-step recurrent replay passed')
print('No hardware commands sent.')
