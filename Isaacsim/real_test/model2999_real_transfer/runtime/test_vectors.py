from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]
m=torch.jit.load(str(p/'policy_actor.ts'),map_location='cpu').eval()
v=torch.load(p/'reference_test_vectors.pt',map_location='cpu',weights_only=True)
h=torch.zeros(1,1,128);c=h.clone();err=0.
with torch.inference_mode():
 for i,obs in enumerate(v['observations']):
  if v['reset'][i]:h.zero_();c.zero_()
  raw,act,h,c=m(obs,h,c)
  err=max(err,float((raw-v['raw_actions'][i]).abs().max()))
  torch.testing.assert_close(raw,v['raw_actions'][i],atol=1e-5,rtol=1e-5)
print('PASS sequence/reset inference; maximum absolute error:',err)
