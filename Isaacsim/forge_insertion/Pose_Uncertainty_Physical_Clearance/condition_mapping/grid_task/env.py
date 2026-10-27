import os,torch
from paired.env import PairedEnv
class GridEnv(PairedEnv):
 def _sample(self,key,context,shape,**kwargs):
  value=super()._sample(key,context,shape,**kwargs)
  if key=='hole_observation':
   # Original named standard-normal tape -> paired uniform XY direction.
   # Parent reset multiplies by nominal sigma=0.001 m; Z remains untouched.
   radius_mm=float(os.environ['PG_ERROR_MM'])
   assert self.cfg.obs_rand.fixed_asset_pos==[.001,.001,.001]
   value[:,:2]=value[:,:2]/torch.linalg.vector_norm(value[:,:2],dim=-1,keepdim=True).clamp_min(1e-12)*radius_mm
  return value
