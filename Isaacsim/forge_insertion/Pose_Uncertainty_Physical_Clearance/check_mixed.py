"""User-launched integration gate: no PPO updates."""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--group',required=True,choices=['P0G0','P1G0','P0G1','P1G1']);AppLauncher.add_app_launcher_args(p);a=p.parse_args()
app=AppLauncher(a).app
env=None;report={'status':'failed','group':a.group}
try:
 import gymnasium as gym,numpy as np,torch
 import isaaclab_tasks,pg_task
 from pg_task import config
 from forge_cad.config import ForgeCadCfg
 from pxr import UsdGeom
 cfg=getattr(config,a.group+'Cfg')();cfg.scene.num_envs=4;cfg.seed=0;cfg.sim.device=a.device
 baseline=ForgeCadCfg().to_dict();candidate=getattr(config,a.group+'Cfg')().to_dict()
 def diff(x,y,prefix=''):
  result=[]
  for k in x.keys()|y.keys():
   path=f'{prefix}.{k}' if prefix else k
   if isinstance(x.get(k),dict) and isinstance(y.get(k),dict):result+=diff(x[k],y[k],path)
   elif str(x.get(k))!=str(y.get(k)):result.append(path)
  return result
 delta=diff(baseline,candidate)
 allowed={'obs_rand.fixed_asset_pos','scene.replicate_physics','scene.clone_in_fabric','task.fixed_asset.spawn.func','task.fixed_asset.spawn.usd_path','task.fixed_asset.spawn.random_choice'}
 assert set(delta)<=allowed,delta
 env=gym.make(f'Isaac-Forge-CAD-PG-{a.group}-Direct-v0',cfg=cfg);r=env.unwrapped
 import omni.usd
 stage=omni.usd.get_context().get_stage();diameters=[]
 for i in range(4):
  path=cfg.task.fixed_asset.prim_path.replace('env_.*',f'env_{i}')
  from pxr import Usd
  meshes=[x for x in Usd.PrimRange(stage.GetPrimAtPath(path)) if x.IsA(UsdGeom.Mesh)]
  assert len(meshes)==1,(path,len(meshes))
  pts=np.array(UsdGeom.Mesh(meshes[0]).GetPointsAttr().Get());rr=np.linalg.norm(pts[:,:2],axis=1)
  diameters.append(float(2*rr.min()*1000-25))
 expected=[.25,.5,1.,2.] if a.group.endswith('G1') else [1.]*4
 np.testing.assert_allclose(diameters,expected,atol=.0001)
 for _ in range(2):
  obs,_=env.reset();bias=r.init_fixed_pos_obs_noise
  if a.group.startswith('P0'):assert torch.count_nonzero(bias[:,:2])==0
  assert all(torch.isfinite(v).all() for v in obs.values())
  for step in range(8):
   obs,reward,term,trunc,info=env.step(r.prev_actions.clone())
   assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
 report.update(status='passed',config_diff=delta,clearances_mm=diameters,training_updates=0)
except BaseException:
 import traceback
 report['error']=traceback.format_exc();print(report['error'],flush=True)
 raise
finally:
 Path('mixed_preflight.json').write_text(json.dumps(report,indent=2))
 if env is not None:env.close()
 app.close()
