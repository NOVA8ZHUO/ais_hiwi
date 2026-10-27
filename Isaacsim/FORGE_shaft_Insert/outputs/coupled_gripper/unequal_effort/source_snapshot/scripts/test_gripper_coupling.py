"""Physical coupling test: unequal finger efforts, opening and partial reset."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import json,sys,types
from pathlib import Path
import gymnasium as gym
import torch
import forge_shaft_insert
from forge_shaft_insert import experiment_archive as archive
from forge_shaft_insert.forge_paper_env_cfg import ForgePaperEnvCfg
from forge_shaft_insert.mdp import forge_paper as paper

out=Path(args.output)
archive.begin(out,argv=sys.argv)
env=None;status='failed'
try:
    cfg=ForgePaperEnvCfg();cfg.scene.num_envs=4;cfg.sim.device=args.device
    cfg.task.gripper_check_delay_s=30.  # Test intentionally opens outside release supervisor.
    cfg.task.gripper_settle_s=30.;cfg.task.grasp_position_error=(0.,0.,0.)
    cfg.task.hole_position_error=(0.,0.,0.);cfg.task.metrics_csv_path=str(out/'episode_metrics.csv')
    (out/'settings.json').write_text(json.dumps(cfg.to_dict(),indent=2,default=str)+'\n')
    env=gym.make('FORGE-Shaft-Insert-Paper-v0',cfg=cfg)
    env.reset(seed=42);raw=env.unwrapped;term=paper.action_term(raw)
    test_efforts=torch.tensor([[-20.,-20.],[0.,-20.],[-20.,0.],[-20.,-20.]],device=raw.device)
    def unequal(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self._asset.set_joint_effort_target(test_efforts[ids],joint_ids=self.fingers,env_ids=env_ids)
    term.hold_gripper=types.MethodType(unequal,term)
    peak=torch.zeros(4,device=raw.device);closed=None;opened=None
    for step in range(105):
        if step==40:
            raw.reset(env_ids=torch.tensor([0],device=raw.device))
            assert not term.releasing[0] and term.elapsed_steps[0]==0
        if step==60:
            closed=paper.gripper_width(raw).clone()
            # Only one finger actively opens in envs 1 and 2.
            test_efforts[:]=torch.tensor([[20.,20.],[0.,20.],[20.,0.],[20.,20.]],device=raw.device)
        obs,_,_,_,_=env.step(torch.zeros(4,5,device=raw.device))
        q=raw.scene['robot'].data.joint_pos[:,term.fingers]
        error=(q[:,0]-q[:,1]).abs();peak=torch.maximum(peak,error)
        assert torch.isfinite(q).all()
        assert (error<1e-4).all(),f'Coupling error: {error.tolist()}'
    opened=paper.gripper_width(raw).clone()
    assert (opened-closed>.005).all(),'Unequal opening effort did not open both jaws.'
    result=dict(max_finger_difference_mm=(peak*1000).tolist(),closed_width_mm=(closed*1000).tolist(),opened_width_mm=(opened*1000).tolist(),partial_reset_pass=True,unequal_effort_pass=True)
    (out/'coupling_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PHYSICAL_COUPLING_PASS',json.dumps(result),flush=True);status='completed'
finally:
    if env is not None:env.close()
    archive.finish(status);app.close()
