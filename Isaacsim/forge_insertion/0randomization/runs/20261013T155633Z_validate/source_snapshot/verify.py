"""Audit config deltas and optionally test two resets plus interval crossing."""
import argparse, json, traceback
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['audit','validate'],required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
app=AppLauncher(a).app
env=None
try:
    import torch, yaml, gymnasium as gym
    import isaaclab_tasks, forge_zero
    from forge_zero.config import ForgeZeroCfg
    from forge_zero.cad_baseline import ForgeCadCfg
    from isaaclab.utils.io import dump_yaml
    root=Path(__file__).resolve().parent
    cfg=ForgeZeroCfg(); baseline=ForgeCadCfg()
    def diff(x,y,prefix=''):
        result=[]
        for k in x.keys() | y.keys():
            name=prefix+'.'+k if prefix else k
            if isinstance(x.get(k),dict) and isinstance(y.get(k),dict): result+=diff(x[k],y[k],name)
            elif x.get(k)!=y.get(k): result.append({'field':name,'baseline':str(x.get(k)),'zero':str(y.get(k))})
        return result
    delta=diff(baseline.to_dict(),cfg.to_dict())
    allowed={'task.fixed_asset_init_pos_noise','task.fixed_asset_init_orn_range_deg','task.hand_init_pos_noise',
        'task.hand_init_orn_noise','task.held_asset_pos_noise','obs_rand.fixed_asset_pos','obs_rand.fingertip_pos',
        'obs_rand.fingertip_rot_deg','obs_rand.ft_force','ctrl.task_prop_gains_noise_level','ctrl.pos_threshold_noise_level',
        'ctrl.rot_threshold_noise_level','ctrl.ema_factor_range','task.contact_penalty_threshold_range',
        'events.object_scale_mass.params.mass_distribution_params',
        'events.fixed_physics_material.params.static_friction_range','events.dead_zone_thresholds'}
    assert {d['field'] for d in delta}==allowed,delta
    assert cfg.scene.num_envs==128 and cfg.episode_length_s==10.0 and cfg.task.held_asset_cfg.mass==.0175
    ref=json.loads((root/'reference.json').read_text())
    original=Path(ref['reference_run'])/'logs/rl_games/Forge/test/params/agent.yaml'
    # YAML contains tuples in the resolved reference; full loader supports those.
    agent=yaml.full_load(original.read_text())
    ours=yaml.safe_load((root/'forge_zero/ppo.yaml').read_text())
    # Resolved training overrides only device/actors/seed/train directory.
    ad=diff(ours,agent)
    permitted={'params.config.train_dir','params.config.device_name'}
    assert all(d['field'] in permitted for d in ad),ad
    assert ours['params']['config']['max_epochs']==200
    assert not ours['params']['load_checkpoint'] and ours['params']['seed']==0
    report={'config_differences':delta,'ppo_runtime_differences':ad,'reference_run':ref['reference_run']}
    dump_yaml(str(Path.cwd()/'zero_env.yaml'),cfg)
    if a.mode=='validate':
        cfg.scene.num_envs=4;cfg.seed=0
        env=gym.make(forge_zero.TASK_ID,cfg=cfg);r=env.unwrapped
        snapshots=[]
        def check():
            assert torch.allclose(r.ema_factor,torch.full_like(r.ema_factor,.0625))
            assert torch.allclose(r.contact_penalty_thresholds,torch.full_like(r.contact_penalty_thresholds,7.5))
            assert torch.allclose(r.dead_zone_thresholds,.5*r.default_dead_zone)
            assert torch.all(r.flip_quats==1)
            assert torch.allclose(r.task_prop_gains,r.default_gains)
            assert torch.allclose(r.pos_threshold,r.default_pos_threshold)
            assert torch.allclose(r.rot_threshold,r.default_rot_threshold)
            assert torch.count_nonzero(r.init_fixed_pos_obs_noise)==0
            assert torch.allclose(r.noisy_force,r.force_sensor_smooth[:,:3])
            assert torch.allclose(r._held_asset.root_physx_view.get_masses(),torch.full_like(r._held_asset.root_physx_view.get_masses(),.0175))
        for trial in range(2):
            obs,_=env.reset();check()
            assert all(torch.isfinite(x).all() for x in obs.values())
            snapshots.append(r.held_pos.detach().clone())
            assert torch.allclose(r.fixed_pos,torch.tensor([.6,0.,.05],device=r.device).expand_as(r.fixed_pos),atol=1e-5)
            width=r.joint_pos[:,7:].sum(-1)
            assert ((width>.028)&(width<.042)).all()
            action=r.prev_actions.clone()
            with torch.no_grad():
                for _ in range(35):
                    obs,rew,*_=env.step(action)
                    assert torch.isfinite(rew).all() and all(torch.isfinite(x).all() for x in obs.values())
            check()
        reset_delta = (snapshots[0]-snapshots[1]).abs().max().item()
        report['reset_shaft_tip_max_difference_m'] = reset_delta
        report['reset_shaft_tip_positions_m'] = [x.cpu().tolist() for x in snapshots]
        Path('reset_diagnostics.json').write_text(json.dumps(report,indent=2))
        assert reset_delta < 1e-4, reset_delta
        report['runtime']='two resets and 35 steps each passed; interval >2s tested'
    Path('passed.json').write_text(json.dumps(report,indent=2))
    print('VERIFICATION_PASSED',flush=True)
except Exception:
    Path('failure.txt').write_text(traceback.format_exc());raise
finally:
    if env is not None:env.close()
    app.close()
