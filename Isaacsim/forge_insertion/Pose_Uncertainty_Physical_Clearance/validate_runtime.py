"""Asset acceptance checks only; never used to alter training behavior."""
import argparse
import json
from pathlib import Path
import traceback
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--task', default='Isaac-Forge-ShaftLarge-CAD-Direct-v0')
p.add_argument('--clearance',type=float,required=True)
p.add_argument('--pose',type=int,choices=[0,1],required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--num_envs', type=int, default=4)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
launcher = AppLauncher(a)
app = launcher.app
env = None
report = {'status': 'running', 'checks': {}}
try:
    import torch
    import gymnasium as gym
    import isaaclab_tasks
    import forge_cad
    from forge_cad.config import ForgeCadCfg
    from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg
    from isaaclab.utils.io import dump_yaml
    official = ForgeTaskPegInsertCfg().to_dict()
    cfg = ForgeCadCfg()
    baseline=cfg.to_dict()
    study_root=Path(__file__).resolve().parent
    cfg.task.fixed_asset_cfg.usd_path=str(study_root/'assets'/f'base_clearance_{a.clearance:g}mm.usda')
    cfg.task.fixed_asset.spawn.usd_path=cfg.task.fixed_asset_cfg.usd_path
    cfg.task.fixed_asset_cfg.diameter=round(.025+a.clearance/1000,9)
    cfg.obs_rand.fixed_asset_pos=[.001*a.pose,.001*a.pose,.001]
    report['pose_xy_sigma_m']=.001*a.pose
    report['clearance_mm']=a.clearance
    def differences(x, y, prefix=''):
        out = []
        for key in x.keys() | y.keys():
            path = prefix + '.' + key if prefix else key
            if isinstance(x.get(key), dict) and isinstance(y.get(key), dict):
                out.extend(differences(x[key], y[key], path))
            elif x.get(key) != y.get(key):
                out.append({'path': path, 'official': str(x.get(key)), 'cad': str(y.get(key))})
        return out
    study_diff=differences(baseline,cfg.to_dict())
    permitted={'task.fixed_asset_cfg.usd_path','task.fixed_asset.spawn.usd_path','task.fixed_asset_cfg.diameter','obs_rand.fixed_asset_pos'}
    assert all(d['path'] in permitted for d in study_diff),study_diff
    assert ForgeCadCfg().to_dict()==baseline,'Mutated baseline defaults'
    report['full_randomized_baseline_diff']=study_diff
    delta = differences(official, cfg.to_dict())
    allowed = {'task.held_asset_cfg.usd_path','task.held_asset_cfg.height','task.held_asset_cfg.diameter',
        'task.held_asset_cfg.mass','task.held_asset.spawn.usd_path','task.held_asset.spawn.mass_props.mass',
        'task.fixed_asset_cfg.usd_path','task.fixed_asset_cfg.height','task.fixed_asset_cfg.diameter',
        'task.fixed_asset.spawn.usd_path','task.hand_init_pos','obs_rand.fixed_asset_pos'}
    assert all(d['path'] in allowed for d in delta), delta
    assert ForgeTaskPegInsertCfg().to_dict() == official, 'CAD configuration mutated original defaults'
    report['configuration_diff'] = delta
    report['checks']['config_whitelist'] = True
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    cfg.sim.device = a.device
    dump_yaml('params/env.yaml', cfg)
    env = gym.make(a.task, cfg=cfg)
    r = env.unwrapped
    assert r._fixed_asset.is_fixed_base and not r._held_asset.is_fixed_base
    assert r._held_asset.num_bodies == 1 and r._held_asset.num_joints == 0
    report['checks']['articulation_structure'] = True
    def snapshot():
        return {'finger_joints_m': r.joint_pos[:,7:].cpu().tolist(),
                'shaft_tip_m': r.held_pos.cpu().tolist(), 'hole_entry_m': r.fixed_pos_obs_frame.cpu().tolist(),
                'shaft_tip_clearance_m': (r.held_pos[:,2]-r.fixed_pos_obs_frame[:,2]).cpu().tolist(),
                'mass_kg': r._held_asset.root_physx_view.get_masses().cpu().tolist()}
    report['resets'] = []
    for trial in range(2):
        obs, _ = env.reset()
        assert all(torch.isfinite(v).all() for v in obs.values())
        record = snapshot()
        bias=r.init_fixed_pos_obs_noise.clone()
        record['hole_estimate_bias_m']=bias.cpu().tolist()
        assert torch.isfinite(bias).all()
        if a.pose==0:assert torch.count_nonzero(bias[:,:2])==0
        else:assert torch.count_nonzero(bias[:,:2])>0
        report['checks']['pose_bias_runtime']=True
        width = r.joint_pos[:,7:].sum(-1)
        assert bool(((width > .028) & (width < .042)).all()), ('invalid grasp width', record)
        assert bool(((r.held_pos[:,2]-r.fixed_pos_obs_frame[:,2]) > 0).all()), ('initial penetration',record)
        m = r._held_asset.root_physx_view.get_masses()
        assert bool(((m >= .012499) & (m <= .022501)).all()), ('mass outside official randomization', record)
        initial_fixed = r.fixed_pos.clone()
        relative = r.held_pos-r.fingertip_midpoint_pos
        action = r.prev_actions.clone()
        with torch.no_grad():
            for step in range(30):
                obs, rew, term, trunc, _ = env.step(action)
                assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(rew).all()
        slip = torch.linalg.vector_norm((r.held_pos-r.fingertip_midpoint_pos)-relative, dim=-1)
        record['hold_relative_translation_m'] = slip.cpu().tolist()
        record['fixed_motion_m'] = torch.linalg.vector_norm(r.fixed_pos-initial_fixed,dim=-1).cpu().tolist()
        assert bool((slip < .010).all()), ('grasp slipped over 10mm during 2s hold',record)
        assert torch.allclose(r.fixed_pos,initial_fixed,atol=1e-5)
        report['resets'].append(record)
        print('Reset/grasp/hold check', trial, record, flush=True)
    report['checks']['reset_grasp_hold'] = True
    # Exercise unchanged success function on controlled geometry; no simulated performance claim.
    saved = r.held_pos
    target = r.fixed_pos.clone()
    r.held_pos = target.clone()
    seated = r._get_curr_successes(r.cfg_task.success_threshold)
    r.held_pos[:,2] = target[:,2]+.002
    too_high = r._get_curr_successes(r.cfg_task.success_threshold)
    r.held_pos = target.clone(); r.held_pos[:,0] += .003
    off_axis = r._get_curr_successes(r.cfg_task.success_threshold)
    r.held_pos = saved
    assert seated.all() and not too_high.any() and not off_axis.any()
    report['checks']['official_success_geometry'] = True
    report['success_depth_min_m'] = .047*(1-.04)
    # Isolated collision probe: move robot away, place shaft in bore vs solid wall.
    ids = torch.arange(r.num_envs,device=r.device)
    r._set_franka_to_default_pose(r.cfg.ctrl.reset_joints,ids)
    probe = []
    for offset in [0.0, .025, a.clearance/4000, a.clearance/2000+.0005]:
        state = r._held_asset.data.default_root_state.clone()
        state[:,:3] = r._fixed_asset.data.root_pos_w.clone()
        # local X offset rotated with base
        import isaacsim.core.utils.torch as tu
        local = torch.zeros((r.num_envs,3),device=r.device); local[:,0]=offset; local[:,2]=.005
        state[:,:3] += tu.quat_apply(r._fixed_asset.data.root_quat_w,local)
        state[:,3:7] = r._fixed_asset.data.root_quat_w
        state[:,7:] = 0
        r._held_asset.write_root_state_to_sim(state)
        r._held_asset.reset()
        initial = state[:,:3].clone()
        for _ in range(20):
            r.step_sim_no_action()
        movement = torch.linalg.vector_norm(r._held_asset.data.root_pos_w-initial,dim=-1)
        probe.append({'lateral_offset_m': offset,'displacement_m': movement.cpu().tolist()})
    report['collision_probe'] = probe
    assert max(probe[0]['displacement_m']) < .002, ('bore blocked or unexpected contact',probe)
    assert min(probe[1]['displacement_m']) > .002, ('solid wall did not repel overlapping shaft',probe)
    report['checks']['bore_and_wall_collision'] = True
    report['small_clearance_contact_check']={
        'free_offset_m':probe[2]['lateral_offset_m'],
        'free_displacement_max_m':max(probe[2]['displacement_m']),
        'overlap_offset_m':probe[3]['lateral_offset_m'],
        'overlap_displacement_min_m':min(probe[3]['displacement_m']),
        'note':'20-step dynamic probe; does not certify machining clearance or SDF error bound'}
    assert max(probe[2]['displacement_m']) < .0001, ('Nominally free near-wall pose moves >0.1mm; investigate SDF/contact offsets',probe)
    assert min(probe[3]['displacement_m']) > .0001, ('Near-wall penetration not resolved >0.1mm',probe)
    report['status'] = 'passed'
except BaseException:
    report['status'] = 'failed'
    report['error'] = traceback.format_exc()
    print(report['error'],flush=True)
    raise
finally:
    Path('validation.json').write_text(json.dumps(report,indent=2))
    if env is not None:
        env.close()
    app.close()
