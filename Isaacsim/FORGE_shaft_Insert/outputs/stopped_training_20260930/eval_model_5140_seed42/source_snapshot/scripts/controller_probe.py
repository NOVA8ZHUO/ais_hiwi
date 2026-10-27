"""Paired hold/yaw tests: old vs explicit low angular damping, no insertion."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--steps', type=int, default=90)
parser.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import json, sys
from pathlib import Path
import gymnasium as gym
import torch
import forge_shaft_insert
from forge_shaft_insert import experiment_archive as archive
from forge_shaft_insert.forge_paper_env_cfg import ForgePaperEnvCfg
from forge_shaft_insert.mdp import forge_paper as paper
from isaaclab.utils.math import quat_apply_inverse

out = Path(args.output)
archive.begin(out, argv=sys.argv)
env = None
try:
    cfg = ForgePaperEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.task.grasp_position_error = (0., 0., 0.)
    cfg.task.hole_position_error = (0., 0., 0.)
    cfg.task.kp_range = (600., 600.)
    cfg.task.deadzone_range = (0., 0.)
    cfg.task.part_friction_range = (.8, .8)
    cfg.task.insertion_speed = 0.
    cfg.task.staged_insertion = False
    for name in ('shaft_friction', 'base_friction'):
        getattr(cfg.events, name).params['static_friction_range'] = (.8, .8)
        getattr(cfg.events, name).params['dynamic_friction_range'] = (.8, .8)
    cfg.task.metrics_csv_path = str(out/'episodes.csv')
    env = gym.make('FORGE-Shaft-Insert-Paper-v0', cfg=cfg)
    obs, _ = env.reset(seed=42)
    raw = env.unwrapped
    term = paper.action_term(raw)
    term.rotation_damping[:, 0] = torch.tensor([30**.5*2, .5, 30**.5*2, .5], device=raw.device)
    labels = ['hold_old_damping', 'hold_low_damping', 'yaw_old_damping', 'yaw_low_damping']
    initial, hand_q, _, _ = term.frame()
    initial = initial.clone()
    shaft = raw.scene['shaft']
    local0 = quat_apply_inverse(hand_q, shaft.data.root_pos_w - raw.scene['robot'].data.body_link_pos_w[:, term.body])
    peak_error = torch.zeros(4, device=raw.device)
    peak_speed = peak_error.clone()
    peak_slip = peak_error.clone()
    peak_tilt = peak_error.clone()
    counts = torch.zeros(4, device=raw.device)
    for step in range(args.steps):
        action = torch.zeros(4, 5, device=raw.device)
        action[:, 4] = -1.
        action[2:, 3] = .8
        obs, reward, terminated, truncated, info = env.step(action)
        p, q, _, velocity = term.frame()
        local = quat_apply_inverse(q, shaft.data.root_pos_w - raw.scene['robot'].data.body_link_pos_w[:, term.body])
        peak_error = torch.maximum(peak_error, (p-initial).norm(dim=-1))
        peak_speed = torch.maximum(peak_speed, velocity[:, :3].norm(dim=-1))
        peak_slip = torch.maximum(peak_slip, (local-local0).norm(dim=-1))
        _, axis = paper.tip_and_axis(raw)
        peak_tilt = torch.maximum(peak_tilt, torch.rad2deg(torch.acos(axis[:, 2].clamp(-1, 1))))
        counts += terminated | truncated
    result = {label: dict(peak_tcp_displacement_m=float(peak_error[i]), peak_sampled_speed_m_s=float(peak_speed[i]),
                           peak_grasp_relative_shift_m=float(peak_slip[i]), peak_world_tilt_deg=float(peak_tilt[i]),
                           resets=int(counts[i])) for i,label in enumerate(labels)}
    (out/'probe.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'task_parameters.json').write_text(json.dumps(cfg.to_dict(),indent=2,default=str)+'\n')
    print('CONTROLLER_PROBE', json.dumps(result), flush=True)
finally:
    if env is not None:
        env.close()
    archive.finish('probe_completed')
    app.close()
