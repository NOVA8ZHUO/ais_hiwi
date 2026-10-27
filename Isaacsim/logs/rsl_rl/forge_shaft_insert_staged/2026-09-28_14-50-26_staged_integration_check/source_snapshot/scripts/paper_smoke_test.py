"""Check reset frames, persistent grip, residual targets and batched resets."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=8)
parser.add_argument('--steps', type=int, default=160)
parser.add_argument('--metrics_csv', default='', help='Completed-episode CSV output (overwritten).')
parser.add_argument('--mode', choices=['hold', 'directed', 'random'], default='hold')
parser.add_argument('--grip_diagnostics', action='store_true')
parser.add_argument('--nominal', action='store_true', help='Disable grasp/hole errors and force dead zone.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import json, sys
from pathlib import Path
from forge_shaft_insert import experiment_archive as archive
import torch
import forge_shaft_insert  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from forge_shaft_insert.forge_paper_env_cfg import ForgePaperEnvCfg
from forge_shaft_insert.mdp import forge_paper as paper


def main():
    cfg = ForgePaperEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.task.metrics_csv_path = args.metrics_csv
    if args.nominal:
        cfg.task.hole_position_error = (0., 0., 0.)
        cfg.task.grasp_position_error = (0., 0., 0.)
        cfg.task.deadzone_range = (0., 0.)
    if args.mode == 'hold':
        # Isolate physical grip stability while retaining all requested offsets.
        cfg.task.gripper_settle_s = cfg.episode_length_s + 1
        cfg.task.deadzone_range = (0., 0.)
    if args.grip_diagnostics:
        from isaaclab.sensors import ContactSensorCfg
        cfg.scene.robot.spawn.activate_contact_sensors = True
        for side in ('left', 'right'):
            setattr(cfg.scene, side + '_grip_contact', ContactSensorCfg(
                prim_path='{ENV_REGEX_NS}/Robot/panda_' + side + 'finger',
                filter_prim_paths_expr=['{ENV_REGEX_NS}/Shaft_large'], history_length=0))
    if args.metrics_csv:
        trace_path = Path(args.metrics_csv)
        archive_dir = trace_path.parent/(trace_path.stem+'_run')
        archive.begin(archive_dir, argv=sys.argv)
        (archive_dir/'settings.json').write_text(json.dumps(cfg.to_dict(), indent=2, default=str)+'\n')
    env = gym.make('FORGE-Shaft-Insert-Paper-v0', cfg=cfg)
    test_status = 'failed' 
    try:
        obs, _ = env.reset(seed=42)
        raw = env.unwrapped
        term = paper.action_term(raw)
        robot, shaft = raw.scene['robot'], raw.scene['shaft']
        assert obs['policy'].shape == (args.num_envs, 28)
        assert obs['critic'].shape == (args.num_envs, 52)
        assert raw.max_episode_length == 150
        gripper = robot.actuators['panda_hand']
        assert torch.allclose(gripper.effort_limit_sim, torch.full_like(gripper.effort_limit_sim, cfg.task.gripper_force))
        assert torch.allclose(gripper.velocity_limit_sim, torch.full_like(gripper.velocity_limit_sim, cfg.task.gripper_closing_speed / 2))
        assert (gripper.stiffness == 0).all() and (gripper.damping == cfg.task.gripper_joint_damping).all()
        print('GRIP_CONFIG_PASS force=20N aperture_speed=20mm/s width=35+/-5mm', flush=True)
        base_start = raw.scene['base'].data.root_pose_w.clone()
        hole_limits = torch.tensor(cfg.task.hole_position_error, device=raw.device)
        grasp_limits = torch.tensor(cfg.task.grasp_position_error, device=raw.device)

        def check_reset(ids):
            assert (term.hole_bias[ids].abs() <= hole_limits + 1e-7).all()
            assert (term.grasp_offset_hand[ids].abs() <= grasp_limits + 1e-7).all()
            assert (term.grasp_offset_hand[ids, 1] == 0).all()
            actual = shaft.data.root_pos_w[ids] - raw.scene.env_origins[ids]
            nominal = torch.tensor(cfg.task.shaft_reset_pos, device=raw.device)
            expected = nominal + quat_apply(term.reset_hand_quaternion.expand(len(ids), -1), term.grasp_offset_hand[ids])
            assert torch.allclose(actual, expected, atol=2e-6, rtol=0)
            assert torch.allclose(raw.scene['base'].data.root_pose_w, base_start, atol=1e-6, rtol=0)

        ids = torch.arange(args.num_envs, device=raw.device)
        check_reset(ids)
        if args.num_envs > 1:
            old_bias, old_grasp = term.hole_bias.clone(), term.grasp_offset_hand.clone()
            raw.reset(env_ids=ids[:1])
            check_reset(ids[:1])
            assert torch.equal(term.hole_bias[1:], old_bias[1:])
            assert torch.equal(term.grasp_offset_hand[1:], old_grasp[1:])
            obs, _ = env.reset(seed=42)
        print('RESET_CONTRACT_PASS', 'actor=28 critic=52; fixed base; local X/Z grasp errors; persistent XYZ hole bias', flush=True)
        print('SAMPLED_RANGES', 'grasp', term.grasp_offset_hand.amin(0).tolist(), term.grasp_offset_hand.amax(0).tolist(),
              'hole', term.hole_bias.amin(0).tolist(), term.hole_bias.amax(0).tolist(), flush=True)

        # Check action semantics, hold gate, and that residuals do not accumulate.
        previous_steps = term.elapsed_steps.clone()
        term.elapsed_steps[:] = int((cfg.task.gripper_settle_s + 1) / raw.step_dt)
        actions = torch.zeros(args.num_envs, 5, device=raw.device)
        actions[:, 0] = 0.1
        current, _, _, _ = term.frame()
        expected = paper.nominal_target(raw)
        expected[:, 0] += 0.1 * cfg.task.residual_position_scale[0]
        expected = current + (expected-current).clamp(-term.limit, term.limit)
        term.process_actions(actions)
        assert torch.allclose(term.target, expected, atol=1e-6)
        first_target = term.target.clone()
        term.elapsed_steps -= 1  # Compare residuals at the same nominal phase.
        term.process_actions(actions)
        assert torch.equal(term.target, first_target), 'Residual incorrectly accumulated.'
        term.elapsed_steps[:] = previous_steps
        # RSL-RL can randomize episode_length_buf before its first rollout;
        # this must not skip the actual gripper-settling stage.
        episode_lengths = raw.episode_length_buf.clone()
        raw.episode_length_buf[:] = 100
        assert not paper.insertion_enabled(raw).any()
        assert (paper.insertion_phase(raw) == 0).all()
        raw.episode_length_buf[:] = episode_lengths
        print('RESIDUAL_CONTRACT_PASS', flush=True)

        counts = {name: 0 for name in raw.termination_manager.active_terms}
        max_force = max_drift = max_slip = max_closing_speed = max_finger_asymmetry = 0.0
        min_width, max_width = 1.0, 0.0
        reference = torch.zeros(args.num_envs, 3, device=raw.device)
        reference_valid = torch.zeros(args.num_envs, dtype=torch.bool, device=raw.device)
        successful_steps = 0
        constant_width_force_steps = 0
        for step in range(args.steps):
            old_bias = term.hole_bias.clone()
            actions = torch.zeros(args.num_envs, 5, device=raw.device)
            actions[:, 4] = -1.0
            if args.mode == 'random':
                actions.uniform_(-1, 1)
            previous_width = paper.gripper_width(raw).clone()
            obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.isfinite(reward).all()
            assert torch.equal(term.hole_bias[~done], old_bias[~done])
            assert torch.allclose(raw.scene['base'].data.root_pose_w, base_start, atol=1e-6, rtol=0)
            assert (term.reference_velocity.norm(dim=-1) <= cfg.task.reference_speed + 1e-6).all()
            assert (term.reference_yaw_velocity.abs() <= cfg.task.reference_yaw_speed + 1e-6).all()
            assert (term.commanded_force.norm(dim=-1) <= cfg.task.translation_force_cap + 1e-5).all()
            assert torch.isfinite(paper.shaft_pitch_penalty(raw)).all()
            # This includes automatically reset environments: no width-based release.
            effort = robot.data.applied_torque[~done][:, term.fingers]
            assert (effort.abs() <= cfg.task.gripper_force + 1e-5).all()
            assert torch.allclose(robot.data.joint_effort_target[:, term.fingers],
                                  torch.full_like(robot.data.joint_effort_target[:, term.fingers], -cfg.task.gripper_force))
            # Aperture change below 0.01 mm per control step; contact velocities can jitter.
            width_stopped = (paper.gripper_width(raw) - previous_width).abs() < 1e-5
            constant_width_force_steps += int((width_stopped & paper.grasp_width_valid(raw) & ~done).sum())
            if done.any():
                check_reset(ids[done])
                reference_valid[done] = False
            for name in counts:
                counts[name] += int(raw.termination_manager.get_term(name).sum())
            force = paper.contact_force(raw).norm(dim=-1)
            p, q, _, _ = term.frame()
            relative = quat_apply_inverse(q, shaft.data.root_pos_w - robot.data.body_link_pos_w[:, term.body])
            settled = (term.elapsed_steps * raw.step_dt >= 0.6) & ~done
            newly_settled = settled & ~reference_valid
            reference[newly_settled] = relative[newly_settled]
            reference_valid[newly_settled] = True
            if settled.any():
                max_slip = max(max_slip, (relative[settled]-reference[settled]).norm(dim=-1).max().item())
                width = paper.gripper_width(raw)[settled]
                min_width = min(min_width, width.min().item())
                max_width = max(max_width, width.max().item())
            asymmetry = (robot.data.joint_pos[:, term.fingers[0]] - robot.data.joint_pos[:, term.fingers[1]]).abs().max().item()
            max_finger_asymmetry = max(max_finger_asymmetry, asymmetry)
            # Physical asymmetry is allowed and reported, not forced to zero.
            max_force = max(max_force, force.max().item())
            max_drift = max(max_drift, (p-term.reset_target).norm(dim=-1).max().item())
            max_closing_speed = max(max_closing_speed, (-robot.data.joint_vel[:, term.fingers].sum(-1)).max().item())
            successful_steps += int(paper.insertion_success(raw).sum())
            if args.grip_diagnostics and step % 30 == 0:
                for side in ('left', 'right'):
                    measured = raw.scene[side + '_grip_contact'].data.force_matrix_w.sum(dim=(1, 2)).norm(dim=-1)
                    print('GRIP_CONTACT', side, measured.tolist(), flush=True)
                print('GRIP_EFFORT', robot.data.applied_torque[:, term.fingers].tolist(), flush=True)
                print('GRIP_LOCAL_SHIFT', (relative-reference).tolist(), flush=True)
            if step % 30 == 0:
                print('RESIDUAL_STEP', step, 'width_mm', (paper.gripper_width(raw)*1000).tolist(),
                      'force_max', force.max().item(), 'grasp_drift_mm', max_slip*1000, flush=True)
        print('RESIDUAL_RESULT', dict(mode=args.mode, nominal=args.nominal, envs=args.num_envs,
              max_contact_force=max_force, max_tcp_displacement=max_drift, max_settled_grasp_drift=max_slip,
              width_range=(min_width, max_width), max_total_closing_speed=max_closing_speed,
              max_finger_asymmetry=max_finger_asymmetry,
              successful_steps=successful_steps, constant_width_force_steps=constant_width_force_steps, terminations=counts), flush=True)
        if args.mode == 'hold':
            assert counts['invalid_grasp_width'] == 0, 'Some randomized grasps failed the width check.'
            assert counts['outside_workspace'] == 0
            # Slip/rotation are physical outcomes, not zero-slip pass criteria.
            # Keep reporting them; validate width, workspace and finite state.
            if args.steps >= 150:
                assert counts['time_out'] >= args.num_envs
        print('EPISODE_METRICS', json.dumps(raw.episode_metrics.summary()), flush=True)
        print('RESIDUAL_TEST_PASS', flush=True)
        test_status = 'completed'
    finally:
        env.close()
        if args.metrics_csv:
            archive.finish(test_status)


exit_code = 0
try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    exit_code = 1
finally:
    app.close()
raise SystemExit(exit_code)
