# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the ordered A--G registration and environment smoke tests."""

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("register", "reset", "zero", "directed", "random"), default="reset")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--position_only", action="store_true")
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument(
    "--diagnostic_no_workspace_termination",
    action="store_true",
    help="Set workspace limits very large so an unstable shaft can be observed without auto-reset.",
)
parser.add_argument("--keep_open", action="store_true", help="Keep stepping zero actions for GUI inspection.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import forge_shaft_insert  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _policy_observation(observation):
    return observation["policy"] if isinstance(observation, dict) else observation


def main() -> None:
    task_name = "FORGE-Shaft-Insert-Position-Only-v0" if args_cli.position_only else "FORGE-Shaft-Insert-v0"
    spec = gym.spec(task_name)
    print(f"[A registration] id={spec.id}, entry_point={spec.entry_point}")

    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.diagnostic_no_workspace_termination:
        env_cfg.task.workspace_xy_limit = 10.0
        env_cfg.task.workspace_z_tolerance = 10.0
    env = gym.make(task_name, cfg=env_cfg)
    observation, _ = env.reset(seed=42)
    policy_obs = _policy_observation(observation)
    expected_dim = 3
    assert policy_obs.shape == (args_cli.num_envs, expected_dim), policy_obs.shape
    assert torch.isfinite(policy_obs).all(), "Reset observation contains NaN or Inf."
    expected_reset_error = torch.tensor(
        env_cfg.task.shaft_reset_pos,
        device=env.unwrapped.device,
        dtype=policy_obs.dtype,
    ) - (
        torch.tensor(env_cfg.task.base_pos, device=env.unwrapped.device, dtype=policy_obs.dtype)
        + torch.tensor(env_cfg.task.hole_offset_in_base, device=env.unwrapped.device, dtype=policy_obs.dtype)
    )
    reset_tolerance = 5.0e-4
    assert torch.all(torch.abs(policy_obs - expected_reset_error) <= reset_tolerance), (
        policy_obs,
        expected_reset_error,
    )

    base = env.unwrapped.scene["base"]
    shaft = env.unwrapped.scene["shaft"]
    robot = env.unwrapped.scene["robot"]
    expected_base_pos = env.unwrapped.scene.env_origins + torch.tensor(
        env_cfg.task.base_pos,
        device=env.unwrapped.device,
        dtype=base.data.root_pos_w.dtype,
    )
    assert torch.allclose(base.data.root_pos_w, expected_base_pos, atol=1.0e-6, rtol=0.0), base.data.root_pos_w
    finger_ids, finger_names = robot.find_joints("panda_finger_joint[1-2]", preserve_order=True)
    diagnostic_body_ids, diagnostic_body_names = robot.find_bodies(
        "panda_hand|panda_leftfinger|panda_rightfinger|panda_fingertip_centered",
        preserve_order=True,
    )
    hand_body_id = diagnostic_body_ids[diagnostic_body_names.index("panda_hand")]
    reset_shaft_in_hand = shaft.data.root_pos_w - robot.data.body_pos_w[:, hand_body_id]

    print(f"[B reset] observation_shape={tuple(policy_obs.shape)} action_shape={env.action_space.shape}")
    print("[C coordinates] observation = Shaft_large origin - hole_large origin, world XYZ [m]")
    print(
        f"[C config] base_pos={env_cfg.task.base_pos} m "
        f"hole_offset_in_base={env_cfg.task.hole_offset_in_base} m "
        f"shaft_reset_pos={env_cfg.task.shaft_reset_pos} m"
    )
    print(f"[C config] panda_hand_to_fingertip_offset={env_cfg.task.tcp_offset} m")
    print(f"[C base env0] {base.data.root_pos_w[0].detach().cpu().tolist()} m (fixed)")
    print(f"[C shaft env0] {shaft.data.root_pos_w[0].detach().cpu().tolist()} m (dynamic, gravity on)")
    print(
        f"[C gripper env0] names={finger_names} "
        f"positions={robot.data.joint_pos[0, finger_ids].detach().cpu().tolist()} m "
        f"command={env_cfg.task.gripper_command:.6f} m"
    )
    for body_id, body_name in zip(diagnostic_body_ids, diagnostic_body_names):
        print(
            f"[C robot body env0] name={body_name} "
            f"xyz={robot.data.body_pos_w[0, body_id].detach().cpu().tolist()} "
            f"quat_wxyz={robot.data.body_quat_w[0, body_id].detach().cpu().tolist()}"
        )
    print(f"[C expected reset observation] {expected_reset_error.detach().cpu().tolist()}")
    print(f"[C observation env0] {policy_obs[0].detach().cpu().tolist()}")

    if args_cli.mode in ("zero", "directed", "random"):
        success_total = 0
        terminated_total = 0
        timeout_total = 0
        auto_reset_total = 0
        max_reset_drift = 0.0
        max_relative_grasp_drift = 0.0
        term_totals = {name: 0 for name in env.unwrapped.termination_manager.active_terms}
        for step in range(args_cli.steps):
            if args_cli.mode == "zero":
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            elif args_cli.mode == "directed":
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                actions[:, :2] = torch.clamp(
                    -policy_obs[:, :2] / env_cfg.task.action_scale,
                    min=-1.0,
                    max=1.0,
                )
            else:
                actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            observation, reward, terminated, truncated, _ = env.step(actions)
            policy_obs = _policy_observation(observation)
            step_reset_drift = torch.max(torch.abs(policy_obs - expected_reset_error)).item()
            max_reset_drift = max(max_reset_drift, step_reset_drift)
            shaft_in_hand = shaft.data.root_pos_w - robot.data.body_pos_w[:, hand_body_id]
            step_relative_drift = torch.max(torch.abs(shaft_in_hand - reset_shaft_in_hand)).item()
            max_relative_grasp_drift = max(max_relative_grasp_drift, step_relative_drift)
            assert torch.isfinite(policy_obs).all(), f"Observation became non-finite at step {step}."
            assert torch.isfinite(reward).all(), f"Reward became non-finite at step {step}."
            assert torch.allclose(base.data.root_pos_w, expected_base_pos, atol=1.0e-6, rtol=0.0), (
                f"Base moved at step {step}: {base.data.root_pos_w}"
            )
            terminated_total += int(terminated.sum().item())
            timeout_total += int(truncated.sum().item())
            for name in term_totals:
                term_totals[name] += int(env.unwrapped.termination_manager.get_term(name).sum().item())
            success = env.unwrapped.termination_manager.get_term("success")
            success_total += int(success.sum().item())
            done = terminated | truncated
            if done.any():
                # ManagerBasedRLEnv returns the freshly reset observation for
                # environments that terminated during this step.
                reset_obs = policy_obs[done]
                assert torch.all(torch.abs(reset_obs - expected_reset_error) <= reset_tolerance), reset_obs
                auto_reset_total += int(done.sum().item())
            if step in (0, args_cli.steps - 1):
                print(
                    f"[{args_cli.mode} step={step + 1}] obs0={policy_obs[0].detach().cpu().tolist()} "
                    f"reward_mean={reward.mean().item():.6f} done={int((terminated | truncated).sum().item())}"
                )
            if args_cli.mode == "zero" and (step < 10 or step % 10 == 9):
                print(
                    f"[zero diagnostic step={step + 1}] shaft0={shaft.data.root_pos_w[0].detach().cpu().tolist()} "
                    f"hand0={robot.data.body_pos_w[0, hand_body_id].detach().cpu().tolist()} "
                    f"fingers0={robot.data.joint_pos[0, finger_ids].detach().cpu().tolist()} "
                    f"world_drift={step_reset_drift:.6f} m relative_drift={step_relative_drift:.6f} m"
                )
        print(
            f"[{args_cli.mode} summary] envs={args_cli.num_envs} steps={args_cli.steps} "
            f"success_events={success_total} terminated_events={terminated_total} timeout_events={timeout_total} "
            f"auto_resets_checked={auto_reset_total} causes={term_totals}"
        )
        if args_cli.mode == "directed":
            assert success_total > 0, "Directed frame-check controller never reached the XY success threshold."
        if args_cli.mode == "zero":
            # With gravity enabled, this catches a dropped or welded-at-the-wrong-pose shaft.
            print(
                f"[zero grasp check] max_world_error_from_reset={max_reset_drift:.6f} m "
                f"max_shaft_in_hand_drift={max_relative_grasp_drift:.6f} m"
            )
            assert terminated_total == 0, "Zero action triggered a non-timeout termination."
            assert max_relative_grasp_drift <= 0.001, "Shaft slipped relative to the closed gripper."
            assert max_reset_drift <= 0.005, "Zero-action Cartesian controller drift exceeded 5 mm."
        if args_cli.steps > env.unwrapped.max_episode_length:
            assert auto_reset_total > 0, "No termination/timeout auto-reset was observed."

    if args_cli.keep_open:
        print("[GUI] Zero-action hold active. Close the Isaac Sim window or press Ctrl+C to exit.")
        hold_step = 0
        while simulation_app.is_running():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            observation, _, _, _, _ = env.step(actions)
            policy_obs = _policy_observation(observation)
            assert torch.allclose(base.data.root_pos_w, expected_base_pos, atol=1.0e-6, rtol=0.0), (
                f"Base moved during GUI hold: {base.data.root_pos_w}"
            )
            if hold_step % 300 == 0:
                print(
                    f"[GUI hold step={hold_step}] base0={base.data.root_pos_w[0].detach().cpu().tolist()} "
                    f"shaft0={shaft.data.root_pos_w[0].detach().cpu().tolist()} "
                    f"obs0={policy_obs[0].detach().cpu().tolist()}"
                )
            hold_step += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
