"""FORGE insertion terms, separate from the validated stage-1 diagnostics.

The control point is a virtual TCP at the nominal held-shaft bottom. Its
transform is fixed, so actor observations never use the true held-object pose.
"""
import torch

from isaaclab.managers import ActionTerm
from isaaclab.utils.math import (
    quat_apply, quat_apply_inverse, quat_from_euler_xyz, compute_pose_error, skew_symmetric_matrix,
)
from .events import reset_grasp_state
from .insertion_supervisor import InsertionSupervisor
from .observations import hole_target_position_w
from .forge_math import logistic_kernel, force_excess, dead_zone, bounded_reference, pitch_cost


class ForgeImpedanceAction(ActionTerm):
    """Residual xyz/yaw over a nominal insertion, plus success prediction."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.joints, _ = self._asset.find_joints('panda_joint[1-7]', preserve_order=True)
        self.body = self._asset.find_bodies('panda_hand')[0][0]
        self.fingers, _ = self._asset.find_joints('panda_finger_joint[1-2]', preserve_order=True)
        self._raw = torch.zeros(self.num_envs, 5, device=self.device)
        self.elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_quat[:, 1] = 1.0
        self.kp = torch.full((self.num_envs, 1), 600.0, device=self.device)
        self.limit = torch.full_like(self.kp, 0.02)
        self.deadzone = torch.zeros(self.num_envs, 3, device=self.device)
        self.threshold = torch.full((self.num_envs,), 7.5, device=self.device)
        self.hole_bias = torch.zeros_like(self.target)
        self.grasp_offset_hand = torch.zeros_like(self.target)
        self.reset_target = torch.zeros_like(self.target)
        self.reset_hand_quaternion = torch.tensor(env.cfg.task.reset_hand_quaternion, device=self.device)
        self.reset_hand_quaternion /= self.reset_hand_quaternion.norm()
        hand_position = torch.tensor(env.cfg.task.reset_hand_position, device=self.device)
        control_offset = torch.tensor(env.cfg.task.control_point_in_hand, device=self.device)
        self.reset_target[:] = env.scene.env_origins + hand_position + quat_apply(
            self.reset_hand_quaternion[None], control_offset[None])
        self.reference_position = self.reset_target.clone()
        self.reference_velocity = torch.zeros_like(self.target)
        self.reference_yaw = torch.zeros(self.num_envs, 1, device=self.device)
        self.reference_yaw_velocity = torch.zeros_like(self.reference_yaw)
        self.yaw_goal = torch.zeros_like(self.reference_yaw)
        self.commanded_force = torch.zeros_like(self.target)
        self.rotation_damping = torch.full((self.num_envs, 1), env.cfg.task.rotation_damping, device=self.device)
        self.max_physics_speed = torch.zeros(self.num_envs, device=self.device)
        self.supervisor = InsertionSupervisor(self.num_envs, self.device)
        self.press_force = torch.zeros(self.num_envs, device=self.device)
        self.releasing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_dwell = torch.zeros(self.num_envs, device=self.device)
        self.release_age = torch.zeros_like(self.success_dwell)
        self.release_position = torch.zeros_like(self.target)
        self.prev_noisy_position = torch.zeros_like(self.target)
        self.obs = None
        self.obs_step = -1
        self.reward_stamp = torch.full_like(self.elapsed_steps, -1)
        self.reward_best_depth = torch.zeros_like(self.press_force)
        self.reward_anchor_depth = torch.zeros_like(self.press_force)
        self.reward_stall_age = torch.zeros_like(self.press_force)
        self.reward_values = torch.zeros(self.num_envs, 4, device=self.device)
        self.fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self):
        return 5

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return torch.cat((self.target, self._raw[:, 3:]), dim=-1)

    def frame(self):
        robot = self._asset
        q = robot.data.body_link_quat_w[:, self.body]
        offset = torch.tensor(self._env.cfg.task.control_point_in_hand, device=self.device).expand(self.num_envs, -1)
        r = quat_apply(q, offset)
        p = robot.data.body_link_pos_w[:, self.body] + r
        # PhysX geometric Jacobians use link origins; shift linear rows to TCP.
        jac = robot.root_physx_view.get_jacobians()[:, self.body - 1, :, self.joints].clone()
        jac[:, :3] -= torch.bmm(skew_symmetric_matrix(r), jac[:, 3:])
        velocity = torch.bmm(jac, robot.data.joint_vel[:, self.joints, None]).squeeze(-1)
        return p, q, jac, velocity

    def process_actions(self, actions):
        self._raw[:] = actions.clamp(-1.0, 1.0)
        current, _, _, _ = self.frame()
        task = self._env.cfg.task
        # Simulation-only release gate uses true geometry; deployment needs an independent depth estimate.
        if task.release_on_success:
            valid = insertion_success(self._env) & ~self.releasing
            self.success_dwell = torch.where(valid, self.success_dwell+self._env.step_dt, 0.)
            begin_release = valid & (self.success_dwell >= task.release_confirm_s)
            self.release_position[begin_release] = current[begin_release]
            self.reference_position[begin_release] = current[begin_release]
            self.reference_velocity[begin_release] = 0.
            self.releasing |= begin_release
            self.release_age[self.releasing] += self._env.step_dt
        scale = torch.tensor(task.residual_position_scale, device=self.device)
        desired = nominal_target(self._env) + self._raw[:, :3] * scale
        ready = insertion_enabled(self._env)
        desired = torch.where(ready[:, None], desired, self.reset_target)
        if task.staged_insertion:
            self.supervisor.update(ready, current, hole_target_position_w(self._env)+self.hole_bias,
                                   contact_force(self._env), self.reference_yaw, self._env.step_dt, task)
            locked = self.supervisor.mode >= 2
            desired[:, :2] = torch.where(locked[:, None], self.supervisor.locked_xy, desired[:, :2])
        desired = torch.where(self.releasing[:, None], self.release_position, desired)
        self.target[:] = current + (desired - current).clamp(-self.limit, self.limit)
        zero = torch.zeros(self.num_envs, device=self.device)
        yaw = torch.where(ready, self._raw[:, 3] * task.yaw_bound, zero)
        if task.staged_insertion:
            yaw = torch.where(self.supervisor.mode >= 2, self.supervisor.locked_yaw[:, 0], yaw)
        yaw = torch.where(self.releasing, self.reference_yaw[:, 0], yaw)
        self.yaw_goal[:, 0] = yaw
        self.elapsed_steps += 1

    def apply_actions(self):
        p, q, jac, velocity = self.frame()
        self.max_physics_speed[:] = torch.maximum(self.max_physics_speed, velocity[:, :3].norm(dim=-1))
        task = self._env.cfg.task
        dt = self._env.physics_dt
        self.reference_position[:], self.reference_velocity[:] = bounded_reference(
            self.reference_position, self.reference_velocity, self.target, dt,
            task.reference_speed, task.reference_acceleration)
        self.reference_yaw[:], self.reference_yaw_velocity[:] = bounded_reference(
            self.reference_yaw, self.reference_yaw_velocity, self.yaw_goal, dt,
            task.reference_yaw_speed, task.reference_yaw_acceleration)
        zero = torch.zeros(self.num_envs, device=self.device)
        self.target_quat[:] = quat_from_euler_xyz(zero + torch.pi, zero, self.reference_yaw[:, 0])
        position_error, rotation_error = compute_pose_error(p, q, self.reference_position, self.target_quat, rot_error_type='axis_angle')
        # Dead zone only affects XY spring drive, never damping or vertical drive.
        drive = self.kp * position_error
        drive[:, :2] = dead_zone(drive[:, :2], self.deadzone[:, :2])
        xy_overspeed = velocity[:, :2].norm(dim=-1) > task.actual_xy_brake_speed
        drive[xy_overspeed, :2] = 0.
        force = drive - 2*self.kp.sqrt()*velocity[:, :3]
        pressed = self.supervisor.mode >= 2 if task.staged_insertion else torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        contact_search = (self.supervisor.mode == 1) & self.supervisor.contact_seen if task.staged_insertion else torch.zeros_like(pressed)
        force_mode = (pressed | contact_search) & ~self.releasing
        low, high = task.press_force_range
        requested_press = low + (self._raw[:, 2]+1)*.5*(high-low)
        requested_press = torch.where(contact_search, task.search_contact_force, requested_press)
        measured_force = contact_force(self._env).norm(dim=-1)
        relief = ((task.press_stop_N-measured_force)/(task.press_stop_N-task.press_relief_N)).clamp(0., 1.)
        requested_press = torch.where(force_mode, requested_press*relief, 0.)
        self.press_force += (requested_press-self.press_force).clamp(-task.press_force_slew*dt, task.press_force_slew*dt)
        # Hybrid vertical force mode: small continuous press at rest, taper at downward speed.
        speed_scale = (1.+velocity[:, 2]/task.press_down_speed).clamp(0., 1.)
        press_z = -self.press_force*speed_scale - 2*self.kp[:, 0].sqrt()*velocity[:, 2]
        force[:, 2] = torch.where(force_mode, press_z, force[:, 2])
        z_overspeed = velocity[:, 2] < -task.actual_z_brake_speed
        force[z_overspeed, 2] = (-2*self.kp[z_overspeed, 0].sqrt()*velocity[z_overspeed, 2]).clamp_min(0.)
        # Independent budgets: lateral corrections cannot consume the vertical budget.
        force[:, :2] *= (task.xy_force_cap/force[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.)
        force[:, 2].clamp_(-task.z_force_cap, task.z_force_cap)
        self.commanded_force[:] = force
        rot_kp = self._env.cfg.task.rotation_stiffness
        torque = rot_kp * rotation_error - self.rotation_damping * velocity[:, 3:]
        wrench = torch.cat((force, torque), dim=-1)
        effort = torch.bmm(jac.transpose(1, 2), wrench.unsqueeze(-1)).squeeze(-1)
        effort += self._asset.root_physx_view.get_gravity_compensation_forces()[:, self.joints]
        self._asset.set_joint_effort_target(effort, joint_ids=self.joints)
        self.hold_gripper()

    def hold_gripper(self, env_ids=None):
        """Always close with force: width acceptance never switches this off."""
        count = self.num_envs if env_ids is None else len(env_ids)
        effort = torch.full((count, len(self.fingers)), -self._env.cfg.task.gripper_force, device=self.device)
        ids = slice(None) if env_ids is None else env_ids
        opening = self.releasing[ids]
        width = self._asset.data.joint_pos[ids][:, self.fingers].sum(dim=-1)
        open_effort = torch.where(width < self._env.cfg.task.release_width,
                                  self._env.cfg.task.gripper_force, 0.)
        effort = torch.where(opening[:, None], open_effort[:, None], effort)
        self._asset.set_joint_effort_target(effort, joint_ids=self.fingers, env_ids=env_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.releasing[ids] = False
        self.success_dwell[ids] = 0
        self.release_age[ids] = 0
        self.release_position[ids] = 0
        self.supervisor.reset(ids)
        self.press_force[ids] = 0
        self.reference_position[ids] = self.reset_target[ids]
        self.reference_velocity[ids] = 0
        self.reference_yaw[ids] = 0
        self.reference_yaw_velocity[ids] = 0
        self.yaw_goal[ids] = 0
        self.commanded_force[ids] = 0
        self.max_physics_speed[ids] = 0
        self._raw[ids] = 0
        self.elapsed_steps[ids] = 0
        self.reward_stamp[ids] = -1
        self.reward_best_depth[ids] = 0
        self.reward_anchor_depth[ids] = 0
        self.reward_stall_age[ids] = 0
        self.reward_values[ids] = 0
        self.fresh[ids] = True
        self.obs_step = -1
        self.hold_gripper(env_ids)


def action_term(env):
    return env.action_manager.get_term('arm_action')


def reset_paper(env, env_ids):
    reset_grasp_state(env, env_ids, gripper_command=env.cfg.task.gripper_expected_width / 2.0)
    task = env.cfg.task
    base = env.scene['base']
    state = base.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    # Fixed base: reset is already above hole_large; no pickup-to-hole travel.
    base.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    base.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]), env_ids=env_ids)
    term = action_term(env)
    for target, bounds in ((term.kp, task.kp_range), (term.limit, task.action_limit_range),
                           (term.deadzone, task.deadzone_range), (term.threshold, task.force_threshold_range)):
        target[env_ids] = torch.empty_like(target[env_ids]).uniform_(*bounds)
    hole_range = torch.tensor(task.hole_position_error, device=env.device)
    term.hole_bias[env_ids] = (2 * torch.rand(len(env_ids), 3, device=env.device) - 1) * hole_range
    grasp_range = torch.tensor(task.grasp_position_error, device=env.device)
    term.grasp_offset_hand[env_ids] = (2 * torch.rand(len(env_ids), 3, device=env.device) - 1) * grasp_range
    shaft = env.scene['shaft']
    shaft_state = shaft.data.default_root_state[env_ids].clone()
    shaft_state[:, :3] += env.scene.env_origins[env_ids]
    quaternion = term.reset_hand_quaternion.expand(len(env_ids), -1)
    shaft_state[:, :3] += quat_apply(quaternion, term.grasp_offset_hand[env_ids])
    shaft_state[:, 7:] = 0.0
    shaft.write_root_pose_to_sim(shaft_state[:, :7], env_ids=env_ids)
    shaft.write_root_velocity_to_sim(shaft_state[:, 7:], env_ids=env_ids)
    term.fresh[env_ids] = True
    term.obs_step = -1


def gripper_width(env):
    term = action_term(env)
    return env.scene['robot'].data.joint_pos[:, term.fingers].sum(dim=-1)


def grasp_width_valid(env):
    task = env.cfg.task
    return (gripper_width(env) - task.gripper_expected_width).abs() <= task.gripper_width_tolerance


def insertion_phase(env):
    task = env.cfg.task
    elapsed = (action_term(env).elapsed_steps * env.step_dt - task.gripper_settle_s).clamp_min(0)
    distance = task.approach_height + task.insertion_depth
    return (elapsed * task.insertion_speed / distance).clamp(0, 1)


def insertion_enabled(env):
    return (action_term(env).elapsed_steps * env.step_dt >= env.cfg.task.gripper_settle_s) & grasp_width_valid(env)


def nominal_target(env):
    task = env.cfg.task
    term = action_term(env)
    desired = hole_target_position_w(env) + term.hole_bias
    desired[:, 2] += task.approach_height - insertion_phase(env) * (task.approach_height + task.insertion_depth)
    return torch.where(insertion_enabled(env)[:, None], desired, term.reset_target)


def invalid_grasp_width(env):
    # Width is a hardware-style grasp check, not proof of force or zero slip.
    elapsed = action_term(env).elapsed_steps * env.step_dt
    return (elapsed >= env.cfg.task.gripper_check_delay_s) & ~grasp_width_valid(env) & ~action_term(env).releasing


def contact_force(env):
    """Shaft-on-base external contact proxy; excludes finger squeeze contacts."""
    matrix = env.scene['insertion_contact'].data.force_matrix_w
    if matrix is None:
        raise RuntimeError('FORGE requires the shaft-to-base filtered contact sensor.')
    return matrix.sum(dim=(1, 2))


def actor_state(env):
    term = action_term(env)
    # ObservationManager probes shapes before the first reset. Cache one noisy
    # sample per policy step so repeated reads cannot change finite differences.
    if term.obs_step == env.common_step_counter and term.obs is not None:
        return term.obs
    p, q, _, vel = term.frame()
    noisy = p + torch.randn_like(p) * env.cfg.task.ee_position_noise_std
    linear = (noisy - term.prev_noisy_position) / env.step_dt
    linear[term.fresh] = 0
    term.prev_noisy_position[:] = noisy
    term.fresh[:] = False
    force = contact_force(env) + torch.randn_like(p) * env.cfg.task.force_noise_std
    relative = noisy - hole_target_position_w(env) - term.hole_bias
    term.obs = torch.cat((relative, q, linear, vel[:, 3:], force,
                          term.threshold[:, None], term.raw_actions,
                          insertion_phase(env)[:, None], gripper_width(env)[:, None], term.supervisor.observation(), term.releasing[:, None].float()), dim=-1)
    term.obs_step = env.common_step_counter
    return term.obs


def tip_and_axis(env):
    shaft = env.scene['shaft']
    axis_local = torch.zeros(env.num_envs, 3, device=env.device)
    axis_local[:, 2] = 1
    axis = quat_apply(shaft.data.root_quat_w, axis_local)
    tip = shaft.data.root_pos_w + axis * env.cfg.task.shaft_bottom_offset
    return tip, axis


def insertion_success(env):
    tip, axis = tip_and_axis(env)
    error = tip - hole_target_position_w(env)
    depth = -error[:, 2]
    # User's teleoperation criterion: depth within the target bore region, no precision/tilt gate.
    return ((error[:, :2].norm(dim=-1) <= env.cfg.task.insertion_region_radius)
            & (depth >= env.cfg.task.success_depth)
            & (depth <= env.cfg.task.insertion_depth + 0.01))


def fully_seated(env):
    tip, axis = tip_and_axis(env)
    error = tip - hole_target_position_w(env)
    return ((error[:, :2].norm(dim=-1) <= env.cfg.task.success_xy_tolerance)
            & ((-error[:, 2]-env.cfg.task.insertion_depth).abs() <= env.cfg.task.depth_tolerance)
            & (axis[:, 2] >= env.cfg.task.success_axis_cos))


def release_complete(env):
    term = action_term(env)
    return term.releasing & (term.release_age >= env.cfg.task.release_observe_s)


def keypoint_reward(env):
    tip, axis = tip_and_axis(env)
    target = hole_target_position_w(env).clone()
    target[:, 2] -= env.cfg.task.success_depth
    offsets = torch.tensor(env.cfg.task.keypoint_offsets, device=env.device)
    actual_points = tip[:, None, :] + axis[:, None, :] * offsets[None, :, None]
    goal_points = target[:, None, :].repeat(1, len(offsets), 1)
    goal_points[:, :, 2] += offsets
    distance = (actual_points - goal_points).norm(dim=-1).mean(dim=-1)
    task = env.cfg.task
    return logistic_kernel(distance, *task.coarse_kernel) + logistic_kernel(distance, *task.fine_kernel)


def place_bonus(env):
    tip, axis = tip_and_axis(env)
    error = tip - hole_target_position_w(env)
    # A centred approach or valid inserted state; deep interpenetration fails.
    return ((error[:, :2].norm(dim=-1) <= env.cfg.task.place_distance)
            & (error[:, 2] <= env.cfg.task.place_distance)
            & (error[:, 2] >= -env.cfg.task.insertion_depth - env.cfg.task.depth_tolerance)
            & (axis[:, 2] >= env.cfg.task.success_axis_cos)).float()


def inserted_bonus(env):
    return insertion_success(env).float()


def excessive_force(env):
    return force_excess(contact_force(env), action_term(env).threshold)


def prediction_error(env):
    prediction = (action_term(env).raw_actions[:, 4] + 1) * 0.5
    return (prediction - insertion_success(env).float()).abs()


def critic_state(env):
    term = action_term(env)
    p, q, _, vel = term.frame()
    shaft = env.scene['shaft']
    hole = hole_target_position_w(env)
    return torch.cat((p-hole, q, vel, shaft.data.root_pos_w-hole,
                      shaft.data.root_quat_w, shaft.data.root_vel_w, contact_force(env),
                      term.kp, term.limit, term.deadzone, term.threshold[:, None], term.raw_actions,
                      insertion_phase(env)[:, None], gripper_width(env)[:, None],
                      term.hole_bias, term.grasp_offset_hand, term.supervisor.observation(), term.releasing[:, None].float()), dim=-1)


def failed_workspace(env):
    tip, _ = tip_and_axis(env)
    error = tip - hole_target_position_w(env)
    return ((error[:, :2].norm(dim=-1) > env.cfg.task.workspace_xy_limit)
            | (error[:, 2] < -env.cfg.task.insertion_depth - 0.01)
            | (error[:, 2] > 0.10))


def shaft_pitch_penalty(env):
    """Soft Y-axis tilt relative to the gripper; nominal shaft direction is hand -Z."""
    term = action_term(env)
    _, axis = tip_and_axis(env)
    hand_q = env.scene['robot'].data.body_link_quat_w[:, term.body]
    local_axis = quat_apply_inverse(hand_q, axis)
    pitch = torch.atan2(local_axis[:, 0], -local_axis[:, 2])
    return pitch_cost(pitch, env.cfg.task.shaft_pitch_tolerance, env.cfg.task.shaft_pitch_scale)


def insertion_reward_v2(env, component: int):
    """Cached per-environment update, evaluated after physics; no actor truth inputs."""
    term = action_term(env)
    stale = term.reward_stamp != term.elapsed_steps
    if stale.any():
        tip, _ = tip_and_axis(env)
        error = tip - hole_target_position_w(env)
        task = env.cfg.task
        depth = -error[:, 2]
        inside = error[:, :2].norm(dim=-1) <= task.insertion_region_radius
        active = insertion_enabled(env) & ~term.releasing
        valid = inside & (depth >= 0) & (depth <= task.insertion_depth + .01) & active
        candidate = torch.where(valid, depth.clamp(0, task.success_depth), term.reward_best_depth)
        best = torch.maximum(candidate, term.reward_best_depth)
        gain_mm = (best-term.reward_best_depth)*1000
        advanced = best >= term.reward_anchor_depth + .0005
        shallow = valid & (depth >= .002) & (depth < task.success_depth)
        age = torch.where(shallow & ~advanced, term.reward_stall_age+env.step_dt, 0.)
        stalled = shallow & (age >= 1.0)
        lateral = contact_force(env)[:, :2].norm(dim=-1)
        outside = active & ~inside & (depth > .002)
        values = torch.stack((gain_mm, stalled.float(),
                              stalled.float()*(lateral-2.).clamp(0, 5), outside.float()), dim=-1)
        term.reward_values[stale] = values[stale]
        term.reward_best_depth[stale] = best[stale]
        term.reward_anchor_depth[stale] = torch.where(advanced, best, term.reward_anchor_depth)[stale]
        term.reward_stall_age[stale] = age[stale]
        term.reward_stamp[stale] = term.elapsed_steps[stale]
    return term.reward_values[:, component]


def release_seated_bonus(env):
    return (release_complete(env) & fully_seated(env)).float()
