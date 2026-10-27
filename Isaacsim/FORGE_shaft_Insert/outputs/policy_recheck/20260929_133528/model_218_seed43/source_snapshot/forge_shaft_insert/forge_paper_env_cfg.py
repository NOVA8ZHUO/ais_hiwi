"""FORGE peg settings adapted to Shaft_large; see docs/FORGE_ALIGNMENT.md."""
from isaaclab.utils import configclass
from isaaclab.managers import (ActionTermCfg, EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from .forge_insert_env_cfg import ForgeInsertSceneCfg, TaskParametersCfg
from .mdp import forge_paper as paper


@configclass
class ForgePaperParametersCfg(TaskParametersCfg):
    # Supplementary Table I, 8 mm peg column, SI units.
    coarse_kernel: tuple[float, float] = (50.0, 2.0)
    fine_kernel: tuple[float, float] = (100.0, 0.0)
    contact_beta: float = 0.25
    place_distance: float = 0.0025
    force_threshold_range: tuple[float, float] = (5.0, 8.0)
    kp_range: tuple[float, float] = (400.0, 800.0)
    action_limit_range: tuple[float, float] = (0.016, 0.025)
    deadzone_range: tuple[float, float] = (0.0, 5.0)
    part_friction_range: tuple[float, float] = (0.5, 1.0)
    # User-specified bounded errors, sampled independently once per episode.
    # Held-object offsets are expressed in panda_hand (the finger closure axis is Y).
    grasp_position_error: tuple[float, float, float] = (0.008, 0.0, 0.005)
    hole_position_error: tuple[float, float, float] = (0.010, 0.010, 0.010)
    ee_position_noise_std: float = 0.00025
    force_noise_std: float = 1.0
    trace_num_envs: int = 4  # First N environments, at every policy step.
    metrics_csv_path: str = ""  # Empty disables CSV; training sets its run directory.
    # User-requested gentler search and soft hand-frame Y tilt penalty.
    reference_speed: float = 0.012  # Vector norm, m/s.
    reference_acceleration: float = 0.040  # Vector norm, m/s^2.
    reference_yaw_speed: float = 0.5  # rad/s.
    reference_yaw_acceleration: float = 1.5  # rad/s^2.
    staged_insertion: bool = True
    xy_force_cap: float = 6.0
    z_force_cap: float = 8.0
    press_force_range: tuple[float, float] = (2.0, 4.0)
    search_contact_force: float = 1.5
    press_force_slew: float = 10.0  # N/s
    press_down_speed: float = 0.008
    actual_xy_brake_speed: float = 0.025
    actual_z_brake_speed: float = 0.015
    probe_contact_N: float = 1.0
    probe_progress: float = 0.003
    probe_timeout_s: float = 0.8
    probe_cooldown_s: float = 0.4
    free_entry_depth: float = 0.016
    entry_lateral_force_N: float = 2.0
    press_relief_N: float = 8.0
    press_stop_N: float = 10.0
    translation_force_cap: float = 10.0  # Commanded Cartesian force, not measured contact.
    shaft_pitch_tolerance: float = 0.05235987756  # 3 degrees.
    shaft_pitch_scale: float = 0.12217304764  # 7 degrees.
    shaft_pitch_weight: float = 0.25
    gripper_joint_damping: float = 200.0  # Implicit prismatic damping, N*s/m; no centering stiffness.
    gripper_force: float = 20.0
    gripper_expected_width: float = 0.035
    gripper_width_tolerance: float = 0.005
    gripper_closing_speed: float = 0.020  # Total aperture speed, not per jaw.
    gripper_settle_s: float = 1.0
    gripper_check_delay_s: float = 1.5
    # Zero positional residual follows a nominal vertical insertion trajectory.
    insertion_speed: float = 0.008
    approach_height: float = 0.0192845565
    residual_position_scale: tuple[float, float, float] = (0.020, 0.020, 0.002)
    # Calibrated FK of the fixed reset joints; no stale PhysX link pose at reset.
    reset_hand_position: tuple[float, float, float] = (0.6275140047, -0.0000764369, 0.2425106019)
    reset_hand_quaternion: tuple[float, float, float, float] = (0.0001641507, 0.9999994636, 0.0000771873, 0.0009570745)
    # Engineering choices / CAD-specific values, NOT reported FORGE settings.
    yaw_bound: float = 0.7853981633974483
    rotation_stiffness: float = 30.0
    rotation_damping: float = 0.5  # Explicit torque damping; old 2*sqrt(30) assumed unit inertia.
    shaft_bottom_offset: float = 0.0
    insertion_region_radius: float = 0.013  # Target bore radius; excludes falling beside the fixture.
    success_depth: float = 0.025  # User: engagement success, distinct from full seating.
    release_on_success: bool = True
    release_confirm_s: float = 0.0
    release_observe_s: float = 2.0
    release_width: float = 0.060
    insertion_depth: float = 0.047
    success_xy_tolerance: float = 0.0005
    depth_tolerance: float = 0.001
    success_axis_cos: float = 0.9961946980917455
    keypoint_offsets: tuple[float, ...] = (0.0, 0.02, 0.04)
    workspace_xy_limit: float = 0.10
    # Virtual TCP at held-shaft bottom, from the measured physical grasp.
    control_point_in_hand: tuple[float, float, float] = (-0.0007295087, -0.0013625964, 0.1732249624)
    shaft_reset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    shaft_reset_pos: tuple[float, float, float] = (0.6271160626399228, -2.079773476359282e-05, 0.06928453282904828)


@configclass
class ForgePaperSceneCfg(ForgeInsertSceneCfg):
    insertion_contact = ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Shaft_large',
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Base'],
        update_period=0.0, history_length=0)


@configclass
class ForgePaperActionsCfg:
    arm_action = ActionTermCfg(class_type=paper.ForgeImpedanceAction, asset_name='robot')


@configclass
class ForgePaperObservationsCfg:
    @configclass
    class Policy(ObsGroup):
        state = ObsTerm(func=paper.actor_state)
        enable_corruption = False
        concatenate_terms = True

    @configclass
    class Critic(ObsGroup):
        state = ObsTerm(func=paper.critic_state)
        enable_corruption = False
        concatenate_terms = True

    policy: Policy = Policy()
    critic: Critic = Critic()


@configclass
class ForgePaperEventsCfg:
    reset = EventTerm(func=paper.reset_paper, mode='reset')
    shaft_friction = EventTerm(func=mdp.randomize_rigid_body_material, mode='reset', params={
        'asset_cfg': SceneEntityCfg('shaft'), 'static_friction_range': (0.5, 1.0),
        'dynamic_friction_range': (0.5, 1.0), 'restitution_range': (0.0, 0.0),
        'num_buckets': 64, 'make_consistent': True})
    base_friction = EventTerm(func=mdp.randomize_rigid_body_material, mode='reset', params={
        'asset_cfg': SceneEntityCfg('base'), 'static_friction_range': (0.5, 1.0),
        'dynamic_friction_range': (0.5, 1.0), 'restitution_range': (0.0, 0.0),
        'num_buckets': 64, 'make_consistent': True})


@configclass
class ForgePaperRewardsCfg:
    keypoints = RewTerm(func=paper.keypoint_reward, weight=1.0)
    place = RewTerm(func=paper.place_bonus, weight=1.0)
    inserted = RewTerm(func=paper.inserted_bonus, weight=1.0)
    contact = RewTerm(func=paper.excessive_force, weight=-0.2)
    shaft_pitch = RewTerm(func=paper.shaft_pitch_penalty, weight=-0.25)
    prediction = RewTerm(func=paper.prediction_error, weight=-1.0)


@configclass
class ForgePaperTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    outside_workspace = DoneTerm(func=paper.failed_workspace)
    invalid_grasp_width = DoneTerm(func=paper.invalid_grasp_width)
    release_complete = DoneTerm(func=paper.release_complete)


@configclass
class ForgePaperEnvCfg(ManagerBasedRLEnvCfg):
    scene: ForgePaperSceneCfg = ForgePaperSceneCfg(num_envs=1, env_spacing=1.5)
    actions: ForgePaperActionsCfg = ForgePaperActionsCfg()
    observations: ForgePaperObservationsCfg = ForgePaperObservationsCfg()
    events: ForgePaperEventsCfg = ForgePaperEventsCfg()
    rewards: ForgePaperRewardsCfg = ForgePaperRewardsCfg()
    terminations: ForgePaperTerminationsCfg = ForgePaperTerminationsCfg()
    task: ForgePaperParametersCfg = ForgePaperParametersCfg()

    def __post_init__(self):
        self.sim.dt = 1.0 / 240.0
        self.decimation = 16
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.viewer.eye = (1.25, 1.0, 0.8)
        self.viewer.lookat = (0.627, 0.0, 0.1)
        self.scene.robot.spawn.rigid_props.disable_gravity = False
        self.scene.shaft.spawn.activate_contact_sensors = True
        self.scene.shaft.init_state.pos = self.task.shaft_reset_pos
        self.scene.shaft.init_state.rot = self.task.shaft_reset_rot
        self.scene.base.init_state.pos = self.task.base_pos
        # Offline IK: fingertip (0.627, 0, 0.13), RPY (180,0,0).
        joints = (-0.0647969916, 0.4782325327, 0.0683593974, -1.9196174145,
                  -0.0469411574, 2.3982622623, 0.8156672716)
        for index, value in enumerate(joints, 1):
            self.scene.robot.init_state.joint_pos[f'panda_joint{index}'] = value
        for name in ('panda_shoulder', 'panda_forearm'):
            self.scene.robot.actuators[name].stiffness = 0.0
            self.scene.robot.actuators[name].damping = 0.0
        # Continuous force command; 20 N per slider is a mapping of
        # the hardware command, not a calibrated measurement of contact force.
        gripper = self.scene.robot.actuators['panda_hand']
        gripper.stiffness = 0.0
        gripper.damping = self.task.gripper_joint_damping
        gripper.effort_limit_sim = self.task.gripper_force
        gripper.velocity_limit_sim = self.task.gripper_closing_speed / 2.0
        for index in (1, 2):
            self.scene.robot.init_state.joint_pos[f'panda_finger_joint{index}'] = self.task.gripper_expected_width / 2.0
        for name in ('shaft_friction', 'base_friction'):
            term = getattr(self.events, name)
            term.params['static_friction_range'] = self.task.part_friction_range
            term.params['dynamic_friction_range'] = self.task.part_friction_range
        # RewardManager multiplies rates by dt; FORGE specifies per-step reward.
        policy_dt = self.sim.dt * self.decimation
        self.rewards.contact.weight = -self.task.contact_beta
        self.rewards.shaft_pitch.weight = -self.task.shaft_pitch_weight
        for name in ('keypoints', 'place', 'inserted', 'contact', 'prediction', 'shaft_pitch'):
            getattr(self.rewards, name).weight /= policy_dt
