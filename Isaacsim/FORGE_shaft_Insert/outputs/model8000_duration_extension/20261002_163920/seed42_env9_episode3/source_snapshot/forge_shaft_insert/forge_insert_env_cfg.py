# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.sensors import CameraCfg
from . import mdp


_PROJECT_DIR = Path(__file__).resolve().parents[1]
_ROBOT_USD = str(_PROJECT_DIR / "assets" / "franka.usda")
_SHAFT_USD = str(_PROJECT_DIR / "assets" / "shaft_large.usda")
_BASE_USD = str(_PROJECT_DIR / "assets" / "base.usda")
_ENVIRONMENT_USD = str(_PROJECT_DIR / "assets" / "environment.usda")


@configclass
class TaskParametersCfg:
    """Stage-1 Cartesian alignment parameters in SI units."""

    # Values read directly from shaft_insertion.usd.  The hole position is
    # local to the Base rigid-body root, whose fixed world pose is (0.5, 0, 0).
    base_pos: tuple[float, float, float] = (0.5, 0.0, 0.0)
    hole_offset_in_base: tuple[float, float, float] = (0.127, -0.00018226745, 0.05)
    # Stable state measured after gradual closure, 120 settling steps, gravity
    # enable, and another 180 physics steps in franka_diff_ik.py.
    shaft_reset_pos: tuple[float, float, float] = (0.4998937547, 0.0005603544, 0.1492845565)
    shaft_reset_rot: tuple[float, float, float, float] = (
        0.9999618530,
        0.0081924731,
        0.0000997439,
        -0.0030323504,
    )

    # Exact panda_hand -> panda_fingertip_centered transform measured in the
    # source USD.  The policy action controls this point.
    tcp_offset: tuple[float, float, float] = (-0.0007295087, -0.0001125964, 0.1125095189)
    tcp_offset_rot: tuple[float, float, float, float] = (
        0.9999995232,
        0.0001637340,
        -0.0009569824,
        0.0000775158,
    )
    nominal_tcp_height_above_hole: float = 0.0992845593
    action_scale: float = 0.001
    # Slightly below the measured contact-limited openings.  Commanding 0.0
    # immediately after a teleported reset creates a large cold-start contact
    # impulse; 0.014 m keeps a moderate physical squeeze without ejecting the
    # 10 g shaft or allowing it to creep during an eight-second episode.
    gripper_command: float = 0.014

    success_xy_tolerance: float = 0.00025
    workspace_xy_limit: float = 0.200
    workspace_z_tolerance: float = 0.050

    xy_progress_scale: float = 0.001
    xy_reward_weight: float = 1.0
    action_penalty_weight: float = -0.001
    success_reward_weight: float = 20.0


@configclass
class WristCameraParametersCfg:
    """Measured RealSense color-stream parameters used by the camera task."""

    # Source: /home/masais/Xinyu/cam_calibration/outputs/shared/camera_intrinsics.json
    # RealSense serial number: 317222075679
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fx: float = 605.37451171875
    fy: float = 604.3340454101562
    cx: float = 318.1541748046875
    cy: float = 255.36512756347656
    near_clip: float = 0.01
    far_clip: float = 3.0


@configclass
class ForgeInsertSceneCfg(InteractiveSceneCfg):
    """Existing shaft-insertion USD assets arranged as cloneable per-environment assets."""

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # Verified by franka_diff_ik.py for fingertip=(0.5, 0, 0.21),
                # RPY=(180, 0, 0) deg in the original /Franka asset.
                "panda_joint1": -0.07760076224803925,
                "panda_joint2": 0.011352632194757462,
                "panda_joint3": 0.0774737149477005,
                "panda_joint4": -2.3616721630096436,
                "panda_joint5": -0.0017386332619935274,
                "panda_joint6": 2.3749072551727295,
                "panda_joint7": 0.7863608598709106,
                # Contact-limited positions measured after the shaft was grasped.
                "panda_finger_joint1": 0.016866788268089294,
                "panda_finger_joint2": 0.017094546929001808,
            },
            joint_vel={".*": 0.0},
        ),
    )
    robot.spawn.usd_path = _ROBOT_USD

    shaft = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shaft_large",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_SHAFT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.4998937547, 0.0005603544, 0.1492845565),
            rot=(0.9999618530, 0.0081924731, 0.0000997439, -0.0030323504),
        ),
    )

    base = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Base",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_BASE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
    )

    workstation = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        spawn=sim_utils.UsdFileCfg(usd_path=_ENVIRONMENT_USD),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2000.0),
    )


@configclass
class ForgeInsertCameraSceneCfg(ForgeInsertSceneCfg):
    """Stage-1 scene with a calibrated wrist RGB-D camera."""

    # The parent Xform is authored in assets/franka.usda using the measured
    # panda_hand <- camera_color_optical_frame hand-eye transform. The child
    # Camera uses identity in that ROS optical frame; Isaac Lab converts it to
    # the OpenGL/USD Camera convention internally.
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/camera_color_optical_frame/Camera",
        update_period=1.0 / 30.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        update_latest_camera_pose=True,
        # Equivalent USD pinhole parameters for the Isaac Sim 5.1-supported
        # effective K: f=(fx+fy)/2=604.8543 px, cx=320, cy=240.
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.945084810256958,
            horizontal_aperture=1.0,
            vertical_aperture=0.75,
            horizontal_aperture_offset=0.0,
            vertical_aperture_offset=0.0,
            clipping_range=(0.01, 3.0),
            lock_camera=True,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )


@configclass
class ActionsCfg:
    """Three-dimensional Cartesian delta-position action."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint[1-7]"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
        scale=0.001,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(-0.0007295087, -0.0001125964, 0.1125095189),
            rot=(0.9999995232, 0.0001637340, -0.0009569824, 0.0000775158),
        ),
    )
    arm_action.class_type = mdp.AccumulatingCartesianDeltaPositionAction


@configclass
class ObservationsCfg:
    """Stage-1 policy observation: ``[dx, dy, dz]`` in world axes [m]."""

    @configclass
    class PolicyCfg(ObsGroup):
        task_state = ObsTerm(
            func=mdp.position_error,
            params={"asset_cfg": SceneEntityCfg("shaft")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Episode reset events."""

    reset_grasp_state = EventTerm(
        func=mdp.reset_grasp_state,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "shaft_cfg": SceneEntityCfg("shaft"),
            "gripper_command": 0.0,
        },
    )


@configclass
class RewardsCfg:
    """Stage-1 reward: align XY, use small actions, and finish."""

    xy_progress = RewTerm(func=mdp.xy_progress, weight=1.0)
    action_magnitude = RewTerm(func=mdp.action_magnitude, weight=-0.001)
    success = RewTerm(func=mdp.success_bonus, weight=20.0)


@configclass
class TerminationsCfg:
    """Stage-1 XY success, compact workspace, and time limit."""

    success = DoneTerm(func=mdp.xy_alignment_success)
    outside_workspace = DoneTerm(func=mdp.outside_workspace)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ForgeInsertEnvCfg(ManagerBasedRLEnvCfg):
    """Stage-1 ManagerBasedRLEnv: move a grasped shaft over the hole center."""

    scene: ForgeInsertSceneCfg = ForgeInsertSceneCfg(num_envs=1, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task: TaskParametersCfg = TaskParametersCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        # At 1 mm/action, the 127 mm pickup-to-hole travel needs more than
        # 127 policy steps.  Eight seconds gives 240 steps at 30 Hz.
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.25, 1.0, 0.8)
        self.viewer.lookat = (0.627, 0.0, 0.15)

        # Keep every user-adjustable scalar sourced from TaskParametersCfg.
        self.actions.arm_action.scale = self.task.action_scale
        self.actions.arm_action.body_offset.pos = self.task.tcp_offset
        self.actions.arm_action.body_offset.rot = self.task.tcp_offset_rot
        self.scene.shaft.init_state.pos = self.task.shaft_reset_pos
        self.scene.shaft.init_state.rot = self.task.shaft_reset_rot
        self.scene.base.init_state.pos = self.task.base_pos
        self.events.reset_grasp_state.params["gripper_command"] = self.task.gripper_command
        self.rewards.xy_progress.weight = self.task.xy_reward_weight
        self.rewards.action_magnitude.weight = self.task.action_penalty_weight
        self.rewards.success.weight = self.task.success_reward_weight


@configclass
class ForgeInsertPositionOnlyEnvCfg(ForgeInsertEnvCfg):
    """Backward-compatible alias for the stage-1 position-only task."""


@configclass
class ForgeInsertCameraEnvCfg(ForgeInsertEnvCfg):
    """Camera-enabled diagnostic variant; policy observation stays [dx, dy, dz]."""

    scene: ForgeInsertCameraSceneCfg = ForgeInsertCameraSceneCfg(num_envs=1, env_spacing=1.5)
    wrist_camera: WristCameraParametersCfg = WristCameraParametersCfg()
