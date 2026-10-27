# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Move the existing ``/Franka`` articulation with Isaac Lab differential IK.

This script opens the existing shaft-insertion USD and wraps its Franka as an
Isaac Lab ``Articulation``.  It does not spawn a second robot and never saves
the stage.  The requested target is split into Cartesian waypoints whose
per-axis displacement is bounded by ``--max_step``.

Run from the Isaac Lab 2.3.2 directory::

    ./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/franka_diff_ik.py \
        --target 0.45 0.0 0.22

Use ``--headless`` for a terminal-only test.  By default, ``--target`` refers
to ``/Franka/panda_fingertip_centered`` while PhysX Jacobians are always read
from the real rigid body ``panda_hand``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


# Easy-to-edit defaults, in world coordinates [m].
TARGET_X = 0.45
TARGET_Y = 0.00
TARGET_Z = 0.22

DEFAULT_USD_PATH = "/home/masais/Xinyu/USD_files/shaft_insertion.usd"
DEFAULT_ROBOT_PRIM = "/Franka"
DEFAULT_HAND_BODY = "panda_hand"
DEFAULT_REFERENCE_PRIM = "/Franka/panda_fingertip_centered"
DEFAULT_SHAFT_PRIM = "/Parts/Shaft_large"
DEFAULT_FINGER_CONTACT_PRIMS = ("/Franka/panda_leftfinger", "/Franka/panda_rightfinger")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd_path", type=Path, default=Path(DEFAULT_USD_PATH), help="USD scene to open.")
parser.add_argument("--robot_prim", type=str, default=DEFAULT_ROBOT_PRIM, help="Existing Franka articulation prim.")
parser.add_argument("--hand_body", type=str, default=DEFAULT_HAND_BODY, help="Rigid body used for the PhysX Jacobian.")
parser.add_argument(
    "--reference_prim",
    type=str,
    default=DEFAULT_REFERENCE_PRIM,
    help="Reference point that should reach --target; use an empty string to target panda_hand directly.",
)
parser.add_argument(
    "--target",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=(TARGET_X, TARGET_Y, TARGET_Z),
    help="Final reference-point position in world XYZ [m].",
)
parser.add_argument(
    "--target_rpy_deg",
    type=float,
    nargs=3,
    metavar=("ROLL", "PITCH", "YAW"),
    default=(180.0, 0.0, -45.0),
    help="Final reference orientation as world-frame XYZ Euler angles [deg].",
)
parser.add_argument("--max_step", type=float, default=0.010, help="Maximum XYZ displacement per waypoint [m].")
parser.add_argument("--position_tolerance", type=float, default=0.0005, help="Waypoint/final tolerance [m].")
parser.add_argument(
    "--orientation_tolerance_deg", type=float, default=1.0, help="Final orientation tolerance [deg]."
)
parser.add_argument("--max_joint_step", type=float, default=0.05, help="Maximum joint command change per sim step [rad].")
parser.add_argument(
    "--gripper_position",
    type=float,
    default=0.0,
    help="Final symmetric finger target after the pose is reached [m].",
)
parser.add_argument(
    "--approach_gripper_position",
    type=float,
    default=0.04,
    help="Symmetric finger target while moving to the grasp pose [m].",
)
parser.add_argument(
    "--drive_mode",
    choices=("direct", "actuator"),
    default="direct",
    help="Direct writes smooth IK joint states; actuator uses the USD drive dynamics.",
)
parser.add_argument("--max_sim_steps", type=int, default=2000, help="Stop after this many physics steps.")
parser.add_argument("--settle_steps", type=int, default=30, help="Hold the current pose before starting IK.")
parser.add_argument(
    "--grasp_settle_steps",
    type=int,
    default=120,
    help="Physics steps to close and stabilize the gripper before the optional lift.",
)
parser.add_argument(
    "--lift_after_grasp",
    type=float,
    default=0.0,
    help="After grasp stabilization, raise the reference by this world-Z distance [m].",
)
parser.add_argument("--shaft_prim", type=str, default=DEFAULT_SHAFT_PRIM, help="Dynamic shaft rigid-body prim.")
parser.add_argument(
    "--finger_contact_prims",
    type=str,
    nargs=2,
    metavar=("LEFT_PRIM", "RIGHT_PRIM"),
    default=DEFAULT_FINGER_CONTACT_PRIMS,
    help="Rigid-body prims used to report the left/right shaft contact forces.",
)
parser.add_argument(
    "--enable_gravity_after_grasp",
    action="store_true",
    help="Keep shaft gravity off during approach, then enable it after the closed gripper has settled.",
)
parser.add_argument(
    "--disturbance_force",
    type=float,
    nargs=3,
    metavar=("FX", "FY", "FZ"),
    default=(0.0, 0.0, 0.0),
    help="Optional constant world-frame force applied to the shaft after lift [N].",
)
parser.add_argument(
    "--disturbance_delay_steps", type=int, default=60, help="Delay after reaching the final pose before force test."
)
parser.add_argument(
    "--disturbance_steps", type=int, default=120, help="Number of physics steps to apply --disturbance_force."
)
parser.add_argument(
    "--force_log_interval", type=int, default=30, help="Print force and shaft-to-gripper slip every N steps."
)
parser.add_argument("--hold_steps", type=int, default=120, help="Hold the final pose before exiting headless mode.")
parser.add_argument(
    "--keep_open",
    action="store_true",
    help="After success, keep simulating until the window is closed or Ctrl+C is pressed.",
)
parser.add_argument("--dry_run", action="store_true", help="Initialize and print poses without commanding motion.")
parser.add_argument(
    "--no_legacy_patch",
    action="store_true",
    help="Do not apply temporary Sim-5.1 compatibility overrides for old mimic/external joints.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402


def _is_below(path: str, root: str) -> bool:
    """Return whether a USD path is the root itself or one of its descendants."""
    normalized = root.rstrip("/")
    return path == normalized or path.startswith(normalized + "/")


def _apply_runtime_compatibility(stage: Usd.Stage, robot_prim_path: str) -> None:
    """Patch legacy physics schemas in the anonymous session layer only."""
    stage.SetEditTarget(stage.GetSessionLayer())
    removed_mimic: list[str] = []
    added_linear_drives: list[str] = []
    removed_nested_roots: list[str] = []
    disabled_external_joints: list[str] = []

    mimic_axes = (
        UsdPhysics.Tokens.rotX,
        UsdPhysics.Tokens.rotY,
        UsdPhysics.Tokens.rotZ,
        UsdPhysics.Tokens.transX,
        UsdPhysics.Tokens.transY,
        UsdPhysics.Tokens.transZ,
    )

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if _is_below(path, robot_prim_path):
            for axis in mimic_axes:
                if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis):
                    prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, axis)
                    removed_mimic.append(f"{path}:{axis}")

                    # The legacy Franka USD relies on the invalid rotational
                    # mimic schema for its second prismatic finger and does not
                    # author an independent linear drive.  Once the mimic is
                    # removed for Sim 5.1 compatibility, create that drive so
                    # both fingers can actually respond to explicit targets.
                    if prim.IsA(UsdPhysics.PrismaticJoint) and path.endswith("/panda_finger_joint2"):
                        drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
                        state_position = prim.GetAttribute("state:linear:physics:position").Get()
                        drive.CreateTargetPositionAttr(float(state_position or 0.0))
                        drive.CreateTargetVelocityAttr(0.0)
                        drive.CreateStiffnessAttr(2.0e3)
                        drive.CreateDampingAttr(1.0e2)
                        drive.CreateMaxForceAttr(200.0)
                        added_linear_drives.append(path)

            if path != robot_prim_path and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                removed_nested_roots.append(path)

        if prim.IsA(UsdPhysics.Joint) and not _is_below(path, robot_prim_path):
            joint = UsdPhysics.Joint(prim)
            targets = joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()
            if any(_is_below(str(target), robot_prim_path) for target in targets):
                prim.SetActive(False)
                disabled_external_joints.append(path)

    print(
        "[compat] session-layer overrides: "
        f"mimic={len(removed_mimic)}, nested_articulation_roots={len(removed_nested_roots)}, "
        f"external_joints={len(disabled_external_joints)}, linear_drives={len(added_linear_drives)}"
    )
    for path in removed_mimic + added_linear_drives + removed_nested_roots + disabled_external_joints:
        print(f"[compat]   {path}")


def _reference_world_pose(
    stage: Usd.Stage, prim_path: str
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Read a reference prim's initial world position and wxyz quaternion from USD."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Reference prim does not exist: {prim_path}")
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    translation: Gf.Vec3d = matrix.ExtractTranslation()
    rotation: Gf.Quatd = matrix.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    position = float(translation[0]), float(translation[1]), float(translation[2])
    quaternion = float(rotation.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])
    return position, quaternion


def _make_robot(robot_prim_path: str) -> Articulation:
    """Wrap the existing USD articulation without spawning another Franka."""
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=robot_prim_path, spawn=None)
    return Articulation(robot_cfg)


def _next_waypoint(current: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
    """Create a waypoint with independently bounded XYZ displacement."""
    return current + torch.clamp(target - current, min=-max_step, max=max_step)


def _resolve_jacobian_body_index(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    hand_body_id: int,
    arm_joint_ids: list[int],
) -> int:
    """Resolve a custom articulation's Jacobian row using finite differences."""
    epsilon = 1.0e-3
    baseline_joint_pos = robot.data.joint_pos[:, arm_joint_ids].clone()
    baseline_hand_pos = robot.data.body_pos_w[:, hand_body_id].clone()
    finite_difference = torch.zeros((3, len(arm_joint_ids)), device=robot.device)

    for column in range(len(arm_joint_ids)):
        perturbed = baseline_joint_pos.clone()
        perturbed[:, column] += epsilon
        robot.write_joint_position_to_sim(perturbed, joint_ids=arm_joint_ids)
        robot.write_joint_velocity_to_sim(torch.zeros_like(perturbed), joint_ids=arm_joint_ids)
        sim.forward()
        robot.update(sim.get_physics_dt())
        finite_difference[:, column] = (
            robot.data.body_pos_w[0, hand_body_id] - baseline_hand_pos[0]
        ) / epsilon
        robot.write_joint_position_to_sim(baseline_joint_pos, joint_ids=arm_joint_ids)
        robot.write_joint_velocity_to_sim(torch.zeros_like(baseline_joint_pos), joint_ids=arm_joint_ids)
        sim.forward()
        robot.update(sim.get_physics_dt())

    all_jacobians = robot.root_physx_view.get_jacobians()
    hand_com_pos_w = robot.data.body_com_pos_w[:, hand_body_id]
    com_to_link_w = baseline_hand_pos - hand_com_pos_w
    errors = []
    for index in range(all_jacobians.shape[1]):
        candidate_6d = all_jacobians[:, index, :, arm_joint_ids]
        candidate = candidate_6d[:, :3, :] - torch.bmm(
            math_utils.skew_symmetric_matrix(com_to_link_w), candidate_6d[:, 3:, :]
        )
        candidate = candidate[0]
        errors.append(torch.linalg.matrix_norm(candidate - finite_difference).item())
    best_index = min(range(len(errors)), key=errors.__getitem__)
    official_index = hand_body_id - 1 if robot.is_fixed_base else hand_body_id
    print(
        f"[setup] jacobian finite-difference check: official_index={official_index} "
        f"error={errors[official_index]:.6f}, selected_index={best_index} error={errors[best_index]:.6f}"
    )
    print(
        f"[setup] panda_hand COM-to-link offset (world)={com_to_link_w[0].cpu().tolist()}, "
        "applied to the PhysX Jacobian"
    )
    return best_index


def run_controller(
    sim: sim_utils.SimulationContext,
    stage: Usd.Stage,
    robot: Articulation,
    shaft: RigidPrim,
) -> bool:
    """Move the selected reference point to the requested world target."""
    arm_joint_ids, arm_joint_names = robot.find_joints("panda_joint[1-7]", preserve_order=True)
    finger_joint_ids, finger_joint_names = robot.find_joints("panda_finger_joint[1-2]", preserve_order=True)
    hand_body_ids, hand_body_names = robot.find_bodies(args_cli.hand_body, preserve_order=True)
    if len(arm_joint_ids) != 7:
        raise RuntimeError(f"Expected panda_joint1..7, found {arm_joint_names}")
    if len(hand_body_ids) != 1:
        raise RuntimeError(f"Expected one '{args_cli.hand_body}' body, found {hand_body_names}")
    if len(finger_joint_ids) != 2:
        raise RuntimeError(f"Expected panda_finger_joint1..2, found {finger_joint_names}")

    hand_body_id = hand_body_ids[0]
    sim_dt = sim.get_physics_dt()
    device = robot.device

    # Update once so the body state reflects the opened stage.
    robot.update(sim_dt)
    jacobian_body_id = _resolve_jacobian_body_index(sim, robot, hand_body_id, arm_joint_ids)
    hand_pose_w = robot.data.body_pose_w[:, hand_body_id]
    initial_hand_pos_w = hand_pose_w[:, :3].clone()
    initial_hand_quat_w = hand_pose_w[:, 3:7].clone()
    initial_arm_joint_pos = robot.data.joint_pos[:, arm_joint_ids].clone()
    final_gripper_target = torch.full(
        (1, len(finger_joint_ids)),
        args_cli.gripper_position,
        device=device,
        dtype=initial_arm_joint_pos.dtype,
    )
    approach_gripper_target = torch.full_like(final_gripper_target, args_cli.approach_gripper_position)

    if args_cli.reference_prim:
        reference_position, reference_quaternion = _reference_world_pose(stage, args_cli.reference_prim)
        initial_reference_pos_w = torch.tensor(
            reference_position, device=device, dtype=initial_hand_pos_w.dtype
        ).unsqueeze(0)
        initial_reference_quat_w = torch.tensor(
            reference_quaternion, device=device, dtype=initial_hand_pos_w.dtype
        ).unsqueeze(0)
        reference_offset_pos_hand, reference_offset_quat_hand = math_utils.subtract_frame_transforms(
            initial_hand_pos_w,
            initial_hand_quat_w,
            initial_reference_pos_w,
            initial_reference_quat_w,
        )
        reference_label = args_cli.reference_prim
    else:
        initial_reference_pos_w = initial_hand_pos_w
        initial_reference_quat_w = initial_hand_quat_w
        reference_offset_pos_hand = torch.zeros((1, 3), device=device, dtype=initial_hand_pos_w.dtype)
        reference_offset_quat_hand = torch.zeros((1, 4), device=device, dtype=initial_hand_pos_w.dtype)
        reference_offset_quat_hand[:, 0] = 1.0
        reference_label = f"{args_cli.robot_prim}/{args_cli.hand_body}"

    grasp_target_reference_w = torch.tensor(
        args_cli.target, device=device, dtype=initial_hand_pos_w.dtype
    ).unsqueeze(0)
    target_reference_w = grasp_target_reference_w.clone()
    target_rpy_w = torch.deg2rad(
        torch.tensor(args_cli.target_rpy_deg, device=device, dtype=initial_hand_pos_w.dtype)
    )
    target_reference_quat_w = math_utils.quat_from_euler_xyz(
        target_rpy_w[0:1], target_rpy_w[1:2], target_rpy_w[2:3]
    )
    orientation_tolerance = torch.deg2rad(
        torch.tensor(args_cli.orientation_tolerance_deg, device=device, dtype=initial_hand_pos_w.dtype)
    )
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=device)

    print(f"[setup] USD={args_cli.usd_path}")
    print(f"[setup] robot={args_cli.robot_prim}, hand_body={hand_body_names[0]}, fixed_base={robot.is_fixed_base}")
    print(f"[setup] arm_joints={arm_joint_names}")
    print(
        f"[setup] finger_joints={finger_joint_names}, approach_target={args_cli.approach_gripper_position:.4f} m, "
        f"final_target={args_cli.gripper_position:.4f} m"
    )
    print(f"[setup] arm_joint_ids={arm_joint_ids}, hand_body_id={hand_body_id}, jacobian_body_id={jacobian_body_id}")
    print(f"[setup] body_names={robot.body_names}")
    print(f"[setup] root_pose_w={robot.data.root_pose_w.cpu().tolist()[0]}")
    print(f"[setup] controlled_reference={reference_label}")
    print(f"[setup] initial_reference_xyz={initial_reference_pos_w.cpu().tolist()[0]}")
    print(
        f"[setup] reference_offset_in_hand_xyz={reference_offset_pos_hand.cpu().tolist()[0]} "
        f"quat_wxyz={reference_offset_quat_hand.cpu().tolist()[0]}"
    )
    print(f"[setup] final_target_xyz={list(args_cli.target)}, max_waypoint_axis_step={args_cli.max_step} m")
    print(f"[setup] final_target_rpy_deg={list(args_cli.target_rpy_deg)}")
    print(
        f"[setup] grasp_settle_steps={args_cli.grasp_settle_steps}, "
        f"lift_after_grasp={args_cli.lift_after_grasp:.4f} m"
    )
    print(
        f"[setup] shaft={args_cli.shaft_prim}, gravity_after_grasp={args_cli.enable_gravity_after_grasp}, "
        f"disturbance_force={list(args_cli.disturbance_force)} N"
    )
    if args_cli.dry_run:
        print("[dry-run] Initialization passed; no joint command was sent.")
        return True

    waypoint_reference_w: torch.Tensor | None = None
    reached_grasp_step: int | None = None
    lift_started = False
    reached_final_step: int | None = None
    gravity_enabled = False
    grasp_relative_pos: torch.Tensor | None = None
    grasp_relative_quat: torch.Tensor | None = None
    disturbance_started_step: int | None = None
    disturbance_finished_step: int | None = None
    disturbance_force = torch.tensor(
        args_cli.disturbance_force, device=device, dtype=initial_hand_pos_w.dtype
    ).unsqueeze(0)
    disturbance_requested = torch.linalg.vector_norm(disturbance_force).item() > 0.0

    step = 0
    while simulation_app.is_running():
        if reached_final_step is None and step >= args_cli.max_sim_steps:
            break

        hand_pose_w = robot.data.body_pose_w[:, hand_body_id]
        hand_pos_w = hand_pose_w[:, :3]
        hand_quat_w = hand_pose_w[:, 3:7]
        current_reference_w, current_reference_quat_w = math_utils.combine_frame_transforms(
            hand_pos_w,
            hand_quat_w,
            reference_offset_pos_hand,
            reference_offset_quat_hand,
        )

        settling = step < args_cli.settle_steps
        if not settling:
            if (
                reached_grasp_step is not None
                and not gravity_enabled
                and step - reached_grasp_step >= args_cli.grasp_settle_steps
            ):
                shaft_pos_w, shaft_quat_w = shaft.get_world_poses()
                shaft_pos_w = torch.as_tensor(shaft_pos_w, device=device)
                shaft_quat_w = torch.as_tensor(shaft_quat_w, device=device)
                grasp_relative_pos, grasp_relative_quat = math_utils.subtract_frame_transforms(
                    current_reference_w, current_reference_quat_w, shaft_pos_w, shaft_quat_w
                )
                if args_cli.enable_gravity_after_grasp:
                    shaft.enable_gravities()
                    print(
                        f"[gravity] step={step} shaft gravity enabled; "
                        "the shaft remains an independent rigid body held only by contact/friction"
                    )
                gravity_enabled = True

            if (
                reached_grasp_step is not None
                and not lift_started
                and args_cli.lift_after_grasp > 0.0
                and step - reached_grasp_step >= args_cli.grasp_settle_steps
            ):
                lift_started = True
                target_reference_w = grasp_target_reference_w.clone()
                target_reference_w[:, 2] += args_cli.lift_after_grasp
                waypoint_reference_w = None
                print(
                    f"[lift] step={step} gripper stabilized; lifting to "
                    f"xyz={target_reference_w.cpu().tolist()[0]}"
                )

            if waypoint_reference_w is None:
                waypoint_reference_w = _next_waypoint(current_reference_w, target_reference_w, args_cli.max_step)
                print(f"[waypoint] step={step} xyz={waypoint_reference_w.cpu().tolist()[0]}")

            waypoint_error = torch.linalg.vector_norm(waypoint_reference_w - current_reference_w, dim=-1)
            if waypoint_error.item() <= args_cli.position_tolerance:
                final_error = torch.linalg.vector_norm(target_reference_w - current_reference_w, dim=-1)
                orientation_error = math_utils.quat_error_magnitude(
                    current_reference_quat_w, target_reference_quat_w
                )
                if final_error.item() <= args_cli.position_tolerance:
                    if orientation_error.item() <= orientation_tolerance.item():
                        if reached_grasp_step is None:
                            reached_grasp_step = step
                            print(
                                f"[grasp] step={step} reference_xyz={current_reference_w.cpu().tolist()[0]} "
                                f"position_error={final_error.item():.6f} m, "
                                f"orientation_error={torch.rad2deg(orientation_error).item():.3f} deg; "
                                "closing gripper"
                            )
                            if args_cli.lift_after_grasp == 0.0:
                                reached_final_step = step
                        elif lift_started and reached_final_step is None:
                            reached_final_step = step
                            print(
                                f"[success] step={step} lift complete at "
                                f"reference_xyz={current_reference_w.cpu().tolist()[0]}"
                            )
                else:
                    waypoint_reference_w = _next_waypoint(
                        current_reference_w, target_reference_w, args_cli.max_step
                    )
                    print(f"[waypoint] step={step} xyz={waypoint_reference_w.cpu().tolist()[0]}")

        root_pose_w = robot.data.root_pose_w
        current_reference_b, current_reference_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], current_reference_w, current_reference_quat_w
        )
        desired_reference_b, desired_reference_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7],
            current_reference_w if settling else waypoint_reference_w,
            current_reference_quat_w if settling else target_reference_quat_w,
        )
        current_joint_pos = robot.data.joint_pos[:, arm_joint_ids]
        if settling:
            desired_joint_pos = initial_arm_joint_pos
        else:
            controller.set_command(torch.cat((desired_reference_b, desired_reference_quat_b), dim=-1))

            jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_body_id, :, arm_joint_ids].clone()
            # PhysX linear Jacobians are evaluated at the rigid body's center
            # of mass.  Translate them to the point actually being controlled
            # (panda_hand link origin or panda_fingertip_centered).
            hand_com_pos_w = robot.data.body_com_pos_w[:, hand_body_id]
            com_to_reference_w = current_reference_w - hand_com_pos_w
            jacobian[:, :3, :] -= torch.bmm(
                math_utils.skew_symmetric_matrix(com_to_reference_w), jacobian[:, 3:, :]
            )
            # PhysX returns the geometric Jacobian in world axes, while both
            # poses above are expressed in the articulation-root frame.
            world_to_root = math_utils.matrix_from_quat(math_utils.quat_inv(root_pose_w[:, 3:7]))
            jacobian[:, :3, :] = torch.bmm(world_to_root, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(world_to_root, jacobian[:, 3:, :])
            desired_joint_pos = controller.compute(
                current_reference_b, current_reference_quat_b, jacobian, current_joint_pos
            )
            desired_joint_pos = current_joint_pos + torch.clamp(
                desired_joint_pos - current_joint_pos,
                min=-args_cli.max_joint_step,
                max=args_cli.max_joint_step,
            )
        joint_limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids]
        desired_joint_pos = torch.clamp(desired_joint_pos, min=joint_limits[..., 0], max=joint_limits[..., 1])

        if step in (args_cli.settle_steps, args_cli.settle_steps + 1, args_cli.settle_steps + 10) or (
            not settling and step % 100 == 0
        ):
            print(
                f"[progress] step={step} reference_xyz={current_reference_w[0].cpu().tolist()} "
                f"waypoint_error={torch.linalg.vector_norm(waypoint_reference_w - current_reference_w).item():.6f} "
                f"max_dq={torch.max(torch.abs(desired_joint_pos - current_joint_pos)).item():.6f}"
            )
            if step == args_cli.settle_steps + 10:
                print(f"[progress] joint_pos={current_joint_pos[0].cpu().tolist()}")
                print(f"[progress] joint_limits={joint_limits[0].cpu().tolist()}")
                print(f"[progress] jacobian_linear={jacobian[0, :3].cpu().tolist()}")

        robot.set_joint_position_target(desired_joint_pos, joint_ids=arm_joint_ids)
        # The legacy USD used an invalid rotational Mimic API on the second
        # prismatic finger.  The compatibility patch removes that API, so both
        # independent finger DOFs must receive the same target explicitly.
        gripper_target = final_gripper_target if reached_grasp_step is not None else approach_gripper_target
        robot.set_joint_position_target(gripper_target, joint_ids=finger_joint_ids)
        if args_cli.drive_mode == "direct":
            robot.write_joint_position_to_sim(desired_joint_pos, joint_ids=arm_joint_ids)
            robot.write_joint_velocity_to_sim(torch.zeros_like(desired_joint_pos), joint_ids=arm_joint_ids)
        robot.write_data_to_sim()
        if (
            disturbance_requested
            and reached_final_step is not None
            and step - reached_final_step >= args_cli.disturbance_delay_steps
            and disturbance_finished_step is None
        ):
            if disturbance_started_step is None:
                disturbance_started_step = step
                print(
                    f"[force-test] step={step} applying world force={list(args_cli.disturbance_force)} N "
                    f"for {args_cli.disturbance_steps} steps ({args_cli.disturbance_steps * sim_dt:.3f} s)"
                )
            if step - disturbance_started_step < args_cli.disturbance_steps:
                shaft.apply_forces(disturbance_force, is_global=True)
            else:
                disturbance_finished_step = step
                print(f"[force-test] step={step} disturbance finished")
        # A real physics step is required even for direct state writes:
        # ArticulationData advances its cache timestamp in ``robot.update``
        # and otherwise keeps returning the previous body pose/Jacobian.
        sim.step()
        robot.update(sim_dt)

        if reached_grasp_step is not None and step % args_cli.force_log_interval == 0:
            net_force = torch.as_tensor(shaft.get_net_contact_forces(dt=sim_dt), device=device)
            finger_forces = torch.as_tensor(shaft.get_contact_force_matrix(dt=sim_dt), device=device)
            finger_force_norms = torch.linalg.vector_norm(finger_forces[0], dim=-1)

            slip_position = float("nan")
            slip_angle_deg = float("nan")
            if grasp_relative_pos is not None and grasp_relative_quat is not None:
                shaft_pos_w, shaft_quat_w = shaft.get_world_poses()
                shaft_pos_w = torch.as_tensor(shaft_pos_w, device=device)
                shaft_quat_w = torch.as_tensor(shaft_quat_w, device=device)
                current_relative_pos, current_relative_quat = math_utils.subtract_frame_transforms(
                    current_reference_w, current_reference_quat_w, shaft_pos_w, shaft_quat_w
                )
                slip_position = torch.linalg.vector_norm(current_relative_pos - grasp_relative_pos).item()
                slip_angle_deg = torch.rad2deg(
                    math_utils.quat_error_magnitude(current_relative_quat, grasp_relative_quat)
                ).item()
            print(
                f"[force] step={step} left={finger_force_norms[0].item():.4f} N "
                f"right={finger_force_norms[1].item():.4f} N "
                f"net_xyz={net_force[0].cpu().tolist()} N "
                f"relative_slip={slip_position:.6f} m relative_rotation={slip_angle_deg:.3f} deg"
            )

        if (
            reached_final_step is not None
            and not args_cli.keep_open
            and step - reached_final_step >= args_cli.hold_steps
            and (not disturbance_requested or disturbance_finished_step is not None)
        ):
            break
        step += 1

    final_hand_pose_w = robot.data.body_pose_w[:, hand_body_id]
    final_reference_w, final_reference_quat_w = math_utils.combine_frame_transforms(
        final_hand_pose_w[:, :3],
        final_hand_pose_w[:, 3:7],
        reference_offset_pos_hand,
        reference_offset_quat_hand,
    )
    final_error = torch.linalg.vector_norm(target_reference_w - final_reference_w, dim=-1).item()
    final_orientation_error = torch.rad2deg(
        math_utils.quat_error_magnitude(final_reference_quat_w, target_reference_quat_w)
    ).item()
    print(f"[result] reference_xyz={final_reference_w.cpu().tolist()[0]}")
    print(f"[result] target_error={final_error:.6f} m")
    print(f"[result] orientation_error={final_orientation_error:.3f} deg")
    print(f"[result] arm_joint_pos={robot.data.joint_pos[:, arm_joint_ids].cpu().tolist()[0]} rad")
    print(
        f"[result] gripper_joint_pos={robot.data.joint_pos[:, finger_joint_ids].cpu().tolist()[0]} m"
    )
    final_shaft_pos_w, final_shaft_quat_w = shaft.get_world_poses()
    print(f"[result] shaft_xyz={torch.as_tensor(final_shaft_pos_w)[0].cpu().tolist()} m")
    print(f"[result] shaft_quat_wxyz={torch.as_tensor(final_shaft_quat_w)[0].cpu().tolist()}")
    for body_name in ("panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_fingertip_centered"):
        body_ids, _ = robot.find_bodies(body_name, preserve_order=True)
        if body_ids:
            body_id = body_ids[0]
            print(
                f"[result] body={body_name} xyz={robot.data.body_pos_w[0, body_id].cpu().tolist()} "
                f"quat_wxyz={robot.data.body_quat_w[0, body_id].cpu().tolist()}"
            )
    if reached_final_step is None:
        print(
            f"[failure] Target was not reached within {args_cli.max_sim_steps} steps; "
            f"final error is {final_error:.6f} m."
        )
        return False
    return True


def main() -> bool:
    if args_cli.max_step <= 0.0 or args_cli.max_step > 0.010:
        raise ValueError("--max_step must be in (0, 0.010] m for this safety-limited script.")
    if args_cli.max_joint_step <= 0.0:
        raise ValueError("--max_joint_step must be positive.")
    if not 0.0 <= args_cli.gripper_position <= 0.04:
        raise ValueError("--gripper_position must be in [0.0, 0.04] m.")
    if not 0.0 <= args_cli.approach_gripper_position <= 0.04:
        raise ValueError("--approach_gripper_position must be in [0.0, 0.04] m.")
    if args_cli.orientation_tolerance_deg <= 0.0:
        raise ValueError("--orientation_tolerance_deg must be positive.")
    if args_cli.grasp_settle_steps < 0:
        raise ValueError("--grasp_settle_steps must be non-negative.")
    if not 0.0 <= args_cli.lift_after_grasp <= 0.10:
        raise ValueError("--lift_after_grasp must be in [0.0, 0.10] m.")
    if args_cli.hold_steps < 0:
        raise ValueError("--hold_steps must be non-negative; use --keep_open for an indefinite hold.")
    if args_cli.disturbance_delay_steps < 0 or args_cli.disturbance_steps < 0:
        raise ValueError("--disturbance_delay_steps and --disturbance_steps must be non-negative.")
    if args_cli.force_log_interval <= 0:
        raise ValueError("--force_log_interval must be positive.")
    usd_path = args_cli.usd_path.expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD scene not found: {usd_path}")
    args_cli.usd_path = usd_path

    if not sim_utils.open_stage(str(usd_path)):
        raise RuntimeError(f"Isaac Sim could not open stage: {usd_path}")
    stage = sim_utils.get_current_stage()
    robot_prim = stage.GetPrimAtPath(args_cli.robot_prim)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim does not exist: {args_cli.robot_prim}")
    shaft_prim = stage.GetPrimAtPath(args_cli.shaft_prim)
    if not shaft_prim.IsValid():
        raise RuntimeError(f"Shaft prim does not exist: {args_cli.shaft_prim}")

    if not args_cli.no_legacy_patch:
        _apply_runtime_compatibility(stage, args_cli.robot_prim)
    sim_utils.modify_rigid_body_properties(
        args_cli.robot_prim,
        sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        stage=stage,
    )
    # Approach with the scene's original gravity-off behavior.  If requested,
    # gravity is turned on only after the fingers have closed and stabilized.
    if args_cli.enable_gravity_after_grasp:
        shaft_physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(shaft_prim)
        shaft_physx_api.CreateDisableGravityAttr().Set(True)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    robot = _make_robot(args_cli.robot_prim)
    # Create before reset so the contact-report APIs are present when PhysX
    # parses the stage and builds its tensor views.
    shaft = RigidPrim(
        prim_paths_expr=args_cli.shaft_prim,
        name="shaft_validation_view",
        reset_xform_properties=False,
        track_contact_forces=True,
        prepare_contact_sensors=True,
        contact_filter_prim_paths_expr=list(args_cli.finger_contact_prims),
        max_contact_count=32,
    )
    sim.reset()
    if not robot.is_initialized:
        raise RuntimeError(f"Failed to initialize the existing articulation at {args_cli.robot_prim}.")
    shaft.initialize()
    if not shaft.is_physics_handle_valid():
        raise RuntimeError(f"Failed to create a rigid-body view for shaft: {args_cli.shaft_prim}")
    if args_cli.enable_gravity_after_grasp:
        shaft.disable_gravities()

    print("[INFO] Existing Franka initialized; starting safety-limited differential IK.")
    try:
        return run_controller(sim, stage, robot, shaft)
    finally:
        # For a stage opened from disk, Isaac Lab 2.3.2 clears callbacks and
        # the singleton without stopping/detaching the externally owned stage.
        # Calling stop() here can block during Kit shutdown on Isaac Sim 5.1.
        sim.clear_all_callbacks()
        sim.clear_instance()


if __name__ == "__main__":
    succeeded = False
    try:
        succeeded = main()
    finally:
        simulation_app.close()
    if not succeeded:
        raise SystemExit(2)
