#!/usr/bin/env python3
"""
Vision-guided waypoint commander for the Franka assembly workflow.

Workflow
--------
1. Move the robot to the configured fixed Global observation pose.
2. Cache one stable Global pose for all required parts.
3. Move to the Base XY at fixed table_z + 0.45 m.
4. Switch to the Local vision stage and record hole_large, hole_small, hole_screw.
5. Visit each part using its locked Global pose and enable Local detection only
   after live-TF arrival verification.
6. At safe height, align Local XY, select the shortest 180-degree-equivalent
   parallel-gripper yaw, verify it, then use Pilz LIN for Z-only descent/retreat.
7. For Shaft_large, Shaft_small and Screw, perform the same high-transfer and
   linear Z-only waypoint sequence above the corresponding recorded hole.

Important
---------
- This first version performs waypoint motion only.
- It does not close the gripper, descend to contact, insert, or use force control.
- Global orientation is ignored.
- Vision supplies waypoint X/Y only. Motion Z is always
  ``table_z_m + configured hover height``; detected Z is logged but ignored.
- Global/observation motion uses a fixed downward orientation. Local grasp yaw
  is applied only at safe height using the nearest 180-degree-equivalent
  parallel-gripper orientation. Descent and retreat use Pilz LIN without an
  OMPL fallback.
- Start with dry_run:=true. Only enable real motion after verifying every pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener


GLOBAL_PARTS = (
    "Base",
    "Shaft_large",
    "Shaft_small",
    "Screw",
    "Gear_compound_handle",
    "Gear_large_handle",
    "Gear_small_handle",
)

GLOBAL_HOLES = (
    "hole_large",
    "hole_small",
    "hole_screw",
)

LOCAL_PARTS_AND_HOLES = (
    "Shaft_large",
    "Shaft_small",
    "Screw",
    "Gear_compound_handle",
    "Gear_large_handle",
    "Gear_small_handle",
    "hole_large",
    "hole_small",
    "hole_screw",
)

HOLE_FOR_PART = {
    "Shaft_large": "hole_large",
    "Shaft_small": "hole_small",
    "Screw": "hole_screw",
}

SIMILAR_PART_FAMILIES = (
    ("Shaft_large", "Shaft_small"),
    (
        "Gear_compound_handle",
        "Gear_large_handle",
        "Gear_small_handle",
    ),
)


@dataclass(frozen=True)
class PartStep:
    part_name: str
    local_target_part: str
    observation_height_m: float
    approach_height_m: float = 0.20
    destination_hole: Optional[str] = None
    detection_observation_height_m: Optional[float] = None


TASK_STEPS = (
    PartStep(
        part_name="Shaft_large",
        local_target_part="Shaft_large",
        observation_height_m=0.40,
        destination_hole="hole_large",
    ),
    PartStep(
        part_name="Shaft_small",
        local_target_part="Shaft_small",
        observation_height_m=0.40,
        destination_hole="hole_small",
        # Use a narrower, closer view only while detecting the small shaft.
        # High XY/yaw alignment remains at the normal 0.40 m safe height.
        detection_observation_height_m=0.35,
    ),
    PartStep(
        part_name="Screw",
        local_target_part="Screw",
        observation_height_m=0.40,
        destination_hole="hole_screw",
    ),
    PartStep(
        part_name="Gear_compound_handle",
        local_target_part="Gear_compound_handle",
        observation_height_m=0.45,
    ),
    PartStep(
        part_name="Gear_large_handle",
        local_target_part="Gear_large_handle",
        observation_height_m=0.45,
    ),
    PartStep(
        part_name="Gear_small_handle",
        local_target_part="Gear_small_handle",
        observation_height_m=0.40,
    ),
)


class VisionGuidedWaypointCommander(Node):
    def __init__(self) -> None:
        super().__init__("vision_guided_waypoint_commander")

        defaults = {
            "base_frame": "panda_link0",
            "ee_link": "panda_hand",
            "move_group_action": "/move_action",
            "execute_trajectory_action": "/execute_trajectory",
            "move_group_name": "panda_arm",
            "local_vision_node": "/local_vision_force_adapter",
            "global_topic_prefix": "/vision/global",
            "local_topic_prefix": "/vision/local",
            "global_wait_timeout_sec": 120.0,
            # Global semantic holes are useful but optional. After all seven
            # parts arrive, briefly retain any hole messages already available
            # without making them a workflow completion requirement.
            "global_optional_hole_collection_sec": 2.0,
            "local_wait_timeout_sec": 60.0,
            "local_node_wait_timeout_sec": 300.0,
            # Table plane in base_frame. Keep this identical to the Global
            # adapter's table_z_m parameter.
            "table_z_m": 0.0,
            # Higher than part observations so the complete long/L-shaped Base
            # and all three holes fit more reliably in the Local image.
            "base_local_observation_height_m": 0.55,
            # Aim the Base Local camera directly at a semantic hole detected
            # during the locked Global snapshot. This follows Base translation
            # and rotation without a table-frame hard-coded offset.
            "base_local_camera_target": "hole_screw",
            # Local observations are specified by the desired camera optical
            # centre above the locked Global XY, not by placing panda_hand at
            # that XY.  The offset is always derived from the calibrated
            # T_gripper_camera; it is not a hand-written camera offset.
            "camera_centered_local_observation": True,
            # After reaching every Base/part Local observation pose, recompute
            # the actual camera optical-axis/table intersection from live TF.
            # Local detection is not enabled unless it lands close to the
            # locked Global target XY.
            "camera_aim_verification_tolerance_m": 0.02,
            "handeye_path": "/home/developer/multipanda_ws/hand_eye_result.npz",
            "settle_time_sec": 1.5,
            "dry_run": True,
            "linear_motion": False,
            "velocity_scale": 0.08,
            "acceleration_scale": 0.04,
            # Commander-side safety envelope. These checks do not alter either
            # notebook detector; they only reject unsafe motion targets.
            # 0.80 x 0.80 m table workspace, assuming panda_link0 is at the
            # midpoint of the rear edge and the tabletop is the link0 z=0 plane.
            "workspace_x_min_m": 0.00,
            "workspace_x_max_m": 0.80,
            "workspace_y_min_m": -0.40,
            "workspace_y_max_m": 0.40,
            "workspace_z_min_m": 0.00,
            "workspace_z_max_m": 0.85,
            # A Local result must remain close to the corresponding cached
            # Global result. Base-hole results are compared with Global Base.
            "max_local_global_xy_delta_m": 0.05,
            # Holes are not the Base center; the previous valid run measured
            # center-to-hole offsets up to about 0.106 m.
            "max_hole_base_xy_delta_m": 0.15,
            "similar_global_association_margin_m": 0.005,
            "min_similar_local_separation_m": 0.02,
            # A MoveIt SUCCESS is not trusted until the live base<-hand TF is
            # independently verified for several consecutive samples.
            "arrival_position_tolerance_m": 0.025,
            "arrival_orientation_tolerance_deg": 15.0,
            "arrival_tf_timeout_sec": 1.0,
            "arrival_tf_max_age_sec": 0.5,
            "arrival_verification_samples": 3,
            "arrival_sample_interval_sec": 0.15,
            # Critical Pilz LIN motions are planned without execution first.
            # The returned joint trajectory must be non-empty and contain
            # meaningful joint motion before it is sent to ExecuteTrajectory.
            "linear_plan_attempts": 2,
            "linear_min_trajectory_points": 2,
            "linear_min_joint_displacement_rad": 0.001,
            "linear_min_duration_sec": 0.05,
            "linear_start_xy_tolerance_m": 0.01,
            "linear_start_orientation_tolerance_deg": 5.0,
            "linear_min_z_displacement_m": 0.02,
            # A timestamped JSON file plus latest.json are updated after every
            # Local candidate, so an emergency stop does not lose earlier data.
            "local_pose_record_directory": "local_pose_records",
            # Fixed T_panda_link0_panda_hand captured with tf2_echo. Quaternion
            # is normalized and stored as w/x/y/z ROS command parameters.
            "move_to_global_observation_on_start": True,
            "global_observation_x_m": 0.357124617,
            "global_observation_y_m": 0.000100923,
            "global_observation_z_m": 0.696015951,
            "global_observation_q_w": 0.026555873828,
            "global_observation_q_x": 0.998979036635,
            "global_observation_q_y": 0.002766559556,
            "global_observation_q_z": 0.036441954906,
            # Real-robot validation showed the Panda finger/closing-axis
            # convention is 90 degrees from the notebook left->right grasp
            # line when interpreted as tool X. Use tool Y parallel to that line.
            "gripper_axis_mode": "y_parallel",
            # Screw is circular, so its detected yaw is physically ambiguous.
            # Use a fixed base-frame grasp yaw after its centre passes spatial
            # consensus instead of following the rotating notebook line yaw.
            "screw_fixed_grasp_yaw_deg": 0.0,
            # Fixed downward pose used by the previous test commander.
            # Verify this quaternion on the real robot before dry_run:=false.
            "down_q_w": 0.0,
            "down_q_x": 1.0,
            "down_q_y": 0.0,
            "down_q_z": 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_link = str(self.get_parameter("ee_link").value)
        self.move_group_name = str(self.get_parameter("move_group_name").value)
        self.global_topic_prefix = str(
            self.get_parameter("global_topic_prefix").value
        ).rstrip("/")
        self.local_topic_prefix = str(
            self.get_parameter("local_topic_prefix").value
        ).rstrip("/")
        self.global_wait_timeout_sec = float(
            self.get_parameter("global_wait_timeout_sec").value
        )
        self.global_optional_hole_collection_sec = float(
            self.get_parameter(
                "global_optional_hole_collection_sec"
            ).value
        )
        self.local_wait_timeout_sec = float(
            self.get_parameter("local_wait_timeout_sec").value
        )
        self.local_node_wait_timeout_sec = float(
            self.get_parameter("local_node_wait_timeout_sec").value
        )
        self.table_z_m = float(self.get_parameter("table_z_m").value)
        if not math.isfinite(self.table_z_m):
            raise ValueError("table_z_m must be finite")
        self.base_local_observation_height_m = float(
            self.get_parameter("base_local_observation_height_m").value
        )
        self.base_local_camera_target = str(
            self.get_parameter("base_local_camera_target").value
        ).strip()
        if self.base_local_camera_target not in GLOBAL_HOLES:
            raise ValueError(
                "base_local_camera_target must be one of "
                + ", ".join(GLOBAL_HOLES)
            )
        self.camera_centered_local_observation = bool(
            self.get_parameter("camera_centered_local_observation").value
        )
        self.camera_aim_verification_tolerance_m = float(
            self.get_parameter(
                "camera_aim_verification_tolerance_m"
            ).value
        )
        self.handeye_path = Path(
            str(self.get_parameter("handeye_path").value)
        ).expanduser()
        self.transform_gripper_camera = self._load_handeye_transform(
            self.handeye_path
        )
        self.settle_time_sec = float(self.get_parameter("settle_time_sec").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.linear_motion = bool(self.get_parameter("linear_motion").value)
        self.velocity_scale = float(self.get_parameter("velocity_scale").value)
        self.acceleration_scale = float(
            self.get_parameter("acceleration_scale").value
        )
        self.workspace_x_min_m = float(
            self.get_parameter("workspace_x_min_m").value
        )
        self.workspace_x_max_m = float(
            self.get_parameter("workspace_x_max_m").value
        )
        self.workspace_y_min_m = float(
            self.get_parameter("workspace_y_min_m").value
        )
        self.workspace_y_max_m = float(
            self.get_parameter("workspace_y_max_m").value
        )
        self.workspace_z_min_m = float(
            self.get_parameter("workspace_z_min_m").value
        )
        self.workspace_z_max_m = float(
            self.get_parameter("workspace_z_max_m").value
        )
        self.max_local_global_xy_delta_m = float(
            self.get_parameter("max_local_global_xy_delta_m").value
        )
        self.max_hole_base_xy_delta_m = float(
            self.get_parameter("max_hole_base_xy_delta_m").value
        )
        self.similar_global_association_margin_m = float(
            self.get_parameter("similar_global_association_margin_m").value
        )
        self.min_similar_local_separation_m = float(
            self.get_parameter("min_similar_local_separation_m").value
        )
        self.arrival_position_tolerance_m = float(
            self.get_parameter("arrival_position_tolerance_m").value
        )
        self.arrival_orientation_tolerance_deg = float(
            self.get_parameter("arrival_orientation_tolerance_deg").value
        )
        self.arrival_tf_timeout_sec = float(
            self.get_parameter("arrival_tf_timeout_sec").value
        )
        self.arrival_tf_max_age_sec = float(
            self.get_parameter("arrival_tf_max_age_sec").value
        )
        self.arrival_verification_samples = int(
            self.get_parameter("arrival_verification_samples").value
        )
        self.arrival_sample_interval_sec = float(
            self.get_parameter("arrival_sample_interval_sec").value
        )
        self.linear_plan_attempts = int(
            self.get_parameter("linear_plan_attempts").value
        )
        self.linear_min_trajectory_points = int(
            self.get_parameter("linear_min_trajectory_points").value
        )
        self.linear_min_joint_displacement_rad = float(
            self.get_parameter("linear_min_joint_displacement_rad").value
        )
        self.linear_min_duration_sec = float(
            self.get_parameter("linear_min_duration_sec").value
        )
        self.linear_start_xy_tolerance_m = float(
            self.get_parameter("linear_start_xy_tolerance_m").value
        )
        self.linear_start_orientation_tolerance_deg = float(
            self.get_parameter(
                "linear_start_orientation_tolerance_deg"
            ).value
        )
        self.linear_min_z_displacement_m = float(
            self.get_parameter("linear_min_z_displacement_m").value
        )
        bounds = (
            self.workspace_x_min_m,
            self.workspace_x_max_m,
            self.workspace_y_min_m,
            self.workspace_y_max_m,
            self.workspace_z_min_m,
            self.workspace_z_max_m,
            self.max_local_global_xy_delta_m,
            self.max_hole_base_xy_delta_m,
            self.similar_global_association_margin_m,
            self.min_similar_local_separation_m,
            self.arrival_position_tolerance_m,
            self.arrival_orientation_tolerance_deg,
            self.arrival_tf_timeout_sec,
            self.arrival_tf_max_age_sec,
            self.arrival_sample_interval_sec,
            self.linear_min_joint_displacement_rad,
            self.linear_min_duration_sec,
            self.linear_start_xy_tolerance_m,
            self.linear_start_orientation_tolerance_deg,
            self.linear_min_z_displacement_m,
            self.camera_aim_verification_tolerance_m,
            self.global_optional_hole_collection_sec,
        )
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("All workspace safety parameters must be finite")
        if not self.workspace_x_min_m < self.workspace_x_max_m:
            raise ValueError("workspace_x_min_m must be less than workspace_x_max_m")
        if not self.workspace_y_min_m < self.workspace_y_max_m:
            raise ValueError("workspace_y_min_m must be less than workspace_y_max_m")
        if not self.workspace_z_min_m < self.workspace_z_max_m:
            raise ValueError("workspace_z_min_m must be less than workspace_z_max_m")
        if self.max_local_global_xy_delta_m <= 0.0:
            raise ValueError("max_local_global_xy_delta_m must be positive")
        if self.max_hole_base_xy_delta_m <= 0.0:
            raise ValueError("max_hole_base_xy_delta_m must be positive")
        if self.similar_global_association_margin_m < 0.0:
            raise ValueError("similar_global_association_margin_m cannot be negative")
        if self.min_similar_local_separation_m <= 0.0:
            raise ValueError("min_similar_local_separation_m must be positive")
        if self.arrival_position_tolerance_m <= 0.0:
            raise ValueError("arrival_position_tolerance_m must be positive")
        if self.arrival_orientation_tolerance_deg <= 0.0:
            raise ValueError("arrival_orientation_tolerance_deg must be positive")
        if self.arrival_tf_timeout_sec <= 0.0:
            raise ValueError("arrival_tf_timeout_sec must be positive")
        if self.arrival_tf_max_age_sec <= 0.0:
            raise ValueError("arrival_tf_max_age_sec must be positive")
        if self.arrival_verification_samples < 1:
            raise ValueError("arrival_verification_samples must be at least 1")
        if self.arrival_sample_interval_sec <= 0.0:
            raise ValueError("arrival_sample_interval_sec must be positive")
        if self.linear_plan_attempts < 1:
            raise ValueError("linear_plan_attempts must be at least 1")
        if self.linear_min_trajectory_points < 2:
            raise ValueError("linear_min_trajectory_points must be at least 2")
        if self.linear_min_joint_displacement_rad <= 0.0:
            raise ValueError(
                "linear_min_joint_displacement_rad must be positive"
            )
        if self.linear_min_duration_sec <= 0.0:
            raise ValueError("linear_min_duration_sec must be positive")
        if self.linear_start_xy_tolerance_m <= 0.0:
            raise ValueError("linear_start_xy_tolerance_m must be positive")
        if self.linear_start_orientation_tolerance_deg <= 0.0:
            raise ValueError(
                "linear_start_orientation_tolerance_deg must be positive"
            )
        if self.linear_min_z_displacement_m <= 0.0:
            raise ValueError("linear_min_z_displacement_m must be positive")
        if self.camera_aim_verification_tolerance_m <= 0.0:
            raise ValueError(
                "camera_aim_verification_tolerance_m must be positive"
            )
        if self.global_optional_hole_collection_sec < 0.0:
            raise ValueError(
                "global_optional_hole_collection_sec cannot be negative"
            )
        self.move_to_global_observation_on_start = bool(
            self.get_parameter("move_to_global_observation_on_start").value
        )
        global_position = tuple(
            float(self.get_parameter(name).value)
            for name in (
                "global_observation_x_m",
                "global_observation_y_m",
                "global_observation_z_m",
            )
        )
        if not all(math.isfinite(value) for value in global_position):
            raise ValueError("Global observation position must be finite")
        (
            self.global_observation_x_m,
            self.global_observation_y_m,
            self.global_observation_z_m,
        ) = global_position
        global_q = tuple(
            float(self.get_parameter(name).value)
            for name in (
                "global_observation_q_w",
                "global_observation_q_x",
                "global_observation_q_y",
                "global_observation_q_z",
            )
        )
        if not all(math.isfinite(value) for value in global_q):
            raise ValueError("Global observation quaternion must be finite")
        global_q_norm = math.sqrt(sum(value * value for value in global_q))
        if global_q_norm < 1.0e-9:
            raise ValueError("Global observation quaternion must be non-zero")
        (
            self.global_observation_q_w,
            self.global_observation_q_x,
            self.global_observation_q_y,
            self.global_observation_q_z,
        ) = (value / global_q_norm for value in global_q)
        self.gripper_axis_mode = str(
            self.get_parameter("gripper_axis_mode").value
        ).strip().lower()
        if self.gripper_axis_mode not in {"x_parallel", "y_parallel"}:
            raise ValueError(
                "gripper_axis_mode must be 'x_parallel' or 'y_parallel'"
            )
        self.screw_fixed_grasp_yaw_deg = float(
            self.get_parameter("screw_fixed_grasp_yaw_deg").value
        )
        if not math.isfinite(self.screw_fixed_grasp_yaw_deg):
            raise ValueError("screw_fixed_grasp_yaw_deg must be finite")

        q = (
            float(self.get_parameter("down_q_w").value),
            float(self.get_parameter("down_q_x").value),
            float(self.get_parameter("down_q_y").value),
            float(self.get_parameter("down_q_z").value),
        )
        norm = math.sqrt(sum(value * value for value in q))
        if norm < 1e-9:
            raise ValueError("Downward quaternion must be non-zero")
        self.down_q_w, self.down_q_x, self.down_q_y, self.down_q_z = (
            value / norm for value in q
        )

        camera_offset_base = self._rotate_vector_wxyz(
            (
                self.down_q_w,
                self.down_q_x,
                self.down_q_y,
                self.down_q_z,
            ),
            tuple(float(value) for value in self.transform_gripper_camera[:3, 3]),
        )
        self.get_logger().info(
            "Local camera-centred observation: "
            f"enabled={self.camera_centered_local_observation}, "
            f"handeye={self.handeye_path}, "
            "R_base_gripper*t_gripper_camera="
            f"({camera_offset_base[0]:.4f}, {camera_offset_base[1]:.4f}, "
            f"{camera_offset_base[2]:.4f}) m at tool-down orientation"
        )

        self.global_poses: Dict[str, PoseStamped] = {}
        self.local_poses: Dict[str, PoseStamped] = {}
        self._global_snapshot_locked = False
        self._last_global_receive_time: Dict[str, float] = {}
        self._last_local_receive_time: Dict[str, float] = {}

        self._subscriptions = []
        for name in GLOBAL_PARTS + GLOBAL_HOLES:
            topic = f"{self.global_topic_prefix}/{name}/pose"
            self._subscriptions.append(
                self.create_subscription(
                    PoseStamped,
                    topic,
                    lambda msg, part=name: self._global_pose_callback(part, msg),
                    10,
                )
            )

        for name in LOCAL_PARTS_AND_HOLES:
            topic = f"{self.local_topic_prefix}/{name}/pose"
            self._subscriptions.append(
                self.create_subscription(
                    PoseStamped,
                    topic,
                    lambda msg, part=name: self._local_pose_callback(part, msg),
                    10,
                )
            )

        move_action = str(self.get_parameter("move_group_action").value)
        self._arm_client = ActionClient(self, MoveGroup, move_action)
        execute_action = str(
            self.get_parameter("execute_trajectory_action").value
        )
        self._execute_trajectory_client = ActionClient(
            self, ExecuteTrajectory, execute_action
        )

        local_node = str(self.get_parameter("local_vision_node").value)
        self._local_parameter_client = self.create_client(
            SetParameters, f"/{local_node.strip('/')}/set_parameters"
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        run_started = datetime.now(timezone.utc)
        record_directory = Path(
            str(self.get_parameter("local_pose_record_directory").value)
        ).expanduser()
        record_directory.mkdir(parents=True, exist_ok=True)
        run_id = run_started.strftime("%Y%m%dT%H%M%S_%fZ")
        self._local_record_path = (
            record_directory / f"local_pose_records_{run_id}.json"
        ).resolve()
        self._local_latest_path = (record_directory / "latest.json").resolve()
        self._local_record = {
            "schema_version": 2,
            "run_started_utc": run_started.isoformat(),
            "run_finished_utc": None,
            "status": "running",
            "failure_reason": None,
            "base_frame": self.base_frame,
            "safety_limits": {
                "workspace_x_m": [self.workspace_x_min_m, self.workspace_x_max_m],
                "workspace_y_m": [self.workspace_y_min_m, self.workspace_y_max_m],
                "workspace_z_m": [self.workspace_z_min_m, self.workspace_z_max_m],
                "max_local_global_xy_delta_m": self.max_local_global_xy_delta_m,
                "max_hole_base_xy_delta_m": self.max_hole_base_xy_delta_m,
                "similar_global_association_margin_m": (
                    self.similar_global_association_margin_m
                ),
                "min_similar_local_separation_m": (
                    self.min_similar_local_separation_m
                ),
                "arrival_position_tolerance_m": self.arrival_position_tolerance_m,
                "arrival_orientation_tolerance_deg": (
                    self.arrival_orientation_tolerance_deg
                ),
                "arrival_tf_max_age_sec": self.arrival_tf_max_age_sec,
            },
            "global_snapshot": {},
            "local_candidates": [],
            "accepted_local_poses": {},
            "motion_attempts": [],
        }
        self._write_local_record()

        self.get_logger().info(
            "Vision workflow commander initialized. "
            f"dry_run={self.dry_run}, frame={self.base_frame}, ee={self.ee_link}"
        )
        self.get_logger().info(
            f"Local pose record: {self._local_record_path} "
            f"(latest: {self._local_latest_path})"
        )
        self.get_logger().info(
            "Motion safety: "
            f"x=[{self.workspace_x_min_m:.3f}, {self.workspace_x_max_m:.3f}], "
            f"y=[{self.workspace_y_min_m:.3f}, {self.workspace_y_max_m:.3f}], "
            f"z=[{self.workspace_z_min_m:.3f}, {self.workspace_z_max_m:.3f}] m; "
            f"Local/Global XY <= {self.max_local_global_xy_delta_m:.3f} m; "
            f"hole/Base XY <= {self.max_hole_base_xy_delta_m:.3f} m; "
            f"similar Local separation >= "
            f"{self.min_similar_local_separation_m:.3f} m"
        )
        self.get_logger().info(
            "Arrival verification: "
            f"position <= {self.arrival_position_tolerance_m:.3f} m, "
            f"orientation <= {self.arrival_orientation_tolerance_deg:.1f} deg, "
            f"samples={self.arrival_verification_samples}; "
            f"TF age <= {self.arrival_tf_max_age_sec:.2f} s; "
            f"TF {self.base_frame} <- {self.ee_link}"
        )
        self.get_logger().info(
            "Validated LIN execution: plan-only attempts="
            f"{self.linear_plan_attempts}, points>="
            f"{self.linear_min_trajectory_points}, joint displacement>="
            f"{self.linear_min_joint_displacement_rad:.4f} rad, duration>="
            f"{self.linear_min_duration_sec:.3f} s; start XY<="
            f"{self.linear_start_xy_tolerance_m:.3f} m, orientation<="
            f"{self.linear_start_orientation_tolerance_deg:.1f} deg"
        )

    @staticmethod
    def _pose_as_dict(msg: PoseStamped) -> dict:
        p = msg.pose.position
        q = msg.pose.orientation
        return {
            "header": {
                "frame_id": msg.header.frame_id,
                "stamp": {
                    "sec": int(msg.header.stamp.sec),
                    "nanosec": int(msg.header.stamp.nanosec),
                },
            },
            "position_m": {
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
            },
            "orientation_xyzw": {
                "x": float(q.x),
                "y": float(q.y),
                "z": float(q.z),
                "w": float(q.w),
            },
        }

    def _write_local_record(self) -> None:
        text = json.dumps(self._local_record, indent=2, sort_keys=True) + "\n"
        for path in (self._local_record_path, self._local_latest_path):
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)

    def _append_local_candidate(self, name: str, msg: PoseStamped) -> int:
        entry = self._pose_as_dict(msg)
        entry.update(
            {
                "name": name,
                "received_utc": datetime.now(timezone.utc).isoformat(),
                "accepted": None,
                "validation": "pending",
            }
        )
        self._local_record["local_candidates"].append(entry)
        self._write_local_record()
        return len(self._local_record["local_candidates"]) - 1

    def _finish_local_candidate(
        self,
        index: int,
        name: str,
        msg: PoseStamped,
        accepted: bool,
        validation: str,
    ) -> None:
        entry = self._local_record["local_candidates"][index]
        entry["accepted"] = bool(accepted)
        entry["validation"] = validation
        if accepted:
            accepted_pose = self._pose_as_dict(msg)
            accepted_pose["accepted_utc"] = datetime.now(timezone.utc).isoformat()
            accepted_pose["validation"] = validation
            self._local_record["accepted_local_poses"][name] = accepted_pose
        self._write_local_record()

    def finalize_local_record(self, status: str, reason: Optional[str] = None) -> None:
        self._local_record["status"] = status
        self._local_record["failure_reason"] = reason
        self._local_record["run_finished_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        self._write_local_record()
        self.get_logger().info(
            f"Local pose record finalized with status={status}: "
            f"{self._local_record_path}"
        )

    def _global_pose_callback(self, name: str, msg: PoseStamped) -> None:
        if self._global_snapshot_locked:
            return
        if msg.header.frame_id != self.base_frame:
            self.get_logger().error(
                f"Reject Global {name}: frame_id={msg.header.frame_id!r}, "
                f"expected {self.base_frame!r}"
            )
            return
        self.global_poses[name] = msg
        self._last_global_receive_time[name] = time.monotonic()

    def _local_pose_callback(self, name: str, msg: PoseStamped) -> None:
        if msg.header.frame_id != self.base_frame:
            self.get_logger().error(
                f"Reject Local {name}: frame_id={msg.header.frame_id!r}, "
                f"expected {self.base_frame!r}"
            )
            return
        self.local_poses[name] = msg
        self._last_local_receive_time[name] = time.monotonic()

    @staticmethod
    def _pose_text(msg: PoseStamped) -> str:
        p = msg.pose.position
        q = msg.pose.orientation
        return (
            f"p=({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) m, "
            f"q_xyzw=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})"
        )

    def _spin_until(self, predicate, timeout_sec: float, description: str) -> None:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise TimeoutError(f"Timeout while waiting for {description}")

    def wait_for_all_global_poses(self) -> Dict[str, PoseStamped]:
        self.get_logger().info(
            "Waiting at the fixed Global observation pose for: "
            + ", ".join(GLOBAL_PARTS)
            + ". Semantic Base holes are optional and will be recorded when available."
        )

        self._spin_until(
            lambda: all(name in self.global_poses for name in GLOBAL_PARTS),
            self.global_wait_timeout_sec,
            "all seven Global part poses",
        )

        if (
            self.global_optional_hole_collection_sec > 0.0
            and not all(name in self.global_poses for name in GLOBAL_HOLES)
        ):
            self.get_logger().info(
                "Seven Global parts received; collecting optional semantic "
                f"Base holes for up to "
                f"{self.global_optional_hole_collection_sec:.1f} s"
            )
            deadline = (
                time.monotonic() + self.global_optional_hole_collection_sec
            )
            while rclpy.ok() and time.monotonic() < deadline:
                if all(name in self.global_poses for name in GLOBAL_HOLES):
                    break
                rclpy.spin_once(self, timeout_sec=0.1)

        available_holes = tuple(
            name for name in GLOBAL_HOLES if name in self.global_poses
        )
        snapshot_names = GLOBAL_PARTS + available_holes
        snapshot = {
            name: self.global_poses[name] for name in snapshot_names
        }
        for name, pose in snapshot.items():
            self._validate_reference_xy(pose, f"Global {name}")
        self._local_record["global_snapshot"] = {
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "poses": {
                name: self._pose_as_dict(pose) for name, pose in snapshot.items()
            },
        }
        # Freeze the seven required part poses plus whichever optional semantic
        # Base holes were available. Later Global publications must not alter
        # this workflow's observation targets.
        self.global_poses = snapshot
        self._global_snapshot_locked = True
        self._write_local_record()
        self.get_logger().info(
            "Global snapshot complete and locked for this workflow:"
        )
        for name, pose in snapshot.items():
            self.get_logger().info(f"  {name}: {self._pose_text(pose)}")
        missing_holes = tuple(
            name for name in GLOBAL_HOLES if name not in snapshot
        )
        if missing_holes:
            self.get_logger().warning(
                "Optional Global holes not available: "
                + ", ".join(missing_holes)
                + "; Base Local observation will fall back to the locked "
                "Global Base centre when the configured camera target is missing"
            )
        return snapshot

    def wait_for_fresh_local_pose(
        self,
        output_name: str,
        request_started_at: float,
    ) -> PoseStamped:
        self.local_poses.pop(output_name, None)
        self._last_local_receive_time.pop(output_name, None)

        self.get_logger().info(
            f"Waiting for fresh Local pose: "
            f"{self.local_topic_prefix}/{output_name}/pose"
        )

        deadline = time.monotonic() + self.local_wait_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            received_at = self._last_local_receive_time.get(output_name, 0.0)
            if output_name not in self.local_poses or received_at <= request_started_at:
                continue

            pose = self.local_poses.pop(output_name)
            self._last_local_receive_time.pop(output_name, None)
            candidate_index = self._append_local_candidate(output_name, pose)
            try:
                validation = self._validate_local_detection(output_name, pose)
            except ValueError as exc:
                reason = str(exc)
                self._finish_local_candidate(
                    candidate_index, output_name, pose, False, reason
                )
                self.get_logger().warning(
                    f"Reject Local {output_name} candidate and keep waiting: "
                    f"{reason}; {self._pose_text(pose)}"
                )
                continue

            self._finish_local_candidate(
                candidate_index, output_name, pose, True, validation
            )
            self.get_logger().info(
                f"Fresh safe Local {output_name}: {self._pose_text(pose)}; "
                f"{validation}"
            )
            return pose

        raise TimeoutError(
            f"Timeout while waiting for a safe fresh Local pose for {output_name}"
        )

    @staticmethod
    def _finite_pose_values(msg: PoseStamped) -> tuple[float, ...]:
        p = msg.pose.position
        q = msg.pose.orientation
        return (
            float(p.x),
            float(p.y),
            float(p.z),
            float(q.x),
            float(q.y),
            float(q.z),
            float(q.w),
        )

    def _validate_reference_xy(self, msg: PoseStamped, label: str) -> None:
        values = self._finite_pose_values(msg)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{label} pose contains a non-finite value")
        x, y = values[:2]
        if not self.workspace_x_min_m <= x <= self.workspace_x_max_m:
            raise ValueError(
                f"{label} x={x:.4f} m is outside "
                f"[{self.workspace_x_min_m:.4f}, {self.workspace_x_max_m:.4f}]"
            )
        if not self.workspace_y_min_m <= y <= self.workspace_y_max_m:
            raise ValueError(
                f"{label} y={y:.4f} m is outside "
                f"[{self.workspace_y_min_m:.4f}, {self.workspace_y_max_m:.4f}]"
            )

    def _validate_motion_target(
        self, x: float, y: float, z: float, label: str
    ) -> None:
        if not all(math.isfinite(value) for value in (x, y, z)):
            raise ValueError(f"{label} motion target contains a non-finite value")
        if not self.workspace_x_min_m <= x <= self.workspace_x_max_m:
            raise ValueError(
                f"SAFETY BLOCK {label}: target x={x:.4f} m is outside "
                f"[{self.workspace_x_min_m:.4f}, {self.workspace_x_max_m:.4f}]"
            )
        if not self.workspace_y_min_m <= y <= self.workspace_y_max_m:
            raise ValueError(
                f"SAFETY BLOCK {label}: target y={y:.4f} m is outside "
                f"[{self.workspace_y_min_m:.4f}, {self.workspace_y_max_m:.4f}]"
            )
        if not self.workspace_z_min_m <= z <= self.workspace_z_max_m:
            raise ValueError(
                f"SAFETY BLOCK {label}: target z={z:.4f} m is outside "
                f"[{self.workspace_z_min_m:.4f}, {self.workspace_z_max_m:.4f}]"
            )

    def _validate_local_detection(
        self, output_name: str, pose: PoseStamped
    ) -> str:
        self._validate_reference_xy(pose, f"Local {output_name}")
        if output_name.startswith("hole_"):
            if output_name in self.global_poses:
                # A Global semantic hole is the strongest protection against
                # Local large/small name swapping.
                reference_name = output_name
                limit = self.max_local_global_xy_delta_m
            else:
                reference_name = "Base"
                limit = self.max_hole_base_xy_delta_m
        else:
            reference_name = output_name
            limit = self.max_local_global_xy_delta_m

        if reference_name not in self.global_poses:
            raise ValueError(
                f"no cached Global {reference_name} pose exists for cross-check"
            )
        reference = self.global_poses[reference_name]
        dx = float(pose.pose.position.x) - float(reference.pose.position.x)
        dy = float(pose.pose.position.y) - float(reference.pose.position.y)
        distance = math.hypot(dx, dy)
        if distance > limit:
            raise ValueError(
                f"Local {output_name} is {distance:.4f} m from Global "
                f"{reference_name}, exceeding {limit:.4f} m"
            )

        family = next(
            (
                members
                for members in SIMILAR_PART_FAMILIES
                if output_name in members
            ),
            None,
        )
        association_note = ""
        if family is not None:
            other_global_distances = {}
            for other_name in family:
                if other_name == output_name or other_name not in self.global_poses:
                    continue
                other_pose = self.global_poses[other_name]
                other_global_distances[other_name] = math.hypot(
                    float(pose.pose.position.x)
                    - float(other_pose.pose.position.x),
                    float(pose.pose.position.y)
                    - float(other_pose.pose.position.y),
                )
            if other_global_distances:
                nearest_other_name, nearest_other_distance = min(
                    other_global_distances.items(), key=lambda item: item[1]
                )
                if (
                    distance + self.similar_global_association_margin_m
                    >= nearest_other_distance
                ):
                    raise ValueError(
                        f"Local {output_name} is not uniquely associated with its "
                        f"Global target: own distance={distance:.4f} m, "
                        f"Global {nearest_other_name} distance="
                        f"{nearest_other_distance:.4f} m, required margin="
                        f"{self.similar_global_association_margin_m:.4f} m"
                    )
                association_note = (
                    f"; nearest other Global {nearest_other_name}="
                    f"{nearest_other_distance:.4f} m"
                )

            accepted = self._local_record["accepted_local_poses"]
            for other_name in family:
                if other_name == output_name or other_name not in accepted:
                    continue
                other_position = accepted[other_name]["position_m"]
                local_separation = math.hypot(
                    float(pose.pose.position.x) - float(other_position["x"]),
                    float(pose.pose.position.y) - float(other_position["y"]),
                )
                if local_separation < self.min_similar_local_separation_m:
                    raise ValueError(
                        f"Local {output_name} nearly duplicates accepted "
                        f"{other_name}: separation={local_separation:.4f} m, "
                        f"minimum={self.min_similar_local_separation_m:.4f} m"
                    )
        return (
            f"XY distance to Global {reference_name}={distance:.4f} m "
            f"(limit {limit:.4f} m){association_note}"
        )

    def wait_for_local_node(self) -> None:
        self.get_logger().info(
            "Waiting for Local vision parameter service. "
            "Stop the Global stage and start run_local_vision_stage.py now."
        )
        deadline = time.monotonic() + self.local_node_wait_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self._local_parameter_client.wait_for_service(timeout_sec=0.5):
                self.get_logger().info("Local vision node is available.")
                return
        raise TimeoutError("Local vision parameter service did not become available")

    def set_local_request(
        self,
        target_part: str,
        base_hole_target: str = "hole_large",
    ) -> float:
        return self._set_local_parameters(
            [
                Parameter("target_part", value=target_part),
                Parameter("base_hole_target", value=base_hole_target),
                Parameter("detection_enabled", value=True),
            ],
            f"target_part={target_part}, "
            f"base_hole_target={base_hole_target}, detection_enabled=true",
        )

    def set_local_detection_enabled(self, enabled: bool) -> float:
        state = "true" if enabled else "false"
        return self._set_local_parameters(
            [Parameter("detection_enabled", value=bool(enabled))],
            f"detection_enabled={state}",
        )

    def _set_local_parameters(
        self,
        parameters: list[Parameter],
        description: str,
    ) -> float:
        request = SetParameters.Request()
        request.parameters = [item.to_parameter_msg() for item in parameters]
        future = self._local_parameter_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if not future.done() or future.result() is None:
            raise RuntimeError("Local parameter request did not complete")

        results = future.result().results
        failures = [
            result.reason or "unknown reason"
            for result in results
            if not result.successful
        ]
        if failures:
            raise RuntimeError(
                "Local parameter update rejected: " + "; ".join(failures)
            )

        started_at = time.monotonic()
        self.get_logger().info(f"Local parameters set: {description}")
        return started_at

    def _wait_for_move_group(self) -> None:
        if self.dry_run:
            return
        if not self._arm_client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError("MoveGroup action server is unavailable")
        if not self._execute_trajectory_client.wait_for_server(
            timeout_sec=20.0
        ):
            raise RuntimeError("ExecuteTrajectory action server is unavailable")

    @staticmethod
    def _normalized_xyzw(reference: PoseStamped) -> tuple[float, float, float, float]:
        q = reference.pose.orientation
        values = (float(q.x), float(q.y), float(q.z), float(q.w))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Local orientation contains a non-finite value")
        norm = math.sqrt(sum(value * value for value in values))
        if norm < 1.0e-9:
            raise ValueError("Local orientation is a zero quaternion")
        return tuple(value / norm for value in values)

    def _target_orientation_wxyz(
        self,
        reference: PoseStamped,
        use_local_orientation: bool,
        label: str,
    ) -> tuple[float, float, float, float]:
        if not use_local_orientation:
            return (
                self.down_q_w,
                self.down_q_x,
                self.down_q_y,
                self.down_q_z,
            )

        # The Local adapter publishes the notebook grasp direction as a planar
        # object-yaw quaternion. Extract that yaw and add roll=pi exactly once,
        # keeping conventional tool +Z downward while rotating the gripper in XY.
        x, y, z, w = self._normalized_xyzw(reference)
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        if self.gripper_axis_mode == "y_parallel":
            yaw += math.pi / 2.0
        half_yaw = 0.5 * yaw
        command = (
            0.0,
            math.cos(half_yaw),
            math.sin(half_yaw),
            0.0,
        )
        self.get_logger().info(
            f"{label}: applying Local grasp yaw={math.degrees(yaw):.2f} deg; "
            f"axis_mode={self.gripper_axis_mode}; tool remains downward"
        )
        return command

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _nearest_parallel_gripper_yaw(
        raw_yaw: float,
        current_yaw: float,
    ) -> float:
        candidates = [raw_yaw + multiple * math.pi for multiple in range(-2, 3)]
        return min(
            candidates,
            key=lambda candidate: abs(
                VisionGuidedWaypointCommander._wrap_angle(
                    candidate - current_yaw
                )
            ),
        )

    def _current_hand_yaw(self) -> float:
        self._spin_for_tf_updates(self.arrival_sample_interval_sec)
        try:
            transform = self._tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_link,
                Time(),
                timeout=Duration(seconds=self.arrival_tf_timeout_sec),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot select shortest grasp yaw because live TF "
                f"{self.base_frame} <- {self.ee_link} is unavailable: {exc}"
            ) from exc
        stamp_ns = (
            int(transform.header.stamp.sec) * 1_000_000_000
            + int(transform.header.stamp.nanosec)
        )
        age_sec = (
            self.get_clock().now().nanoseconds - stamp_ns
        ) / 1_000_000_000.0
        if not -0.05 <= age_sec <= self.arrival_tf_max_age_sec:
            raise RuntimeError(
                f"Cannot select shortest grasp yaw from stale TF age="
                f"{age_sec:.3f} s"
            )
        rotation = transform.transform.rotation
        x, y, z, w = map(
            float,
            (rotation.x, rotation.y, rotation.z, rotation.w),
        )
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def _shortest_equivalent_local_orientation(
        self,
        reference: PoseStamped,
        label: str,
    ) -> tuple[float, float, float, float]:
        raw = self._target_orientation_wxyz(reference, True, label)
        q_w, q_x, q_y, q_z = raw
        raw_yaw = math.atan2(
            2.0 * (q_w * q_z + q_x * q_y),
            1.0 - 2.0 * (q_y * q_y + q_z * q_z),
        )
        current_yaw = self._current_hand_yaw()
        selected_yaw = self._nearest_parallel_gripper_yaw(
            raw_yaw,
            current_yaw,
        )
        rotation_delta = self._wrap_angle(selected_yaw - current_yaw)
        half_yaw = 0.5 * selected_yaw
        selected = (
            0.0,
            math.cos(half_yaw),
            math.sin(half_yaw),
            0.0,
        )
        self.get_logger().info(
            f"{label}: symmetric gripper shortest-yaw selection: "
            f"notebook line yaw={math.degrees(raw_yaw):.2f} deg, "
            f"current hand yaw={math.degrees(current_yaw):.2f} deg, "
            f"selected equivalent yaw={math.degrees(selected_yaw):.2f} deg, "
            f"rotation delta={math.degrees(rotation_delta):.2f} deg"
        )
        return selected

    @staticmethod
    def _motion_pose_dict(
        position_xyz: tuple[float, float, float],
        quaternion_wxyz: tuple[float, float, float, float],
    ) -> dict:
        x, y, z = position_xyz
        q_w, q_x, q_y, q_z = quaternion_wxyz
        return {
            "position_m": {"x": x, "y": y, "z": z},
            "orientation_xyzw": {
                "x": q_x,
                "y": q_y,
                "z": q_z,
                "w": q_w,
            },
        }

    def _append_motion_attempt(
        self,
        label: str,
        target_position: tuple[float, float, float],
        target_quaternion: tuple[float, float, float, float],
    ) -> int:
        entry = {
            "label": label,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "finished_utc": None,
            "status": "sending",
            "moveit_error_code": None,
            "target": self._motion_pose_dict(
                target_position, target_quaternion
            ),
            "arrival_verification": {
                "base_frame": self.base_frame,
                "ee_frame": self.ee_link,
                "position_tolerance_m": self.arrival_position_tolerance_m,
                "orientation_tolerance_deg": (
                    self.arrival_orientation_tolerance_deg
                ),
                "tf_max_age_sec": self.arrival_tf_max_age_sec,
                "samples": [],
            },
        }
        self._local_record["motion_attempts"].append(entry)
        self._write_local_record()
        return len(self._local_record["motion_attempts"]) - 1

    def _set_motion_status(
        self,
        index: int,
        status: str,
        moveit_error_code: Optional[int] = None,
        reason: Optional[str] = None,
        finished: bool = True,
    ) -> None:
        entry = self._local_record["motion_attempts"][index]
        entry["status"] = status
        if moveit_error_code is not None:
            entry["moveit_error_code"] = int(moveit_error_code)
        if reason is not None:
            entry["failure_reason"] = reason
        if finished:
            entry["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self._write_local_record()

    @staticmethod
    def _quaternion_angle_error_deg(
        target_wxyz: tuple[float, float, float, float],
        actual_wxyz: tuple[float, float, float, float],
    ) -> float:
        target_norm = math.sqrt(sum(value * value for value in target_wxyz))
        actual_norm = math.sqrt(sum(value * value for value in actual_wxyz))
        if target_norm < 1.0e-9 or actual_norm < 1.0e-9:
            raise ValueError("Cannot compare a zero quaternion")
        dot = sum(
            target * actual
            for target, actual in zip(target_wxyz, actual_wxyz)
        ) / (target_norm * actual_norm)
        # q and -q represent the same physical orientation.
        dot = min(1.0, max(-1.0, abs(dot)))
        return math.degrees(2.0 * math.acos(dot))

    def _spin_for_tf_updates(self, duration_sec: float) -> None:
        deadline = time.monotonic() + max(0.0, duration_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.1, max(0.0, remaining)))

    def _verify_arrival(
        self,
        attempt_index: int,
        label: str,
        target_position: tuple[float, float, float],
        target_quaternion: tuple[float, float, float, float],
    ) -> bool:
        # Keep processing TF subscriptions while waiting for mechanical settling.
        self._spin_for_tf_updates(self.settle_time_sec)
        entry = self._local_record["motion_attempts"][attempt_index]
        samples = entry["arrival_verification"]["samples"]
        previous_tf_stamp_ns: Optional[int] = None

        for sample_number in range(1, self.arrival_verification_samples + 1):
            self._spin_for_tf_updates(self.arrival_sample_interval_sec)
            try:
                transform = self._tf_buffer.lookup_transform(
                    self.base_frame,
                    self.ee_link,
                    Time(),
                    timeout=Duration(seconds=self.arrival_tf_timeout_sec),
                )
            except Exception as exc:
                reason = (
                    f"Arrival TF {self.base_frame} <- {self.ee_link} unavailable: "
                    f"{exc}"
                )
                self._set_motion_status(
                    attempt_index, "arrival_tf_unavailable", reason=reason
                )
                self.get_logger().error(f"{label}: {reason}")
                return False

            translation = transform.transform.translation
            rotation = transform.transform.rotation
            actual_position = (
                float(translation.x),
                float(translation.y),
                float(translation.z),
            )
            actual_quaternion = (
                float(rotation.w),
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
            )
            position_error = math.sqrt(
                sum(
                    (actual - target) ** 2
                    for actual, target in zip(actual_position, target_position)
                )
            )
            orientation_error = self._quaternion_angle_error_deg(
                target_quaternion, actual_quaternion
            )
            tf_stamp_ns = (
                int(transform.header.stamp.sec) * 1_000_000_000
                + int(transform.header.stamp.nanosec)
            )
            tf_age_sec = (
                self.get_clock().now().nanoseconds - tf_stamp_ns
            ) / 1_000_000_000.0
            tf_is_fresh = (
                -0.05 <= tf_age_sec <= self.arrival_tf_max_age_sec
            )
            tf_is_new_sample = (
                previous_tf_stamp_ns is None
                or tf_stamp_ns > previous_tf_stamp_ns
            )
            sample_ok = (
                position_error <= self.arrival_position_tolerance_m
                and orientation_error <= self.arrival_orientation_tolerance_deg
                and tf_is_fresh
                and tf_is_new_sample
            )
            sample = {
                "sample": sample_number,
                "captured_utc": datetime.now(timezone.utc).isoformat(),
                "tf_stamp": {
                    "sec": int(transform.header.stamp.sec),
                    "nanosec": int(transform.header.stamp.nanosec),
                },
                "actual": self._motion_pose_dict(
                    actual_position, actual_quaternion
                ),
                "position_error_m": position_error,
                "orientation_error_deg": orientation_error,
                "tf_age_sec": tf_age_sec,
                "tf_is_fresh": tf_is_fresh,
                "tf_timestamp_increased": tf_is_new_sample,
                "within_tolerance": sample_ok,
            }
            samples.append(sample)
            self._write_local_record()
            previous_tf_stamp_ns = tf_stamp_ns

            if not sample_ok:
                reason = (
                    f"arrival mismatch: position error={position_error:.4f} m "
                    f"(limit {self.arrival_position_tolerance_m:.4f}), "
                    f"orientation error={orientation_error:.2f} deg "
                    f"(limit {self.arrival_orientation_tolerance_deg:.2f}), "
                    f"TF age={tf_age_sec:.3f} s "
                    f"(limit {self.arrival_tf_max_age_sec:.3f}), "
                    f"TF timestamp increased={tf_is_new_sample}"
                )
                self._set_motion_status(
                    attempt_index, "arrival_mismatch", reason=reason
                )
                self.get_logger().error(
                    f"SAFETY ABORT {label}: {reason}; no subsequent motion "
                    "will be sent"
                )
                return False

        maximum_position_error = max(
            sample["position_error_m"] for sample in samples
        )
        maximum_orientation_error = max(
            sample["orientation_error_deg"] for sample in samples
        )
        entry["arrival_verification"]["maximum_position_error_m"] = (
            maximum_position_error
        )
        entry["arrival_verification"]["maximum_orientation_error_deg"] = (
            maximum_orientation_error
        )
        self._set_motion_status(attempt_index, "verified_arrival")
        self.get_logger().info(
            f"{label}: MoveIt SUCCESS and live TF arrival verified; "
            f"max position error={maximum_position_error:.4f} m, "
            f"max orientation error={maximum_orientation_error:.2f} deg"
        )
        return True

    def _record_dry_run_motion(
        self,
        label: str,
        target_position: tuple[float, float, float],
        target_quaternion: tuple[float, float, float, float],
    ) -> None:
        index = self._append_motion_attempt(
            label, target_position, target_quaternion
        )
        self._set_motion_status(index, "dry_run_skipped")

    def _current_hand_pose_wxyz(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
        float,
    ]:
        self._spin_for_tf_updates(self.arrival_sample_interval_sec)
        transform = self._tf_buffer.lookup_transform(
            self.base_frame,
            self.ee_link,
            Time(),
            timeout=Duration(seconds=self.arrival_tf_timeout_sec),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        stamp_ns = (
            int(transform.header.stamp.sec) * 1_000_000_000
            + int(transform.header.stamp.nanosec)
        )
        age_sec = (
            self.get_clock().now().nanoseconds - stamp_ns
        ) / 1_000_000_000.0
        if not -0.05 <= age_sec <= self.arrival_tf_max_age_sec:
            raise RuntimeError(
                f"Current TF is stale: age={age_sec:.3f} s, limit="
                f"{self.arrival_tf_max_age_sec:.3f} s"
            )
        return (
            (
                float(translation.x),
                float(translation.y),
                float(translation.z),
            ),
            (
                float(rotation.w),
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
            ),
            age_sec,
        )

    @staticmethod
    def _duration_seconds(duration) -> float:
        return float(duration.sec) + float(duration.nanosec) * 1.0e-9

    def _trajectory_metrics(self, trajectory) -> dict:
        joint_trajectory = trajectory.joint_trajectory
        points = list(joint_trajectory.points)
        metrics = {
            "joint_names": list(joint_trajectory.joint_names),
            "point_count": len(points),
            "duration_sec": (
                self._duration_seconds(points[-1].time_from_start)
                if points else 0.0
            ),
            "first_positions": (
                [float(value) for value in points[0].positions]
                if points else []
            ),
            "last_positions": (
                [float(value) for value in points[-1].positions]
                if points else []
            ),
            "maximum_joint_displacement_rad": 0.0,
        }
        if len(points) >= 2:
            first = metrics["first_positions"]
            last = metrics["last_positions"]
            if len(first) == len(last) and first:
                metrics["maximum_joint_displacement_rad"] = max(
                    abs(end - start) for start, end in zip(first, last)
                )
        return metrics

    def _validate_linear_plan(self, metrics: dict) -> Optional[str]:
        if metrics["point_count"] < self.linear_min_trajectory_points:
            return (
                f"trajectory has {metrics['point_count']} points; expected at "
                f"least {self.linear_min_trajectory_points}"
            )
        if not metrics["joint_names"]:
            return "trajectory has no joint names"
        if (
            len(metrics["first_positions"]) != len(metrics["joint_names"])
            or len(metrics["last_positions"]) != len(metrics["joint_names"])
        ):
            return "trajectory endpoint dimensions do not match joint names"
        if metrics["duration_sec"] < self.linear_min_duration_sec:
            return (
                f"trajectory duration={metrics['duration_sec']:.4f} s; "
                f"minimum={self.linear_min_duration_sec:.4f} s"
            )
        if (
            metrics["maximum_joint_displacement_rad"]
            < self.linear_min_joint_displacement_rad
        ):
            return (
                "trajectory maximum joint displacement="
                f"{metrics['maximum_joint_displacement_rad']:.6f} rad; "
                f"minimum={self.linear_min_joint_displacement_rad:.6f} rad"
            )
        return None

    def _execute_validated_linear_goal(
        self,
        goal: MoveGroup.Goal,
        label: str,
        target_position: tuple[float, float, float],
        target_quaternion: tuple[float, float, float, float],
    ) -> bool:
        attempt_index = self._append_motion_attempt(
            label, target_position, target_quaternion
        )
        entry = self._local_record["motion_attempts"][attempt_index]
        entry["planning_pipeline"] = str(goal.request.pipeline_id)
        entry["planner_id"] = str(goal.request.planner_id)
        entry["execution_mode"] = "plan_validate_execute"

        try:
            start_position, start_quaternion, tf_age_sec = (
                self._current_hand_pose_wxyz()
            )
        except Exception as exc:
            reason = f"LIN start TF unavailable: {exc}"
            self._set_motion_status(
                attempt_index, "linear_start_tf_unavailable", reason=reason
            )
            self.get_logger().error(f"{label}: {reason}")
            return False

        xy_error = math.hypot(
            start_position[0] - target_position[0],
            start_position[1] - target_position[1],
        )
        orientation_error = self._quaternion_angle_error_deg(
            target_quaternion, start_quaternion
        )
        z_displacement = target_position[2] - start_position[2]
        entry["linear_start_check"] = {
            "actual": self._motion_pose_dict(
                start_position, start_quaternion
            ),
            "target": self._motion_pose_dict(
                target_position, target_quaternion
            ),
            "xy_error_m": float(xy_error),
            "orientation_error_deg": float(orientation_error),
            "z_displacement_m": float(z_displacement),
            "tf_age_sec": float(tf_age_sec),
        }
        self._write_local_record()
        start_problem = None
        if xy_error > self.linear_start_xy_tolerance_m:
            start_problem = (
                f"LIN start XY error={xy_error:.4f} m exceeds "
                f"{self.linear_start_xy_tolerance_m:.4f} m"
            )
        elif (
            orientation_error
            > self.linear_start_orientation_tolerance_deg
        ):
            start_problem = (
                f"LIN start orientation error={orientation_error:.2f} deg "
                f"exceeds {self.linear_start_orientation_tolerance_deg:.2f} deg"
            )
        elif abs(z_displacement) < self.linear_min_z_displacement_m:
            start_problem = (
                f"LIN requested Z displacement={z_displacement:.4f} m is "
                f"smaller than {self.linear_min_z_displacement_m:.4f} m"
            )
        if start_problem is not None:
            self._set_motion_status(
                attempt_index, "linear_start_rejected", reason=start_problem
            )
            self.get_logger().error(f"SAFETY ABORT {label}: {start_problem}")
            return False

        goal.planning_options.plan_only = True
        entry["linear_plan_attempts"] = []
        planned_trajectory = None
        for plan_attempt in range(1, self.linear_plan_attempts + 1):
            send_future = self._arm_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()
            plan_record = {
                "attempt": plan_attempt,
                "accepted": bool(
                    goal_handle is not None and goal_handle.accepted
                ),
            }
            entry["linear_plan_attempts"].append(plan_record)
            self._write_local_record()
            if goal_handle is None or not goal_handle.accepted:
                plan_record["failure_reason"] = "MoveGroup plan goal rejected"
                continue

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            wrapped_result = result_future.result()
            if wrapped_result is None:
                plan_record["failure_reason"] = "MoveGroup returned no plan"
                continue
            result = wrapped_result.result
            error_code = int(result.error_code.val)
            plan_record["moveit_error_code"] = error_code
            plan_record["planning_time_sec"] = float(result.planning_time)
            if error_code != MoveItErrorCodes.SUCCESS:
                plan_record["failure_reason"] = (
                    f"planning failed with MoveIt code {error_code}"
                )
                continue

            metrics = self._trajectory_metrics(result.planned_trajectory)
            plan_record["trajectory"] = metrics
            validation_error = self._validate_linear_plan(metrics)
            if validation_error is not None:
                plan_record["failure_reason"] = validation_error
                self.get_logger().warning(
                    f"{label}: reject LIN plan attempt {plan_attempt}: "
                    f"{validation_error}"
                )
                continue
            plan_record["validated"] = True
            planned_trajectory = result.planned_trajectory
            self.get_logger().info(
                f"{label}: validated LIN plan attempt {plan_attempt}: "
                f"points={metrics['point_count']}, duration="
                f"{metrics['duration_sec']:.3f} s, max joint displacement="
                f"{metrics['maximum_joint_displacement_rad']:.4f} rad"
            )
            break

        if planned_trajectory is None:
            reason = (
                f"No valid LIN trajectory after {self.linear_plan_attempts} "
                "plan-only attempts"
            )
            self._set_motion_status(
                attempt_index, "linear_plan_invalid", reason=reason
            )
            self.get_logger().error(f"SAFETY ABORT {label}: {reason}")
            return False

        execute_goal = ExecuteTrajectory.Goal()
        execute_goal.trajectory = planned_trajectory
        execute_future = self._execute_trajectory_client.send_goal_async(
            execute_goal
        )
        rclpy.spin_until_future_complete(self, execute_future)
        execute_handle = execute_future.result()
        if execute_handle is None or not execute_handle.accepted:
            reason = "ExecuteTrajectory goal rejected"
            self._set_motion_status(
                attempt_index, "execute_trajectory_rejected", reason=reason
            )
            self.get_logger().error(f"{label}: {reason}")
            return False

        self._set_motion_status(
            attempt_index, "execute_trajectory_running", finished=False
        )
        result_future = execute_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            reason = "ExecuteTrajectory returned no result"
            self._set_motion_status(
                attempt_index, "execute_trajectory_no_result", reason=reason
            )
            self.get_logger().error(f"{label}: {reason}")
            return False
        error_code = int(wrapped_result.result.error_code.val)
        if error_code != MoveItErrorCodes.SUCCESS:
            reason = f"ExecuteTrajectory failed with MoveIt code {error_code}"
            self._set_motion_status(
                attempt_index,
                "execute_trajectory_failed",
                moveit_error_code=error_code,
                reason=reason,
            )
            self.get_logger().error(f"{label}: {reason}")
            return False

        self._set_motion_status(
            attempt_index,
            "execute_success_pending_tf_verification",
            moveit_error_code=error_code,
            finished=False,
        )
        self.get_logger().info(
            f"{label}: validated trajectory execution returned SUCCESS; "
            f"verifying live TF {self.base_frame} <- {self.ee_link}"
        )
        return self._verify_arrival(
            attempt_index,
            label,
            target_position,
            target_quaternion,
        )

    def _execute_move_group_goal(
        self,
        goal: MoveGroup.Goal,
        label: str,
        target_position: tuple[float, float, float],
        target_quaternion: tuple[float, float, float, float],
    ) -> bool:
        attempt_index = self._append_motion_attempt(
            label, target_position, target_quaternion
        )
        attempt = self._local_record["motion_attempts"][attempt_index]
        attempt["planning_pipeline"] = str(goal.request.pipeline_id)
        attempt["planner_id"] = str(goal.request.planner_id)
        self._write_local_record()
        send_future = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._set_motion_status(attempt_index, "moveit_goal_rejected")
            self.get_logger().error(f"{label}: MoveGroup goal rejected")
            return False

        self._set_motion_status(
            attempt_index, "moveit_goal_accepted", finished=False
        )

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            self._set_motion_status(attempt_index, "moveit_no_result")
            self.get_logger().error(f"{label}: MoveGroup returned no result")
            return False

        error_code = int(wrapped_result.result.error_code.val)
        if error_code != MoveItErrorCodes.SUCCESS:
            self._set_motion_status(
                attempt_index, "moveit_failed", moveit_error_code=error_code
            )
            self.get_logger().error(
                f"{label}: MoveIt failed with error code {error_code}"
            )
            return False

        self._set_motion_status(
            attempt_index,
            "moveit_success_pending_tf_verification",
            moveit_error_code=error_code,
            finished=False,
        )
        self.get_logger().info(
            f"{label}: MoveIt returned SUCCESS; verifying live "
            f"TF {self.base_frame} <- {self.ee_link} before continuing"
        )
        return self._verify_arrival(
            attempt_index,
            label,
            target_position,
            target_quaternion,
        )

    def move_to_global_observation_pose(self) -> bool:
        label = "fixed Global observation pose"
        self._validate_motion_target(
            self.global_observation_x_m,
            self.global_observation_y_m,
            self.global_observation_z_m,
            label,
        )
        self.get_logger().info(
            f"MOVE {label}: "
            f"p=({self.global_observation_x_m:.9f}, "
            f"{self.global_observation_y_m:.9f}, "
            f"{self.global_observation_z_m:.9f}) m, "
            f"q_wxyz=({self.global_observation_q_w:.9f}, "
            f"{self.global_observation_q_x:.9f}, "
            f"{self.global_observation_q_y:.9f}, "
            f"{self.global_observation_q_z:.9f})"
        )
        if self.dry_run:
            self._record_dry_run_motion(
                label,
                (
                    self.global_observation_x_m,
                    self.global_observation_y_m,
                    self.global_observation_z_m,
                ),
                (
                    self.global_observation_q_w,
                    self.global_observation_q_x,
                    self.global_observation_q_y,
                    self.global_observation_q_z,
                ),
            )
            self.get_logger().warning("dry_run=true: no robot motion sent")
            time.sleep(0.2)
            return True
        goal = self._build_move_group_goal(
            self.global_observation_x_m,
            self.global_observation_y_m,
            self.global_observation_z_m,
            self.global_observation_q_w,
            self.global_observation_q_x,
            self.global_observation_q_y,
            self.global_observation_q_z,
        )
        return self._execute_move_group_goal(
            goal,
            label,
            (
                self.global_observation_x_m,
                self.global_observation_y_m,
                self.global_observation_z_m,
            ),
            (
                self.global_observation_q_w,
                self.global_observation_q_x,
                self.global_observation_q_y,
                self.global_observation_q_z,
            ),
        )

    @staticmethod
    def _load_handeye_transform(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(
                "Camera-centred Local observation requires the hand-eye file: "
                f"{path}"
            )
        with np.load(path, allow_pickle=False) as archive:
            if "T_gripper_camera" not in archive.files:
                raise KeyError(
                    f"Hand-eye file {path} lacks T_gripper_camera; "
                    f"keys={archive.files}"
                )
            transform = np.asarray(
                archive["T_gripper_camera"], dtype=np.float64
            )
        if transform.shape != (4, 4):
            raise ValueError(
                f"T_gripper_camera must be 4x4, got {transform.shape}"
            )
        if not np.all(np.isfinite(transform)):
            raise ValueError("T_gripper_camera contains non-finite values")
        if not np.allclose(
            transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1.0e-8
        ):
            raise ValueError("T_gripper_camera has an invalid last row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
            raise ValueError("T_gripper_camera rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
            raise ValueError("T_gripper_camera rotation determinant is not +1")
        return transform

    @staticmethod
    def _rotate_vector_wxyz(
        quaternion_wxyz: tuple[float, float, float, float],
        vector_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        w, x, y, z = quaternion_wxyz
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1.0e-9:
            raise ValueError("Cannot rotate camera offset with a zero quaternion")
        w, x, y, z = (value / norm for value in (w, x, y, z))
        rotation = (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        )
        vx, vy, vz = vector_xyz
        return tuple(
            row[0] * vx + row[1] * vy + row[2] * vz for row in rotation
        )

    def _verify_camera_aim_at_table(
        self,
        desired_xy: tuple[float, float],
        label: str,
    ) -> bool:
        """Verify the live calibrated camera centre ray reaches desired table XY."""
        try:
            hand_position, hand_quaternion, tf_age_sec = (
                self._current_hand_pose_wxyz()
            )
            camera_offset_base = self._rotate_vector_wxyz(
                hand_quaternion,
                tuple(
                    float(value)
                    for value in self.transform_gripper_camera[:3, 3]
                ),
            )
            optical_axis_base = self._rotate_vector_wxyz(
                hand_quaternion,
                tuple(
                    float(value)
                    for value in self.transform_gripper_camera[:3, 2]
                ),
            )
            camera_origin = tuple(
                hand + offset
                for hand, offset in zip(hand_position, camera_offset_base)
            )
            if abs(optical_axis_base[2]) < 1.0e-6:
                raise RuntimeError(
                    "calibrated camera optical axis is parallel to table"
                )
            ray_distance = (
                self.table_z_m - camera_origin[2]
            ) / optical_axis_base[2]
            if ray_distance <= 0.0:
                raise RuntimeError(
                    "calibrated camera optical axis points away from table"
                )
            actual_xy = (
                camera_origin[0] + ray_distance * optical_axis_base[0],
                camera_origin[1] + ray_distance * optical_axis_base[1],
            )
            error_m = math.hypot(
                actual_xy[0] - desired_xy[0],
                actual_xy[1] - desired_xy[1],
            )
        except Exception as exc:
            self.get_logger().error(
                f"SAFETY ABORT {label}: camera aim verification failed: {exc}"
            )
            return False

        verification = {
            "desired_table_xy_m": {
                "x": desired_xy[0],
                "y": desired_xy[1],
            },
            "actual_table_xy_m": {
                "x": actual_xy[0],
                "y": actual_xy[1],
            },
            "error_m": error_m,
            "tolerance_m": self.camera_aim_verification_tolerance_m,
            "tf_age_sec": tf_age_sec,
            "handeye_path": str(self.handeye_path),
        }
        if self._local_record["motion_attempts"]:
            self._local_record["motion_attempts"][-1][
                "camera_aim_verification"
            ] = verification
            self._write_local_record()

        if error_m > self.camera_aim_verification_tolerance_m:
            self.get_logger().error(
                f"SAFETY ABORT {label}: actual camera table aim="
                f"({actual_xy[0]:.4f}, {actual_xy[1]:.4f}), desired="
                f"({desired_xy[0]:.4f}, {desired_xy[1]:.4f}), "
                f"error={error_m:.4f} m exceeds "
                f"{self.camera_aim_verification_tolerance_m:.4f} m; "
                "Local detection remains disabled"
            )
            return False

        self.get_logger().info(
            f"CAMERA AIM VERIFIED {label}: table intersection="
            f"({actual_xy[0]:.4f}, {actual_xy[1]:.4f}), desired="
            f"({desired_xy[0]:.4f}, {desired_xy[1]:.4f}), "
            f"error={error_m:.4f} m"
        )
        return True

    def move_above_pose(
        self,
        reference: PoseStamped,
        height_m: float,
        label: str,
        use_local_orientation: bool = False,
        orientation_override_wxyz: Optional[
            tuple[float, float, float, float]
        ] = None,
        force_linear: bool = False,
        center_camera_over_reference: bool = False,
        camera_target_offset_xy: tuple[float, float] = (0.0, 0.0),
    ) -> bool:
        target_x = float(reference.pose.position.x)
        target_y = float(reference.pose.position.y)
        # XY comes from vision, while every safe hover/observation Z is anchored
        # to the configured table plane. Object thickness or depth noise must
        # not raise/lower these waypoints.
        target_z = self.table_z_m + float(height_m)
        if orientation_override_wxyz is None:
            target_q_w, target_q_x, target_q_y, target_q_z = (
                self._target_orientation_wxyz(
                    reference, use_local_orientation, label
                )
            )
        else:
            target_q_w, target_q_x, target_q_y, target_q_z = (
                orientation_override_wxyz
            )

        if (
            center_camera_over_reference
            and not self.camera_centered_local_observation
        ):
            raise RuntimeError(
                f"{label}: camera-centred Local observation was requested but "
                "camera_centered_local_observation=false"
            )

        if center_camera_over_reference:
            desired_camera_x = (
                float(reference.pose.position.x)
                + float(camera_target_offset_xy[0])
            )
            desired_camera_y = (
                float(reference.pose.position.y)
                + float(camera_target_offset_xy[1])
            )
            if use_local_orientation or orientation_override_wxyz is not None:
                raise ValueError(
                    "Camera-centred observation requires the fixed tool-down "
                    "orientation"
                )
            camera_origin_offset_base = self._rotate_vector_wxyz(
                (target_q_w, target_q_x, target_q_y, target_q_z),
                tuple(
                    float(value)
                    for value in self.transform_gripper_camera[:3, 3]
                ),
            )
            optical_axis_base = self._rotate_vector_wxyz(
                (target_q_w, target_q_x, target_q_y, target_q_z),
                tuple(
                    float(value)
                    for value in self.transform_gripper_camera[:3, 2]
                ),
            )
            camera_origin_z = target_z + camera_origin_offset_base[2]
            if abs(optical_axis_base[2]) < 1.0e-6:
                raise ValueError("Calibrated camera optical axis is parallel to table")
            ray_distance = (
                self.table_z_m - camera_origin_z
            ) / optical_axis_base[2]
            if ray_distance <= 0.0:
                raise ValueError(
                    "Calibrated camera optical axis points away from the table"
                )
            ray_offset_x = (
                camera_origin_offset_base[0]
                + ray_distance * optical_axis_base[0]
            )
            ray_offset_y = (
                camera_origin_offset_base[1]
                + ray_distance * optical_axis_base[1]
            )
            # Solve for the hand XY whose calibrated camera centre ray
            # intersects the table plane at the requested inspection centre.
            target_x = desired_camera_x - ray_offset_x
            target_y = desired_camera_y - ray_offset_y
            self.get_logger().info(
                f"CAMERA-CENTRED {label}: desired camera XY="
                f"({desired_camera_x:.4f}, {desired_camera_y:.4f}); "
                f"reference Global XY="
                f"({float(reference.pose.position.x):.4f}, "
                f"{float(reference.pose.position.y):.4f}); "
                f"inspection offset="
                f"({float(camera_target_offset_xy[0]):.4f}, "
                f"{float(camera_target_offset_xy[1]):.4f}); commanded hand XY="
                f"({target_x:.4f}, {target_y:.4f}); calibrated table-ray offset="
                f"({ray_offset_x:.4f}, {ray_offset_y:.4f}) m"
            )

        self._validate_motion_target(target_x, target_y, target_z, label)

        self.get_logger().info(
            f"MOVE {label}: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f}) m "
            f"= table_z({self.table_z_m:.4f}) + {height_m:.3f} m; "
            f"detected_z={float(reference.pose.position.z):.4f} m (not used for motion); "
            f"q_wxyz=({target_q_w:.4f}, {target_q_x:.4f}, "
            f"{target_q_y:.4f}, {target_q_z:.4f}); "
            f"planner={'Pilz LIN' if force_linear else 'configured/OMPL'}"
        )

        if self.dry_run:
            self._record_dry_run_motion(
                label,
                (target_x, target_y, target_z),
                (target_q_w, target_q_x, target_q_y, target_q_z),
            )
            self.get_logger().warning("dry_run=true: no robot motion sent")
            time.sleep(0.2)
            return True

        goal = self._build_move_group_goal(
            target_x,
            target_y,
            target_z,
            target_q_w,
            target_q_x,
            target_q_y,
            target_q_z,
            force_linear=force_linear,
        )
        if force_linear or self.linear_motion:
            moved = self._execute_validated_linear_goal(
                goal,
                label,
                (target_x, target_y, target_z),
                (target_q_w, target_q_x, target_q_y, target_q_z),
            )
        else:
            moved = self._execute_move_group_goal(
                goal,
                label,
                (target_x, target_y, target_z),
                (target_q_w, target_q_x, target_q_y, target_q_z),
            )
        if not moved:
            return False
        if center_camera_over_reference:
            return self._verify_camera_aim_at_table(
                (
                    desired_camera_x,
                    desired_camera_y,
                ),
                label,
            )
        return True

    def _build_move_group_goal(
        self,
        x: float,
        y: float,
        z: float,
        q_w: float,
        q_x: float,
        q_y: float,
        q_z: float,
        force_linear: bool = False,
    ) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = self.move_group_name
        request.num_planning_attempts = 20
        request.allowed_planning_time = 8.0
        request.max_velocity_scaling_factor = self.velocity_scale
        request.max_acceleration_scaling_factor = self.acceleration_scale

        if force_linear or self.linear_motion:
            request.pipeline_id = "pilz_industrial_motion_planner"
            request.planner_id = "LIN"
        else:
            request.pipeline_id = "ompl"
            request.planner_id = "RRTConnectkConfigDefault"

        constraints = Constraints()

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.base_frame
        position_constraint.link_name = self.ee_link
        position_constraint.weight = 1.0

        region = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.002]

        target_pose = PoseStamped()
        target_pose.header.frame_id = self.base_frame
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        target_pose.pose.orientation.x = q_x
        target_pose.pose.orientation.y = q_y
        target_pose.pose.orientation.z = q_z
        target_pose.pose.orientation.w = q_w

        region.primitives.append(sphere)
        region.primitive_poses.append(target_pose.pose)
        position_constraint.constraint_region = region
        constraints.position_constraints.append(position_constraint)

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.base_frame
        orientation_constraint.link_name = self.ee_link
        orientation_constraint.orientation = target_pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.03
        orientation_constraint.absolute_y_axis_tolerance = 0.03
        orientation_constraint.absolute_z_axis_tolerance = 0.03
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints.append(orientation_constraint)

        request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        return goal

    def record_base_holes(self) -> Dict[str, PoseStamped]:
        # The robot has already moved to Base + 0.45 m before the Local node starts.
        holes: Dict[str, PoseStamped] = {}
        self.set_local_detection_enabled(False)
        try:
            for hole_name in ("hole_large", "hole_small", "hole_screw"):
                request_started = self.set_local_request(
                    target_part="Base",
                    base_hole_target=hole_name,
                )
                holes[hole_name] = self.wait_for_fresh_local_pose(
                    hole_name,
                    request_started,
                )
        finally:
            self.set_local_detection_enabled(False)

        self.get_logger().info("Recorded all three Local hole poses:")
        for name, pose in holes.items():
            self.get_logger().info(f"  {name}: {self._pose_text(pose)}")
        return holes

    def execute_part_step(
        self,
        step: PartStep,
        holes: Dict[str, PoseStamped],
    ) -> None:
        self.get_logger().info("=" * 72)
        self.get_logger().info(f"Starting step: {step.part_name}")

        # Never let the previous target keep detecting while the camera/robot is
        # moving. The new target is enabled only after live-TF arrival succeeds.
        self.set_local_detection_enabled(False)
        coarse_pose = self.global_poses[step.part_name]
        detection_observation_height_m = (
            step.detection_observation_height_m
            if step.detection_observation_height_m is not None
            else step.observation_height_m
        )
        if not self.move_above_pose(
            coarse_pose,
            detection_observation_height_m,
            f"{step.part_name} Local observation pose",
            center_camera_over_reference=True,
        ):
            raise RuntimeError(
                f"Could not move above Global pose of {step.part_name}"
            )

        request_started = self.set_local_request(
            target_part=step.local_target_part,
            base_hole_target="hole_large",
        )
        try:
            refined_pose = self.wait_for_fresh_local_pose(
                step.part_name,
                request_started,
            )
        except Exception:
            # Preserve the original detection timeout/validation exception.
            # A stopped Local node must not replace it with the less useful
            # "Local parameter request did not complete" cleanup error.
            try:
                self.set_local_detection_enabled(False)
            except Exception as cleanup_error:
                self.get_logger().warning(
                    "Could not disable Local detection while handling the "
                    f"original failure: {cleanup_error}"
                )
            raise
        # Detection must be off before grasp/placement motion starts.
        self.set_local_detection_enabled(False)

        # Stage 1: translate to refined XY while remaining at the safe Local
        # observation height and keeping the fixed tool-down orientation.
        if not self.move_above_pose(
            refined_pose,
            step.observation_height_m,
            f"{step.part_name} high Local XY alignment",
        ):
            raise RuntimeError(
                f"Could not align high above Local XY of {step.part_name}"
            )

        # Stage 2: the circular Screw has no stable visual yaw. Once its centre
        # has passed Local spatial consensus, use a configured fixed yaw.
        # All non-circular parts retain their notebook grasp direction.
        if step.part_name == "Screw":
            fixed_yaw = math.radians(self.screw_fixed_grasp_yaw_deg)
            half_yaw = 0.5 * fixed_yaw
            grasp_orientation = (
                0.0,
                math.cos(half_yaw),
                math.sin(half_yaw),
                0.0,
            )
            self.get_logger().info(
                "Screw centre consensus accepted; ignoring rotationally "
                "ambiguous Local yaw and applying fixed grasp yaw="
                f"{self.screw_fixed_grasp_yaw_deg:.2f} deg"
            )
        else:
            # Exploit the 180-degree symmetry of a parallel gripper and rotate
            # to the equivalent notebook yaw requiring the least hand rotation.
            grasp_orientation = self._shortest_equivalent_local_orientation(
                refined_pose,
                f"{step.part_name} high grasp-yaw alignment",
            )
        if not self.move_above_pose(
            refined_pose,
            step.observation_height_m,
            f"{step.part_name} high grasp-yaw alignment",
            orientation_override_wxyz=grasp_orientation,
        ):
            raise RuntimeError(
                f"Could not align grasp yaw of {step.part_name} at safe height"
            )

        # Stage 3: no XY or yaw change is permitted during descent. Pilz LIN is
        # mandatory; failure aborts and never falls back to an OMPL trajectory.
        if not self.move_above_pose(
            refined_pose,
            step.approach_height_m,
            f"{step.part_name} linear Z descent to refined grasp waypoint",
            orientation_override_wxyz=grasp_orientation,
            force_linear=True,
        ):
            raise RuntimeError(
                f"Could not linearly descend above Local pose of {step.part_name}"
            )

        # Return to the safe height with the same XY/yaw before any horizontal
        # transfer to a destination or the next Global observation target.
        if not self.move_above_pose(
            refined_pose,
            step.observation_height_m,
            f"{step.part_name} linear Z retreat from grasp waypoint",
            orientation_override_wxyz=grasp_orientation,
            force_linear=True,
        ):
            raise RuntimeError(
                f"Could not linearly retreat from {step.part_name}"
            )

        if step.destination_hole is not None:
            hole_pose = holes[step.destination_hole]
            if not self.move_above_pose(
                hole_pose,
                step.observation_height_m,
                f"{step.part_name} high transfer above {step.destination_hole}",
            ):
                raise RuntimeError(
                    f"Could not transfer high above {step.destination_hole}"
                )
            if not self.move_above_pose(
                hole_pose,
                step.approach_height_m,
                f"{step.part_name} linear Z descent above {step.destination_hole}",
                force_linear=True,
            ):
                raise RuntimeError(
                    f"Could not linearly descend above {step.destination_hole}"
                )
            if not self.move_above_pose(
                hole_pose,
                step.observation_height_m,
                f"{step.part_name} linear Z retreat above {step.destination_hole}",
                force_linear=True,
            ):
                raise RuntimeError(
                    f"Could not linearly retreat above {step.destination_hole}"
                )

    def run_workflow(self) -> None:
        self._wait_for_move_group()

        if self.move_to_global_observation_on_start:
            if not self.move_to_global_observation_pose():
                raise RuntimeError("Could not move to fixed Global observation pose")
        else:
            self.get_logger().warning(
                "move_to_global_observation_on_start=false: assuming the robot "
                "is already at the calibrated Global observation pose"
            )

        self.wait_for_all_global_poses()

        self.get_logger().info(
            "Global poses are cached. They remain available after the Global "
            "vision process is stopped."
        )

        # Prefer the configured locked Global semantic hole. If that optional
        # detection was unavailable, fall back to the required Global Base
        # centre rather than blocking the Global -> Local stage transition.
        if self.base_local_camera_target in self.global_poses:
            base_camera_target_name = self.base_local_camera_target
            base_camera_target = self.global_poses[base_camera_target_name]
            self.get_logger().info(
                "Base Local camera target comes from locked Global "
                f"{base_camera_target_name}: "
                f"{self._pose_text(base_camera_target)}"
            )
        else:
            base_camera_target_name = "Base"
            base_camera_target = self.global_poses["Base"]
            self.get_logger().warning(
                "Configured optional Global camera target "
                f"{self.base_local_camera_target} was not detected; using "
                "locked Global Base centre for the higher Local observation: "
                f"{self._pose_text(base_camera_target)}"
            )
        if not self.move_above_pose(
            base_camera_target,
            self.base_local_observation_height_m,
            f"Base Local hover centred on Global {base_camera_target_name}",
            center_camera_over_reference=True,
        ):
            raise RuntimeError(
                "Could not centre the camera above Global "
                f"{base_camera_target_name}"
            )

        self.wait_for_local_node()
        self.set_local_detection_enabled(False)
        holes = self.record_base_holes()

        for step in TASK_STEPS:
            self.execute_part_step(step, holes)

        self.get_logger().info("=" * 72)
        self.get_logger().info(
            "Waypoint workflow completed. No gripper closing, contact descent, "
            "force-control insertion or gear placement was executed."
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionGuidedWaypointCommander()
    try:
        node.run_workflow()
        node.finalize_local_record("completed")
    except KeyboardInterrupt:
        node.finalize_local_record("interrupted", "KeyboardInterrupt")
        node.get_logger().warning("Workflow interrupted by user")
    except Exception as exc:
        node.finalize_local_record("aborted", str(exc))
        node.get_logger().fatal(f"Workflow aborted: {exc}")
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
