#!/usr/bin/env python3
"""ROS2 wrapper around the exact local RGB-D notebook detector."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import local_notebook_detector as notebook
from vision_common import (
    BASE_HOLE_NAMES,
    DEFAULT_HANDEYE_PATH,
    LOCAL_REALSENSE_ROS_PARAMETERS,
    PART_NAME_MAP,
    RealSenseParameterEnforcer,
    ROSAlignedRGBDCamera,
    SpatialConsensusFilter,
    TFTransformProvider,
    apply_black_background_workspace,
    force_control_part_name,
    load_handeye_transform,
    normalize_part_key,
    planar_object_quaternion,
    quaternion_from_roll_pitch_yaw,
    transform_points_camera_to_base,
)

try:
    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except ImportError:
    rclpy = None
    PoseStamped = Any
    Image = Any

    class Node:  # type: ignore[no-redef]
        pass


LOCAL_TOPIC_PREFIX = "/vision/local"


@dataclass(frozen=True)
class LocalDetection:
    part_key: str
    output_name: str
    confidence: str
    source: str
    raw_result: Mapping[str, Any]
    annotated: np.ndarray
    debug_images: Mapping[str, np.ndarray] = field(default_factory=dict)


class LocalGraspDetector:
    """Pure notebook detector. No thresholds or branch ordering are redefined."""

    def __init__(self) -> None:
        self.last_annotated: np.ndarray | None = None
        self.last_debug_images: dict[str, np.ndarray] = {}

    def detect(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: Mapping[str, float],
        part_key: str,
        base_hole_target: str = "hole_large",
    ) -> LocalDetection | None:
        key = normalize_part_key(part_key)
        filtered_color, filtered_depth, workspace_mask = (
            apply_black_background_workspace(color_bgr, depth_m)
        )
        result, annotated, debug = notebook.run_local_refinement(
            filtered_color, filtered_depth, dict(intrinsics), key
        )
        annotated[workspace_mask == 0] = color_bgr[workspace_mask == 0]
        workspace_contours, _ = cv2.findContours(
            workspace_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, workspace_contours, -1, (0, 255, 0), 2)
        debug = dict(debug)
        debug["black_background_workspace_mask"] = workspace_mask
        debug["black_background_masked_input"] = filtered_color
        self.last_annotated = annotated
        self.last_debug_images = debug
        if result.get("status") != "ok":
            return None
        if key == "base":
            if base_hole_target not in BASE_HOLE_NAMES:
                raise ValueError(
                    f"base_hole_target must be one of {BASE_HOLE_NAMES}"
                )
            if base_hole_target not in result.get("pixel_points", {}):
                return None
            output_name = base_hole_target
        else:
            output_name = force_control_part_name(key)
        return LocalDetection(
            part_key=key,
            output_name=output_name,
            confidence=str(result.get("confidence", "low")),
            source=str(
                result.get("yaw_source")
                or result.get("base_hole_source")
                or result.get("rule_note", "notebook")
            ),
            raw_result=result,
            annotated=annotated,
            debug_images=debug,
        )

    def failed_debug_detection(
        self,
        part_key: str,
        base_hole_target: str,
    ) -> LocalDetection | None:
        """Expose current-frame preprocessing debug without a pose candidate."""
        if self.last_annotated is None:
            return None
        key = normalize_part_key(part_key)
        output_name = (
            base_hole_target if key == "base" else force_control_part_name(key)
        )
        return LocalDetection(
            part_key=key,
            output_name=output_name,
            confidence="low",
            source="notebook_detection_failed",
            raw_result={},
            annotated=self.last_annotated,
            debug_images=self.last_debug_images,
        )


class LocalVisionForceAdapter(Node):
    """Publish only the requested local target using synchronized image-time TF."""

    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("ROS2 Python packages are required to run this node")
        super().__init__("local_vision_force_adapter")
        defaults = {
            "target_part": "Shaft_large",
            "base_hole_target": "hole_large",
            # Outer ROS execution gate only. The notebook detector is unchanged
            # and runs only after Commander verifies arrival over the cached
            # Global target and explicitly enables this parameter.
            "detection_enabled": False,
            "color_topic": "/camera/camera/color/image_raw",
            "aligned_depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "realsense_camera_node": "/camera/camera",
            "enforce_fixed_camera_parameters": True,
            "strict_camera_parameter_verification": True,
            "camera_parameter_retry_sec": 2.0,
            "base_frame": "panda_link0",
            "ee_frame": "panda_hand",
            "camera_frame": "camera_color_optical_frame",
            "handeye_path": DEFAULT_HANDEYE_PATH,
            "handeye_yaml_path": "",
            "topic_prefix": LOCAL_TOPIC_PREFIX,
            "expected_width": 640,
            "expected_height": 480,
            "warmup_frames": 20,
            "sync_queue_size": 10,
            "sync_slop_sec": 0.05,
            "motion_threshold": 5.0,
            "stable_frames_required": 5,
            "motion_reset_threshold": 15.0,
            "motion_reset_frames": 20,
            "consensus_required_frames": 5,
            "consensus_tolerance_m": 0.008,
            # Screw is rotationally symmetric in the Local view. Its notebook
            # midpoint remains authoritative, but its frame-to-frame yaw is not
            # used for publication stability or robot orientation.
            "screw_fixed_yaw_deg": 0.0,
            "publish_debug_images": True,
            "publish_low_confidence": False,
            "publish_rate": 10.0,
            "tf_timeout_sec": 0.2,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.topic_prefix = str(self.get_parameter("topic_prefix").value).rstrip("/")
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.publish_debug_images = bool(
            self.get_parameter("publish_debug_images").value
        )
        self.publish_low_confidence = bool(
            self.get_parameter("publish_low_confidence").value
        )
        self.screw_fixed_yaw_deg = float(
            self.get_parameter("screw_fixed_yaw_deg").value
        )
        if not math.isfinite(self.screw_fixed_yaw_deg):
            raise ValueError("screw_fixed_yaw_deg must be finite")
        self.motion_reset_threshold = float(
            self.get_parameter("motion_reset_threshold").value
        )
        self.motion_reset_frames = int(
            self.get_parameter("motion_reset_frames").value
        )
        if self.motion_reset_threshold <= 0.0:
            raise ValueError("motion_reset_threshold must be positive")
        if self.motion_reset_frames < 1:
            raise ValueError("motion_reset_frames must be positive")
        self._unstable_frame_count = 0
        handeye_path = str(self.get_parameter("handeye_path").value).strip()
        legacy_path = str(self.get_parameter("handeye_yaml_path").value).strip()
        self.transform_gripper_camera = load_handeye_transform(legacy_path or handeye_path)
        self.tf_provider = TFTransformProvider(self, self.base_frame, self.ee_frame)
        self.detector = LocalGraspDetector()
        self.consensus = SpatialConsensusFilter(
            int(self.get_parameter("consensus_required_frames").value),
            float(self.get_parameter("consensus_tolerance_m").value),
        )
        output_names = list(PART_NAME_MAP.values()) + list(BASE_HOLE_NAMES)
        self._pose_publishers = {
            name: self.create_publisher(
                PoseStamped, f"{self.topic_prefix}/{name}/pose", 10
            )
            for name in output_names
        }
        self._debug_publishers: dict[str, Any] = {}
        self._bridge = CvBridge()
        self._active_request: tuple[str, str] | None = None
        self._compound_axis_reference: np.ndarray | None = None
        self._detection_was_enabled = False
        self._enabled_after_ros_ns: int | None = None
        self.camera = ROSAlignedRGBDCamera(
            self,
            str(self.get_parameter("color_topic").value),
            str(self.get_parameter("aligned_depth_topic").value),
            str(self.get_parameter("camera_info_topic").value),
            int(self.get_parameter("warmup_frames").value),
            int(self.get_parameter("sync_queue_size").value),
            float(self.get_parameter("sync_slop_sec").value),
            float(self.get_parameter("motion_threshold").value),
            int(self.get_parameter("stable_frames_required").value),
            int(self.get_parameter("expected_width").value),
            int(self.get_parameter("expected_height").value),
        )
        self.camera_parameters = RealSenseParameterEnforcer(
            self,
            str(self.get_parameter("realsense_camera_node").value),
            bool(self.get_parameter("enforce_fixed_camera_parameters").value),
            bool(self.get_parameter("strict_camera_parameter_verification").value),
            float(self.get_parameter("camera_parameter_retry_sec").value),
            on_ready=self.camera.reset_after_reconfigure,
            parameters=LOCAL_REALSENSE_ROS_PARAMETERS,
            profile_name="local",
        )
        rate = float(self.get_parameter("publish_rate").value)
        if rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        self.timer = self.create_timer(1.0 / rate, self._process_frame)
        self.get_logger().info(
            f"Notebook-exact local detector; TF {self.base_frame} <- {self.ee_frame}; "
            "target_part is read every frame for runtime switching"
        )
        self.get_logger().info(
            "Outer publication gate: "
            f"motion_threshold={self.get_parameter('motion_threshold').value}, "
            f"stable_frames={self.get_parameter('stable_frames_required').value}, "
            f"hard_reset_score={self.motion_reset_threshold}, "
            f"hard_reset_frames={self.motion_reset_frames}; "
            "sparse valid detections retain consensus"
        )

    def _current_request(self) -> tuple[str, str]:
        key = normalize_part_key(str(self.get_parameter("target_part").value))
        hole = str(self.get_parameter("base_hole_target").value).strip()
        if key == "base" and hole not in BASE_HOLE_NAMES:
            raise ValueError(f"base_hole_target must be one of {BASE_HOLE_NAMES}")
        return key, hole

    @staticmethod
    def _camera_xyz(point: Mapping[str, Any]) -> np.ndarray | None:
        values = [
            point.get("camera_x_m"),
            point.get("camera_y_m"),
            point.get("camera_z_m"),
        ]
        if any(value is None for value in values):
            return None
        array = np.asarray(values, dtype=np.float64)
        return array if np.all(np.isfinite(array)) else None

    def _base_geometry(
        self,
        detection: LocalDetection,
        transform_base_gripper: np.ndarray,
        base_hole_target: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        result = detection.raw_result
        if detection.part_key == "base":
            camera = self._camera_xyz(result["camera_points"][base_hole_target])
            if camera is None:
                raise ValueError(f"{base_hole_target} has invalid ring depth")
            base = transform_points_camera_to_base(
                [camera], transform_base_gripper, self.transform_gripper_camera
            )[0]
            return np.asarray([base]), quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)

        if detection.part_key == "screw":
            # A circular screw has no meaningful visual yaw.  Require only the
            # notebook/fallback centre to have valid camera XYZ, then perform
            # spatial consensus on that centre alone.  Endpoint depth and
            # top-bar orientation must not prevent a safe fixed-yaw grasp.
            midpoint = self._camera_xyz(result.get("grasp_midpoint", {}))
            if midpoint is None:
                raise ValueError("screw grasp midpoint lacks valid camera XYZ")
            base_midpoint = transform_points_camera_to_base(
                [midpoint],
                transform_base_gripper,
                self.transform_gripper_camera,
            )[0]
            fixed_yaw = math.radians(self.screw_fixed_yaw_deg)
            return (
                np.asarray([base_midpoint], dtype=np.float64),
                quaternion_from_roll_pitch_yaw(0.0, 0.0, fixed_yaw),
            )

        left = self._camera_xyz(result.get("grasp_left", {}))
        right = self._camera_xyz(result.get("grasp_right", {}))
        midpoint = self._camera_xyz(result.get("grasp_midpoint", {}))
        if left is None or right is None or midpoint is None:
            raise ValueError("notebook grasp endpoints/midpoint lack valid camera XYZ")
        points = transform_points_camera_to_base(
            [left, right, midpoint],
            transform_base_gripper,
            self.transform_gripper_camera,
        )
        if detection.part_key == "gear_compound":
            # The two physical contacts define an undirected line: swapping
            # "left" and "right" is the same grasp and must not reset
            # consensus.  Keep a continuous sign relative to the first valid
            # axis, and use a fixed half-length so consensus measures only the
            # midpoint and direction rather than noisy endpoint labelling or
            # separation.
            axis_xy = np.asarray(points[1, :2] - points[0, :2], dtype=np.float64)
            axis_norm = float(np.linalg.norm(axis_xy))
            if axis_norm <= 1e-6:
                raise ValueError("gear_compound grasp direction is degenerate")
            axis_xy /= axis_norm
            if self._compound_axis_reference is None:
                self._compound_axis_reference = axis_xy.copy()
            elif float(np.dot(axis_xy, self._compound_axis_reference)) < 0.0:
                axis_xy = -axis_xy
            self._compound_axis_reference = axis_xy.copy()
            midpoint_base = np.asarray(points[2], dtype=np.float64)
            half_axis = np.array(
                [0.04 * axis_xy[0], 0.04 * axis_xy[1], 0.0],
                dtype=np.float64,
            )
            points = np.vstack(
                (
                    midpoint_base - half_axis,
                    midpoint_base + half_axis,
                    midpoint_base,
                )
            )
        quaternion = planar_object_quaternion(points[0], points[1], "x_parallel")
        return points, quaternion

    def _publish_debug(
        self, detection: LocalDetection | None, frame: Any, gate_text: str
    ) -> None:
        if not self.publish_debug_images:
            return
        images: dict[str, np.ndarray] = {}
        if detection is not None:
            images.update(detection.debug_images)
            images["annotated_result"] = detection.annotated
        gate = (
            detection.annotated.copy()
            if detection is not None
            else frame.color_bgr.copy()
        )
        cv2.putText(
            gate,
            gate_text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        images["publication_gate"] = gate
        for name, image in images.items():
            if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
                continue
            safe_name = str(name).strip("/").replace(" ", "_")
            publisher = self._debug_publishers.get(safe_name)
            if publisher is None:
                publisher = self.create_publisher(
                    Image, f"{self.topic_prefix}/debug/{safe_name}", 1
                )
                self._debug_publishers[safe_name] = publisher
            encoding = "mono8" if image.ndim == 2 else "bgr8"
            message = self._bridge.cv2_to_imgmsg(image, encoding=encoding)
            message.header.stamp.sec = int(frame.stamp_sec)
            message.header.stamp.nanosec = int(frame.stamp_nanosec)
            message.header.frame_id = self.camera_frame
            publisher.publish(message)

    def _process_frame(self) -> None:
        if not self.camera_parameters.ready:
            return
        frame = self.camera.capture()
        if frame is None:
            return
        detection_enabled = bool(self.get_parameter("detection_enabled").value)
        if not detection_enabled:
            if self._detection_was_enabled:
                self.get_logger().info(
                    "Local detection disabled; consensus and active request cleared"
                )
            self._detection_was_enabled = False
            self._enabled_after_ros_ns = None
            self.consensus.clear()
            self._unstable_frame_count = 0
            self._active_request = None
            self._publish_debug(
                None,
                frame,
                "BLOCKED: detection disabled until verified Global arrival",
            )
            return

        frame_stamp_ns = (
            int(frame.stamp_sec) * 1_000_000_000 + int(frame.stamp_nanosec)
        )
        if not self._detection_was_enabled:
            # The frame returned in the same timer tick may have been queued
            # before the parameter service enabled detection. Discard it and
            # require a strictly newer RGB-D acquisition timestamp.
            self._detection_was_enabled = True
            self._enabled_after_ros_ns = self.get_clock().now().nanoseconds
            self.consensus.clear()
            self._unstable_frame_count = 0
            self._active_request = None
            self.get_logger().info(
                "Local detection enabled after verified Global arrival; "
                "discarding pre-enable RGB-D frames"
            )
            self._publish_debug(
                None,
                frame,
                "BLOCKED: discarding pre-enable RGB-D frame",
            )
            return
        if (
            self._enabled_after_ros_ns is None
            or frame_stamp_ns <= self._enabled_after_ros_ns
        ):
            self._publish_debug(
                None,
                frame,
                "BLOCKED: waiting for post-enable RGB-D frame",
            )
            return
        try:
            part_key, base_hole_target = self._current_request()
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        request = (part_key, base_hole_target)
        if request != self._active_request:
            self.consensus.clear()
            self._compound_axis_reference = None
            self._unstable_frame_count = 0
            self._active_request = request
            self.get_logger().info(f"Local target switched to {request}")

        # Single-frame notebook detection precedes motion, consensus and TF gates.
        detection = self.detector.detect(
            frame.color_bgr,
            frame.depth_m,
            frame.intrinsics,
            part_key,
            base_hole_target,
        )
        if not frame.is_stable:
            self._unstable_frame_count += 1
            hard_motion = (
                frame.motion_score >= self.motion_reset_threshold
                or self._unstable_frame_count >= self.motion_reset_frames
            )
            if hard_motion:
                self.consensus.clear()
                gate_text = "BLOCKED: sustained/large motion; consensus reset"
            else:
                gate_text = "BLOCKED: mild image motion; consensus retained"
            self._publish_debug(detection, frame, gate_text)
            return
        self._unstable_frame_count = 0
        if detection is None:
            failed_debug = self.detector.failed_debug_detection(
                part_key, base_hole_target
            )
            self._publish_debug(
                failed_debug,
                frame,
                "BLOCKED: notebook detection failed; consensus retained",
            )
            return
        if detection.confidence == "low" and not self.publish_low_confidence:
            self._publish_debug(
                detection,
                frame,
                "BLOCKED: low confidence; consensus retained",
            )
            return
        transform_base_gripper = self.tf_provider.lookup_base_gripper(
            self.tf_timeout_sec, frame.stamp_sec, frame.stamp_nanosec
        )
        if transform_base_gripper is None:
            self.consensus.clear()
            self._publish_debug(detection, frame, "BLOCKED: image-time TF unavailable")
            return
        try:
            points, quaternion = self._base_geometry(
                detection, transform_base_gripper, base_hole_target
            )
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            self._publish_debug(
                detection,
                frame,
                "BLOCKED: invalid depth; consensus retained",
            )
            return
        consensus = self.consensus.add(detection.output_name, points)
        if consensus is None:
            self._publish_debug(detection, frame, "BLOCKED: collecting consensus")
            return
        position = (
            consensus[0]
            if part_key in {"base", "screw"}
            else consensus[2]
        )
        if part_key not in {"base", "screw"}:
            quaternion = planar_object_quaternion(consensus[0], consensus[1], "x_parallel")
        quaternion = quaternion / np.linalg.norm(quaternion)
        message = PoseStamped()
        message.header.stamp.sec = int(frame.stamp_sec)
        message.header.stamp.nanosec = int(frame.stamp_nanosec)
        message.header.frame_id = self.base_frame
        message.pose.position.x, message.pose.position.y, message.pose.position.z = map(
            float, position
        )
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = map(float, quaternion)
        self._pose_publishers[detection.output_name].publish(message)
        self._publish_debug(
            detection,
            frame,
            f"PUBLISHED: {detection.output_name} source={detection.source}",
        )


def main(args: list[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError("rclpy and geometry_msgs are required")
    rclpy.init(args=args)
    node = LocalVisionForceAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
