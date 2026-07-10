#!/usr/bin/env python3
"""ROS2 node using the extracted local_rgbd_grasp_refinement notebook core."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
JIARUI_DIR = THIS_DIR.parent
REPO_ROOT = JIARUI_DIR.parent
for path in (THIS_DIR, JIARUI_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import local_rgbd_grasp_refiner_core as local_core
from camera_profiles import get_camera_profile
from local_vision_force_adapter import LocalDetection, LocalVisionForceAdapter, PoseStamped
from robust_pose_filters import LocalRobustPoseBuffer, StabilityGate
from vision_common import (
    BASE_HOLE_NAMES,
    downward_grasp_quaternion,
    transform_points_camera_to_base,
)


class NotebookLocalRefiner:
    """Adapter from notebook local-refinement results to the ROS adapter interface."""

    def __init__(self) -> None:
        self.last_result: dict[str, Any] | None = None
        self.last_debug_images: dict[str, np.ndarray] = {}
        self.last_annotated: np.ndarray | None = None

    @staticmethod
    def _point_uv(result: dict[str, Any], name: str) -> tuple[float, float] | None:
        point = (result.get("pixel_points") or {}).get(name) or {}
        if point.get("u") is None or point.get("v") is None:
            return None
        return float(point["u"]), float(point["v"])

    @staticmethod
    def _point_depth(result: dict[str, Any], name: str) -> float | None:
        point = (result.get("camera_points") or {}).get(name) or {}
        depth = point.get("depth_m")
        if depth is None:
            return None
        depth_value = float(depth)
        return depth_value if np.isfinite(depth_value) and depth_value > 0.0 else None

    @staticmethod
    def _point_xyz(result: dict[str, Any], name: str) -> np.ndarray | None:
        point = (result.get("camera_points") or {}).get(name) or {}
        xyz = [
            point.get("camera_x_m"),
            point.get("camera_y_m"),
            point.get("camera_z_m"),
        ]
        if any(value is None for value in xyz):
            return None
        values = np.asarray(xyz, dtype=np.float64)
        return values if np.all(np.isfinite(values)) else None

    def detect(
        self,
        image: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: dict[str, float],
        part_key: str,
        base_hole_target: str = "hole_screw",
    ) -> LocalDetection | None:
        result, annotated, debug_images = local_core.run_local_refinement(
            image,
            depth_m,
            intrinsics,
            part_key,
        )
        self.last_result = result
        self.last_annotated = annotated if isinstance(annotated, np.ndarray) else None
        self.last_debug_images = {
            key: value for key, value in debug_images.items() if isinstance(value, np.ndarray)
        }

        if result.get("status") not in (None, "ok", "success"):
            return None
        if part_key == "base":
            if base_hole_target not in BASE_HOLE_NAMES:
                raise ValueError(
                    f"base_hole_target must be one of: {', '.join(BASE_HOLE_NAMES)}"
                )
            left_uv = self._point_uv(result, "hole_large")
            right_uv = self._point_uv(result, "hole_small")
            midpoint_uv = self._point_uv(result, base_hole_target)
            point_names = {
                "left": "hole_large",
                "right": "hole_small",
                "midpoint": base_hole_target,
            }
        else:
            left_uv = self._point_uv(result, "grasp_left")
            right_uv = self._point_uv(result, "grasp_right")
            midpoint_uv = self._point_uv(result, "grasp_midpoint")
            point_names = {
                "left": "grasp_left",
                "right": "grasp_right",
                "midpoint": "grasp_midpoint",
            }
        if left_uv is None or right_uv is None or midpoint_uv is None:
            return None

        depth_overrides = {}
        camera_points_m = {}
        for role, point_name in point_names.items():
            depth = self._point_depth(result, point_name)
            if depth is not None:
                depth_overrides[role] = depth
            xyz = self._point_xyz(result, point_name)
            if xyz is not None:
                camera_points_m[role] = xyz

        detection = LocalDetection(
            part_key=part_key,
            grasp_left_uv=left_uv,
            grasp_right_uv=right_uv,
            grasp_midpoint_uv=midpoint_uv,
            confidence=str(result.get("confidence", "medium")),
            source=str(result.get("yaw_source") or result.get("rule_note") or "notebook_core"),
            debug_images=self.last_debug_images,
            depth_overrides_m=depth_overrides,
        )
        detection.camera_points_m = camera_points_m
        detection.core_result = result
        return detection

    def annotate(self, image: np.ndarray, detection: LocalDetection | None) -> np.ndarray:
        if self.last_annotated is not None:
            return self.last_annotated.copy()
        return image.copy()


class LocalRGBDGraspRefinementNode(LocalVisionForceAdapter):
    """Local ROS node with notebook-extracted RGB-D grasp-refinement logic."""

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("wait_after_motion_sec", 0.5)
        self.declare_parameter("min_stable_frames", 8)
        self.declare_parameter("image_motion_threshold_px", 1.5)
        self.declare_parameter("tf_motion_threshold_m", 0.002)
        self.declare_parameter("tf_rotation_threshold_deg", 1.0)
        self.declare_parameter("max_point_jump_m", 0.015)
        self.declare_parameter("max_midpoint_jump_m", 0.012)
        self.declare_parameter("max_yaw_jump_deg", 15.0)
        self.declare_parameter("outlier_window_size", 7)
        self.declare_parameter("min_inlier_count", 4)
        self.declare_parameter("max_missed_frames", 5)
        self.declare_parameter("allow_low_confidence_for_debug", True)
        self.declare_parameter("low_confidence_requires_extra_frames", True)
        self.declare_parameter("low_confidence_consensus_frames", 8)
        self.detector = NotebookLocalRefiner()
        self.camera_profile = get_camera_profile("local_rgbd")
        self.stability_gate = StabilityGate(
            int(self.get_parameter("min_stable_frames").value),
            float(self.get_parameter("image_motion_threshold_px").value),
            float(self.get_parameter("tf_motion_threshold_m").value),
            float(self.get_parameter("tf_rotation_threshold_deg").value),
            float(self.get_parameter("wait_after_motion_sec").value),
        )
        self.robust_buffer = LocalRobustPoseBuffer(
            int(self.get_parameter("outlier_window_size").value),
            int(self.get_parameter("min_inlier_count").value),
            int(self.get_parameter("consensus_frames").value),
            int(self.get_parameter("low_confidence_consensus_frames").value),
            float(self.get_parameter("max_point_jump_m").value),
            float(self.get_parameter("max_midpoint_jump_m").value),
            float(self.get_parameter("max_yaw_jump_deg").value),
        )
        self.max_missed_frames = int(self.get_parameter("max_missed_frames").value)
        self.allow_low_confidence_for_debug = bool(
            self.get_parameter("allow_low_confidence_for_debug").value
        )
        self.low_confidence_requires_extra_frames = bool(
            self.get_parameter("low_confidence_requires_extra_frames").value
        )
        self.low_confidence_consensus_frames = int(
            self.get_parameter("low_confidence_consensus_frames").value
        )
        self.missed_count = 0
        self._last_status_log = ""
        self.get_logger().info(
            "Using extracted local_rgbd_grasp_refinement notebook core; "
            f"camera_profile=local_rgbd, rgb_settings={self.camera_profile['rgb']}, "
            f"depth_settings={self.camera_profile['depth']}"
        )

    def _log_state_once(self, message: str, level: str = "info") -> None:
        if message == self._last_status_log:
            return
        self._last_status_log = message
        logger = self.get_logger()
        getattr(logger, level)(message)

    def _clear_motion_state(self, reason: str) -> None:
        self.robust_buffer.clear()
        self.consensus.clear(self.output_name)
        self.missed_count = 0
        self._log_state_once(reason)

    def _process_frame(self) -> None:
        frame = self.camera.capture()
        if frame is None:
            return
        transform_base_gripper = self.tf_provider.lookup_base_gripper(
            self.tf_timeout_sec,
            frame.stamp_sec,
            frame.stamp_nanosec,
        )
        if not self.stability_gate.update(frame.motion_score, transform_base_gripper):
            reason = self.stability_gate.last_reason
            if "unstable" in reason or "unavailable" in reason:
                self._clear_motion_state(reason)
            else:
                self._log_state_once(reason)
            return

        try:
            detection = self.detector.detect(
                frame.color_bgr,
                frame.depth_m,
                frame.intrinsics,
                self.target_key,
                self.base_hole_target,
            )
        except Exception as exc:
            self.get_logger().warning(f"Local notebook-core detection failed: {exc}")
            self.missed_count += 1
            if self.missed_count >= self.max_missed_frames:
                self._clear_motion_state("detection failed too many times; clearing buffer")
            else:
                self._log_state_once(
                    f"detection failed but keeping previous buffer: {self.missed_count}/"
                    f"{self.max_missed_frames}",
                    "warning",
                )
            return
        if detection is None:
            self.missed_count += 1
            if self.missed_count >= self.max_missed_frames:
                self._clear_motion_state("detection failed too many times; clearing buffer")
            else:
                self._log_state_once(
                    f"detection failed but keeping previous buffer: {self.missed_count}/"
                    f"{self.max_missed_frames}",
                    "warning",
                )
            self._show_debug(frame.color_bgr, None)
            return
        self.missed_count = 0
        confidence = str(detection.confidence).lower()
        if confidence == "low" and not self.publish_low_confidence:
            self.get_logger().warning(
                f"{self.output_name}: rejected low-confidence fallback "
                f"source={detection.source}"
            )
            if self.allow_low_confidence_for_debug:
                self._show_debug(frame.color_bgr, detection)
            return

        pixels = {
            "left": detection.grasp_left_uv,
            "right": detection.grasp_right_uv,
            "midpoint": detection.grasp_midpoint_uv,
        }
        core_camera_points = getattr(detection, "camera_points_m", {}) or {}
        camera_points = {}
        for name, pixel in pixels.items():
            point = core_camera_points.get(name)
            if point is not None and np.all(np.isfinite(point)):
                camera_points[name] = np.asarray(point, dtype=np.float64)
            else:
                camera_points[name] = self._camera_point(
                    name, pixel, detection, frame.depth_m, frame.intrinsics
                )

        invalid = [name for name, point in camera_points.items() if point is None]
        if invalid:
            self.get_logger().warning(
                f"{self.output_name}: invalid camera XYZ for {invalid}; "
                "local pose not published"
            )
            self._show_debug(frame.color_bgr, detection)
            return
        ordered_camera = np.asarray(
            [camera_points["left"], camera_points["right"], camera_points["midpoint"]]
        )
        ordered_base = transform_points_camera_to_base(
            ordered_camera,
            transform_base_gripper,
            self.transform_gripper_camera,
        )

        reason = self.robust_buffer.reject_reason(ordered_base)
        if reason is not None:
            self.get_logger().warning(f"rejected outlier: reason={reason}")
            self._show_debug(frame.color_bgr, detection)
            return

        required_frames = (
            self.low_confidence_consensus_frames
            if confidence == "low" and self.low_confidence_requires_extra_frames
            else int(self.get_parameter("consensus_frames").value)
        )
        consensus_base = self.robust_buffer.add(
            ordered_base,
            confidence,
            required_frames=required_frames,
        )
        if consensus_base is None:
            self._log_state_once(
                f"collecting consensus for {self.output_name}: "
                f"need {required_frames} valid frames"
            )
            return
        left_base, right_base, midpoint_base = consensus_base
        try:
            quaternion = downward_grasp_quaternion(
                left_base, right_base, self.gripper_axis_mode
            )
        except ValueError as exc:
            self.get_logger().warning(f"Cannot compute local orientation: {exc}")
            return

        message = PoseStamped()
        message.header.stamp.sec = int(frame.stamp_sec)
        message.header.stamp.nanosec = int(frame.stamp_nanosec)
        message.header.frame_id = self.base_frame
        message.pose.position.x = float(midpoint_base[0])
        message.pose.position.y = float(midpoint_base[1])
        message.pose.position.z = float(midpoint_base[2])
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = map(float, quaternion)
        self.publisher.publish(message)
        self.get_logger().info(
            f"published robust pose for {self.output_name}: "
            f"left_pixel={pixels['left']}, right_pixel={pixels['right']}, "
            f"midpoint_pixel={pixels['midpoint']}, "
            f"camera XYZ left/right/mid={ordered_camera.tolist()}, "
            f"{self.base_frame} XYZ midpoint={midpoint_base.tolist()}, "
            f"source={detection.source}, published {self.output_topic}"
        )
        self._show_debug(frame.color_bgr, detection)


def main(args: list[str] | None = None) -> None:
    import rclpy

    rclpy.init(args=args)
    node = None
    try:
        node = LocalRGBDGraspRefinementNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        rclpy.logging.get_logger("local_rgbd_grasp_refinement_node").fatal(str(exc))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
