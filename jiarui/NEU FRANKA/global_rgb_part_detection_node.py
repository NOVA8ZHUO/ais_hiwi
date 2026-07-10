#!/usr/bin/env python3
"""ROS2 node using the extracted global_rgb_part_detection notebook core.

The 2D detections are produced by ``global_core.detect_live_frame_bgr``. The
published 3D poses intentionally keep the existing global coarse-localization
behavior from ``GlobalVisionForceAdapter``: each pixel ray is intersected with
the configured robot-base table plane. This is suitable for high-level coarse
hover/localization. It is not a depth-median reproduction of any notebook
camera-point output. A depth-aware global wrapper should be added if the robot
workflow needs the notebook's depth-derived 3D semantics instead.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
JIARUI_DIR = THIS_DIR.parent
REPO_ROOT = JIARUI_DIR.parent
for path in (THIS_DIR, JIARUI_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import global_rgb_part_detector_core as global_core
from camera_profiles import get_camera_profile
from global_table_intersection_utils import pixel_to_table_intersection_camera_and_base
from global_vision_force_adapter import GlobalVisionForceAdapter, PoseStamped, _output_name
from robust_pose_filters import GlobalRobustPoseBuffer, StabilityGate
from vision_common import BASE_HOLE_NAMES, quaternion_from_roll_pitch_yaw


CLASS_TO_PART_KEY = {
    "Gear_large": "gear_large",
    "Gear_small": "gear_small",
    "Gear_compound": "gear_compound",
    "Screw": "screw",
    "Shaft_large": "shaft_large",
    "Shaft_small": "shaft_small",
    "Base": "base",
}


class CoreGlobalDetection:
    def __init__(
        self,
        part_key: str,
        center_uv: tuple[float, float],
        contour: np.ndarray,
        confidence: str,
        hole_radius_px: float | None = None,
    ) -> None:
        self.part_key = part_key
        self.center_uv = center_uv
        self.contour = contour
        self.confidence = confidence
        self.hole_radius_px = hole_radius_px


def _part_key_from_core_result(result: dict[str, Any]) -> str | None:
    hole_name = result.get("hole_name")
    if isinstance(hole_name, str) and hole_name.startswith("hole_"):
        return hole_name
    class_name = str(result.get("class_name", ""))
    return CLASS_TO_PART_KEY.get(class_name)


class NotebookGlobalDetector:
    """Adapter from notebook result dictionaries to the ROS adapter interface."""

    def __init__(self) -> None:
        self.last_debug_views: dict[str, np.ndarray] = {}
        self.last_annotated: np.ndarray | None = None

    def detect(self, image: np.ndarray) -> tuple[list[CoreGlobalDetection], dict[str, np.ndarray]]:
        annotated, results, debug_views = global_core.detect_live_frame_bgr(image)
        self.last_debug_views = {
            key: value for key, value in debug_views.items() if isinstance(value, np.ndarray)
        }
        self.last_annotated = annotated if isinstance(annotated, np.ndarray) else None

        detections: list[CoreGlobalDetection] = []
        for result in results:
            part_key = _part_key_from_core_result(result)
            contour = result.get("_contour")
            if part_key is None or not isinstance(contour, np.ndarray):
                continue
            detections.append(
                CoreGlobalDetection(
                    part_key=part_key,
                    center_uv=(
                        float(result["center_u"]),
                        float(result["center_v"]),
                    ),
                    contour=contour,
                    confidence=str(result.get("confidence", "medium")),
                    hole_radius_px=(
                        float(result["hole_radius_px"])
                        if result.get("hole_radius_px") is not None
                        else None
                    ),
                )
            )
        return detections, self.last_debug_views

    @staticmethod
    def annotate(image: np.ndarray, detections: list[CoreGlobalDetection]) -> np.ndarray:
        del detections
        annotated, _, _ = global_core.detect_live_frame_bgr(image)
        return annotated


class GlobalRGBPartDetectionNode(GlobalVisionForceAdapter):
    """Global ROS node with notebook-extracted detection logic."""

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("wait_after_motion_sec", 0.5)
        self.declare_parameter("min_stable_frames", 8)
        self.declare_parameter("image_motion_threshold_px", 1.5)
        self.declare_parameter("tf_motion_threshold_m", 0.002)
        self.declare_parameter("tf_rotation_threshold_deg", 1.0)
        self.declare_parameter("max_position_jump_m", 0.015)
        self.declare_parameter("outlier_window_size", 7)
        self.declare_parameter("min_inlier_count", 4)
        self.declare_parameter("per_part_consensus_frames", 5)
        self.declare_parameter("max_missed_frames", 5)
        self.detector = NotebookGlobalDetector()
        self.camera_profile = get_camera_profile("global_rgb")
        self.stability_gate = StabilityGate(
            int(self.get_parameter("min_stable_frames").value),
            float(self.get_parameter("image_motion_threshold_px").value),
            float(self.get_parameter("tf_motion_threshold_m").value),
            float(self.get_parameter("tf_rotation_threshold_deg").value),
            float(self.get_parameter("wait_after_motion_sec").value),
        )
        self.robust_buffer = GlobalRobustPoseBuffer(
            int(self.get_parameter("outlier_window_size").value),
            int(self.get_parameter("min_inlier_count").value),
            int(self.get_parameter("per_part_consensus_frames").value),
            float(self.get_parameter("max_position_jump_m").value),
        )
        self.max_missed_frames = int(self.get_parameter("max_missed_frames").value)
        self.missed_counts = {
            output_name: 0 for output_name in list(self._pose_publishers.keys())
        }
        for name in BASE_HOLE_NAMES:
            self.missed_counts.setdefault(name, 0)
        self._last_status_log = ""
        self.get_logger().info(
            "Using extracted global_rgb_part_detection notebook core; "
            f"camera_profile=global_rgb, rgb_settings={self.camera_profile['rgb']}"
        )

    def _log_state_once(self, message: str, level: str = "info") -> None:
        if message == self._last_status_log:
            return
        self._last_status_log = message
        getattr(self.get_logger(), level)(message)

    def _clear_motion_state(self, reason: str) -> None:
        self.robust_buffer.clear()
        self.consensus.clear()
        for key in self.missed_counts:
            self.missed_counts[key] = 0
        self._log_state_once(reason)

    def _note_missed(self, output_name: str) -> None:
        self.missed_counts[output_name] = self.missed_counts.get(output_name, 0) + 1
        missed = self.missed_counts[output_name]
        if missed >= self.max_missed_frames:
            self.robust_buffer.clear(output_name)
            self.consensus.clear(output_name)
            self._log_state_once(
                f"{output_name}: detection failed too many times; clearing buffer",
                "warning",
            )
        else:
            self._log_state_once(
                f"{output_name}: detection failed but keeping previous buffer "
                f"{missed}/{self.max_missed_frames}",
                "warning",
            )

    def _active_output_names(self) -> list[str]:
        if self.target_key:
            return [_output_name(self.target_key)]
        return list(self._pose_publishers.keys())

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
            detections, _ = self.detector.detect(frame.color_bgr)
        except Exception as exc:
            self.get_logger().warning(f"Global notebook-core detection failed: {exc}")
            for output_name in self._active_output_names():
                self._note_missed(output_name)
            return
        if self.target_key:
            detections = [item for item in detections if item.part_key == self.target_key]
        if not self.publish_low_confidence:
            detections = [item for item in detections if item.confidence != "low"]
        if not detections:
            for output_name in self._active_output_names():
                self._note_missed(output_name)
            self._show_debug(frame.color_bgr, [])
            return

        camera_matrix = np.array(
            [
                [float(frame.intrinsics["fx"]), 0.0, float(frame.intrinsics["ppx"])],
                [0.0, float(frame.intrinsics["fy"]), float(frame.intrinsics["ppy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        detected_names = {_output_name(item.part_key) for item in detections}
        for output_name in self._active_output_names():
            if output_name not in detected_names:
                self._note_missed(output_name)

        for detection in detections:
            part_name = _output_name(detection.part_key)
            self.missed_counts[part_name] = 0
            u, v = detection.center_uv
            table_result = pixel_to_table_intersection_camera_and_base(
                u,
                v,
                camera_matrix,
                transform_base_gripper,
                self.transform_gripper_camera,
                self.table_z_m,
            )
            if not table_result["ray_table_intersection_valid"]:
                self._note_missed(part_name)
                self.get_logger().warning(
                    f"{part_name}: table intersection failed at pixel ({u:.1f}, {v:.1f}): "
                    f"{table_result['ray_table_intersection_reason']}"
                )
                continue
            base_point = np.asarray(table_result["center_base_on_table_m"], dtype=np.float64)
            reason = self.robust_buffer.reject_reason(part_name, base_point)
            if reason is not None:
                self.get_logger().warning(
                    f"{part_name}: rejected outlier: reason={reason}"
                )
                continue
            consensus_point = self.robust_buffer.add(part_name, base_point)
            if consensus_point is None:
                self._log_state_once(
                    f"{part_name}: collecting consensus; need "
                    f"{int(self.get_parameter('per_part_consensus_frames').value)} valid frames"
                )
                continue

            message = PoseStamped()
            message.header.stamp.sec = int(frame.stamp_sec)
            message.header.stamp.nanosec = int(frame.stamp_nanosec)
            message.header.frame_id = self.base_frame
            message.pose.position.x = float(consensus_point[0])
            message.pose.position.y = float(consensus_point[1])
            message.pose.position.z = float(consensus_point[2])
            quaternion = quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)
            (
                message.pose.orientation.x,
                message.pose.orientation.y,
                message.pose.orientation.z,
                message.pose.orientation.w,
            ) = map(float, quaternion)
            self._pose_publishers[part_name].publish(message)
            self.get_logger().info(
                f"published robust pose for {part_name}: pixel=({u:.1f},{v:.1f}), "
                f"{self.base_frame} XYZ={consensus_point.tolist()}, "
                f"source=table_intersection, topic={self.topic_prefix}/{part_name}/pose"
            )
        self._show_debug(frame.color_bgr, detections)


def main(args: list[str] | None = None) -> None:
    import rclpy

    rclpy.init(args=args)
    node = None
    try:
        node = GlobalRGBPartDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        rclpy.logging.get_logger("global_rgb_part_detection_node").fatal(str(exc))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
