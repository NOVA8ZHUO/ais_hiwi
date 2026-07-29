#!/usr/bin/env python3
"""ROS2 wrapper around the exact global notebook detector."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import global_notebook_detector as notebook
from global_table_intersection_utils import pixel_to_table_intersection_camera_and_base
from vision_common import (
    BASE_HOLE_NAMES,
    DEFAULT_HANDEYE_PATH,
    GLOBAL_REALSENSE_ROS_PARAMETERS,
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


GLOBAL_TOPIC_PREFIX = "/vision/global"

_CLASS_TO_KEY = {
    "Base": "base",
    "Shaft_small": "shaft_small",
    "Shaft_large": "shaft_large",
    "Screw": "screw",
    "Gear_small": "gear_small",
    "Gear_compound": "gear_compound",
    "Gear_large": "gear_large",
}


@dataclass(frozen=True)
class GlobalDetection:
    """Lossless ROS-facing view of one notebook result dictionary."""

    part_key: str
    center_uv: tuple[float, float]
    confidence: str
    raw_result: Mapping[str, Any]

    @property
    def contour(self) -> np.ndarray | None:
        value = self.raw_result.get("_contour")
        return value if isinstance(value, np.ndarray) else None


def _result_key(result: Mapping[str, Any]) -> str | None:
    hole_name = result.get("hole_name")
    if hole_name in BASE_HOLE_NAMES:
        return str(hole_name)
    return _CLASS_TO_KEY.get(str(result.get("class_name")))


def _output_name(part_key: str) -> str:
    return part_key if part_key in BASE_HOLE_NAMES else force_control_part_name(part_key)


class GlobalPartDetector:
    """Pure detector; every mask, threshold and selection rule is notebook code."""

    def detect(
        self, color_bgr: np.ndarray
    ) -> tuple[list[GlobalDetection], dict[str, np.ndarray]]:
        filtered_color, _, workspace_mask = apply_black_background_workspace(
            color_bgr
        )
        annotated, raw_results, debug = notebook.detect_live_frame_bgr(
            filtered_color
        )
        # Display the real exterior for operator context, but erase any drawing
        # outside the accepted workspace and show its boundary explicitly.
        annotated[workspace_mask == 0] = color_bgr[workspace_mask == 0]
        workspace_contours, _ = cv2.findContours(
            workspace_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, workspace_contours, -1, (0, 255, 0), 2)
        detections: list[GlobalDetection] = []
        for result in raw_results:
            key = _result_key(result)
            if key is None:
                continue
            detections.append(
                GlobalDetection(
                    part_key=key,
                    center_uv=(float(result["center_u"]), float(result["center_v"])),
                    confidence=str(result.get("confidence", "low")),
                    raw_result=result,
                )
            )
        debug = dict(debug)
        debug["black_background_workspace_mask"] = workspace_mask
        debug["black_background_masked_input"] = filtered_color
        debug["annotated"] = annotated
        return detections, debug

    @staticmethod
    def annotate(
        image: np.ndarray, detections: Sequence[GlobalDetection]
    ) -> np.ndarray:
        return notebook.draw_results(image, [item.raw_result for item in detections])


class GlobalVisionForceAdapter(Node):
    """Publish notebook global detections using image-time Franka TF."""

    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("ROS2 Python packages are required to run this node")
        super().__init__("global_vision_force_adapter")
        defaults = {
            "target_part": "",
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
            "topic_prefix": GLOBAL_TOPIC_PREFIX,
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
            "consensus_tolerance_m": 0.015,
            "publish_debug_images": True,
            "publish_low_confidence": False,
            "publish_rate": 10.0,
            "tf_timeout_sec": 0.2,
            "table_z_m": 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.topic_prefix = str(self.get_parameter("topic_prefix").value).rstrip("/")
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.table_z_m = float(self.get_parameter("table_z_m").value)
        self.publish_low_confidence = bool(
            self.get_parameter("publish_low_confidence").value
        )
        self.publish_debug_images = bool(
            self.get_parameter("publish_debug_images").value
        )
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
        target = str(self.get_parameter("target_part").value).strip()
        self.target_key = (
            target if target in BASE_HOLE_NAMES else normalize_part_key(target)
        ) if target else None
        handeye_path = str(self.get_parameter("handeye_path").value).strip()
        legacy_path = str(self.get_parameter("handeye_yaml_path").value).strip()
        self.transform_gripper_camera = load_handeye_transform(legacy_path or handeye_path)

        self.detector = GlobalPartDetector()
        self.tf_provider = TFTransformProvider(self, self.base_frame, self.ee_frame)
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
            parameters=GLOBAL_REALSENSE_ROS_PARAMETERS,
            profile_name="global",
        )
        rate = float(self.get_parameter("publish_rate").value)
        if rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        self.timer = self.create_timer(1.0 / rate, self._process_frame)
        self.get_logger().info(
            f"Notebook-exact global detector; TF {self.base_frame} <- {self.ee_frame}; "
            "T_base_camera=T_base_gripper@T_gripper_camera"
        )
        self.get_logger().info(
            "Outer publication gate: "
            f"motion_threshold={self.get_parameter('motion_threshold').value}, "
            f"stable_frames={self.get_parameter('stable_frames_required').value}, "
            f"hard_reset_score={self.motion_reset_threshold}, "
            f"hard_reset_frames={self.motion_reset_frames}; "
            "sparse valid detections retain consensus"
        )

    @staticmethod
    def _camera_matrix(intrinsics: Mapping[str, float]) -> np.ndarray:
        return np.array(
            [
                [intrinsics["fx"], 0.0, intrinsics["ppx"]],
                [0.0, intrinsics["fy"], intrinsics["ppy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _table_point(
        self,
        uv: Sequence[float],
        camera_matrix: np.ndarray,
        transform_base_gripper: np.ndarray,
    ) -> np.ndarray | None:
        result = pixel_to_table_intersection_camera_and_base(
            float(uv[0]),
            float(uv[1]),
            camera_matrix,
            transform_base_gripper,
            self.transform_gripper_camera,
            self.table_z_m,
        )
        value = result.get("center_base_on_table_m")
        return None if value is None else np.asarray(value, dtype=np.float64)

    def _orientation(
        self,
        detection: GlobalDetection,
        center_base: np.ndarray,
        camera_matrix: np.ndarray,
        transform_base_gripper: np.ndarray,
    ) -> np.ndarray:
        yaw_deg = detection.raw_result.get("yaw_deg")
        if yaw_deg is None:
            return quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)
        angle = math.radians(float(yaw_deg))
        direction_uv = np.asarray(detection.center_uv) + 10.0 * np.array(
            [math.cos(angle), math.sin(angle)]
        )
        direction_base = self._table_point(
            direction_uv, camera_matrix, transform_base_gripper
        )
        if direction_base is None:
            return quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)
        return planar_object_quaternion(center_base, direction_base, "x_parallel")

    def _publish_debug(
        self, debug: Mapping[str, np.ndarray], frame: Any, gate_text: str
    ) -> None:
        if not self.publish_debug_images:
            return
        images = dict(debug)
        gate = np.asarray(debug.get("annotated", frame.color_bgr)).copy()
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
        # Single-frame notebook detection always runs before any publication gate.
        detections, debug = self.detector.detect(frame.color_bgr)
        if self.target_key:
            detections = [item for item in detections if item.part_key == self.target_key]
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
            self._publish_debug(debug, frame, gate_text)
            return
        self._unstable_frame_count = 0
        transform_base_gripper = self.tf_provider.lookup_base_gripper(
            self.tf_timeout_sec, frame.stamp_sec, frame.stamp_nanosec
        )
        if transform_base_gripper is None:
            self.consensus.clear()
            self._publish_debug(debug, frame, "BLOCKED: image-time TF unavailable")
            return
        camera_matrix = self._camera_matrix(frame.intrinsics)
        published = 0
        for detection in detections:
            name = _output_name(detection.part_key)
            if detection.confidence == "low" and not self.publish_low_confidence:
                # A low-confidence or missing frame contributes nothing, but it
                # must not erase earlier valid notebook detections.  This keeps
                # the single-frame algorithm unchanged while allowing sparse
                # valid frames to form publication consensus over time.
                continue
            center_base = self._table_point(
                detection.center_uv, camera_matrix, transform_base_gripper
            )
            if center_base is None:
                # This frame cannot contribute a point, but an intermittent
                # projection failure is not evidence that prior points moved.
                continue
            consensus = self.consensus.add(name, center_base)
            if consensus is None:
                continue
            position = consensus[0]
            quaternion = self._orientation(
                detection, position, camera_matrix, transform_base_gripper
            )
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
            ) = map(float, quaternion / np.linalg.norm(quaternion))
            self._pose_publishers[name].publish(message)
            published += 1
        self._publish_debug(debug, frame, f"PUBLISHED: {published}")


def main(args: list[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError("rclpy and geometry_msgs are required")
    rclpy.init(args=args)
    node = GlobalVisionForceAdapter()
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
