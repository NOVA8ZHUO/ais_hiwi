#!/usr/bin/env python3
"""Publish all global part detections for robot-side consumption.

Coordinates are in ``camera_color_optical_frame``. This node does not select a
target and does not transform points into robot base frame. Target selection,
TF transformation, and robot motion are handled by robot-side code.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from global_rgb_detector_core import detect_global_parts_bgr, draw_global_results
from rgbd_coordinate_utils import (
    add_camera_xyz_to_global_result,
    add_invalid_camera_fields,
)

EXPECTED_INSTANCES = {
    "gear_large",
    "gear_small",
    "gear_compound",
    "screw",
    "shaft_large",
    "shaft_small",
    "base",
}
EXPECTED_BASE_HOLES = {"hole_l", "hole_m", "hole_r"}
DEBUG_MASK_TOPICS = {
    "work_area_mask": "/vision/global/debug/work_area_mask",
    "red_mask": "/vision/global/debug/red_mask",
    "black_mask": "/vision/global/debug/black_mask",
    "silver_base_mask": "/vision/global/debug/silver_base_mask",
    "base_surface_mask": "/vision/global/debug/base_surface_mask",
    "base_hole_search_mask": "/vision/global/debug/base_hole_search_mask",
}


class GlobalVisionNode(Node):
    """Subscribe to RealSense streams and publish annotated images plus JSON."""

    def __init__(self) -> None:
        super().__init__("global_vision_node")
        self.declare_parameter(
            "color_topic", "/camera/camera/color/image_raw"
        )
        self.declare_parameter(
            "aligned_depth_topic",
            "/camera/camera/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )
        self.declare_parameter(
            "annotated_topic", "/vision/global/annotated_image"
        )
        self.declare_parameter(
            "detections_json_topic", "/vision/global/detections_json"
        )
        self.declare_parameter(
            "camera_frame_id", "camera_color_optical_frame"
        )
        self.declare_parameter("depth_scale_for_integer_images", 0.001)
        self.declare_parameter("publish_debug_masks", False)

        color_topic = self.get_parameter("color_topic").value
        depth_topic = self.get_parameter("aligned_depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        annotated_topic = self.get_parameter("annotated_topic").value
        detections_topic = self.get_parameter("detections_json_topic").value
        self.publish_debug_masks = bool(
            self.get_parameter("publish_debug_masks").value
        )

        self.bridge = CvBridge()
        self.latest_depth_m: np.ndarray | None = None
        self.latest_camera_info: CameraInfo | None = None
        self._last_warning_time: dict[str, float] = {}

        # Latest-frame reuse is acceptable for this first stationary-camera
        # stage. Replace it with synchronized RGB-D messages before detecting
        # while an eye-in-hand camera is moving.
        self.create_subscription(
            Image, color_topic, self._color_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, depth_topic, self._depth_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.annotated_publisher = self.create_publisher(
            Image, annotated_topic, 10
        )
        self.json_publisher = self.create_publisher(
            String, detections_topic, 10
        )
        self.debug_publishers = {}
        if self.publish_debug_masks:
            self.debug_publishers = {
                name: self.create_publisher(Image, topic, 10)
                for name, topic in DEBUG_MASK_TOPICS.items()
            }
        self.get_logger().info(
            f"Global vision node listening on color={color_topic}, depth={depth_topic}"
        )

    def _depth_callback(self, message: Image) -> None:
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(
                message, desired_encoding="passthrough"
            )
            depth_array = np.asarray(depth_raw)
            if depth_array.ndim != 2:
                raise ValueError(
                    f"expected a 2D depth image, got shape {depth_array.shape}"
                )
            if np.issubdtype(depth_array.dtype, np.integer):
                scale = float(
                    self.get_parameter("depth_scale_for_integer_images").value
                )
                self.latest_depth_m = depth_array.astype(np.float32) * scale
            else:
                self.latest_depth_m = depth_array.astype(np.float32, copy=False)
        except (CvBridgeError, ValueError, TypeError) as exc:
            self._warn_throttled(
                "depth_conversion",
                f"Could not convert aligned depth image: {exc}",
            )

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self.latest_camera_info = message

    def _color_callback(self, message: Image) -> None:
        try:
            color_bgr = self.bridge.imgmsg_to_cv2(
                message, desired_encoding="bgr8"
            )
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not convert color image: {exc}")
            return

        result, masks = detect_global_parts_bgr(color_bgr, return_debug=True)
        result["timestamp_sec"] = _stamp_to_seconds(message)
        camera_frame_id = str(
            self.get_parameter("camera_frame_id").value
            or "camera_color_optical_frame"
        )

        if self.latest_depth_m is not None and self.latest_camera_info is not None:
            try:
                camera_info = {
                    "k": list(self.latest_camera_info.k),
                    "frame_id": camera_frame_id,
                }
                result = add_camera_xyz_to_global_result(
                    result,
                    self.latest_depth_m,
                    camera_info,
                    masks=masks,
                )
            except (ValueError, TypeError) as exc:
                result = add_invalid_camera_fields(
                    result,
                    camera_frame_id,
                    reason=f"RGB-D conversion unavailable: {exc}",
                )
                self._warn_throttled("rgbd_conversion", str(exc))
        else:
            missing = []
            if self.latest_depth_m is None:
                missing.append("aligned depth")
            if self.latest_camera_info is None:
                missing.append("camera_info")
            result = add_invalid_camera_fields(
                result,
                camera_frame_id,
                reason=f"waiting for {' and '.join(missing)}; publishing pixel-only result",
            )

        result = _stabilize_result_schema(result, camera_frame_id)
        self._log_runtime_warnings(result)
        annotated = draw_global_results(color_bgr, result)
        self._publish_annotated(annotated, message, camera_frame_id)
        if self.publish_debug_masks:
            self._publish_debug_images(masks, message, camera_frame_id)
        self._publish_json(result)

    def _publish_annotated(
        self, annotated_bgr: np.ndarray, source: Image, frame_id: str
    ) -> None:
        try:
            message = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding="bgr8")
            message.header = source.header
            if not message.header.frame_id:
                message.header.frame_id = frame_id
            self.annotated_publisher.publish(message)
        except CvBridgeError as exc:
            self.get_logger().error(f"Could not publish annotated image: {exc}")

    def _publish_debug_images(
        self,
        masks: dict[str, np.ndarray],
        source: Image,
        frame_id: str,
    ) -> None:
        for name, publisher in self.debug_publishers.items():
            mask = masks.get(name)
            if mask is None:
                continue
            try:
                message = self.bridge.cv2_to_imgmsg(
                    np.asarray(mask, dtype=np.uint8), encoding="mono8"
                )
                message.header = source.header
                if not message.header.frame_id:
                    message.header.frame_id = frame_id
                publisher.publish(message)
            except (CvBridgeError, TypeError, ValueError) as exc:
                self._warn_throttled(
                    f"debug_mask_{name}",
                    f"Could not publish debug mask {name}: {exc}",
                )

    def _publish_json(self, result: dict[str, Any]) -> None:
        try:
            message = String()
            message.data = json.dumps(
                result, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            self.json_publisher.publish(message)
        except (TypeError, ValueError) as exc:
            self.get_logger().error(f"Detection result is not valid JSON: {exc}")

    def _log_runtime_warnings(self, result: dict[str, Any]) -> None:
        missing_instances = result.get("missing_instances", [])
        missing_holes = result.get("missing_base_holes", [])
        invalid_depth = (
            result.get("num_detections", 0)
            + result.get("num_base_holes", 0)
            - result.get("num_valid_depth", 0)
        )
        if missing_instances or missing_holes:
            self._warn_throttled(
                "missing_detections",
                "Missing global detections: "
                f"parts={missing_instances}, base_holes={missing_holes}",
            )
        if invalid_depth:
            self._warn_throttled(
                "invalid_depth",
                f"{invalid_depth} published detection(s) have invalid depth",
            )

    def _warn_throttled(
        self, key: str, message: str, period_sec: float = 2.0
    ) -> None:
        now = time.monotonic()
        if now - self._last_warning_time.get(key, -period_sec) >= period_sec:
            self.get_logger().warning(message)
            self._last_warning_time[key] = now


def _stabilize_result_schema(
    result: dict[str, Any], camera_frame_id: str
) -> dict[str, Any]:
    """Guarantee fixed robot-facing fields and add a compact runtime summary."""
    detection_defaults = {
        "instance_name": None,
        "class_name": None,
        "center_u": None,
        "center_v": None,
        "bbox_x": None,
        "bbox_y": None,
        "bbox_w": None,
        "bbox_h": None,
        "area_px": None,
        "yaw_deg": None,
        "confidence": None,
        "depth_valid": False,
        "depth_m": None,
        "depth_valid_count": 0,
        "depth_source": None,
        "camera_x_m": None,
        "camera_y_m": None,
        "camera_z_m": None,
        "camera_frame_id": camera_frame_id,
    }
    hole_defaults = {
        "name": None,
        "center_u": None,
        "center_v": None,
        "radius_px": None,
        "confidence": None,
        "depth_valid": False,
        "depth_m": None,
        "depth_valid_count": 0,
        "depth_source": None,
        "camera_x_m": None,
        "camera_y_m": None,
        "camera_z_m": None,
        "camera_frame_id": camera_frame_id,
    }
    detections = result.setdefault("detections", [])
    base_holes = result.setdefault("base_holes", {})
    for detection in detections:
        for key, value in detection_defaults.items():
            detection.setdefault(key, value)
        if not detection["depth_valid"]:
            _clear_invalid_depth(detection)
    for name, hole in base_holes.items():
        hole.setdefault("name", name)
        for key, value in hole_defaults.items():
            hole.setdefault(key, value)
        if not hole["depth_valid"]:
            _clear_invalid_depth(hole)

    found_instances = {
        item.get("instance_name") for item in detections if item.get("instance_name")
    }
    found_holes = set(base_holes)
    result["frame_id"] = camera_frame_id
    result["num_detections"] = int(len(detections))
    result["num_base_holes"] = int(len(base_holes))
    result["num_valid_depth"] = int(
        sum(bool(item.get("depth_valid")) for item in detections)
        + sum(bool(item.get("depth_valid")) for item in base_holes.values())
    )
    result["missing_instances"] = sorted(EXPECTED_INSTANCES - found_instances)
    result["missing_base_holes"] = sorted(EXPECTED_BASE_HOLES - found_holes)
    return result


def _clear_invalid_depth(item: dict[str, Any]) -> None:
    item["depth_valid"] = False
    item["depth_m"] = None
    item["camera_x_m"] = None
    item["camera_y_m"] = None
    item["camera_z_m"] = None


def _stamp_to_seconds(message: Image) -> float:
    return float(message.header.stamp.sec) + float(
        message.header.stamp.nanosec
    ) * 1e-9


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GlobalVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
