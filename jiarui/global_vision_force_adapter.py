#!/usr/bin/env python3
"""ROS2 global RGB-D coarse localization adapter for force_control.

The rule-based detector is extracted from global_rgb_part_detection.ipynb. It
does no robot motion; it only publishes base-frame part centers.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_table_intersection_utils import pixel_to_table_intersection_camera_and_base

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
except ImportError:  # Keep the pure detector importable for offline tests.
    rclpy = None
    PoseStamped = Any

    class Node:  # type: ignore[no-redef]
        pass

from vision_common import (
    ROSAlignedRGBDCamera,
    BASE_HOLE_NAMES,
    DEFAULT_HANDEYE_PATH,
    PART_NAME_MAP,
    SpatialConsensusFilter,
    TFTransformProvider,
    assign_base_hole_names,
    force_control_part_name,
    load_handeye_transform,
    normalize_part_key,
    quaternion_from_roll_pitch_yaw,
)


GLOBAL_TOPIC_PREFIX = "/vision/global"


@dataclass
class GlobalDetection:
    part_key: str
    center_uv: tuple[float, float]
    contour: np.ndarray
    confidence: str
    hole_radius_px: float | None = None


def _output_name(key: str) -> str:
    return key if key in BASE_HOLE_NAMES else force_control_part_name(key)


class GlobalPartDetector:
    """Notebook-derived red/black/silver segmentation without ROS concerns."""

    def __init__(self, work_area_polygon: Sequence[int] | None = None) -> None:
        flat = list(work_area_polygon or [104, 0, 568, 0, 588, 330, 455, 480, 120, 480])
        if len(flat) < 6 or len(flat) % 2:
            raise ValueError("work_area_polygon must be a flat [u0,v0,u1,v1,...] list")
        self.work_area_polygon = np.asarray(flat, dtype=np.int32).reshape(-1, 2)

    @staticmethod
    def _kernel(size: int) -> np.ndarray:
        size = max(1, int(size))
        size += int(size % 2 == 0)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    @staticmethod
    def _center(contour: np.ndarray) -> tuple[float, float]:
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-8:
            return (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
        (u, v), _, _ = cv2.minAreaRect(contour)
        return float(u), float(v)

    @staticmethod
    def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == label] = 255
        return cleaned

    def _work_mask(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        polygon = self.work_area_polygon.copy()
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return cv2.erode(mask, self._kernel(7))

    def _red_mask(self, image: np.ndarray, work_mask: np.ndarray) -> np.ndarray:
        segmentation_image = cv2.bilateralFilter(image, 5, 35, 35)
        hsv = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 50, 40), (12, 255, 255)),
            cv2.inRange(hsv, (168, 50, 40), (179, 255, 255)),
        )
        blue, green, red_channel = cv2.split(segmentation_image)
        red_dominant = (
            (red_channel.astype(np.int16) >= green.astype(np.int16) + 8)
            & (red_channel.astype(np.int16) >= blue.astype(np.int16) + 8)
        )
        red[~red_dominant] = 0
        red = cv2.bitwise_and(red, work_mask)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, self._kernel(3))
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, self._kernel(7), iterations=2)
        red = cv2.dilate(red, self._kernel(3), iterations=1)
        red = cv2.bitwise_and(red, work_mask)
        return self._remove_small(red, 180)

    def _black_mask(self, image: np.ndarray, work_mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        black = cv2.inRange(hsv, (0, 0, 0), (179, 255, 82))
        black = cv2.bitwise_and(black, work_mask)
        black = cv2.morphologyEx(black, cv2.MORPH_OPEN, self._kernel(3))
        return cv2.morphologyEx(black, cv2.MORPH_CLOSE, self._kernel(5))

    def _silver_mask(
        self,
        image: np.ndarray,
        work_mask: np.ndarray,
        red_mask: np.ndarray,
        black_mask: np.ndarray,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, saturation, value = cv2.split(hsv)
        lightness, a_channel, b_channel = cv2.split(lab)
        silver = (
            (np.abs(a_channel.astype(np.int16) - 128) <= 35)
            & (np.abs(b_channel.astype(np.int16) - 128) <= 45)
            & (saturation <= 95)
            & (value >= 85)
            & (value <= 245)
            & (lightness >= 95)
            & (lightness <= 235)
        )
        mask = cv2.bitwise_and(np.uint8(silver) * 255, work_mask)
        exclusion = cv2.dilate(cv2.bitwise_or(red_mask, black_mask), self._kernel(7))
        mask[exclusion > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel(3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel(11))
        return self._remove_small(mask, 250)

    def _white_shaft_mask(self, image: np.ndarray, work_mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, saturation, value = cv2.split(hsv)
        lightness, a_channel, b_channel = cv2.split(lab)
        neutral = (
            (np.abs(a_channel.astype(np.int16) - 128) <= 28)
            & (np.abs(b_channel.astype(np.int16) - 128) <= 35)
        )
        white = (
            neutral
            & (saturation <= 85)
            & (value >= 118)
            & (lightness >= 118)
        )
        mask = cv2.bitwise_and(np.uint8(white) * 255, work_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel(3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel(5))
        return self._remove_small(mask, 80)

    @staticmethod
    def _holes_inside(contour: np.ndarray, object_mask: np.ndarray) -> int:
        filled = np.zeros_like(object_mask)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        holes = cv2.bitwise_and(filled, cv2.bitwise_not(object_mask))
        contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(cv2.contourArea(item) >= 90 for item in contours)

    def _detect_red(self, image: np.ndarray, red_mask: np.ndarray) -> list[GlobalDetection]:
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        screw: list[GlobalDetection] = []
        gears: list[tuple[np.ndarray, float, int]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not 180 <= area <= 30000:
                continue
            _, _, width, height = cv2.boundingRect(contour)
            if area <= 1500 and max(width, height) <= 65:
                center = cv2.minAreaRect(contour)[0]
                screw.append(GlobalDetection("screw", center, contour, "high"))
            elif area >= 2500:
                gears.append((contour, area, self._holes_inside(contour, red_mask)))

        gear_results: list[GlobalDetection] = []
        if len(gears) == 3:
            large = max(gears, key=lambda item: (item[2], item[1]))
            remaining = [item for item in gears if item is not large]
            compound = max(remaining, key=lambda item: item[1])
            small = min(remaining, key=lambda item: item[1])
            for part_key, item in (
                ("gear_large", large),
                ("gear_compound", compound),
                ("gear_small", small),
            ):
                gear_results.append(
                    GlobalDetection(part_key, self._center(item[0]), item[0], "high")
                )
        else:
            for contour, area, hole_count in gears:
                equivalent_diameter = math.sqrt(4.0 * area / math.pi)
                if hole_count >= 3:
                    name = "gear_large"
                elif equivalent_diameter <= 125 and hole_count >= 1:
                    name = "gear_small"
                elif area > 6000:
                    name = "gear_compound"
                else:
                    name = "gear_small"
                gear_results.append(
                    GlobalDetection(name, self._center(contour), contour, "medium")
                )
        return screw + gear_results

    def _detect_base(self, image: np.ndarray, silver_mask: np.ndarray) -> GlobalDetection | None:
        contours, _ = cv2.findContours(silver_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [c for c in contours if 5000 <= cv2.contourArea(c) <= 60000]
        if not candidates:
            return None
        contour = max(candidates, key=cv2.contourArea)
        return GlobalDetection("base", self._center(contour), contour, "high")

    def _detect_base_holes(
        self, image: np.ndarray, base: GlobalDetection
    ) -> list[GlobalDetection]:
        base_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(base_mask, [base.contour], -1, 255, cv2.FILLED)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hole_mask = cv2.inRange(hsv, (0, 0, 0), (179, 255, 100))
        hole_mask = cv2.bitwise_and(hole_mask, base_mask)
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, self._kernel(3))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, self._kernel(3))
        contours, _ = cv2.findContours(
            hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            x, y, width, height = cv2.boundingRect(contour)
            aspect = min(width, height) / max(width, height) if max(width, height) else 0.0
            (_, _), radius = cv2.minEnclosingCircle(contour)
            u, v = self._center(contour)
            boundary = cv2.pointPolygonTest(base.contour, (u, v), True)
            fill_ratio = area / (math.pi * radius * radius) if radius > 1e-6 else 0.0
            ellipse_aspect = aspect
            if len(contour) >= 5:
                try:
                    (_, _), ellipse_size, _ = cv2.fitEllipse(contour)
                    ellipse_major = float(max(ellipse_size))
                    ellipse_minor = float(min(ellipse_size))
                    ellipse_aspect = ellipse_minor / max(ellipse_major, 1e-6)
                except cv2.error:
                    (_, _), rect_size, _ = cv2.minAreaRect(contour)
                    ellipse_minor = float(min(rect_size))
            else:
                (_, _), rect_size, _ = cv2.minAreaRect(contour)
                ellipse_minor = float(min(rect_size))
            if not (
                100 <= area <= 2000
                and 5.0 <= radius <= 30.0
                and circularity >= 0.65
                and aspect >= 0.48
                and ellipse_aspect >= 0.62
                and fill_ratio >= 0.50
                and boundary >= 3.0
            ):
                continue
            candidates.append((circularity + aspect + ellipse_aspect, u, v, radius, ellipse_minor, contour))
        selected = list(
            max(
                itertools.combinations(candidates, 3),
                key=lambda items: (
                    sum(item[0] for item in items)
                    + min(
                        math.hypot(a[1] - b[1], a[2] - b[2])
                        for a, b in itertools.combinations(items, 2)
                    )
                    / 45.0
                ),
            )
        ) if len(candidates) >= 3 else candidates[:]
        if len(selected) != 3:
            return []
        named_holes = assign_base_hole_names(
            selected,
            get_u=lambda item: item[1],
            get_v=lambda item: item[2],
            get_minor_size=lambda item: item[4],
        )
        return [
            GlobalDetection(name, (item[1], item[2]), item[5], "high", item[3])
            for name, item in named_holes.items()
        ]

    def _score_white_shaft(
        self, contour: np.ndarray, image_shape: tuple[int, ...], relaxed: bool
    ) -> tuple[float, float, np.ndarray] | None:
        area = float(cv2.contourArea(contour))
        min_area = 180 if relaxed else 420
        min_rectangularity = 0.36 if relaxed else 0.58
        min_short_side = 8 if relaxed else 14
        min_score = 2.05 if relaxed else 2.55
        if not min_area <= area <= 5000:
            return None
        x, y, width, height = cv2.boundingRect(contour)
        if (
            x <= 5
            or y <= 5
            or x + width >= image_shape[1] - 5
            or y + height >= image_shape[0] - 5
        ):
            return None
        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
        long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
        rect_area = float(rect_width * rect_height)
        if rect_area <= 0.0:
            return None
        rectangularity = area / rect_area
        aspect_ratio = long_side / max(short_side, 1e-6)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter else 0
        if (
            circularity >= 0.86
            or rectangularity < min_rectangularity
            or long_side > 120
            or short_side < min_short_side
            or aspect_ratio > 4.2
        ):
            return None
        score = (
            min(area / 900.0, 1.5)
            + min(rectangularity / max(min_rectangularity, 1e-6), 1.6)
            + min(short_side / max(min_short_side, 1e-6), 1.4)
            + max(0.0, 1.0 - max(aspect_ratio - 2.2, 0.0) / 2.0)
            - (0.25 if relaxed else 0.0)
        )
        if score < min_score:
            return None
        return score, rect_area, contour

    def _detect_shafts(self, image: np.ndarray, white_mask: np.ndarray) -> list[GlobalDetection]:
        shaft_candidates: list[tuple[float, float, np.ndarray]] = []
        for close_size, relaxed in ((5, False), (7, True)):
            mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, self._kernel(close_size))
            if relaxed:
                mask = cv2.dilate(mask, self._kernel(3), iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                candidate = self._score_white_shaft(contour, image.shape, relaxed)
                if candidate is not None:
                    shaft_candidates.append(candidate)
        deduplicated: list[tuple[float, float, np.ndarray]] = []
        for candidate in sorted(shaft_candidates, key=lambda item: (item[0], item[1]), reverse=True):
            center = np.asarray(self._center(candidate[2]), dtype=float)
            if all(np.linalg.norm(center - np.asarray(self._center(item[2]), dtype=float)) >= 12.0 for item in deduplicated):
                deduplicated.append(candidate)
        candidates = sorted(deduplicated, key=lambda item: item[1], reverse=True)[:2]
        results = []
        type_certain = len(candidates) == 2 and (candidates[0][1] - candidates[1][1]) / max(candidates[0][1], 1e-6) >= 0.08
        for index, (_, _, contour) in enumerate(candidates):
            name = "shaft_large" if index == 0 else "shaft_small"
            confidence = "high" if type_certain else "medium"
            results.append(GlobalDetection(name, self._center(contour), contour, confidence))
        return results

    @staticmethod
    def _deduplicate_by_part(
        detections: Sequence[GlobalDetection],
    ) -> list[GlobalDetection]:
        confidence_rank = {"high": 2, "medium": 1, "low": 0}
        best: dict[str, GlobalDetection] = {}
        for detection in detections:
            current = best.get(detection.part_key)
            if current is None:
                best[detection.part_key] = detection
                continue
            score = (
                confidence_rank.get(detection.confidence, -1),
                float(cv2.contourArea(detection.contour)),
            )
            current_score = (
                confidence_rank.get(current.confidence, -1),
                float(cv2.contourArea(current.contour)),
            )
            if score > current_score:
                best[detection.part_key] = detection
        return list(best.values())

    def detect(self, image: np.ndarray) -> tuple[list[GlobalDetection], dict[str, np.ndarray]]:
        work_mask = self._work_mask(image)
        red_mask = self._red_mask(image, work_mask)
        black_mask = self._black_mask(image, work_mask)
        white_shaft_mask = self._white_shaft_mask(image, work_mask)
        silver_mask = self._silver_mask(image, work_mask, red_mask, black_mask)
        base = self._detect_base(image, silver_mask)

        shaft_mask = white_shaft_mask.copy()
        red_object_exclusion = np.zeros_like(shaft_mask)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if red_contours:
            cv2.drawContours(red_object_exclusion, red_contours, -1, 255, cv2.FILLED)
            red_object_exclusion = cv2.dilate(red_object_exclusion, self._kernel(15))
            shaft_mask[red_object_exclusion > 0] = 0
        if base is not None:
            base_region = np.zeros_like(shaft_mask)
            cv2.drawContours(base_region, [base.contour], -1, 255, cv2.FILLED)
            base_region = cv2.dilate(base_region, self._kernel(3))
            shaft_mask[base_region > 0] = 0

        results = self._detect_red(image, red_mask) + self._detect_shafts(image, shaft_mask)
        if base is not None:
            results.append(base)
            results.extend(self._detect_base_holes(image, base))
        results = self._deduplicate_by_part(results)
        return results, {
            "work_mask": work_mask,
            "red_mask": red_mask,
            "black_mask": black_mask,
            "white_shaft_mask": white_shaft_mask,
            "shaft_mask": shaft_mask,
            "silver_mask": silver_mask,
        }

    @staticmethod
    def annotate(image: np.ndarray, detections: Sequence[GlobalDetection]) -> np.ndarray:
        annotated = image.copy()
        for detection in detections:
            color = (0, 220, 0) if detection.part_key == "base" else (0, 180, 255)
            cv2.drawContours(annotated, [detection.contour], -1, color, 2)
            center = tuple(np.round(detection.center_uv).astype(int))
            cv2.drawMarker(annotated, center, color, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                annotated,
                _output_name(detection.part_key),
                (center[0] + 7, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated


class GlobalVisionForceAdapter(Node):
    def __init__(self) -> None:
        super().__init__("global_vision_force_adapter")
        self.declare_parameter("target_part", "")
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("ee_frame", "panda_hand")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("handeye_yaml_path", DEFAULT_HANDEYE_PATH)
        self.declare_parameter("publish_rate", 2.0)
        self.declare_parameter("topic_prefix", GLOBAL_TOPIC_PREFIX)
        self.declare_parameter("debug_view", False)
        self.declare_parameter("depth_window", 7)
        self.declare_parameter("min_valid_depth_points", 3)
        self.declare_parameter("table_z_m", 0.0)
        self.declare_parameter("tf_timeout_sec", 0.2)
        self.declare_parameter("realsense_warmup_frames", 15)
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter(
            "aligned_depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("motion_threshold", 2.0)
        self.declare_parameter("stable_frames_required", 10)
        self.declare_parameter("consensus_frames", 5)
        self.declare_parameter("consensus_tolerance_m", 0.015)
        self.declare_parameter("publish_low_confidence", False)
        self.declare_parameter("expected_width", 640)
        self.declare_parameter("expected_height", 480)
        self.declare_parameter(
            "work_area_polygon", [104, 0, 568, 0, 588, 330, 455, 480, 120, 480]
        )

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        handeye_path = str(self.get_parameter("handeye_yaml_path").value)
        if not handeye_path:
            raise RuntimeError("handeye_yaml_path must point to an NPZ, YAML, or JSON file")
        self.transform_gripper_camera = load_handeye_transform(handeye_path)

        target = str(self.get_parameter("target_part").value).strip()
        normalized_target = target.lower().replace("-", "_").replace(" ", "_")
        self.target_key = (
            normalized_target
            if normalized_target in BASE_HOLE_NAMES
            else normalize_part_key(target) if target else None
        )
        self.debug_view = bool(self.get_parameter("debug_view").value)
        self.publish_low_confidence = bool(
            self.get_parameter("publish_low_confidence").value
        )
        self.depth_window = int(self.get_parameter("depth_window").value)
        self.min_valid_depth_points = int(
            self.get_parameter("min_valid_depth_points").value
        )
        self.table_z_m = float(self.get_parameter("table_z_m").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        if self.depth_window < 1:
            raise ValueError("depth_window must be positive")
        if self.min_valid_depth_points < 1:
            raise ValueError("min_valid_depth_points must be positive")
        if not math.isfinite(self.table_z_m):
            raise ValueError("table_z_m must be finite")
        if self.tf_timeout_sec < 0.0:
            raise ValueError("tf_timeout_sec must be non-negative")
        self.topic_prefix = str(self.get_parameter("topic_prefix").value).rstrip("/")
        if not self.topic_prefix:
            raise ValueError("topic_prefix must not be empty")

        self.detector = GlobalPartDetector(
            self.get_parameter("work_area_polygon").value
        )
        self.tf_provider = TFTransformProvider(self, self.base_frame, self.ee_frame)
        rate = float(self.get_parameter("publish_rate").value)
        if rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        output_names = list(PART_NAME_MAP.values()) + list(BASE_HOLE_NAMES)
        self._pose_publishers = {
            output_name: self.create_publisher(
                PoseStamped, f"{self.topic_prefix}/{output_name}/pose", 10
            )
            for output_name in output_names
        }
        self.consensus = SpatialConsensusFilter(
            int(self.get_parameter("consensus_frames").value),
            float(self.get_parameter("consensus_tolerance_m").value),
        )
        self.camera = ROSAlignedRGBDCamera(
            self,
            str(self.get_parameter("color_topic").value),
            str(self.get_parameter("aligned_depth_topic").value),
            str(self.get_parameter("camera_info_topic").value),
            int(self.get_parameter("realsense_warmup_frames").value),
            int(self.get_parameter("sync_queue_size").value),
            float(self.get_parameter("sync_slop_sec").value),
            float(self.get_parameter("motion_threshold").value),
            int(self.get_parameter("stable_frames_required").value),
            int(self.get_parameter("expected_width").value),
            int(self.get_parameter("expected_height").value),
        )
        self.timer = self.create_timer(1.0 / rate, self._process_frame)
        self.get_logger().info(
            f"Publishing {self.topic_prefix}/<part_name>/pose; "
            f"base={self.base_frame}, ee={self.ee_frame}, "
            f"camera={self.camera_frame}, target={self.target_key or 'all'}"
        )

    def _process_frame(self) -> None:
        frame = self.camera.capture()
        if frame is None:
            return
        if not frame.is_stable:
            self.consensus.clear()
            self.get_logger().info(
                f"Global image not stable: motion={frame.motion_score:.3f}, "
                f"stable_frames={frame.stable_frame_count}; suppressing output"
            )
            return
        detections, _ = self.detector.detect(frame.color_bgr)
        if self.target_key:
            detections = [item for item in detections if item.part_key == self.target_key]
        if not self.publish_low_confidence:
            detections = [item for item in detections if item.confidence != "low"]
        if not detections:
            self.consensus.clear()
            self.get_logger().warning("No requested global part was detected")
            self._show_debug(frame.color_bgr, [])
            return

        detected_names = {_output_name(item.part_key) for item in detections}
        for output_name in self._pose_publishers:
            if output_name not in detected_names:
                self.consensus.clear(output_name)

        transform_base_gripper = self.tf_provider.lookup_base_gripper(
            self.tf_timeout_sec,
            frame.stamp_sec,
            frame.stamp_nanosec,
        )
        if transform_base_gripper is None:
            self.consensus.clear()
            return
        self.get_logger().info(
            f"T_{self.base_frame}_{self.ee_frame} read successfully"
        )
        camera_matrix = np.array(
            [
                [float(frame.intrinsics["fx"]), 0.0, float(frame.intrinsics["ppx"])],
                [0.0, float(frame.intrinsics["fy"]), float(frame.intrinsics["ppy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        for detection in detections:
            u, v = detection.center_uv
            part_name = _output_name(detection.part_key)
            table_result = pixel_to_table_intersection_camera_and_base(
                u,
                v,
                camera_matrix,
                transform_base_gripper,
                self.transform_gripper_camera,
                self.table_z_m,
            )
            if not table_result["ray_table_intersection_valid"]:
                self.consensus.clear(part_name)
                self.get_logger().warning(
                    f"{part_name}: table intersection failed at pixel ({u:.1f}, {v:.1f}): "
                    f"{table_result['ray_table_intersection_reason']}; not publishing"
                )
                continue
            camera_point = np.asarray(table_result["center_camera_on_table_m"], dtype=np.float64)
            base_point = np.asarray(table_result["center_base_on_table_m"], dtype=np.float64)
            consensus_points = self.consensus.add(part_name, base_point)
            if consensus_points is None:
                self.get_logger().info(
                    f"{part_name}: collecting stable position consensus; not publishing yet"
                )
                continue
            base_point = consensus_points[0]
            if detection.confidence == "low":
                self.get_logger().warning(
                    f"{part_name}: publishing LOW-confidence global result; "
                    "use it only for coarse hover/debug, not final grasp"
                )
            message = PoseStamped()
            message.header.stamp.sec = int(frame.stamp_sec)
            message.header.stamp.nanosec = int(frame.stamp_nanosec)
            message.header.frame_id = self.base_frame
            message.pose.position.x = float(base_point[0])
            message.pose.position.y = float(base_point[1])
            message.pose.position.z = float(base_point[2])
            # Global detection has no reliable planar orientation estimate.
            quaternion = quaternion_from_roll_pitch_yaw(0.0, 0.0, 0.0)
            (
                message.pose.orientation.x,
                message.pose.orientation.y,
                message.pose.orientation.z,
                message.pose.orientation.w,
            ) = map(float, quaternion)
            topic = f"{self.topic_prefix}/{part_name}/pose"
            self._pose_publishers[part_name].publish(message)
            self.get_logger().info(
                f"Detected {part_name}: pixel=({u:.1f},{v:.1f}), "
                f"camera table XYZ={camera_point.tolist()}, {self.base_frame} table XYZ={base_point.tolist()}, "
                f"confidence={detection.confidence}, published {topic}"
            )
        self._show_debug(frame.color_bgr, detections)

    def _show_debug(
        self, image: np.ndarray, detections: Sequence[GlobalDetection]
    ) -> None:
        if not self.debug_view:
            return
        cv2.imshow("global_vision_force_adapter", self.detector.annotate(image, detections))
        cv2.waitKey(1)

    def destroy_node(self) -> bool:
        if self.debug_view:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError("rclpy and geometry_msgs are required to run this ROS2 node")
    rclpy.init(args=args)
    node = None
    try:
        node = GlobalVisionForceAdapter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        rclpy.logging.get_logger("global_vision_force_adapter").fatal(str(exc))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
