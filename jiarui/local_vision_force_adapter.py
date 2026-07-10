#!/usr/bin/env python3
"""ROS2 local RGB-D grasp refinement adapter for force_control.

Detection is derived from local_rgbd_grasp_refinement.ipynb. Robot motion is
deliberately outside this node; it publishes one refined base-frame pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from typing import Any, Sequence

import cv2
import numpy as np

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
    SpatialConsensusFilter,
    TFTransformProvider,
    assign_base_hole_names,
    deproject_pixel_with_depth,
    downward_grasp_quaternion,
    force_control_part_name,
    load_handeye_transform,
    median_depth_around_pixel,
    normalize_part_key,
    pixel_to_camera_point,
    transform_points_camera_to_base,
    valid_depth_values,
)


LOCAL_TOPIC_PREFIX = "/vision/local"


@dataclass
class LocalDetection:
    part_key: str
    grasp_left_uv: tuple[float, float]
    grasp_right_uv: tuple[float, float]
    grasp_midpoint_uv: tuple[float, float]
    confidence: str
    source: str
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    depth_overrides_m: dict[str, float] = field(default_factory=dict)


class LocalGraspDetector:
    """Pure OpenCV/NumPy local detector with no ROS dependencies."""

    def __init__(
        self,
        min_valid_depth_points: int = 5,
        target_roi_width_px: int = 640,
        target_roi_height_px: int = 480,
    ) -> None:
        self.min_valid_depth_points = max(1, int(min_valid_depth_points))
        self.target_roi_width_px = max(20, int(target_roi_width_px))
        self.target_roi_height_px = max(20, int(target_roi_height_px))

    def _center_roi(self, image_shape: tuple[int, ...]) -> np.ndarray:
        height, width = image_shape[:2]
        roi_width = min(width, self.target_roi_width_px)
        roi_height = min(height, self.target_roi_height_px)
        x0 = max(0, (width - roi_width) // 2)
        y0 = max(0, (height - roi_height) // 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0 : y0 + roi_height, x0 : x0 + roi_width] = 255
        return mask

    @staticmethod
    def _kernel(size: int) -> np.ndarray:
        size = max(1, int(size))
        size += int(size % 2 == 0)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    @classmethod
    def _clean(cls, mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
        result = mask.copy()
        if open_size > 1:
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, cls._kernel(open_size))
        if close_size > 1:
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, cls._kernel(close_size))
        return result

    @classmethod
    def _red_mask(cls, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 70, 40), (12, 255, 255)),
            cv2.inRange(hsv, (165, 70, 40), (179, 255, 255)),
        )
        return cls._clean(mask, 3, 9)

    @classmethod
    def _black_mask(cls, image: np.ndarray, maximum_value: int = 95) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return cls._clean(
            cv2.inRange(hsv, (0, 0, 0), (179, 255, int(maximum_value))), 3, 7
        )

    @classmethod
    def _white_shaft_mask(cls, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, saturation, value = cv2.split(hsv)
        lightness, a_channel, b_channel = cv2.split(lab)
        neutral = (
            (np.abs(a_channel.astype(np.int16) - 128) <= 32)
            & (np.abs(b_channel.astype(np.int16) - 128) <= 40)
        )
        white = (
            neutral
            & (saturation <= 95)
            & (value >= 130)
            & (lightness >= 135)
        )
        return cls._clean(np.uint8(white) * 255, 3, 5)

    @classmethod
    def _silver_masks(cls, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, saturation, value = cv2.split(hsv)
        lightness, a_channel, b_channel = cv2.split(lab)
        neutral = (
            (np.abs(a_channel.astype(np.int16) - 128) <= 34)
            & (np.abs(b_channel.astype(np.int16) - 128) <= 42)
        )
        silver = (
            neutral
            & (saturation <= 75)
            & (value >= 45)
            & (value <= 210)
            & (lightness <= 205)
        )
        support = (
            (np.abs(a_channel.astype(np.int16) - 128) <= 42)
            & (np.abs(b_channel.astype(np.int16) - 128) <= 50)
            & (saturation <= 105)
            & (value >= 45)
        )
        return (
            cls._clean(np.uint8(silver) * 255, 3, 13),
            cls._clean(np.uint8(support) * 255, 3, 5),
        )

    @staticmethod
    def _center(contour: np.ndarray) -> tuple[float, float]:
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-8:
            return (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
        return tuple(map(float, cv2.minAreaRect(contour)[0]))

    @staticmethod
    def _rect_axes(rect: tuple) -> tuple[np.ndarray, np.ndarray, float, float]:
        box = cv2.boxPoints(rect).astype(np.float64)
        edges = [box[(index + 1) % 4] - box[index] for index in range(4)]
        lengths = [float(np.linalg.norm(edge)) for edge in edges]
        long_axis = edges[int(np.argmax(lengths))]
        long_axis /= max(float(np.linalg.norm(long_axis)), 1e-9)
        short_axis = np.array([-long_axis[1], long_axis[0]])
        return long_axis, short_axis, max(lengths), min(lengths)

    @staticmethod
    def _mask_median_depth(
        depth_m: np.ndarray, mask: np.ndarray
    ) -> tuple[float | None, int]:
        values = valid_depth_values(np.asarray(depth_m)[mask > 0])
        if values.size == 0:
            return None, 0
        return float(np.median(values)), int(values.size)

    @staticmethod
    def _ring_depth(
        depth_m: np.ndarray,
        support_mask: np.ndarray,
        u: float,
        v: float,
        inner_radius: float,
        outer_radius: float,
    ) -> tuple[float | None, int]:
        height, width = depth_m.shape
        x0 = max(0, int(math.floor(u - outer_radius)))
        x1 = min(width, int(math.ceil(u + outer_radius)) + 1)
        y0 = max(0, int(math.floor(v - outer_radius)))
        y1 = min(height, int(math.ceil(v + outer_radius)) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance_squared = (xx - u) ** 2 + (yy - v) ** 2
        ring = (
            (distance_squared >= inner_radius**2)
            & (distance_squared <= outer_radius**2)
            & (support_mask[y0:y1, x0:x1] > 0)
        )
        values = valid_depth_values(depth_m[y0:y1, x0:x1][ring])
        if values.size == 0:
            return None, 0
        return float(np.median(values)), int(values.size)

    def detect_base(
        self,
        image: np.ndarray,
        depth_m: np.ndarray,
        hole_target: str = "hole_screw",
    ) -> LocalDetection | None:
        target = str(hole_target).strip().lower()
        if target not in BASE_HOLE_NAMES:
            raise ValueError(
                f"base_hole_target must be one of: {', '.join(BASE_HOLE_NAMES)}"
            )
        center_roi = self._center_roi(image.shape)
        silver, support = self._silver_masks(image)
        roi_source = cv2.morphologyEx(silver, cv2.MORPH_CLOSE, self._kernel(31), iterations=2)
        contours, _ = cv2.findContours(roi_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [contour for contour in contours if cv2.contourArea(contour) >= 5000]
        if not candidates:
            return None
        base_contour = max(candidates, key=cv2.contourArea)
        base_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(base_mask, [base_contour], -1, 255, cv2.FILLED)
        black = self._black_mask(image)
        hole_search = cv2.bitwise_and(black, base_mask)
        hole_contours, _ = cv2.findContours(
            hole_search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        holes = []
        for contour in hole_contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0
            x, y, width, height = cv2.boundingRect(contour)
            aspect = min(width, height) / max(width, height) if max(width, height) else 0
            (_, _), radius = cv2.minEnclosingCircle(contour)
            if len(contour) >= 5:
                (_, _), ellipse_size, _ = cv2.fitEllipse(contour)
                ellipse_minor = float(min(ellipse_size))
            else:
                (_, _), rect_size, _ = cv2.minAreaRect(contour)
                ellipse_minor = float(min(rect_size))
            if not (
                180 <= area <= 8000
                and 18 <= radius <= 55
                and circularity >= 0.68
                and aspect >= 0.55
            ):
                continue
            u, v = self._center(contour)
            if (
                u - radius <= 8
                or v - radius <= 8
                or u + radius >= image.shape[1] - 8
                or v + radius >= image.shape[0] - 8
            ):
                continue
            if cv2.pointPolygonTest(base_contour, (u, v), False) < 0:
                continue
            ring_outer = radius + 13
            x0 = max(0, int(math.floor(u - ring_outer)))
            x1 = min(image.shape[1], int(math.ceil(u + ring_outer)) + 1)
            y0 = max(0, int(math.floor(v - ring_outer)))
            y1 = min(image.shape[0], int(math.ceil(v + ring_outer)) + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            distance_squared = (xx - u) ** 2 + (yy - v) ** 2
            ring = (
                (distance_squared >= (radius + 3) ** 2)
                & (distance_squared <= ring_outer**2)
            )
            silver_fraction = (
                float(np.mean(support[y0:y1, x0:x1][ring] > 0))
                if np.any(ring)
                else 0.0
            )
            if silver_fraction < 0.12:
                continue
            depth, count = self._ring_depth(
                depth_m, support, u, v, radius + 3, radius + 13
            )
            if depth is None or count < self.min_valid_depth_points:
                continue
            holes.append((u, v, radius, depth, circularity, ellipse_minor))
        holes = sorted(holes, key=lambda item: item[4], reverse=True)[:3]
        holes = sorted(holes, key=lambda item: item[0])
        if len(holes) != 3:
            return None

        named_holes = assign_base_hole_names(
            holes,
            get_u=lambda item: item[0],
            get_v=lambda item: item[1],
            get_minor_size=lambda item: item[5],
        )
        outer_holes = sorted(
            (named_holes["hole_large"], named_holes["hole_small"]),
            key=lambda item: item[0],
        )
        left_uv = outer_holes[0][:2]
        right_uv = outer_holes[1][:2]
        selected_hole = named_holes[target]
        midpoint_uv, midpoint_depth = selected_hole[:2], selected_hole[3]
        return LocalDetection(
            part_key="base",
            grasp_left_uv=left_uv,
            grasp_right_uv=right_uv,
            grasp_midpoint_uv=midpoint_uv,
            confidence="high",
            source=f"three_base_holes_ring_depth:{target}",
            debug_images={
                "target_center_roi": center_roi,
                "base_mask": base_mask,
                "hole_search_mask": hole_search,
            },
            depth_overrides_m={
                "left": outer_holes[0][3],
                "right": outer_holes[1][3],
                "midpoint": midpoint_depth,
            },
        )

    def detect_shaft(
        self, image: np.ndarray, depth_image_m: np.ndarray, part_key: str
    ) -> LocalDetection | None:
        center_roi = self._center_roi(image.shape)
        shaft_mask = self._white_shaft_mask(image)
        contours, _ = cv2.findContours(
            shaft_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not 350 <= area <= 30000:
                continue
            rect = cv2.minAreaRect(contour)
            (_, _), (width, height), _ = rect
            rect_area = float(width * height)
            rectangularity = area / rect_area if rect_area else 0.0
            if rectangularity >= 0.55:
                candidates.append((rect_area, contour, rect, rectangularity))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        if part_key == "shaft_small" and len(candidates) >= 2:
            selected = candidates[1]
            selection_reason = "two_or_more_shafts_detected_select_smallest_area"
        else:
            selected = candidates[0]
            selection_reason = (
                "two_or_more_shafts_detected_select_largest_area"
                if part_key == "shaft_large" and len(candidates) >= 2
                else "single_or_dominant_shaft_candidate"
            )
        _, contour, rect, rectangularity = selected
        box = cv2.boxPoints(rect).astype(np.float64)
        edges = []
        for index in range(4):
            first, second = box[index], box[(index + 1) % 4]
            edges.append((float(np.linalg.norm(second - first)), (first + second) / 2.0))
        side_midpoints = [item[1] for item in sorted(edges, key=lambda item: item[0], reverse=True)[:2]]
        side_midpoints.sort(key=lambda point: (point[0], point[1]))
        selected_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(selected_mask, [contour], -1, 255, cv2.FILLED)
        object_depth, object_count = self._mask_median_depth(depth_image_m, selected_mask)
        depth_overrides = {}
        for name, point in (
            ("left", side_midpoints[0]),
            ("right", side_midpoints[1]),
            ("midpoint", rect[0]),
        ):
            local_depth, local_count = median_depth_around_pixel(
                depth_image_m, point[0], point[1], 7
            )
            if local_depth is not None and local_count >= self.min_valid_depth_points:
                depth_overrides[name] = local_depth
            elif object_depth is not None and object_count >= self.min_valid_depth_points:
                depth_overrides[name] = object_depth
        return LocalDetection(
            part_key=part_key,
            grasp_left_uv=tuple(side_midpoints[0]),
            grasp_right_uv=tuple(side_midpoints[1]),
            grasp_midpoint_uv=tuple(map(float, rect[0])),
            confidence="high" if rectangularity >= 0.72 else "medium",
            source=f"{selection_reason}:white_shaft_minAreaRect_long_edge_midpoints",
            debug_images={
                "target_center_roi": center_roi,
                "shaft_mask": shaft_mask,
                "selected_mask": selected_mask,
            },
            depth_overrides_m=depth_overrides,
        )

    @classmethod
    def _marker_components(
        cls,
        image: np.ndarray,
        roi_mask: np.ndarray,
        depth_m: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        """Notebook-final black-marker segmentation with component depths."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        full_mask = cls._clean(
            cv2.inRange(hsv, (0, 0, 0), (179, 255, 80)), 3, 3
        )
        restricted = cv2.bitwise_and(full_mask, roi_mask)
        contours, _ = cv2.findContours(
            restricted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        records: list[dict[str, object]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not 30 <= area <= 5000:
                continue
            rect = cv2.minAreaRect(contour)
            long_axis, short_axis, long_side, short_side = cls._rect_axes(rect)
            component_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, cv2.FILLED)
            component_depth, depth_count = cls._mask_median_depth(
                depth_m, component_mask
            )
            records.append(
                {
                    "area": area,
                    "contour": contour,
                    "rect": rect,
                    "center": np.asarray(rect[0], dtype=np.float64),
                    "long_axis": long_axis,
                    "short_axis": short_axis,
                    "long_side": long_side,
                    "short_side": short_side,
                    "mask": component_mask,
                    "depth": component_depth,
                    "depth_count": depth_count,
                }
            )
        records.sort(key=lambda item: float(item["area"]), reverse=True)
        return restricted, records

    def _extract_top_surface(
        self, depth_m: np.ndarray, object_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = (
            (object_mask > 0)
            & np.isfinite(depth_m)
            & (depth_m >= 0.05)
            & (depth_m <= 2.0)
        )
        if np.count_nonzero(valid) < max(50, self.min_valid_depth_points):
            return np.zeros_like(object_mask), np.uint8(valid) * 255
        yy, xx = np.nonzero(valid)
        values = depth_m[valid].astype(np.float64)
        design = np.column_stack([xx, yy, np.ones_like(xx)])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - design @ coefficients
        top = np.zeros_like(object_mask)
        top[yy[residual <= np.percentile(residual, 20)], xx[residual <= np.percentile(residual, 20)]] = 255
        top = self._clean(top, 3, 7)
        top = cv2.bitwise_and(top, object_mask)
        return top, np.uint8(valid) * 255

    def _split_top_surface(self, top: np.ndarray) -> list[np.ndarray]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(top, 8)
        components = [
            np.uint8(labels == index) * 255
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] >= 80
        ]
        components.sort(key=cv2.countNonZero, reverse=True)
        if len(components) == 2:
            return components
        yy, xx = np.nonzero(top > 0)
        if len(xx) < 160:
            return []
        samples = np.column_stack([xx, yy]).astype(np.float32)
        cv2.setRNGSeed(0)
        _, cluster_labels, _ = cv2.kmeans(
            samples,
            2,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2),
            10,
            cv2.KMEANS_PP_CENTERS,
        )
        split = []
        for index in range(2):
            mask = np.zeros_like(top)
            selected = cluster_labels.ravel() == index
            mask[yy[selected], xx[selected]] = 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel(3))
            if cv2.countNonZero(mask) >= 80:
                split.append(mask)
        return split

    def _outer_component_point(
        self, component_mask: np.ndarray, object_center: np.ndarray
    ) -> np.ndarray:
        working = cv2.erode(component_mask, self._kernel(3))
        if cv2.countNonZero(working) == 0:
            working = component_mask
        yy, xx = np.nonzero(working > 0)
        points = np.column_stack([xx, yy]).astype(np.float64)
        center = np.mean(points, axis=0)
        outward = center - object_center
        norm = float(np.linalg.norm(outward))
        outward = outward / norm if norm >= 1e-6 else np.array([1.0, 0.0])
        projections = (points - center) @ outward
        return np.median(points[projections >= np.percentile(projections, 85)], axis=0)

    def _gear_depth_pair(
        self,
        red_mask: np.ndarray,
        contour: np.ndarray,
        depth_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, np.ndarray]] | None:
        filled = np.zeros_like(red_mask)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        object_mask = cv2.bitwise_and(red_mask, filled)
        center = np.asarray(self._center(contour), dtype=np.float64)
        (_, _), radius = cv2.minEnclosingCircle(contour)
        inner = np.zeros_like(red_mask)
        cv2.circle(
            inner,
            tuple(np.round(center).astype(int)),
            max(4, int(radius * 0.46)),
            255,
            cv2.FILLED,
        )
        central = cv2.bitwise_and(object_mask, inner)
        top, valid_depth = self._extract_top_surface(depth_m, central)
        components = self._split_top_surface(top)
        if len(components) != 2:
            return None
        items = []
        combined = np.zeros_like(red_mask)
        for component in components:
            point = self._outer_component_point(component, center)
            component_depth, count = self._mask_median_depth(depth_m, component)
            if component_depth is None or count < self.min_valid_depth_points:
                return None
            if object_mask[int(round(point[1])), int(round(point[0]))] == 0:
                return None
            items.append((point, component_depth))
            combined = cv2.bitwise_or(combined, component)
        items.sort(key=lambda item: item[0][0])
        if float(np.linalg.norm(items[1][0] - items[0][0])) < 12.0:
            return None
        return (
            items[0][0],
            items[1][0],
            {
                "left": float(items[0][1]),
                "right": float(items[1][1]),
                "midpoint": float(0.5 * (items[0][1] + items[1][1])),
            },
            {
                "gear_object_mask": object_mask,
                "gear_valid_depth_mask": valid_depth,
                "gear_top_surface_mask": top,
                "gear_selected_components": combined,
            },
        )

    def detect_gear(
        self, image: np.ndarray, depth_m: np.ndarray, part_key: str
    ) -> LocalDetection | None:
        center_roi = self._center_roi(image.shape)
        red = self._red_mask(image)
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            if cv2.contourArea(contour) < 2500:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if x <= 1 or y <= 1 or x + width >= image.shape[1] - 1 or y + height >= image.shape[0] - 1:
                continue
            candidates.append(contour)
        if not candidates:
            return None
        contour = min(candidates, key=cv2.contourArea) if part_key == "gear_small" else max(candidates, key=cv2.contourArea)
        object_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)
        marker_mask, markers = self._marker_components(image, object_mask, depth_m)
        depth_overrides: dict[str, float] = {}
        debug_images = {
            "target_center_roi": center_roi,
            "gear_red_mask": red,
            "gear_object_mask": object_mask,
            "gear_marker_mask": marker_mask,
        }
        if len(markers) >= 2:
            selected = sorted(markers[:2], key=lambda item: tuple(item["center"]))
            left = np.asarray(selected[0]["center"], dtype=np.float64)
            right = np.asarray(selected[1]["center"], dtype=np.float64)
            object_depth, object_count = self._mask_median_depth(depth_m, object_mask)
            for name, marker in (("left", selected[0]), ("right", selected[1])):
                if marker["depth"] is not None and int(marker["depth_count"]) >= self.min_valid_depth_points:
                    depth_overrides[name] = float(marker["depth"])
                elif object_depth is not None and object_count >= self.min_valid_depth_points:
                    depth_overrides[name] = float(object_depth)
            if "left" in depth_overrides and "right" in depth_overrides:
                depth_overrides["midpoint"] = float(
                    np.median([depth_overrides["left"], depth_overrides["right"]])
                )
            confidence, source = "high", "two_black_grasp_markers"
        else:
            depth_pair = self._gear_depth_pair(red, contour, depth_m)
            if depth_pair is not None:
                left, right, depth_overrides, depth_debug = depth_pair
                debug_images.update(depth_debug)
                confidence, source = "low", "gear_depth_top_surface_fallback"
            else:
                center = np.asarray(self._center(contour), dtype=np.float64)
                (_, _), radius = cv2.minEnclosingCircle(contour)
                inner = np.zeros_like(red)
                cv2.circle(inner, tuple(np.round(center).astype(int)), max(4, int(radius * 0.46)), 255, cv2.FILLED)
                central = cv2.bitwise_and(red, inner)
                yy, xx = np.nonzero(central > 0)
                left_pixels = np.column_stack([xx[xx < center[0]], yy[xx < center[0]]])
                right_pixels = np.column_stack([xx[xx >= center[0]], yy[xx >= center[0]]])
                if len(left_pixels) == 0 or len(right_pixels) == 0:
                    return None
                left = np.median(left_pixels, axis=0)
                right = np.median(right_pixels, axis=0)
                confidence, source = "low", "gear_rgb_geometry_fallback"
        left, right = sorted((np.asarray(left), np.asarray(right)), key=lambda point: (point[0], point[1]))
        midpoint = tuple((np.asarray(left) + np.asarray(right)) / 2.0)
        return LocalDetection(
            part_key=part_key,
            grasp_left_uv=left,
            grasp_right_uv=right,
            grasp_midpoint_uv=midpoint,
            confidence=confidence,
            source=source,
            debug_images=debug_images,
            depth_overrides_m=depth_overrides,
        )

    def _screw_depth_pair(
        self, red: np.ndarray, contour: np.ndarray, depth_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, np.ndarray]] | None:
        filled = np.zeros_like(red)
        cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        object_mask = cv2.bitwise_and(red, filled)
        x, y, width, height = cv2.boundingRect(contour)
        screw_center = np.asarray(self._center(contour), dtype=np.float64)
        center_roi = np.zeros_like(red)
        cv2.ellipse(
            center_roi,
            tuple(np.round(screw_center).astype(int)),
            (max(3, int(0.5 * width * 0.55)), max(3, int(0.5 * height * 0.55))),
            0, 0, 360, 255, cv2.FILLED,
        )
        search = cv2.bitwise_and(object_mask, center_roi)
        valid = (
            (search > 0)
            & np.isfinite(depth_m)
            & (depth_m >= 0.05)
            & (depth_m <= 2.0)
        )
        values = depth_m[valid]
        if values.size < max(50, self.min_valid_depth_points):
            return None
        threshold = max(float(np.percentile(values, 20)), float(np.min(values)) + 0.0005)
        top = np.uint8(valid & (depth_m <= threshold)) * 255
        top = self._clean(top, 3, 5)
        contours, _ = cv2.findContours(top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        max_offset = 0.22 * max(width, height)
        for candidate in contours:
            area = float(cv2.contourArea(candidate))
            if area < 80:
                continue
            rect = cv2.minAreaRect(candidate)
            _, _, long_side, short_side = self._rect_axes(rect)
            aspect = long_side / max(short_side, 1.0)
            offset = float(np.linalg.norm(np.asarray(rect[0]) - screw_center))
            if aspect >= 1.25 and offset <= max_offset:
                candidates.append((2.0 * aspect + 0.01 * area - offset / max(max_offset, 1.0), candidate, rect))
        if not candidates:
            return None
        _, selected_contour, rect = max(candidates, key=lambda item: item[0])
        selected_mask = np.zeros_like(red)
        cv2.drawContours(selected_mask, [selected_contour], -1, 255, cv2.FILLED)
        _, short_axis, _, short_side = self._rect_axes(rect)
        center = np.asarray(rect[0], dtype=np.float64)
        half_offset = max(4.0, 0.45 * short_side)
        targets = (center - half_offset * short_axis, center + half_offset * short_axis)
        yy, xx = np.nonzero(selected_mask > 0)
        component_points = np.column_stack([xx, yy]).astype(np.float64)
        points = [component_points[np.argmin(np.sum((component_points - target) ** 2, axis=1))] for target in targets]
        points.sort(key=lambda point: (point[0], point[1]))
        if float(np.linalg.norm(points[1] - points[0])) < 8.0:
            return None
        signed = (component_points - center) @ short_axis
        half_depths = []
        for negative in (True, False):
            half_mask = np.zeros_like(red)
            chosen = signed <= 0 if negative else signed > 0
            half_mask[component_points[chosen, 1].astype(int), component_points[chosen, 0].astype(int)] = 255
            half_depths.append(self._mask_median_depth(depth_m, half_mask)[0])
        if any(value is None for value in half_depths):
            return None
        ordered_depths = sorted(zip([target[0] for target in targets], half_depths), key=lambda item: item[0])
        depths = {
            "left": float(ordered_depths[0][1]),
            "right": float(ordered_depths[1][1]),
        }
        depths["midpoint"] = 0.5 * (depths["left"] + depths["right"])
        return points[0], points[1], depths, {
            "screw_object_mask": object_mask,
            "screw_center_roi": center_roi,
            "screw_top_surface_mask": top,
            "screw_selected_top": selected_mask,
        }

    def detect_screw(
        self, image: np.ndarray, depth_m: np.ndarray
    ) -> LocalDetection | None:
        center_roi = self._center_roi(image.shape)
        red = self._red_mask(image)
        contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            _, _, width, height = cv2.boundingRect(contour)
            if 120 <= area <= 6000 and max(width, height) <= 130:
                candidates.append(contour)
        if not candidates:
            return None
        contour = max(candidates, key=cv2.contourArea)
        object_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)
        marker_mask, markers = self._marker_components(image, object_mask, depth_m)
        depth_overrides: dict[str, float] = {}
        debug_images = {
            "target_center_roi": center_roi,
            "screw_red_mask": red,
            "screw_object_mask": object_mask,
            "screw_marker_mask": marker_mask,
        }
        if markers:
            marker = markers[0]
            rect = marker["rect"]
            _, short_axis, _, short_side = self._rect_axes(rect)
            center = np.asarray(rect[0], dtype=np.float64)
            left, right = sorted(
                (center - 0.5 * short_side * short_axis, center + 0.5 * short_side * short_axis),
                key=lambda point: (point[0], point[1]),
            )
            object_depth, object_count = self._mask_median_depth(depth_m, object_mask)
            selected_depth = (
                float(marker["depth"])
                if marker["depth"] is not None
                and int(marker["depth_count"]) >= self.min_valid_depth_points
                else float(object_depth)
                if object_depth is not None
                and object_count >= self.min_valid_depth_points
                else None
            )
            if selected_depth is not None:
                depth_overrides = {
                    "left": selected_depth,
                    "right": selected_depth,
                    "midpoint": selected_depth,
                }
            confidence, source = "high", "screw_marker_short_axis"
        else:
            fallback = self._screw_depth_pair(red, contour, depth_m)
            if fallback is not None:
                left, right, depth_overrides, depth_debug = fallback
                debug_images.update(depth_debug)
                center = 0.5 * (np.asarray(left) + np.asarray(right))
                confidence, source = "low", "screw_depth_top_bar_fallback"
            else:
                rect = cv2.minAreaRect(contour)
                _, short_axis, _, short_side = self._rect_axes(rect)
                center = np.asarray(rect[0], dtype=np.float64)
                left, right = sorted(
                    (center - 0.5 * short_side * short_axis, center + 0.5 * short_side * short_axis),
                    key=lambda point: (point[0], point[1]),
                )
                confidence, source = "low", "screw_minAreaRect_fallback"
        return LocalDetection(
            part_key="screw",
            grasp_left_uv=left,
            grasp_right_uv=right,
            grasp_midpoint_uv=tuple(center),
            confidence=confidence,
            source=source,
            debug_images=debug_images,
            depth_overrides_m=depth_overrides,
        )

    def detect(
        self,
        image: np.ndarray,
        depth_m: np.ndarray,
        part_key: str,
        base_hole_target: str = "hole_screw",
    ) -> LocalDetection | None:
        if part_key == "base":
            return self.detect_base(image, depth_m, base_hole_target)
        if part_key.startswith("shaft_"):
            return self.detect_shaft(image, depth_m, part_key)
        if part_key.startswith("gear_"):
            return self.detect_gear(image, depth_m, part_key)
        if part_key == "screw":
            return self.detect_screw(image, depth_m)
        raise ValueError(f"No local detector for {part_key}")

    @staticmethod
    def annotate(image: np.ndarray, detection: LocalDetection | None) -> np.ndarray:
        annotated = image.copy()
        if detection is None:
            return annotated
        left = tuple(np.round(detection.grasp_left_uv).astype(int))
        right = tuple(np.round(detection.grasp_right_uv).astype(int))
        midpoint = tuple(np.round(detection.grasp_midpoint_uv).astype(int))
        cv2.line(annotated, left, right, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(annotated, left, 7, (255, 100, 20), -1)
        cv2.circle(annotated, right, 7, (20, 20, 255), -1)
        cv2.drawMarker(annotated, midpoint, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(
            annotated,
            f"{force_control_part_name(detection.part_key)} {detection.source}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated


class LocalVisionForceAdapter(Node):
    def __init__(self) -> None:
        super().__init__("local_vision_force_adapter")
        self.declare_parameter("target_part", "Shaft_large")
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("ee_frame", "panda_hand")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("handeye_yaml_path", DEFAULT_HANDEYE_PATH)
        self.declare_parameter("publish_rate", 5.0)
        self.declare_parameter("topic_prefix", LOCAL_TOPIC_PREFIX)
        self.declare_parameter("debug_view", False)
        self.declare_parameter("gripper_axis_mode", "x_parallel")
        self.declare_parameter("min_valid_depth_points", 5)
        self.declare_parameter("depth_window", 7)
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
        self.declare_parameter("consensus_tolerance_m", 0.008)
        self.declare_parameter("target_roi_width_px", 640)
        self.declare_parameter("target_roi_height_px", 480)
        self.declare_parameter("publish_low_confidence", False)
        self.declare_parameter("expected_width", 640)
        self.declare_parameter("expected_height", 480)
        self.declare_parameter("base_hole_target", "hole_screw")

        target = str(self.get_parameter("target_part").value).strip()
        if not target:
            raise RuntimeError("target_part is required for local refinement")
        self.target_key = normalize_part_key(target)
        self.part_name = force_control_part_name(self.target_key)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.gripper_axis_mode = str(self.get_parameter("gripper_axis_mode").value)
        self.min_valid_depth_points = int(
            self.get_parameter("min_valid_depth_points").value
        )
        self.depth_window = int(self.get_parameter("depth_window").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        if self.depth_window < 1:
            raise ValueError("depth_window must be positive")
        if self.min_valid_depth_points < 1:
            raise ValueError("min_valid_depth_points must be positive")
        if self.tf_timeout_sec < 0.0:
            raise ValueError("tf_timeout_sec must be non-negative")
        self.base_hole_target = str(
            self.get_parameter("base_hole_target").value
        ).strip().lower()
        if self.base_hole_target not in BASE_HOLE_NAMES:
            raise ValueError(
                f"base_hole_target must be one of: {', '.join(BASE_HOLE_NAMES)}"
            )
        self.debug_view = bool(self.get_parameter("debug_view").value)
        self.publish_low_confidence = bool(
            self.get_parameter("publish_low_confidence").value
        )
        self.topic_prefix = str(self.get_parameter("topic_prefix").value).rstrip("/")
        if not self.topic_prefix:
            raise ValueError("topic_prefix must not be empty")
        self.output_name = (
            self.base_hole_target if self.target_key == "base" else self.part_name
        )
        self.output_topic = f"{self.topic_prefix}/{self.output_name}/pose"

        handeye_path = str(self.get_parameter("handeye_yaml_path").value)
        if not handeye_path:
            raise RuntimeError("handeye_yaml_path must point to an NPZ, YAML, or JSON file")
        self.transform_gripper_camera = load_handeye_transform(handeye_path)

        self.detector = LocalGraspDetector(
            self.min_valid_depth_points,
            int(self.get_parameter("target_roi_width_px").value),
            int(self.get_parameter("target_roi_height_px").value),
        )
        self.consensus = SpatialConsensusFilter(
            int(self.get_parameter("consensus_frames").value),
            float(self.get_parameter("consensus_tolerance_m").value),
        )
        self.tf_provider = TFTransformProvider(self, self.base_frame, self.ee_frame)
        rate = float(self.get_parameter("publish_rate").value)
        if rate <= 0.0:
            raise ValueError("publish_rate must be positive")
        self.publisher = self.create_publisher(PoseStamped, self.output_topic, 10)
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
            f"Publishing {self.output_topic}; target={self.output_name}, base={self.base_frame}, "
            f"ee={self.ee_frame}, camera={self.camera_frame}, axis={self.gripper_axis_mode}"
        )

    def _camera_point(
        self,
        name: str,
        pixel: tuple[float, float],
        detection: LocalDetection,
        depth_m: np.ndarray,
        intrinsics: dict[str, float],
    ) -> np.ndarray | None:
        override = detection.depth_overrides_m.get(name)
        if override is not None:
            return deproject_pixel_with_depth(pixel[0], pixel[1], override, intrinsics)
        return pixel_to_camera_point(
            pixel[0],
            pixel[1],
            depth_m,
            intrinsics,
            window_size=self.depth_window,
            min_valid_depth_points=self.min_valid_depth_points,
        )

    def _process_frame(self) -> None:
        frame = self.camera.capture()
        if frame is None:
            return
        if not frame.is_stable:
            self.consensus.clear()
            self.get_logger().info(
                f"Local image not stable: motion={frame.motion_score:.3f}, "
                f"stable_frames={frame.stable_frame_count}; suppressing output"
            )
            return
        try:
            detection = self.detector.detect(
                frame.color_bgr,
                frame.depth_m,
                self.target_key,
                self.base_hole_target,
            )
        except Exception as exc:
            self.get_logger().warning(f"Local detection failed: {exc}")
            return
        if detection is None:
            self.consensus.clear(self.output_name)
            self.get_logger().warning(f"No local pose detected for {self.output_name}")
            self._show_debug(frame.color_bgr, None)
            return
        if detection.confidence == "low" and not self.publish_low_confidence:
            self.consensus.clear(self.output_name)
            self.get_logger().warning(
                f"{self.output_name}: rejected low-confidence fallback "
                f"source={detection.source}"
            )
            self._show_debug(frame.color_bgr, detection)
            return

        pixels = {
            "left": detection.grasp_left_uv,
            "right": detection.grasp_right_uv,
            "midpoint": detection.grasp_midpoint_uv,
        }
        camera_points = {
            name: self._camera_point(
                name, pixel, detection, frame.depth_m, frame.intrinsics
            )
            for name, pixel in pixels.items()
        }
        invalid = [name for name, point in camera_points.items() if point is None]
        if invalid:
            self.consensus.clear(self.output_name)
            self.get_logger().warning(
                f"{self.output_name}: invalid depth for {invalid}; local pose not published"
            )
            self._show_debug(frame.color_bgr, detection)
            return

        transform_base_gripper = self.tf_provider.lookup_base_gripper(
            self.tf_timeout_sec,
            frame.stamp_sec,
            frame.stamp_nanosec,
        )
        if transform_base_gripper is None:
            self.consensus.clear(self.output_name)
            return
        self.get_logger().info(
            f"T_{self.base_frame}_{self.ee_frame} read successfully"
        )
        ordered_camera = np.asarray(
            [camera_points["left"], camera_points["right"], camera_points["midpoint"]]
        )
        ordered_base = transform_points_camera_to_base(
            ordered_camera,
            transform_base_gripper,
            self.transform_gripper_camera,
        )
        consensus_base = self.consensus.add(self.output_name, ordered_base)
        if consensus_base is None:
            self.get_logger().info(
                f"{self.output_name}: collecting stable local-pose consensus; "
                "not publishing yet"
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
            f"Detected {self.output_name}: left_pixel={pixels['left']}, "
            f"right_pixel={pixels['right']}, midpoint_pixel={pixels['midpoint']}, "
            f"camera XYZ left/right/mid={ordered_camera.tolist()}, "
            f"{self.base_frame} XYZ midpoint={midpoint_base.tolist()}, "
            f"source={detection.source}, published {self.output_topic}"
        )
        self._show_debug(frame.color_bgr, detection)

    def _show_debug(
        self, image: np.ndarray, detection: LocalDetection | None
    ) -> None:
        if not self.debug_view:
            return
        cv2.imshow("local_vision_force_adapter", self.detector.annotate(image, detection))
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
        node = LocalVisionForceAdapter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        rclpy.logging.get_logger("local_vision_force_adapter").fatal(str(exc))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
