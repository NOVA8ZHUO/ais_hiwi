"""ROS-independent global RGB detector for the validated fixed camera view.

The thresholds and geometric rules in this module are copied from the verified
``global_rgb_part_detection.ipynb`` pipeline. Public results contain only
JSON-compatible Python values. OpenCV contours and NumPy masks remain internal.
"""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np


CAMERA_FRAME_ID = "camera_color_optical_frame"

# Manual work area for the current 640 x 480 global camera view.
USE_MANUAL_WORK_AREA = True
MANUAL_WORK_AREA_POLYGON = np.array(
    [[104, 0], [568, 0], [588, 330], [455, 480], [120, 480]],
    dtype=np.int32,
)
WORK_MASK_ERODE_PX = 3

# Red segmentation.
RED_HSV_LOW_1 = (0, 90, 55)
RED_HSV_HIGH_1 = (12, 255, 255)
RED_HSV_LOW_2 = (168, 90, 55)
RED_HSV_HIGH_2 = (179, 255, 255)
RED_OPEN_KERNEL = 3
RED_CLOSE_KERNEL = 7
RED_MIN_AREA_PX = 180
RED_MAX_AREA_PX = 30000
RED_SCREW_MAX_AREA_PX = 1500
RED_SCREW_MAX_DIM_PX = 55
RED_HOLE_MIN_AREA_PX = 90
RED_GEAR_SMALL_MAX_EQ_DIAMETER_PX = 125

# Black shaft segmentation and filtering.
BLACK_MAX_VALUE = 82
BLACK_MIN_AREA_PX = 550
BLACK_MAX_AREA_PX = 4200
BLACK_MIN_RECTANGULARITY = 0.68
BLACK_MAX_LONG_SIDE_PX = 90
BLACK_MIN_SHORT_SIDE_PX = 18
BLACK_MAX_CIRCULARITY_FOR_ROUND_BLOB = 0.86
EXPECTED_SHAFT_COUNT = 2

# Black insert holes inside the silver base.
BASE_INSERT_HOLE_MAX_VALUE = 100
BASE_INSERT_HOLE_OPEN_KERNEL = 3
BASE_INSERT_HOLE_CLOSE_KERNEL = 3
BASE_INSERT_HOLE_MIN_AREA = 80
BASE_INSERT_HOLE_MAX_AREA = 2000
BASE_INSERT_HOLE_MIN_RADIUS = 5.0
BASE_INSERT_HOLE_MAX_RADIUS = 30.0
BASE_INSERT_HOLE_MIN_CIRCULARITY = 0.50
BASE_INSERT_HOLE_MIN_ASPECT_RATIO = 0.40
BASE_INSERT_HOLE_MIN_BOUNDARY_DISTANCE = 3.0
BASE_INSERT_HOLE_EXPECTED_COUNT = 3

# Silver base segmentation.
SILVER_MAX_SATURATION = 50
SILVER_MIN_VALUE = 70
SILVER_MAX_VALUE = 175
SILVER_MAX_L = 168
SILVER_LAB_A_TOLERANCE = 30
SILVER_LAB_B_TOLERANCE = 38
BASE_MIN_AREA_PX = 5000
BASE_MAX_AREA_PX = 60000

IMAGE_BORDER_MARGIN_PX = 5
MASK_EXCLUSION_DILATE_PX = 7

EXPECTED_INSTANCES = {
    "gear_large",
    "gear_small",
    "gear_compound",
    "screw",
    "shaft_large",
    "shaft_small",
    "base",
}

CLASS_COLORS = {
    "gear_large": (20, 20, 255),
    "gear_small": (0, 110, 255),
    "gear_compound": (180, 0, 255),
    "screw": (0, 220, 255),
    "shaft_large": (255, 80, 20),
    "shaft_small": (255, 180, 20),
    "base": (40, 220, 40),
    "hole_l": (255, 255, 0),
    "hole_m": (255, 255, 0),
    "hole_r": (255, 255, 0),
}


def _empty_result(
    image_width: int = 0,
    image_height: int = 0,
    timestamp_sec: float | None = None,
) -> dict[str, Any]:
    return {
        "timestamp_sec": float(time.time() if timestamp_sec is None else timestamp_sec),
        "frame_id": CAMERA_FRAME_ID,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "coordinate_type": "pixel_2d",
        "detections": [],
        "base_holes": {},
        "status": {"ok": False, "warnings": [], "errors": []},
    }


def _odd_kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 255
    return clean


def _auto_detect_white_table(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 145), (179, 90, 255))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not contours:
        mask[:] = 255
        return mask
    cv2.drawContours(mask, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 255, -1)
    return mask


def create_work_area_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if USE_MANUAL_WORK_AREA:
        polygon = MANUAL_WORK_AREA_POLYGON.copy()
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [polygon], 255)
    else:
        mask = _auto_detect_white_table(image)
    if WORK_MASK_ERODE_PX > 0:
        mask = cv2.erode(mask, _odd_kernel(2 * WORK_MASK_ERODE_PX + 1))
    return mask


def segment_red_objects(image: np.ndarray, work_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, RED_HSV_LOW_1, RED_HSV_HIGH_1),
        cv2.inRange(hsv, RED_HSV_LOW_2, RED_HSV_HIGH_2),
    )
    red = cv2.bitwise_and(red, work_mask)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, _odd_kernel(RED_OPEN_KERNEL))
    red = cv2.morphologyEx(
        red, cv2.MORPH_CLOSE, _odd_kernel(RED_CLOSE_KERNEL), iterations=2
    )
    return _remove_small_components(red, RED_MIN_AREA_PX)


def segment_black_objects(image: np.ndarray, work_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    black = cv2.inRange(hsv, (0, 0, 0), (179, 255, BLACK_MAX_VALUE))
    black = cv2.bitwise_and(black, work_mask)
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, _odd_kernel(3))
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, _odd_kernel(5))
    return _remove_small_components(black, 80)


def segment_silver_base(
    image: np.ndarray,
    work_mask: np.ndarray,
    red_mask: np.ndarray,
    black_mask: np.ndarray,
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, saturation, value = cv2.split(hsv)
    lab_l, lab_a, lab_b = cv2.split(lab)
    neutral_lab = (
        (np.abs(lab_a.astype(np.int16) - 128) <= SILVER_LAB_A_TOLERANCE)
        & (np.abs(lab_b.astype(np.int16) - 128) <= SILVER_LAB_B_TOLERANCE)
    )
    gray_hsv = (
        (saturation <= SILVER_MAX_SATURATION)
        & (value >= SILVER_MIN_VALUE)
        & (value <= SILVER_MAX_VALUE)
        & (lab_l <= SILVER_MAX_L)
    )
    silver = np.uint8(neutral_lab & gray_hsv) * 255
    silver = cv2.bitwise_and(silver, work_mask)
    exclusion = cv2.dilate(
        cv2.bitwise_or(red_mask, black_mask), _odd_kernel(MASK_EXCLUSION_DILATE_PX)
    )
    silver[exclusion > 0] = 0
    silver = cv2.morphologyEx(silver, cv2.MORPH_OPEN, _odd_kernel(3))
    silver = cv2.morphologyEx(silver, cv2.MORPH_CLOSE, _odd_kernel(11))
    return _remove_small_components(silver, 250)


def _contour_touches_image_border(contour: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    x, y, width, height = cv2.boundingRect(contour)
    image_height, image_width = image_shape[:2]
    return (
        x <= IMAGE_BORDER_MARGIN_PX
        or y <= IMAGE_BORDER_MARGIN_PX
        or x + width >= image_width - IMAGE_BORDER_MARGIN_PX
        or y + height >= image_height - IMAGE_BORDER_MARGIN_PX
    )


def _contour_center(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-8:
        return (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
    x, y, width, height = cv2.boundingRect(contour)
    return float(x + width / 2.0), float(y + height / 2.0)


def _estimate_hole_center(contour: np.ndarray) -> tuple[float, float]:
    """Estimate a hole center from its fitted ellipse or enclosing circle."""
    if len(contour) >= 5:
        try:
            (center_u, center_v), _, _ = cv2.fitEllipse(contour)
            if math.isfinite(center_u) and math.isfinite(center_v):
                return float(center_u), float(center_v)
        except cv2.error:
            pass
    try:
        (center_u, center_v), _ = cv2.minEnclosingCircle(contour)
        if math.isfinite(center_u) and math.isfinite(center_v):
            return float(center_u), float(center_v)
    except cv2.error:
        pass
    return _contour_center(contour)


def _rotated_rect_features(contour: np.ndarray) -> dict[str, Any]:
    rect = cv2.minAreaRect(contour)
    (center_u, center_v), (raw_width, raw_height), raw_angle = rect
    if raw_width >= raw_height:
        long_side = float(raw_width)
        short_side = float(raw_height)
        yaw_deg = float(raw_angle)
    else:
        long_side = float(raw_height)
        short_side = float(raw_width)
        yaw_deg = float(raw_angle + 90.0)
    yaw_deg = ((yaw_deg + 90.0) % 180.0) - 90.0
    box = cv2.boxPoints(rect)
    return {
        "center": [float(center_u), float(center_v)],
        "raw_size": [float(raw_width), float(raw_height)],
        "long_side_px": long_side,
        "short_side_px": short_side,
        "yaw_deg": yaw_deg,
        "box_points": [[float(x), float(y)] for x, y in box],
    }


def _basic_contour_features(contour: np.ndarray) -> dict[str, float | int]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    return {
        "area": area,
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(width),
        "bbox_h": int(height),
        "circularity": 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0,
        "equivalent_diameter": math.sqrt(4.0 * area / math.pi) if area else 0.0,
        "contour_complexity": int(len(cv2.approxPolyDP(contour, 0.01 * perimeter, True))),
        "solidity": area / hull_area if hull_area else 0.0,
    }


def _find_holes_inside_contour(
    contour: np.ndarray, object_mask: np.ndarray
) -> list[np.ndarray]:
    filled = np.zeros_like(object_mask)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    holes = cv2.bitwise_and(filled, cv2.bitwise_not(object_mask))
    hole_contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [
        hole for hole in hole_contours if cv2.contourArea(hole) >= RED_HOLE_MIN_AREA_PX
    ]


def _new_internal_result(
    class_name: str, contour: np.ndarray, confidence: str, rule_note: str
) -> dict[str, Any]:
    features = _basic_contour_features(contour)
    rotated = _rotated_rect_features(contour)
    center_u, center_v = _contour_center(contour)
    return {
        "instance_name": class_name,
        "class_name": class_name,
        "center_u": round(center_u, 2),
        "center_v": round(center_v, 2),
        "bbox_x": int(features["bbox_x"]),
        "bbox_y": int(features["bbox_y"]),
        "bbox_w": int(features["bbox_w"]),
        "bbox_h": int(features["bbox_h"]),
        "area_px": round(float(features["area"]), 1),
        "yaw_deg": round(float(rotated["yaw_deg"]), 2),
        "confidence": str(confidence),
        "rule_note": str(rule_note),
        "box_points": rotated["box_points"],
        "_contour": contour,
        "_rotated_rect": rotated,
        "_features": features,
    }


def _classify_red_part(
    contour: np.ndarray, red_mask: np.ndarray
) -> tuple[str, str, str]:
    features = _basic_contour_features(contour)
    max_dim = max(int(features["bbox_w"]), int(features["bbox_h"]))
    hole_count = len(_find_holes_inside_contour(contour, red_mask))
    if features["area"] <= RED_SCREW_MAX_AREA_PX and max_dim <= RED_SCREW_MAX_DIM_PX:
        return "screw", "high", "small red component; compact non-gear geometry"
    if hole_count >= 3:
        return (
            "gear_large",
            "high",
            f"large gear-like contour with {hole_count} significant internal holes",
        )
    if (
        features["equivalent_diameter"] <= RED_GEAR_SMALL_MAX_EQ_DIAMETER_PX
        and hole_count >= 1
    ):
        return "gear_small", "high", "medium red gear-like contour with a central hole"
    if features["area"] > 6000 and hole_count <= 2:
        return (
            "gear_compound",
            "medium",
            "large gear-like contour; compound structure inferred by size and few through-holes",
        )
    return "gear_small", "medium", "red gear candidate classified mainly by size"


def _detect_red_parts(image: np.ndarray, red_mask: np.ndarray) -> list[dict[str, Any]]:
    del image
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not RED_MIN_AREA_PX <= area <= RED_MAX_AREA_PX:
            continue
        class_name, confidence, note = _classify_red_part(contour, red_mask)
        item = _new_internal_result(class_name, contour, confidence, note)
        item["_hole_count"] = len(_find_holes_inside_contour(contour, red_mask))
        candidates.append(item)

    gear_candidates = [item for item in candidates if item["class_name"] != "screw"]
    if len(gear_candidates) == 3:
        large = max(gear_candidates, key=lambda item: (item["_hole_count"], item["area_px"]))
        remaining = [item for item in gear_candidates if item is not large]
        compound = max(remaining, key=lambda item: item["area_px"])
        small = min(remaining, key=lambda item: item["area_px"])
        large.update(
            instance_name="gear_large",
            class_name="gear_large",
            confidence="high" if large["_hole_count"] >= 3 else "medium",
            rule_note=f"largest multi-hole gear; {large['_hole_count']} significant holes",
        )
        small.update(
            instance_name="gear_small",
            class_name="gear_small",
            confidence="high" if small["_hole_count"] >= 1 else "medium",
            rule_note="smallest gear candidate; central-hole evidence included",
        )
        compound.update(
            instance_name="gear_compound",
            class_name="gear_compound",
            confidence="medium",
            rule_note=(
                "large remaining gear with few through-holes; size/structure fallback, "
                "not y-position only"
            ),
        )
    return candidates


def _detect_black_shafts(
    image: np.ndarray, black_mask: np.ndarray
) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        features = _basic_contour_features(contour)
        if not BLACK_MIN_AREA_PX <= features["area"] <= BLACK_MAX_AREA_PX:
            continue
        if _contour_touches_image_border(contour, image.shape):
            continue
        rotated = _rotated_rect_features(contour)
        rect_area = rotated["long_side_px"] * rotated["short_side_px"]
        rectangularity = features["area"] / rect_area if rect_area > 0 else 0.0
        looks_round = (
            features["circularity"] >= BLACK_MAX_CIRCULARITY_FOR_ROUND_BLOB
            and features["contour_complexity"] > 6
        )
        if (
            looks_round
            or rectangularity < BLACK_MIN_RECTANGULARITY
            or rotated["long_side_px"] > BLACK_MAX_LONG_SIDE_PX
            or rotated["short_side_px"] < BLACK_MIN_SHORT_SIDE_PX
        ):
            continue
        provisional = "shaft_large" if rect_area >= 1450 else "shaft_small"
        item = _new_internal_result(
            provisional,
            contour,
            "high" if provisional == "shaft_large" else "medium",
            "dark rectangular component selected by rotated geometry",
        )
        item["_rect_area"] = float(rect_area)
        candidates.append(item)

    candidates.sort(key=lambda item: item["_rect_area"], reverse=True)
    candidates = candidates[:EXPECTED_SHAFT_COUNT]
    if candidates:
        candidates[0].update(
            instance_name="shaft_large",
            class_name="shaft_large",
            confidence="high",
            rule_note="largest valid dark rectangular component; circular holes rejected",
        )
    if len(candidates) >= 2:
        candidates[1].update(
            instance_name="shaft_small",
            class_name="shaft_small",
            confidence="high",
            rule_note="smaller valid dark rectangular component; circular holes rejected",
        )
    return candidates


def _detect_silver_base(
    image: np.ndarray, base_mask: np.ndarray
) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [
        contour
        for contour in contours
        if BASE_MIN_AREA_PX <= cv2.contourArea(contour) <= BASE_MAX_AREA_PX
        and not _contour_touches_image_border(contour, image.shape)
    ]
    if not candidates:
        return []
    contour = max(candidates, key=cv2.contourArea)
    return [
        _new_internal_result(
            "base",
            contour,
            "high",
            "largest neutral-gray object; black insert holes detected separately",
        )
    ]


def _detect_base_insert_holes(
    image: np.ndarray, base_contour: np.ndarray, black_mask: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    base_roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(base_roi_mask, [base_contour], -1, 255, thickness=-1)

    # Use a slightly more permissive dark threshold only inside the base ROI.
    # This does not affect the global black mask used by shaft detection.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    base_dark_mask = cv2.inRange(
        hsv, (0, 0, 0), (179, 255, BASE_INSERT_HOLE_MAX_VALUE)
    )
    hole_search_mask = cv2.bitwise_and(base_dark_mask, base_roi_mask)
    hole_search_mask = cv2.bitwise_or(
        hole_search_mask, cv2.bitwise_and(black_mask, base_roi_mask)
    )
    hole_search_mask = cv2.morphologyEx(
        hole_search_mask,
        cv2.MORPH_OPEN,
        _odd_kernel(BASE_INSERT_HOLE_OPEN_KERNEL),
    )
    hole_search_mask = cv2.morphologyEx(
        hole_search_mask,
        cv2.MORPH_CLOSE,
        _odd_kernel(BASE_INSERT_HOLE_CLOSE_KERNEL),
    )
    contours, _ = cv2.findContours(
        hole_search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not BASE_INSERT_HOLE_MIN_AREA <= area <= BASE_INSERT_HOLE_MAX_AREA:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else 0.0
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) else 0.0
        (_, _), radius = cv2.minEnclosingCircle(contour)
        center_u, center_v = _estimate_hole_center(contour)
        ellipse_major = 2.0 * float(radius)
        ellipse_minor = 2.0 * float(radius)
        if len(contour) >= 5:
            try:
                _, (ellipse_width, ellipse_height), _ = cv2.fitEllipse(contour)
                ellipse_major = max(float(ellipse_width), float(ellipse_height))
                ellipse_minor = min(float(ellipse_width), float(ellipse_height))
            except cv2.error:
                pass
        ellipse_aspect_ratio = (
            ellipse_minor / ellipse_major if ellipse_major > 1e-6 else 0.0
        )
        boundary_distance = cv2.pointPolygonTest(
            base_contour, (float(center_u), float(center_v)), True
        )
        if (
            circularity < BASE_INSERT_HOLE_MIN_CIRCULARITY
            or aspect_ratio < BASE_INSERT_HOLE_MIN_ASPECT_RATIO
            or not BASE_INSERT_HOLE_MIN_RADIUS <= radius <= BASE_INSERT_HOLE_MAX_RADIUS
            or boundary_distance < BASE_INSERT_HOLE_MIN_BOUNDARY_DISTANCE
        ):
            continue

        circularity_score = min(
            max(
                (circularity - BASE_INSERT_HOLE_MIN_CIRCULARITY)
                / max(1.0 - BASE_INSERT_HOLE_MIN_CIRCULARITY, 1e-6),
                0.0,
            ),
            1.0,
        )
        ellipse_score = min(
            max(
                (ellipse_aspect_ratio - BASE_INSERT_HOLE_MIN_ASPECT_RATIO)
                / max(1.0 - BASE_INSERT_HOLE_MIN_ASPECT_RATIO, 1e-6),
                0.0,
            ),
            1.0,
        )
        radius_midpoint = 0.5 * (
            BASE_INSERT_HOLE_MIN_RADIUS + BASE_INSERT_HOLE_MAX_RADIUS
        )
        radius_half_range = 0.5 * (
            BASE_INSERT_HOLE_MAX_RADIUS - BASE_INSERT_HOLE_MIN_RADIUS
        )
        radius_score = max(
            0.0, 1.0 - abs(radius - radius_midpoint) / radius_half_range
        )
        boundary_score = min(max(boundary_distance / 25.0, 0.0), 1.0)
        area_score = min(area / BASE_INSERT_HOLE_MAX_AREA, 1.0)
        score = (
            0.35 * circularity_score
            + 0.30 * ellipse_score
            + 0.20 * boundary_score
            + 0.10 * radius_score
            + 0.05 * area_score
        )
        candidates.append(
            {
                "contour": contour,
                "center_u": float(center_u),
                "center_v": float(center_v),
                "radius_px": float(radius),
                "area_px": area,
                "circularity": float(circularity),
                "ellipse_major_px": float(ellipse_major),
                "ellipse_minor_px": float(ellipse_minor),
                "ellipse_aspect_ratio": float(ellipse_aspect_ratio),
                "boundary_distance_px": float(boundary_distance),
                "score": float(score),
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[:BASE_INSERT_HOLE_EXPECTED_COUNT]
    selected.sort(key=lambda item: item["center_u"])
    holes = []
    for name, candidate in zip(("hole_l", "hole_m", "hole_r"), selected):
        holes.append(
            {
                "name": name,
                "center_u": round(candidate["center_u"], 2),
                "center_v": round(candidate["center_v"], 2),
                "radius_px": round(candidate["radius_px"], 2),
                "area_px": round(candidate["area_px"], 1),
                "confidence": "high" if candidate["circularity"] >= 0.75 else "medium",
                "rule_note": (
                    "dark insert-hole contour inside base ROI; "
                    "pixel center from ellipse/circle fit"
                ),
                "_contour": candidate["contour"],
            }
        )
    candidates_mask = np.zeros_like(hole_search_mask)
    for candidate in candidates:
        cv2.drawContours(
            candidates_mask, [candidate["contour"]], -1, 255, thickness=1
        )
        cv2.circle(
            candidates_mask,
            (
                int(round(candidate["center_u"])),
                int(round(candidate["center_v"])),
            ),
            2,
            255,
            thickness=-1,
        )
    return holes, hole_search_mask, candidates_mask


def _public_detection(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_name": str(item["instance_name"]),
        "class_name": str(item["class_name"]),
        "center_u": float(item["center_u"]),
        "center_v": float(item["center_v"]),
        "bbox_x": int(item["bbox_x"]),
        "bbox_y": int(item["bbox_y"]),
        "bbox_w": int(item["bbox_w"]),
        "bbox_h": int(item["bbox_h"]),
        "area_px": float(item["area_px"]),
        "yaw_deg": float(item["yaw_deg"]),
        "confidence": str(item["confidence"]),
        "rule_note": str(item["rule_note"]),
        "box_points": [
            [float(point[0]), float(point[1])] for point in item["box_points"]
        ],
    }


def _public_hole(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(item["name"]),
        "center_u": float(item["center_u"]),
        "center_v": float(item["center_v"]),
        "radius_px": float(item["radius_px"]),
        "area_px": float(item["area_px"]),
        "confidence": str(item["confidence"]),
        "rule_note": str(item["rule_note"]),
    }


def detect_global_parts_bgr(
    color_image_bgr: np.ndarray, return_debug: bool = False
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Detect global parts in one OpenCV BGR image.

    With ``return_debug=False`` this returns a JSON-compatible result dict.
    With ``return_debug=True`` it returns ``(result, debug_masks)``. Debug masks
    are NumPy arrays and are intentionally kept outside the publishable result.
    """
    if not isinstance(color_image_bgr, np.ndarray) or color_image_bgr.ndim != 3:
        result = _empty_result()
        result["status"]["errors"].append("color_image_bgr must be an HxWx3 NumPy array")
        return (result, {}) if return_debug else result
    height, width = color_image_bgr.shape[:2]
    result = _empty_result(width, height)
    if color_image_bgr.shape[2] != 3 or color_image_bgr.size == 0:
        result["status"]["errors"].append("color_image_bgr must have exactly 3 BGR channels")
        return (result, {}) if return_debug else result

    debug_masks: dict[str, np.ndarray] = {}
    try:
        work_mask = create_work_area_mask(color_image_bgr)
        red_mask = segment_red_objects(color_image_bgr, work_mask)
        black_mask = segment_black_objects(color_image_bgr, work_mask)
        base_mask = segment_silver_base(
            color_image_bgr, work_mask, red_mask, black_mask
        )
        red_results = _detect_red_parts(color_image_bgr, red_mask)
        shaft_results = _detect_black_shafts(color_image_bgr, black_mask)
        base_results = _detect_silver_base(color_image_bgr, base_mask)

        hole_results: list[dict[str, Any]] = []
        hole_search_mask = np.zeros((height, width), dtype=np.uint8)
        hole_candidates_mask = np.zeros((height, width), dtype=np.uint8)
        base_surface_mask = np.zeros((height, width), dtype=np.uint8)
        if base_results:
            cv2.drawContours(
                base_surface_mask, [base_results[0]["_contour"]], -1, 255, -1
            )
            base_surface_mask = cv2.bitwise_and(base_surface_mask, base_mask)
            hole_results, hole_search_mask, hole_candidates_mask = _detect_base_insert_holes(
                color_image_bgr, base_results[0]["_contour"], black_mask
            )

        all_parts = red_results + shaft_results + base_results
        result["detections"] = [_public_detection(item) for item in all_parts]
        result["base_holes"] = {
            item["name"]: _public_hole(item) for item in hole_results
        }

        found_instances = {item["instance_name"] for item in result["detections"]}
        for missing in sorted(EXPECTED_INSTANCES - found_instances):
            result["status"]["warnings"].append(f"missing detection: {missing}")
        for missing in sorted({"hole_l", "hole_m", "hole_r"} - set(result["base_holes"])):
            result["status"]["warnings"].append(f"missing base hole: {missing}")
        result["status"]["ok"] = not result["status"]["errors"]

        debug_masks = {
            "work_area_mask": work_mask,
            "red_mask": red_mask,
            "black_mask": black_mask,
            "silver_base_mask": base_mask,
            "base_surface_mask": base_surface_mask,
            "base_hole_search_mask": hole_search_mask,
            "base_hole_candidates_mask": hole_candidates_mask,
        }
    except (cv2.error, ValueError, TypeError) as exc:
        result["status"]["errors"].append(f"detection pipeline failed: {exc}")
        result["status"]["ok"] = False

    return (result, debug_masks) if return_debug else result


def draw_global_results(
    color_image_bgr: np.ndarray,
    result: dict[str, Any],
    show_depth_status: bool = True,
) -> np.ndarray:
    """Return an annotated BGR image using only fields in the public result."""
    annotated = color_image_bgr.copy()
    for detection in result.get("detections", []):
        name = str(detection.get("instance_name", detection.get("class_name", "part")))
        color = CLASS_COLORS.get(name, (255, 255, 0))
        center = (
            int(round(float(detection["center_u"]))),
            int(round(float(detection["center_v"]))),
        )
        box_points = detection.get("box_points")
        if box_points:
            box = np.round(np.asarray(box_points, dtype=np.float32)).astype(np.int32)
            cv2.polylines(annotated, [box], True, color, 2, cv2.LINE_AA)
        else:
            x, y = int(detection["bbox_x"]), int(detection["bbox_y"])
            w, h = int(detection["bbox_w"]), int(detection["bbox_h"])
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.circle(annotated, center, 4, color, -1, cv2.LINE_AA)

        yaw_rad = math.radians(float(detection.get("yaw_deg", 0.0)))
        line_length = max(24.0, 0.35 * max(detection["bbox_w"], detection["bbox_h"]))
        endpoint = (
            int(round(center[0] + line_length * math.cos(yaw_rad))),
            int(round(center[1] + line_length * math.sin(yaw_rad))),
        )
        cv2.line(annotated, center, endpoint, color, 2, cv2.LINE_AA)
        depth_text = ""
        if show_depth_status and "depth_valid" in detection:
            depth_text = " depth=ok" if detection["depth_valid"] else " depth=invalid"
        label = f"{name} ({center[0]},{center[1]}){depth_text}"
        text_position = (
            int(detection.get("bbox_x", center[0])),
            max(18, int(detection.get("bbox_y", center[1])) - 7),
        )
        _draw_label(annotated, label, text_position, color)

    for name, hole in result.get("base_holes", {}).items():
        color = CLASS_COLORS.get(name, (255, 255, 0))
        center = (
            int(round(float(hole["center_u"]))),
            int(round(float(hole["center_v"]))),
        )
        cv2.circle(annotated, center, 3, color, -1, cv2.LINE_AA)
        cv2.drawMarker(annotated, center, color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        depth_text = ""
        if show_depth_status and "depth_valid" in hole:
            depth_text = " depth=ok" if hole["depth_valid"] else " depth=invalid"
        _draw_label(
            annotated,
            f"{name} ({center[0]},{center[1]}){depth_text}",
            (center[0] + 8, center[1] - 8),
            color,
        )
    return annotated


def _draw_label(
    image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int]
) -> None:
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
    )
