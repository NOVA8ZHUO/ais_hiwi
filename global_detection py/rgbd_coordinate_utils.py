"""Depth sampling and pixel-to-camera deprojection utilities.

All 3D coordinates produced here are in the RealSense color optical camera
frame, normally ``camera_color_optical_frame``. They are not robot base-frame
coordinates.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_CAMERA_FRAME_ID = "camera_color_optical_frame"


def deproject_pixel_to_camera(
    u: float,
    v: float,
    depth_m: float,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
) -> dict[str, float]:
    """Deproject one color pixel into the optical camera frame in meters."""
    values = (u, v, depth_m, fx, fy, ppx, ppy)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("deprojection inputs must be finite")
    if float(depth_m) <= 0.0:
        raise ValueError("depth_m must be positive")
    if float(fx) == 0.0 or float(fy) == 0.0:
        raise ValueError("fx and fy must be non-zero")
    z = float(depth_m)
    return {
        "camera_x_m": float((float(u) - float(ppx)) / float(fx) * z),
        "camera_y_m": float((float(v) - float(ppy)) / float(fy) * z),
        "camera_z_m": z,
    }


def _valid_depth_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values) & (values > 0.0)]


def get_median_depth_around_pixel(
    depth_image_m: np.ndarray,
    u: float,
    v: float,
    window_size: int = 7,
) -> tuple[float | None, int]:
    """Return median valid depth in a clipped square window around ``(u, v)``."""
    depth = _validate_depth_image(depth_image_m)
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    center_u = int(round(float(u)))
    center_v = int(round(float(v)))
    half = window_size // 2
    x0 = max(0, center_u - half)
    x1 = min(depth.shape[1], center_u + half + 1)
    y0 = max(0, center_v - half)
    y1 = min(depth.shape[0], center_v + half + 1)
    if x0 >= x1 or y0 >= y1:
        return None, 0
    valid = _valid_depth_values(depth[y0:y1, x0:x1])
    if valid.size == 0:
        return None, 0
    return float(np.median(valid)), int(valid.size)


def get_ring_median_depth(
    depth_image_m: np.ndarray,
    u: float,
    v: float,
    r_inner: float,
    r_outer: float,
) -> tuple[float | None, int]:
    """Return median valid depth from an annulus centered at pixel ``(u, v)``."""
    depth = _validate_depth_image(depth_image_m)
    r_inner = max(0.0, float(r_inner))
    r_outer = float(r_outer)
    if not math.isfinite(r_outer) or r_outer <= r_inner:
        raise ValueError("r_outer must be finite and greater than r_inner")

    center_u = float(u)
    center_v = float(v)
    x0 = max(0, int(math.floor(center_u - r_outer)))
    x1 = min(depth.shape[1], int(math.ceil(center_u + r_outer)) + 1)
    y0 = max(0, int(math.floor(center_v - r_outer)))
    y1 = min(depth.shape[0], int(math.ceil(center_v + r_outer)) + 1)
    if x0 >= x1 or y0 >= y1:
        return None, 0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance_squared = (xx - center_u) ** 2 + (yy - center_v) ** 2
    ring = (distance_squared >= r_inner**2) & (distance_squared <= r_outer**2)
    valid = _valid_depth_values(depth[y0:y1, x0:x1][ring])
    if valid.size == 0:
        return None, 0
    return float(np.median(valid)), int(valid.size)


def get_mask_median_depth(
    depth_image_m: np.ndarray, mask: np.ndarray
) -> tuple[float | None, int]:
    """Return median valid depth wherever a same-sized mask is non-zero."""
    depth = _validate_depth_image(depth_image_m)
    mask_array = np.asarray(mask)
    if mask_array.shape != depth.shape:
        raise ValueError(
            f"mask shape {mask_array.shape} does not match depth shape {depth.shape}"
        )
    valid = _valid_depth_values(depth[mask_array > 0])
    if valid.size == 0:
        return None, 0
    return float(np.median(valid)), int(valid.size)


def add_camera_xyz_to_global_result(
    result: dict[str, Any],
    depth_image_m: np.ndarray,
    camera_info: Any,
    masks: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Add aligned-depth camera coordinates while retaining every detection.

    ``depth_image_m`` must already be aligned to the color image and expressed
    in meters. ``camera_info`` may be a dict with ``fx/fy/ppx/ppy`` fields, a
    dict with a ROS-style ``k`` array, or a ``sensor_msgs/CameraInfo`` object.
    """
    output = copy.deepcopy(result)
    depth = _validate_depth_image(depth_image_m)
    fx, fy, ppx, ppy, camera_frame_id = _parse_camera_info(camera_info)
    expected_shape = (
        int(output.get("image_height", depth.shape[0])),
        int(output.get("image_width", depth.shape[1])),
    )
    if depth.shape != expected_shape:
        raise ValueError(
            f"depth shape {depth.shape} does not match color image shape {expected_shape}"
        )

    output["frame_id"] = camera_frame_id
    output["coordinate_type"] = "pixel_2d_and_camera_3d"
    status = output.setdefault("status", {"ok": True, "warnings": [], "errors": []})
    status.setdefault("warnings", [])
    status.setdefault("errors", [])

    for detection in output.get("detections", []):
        instance_name = str(detection.get("instance_name", ""))
        base_depth_mask = None
        if instance_name == "base" and masks:
            base_depth_mask = masks.get("base_surface_mask")
            if base_depth_mask is None:
                base_depth_mask = masks.get("silver_base_mask")
        if base_depth_mask is not None:
            depth_m, valid_count = get_mask_median_depth(
                depth, base_depth_mask
            )
            source = "base_mask_median"
        else:
            window_size = 11 if instance_name.startswith("gear_") else 7
            depth_m, valid_count = get_median_depth_around_pixel(
                depth,
                detection["center_u"],
                detection["center_v"],
                window_size=window_size,
            )
            source = "center_window_median"
        _set_camera_fields(
            detection,
            depth_m,
            valid_count,
            source,
            fx,
            fy,
            ppx,
            ppy,
            camera_frame_id,
        )

    for hole in output.get("base_holes", {}).values():
        radius = max(1.0, float(hole.get("radius_px", 1.0)))
        depth_m, valid_count = get_ring_median_depth(
            depth,
            hole["center_u"],
            hole["center_v"],
            r_inner=radius + 2.0,
            r_outer=radius + 10.0,
        )
        _set_camera_fields(
            hole,
            depth_m,
            valid_count,
            "ring_median_around_hole",
            fx,
            fy,
            ppx,
            ppy,
            camera_frame_id,
        )

    invalid_count = sum(
        not item.get("depth_valid", False) for item in output.get("detections", [])
    ) + sum(
        not item.get("depth_valid", False)
        for item in output.get("base_holes", {}).values()
    )
    if invalid_count:
        status["warnings"].append(
            f"{invalid_count} detection(s) have no valid aligned depth"
        )
    status["ok"] = not status.get("errors")
    return output


def add_invalid_camera_fields(
    result: dict[str, Any],
    camera_frame_id: str = DEFAULT_CAMERA_FRAME_ID,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return a pixel-only result with explicit invalid 3D fields."""
    output = copy.deepcopy(result)
    output["frame_id"] = str(camera_frame_id or DEFAULT_CAMERA_FRAME_ID)
    output["coordinate_type"] = "pixel_2d_and_camera_3d"
    for item in list(output.get("detections", [])) + list(
        output.get("base_holes", {}).values()
    ):
        item.update(
            {
                "depth_valid": False,
                "depth_m": None,
                "depth_valid_count": 0,
                "depth_source": None,
                "camera_x_m": None,
                "camera_y_m": None,
                "camera_z_m": None,
                "camera_frame_id": output["frame_id"],
            }
        )
    if reason:
        status = output.setdefault("status", {"ok": True, "warnings": [], "errors": []})
        warnings = status.setdefault("warnings", [])
        if reason not in warnings:
            warnings.append(reason)
    return output


def _set_camera_fields(
    item: dict[str, Any],
    depth_m: float | None,
    valid_count: int,
    depth_source: str,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
    camera_frame_id: str,
) -> None:
    valid = depth_m is not None and math.isfinite(depth_m) and depth_m > 0.0
    item["depth_valid"] = bool(valid)
    item["depth_m"] = float(depth_m) if valid else None
    item["depth_valid_count"] = int(valid_count)
    item["depth_source"] = str(depth_source)
    item["camera_frame_id"] = str(camera_frame_id)
    if not valid:
        item["camera_x_m"] = None
        item["camera_y_m"] = None
        item["camera_z_m"] = None
        return
    item.update(
        deproject_pixel_to_camera(
            item["center_u"], item["center_v"], depth_m, fx, fy, ppx, ppy
        )
    )


def _validate_depth_image(depth_image_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_image_m)
    if depth.ndim != 2:
        raise ValueError("depth_image_m must be a 2D array")
    if not np.issubdtype(depth.dtype, np.number):
        raise TypeError("depth_image_m must have a numeric dtype")
    return depth


def _parse_camera_info(camera_info: Any) -> tuple[float, float, float, float, str]:
    frame_id = DEFAULT_CAMERA_FRAME_ID
    if isinstance(camera_info, Mapping):
        if all(key in camera_info for key in ("fx", "fy", "ppx", "ppy")):
            fx = camera_info["fx"]
            fy = camera_info["fy"]
            ppx = camera_info["ppx"]
            ppy = camera_info["ppy"]
        else:
            k = camera_info.get("k", camera_info.get("K"))
            if k is None or len(k) < 9:
                raise ValueError("camera_info must provide fx/fy/ppx/ppy or a 3x3 K array")
            fx, fy, ppx, ppy = k[0], k[4], k[2], k[5]
        frame_id = str(camera_info.get("frame_id", frame_id) or frame_id)
    else:
        k = getattr(camera_info, "k", None)
        if k is None or len(k) < 9:
            raise ValueError("camera_info object must provide a ROS-style k array")
        fx, fy, ppx, ppy = k[0], k[4], k[2], k[5]
        header = getattr(camera_info, "header", None)
        frame_id = str(getattr(header, "frame_id", frame_id) or frame_id)

    intrinsics = tuple(float(value) for value in (fx, fy, ppx, ppy))
    if not all(math.isfinite(value) for value in intrinsics):
        raise ValueError("camera intrinsics must be finite")
    if intrinsics[0] <= 0.0 or intrinsics[1] <= 0.0:
        raise ValueError("fx and fy must be positive")
    return (*intrinsics, frame_id)
