from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np


RGB_TABLE_INTERSECTION_SOURCE = "rgb_ray_table_intersection"
DEFAULT_CAMERA_FRAME_ID = "camera_color_optical_frame"


def _as_matrix(array_like: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    matrix = np.asarray(array_like, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def load_hand_eye_npz(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(Path(npz_path))
    if "T_gripper_camera" not in data:
        raise KeyError("hand-eye npz is missing T_gripper_camera")
    if "camera_matrix" not in data:
        raise KeyError("hand-eye npz is missing camera_matrix")
    if "dist_coeffs" not in data:
        raise KeyError("hand-eye npz is missing dist_coeffs")
    T_gripper_camera = _as_matrix(
        data["T_gripper_camera"], (4, 4), "T_gripper_camera"
    )
    camera_matrix = _as_matrix(data["camera_matrix"], (3, 3), "camera_matrix")
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
    if dist_coeffs.ndim > 2 or not np.all(np.isfinite(dist_coeffs)):
        raise ValueError("dist_coeffs must be a finite 1D or 2D array")
    return T_gripper_camera, camera_matrix, dist_coeffs


def _compute_table_intersection(
    u: float,
    v: float,
    K: Any,
    T_base_gripper: Any,
    T_gripper_camera: Any,
    z_table: float,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    uv = np.asarray([u, v], dtype=np.float64)
    if uv.shape != (2,) or not np.all(np.isfinite(uv)):
        raise ValueError("pixel center must contain two finite values")
    K_matrix = _as_matrix(K, (3, 3), "camera_matrix")
    T_bg = _as_matrix(T_base_gripper, (4, 4), "T_base_gripper")
    T_gc = _as_matrix(T_gripper_camera, (4, 4), "T_gripper_camera")
    z_table_value = float(z_table)
    if not np.isfinite(z_table_value):
        raise ValueError("z_table must be finite")

    ray_camera = np.linalg.inv(K_matrix) @ np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    if not np.all(np.isfinite(ray_camera)):
        raise ValueError("camera ray contains non-finite values")

    T_base_camera = T_bg @ T_gc
    R_base_camera = T_base_camera[:3, :3]
    o_base = T_base_camera[:3, 3]
    ray_base = R_base_camera @ ray_camera

    if abs(float(ray_base[2])) < 1e-6:
        raise ValueError("ray is nearly parallel to the table plane")

    scale = (z_table_value - float(o_base[2])) / float(ray_base[2])
    if scale <= 0.0:
        raise ValueError("table intersection lies behind the camera")

    p_base_on_table = o_base + scale * ray_base
    T_camera_base = np.linalg.inv(T_base_camera)
    p_camera_on_table_h = T_camera_base @ np.array(
        [p_base_on_table[0], p_base_on_table[1], p_base_on_table[2], 1.0],
        dtype=np.float64,
    )
    p_camera_on_table = p_camera_on_table_h[:3]
    if not np.all(np.isfinite(p_camera_on_table)):
        raise ValueError("camera-frame table intersection contains non-finite values")
    if float(p_camera_on_table[2]) <= 0.0:
        raise ValueError("camera-frame table intersection has non-positive z")

    return p_camera_on_table, p_base_on_table, z_table_value, [float(uv[0]), float(uv[1])]


def pixel_to_table_intersection_camera(
    u: float,
    v: float,
    K: Any,
    T_base_gripper: Any,
    T_gripper_camera: Any,
    z_table: float,
    camera_frame_id: str = DEFAULT_CAMERA_FRAME_ID,
) -> dict[str, Any]:
    """Project one RGB pixel ray to the table and return camera-frame coordinates only.

    The table plane is defined in robot base frame as base_z = z_table, so the
    current T_base_gripper is still required even though only camera-frame
    coordinates are returned.
    """
    result = {
        "center_px": [None, None],
        "camera_x_m": None,
        "camera_y_m": None,
        "camera_z_m": None,
        "camera_frame_id": camera_frame_id,
        "depth_valid": False,
        "depth_m": None,
        "depth_source": RGB_TABLE_INTERSECTION_SOURCE,
        "ray_table_intersection_valid": False,
        "ray_table_intersection_reason": None,
        "table_z_m": None,
        "center_camera_on_table_m": None,
        "used_depth_source": RGB_TABLE_INTERSECTION_SOURCE,
    }
    try:
        p_camera_on_table, _, z_table_value, center_px = _compute_table_intersection(
            u, v, K, T_base_gripper, T_gripper_camera, z_table
        )
        result["center_px"] = center_px
        result["table_z_m"] = z_table_value
        result["center_camera_on_table_m"] = [
            float(value) for value in p_camera_on_table
        ]
        result["camera_x_m"] = float(p_camera_on_table[0])
        result["camera_y_m"] = float(p_camera_on_table[1])
        result["camera_z_m"] = float(p_camera_on_table[2])
        result["depth_m"] = float(p_camera_on_table[2])
        result["depth_valid"] = True
        result["ray_table_intersection_valid"] = True
        result["ray_table_intersection_reason"] = None
        return result
    except (ValueError, np.linalg.LinAlgError) as exc:
        try:
            uv = np.asarray([u, v], dtype=np.float64)
            if uv.shape == (2,) and np.all(np.isfinite(uv)):
                result["center_px"] = [float(uv[0]), float(uv[1])]
        except (TypeError, ValueError):
            pass
        result["ray_table_intersection_reason"] = str(exc)
        return result


def pixel_to_table_intersection_camera_and_base(
    u: float,
    v: float,
    K: Any,
    T_base_gripper: Any,
    T_gripper_camera: Any,
    z_table: float,
) -> dict[str, Any]:
    result = pixel_to_table_intersection_camera(
        u,
        v,
        K,
        T_base_gripper,
        T_gripper_camera,
        z_table,
        camera_frame_id=DEFAULT_CAMERA_FRAME_ID,
    )
    result["center_base_on_table_m"] = None
    if not result["ray_table_intersection_valid"]:
        return result
    try:
        _, p_base_on_table, _, _ = _compute_table_intersection(
            u, v, K, T_base_gripper, T_gripper_camera, z_table
        )

        result["center_base_on_table_m"] = [float(value) for value in p_base_on_table]
        return result
    except (ValueError, np.linalg.LinAlgError) as exc:
        result["ray_table_intersection_reason"] = str(exc)
        result["ray_table_intersection_valid"] = False
        result["depth_valid"] = False
        result["camera_x_m"] = None
        result["camera_y_m"] = None
        result["camera_z_m"] = None
        result["depth_m"] = None
        return result


def add_camera_table_intersection_to_detections(
    detections: list[dict[str, Any]],
    K: Any,
    T_base_gripper: Any,
    T_gripper_camera: Any,
    z_table: float,
    camera_frame_id: str = DEFAULT_CAMERA_FRAME_ID,
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for detection in detections:
        item = copy.deepcopy(detection)
        item.update(
            pixel_to_table_intersection_camera(
                item.get("center_u"),
                item.get("center_v"),
                K,
                T_base_gripper,
                T_gripper_camera,
                z_table,
                camera_frame_id=camera_frame_id,
            )
        )
        augmented.append(item)
    return augmented


def add_camera_table_intersection_to_result_dict(
    result: dict[str, Any],
    K: Any,
    T_base_gripper: Any,
    T_gripper_camera: Any,
    z_table: float,
    camera_frame_id: str = DEFAULT_CAMERA_FRAME_ID,
) -> dict[str, Any]:
    augmented = copy.deepcopy(result)
    detections = augmented.get("detections")
    if isinstance(detections, list):
        augmented["detections"] = add_camera_table_intersection_to_detections(
            detections,
            K,
            T_base_gripper,
            T_gripper_camera,
            z_table,
            camera_frame_id=camera_frame_id,
        )
    return augmented


def add_table_intersection_to_detections(
    detections: list[dict[str, Any]],
    K: Any,
    T_base_gripper: Any,
    T_gripper_camera: Any,
    z_table: float,
) -> list[dict[str, Any]]:
    return add_camera_table_intersection_to_detections(
        detections, K, T_base_gripper, T_gripper_camera, z_table
    )


def add_table_intersection_to_result_dict(
    result: dict[str, Any],
    K: Any,
    T_base_gripper: Any,
    T_gripper_camera: Any,
    z_table: float,
) -> dict[str, Any]:
    return add_camera_table_intersection_to_result_dict(
        result, K, T_base_gripper, T_gripper_camera, z_table
    )
