from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


EXPECTED_CONVENTION = "p_gripper = T_gripper_camera @ p_camera"


def _load_npz(npz_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(npz_path))
    return {key: np.asarray(data[key]) for key in data.files}


def _load_json(json_path: str | Path | None) -> dict[str, Any] | None:
    if json_path is None:
        return None
    path = Path(json_path)
    if not path.exists():
        print(f"WARNING: JSON path does not exist: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_matrix(matrix: Any, precision: int = 9) -> str:
    array = np.asarray(matrix, dtype=np.float64)
    return np.array2string(array, precision=precision, suppress_small=False)


def _as_transform(T: Any, name: str) -> np.ndarray:
    matrix = np.asarray(T, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}")
    return matrix


def _as_K(K: Any) -> np.ndarray:
    matrix = np.asarray(K, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {matrix.shape}")
    return matrix


def _rotation_angle_deg(R: np.ndarray) -> float:
    value = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def _json_intrinsics_to_K(payload: dict[str, Any]) -> np.ndarray | None:
    intrinsics = payload.get("camera_intrinsics") or {}
    keys = ("fx", "fy", "cx", "cy")
    if not all(key in intrinsics for key in keys):
        return None
    return np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def debug_loaded_calibration(npz_path: str | Path, json_path: str | Path | None = None):
    """Print exactly what hand-eye/intrinsic calibration would be loaded."""
    npz_path = Path(npz_path)
    json_path = Path(json_path) if json_path is not None else None
    data = _load_npz(npz_path)
    payload = _load_json(json_path)

    print("\n========== Loaded Calibration ==========")
    print("NPZ path:", npz_path)
    print("JSON path:", json_path)
    print("NPZ keys:", sorted(data))

    T = np.asarray(data.get("T_gripper_camera"), dtype=np.float64)
    K = np.asarray(data.get("camera_matrix"), dtype=np.float64)
    dist = np.asarray(data.get("dist_coeffs"), dtype=np.float64)
    print("\nNPZ T_gripper_camera:\n", _fmt_matrix(T))
    print("NPZ camera_matrix:\n", _fmt_matrix(K))
    print("NPZ dist_coeffs:", dist.tolist())

    if payload is not None:
        intr = payload.get("camera_intrinsics") or {}
        print("\nJSON status:", payload.get("status"))
        print("JSON date_utc:", payload.get("date_utc"))
        print("JSON method:", payload.get("method"))
        print("JSON coordinate_convention:", payload.get("coordinate_convention"))
        print("JSON serial_number:", intr.get("serial_number"))
        print("JSON translation_m:", payload.get("translation_m"))
        print("JSON quality:", payload.get("quality"))
        print("JSON intrinsics fx/fy/cx/cy:", intr.get("fx"), intr.get("fy"), intr.get("cx"), intr.get("cy"))
        print("JSON intrinsics width/height/fps:", intr.get("width"), intr.get("height"), intr.get("fps"))

        T_json = np.asarray(payload.get("T_gripper_camera"), dtype=np.float64)
        K_json = _json_intrinsics_to_K(payload)
        T_match = T_json.shape == (4, 4) and np.allclose(T, T_json, atol=1e-12)
        K_match = K_json is not None and np.allclose(K, K_json, atol=1e-9)
        print("NPZ/JSON T_gripper_camera match:", T_match)
        print("NPZ/JSON camera_matrix match:", K_match)
        if not T_match:
            print("WARNING: NPZ and JSON T_gripper_camera differ.")
        if not K_match:
            print("WARNING: NPZ and JSON camera intrinsics differ.")
        if payload.get("coordinate_convention") != EXPECTED_CONVENTION:
            print(f"WARNING: coordinate_convention is not expected: {EXPECTED_CONVENTION}")

    print("\nFINAL T_gripper_camera used for calculation:\n", _fmt_matrix(T))
    print("FINAL K used for calculation:\n", _fmt_matrix(K))
    print("========================================\n")
    return T, K, dist


def compare_calibrations(
    new_npz: str | Path,
    new_json: str | Path,
    old_npz: str | Path,
    old_json: str | Path,
):
    new_data = _load_npz(new_npz)
    old_data = _load_npz(old_npz)
    new_payload = _load_json(new_json) or {}
    old_payload = _load_json(old_json) or {}
    T_new = np.asarray(new_data["T_gripper_camera"], dtype=np.float64)
    T_old = np.asarray(old_data["T_gripper_camera"], dtype=np.float64)
    K_new = np.asarray(new_data["camera_matrix"], dtype=np.float64)
    K_old = np.asarray(old_data["camera_matrix"], dtype=np.float64)
    dist_new = np.asarray(new_data["dist_coeffs"], dtype=np.float64)
    dist_old = np.asarray(old_data["dist_coeffs"], dtype=np.float64)

    T_rel = np.linalg.inv(T_old) @ T_new
    t_delta_mm = (T_new[:3, 3] - T_old[:3, 3]) * 1000.0
    intr_delta = {
        "fx": float(K_new[0, 0] - K_old[0, 0]),
        "fy": float(K_new[1, 1] - K_old[1, 1]),
        "cx": float(K_new[0, 2] - K_old[0, 2]),
        "cy": float(K_new[1, 2] - K_old[1, 2]),
    }
    new_serial = (new_payload.get("camera_intrinsics") or {}).get("serial_number")
    old_serial = (old_payload.get("camera_intrinsics") or {}).get("serial_number")

    print("\n========== New vs Old Calibration ==========")
    print("Relative rotation old->new [deg]:", _rotation_angle_deg(T_rel[:3, :3]))
    print("Translation new-old [mm]:", t_delta_mm.tolist(), "norm:", float(np.linalg.norm(t_delta_mm)))
    print("K new-old:", intr_delta)
    print("dist_coeffs match:", np.array_equal(dist_new, dist_old))
    print("RealSense serial new/old:", new_serial, old_serial, "match:", new_serial == old_serial)
    print("New quality:", new_payload.get("quality"))
    print("Old quality:", old_payload.get("quality"))
    print("============================================\n")
    return {
        "relative_rotation_deg": _rotation_angle_deg(T_rel[:3, :3]),
        "translation_delta_mm": t_delta_mm,
        "K_delta": intr_delta,
        "serial_match": new_serial == old_serial,
    }


def debug_runtime_realsense_serial(expected_serial: str | None = None):
    print("\n========== Runtime RealSense Serial ==========")
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        print(f"WARNING: pyrealsense2 is not available: {exc}")
        return []

    try:
        ctx = rs.context()
        devices = list(ctx.query_devices())
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        print("Connected RealSense serials:", serials)
        if not serials:
            print("WARNING: no RealSense device connected.")
        if expected_serial is not None and expected_serial not in serials:
            print("WARNING: runtime RealSense serial does not match calibration serial. Do not mix this calibration with current camera.")
        return serials
    except Exception as exc:
        print(f"WARNING: cannot query RealSense devices: {exc}")
        return []
    finally:
        print("==============================================\n")


def debug_runtime_realsense_color_intrinsics(
    expected_serial: str | None = None,
    expected_K: Any | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_fps: int | None = None,
    start_stream: bool = True,
):
    print("\n========== Runtime RealSense Color Intrinsics ==========")
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        print(f"WARNING: pyrealsense2 is not available: {exc}")
        print("========================================================\n")
        return None

    profile = None
    pipeline = None
    try:
        ctx = rs.context()
        devices = list(ctx.query_devices())
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        print("Connected RealSense serials:", serials)
        if expected_serial is not None:
            print("Expected calibration serial:", expected_serial)
            if expected_serial not in serials:
                print("WARNING: runtime RealSense serial does not match calibration serial. Do not mix this calibration with current camera.")
        if not start_stream:
            return {"serials": serials}

        pipeline = rs.pipeline()
        config = rs.config()
        if expected_serial:
            config.enable_device(str(expected_serial))
        width = int(expected_width or 640)
        height = int(expected_height or 480)
        fps = int(expected_fps or 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = pipeline.start(config)
        stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream_profile.get_intrinsics()
        runtime_K = np.array(
            [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        active_device = profile.get_device()
        runtime_serial = active_device.get_info(rs.camera_info.serial_number)
        print("Active color serial:", runtime_serial)
        print("Runtime color width/height:", intr.width, intr.height)
        print("Runtime color fx/fy/cx/cy:", intr.fx, intr.fy, intr.ppx, intr.ppy)
        print("Runtime color model:", intr.model)
        print("Runtime color coeffs:", list(intr.coeffs))
        print("Runtime K:\n", _fmt_matrix(runtime_K))
        if expected_K is not None:
            expected_K_array = _as_K(expected_K)
            delta = runtime_K - expected_K_array
            print("Runtime K - expected K:\n", _fmt_matrix(delta))
            print("K allclose atol=1e-6:", bool(np.allclose(runtime_K, expected_K_array, atol=1e-6)))
            if not np.allclose(runtime_K, expected_K_array, atol=1e-6):
                print("WARNING: runtime color intrinsics differ from loaded calibration K.")
        if expected_width is not None and int(intr.width) != int(expected_width):
            print("WARNING: runtime color width differs from calibration width.")
        if expected_height is not None and int(intr.height) != int(expected_height):
            print("WARNING: runtime color height differs from calibration height.")
        if expected_serial is not None and runtime_serial != str(expected_serial):
            print("WARNING: active RealSense serial does not match calibration serial.")
        return {
            "serial": runtime_serial,
            "serials": serials,
            "K": runtime_K,
            "width": int(intr.width),
            "height": int(intr.height),
            "coeffs": list(intr.coeffs),
            "model": str(intr.model),
        }
    except Exception as exc:
        print(f"WARNING: cannot read runtime RealSense color intrinsics: {exc}")
        return None
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                print(f"WARNING: could not stop temporary RealSense pipeline cleanly: {exc}")
        print("========================================================\n")


def debug_T_base_gripper(T_base_gripper: Any, name: str = "T_base_gripper"):
    T = _as_transform(T_base_gripper, name)
    R = T[:3, :3]
    det = float(np.linalg.det(R))
    orth = R.T @ R
    orth_ok = bool(np.allclose(orth, np.eye(3), atol=1e-4))
    det_ok = bool(np.isclose(det, 1.0, atol=1e-4))
    print(f"\n========== {name} Sanity ==========")
    print(name + ":\n", _fmt_matrix(T))
    print("translation:", T[:3, 3].tolist())
    print("translation z:", float(T[2, 3]))
    print("gripper x-axis in base:", R[:, 0].tolist())
    print("gripper y-axis in base:", R[:, 1].tolist())
    print("gripper z-axis in base:", R[:, 2].tolist())
    print("det(R):", det)
    print("R.T @ R:\n", _fmt_matrix(orth))
    print("orthogonal:", orth_ok)
    print("det close to 1:", det_ok)
    if T[2, 3] <= 0.0:
        print("WARNING: gripper z is not above table/base z=0.")
    if not orth_ok:
        print("WARNING: rotation is not orthogonal.")
    if not det_ok:
        print("WARNING: rotation determinant is not close to 1.")
    print("===================================\n")
    return {"det": det, "orthogonal": orth_ok, "det_ok": det_ok, "z": float(T[2, 3])}


def debug_franka_reshape(O_T_EE: Iterable[float], sample_pixels, K, T_gripper_camera, z_table=0.0):
    values = np.asarray(list(O_T_EE), dtype=np.float64)
    if values.size != 16:
        raise ValueError(f"O_T_EE must contain 16 values, got {values.size}")
    row_major = values.reshape(4, 4)
    col_major = values.reshape(4, 4, order="F")
    print("\n========== Franka O_T_EE Reshape Check ==========")
    for name, T in (("row-major reshape(4,4)", row_major), ("column-major reshape(4,4, order='F')", col_major)):
        debug_T_base_gripper(T, name)
        for u, v in sample_pixels:
            debug_pixel_ray_table_intersection(u, v, K, T, T_gripper_camera, z_table, label=name)
    print("=================================================\n")
    return row_major, col_major


def debug_quaternion_order(raw_quaternion, raw_order: str = "xyzw"):
    q = np.asarray(raw_quaternion, dtype=np.float64).reshape(4)
    print("\n========== Quaternion Order Check ==========")
    print("raw quaternion:", q.tolist())
    print("raw order:", raw_order)
    if raw_order.lower() == "wxyz":
        q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    elif raw_order.lower() == "xyzw":
        q_xyzw = q
    else:
        raise ValueError("raw_order must be 'xyzw' or 'wxyz'")
    norm = float(np.linalg.norm(q_xyzw))
    print("scipy/ROS xyzw quaternion:", q_xyzw.tolist())
    print("norm:", norm)
    if not np.isclose(norm, 1.0, atol=1e-3):
        print("WARNING: quaternion norm is not close to 1.")
    print("============================================\n")
    return q_xyzw / norm if norm > 0 else q_xyzw


def _pixel_ray_values(u, v, K, T_base_gripper, T_gripper_camera, z_table):
    K = _as_K(K)
    T_bg = _as_transform(T_base_gripper, "T_base_gripper")
    T_gc = _as_transform(T_gripper_camera, "T_gripper_camera")
    ray_camera = np.linalg.inv(K) @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    T_base_camera = T_bg @ T_gc
    origin_base = T_base_camera[:3, 3]
    ray_base = T_base_camera[:3, :3] @ ray_camera
    if abs(float(ray_base[2])) < 1e-9:
        lam = math.inf
    else:
        lam = (float(z_table) - float(origin_base[2])) / float(ray_base[2])
    P_camera = lam * ray_camera
    P_base = origin_base + lam * ray_base
    return T_base_camera, origin_base, ray_camera, ray_base, lam, P_camera, P_base


def debug_pixel_ray_table_intersection(
    u,
    v,
    K,
    T_base_gripper,
    T_gripper_camera,
    z_table: float = 0.0,
    label: str = "",
):
    T_base_camera, origin_base, ray_camera, ray_base, lam, P_camera, P_base = _pixel_ray_values(
        u, v, K, T_base_gripper, T_gripper_camera, z_table
    )
    print(f"\n========== Pixel Ray/Table Intersection {label} ==========")
    print("pixel:", (float(u), float(v)))
    print("K:\n", _fmt_matrix(K))
    print("T_base_gripper:\n", _fmt_matrix(T_base_gripper))
    print("T_gripper_camera:\n", _fmt_matrix(T_gripper_camera))
    print("T_base_camera:\n", _fmt_matrix(T_base_camera))
    print("origin_base:", origin_base.tolist())
    print("ray_camera:", ray_camera.tolist())
    print("ray_base:", ray_base.tolist())
    print("ray_base[2]:", float(ray_base[2]))
    print("lambda:", float(lam))
    print("P_camera:", P_camera.tolist())
    print("P_base:", P_base.tolist())
    print("P_base[2] - z_table:", float(P_base[2] - z_table))
    checks = {
        "lambda > 0": bool(lam > 0),
        "P_camera[2] > 0": bool(P_camera[2] > 0),
        "ray_base[2] < 0": bool(ray_base[2] < 0),
        "origin_base[2] > z_table": bool(origin_base[2] > z_table),
    }
    print("checks:", checks)
    if abs(float(ray_base[2])) < 1e-6:
        print("ERROR: ray is almost parallel to table plane.")
    if not checks["lambda > 0"]:
        print("ERROR: table intersection is behind the camera. Check T_base_gripper direction, T_gripper_camera direction, K, or z_table.")
    print("==========================================================\n")
    return {
        "T_base_camera": T_base_camera,
        "origin_base": origin_base,
        "ray_camera": ray_camera,
        "ray_base": ray_base,
        "lambda": lam,
        "P_camera": P_camera,
        "P_base": P_base,
        "checks": checks,
    }


def debug_transform_direction_candidates(u, v, K, T_base_gripper, T_gripper_camera, z_table: float = 0.0):
    T_bg = _as_transform(T_base_gripper, "T_base_gripper")
    T_gc = _as_transform(T_gripper_camera, "T_gripper_camera")
    candidates = {
        "T_base_gripper @ T_gripper_camera": T_bg @ T_gc,
        "T_base_gripper @ inv(T_gripper_camera)": T_bg @ np.linalg.inv(T_gc),
        "inv(T_base_gripper) @ T_gripper_camera": np.linalg.inv(T_bg) @ T_gc,
        "inv(T_base_gripper) @ inv(T_gripper_camera)": np.linalg.inv(T_bg) @ np.linalg.inv(T_gc),
    }
    print("\n========== Transform Direction Candidates ==========")
    ray_camera = np.linalg.inv(_as_K(K)) @ np.array([float(u), float(v), 1.0], dtype=np.float64)
    for name, T_bc in candidates.items():
        origin_base = T_bc[:3, 3]
        ray_base = T_bc[:3, :3] @ ray_camera
        lam = (float(z_table) - float(origin_base[2])) / float(ray_base[2]) if abs(ray_base[2]) > 1e-9 else math.inf
        P_camera = lam * ray_camera
        P_base = origin_base + lam * ray_base
        print("\nCandidate:", name)
        print("camera origin in base:", origin_base.tolist())
        print("ray direction in base:", ray_base.tolist())
        print("ray_base[2]:", float(ray_base[2]))
        print("lambda:", float(lam))
        print("P_base:", P_base.tolist())
        print("P_camera:", P_camera.tolist())
        print("in camera front:", bool(P_camera[2] > 0))
        print("on table:", bool(np.isfinite(P_base[2]) and abs(P_base[2] - z_table) < 1e-6))
    print("====================================================\n")


def debug_old_camera_point_new_transform(
    u,
    v,
    old_K,
    old_T_gripper_camera,
    new_K,
    new_T_gripper_camera,
    T_base_gripper,
    z_table: float = 0.0,
):
    print("\n========== Old Camera Point + New Transform ==========")
    old = debug_pixel_ray_table_intersection(
        u, v, old_K, T_base_gripper, old_T_gripper_camera, z_table, label="old full"
    )
    new = debug_pixel_ray_table_intersection(
        u, v, new_K, T_base_gripper, new_T_gripper_camera, z_table, label="new full"
    )
    P_camera_old_h = np.array([*old["P_camera"], 1.0], dtype=np.float64)
    P_base_mixed_h = _as_transform(T_base_gripper, "T_base_gripper") @ _as_transform(
        new_T_gripper_camera, "new_T_gripper_camera"
    ) @ P_camera_old_h
    P_base_old = old["P_base"]
    P_base_new = new["P_base"]
    P_base_mixed = P_base_mixed_h[:3]
    print("P_base_old_full:", P_base_old.tolist())
    print("P_base_new_full:", P_base_new.tolist())
    print("P_base_mixed old-camera + new-T:", P_base_mixed.tolist())
    print("new-old [mm]:", ((P_base_new - P_base_old) * 1000.0).tolist())
    print("mixed-old [mm]:", ((P_base_mixed - P_base_old) * 1000.0).tolist())
    print("mixed-new [mm]:", ((P_base_mixed - P_base_new) * 1000.0).tolist())
    print("=====================================================\n")
    return {"old": P_base_old, "new": P_base_new, "mixed": P_base_mixed}


def debug_z_table_sweep(u, v, K, T_base_gripper, T_gripper_camera, z_values=(0.0,)):
    print("\n========== z_table Sweep ==========")
    for z_table in z_values:
        result = debug_pixel_ray_table_intersection(
            u, v, K, T_base_gripper, T_gripper_camera, z_table=z_table, label=f"z_table={z_table}"
        )
        print("z_table:", z_table, "lambda:", result["lambda"], "P_base:", result["P_base"].tolist())
    print("===================================\n")


def default_sample_pixels_from_results(results, max_points: int = 3):
    pixels = [(320.0, 240.0)]
    if isinstance(results, list):
        for item in results:
            if item.get("center_u") is not None and item.get("center_v") is not None:
                pixels.append((float(item["center_u"]), float(item["center_v"])))
                break
        for item in results:
            if item.get("hole_name") and item.get("center_u") is not None and item.get("center_v") is not None:
                pixels.append((float(item["center_u"]), float(item["center_v"])))
                break
    return pixels[:max_points]


def run_global_chain_diagnostics(
    T_base_gripper,
    detections: list[dict[str, Any]] | None = None,
    current_npz: str | Path = "cam_calibration/outputs/03_hand_eye/hand_eye_result.npz",
    current_json: str | Path = "cam_calibration/outputs/03_hand_eye/hand_eye_result.json",
    new_npz: str | Path = "cam_calibration/outputs/03_hand_eye/hand_eye_result.npz",
    new_json: str | Path = "cam_calibration/outputs/03_hand_eye/hand_eye_result.json",
    old_npz: str | Path = "cam_calibration/outputs/03_hand_eye/hand_eye_result_alt.npz",
    old_json: str | Path = "cam_calibration/outputs/03_hand_eye/hand_eye_result_alt.json",
    z_table: float = 0.0,
):
    print("\n################ Global Chain Diagnostic Report ################")
    T_current, K_current, _ = debug_loaded_calibration(current_npz, current_json)
    current_payload = _load_json(current_json) or {}
    serial = (current_payload.get("camera_intrinsics") or {}).get("serial_number")
    debug_runtime_realsense_serial(serial)
    compare_calibrations(new_npz, new_json, old_npz, old_json)
    debug_T_base_gripper(T_base_gripper)

    new_data = _load_npz(new_npz)
    old_data = _load_npz(old_npz)
    T_new = new_data["T_gripper_camera"]
    K_new = new_data["camera_matrix"]
    T_old = old_data["T_gripper_camera"]
    K_old = old_data["camera_matrix"]

    pixels = default_sample_pixels_from_results(detections or [])
    print("Sample pixels:", pixels)
    for u, v in pixels:
        debug_pixel_ray_table_intersection(u, v, K_new, T_base_gripper, T_new, z_table, label="new K + new T")
        debug_pixel_ray_table_intersection(u, v, K_old, T_base_gripper, T_old, z_table, label="old K + old T")
        debug_pixel_ray_table_intersection(u, v, K_old, T_base_gripper, T_new, z_table, label="old K + new T")
        debug_pixel_ray_table_intersection(u, v, K_new, T_base_gripper, T_old, z_table, label="new K + old T")
        debug_old_camera_point_new_transform(u, v, K_old, T_old, K_new, T_new, T_base_gripper, z_table)
        debug_z_table_sweep(u, v, K_new, T_base_gripper, T_new, (z_table, z_table + 0.02, z_table - 0.02))
        debug_transform_direction_candidates(u, v, K_new, T_base_gripper, T_new, z_table)

    print("\nSummary checklist:")
    print("1. Confirm current NPZ/JSON path above is the intended new/old calibration.")
    print("2. Confirm runtime RealSense serial matches calibration serial.")
    print("3. Confirm T_base_gripper is the pose at the exact RGB capture time.")
    print("4. For a downward view, ray_base[2] should be negative and lambda positive.")
    print("5. If mixed old-camera + new-T works but new full ray fails, focus on K/T/pose used during pixel-ray intersection.")
    print("6. MoveIt -31 is more likely target pose/orientation/frame/IK/collision if vision base point is finite and plausible.")
    print("################################################################\n")
