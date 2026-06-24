#!/usr/bin/env python3
"""Offline single-image test for the global RGB/RGB-D detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("color_image", type=Path, help="Input color image")
    parser.add_argument(
        "--depth-npy",
        type=Path,
        help="Optional aligned-to-color depth NumPy file",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Multiply loaded depth values by this scale to obtain meters",
    )
    parser.add_argument("--fx", type=float, help="Color camera focal length fx")
    parser.add_argument("--fy", type=float, help="Color camera focal length fy")
    parser.add_argument("--ppx", type=float, help="Color camera principal point x")
    parser.add_argument("--ppy", type=float, help="Color camera principal point y")
    parser.add_argument(
        "--camera-frame-id",
        default="camera_color_optical_frame",
        help="Frame name stored with camera coordinates",
    )
    parser.add_argument(
        "--annotated-output",
        type=Path,
        default=Path("global_result_annotated.png"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("global_result.json"),
    )
    return parser


def run_offline(args: argparse.Namespace) -> dict[str, Any]:
    color_bgr = cv2.imread(str(args.color_image), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"Could not read color image: {args.color_image}")

    result, masks = detect_global_parts_bgr(color_bgr, return_debug=True)
    if args.depth_npy is not None:
        missing = [
            name
            for name in ("fx", "fy", "ppx", "ppy")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(
                "Depth input requires camera intrinsics: " + ", ".join(missing)
            )
        depth_m = np.asarray(np.load(args.depth_npy), dtype=np.float32)
        depth_m *= float(args.depth_scale)
        camera_info = {
            "fx": float(args.fx),
            "fy": float(args.fy),
            "ppx": float(args.ppx),
            "ppy": float(args.ppy),
            "frame_id": str(args.camera_frame_id),
        }
        result = add_camera_xyz_to_global_result(
            result, depth_m, camera_info, masks=masks
        )
    else:
        result = add_invalid_camera_fields(
            result,
            camera_frame_id=str(args.camera_frame_id),
            reason="no depth file supplied; result contains pixel coordinates only",
        )

    result = _stabilize_result_schema(result, str(args.camera_frame_id))
    annotated = draw_global_results(color_bgr, result)
    args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.annotated_output), annotated):
        raise OSError(f"Could not write annotated image: {args.annotated_output}")
    with args.json_output.open("w", encoding="utf-8") as output_file:
        json.dump(
            result,
            output_file,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    return result


def _stabilize_result_schema(
    result: dict[str, Any], camera_frame_id: str
) -> dict[str, Any]:
    """Match the stable JSON schema published by the ROS2 node."""
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
    result["frame_id"] = camera_frame_id
    result["num_detections"] = int(len(detections))
    result["num_base_holes"] = int(len(base_holes))
    result["num_valid_depth"] = int(
        sum(bool(item.get("depth_valid")) for item in detections)
        + sum(bool(item.get("depth_valid")) for item in base_holes.values())
    )
    result["missing_instances"] = sorted(EXPECTED_INSTANCES - found_instances)
    result["missing_base_holes"] = sorted(
        EXPECTED_BASE_HOLES - set(base_holes)
    )
    return result


def _clear_invalid_depth(item: dict[str, Any]) -> None:
    item["depth_valid"] = False
    item["depth_m"] = None
    item["camera_x_m"] = None
    item["camera_y_m"] = None
    item["camera_z_m"] = None


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        result = run_offline(args)
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
        return

    print(
        f"Detected {len(result['detections'])} parts and "
        f"{len(result['base_holes'])} base holes."
    )
    print(f"Annotated image: {args.annotated_output}")
    print(f"JSON result:     {args.json_output}")


if __name__ == "__main__":
    main()
