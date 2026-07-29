#!/usr/bin/env python3
"""Execute authoritative notebook cells and exported detectors on identical frames."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import global_notebook_detector as global_ros
import local_notebook_detector as local_ros
import vision_common


def _notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_source(document: dict[str, Any], index: int) -> str:
    return "".join(document["cells"][index].get("source", []))


def _definition_source(document: dict[str, Any], index: int, name: str) -> str:
    source = _cell_source(document, index)
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def _execute_cells(
    path: Path,
    indexes: list[int],
    extracted_definitions: list[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    document = _notebook(path)
    namespace: dict[str, Any] = {"__name__": "notebook_alignment_reference"}
    for index in indexes:
        source = _cell_source(document, index)
        exec(compile(source, f"{path.name}:cell_{index}", "exec"), namespace)
    for index, name in extracted_definitions or []:
        source = _definition_source(document, index, name)
        exec(compile(source, f"{path.name}:cell_{index}:{name}", "exec"), namespace)
    return namespace


def _top_level_algorithm_nodes(source: str) -> list[str]:
    tree = ast.parse(source)
    return [
        ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))
    ]


def check_source_structure() -> None:
    checks = [
        (
            PROJECT / "global_rgb_part_detection.ipynb",
            [1, 3, 5, 7, 9, 11, 13, 15, 17],
            [(22, "detect_live_frame_bgr")],
            HERE / "global_notebook_detector.py",
            "global",
        ),
        (
            PROJECT / "local_rgbd_grasp_refinement.ipynb",
            [2, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            [(23, "save_local_outputs")],
            HERE / "local_notebook_detector.py",
            "local",
        ),
    ]
    for notebook_path, cells, definitions, module_path, label in checks:
        document = _notebook(notebook_path)
        expected = "\n\n".join(_cell_source(document, index) for index in cells)
        expected += "\n\n" + "\n\n".join(
            _definition_source(document, index, name)
            for index, name in definitions
        )
        expected_nodes = _top_level_algorithm_nodes(expected)
        actual_nodes = _top_level_algorithm_nodes(module_path.read_text(encoding="utf-8"))
        if expected_nodes != actual_nodes:
            raise AssertionError(f"{label}: exported top-level functions/constants differ")
        print(f"PASS {label} source structure: {len(actual_nodes)} functions/constants")


def _without_time(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_time(item)
            for key, item in value.items()
            if key != "timestamp_sec"
        }
    if isinstance(value, list):
        return [_without_time(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_time(item) for item in value)
    return value


def _assert_equal(reference: Any, exported: Any, path: str = "root") -> None:
    reference = _without_time(reference)
    exported = _without_time(exported)
    if isinstance(reference, np.ndarray) or isinstance(exported, np.ndarray):
        left, right = np.asarray(reference), np.asarray(exported)
        if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            raise AssertionError(f"{path}: arrays differ ({left.shape}/{right.shape})")
        return
    if isinstance(reference, dict) and isinstance(exported, dict):
        if set(reference) != set(exported):
            raise AssertionError(
                f"{path}: keys differ: reference-only={set(reference)-set(exported)}, "
                f"export-only={set(exported)-set(reference)}"
            )
        for key in reference:
            _assert_equal(reference[key], exported[key], f"{path}.{key}")
        return
    if isinstance(reference, (list, tuple)) and isinstance(exported, (list, tuple)):
        if len(reference) != len(exported):
            raise AssertionError(f"{path}: lengths differ")
        for index, (left, right) in enumerate(zip(reference, exported)):
            _assert_equal(left, right, f"{path}[{index}]")
        return
    if isinstance(reference, (float, np.floating)) or isinstance(exported, (float, np.floating)):
        if not np.isclose(float(reference), float(exported), rtol=0.0, atol=1e-12):
            raise AssertionError(f"{path}: {reference!r} != {exported!r}")
        return
    if reference != exported:
        raise AssertionError(f"{path}: {reference!r} != {exported!r}")


def check_global() -> None:
    notebook_path = PROJECT / "global_rgb_part_detection.ipynb"
    reference = _execute_cells(
        notebook_path,
        [1, 3, 5, 7, 9, 11, 13, 15, 17],
        [(22, "detect_live_frame_bgr")],
    )
    for stem in ("20260624_160418", "20260624_165150"):
        image_path = PROJECT / f"realsense_pic/realsense_color_{stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        ref_annotated, ref_results, ref_debug = reference["detect_live_frame_bgr"](image)
        ros_annotated, ros_results, ros_debug = global_ros.detect_live_frame_bgr(image)
        _assert_equal(ref_results, ros_results, f"global.{stem}.results")
        _assert_equal(ref_annotated, ros_annotated, f"global.{stem}.annotated")
        _assert_equal(ref_debug, ros_debug, f"global.{stem}.debug")
        names = [item.get("hole_name") or item.get("class_name") for item in ros_results]
        print(f"PASS global {stem}: {len(ros_results)} results {names}")


def check_local(targets: list[str]) -> None:
    notebook_path = PROJECT / "local_rgbd_grasp_refinement.ipynb"
    reference = _execute_cells(
        notebook_path,
        [2, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    )
    samples = {
        "base": "20260707_165106",
        # Real workspace frame containing both white shafts plus the silver
        # Base. This validates small/large rank selection and Base rejection;
        # the former 20260707_165106 sample was an ArUco calibration board.
        "shaft_small": "20260624_160418",
        "shaft_large": "20260624_160418",
        "screw": "20260610_141142",
        "gear_small": "20260706_135615",
        "gear_compound": "20260610_141258",
        "gear_large": "20260624_165150",
    }
    calibration = np.load(
        PROJECT / "cam_calibration/outputs/03_hand_eye/hand_eye_result.npz"
    )
    matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    intrinsics = {
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "ppx": float(matrix[0, 2]),
        "ppy": float(matrix[1, 2]),
    }
    for target in targets:
        stem = samples[target]
        color_path = PROJECT / f"realsense_pic/realsense_color_{stem}.png"
        depth_path = PROJECT / f"realsense_pic/realsense_depth_raw_{stem}.npy"
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color is None or not depth_path.is_file():
            raise FileNotFoundError(f"missing paired sample {stem}")
        depth = np.load(depth_path).astype(np.float32) * 0.001
        ref_result, ref_annotated, ref_debug = reference["run_local_refinement"](
            color, depth, intrinsics, target
        )
        ros_result, ros_annotated, ros_debug = local_ros.run_local_refinement(
            color, depth, intrinsics, target
        )
        _assert_equal(ref_result, ros_result, f"local.{target}.result")
        _assert_equal(ref_annotated, ros_annotated, f"local.{target}.annotated")
        _assert_equal(ref_debug, ros_debug, f"local.{target}.debug")
        if target == "shaft_small":
            areas = ros_result.get("shaft_candidate_areas_px", [])
            if len(areas) < 2:
                raise AssertionError(
                    "shaft_small real-frame test must retain both shaft candidates"
                )
            expected_rank = 2 if len(areas) >= 2 else 1
            if ros_result.get("shaft_selected_area_rank") != expected_rank:
                raise AssertionError(
                    "shaft_small must select rank 2 when interference candidates exist"
                )
            expected_area = sorted(areas, reverse=True)[expected_rank - 1]
            if not np.isclose(ros_result.get("shaft_area_px"), expected_area):
                raise AssertionError("shaft_small selected area does not match its rank")
        if target == "shaft_large":
            areas = ros_result.get("shaft_candidate_areas_px", [])
            if len(areas) < 2 or ros_result.get("shaft_selected_area_rank") != 1:
                raise AssertionError(
                    "shaft_large real-frame test must select the largest of both shafts"
                )
            if not np.isclose(ros_result.get("shaft_area_px"), max(areas)):
                raise AssertionError("shaft_large selected area is not the maximum")
        print(
            f"PASS local {target}: status={ros_result.get('status')} "
            f"confidence={ros_result.get('confidence')} "
            f"source={ros_result.get('yaw_source') or ros_result.get('base_hole_source')}"
        )


def check_ros_camera_parameters() -> None:
    expected_local = {
        **{f"rgb_camera.{key}": value for key, value in local_ros.FIXED_RGB_SETTINGS.items()},
        **{
            f"depth_module.{key}": value
            for key, value in local_ros.FIXED_DEPTH_SETTINGS.items()
            if value is not None
        },
        "spatial_filter.enable": local_ros.FIXED_DEPTH_FILTER_SETTINGS["enable_spatial"],
        "temporal_filter.enable": local_ros.FIXED_DEPTH_FILTER_SETTINGS["enable_temporal"],
        "hole_filling_filter.enable": local_ros.FIXED_DEPTH_FILTER_SETTINGS["enable_hole_filling"],
        "spatial_filter.filter_magnitude": local_ros.FIXED_DEPTH_FILTER_SETTINGS["spatial_filter_magnitude"],
        "spatial_filter.filter_smooth_alpha": local_ros.FIXED_DEPTH_FILTER_SETTINGS["spatial_smooth_alpha"],
        "spatial_filter.filter_smooth_delta": local_ros.FIXED_DEPTH_FILTER_SETTINGS["spatial_smooth_delta"],
        "spatial_filter.holes_fill": 0,
        "temporal_filter.filter_smooth_alpha": local_ros.FIXED_DEPTH_FILTER_SETTINGS["temporal_smooth_alpha"],
        "temporal_filter.filter_smooth_delta": local_ros.FIXED_DEPTH_FILTER_SETTINGS["temporal_smooth_delta"],
    }
    actual_local = dict(vision_common.LOCAL_REALSENSE_ROS_PARAMETERS)
    _assert_equal(expected_local, actual_local, "local_camera_parameters")

    global_settings = None
    for cell in _notebook(PROJECT / "global_rgb_part_detection.ipynb")["cells"]:
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name)
                and target.id == "GLOBAL_FIXED_RGB_SETTINGS"
                for target in node.targets
            ):
                global_settings = ast.literal_eval(node.value)
    if global_settings is None:
        raise AssertionError("Global notebook camera settings were not found")
    expected_global = dict(expected_local)
    expected_global.update(
        {f"rgb_camera.{name}": value for name, value in global_settings.items()}
    )
    actual_global = dict(vision_common.GLOBAL_REALSENSE_ROS_PARAMETERS)
    _assert_equal(expected_global, actual_global, "global_camera_parameters")
    print(
        "PASS ROS camera parameters: "
        f"Local={len(actual_local)}, Global={len(actual_global)} notebook-aligned values"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        nargs="*",
        default=[
            "base", "shaft_small", "shaft_large", "screw",
            "gear_small", "gear_compound", "gear_large",
        ],
    )
    args = parser.parse_args()
    check_source_structure()
    check_ros_camera_parameters()
    check_global()
    check_local(args.targets)


if __name__ == "__main__":
    main()
