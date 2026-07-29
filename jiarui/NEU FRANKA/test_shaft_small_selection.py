#!/usr/bin/env python3
"""Deterministic tests for target-aware Shaft_small candidate selection."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import local_notebook_detector as detector


SHAPE = (480, 640)
INTRINSICS = {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}


def _frame(rectangles: list[tuple[tuple[int, int], tuple[int, int]]]):
    color = np.zeros((*SHAPE, 3), dtype=np.uint8)
    for corner_a, corner_b in rectangles:
        cv2.rectangle(color, corner_a, corner_b, (220, 220, 220), cv2.FILLED)
    depth = np.full(SHAPE, 0.5, dtype=np.float32)
    return color, depth


def _detect(rectangles):
    color, depth = _frame(rectangles)
    result, _, debug = detector.run_local_refinement(
        color, depth, INTRINSICS, "shaft_small"
    )
    return result, debug


def test_three_square_candidates_select_second_largest() -> None:
    result, debug = _detect([
        ((297, 213), (343, 267)),  # intended Shaft_small, second-largest
        ((365, 195), (435, 275)),  # nearby Shaft_large, largest
        ((250, 220), (285, 260)),  # smaller square distractor
        ((20, 190), (170, 225)),   # elongated Base-like distractor
    ])
    assert result["status"] == "ok"
    assert result["shaft_candidate_count"] == 3
    assert result["shaft_selected_area_rank"] == 2
    areas = result["shaft_candidate_areas_px"]
    assert result["shaft_area_px"] == sorted(areas, reverse=True)[1]
    assert result["shaft_selection_reason"] == (
        "target_square_roi_with_interference_select_second_largest_area"
    )
    midpoint = result["grasp_midpoint"]
    assert abs(midpoint["u"] - 320.0) < 1.0e-6
    assert abs(midpoint["v"] - 240.0) < 1.0e-6
    assert np.count_nonzero(debug["shaft_rejected_mask"]) > 0


def test_single_centered_small_candidate_is_retained() -> None:
    result, _ = _detect([((306, 222), (334, 258))])
    assert result["status"] == "ok"
    assert result["shaft_candidate_count"] == 1
    assert result["shaft_selection_reason"] == (
        "single_target_square_small_shaft_candidate"
    )


def test_off_center_base_like_candidate_is_rejected() -> None:
    result, debug = _detect([((20, 190), (170, 225))])
    assert result["status"] == "failed"
    assert result["shaft_candidate_count"] == 0
    reasons = result["shaft_rejected_candidates"][0]["rejection_reasons"]
    assert "aspect_ratio_above_small_limit" in reasons
    assert "center_outside_square_roi" in reasons
    assert np.count_nonzero(debug["shaft_rejected_mask"]) > 0


def test_close_large_pixel_square_is_retained() -> None:
    result, _ = _detect([((270, 185), (370, 295))])
    assert result["status"] == "ok"
    assert result["shaft_candidate_count"] == 1
    assert result["shaft_area_px"] > 6000


def test_two_nearby_shafts_select_second_largest() -> None:
    result, _ = _detect([
        ((365, 195), (435, 275)),  # nearby Shaft_large
        ((297, 213), (343, 267)),  # intended Shaft_small
    ])
    assert result["status"] == "ok"
    assert result["shaft_candidate_count"] == 2
    assert result["shaft_selected_area_rank"] == 2
    assert result["shaft_area_px"] == sorted(
        result["shaft_candidate_areas_px"], reverse=True
    )[1]


def test_perspective_stretched_small_shaft_is_retained() -> None:
    result, _ = _detect([((280, 215), (360, 265))])
    assert result["status"] == "ok"
    assert result["shaft_candidate_count"] == 1
    assert 1.35 < result["shaft_candidate_metrics"][0]["aspect_ratio"] <= 1.65


def test_warm_cast_is_accepted_only_for_small_shaft() -> None:
    color = np.zeros((*SHAPE, 3), dtype=np.uint8)
    # This stable warm cast fails the original neutral-white saturation limit
    # while remaining far less saturated than the red parts in the workspace.
    cv2.rectangle(color, (280, 210), (360, 270), (90, 110, 190), cv2.FILLED)
    depth = np.full(SHAPE, 0.5, dtype=np.float32)
    small, _, _ = detector.run_local_refinement(
        color, depth, INTRINSICS, "shaft_small"
    )
    large, _, _ = detector.run_local_refinement(
        color, depth, INTRINSICS, "shaft_large"
    )
    assert small["status"] == "ok"
    assert large["status"] == "failed"
    assert cv2.countNonZero(detector.make_red_mask(color)) == 0


def test_saturated_red_square_is_not_accepted_as_small_shaft() -> None:
    color = np.zeros((*SHAPE, 3), dtype=np.uint8)
    cv2.rectangle(color, (280, 210), (360, 270), (0, 0, 220), cv2.FILLED)
    depth = np.full(SHAPE, 0.5, dtype=np.float32)
    result, _, _ = detector.run_local_refinement(
        color, depth, INTRINSICS, "shaft_small"
    )
    assert result["status"] == "failed"


def test_near_square_axis_swap_uses_previous_equivalent_yaw() -> None:
    detector.reset_shaft_small_orientation_state()
    results = []
    for size in ((64.0, 60.0), (60.0, 64.0), (64.0, 60.0)):
        color = np.zeros((*SHAPE, 3), dtype=np.uint8)
        box = cv2.boxPoints(((320.0, 240.0), size, 30.0)).astype(np.int32)
        cv2.fillConvexPoly(color, box, (220, 220, 220))
        depth = np.full(SHAPE, 0.5, dtype=np.float32)
        result, _, _ = detector.run_local_refinement(
            color, depth, INTRINSICS, "shaft_small"
        )
        results.append(result)
    yaws = [item["shaft_orientation_selected_yaw_deg"] for item in results]
    assert all(item["shaft_orientation_ambiguous"] for item in results)
    assert results[0]["shaft_orientation_source"] == (
        "initial_square_axis_from_minAreaRect"
    )
    assert results[1]["shaft_orientation_source"] == (
        "nearest_previous_equivalent_square_axis"
    )
    assert max(abs(yaw - yaws[0]) for yaw in yaws[1:]) < 1.0
    assert all(item["depth_valid"] for item in results)


def main() -> None:
    tests = [
        test_three_square_candidates_select_second_largest,
        test_single_centered_small_candidate_is_retained,
        test_off_center_base_like_candidate_is_rejected,
        test_close_large_pixel_square_is_retained,
        test_two_nearby_shafts_select_second_largest,
        test_perspective_stretched_small_shaft_is_retained,
        test_warm_cast_is_accepted_only_for_small_shaft,
        test_saturated_red_square_is_not_accepted_as_small_shaft,
        test_near_square_axis_swap_uses_previous_equivalent_yaw,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
