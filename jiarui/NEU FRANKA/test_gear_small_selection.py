#!/usr/bin/env python3
"""Deterministic tests for structure-gated Gear_small selection."""

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


def _depth():
    return np.full(SHAPE, 0.5, dtype=np.float32)


def test_all_red_candidates_are_checked_for_center_square_structure() -> None:
    color = np.zeros((*SHAPE, 3), dtype=np.uint8)
    # Smaller invalid red distractor: the old area-minimum rule selected this.
    cv2.circle(color, (180, 240), 38, (0, 0, 220), cv2.FILLED)
    # Larger valid Gear_small-style red body with a closed central square hole.
    cv2.circle(color, (400, 240), 62, (0, 0, 220), cv2.FILLED)
    cv2.rectangle(color, (382, 222), (418, 258), (0, 0, 0), cv2.FILLED)

    result, _, _ = detector.run_local_refinement(
        color, _depth(), INTRINSICS, "gear_small"
    )

    assert result["status"] == "ok"
    assert result["gear_candidate_count"] == 2
    assert result["gear_structured_candidate_count"] == 1
    assert result["gear_selection_reason"] == (
        "gear_small_center_square_structure_score_over_red_contours"
    )
    assert result["gear_selected_candidate_area_px"] == max(
        result["gear_candidate_areas_px"]
    )


def test_invalid_red_candidate_cannot_fall_back_to_warm_shaft() -> None:
    color = np.zeros((*SHAPE, 3), dtype=np.uint8)
    cv2.circle(color, (180, 220), 42, (0, 0, 220), cv2.FILLED)
    # Warm Shaft-like raised square elsewhere in the scene.
    cv2.rectangle(color, (285, 250), (365, 330), (90, 110, 190), cv2.FILLED)
    depth = _depth()
    depth[250:331, 285:366] = 0.47

    result, _, _ = detector.run_local_refinement(
        color, depth, INTRINSICS, "gear_small"
    )

    assert result["status"] == "failed"
    assert result["confidence"] == "low"
    assert result["gear_selection_reason"] == (
        "gear_small_structure_required_fail_closed"
    )
    assert result["gear_structured_candidate_count"] == 0


def main() -> None:
    tests = [
        test_all_red_candidates_are_checked_for_center_square_structure,
        test_invalid_red_candidate_cannot_fall_back_to_warm_shaft,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
