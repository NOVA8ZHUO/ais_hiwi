#!/usr/bin/env python3
"""Stability gates and robust pose buffers for Franka vision nodes."""

from __future__ import annotations

from collections import defaultdict, deque
import math
import time

import numpy as np


def rotation_delta_deg(transform_a: np.ndarray, transform_b: np.ndarray) -> float:
    rotation = np.asarray(transform_a, dtype=np.float64)[:3, :3].T @ np.asarray(
        transform_b, dtype=np.float64
    )[:3, :3]
    value = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(value)))


def translation_delta_m(transform_a: np.ndarray, transform_b: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(transform_b, dtype=np.float64)[:3, 3]
            - np.asarray(transform_a, dtype=np.float64)[:3, 3]
        )
    )


def grasp_yaw_deg(points_base_m: np.ndarray) -> float:
    points = np.asarray(points_base_m, dtype=np.float64).reshape(3, 3)
    delta = points[1, :2] - points[0, :2]
    return float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))))


def angle_diff_deg(a: float, b: float) -> float:
    return abs(float((a - b + 180.0) % 360.0 - 180.0))


def circular_mean_deg(values: list[float]) -> float:
    radians = np.radians(np.asarray(values, dtype=np.float64))
    sin_sum = float(np.sum(np.sin(radians)))
    cos_sum = float(np.sum(np.cos(radians)))
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return float(values[-1])
    return float(math.degrees(math.atan2(sin_sum, cos_sum)))


class StabilityGate:
    """Require image and TF stability for several consecutive frames."""

    def __init__(
        self,
        min_stable_frames: int,
        image_motion_threshold_px: float,
        tf_motion_threshold_m: float,
        tf_rotation_threshold_deg: float,
        wait_after_motion_sec: float,
    ) -> None:
        self.min_stable_frames = max(1, int(min_stable_frames))
        self.image_motion_threshold_px = float(image_motion_threshold_px)
        self.tf_motion_threshold_m = float(tf_motion_threshold_m)
        self.tf_rotation_threshold_deg = float(tf_rotation_threshold_deg)
        self.wait_after_motion_sec = max(0.0, float(wait_after_motion_sec))
        self._last_tf: np.ndarray | None = None
        self._stable_count = 0
        self._last_motion_time = time.monotonic()
        self._last_reason = "waiting for stable frame"

    @property
    def stable_count(self) -> int:
        return self._stable_count

    @property
    def last_reason(self) -> str:
        return self._last_reason

    def clear(self, reason: str = "motion") -> None:
        self._stable_count = 0
        self._last_tf = None
        self._last_motion_time = time.monotonic()
        self._last_reason = reason

    def update(self, image_motion_score: float, transform_base_gripper: np.ndarray | None) -> bool:
        now = time.monotonic()
        if transform_base_gripper is None:
            self.clear("tf unavailable")
            return False

        image_stable = (
            np.isfinite(image_motion_score)
            and float(image_motion_score) <= self.image_motion_threshold_px
        )
        tf_stable = True
        if self._last_tf is not None:
            translation_jump = translation_delta_m(self._last_tf, transform_base_gripper)
            rotation_jump = rotation_delta_deg(self._last_tf, transform_base_gripper)
            tf_stable = (
                translation_jump <= self.tf_motion_threshold_m
                and rotation_jump <= self.tf_rotation_threshold_deg
            )
            if not tf_stable:
                self._last_reason = (
                    f"tf unstable: translation={translation_jump:.4f} m, "
                    f"rotation={rotation_jump:.2f} deg"
                )

        if not image_stable:
            self._stable_count = 0
            self._last_motion_time = now
            self._last_tf = np.asarray(transform_base_gripper, dtype=np.float64).copy()
            self._last_reason = f"image unstable: motion={float(image_motion_score):.3f}"
            return False
        if not tf_stable:
            self._stable_count = 0
            self._last_motion_time = now
            self._last_tf = np.asarray(transform_base_gripper, dtype=np.float64).copy()
            return False

        self._last_tf = np.asarray(transform_base_gripper, dtype=np.float64).copy()
        if now - self._last_motion_time < self.wait_after_motion_sec:
            self._last_reason = "waiting after motion"
            return False
        self._stable_count += 1
        if self._stable_count < self.min_stable_frames:
            self._last_reason = (
                f"waiting for stable frame: {self._stable_count}/{self.min_stable_frames}"
            )
            return False
        self._last_reason = "stable"
        return True


class LocalRobustPoseBuffer:
    """Reject local grasp outliers and return median multi-frame points."""

    def __init__(
        self,
        window_size: int,
        min_inlier_count: int,
        consensus_frames: int,
        low_confidence_consensus_frames: int,
        max_point_jump_m: float,
        max_midpoint_jump_m: float,
        max_yaw_jump_deg: float,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.min_inlier_count = max(1, int(min_inlier_count))
        self.consensus_frames = max(1, int(consensus_frames))
        self.low_confidence_consensus_frames = max(
            self.consensus_frames, int(low_confidence_consensus_frames)
        )
        self.max_point_jump_m = float(max_point_jump_m)
        self.max_midpoint_jump_m = float(max_midpoint_jump_m)
        self.max_yaw_jump_deg = float(max_yaw_jump_deg)
        self._history: deque[tuple[np.ndarray, float, str]] = deque(maxlen=self.window_size)

    def clear(self) -> None:
        self._history.clear()

    def reject_reason(self, points_base_m: np.ndarray) -> str | None:
        points = np.asarray(points_base_m, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(points)):
            return "invalid_point"
        if len(self._history) < self.min_inlier_count:
            return None
        samples = np.stack([item[0] for item in self._history], axis=0)
        median = np.median(samples, axis=0)
        point_jump = float(np.max(np.linalg.norm(points[:2] - median[:2], axis=1)))
        if point_jump > self.max_point_jump_m:
            return "point_jump"
        midpoint_jump = float(np.linalg.norm(points[2] - median[2]))
        if midpoint_jump > self.max_midpoint_jump_m:
            return "midpoint_jump"
        history_yaw = circular_mean_deg([item[1] for item in self._history])
        if angle_diff_deg(grasp_yaw_deg(points), history_yaw) > self.max_yaw_jump_deg:
            return "yaw_jump"
        return None

    def add(
        self,
        points_base_m: np.ndarray,
        confidence: str,
        required_frames: int | None = None,
    ) -> np.ndarray | None:
        points = np.asarray(points_base_m, dtype=np.float64).reshape(3, 3)
        self._history.append((points.copy(), grasp_yaw_deg(points), str(confidence)))
        required = (
            int(required_frames)
            if required_frames is not None
            else self.low_confidence_consensus_frames
            if str(confidence).lower() == "low"
            else self.consensus_frames
        )
        if len(self._history) < required:
            return None
        samples = np.stack([item[0] for item in list(self._history)[-required:]], axis=0)
        return np.median(samples, axis=0)


class GlobalRobustPoseBuffer:
    """Per-part robust position buffer for global detections."""

    def __init__(
        self,
        window_size: int,
        min_inlier_count: int,
        consensus_frames: int,
        max_position_jump_m: float,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.min_inlier_count = max(1, int(min_inlier_count))
        self.consensus_frames = max(1, int(consensus_frames))
        self.max_position_jump_m = float(max_position_jump_m)
        self._history: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def clear(self, key: str | None = None) -> None:
        if key is None:
            self._history.clear()
        else:
            self._history.pop(str(key), None)

    def reject_reason(self, key: str, point_base_m: np.ndarray) -> str | None:
        point = np.asarray(point_base_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(point)):
            return "invalid_position"
        history = self._history[str(key)]
        if len(history) < self.min_inlier_count:
            return None
        median = np.median(np.stack(tuple(history), axis=0), axis=0)
        if float(np.linalg.norm(point - median)) > self.max_position_jump_m:
            return "position_jump"
        return None

    def add(self, key: str, point_base_m: np.ndarray) -> np.ndarray | None:
        point = np.asarray(point_base_m, dtype=np.float64).reshape(3)
        history = self._history[str(key)]
        history.append(point.copy())
        if len(history) < self.consensus_frames:
            return None
        samples = np.stack(list(history)[-self.consensus_frames :], axis=0)
        return np.median(samples, axis=0)
