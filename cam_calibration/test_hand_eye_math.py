from __future__ import annotations

import cv2
import numpy as np


def make_transform(rotation_vector, translation):
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=float))
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def invert_transform(transform):
    result = np.eye(4)
    rotation = transform[:3, :3]
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ transform[:3, 3]
    return result


def rotation_error_deg(expected, actual):
    delta = expected[:3, :3].T @ actual[:3, :3]
    value = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def test_opencv_hand_eye_direction():
    rng = np.random.default_rng(7)
    expected = make_transform(
        [0.18, -0.25, 0.12],
        [0.055, -0.018, 0.095],
    )
    base_target = make_transform(
        [0.05, 0.02, -0.10],
        [0.55, 0.03, 0.02],
    )

    base_gripper = []
    camera_target = []
    for _ in range(25):
        pose = make_transform(
            rng.uniform(-0.8, 0.8, 3),
            [
                0.35 + rng.uniform(-0.15, 0.15),
                rng.uniform(-0.25, 0.25),
                0.35 + rng.uniform(-0.12, 0.15),
            ],
        )
        target = invert_transform(expected) @ invert_transform(pose) @ base_target
        base_gripper.append(pose)
        camera_target.append(target)

    rotation, translation = cv2.calibrateHandEye(
        [pose[:3, :3] for pose in base_gripper],
        [pose[:3, 3] for pose in base_gripper],
        [pose[:3, :3] for pose in camera_target],
        [pose[:3, 3] for pose in camera_target],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    actual = np.eye(4)
    actual[:3, :3] = rotation
    actual[:3, 3] = np.asarray(translation).reshape(3)

    assert np.linalg.norm(actual[:3, 3] - expected[:3, 3]) < 1e-9
    assert rotation_error_deg(expected, actual) < 1e-6


if __name__ == "__main__":
    test_opencv_hand_eye_direction()
    print("hand-eye synthetic direction test passed")

