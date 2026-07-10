#!/usr/bin/env python3
"""Camera profiles matched to the extracted global/local detector strategies.

These values are the RealSense option targets used by the validated notebooks.
They are stored here for launch/config consistency only. The ROS nodes in this
folder subscribe to camera topics and do not own the RealSense device, so this
module does not pretend that the settings have been applied. Apply the selected
profile in the ``realsense2_camera`` launch file or through its ROS parameter
services before each phase.
"""

from __future__ import annotations


GLOBAL_RGB_SETTINGS = {
    "enable_auto_exposure": False,
    "auto_exposure_priority": 0,
    "power_line_frequency": 1,
    "exposure": 260,
    "gain": 96,
    "enable_auto_white_balance": False,
    "white_balance": 3600,
    "brightness": 0,
    "contrast": 45,
    "saturation": 58,
    "sharpness": 35,
    "gamma": 300,
}

LOCAL_RGB_SETTINGS = {
    "enable_auto_exposure": False,
    "auto_exposure_priority": 0,
    "power_line_frequency": 1,
    "exposure": 200,
    "gain": 96,
    "enable_auto_white_balance": False,
    "white_balance": 4600,
    "brightness": 0,
    "contrast": 45,
    "saturation": 58,
    "sharpness": 35,
    "gamma": 300,
}

LOCAL_DEPTH_SETTINGS = {
    "enable_auto_exposure": False,
    "exposure": 6000,
    "gain": 16,
    "emitter_enabled": 1,
    "emitter_always_on": None,
    "laser_power": 150,
    "visual_preset": 4,
}

LOCAL_DEPTH_FILTER_SETTINGS = {
    "enable_spatial": True,
    "enable_temporal": True,
    "enable_hole_filling": False,
    "spatial_filter_magnitude": 1,
    "spatial_smooth_alpha": 0.4,
    "spatial_smooth_delta": 10,
    "temporal_smooth_alpha": 0.4,
    "temporal_smooth_delta": 20,
}

CAMERA_PROFILES = {
    "global_rgb": {
        "rgb": GLOBAL_RGB_SETTINGS,
        "depth": {},
        "depth_filters": {},
    },
    "local_rgbd": {
        "rgb": LOCAL_RGB_SETTINGS,
        "depth": LOCAL_DEPTH_SETTINGS,
        "depth_filters": LOCAL_DEPTH_FILTER_SETTINGS,
    },
}


def get_camera_profile(name: str) -> dict[str, dict[str, object]]:
    key = str(name).strip().lower()
    if key not in CAMERA_PROFILES:
        valid = ", ".join(sorted(CAMERA_PROFILES))
        raise KeyError(f"unknown camera profile {name!r}; valid profiles: {valid}")
    return CAMERA_PROFILES[key]


def apply_camera_profile_to_ros_parameters(*args, **kwargs) -> None:
    """Placeholder for RealSense ROS parameter-service integration.

    TODO: Implement this with an rclpy parameter client targeting the active
    ``realsense2_camera`` node when the launch/node names are finalized. Until
    then, apply ``GLOBAL_RGB_SETTINGS`` or ``LOCAL_RGB_SETTINGS``/
    ``LOCAL_DEPTH_SETTINGS`` in the RealSense launch configuration.
    """
    del args, kwargs
    raise NotImplementedError(
        "Camera profiles are packaged but not automatically applied. Configure "
        "realsense2_camera launch parameters or implement this parameter client."
    )
