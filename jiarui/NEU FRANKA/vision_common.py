"""Shared RGB-D, hand-eye, TF2, and pose utilities for vision adapters.

All translations and 3D points in this module use meters. Matrices follow the
column-vector convention ``p_target = T_target_source @ p_source``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import cv2
import numpy as np
import yaml

PART_NAME_MAP = {
    "base": "Base",
    "shaft_small": "Shaft_small",
    "shaft_large": "Shaft_large",
    "screw": "Screw",
    "gear_small": "Gear_small_handle",
    "gear_compound": "Gear_compound_handle",
    "gear_large": "Gear_large_handle",
}

BASE_HOLE_NAMES = ("hole_large", "hole_small", "hole_screw")

# ROS-only workspace isolation.  The notebooks remain unchanged: before their
# detector is called, the wrappers retain only the region enclosed by the
# largest connected black-table background.  Objects placed on the table are
# holes in that background component; filling its external contour deliberately
# keeps those objects without expanding into concave non-table regions.
BLACK_WORKSPACE_MAX_VALUE = 95
BLACK_WORKSPACE_OPEN_KERNEL = 5
BLACK_WORKSPACE_CLOSE_KERNEL = 15
BLACK_WORKSPACE_ERODE_PX = 3
BLACK_WORKSPACE_MIN_AREA_FRACTION = 0.15
BLACK_WORKSPACE_EXCLUDED_BGR = (0, 255, 0)
DEFAULT_HANDEYE_PATH = str(
    Path(__file__).resolve().parent.parent
    / "cam_calibration"
    / "outputs"
    / "03_hand_eye"
    / "hand_eye_result.npz"
)
_HoleT = TypeVar("_HoleT")


def black_background_workspace_mask(color_bgr: np.ndarray) -> np.ndarray:
    """Return the filled outer region of the largest black background component.

    A zero mask is returned when no credible black workspace exists.  This is
    fail-closed: the ROS nodes publish no visual target instead of silently
    accepting detections from outside the intended black tabletop.
    """
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("color_bgr must be a uint8 HxWx3 image")
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    black_seed = cv2.inRange(
        hsv,
        np.array([0, 0, 0], dtype=np.uint8),
        np.array([179, 255, BLACK_WORKSPACE_MAX_VALUE], dtype=np.uint8),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BLACK_WORKSPACE_OPEN_KERNEL, BLACK_WORKSPACE_OPEN_KERNEL),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BLACK_WORKSPACE_CLOSE_KERNEL, BLACK_WORKSPACE_CLOSE_KERNEL),
    )
    black_seed = cv2.morphologyEx(black_seed, cv2.MORPH_OPEN, open_kernel)
    black_seed = cv2.morphologyEx(black_seed, cv2.MORPH_CLOSE, close_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        black_seed, connectivity=8
    )
    if count <= 1:
        return np.zeros((height, width), dtype=np.uint8)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component_area = int(stats[label, cv2.CC_STAT_AREA])
    if component_area < int(height * width * BLACK_WORKSPACE_MIN_AREA_FRACTION):
        return np.zeros((height, width), dtype=np.uint8)

    component = np.zeros((height, width), dtype=np.uint8)
    component[labels == label] = 255
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros((height, width), dtype=np.uint8)
    # Fill only holes that are completely enclosed by this black component.
    # Padding guarantees that (0, 0) is exterior even when the black table
    # touches an image corner. Flood-filled complement pixels are connected to
    # the image boundary and therefore remain excluded. Non-black parts lying
    # wholly on the table are enclosed holes and are retained.
    padded = cv2.copyMakeBorder(
        component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    exterior_flood = padded.copy()
    flood_mask = np.zeros(
        (exterior_flood.shape[0] + 2, exterior_flood.shape[1] + 2),
        dtype=np.uint8,
    )
    cv2.floodFill(exterior_flood, flood_mask, (0, 0), 255)
    enclosed_holes = cv2.bitwise_not(exterior_flood)
    workspace = cv2.bitwise_or(padded, enclosed_holes)[1:-1, 1:-1]
    if BLACK_WORKSPACE_ERODE_PX > 0:
        size = 2 * BLACK_WORKSPACE_ERODE_PX + 1
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (size, size)
        )
        workspace = cv2.erode(workspace, erode_kernel, iterations=1)
    return workspace


def apply_black_background_workspace(
    color_bgr: np.ndarray,
    depth_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Mask ROS detector inputs to the black-table workspace only."""
    workspace = black_background_workspace_mask(color_bgr)
    filtered_color = np.asarray(color_bgr).copy()
    filtered_color[workspace == 0] = BLACK_WORKSPACE_EXCLUDED_BGR
    filtered_depth = None
    if depth_m is not None:
        filtered_depth = np.asarray(depth_m).copy()
        if filtered_depth.shape != workspace.shape:
            raise ValueError("depth_m and color_bgr must have matching dimensions")
        filtered_depth[workspace == 0] = 0.0
    return filtered_color, filtered_depth, workspace


# Exact acquisition values from local_rgbd_grasp_refinement.ipynb.  These are
# driver parameters, not detector thresholds.  Keep this list in notebook
# order: librealsense applies visual_preset last in the authoritative cell.
LOCAL_REALSENSE_ROS_PARAMETERS: tuple[tuple[str, Any], ...] = (
    ("rgb_camera.enable_auto_exposure", False),
    ("rgb_camera.auto_exposure_priority", False),
    ("rgb_camera.power_line_frequency", 1),
    ("rgb_camera.exposure", 220),
    ("rgb_camera.gain", 96),
    ("rgb_camera.enable_auto_white_balance", False),
    ("rgb_camera.white_balance", 4600),
    ("rgb_camera.brightness", 0),
    ("rgb_camera.contrast", 45),
    ("rgb_camera.saturation", 58),
    ("rgb_camera.sharpness", 35),
    ("rgb_camera.gamma", 300),
    ("depth_module.enable_auto_exposure", False),
    ("depth_module.exposure", 6000),
    ("depth_module.gain", 16),
    ("depth_module.emitter_enabled", 1),
    # emitter_always_on is None in the notebook and is intentionally skipped.
    ("depth_module.laser_power", 150.0),
    ("depth_module.visual_preset", 4),
    ("spatial_filter.enable", True),
    ("temporal_filter.enable", True),
    ("hole_filling_filter.enable", False),
    ("spatial_filter.filter_magnitude", 1),
    ("spatial_filter.filter_smooth_alpha", 0.4),
    ("spatial_filter.filter_smooth_delta", 10),
    ("spatial_filter.holes_fill", 0),
    ("temporal_filter.filter_smooth_alpha", 0.4),
    ("temporal_filter.filter_smooth_delta", 20),
)

# Exact Global live-view differences: exposure=260 and white_balance=3600.
# Every other fixed value inherits the Local notebook profile above.
GLOBAL_REALSENSE_ROS_PARAMETERS: tuple[tuple[str, Any], ...] = tuple(
    (
        name,
        260
        if name == "rgb_camera.exposure"
        else 3600
        if name == "rgb_camera.white_balance"
        else value,
    )
    for name, value in LOCAL_REALSENSE_ROS_PARAMETERS
)

# Backward-compatible alias for code that previously meant the Local profile.
FIXED_REALSENSE_ROS_PARAMETERS = LOCAL_REALSENSE_ROS_PARAMETERS


def assign_base_hole_names(
    holes: Sequence[_HoleT],
    get_u: Callable[[_HoleT], float],
    get_v: Callable[[_HoleT], float],
    get_minor_size: Callable[[_HoleT], float],
) -> dict[str, _HoleT]:
    """Apply the shared global/local semantic assignment for three base holes."""
    if len(holes) != 3:
        return {}
    horizontal = sorted(holes, key=get_u)
    shaft_holes = sorted(
        (horizontal[0], horizontal[2]),
        key=lambda item: (get_minor_size(item), -get_v(item)),
        reverse=True,
    )
    return {
        "hole_large": shaft_holes[0],
        "hole_small": shaft_holes[1],
        "hole_screw": horizontal[1],
    }

_PART_ALIASES = {
    **{key: key for key in PART_NAME_MAP},
    **{value.lower(): key for key, value in PART_NAME_MAP.items()},
    "gear_small_handle": "gear_small",
    "gear_compound_handle": "gear_compound",
    "gear_large_handle": "gear_large",
}


def normalize_part_key(name: str) -> str:
    """Return the notebook-style lower-case key for a supported part name."""
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in _PART_ALIASES:
        valid = ", ".join(PART_NAME_MAP.values())
        raise ValueError(f"Unsupported part name {name!r}. Expected one of: {valid}")
    return _PART_ALIASES[normalized]


def force_control_part_name(name: str) -> str:
    """Map either notebook or force_control spelling to force_control spelling."""
    return PART_NAME_MAP[normalize_part_key(name)]


@dataclass(frozen=True)
class RGBDFrame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: dict[str, float]
    stamp_sec: int = 0
    stamp_nanosec: int = 0
    motion_score: float = float("inf")
    stable_frame_count: int = 0
    is_stable: bool = False


class SpatialConsensusFilter:
    """Require several spatially consistent observations before publication."""

    def __init__(
        self,
        required_frames: int,
        tolerance_m: float,
        reset_distance_m: float | None = None,
    ) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be positive")
        if tolerance_m <= 0.0:
            raise ValueError("tolerance_m must be positive")
        self.required_frames = int(required_frames)
        self.tolerance_m = float(tolerance_m)
        self.reset_distance_m = float(
            reset_distance_m
            if reset_distance_m is not None
            else max(4.0 * tolerance_m, 0.03)
        )
        self._history: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.required_frames)
        )

    def clear(self, key: str | None = None) -> None:
        if key is None:
            self._history.clear()
        else:
            self._history.pop(str(key), None)

    def add(self, key: str, points_m: np.ndarray) -> np.ndarray | None:
        """Return averaged points only after a complete, tight consensus window."""
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points_m must be a finite Nx3 array")
        history = self._history[str(key)]
        if history:
            jump = np.linalg.norm(points - history[-1], axis=1)
            if float(np.max(jump)) > self.reset_distance_m:
                history.clear()
        history.append(points.copy())
        if len(history) < self.required_frames:
            return None
        samples = np.stack(tuple(history), axis=0)
        median = np.median(samples, axis=0)
        deviations = np.linalg.norm(samples - median[None, :, :], axis=2)
        if float(np.max(deviations)) > self.tolerance_m:
            return None
        return np.mean(samples, axis=0)


class RealSenseParameterEnforcer:
    """Set and read back notebook camera settings through ROS parameter services.

    The realsense2_camera process remains the sole device owner.  ``ready`` is
    false until every requested value has been accepted and verified, allowing
    callers to block pose publication rather than silently use camera defaults.
    """

    def __init__(
        self,
        node: Any,
        camera_node: str = "/camera/camera",
        enabled: bool = True,
        strict: bool = True,
        retry_sec: float = 2.0,
        on_ready: Callable[[], None] | None = None,
        parameters: Sequence[tuple[str, Any]] = LOCAL_REALSENSE_ROS_PARAMETERS,
        profile_name: str = "local",
    ) -> None:
        self._node = node
        self._enabled = bool(enabled)
        self._strict = bool(strict)
        self._retry_sec = max(0.25, float(retry_sec))
        self._on_ready = on_ready
        self._state = "waiting"
        self._future: Any = None
        self._last_attempt_ns = 0
        self._reported_waiting = False
        self.ready = not self._enabled
        self.camera_node = "/" + str(camera_node).strip("/")
        self._parameters = tuple(parameters)
        self._desired = dict(self._parameters)
        if len(self._desired) != len(self._parameters):
            raise ValueError("RealSense parameter profile contains duplicate names")
        self.profile_name = str(profile_name)

        if not self._enabled:
            node.get_logger().warn(
                "Fixed RealSense parameter enforcement is disabled"
            )
            return
        try:
            from rcl_interfaces.srv import (
                DescribeParameters,
                GetParameters,
                SetParameters,
            )
        except ImportError as exc:
            raise RuntimeError(
                "rcl_interfaces is required to enforce RealSense parameters"
            ) from exc
        self._get_service_type = GetParameters
        self._set_service_type = SetParameters
        self._describe_service_type = DescribeParameters
        self._set_client = node.create_client(
            SetParameters, f"{self.camera_node}/set_parameters"
        )
        self._get_client = node.create_client(
            GetParameters, f"{self.camera_node}/get_parameters"
        )
        self._describe_client = node.create_client(
            DescribeParameters, f"{self.camera_node}/describe_parameters"
        )
        self._timer = node.create_timer(0.25, self._tick)
        node.get_logger().info(
            "Enforcing notebook RealSense settings through "
            f"{self.camera_node} parameter services; profile={self.profile_name}; "
            "pose publication is gated"
        )

    @staticmethod
    def _matches(actual: Any, expected: Any) -> bool:
        if isinstance(expected, bool):
            return isinstance(actual, (bool, np.bool_)) and bool(actual) == expected
        if isinstance(expected, float):
            try:
                return math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1.0e-6
                )
            except (TypeError, ValueError):
                return False
        if isinstance(expected, int):
            return (
                not isinstance(actual, (bool, np.bool_))
                and isinstance(actual, (int, float, np.integer, np.floating))
                and math.isclose(
                    float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-6
                )
            )
        return actual == expected

    def _retry(self, message: str) -> None:
        self.ready = False
        self._state = "waiting"
        self._future = None
        self._last_attempt_ns = self._node.get_clock().now().nanoseconds
        log = self._node.get_logger().error if self._strict else self._node.get_logger().warn
        log(message + f"; retrying in {self._retry_sec:.1f}s")
        if not self._strict:
            self.ready = True

    def _tick(self) -> None:
        if not self._enabled or self.ready or self._future is not None:
            return
        now_ns = self._node.get_clock().now().nanoseconds
        if now_ns - self._last_attempt_ns < int(self._retry_sec * 1.0e9):
            return
        if (
            not self._set_client.service_is_ready()
            or not self._get_client.service_is_ready()
            or not self._describe_client.service_is_ready()
        ):
            if not self._reported_waiting:
                self._node.get_logger().warn(
                    "Waiting for RealSense parameter services at " + self.camera_node
                )
                self._reported_waiting = True
            return

        request = self._describe_service_type.Request()
        request.names = list(self._desired)
        self._state = "describing"
        self._future = self._describe_client.call_async(request)
        self._future.add_done_callback(self._after_describe)

    def _after_describe(self, future: Any) -> None:
        self._future = None
        try:
            from rcl_interfaces.msg import ParameterType
            from rclpy.parameter import Parameter

            response = future.result()
            if len(response.descriptors) != len(self._desired):
                self._retry("RealSense parameter description list is incomplete")
                return
            parameters = []
            failures = []
            for (name, expected), descriptor in zip(
                self._desired.items(), response.descriptors
            ):
                declared_type = int(descriptor.type)
                if declared_type == ParameterType.PARAMETER_BOOL:
                    value = bool(expected)
                elif declared_type == ParameterType.PARAMETER_INTEGER:
                    value = int(expected)
                elif declared_type == ParameterType.PARAMETER_DOUBLE:
                    value = float(expected)
                else:
                    failures.append(
                        f"{name}: absent or unsupported declared type {declared_type}"
                    )
                    continue
                parameters.append(
                    Parameter(name=name, value=value).to_parameter_msg()
                )
            if failures:
                self._retry("RealSense parameter discovery failed: " + "; ".join(failures))
                return
        except Exception as exc:
            self._retry(f"RealSense describe_parameters call failed: {exc}")
            return

        request = self._set_service_type.Request()
        request.parameters = parameters
        self._state = "setting"
        self._future = self._set_client.call_async(request)
        self._future.add_done_callback(self._after_set)

    def _after_set(self, future: Any) -> None:
        self._future = None
        try:
            response = future.result()
            failures = [
                f"{name}: {result.reason or 'rejected'}"
                for (name, _), result in zip(
                    self._parameters, response.results
                )
                if not result.successful
            ]
            if len(response.results) != len(self._parameters):
                failures.append("driver returned an incomplete result list")
            if failures:
                self._retry("RealSense setting failed: " + "; ".join(failures))
                return
        except Exception as exc:
            self._retry(f"RealSense set_parameters call failed: {exc}")
            return

        request = self._get_service_type.Request()
        request.names = list(self._desired)
        self._state = "verifying"
        self._future = self._get_client.call_async(request)
        self._future.add_done_callback(self._after_get)

    def _after_get(self, future: Any) -> None:
        self._future = None
        try:
            from rcl_interfaces.msg import Parameter as ParameterMessage
            from rclpy.parameter import Parameter

            response = future.result()
            if len(response.values) != len(self._desired):
                self._retry("RealSense readback returned an incomplete value list")
                return
            mismatches = []
            for (name, expected), value_message in zip(
                self._desired.items(), response.values
            ):
                actual = Parameter.from_parameter_msg(
                    ParameterMessage(name=name, value=value_message)
                ).value
                if not self._matches(actual, expected):
                    mismatches.append(f"{name}: expected {expected!r}, got {actual!r}")
            if mismatches:
                self._retry("RealSense readback mismatch: " + "; ".join(mismatches))
                return
        except Exception as exc:
            self._retry(f"RealSense get_parameters verification failed: {exc}")
            return

        self.ready = True
        self._state = "ready"
        self._node.get_logger().info(
            f"Verified {len(self._desired)} fixed {self.profile_name} "
            "RealSense parameters"
        )
        if self._on_ready is not None:
            self._on_ready()


class ROSAlignedRGBDCamera:
    """Synchronize RealSense ROS color/aligned-depth topics into RGBDFrame."""

    def __init__(
        self,
        node: Any,
        color_topic: str,
        aligned_depth_topic: str,
        camera_info_topic: str,
        warmup_frames: int = 15,
        sync_queue_size: int = 10,
        sync_slop_sec: float = 0.05,
        motion_threshold: float = 2.0,
        stable_frames_required: int = 10,
        expected_width: int = 640,
        expected_height: int = 480,
    ) -> None:
        try:
            from cv_bridge import CvBridge
            from message_filters import ApproximateTimeSynchronizer, Subscriber
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise RuntimeError(
                "cv_bridge, message_filters, and sensor_msgs are required for "
                "ROS RGB-D input"
            ) from exc
        if sync_queue_size < 1:
            raise ValueError("sync_queue_size must be positive")
        if sync_slop_sec < 0.0:
            raise ValueError("sync_slop_sec must be non-negative")
        if motion_threshold <= 0.0:
            raise ValueError("motion_threshold must be positive")
        if stable_frames_required < 1:
            raise ValueError("stable_frames_required must be positive")

        self._node = node
        self._bridge = CvBridge()
        self._latest: tuple[tuple[int, int], RGBDFrame] | None = None
        self._last_returned_stamp: tuple[int, int] | None = None
        self._warmup_frames = max(0, int(warmup_frames))
        self._synchronized_frame_count = 0
        self._motion_threshold = float(motion_threshold)
        self._stable_frames_required = int(stable_frames_required)
        self._expected_width = int(expected_width)
        self._expected_height = int(expected_height)
        self._previous_gray: np.ndarray | None = None
        self._stable_frame_count = 0
        self._color_sub = Subscriber(
            node, Image, str(color_topic), qos_profile=qos_profile_sensor_data
        )
        self._depth_sub = Subscriber(
            node, Image, str(aligned_depth_topic), qos_profile=qos_profile_sensor_data
        )
        self._info_sub = Subscriber(
            node, CameraInfo, str(camera_info_topic), qos_profile=qos_profile_sensor_data
        )
        self._synchronizer = ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub, self._info_sub],
            queue_size=int(sync_queue_size),
            slop=float(sync_slop_sec),
        )
        self._synchronizer.registerCallback(self._rgbd_callback)
        node.get_logger().info(
            "Waiting for synchronized ROS RGB-D: "
            f"color={color_topic}, depth={aligned_depth_topic}, "
            f"camera_info={camera_info_topic}"
        )

    def reset_after_reconfigure(self) -> None:
        """Restart warmup/motion accounting after camera parameters change."""
        self._synchronized_frame_count = 0
        self._previous_gray = None
        self._stable_frame_count = 0
        self._latest = None
        self._last_returned_stamp = None
        self._node.get_logger().info(
            "RealSense parameters verified; restarting RGB-D warmup"
        )

    def _rgbd_callback(
        self, color_message: Any, depth_message: Any, info_message: Any
    ) -> None:
        try:
            if (
                len(info_message.k) != 9
                or info_message.k[0] <= 0.0
                or info_message.k[4] <= 0.0
            ):
                raise ValueError("invalid synchronized color CameraInfo intrinsics")
            color_bgr = self._bridge.imgmsg_to_cv2(
                color_message, desired_encoding="bgr8"
            )
            depth_raw = self._bridge.imgmsg_to_cv2(
                depth_message, desired_encoding="passthrough"
            )
            if depth_message.encoding in ("16UC1", "mono16"):
                depth_m = np.asarray(depth_raw, dtype=np.float32) * 0.001
            elif depth_message.encoding == "32FC1":
                depth_m = np.asarray(depth_raw, dtype=np.float32)
            else:
                raise ValueError(
                    f"Unsupported aligned depth encoding: {depth_message.encoding}"
                )
            color_bgr = np.asarray(color_bgr)
            if color_bgr.shape[:2] != depth_m.shape:
                raise ValueError(
                    f"Color/depth size mismatch: {color_bgr.shape[:2]} vs {depth_m.shape}"
                )
            if color_bgr.shape[:2] != (
                self._expected_height,
                self._expected_width,
            ):
                raise ValueError(
                    "RGB-D resolution must match notebook tuning: expected "
                    f"{self._expected_width}x{self._expected_height}, got "
                    f"{color_bgr.shape[1]}x{color_bgr.shape[0]}"
                )
            if (
                int(info_message.width) != color_bgr.shape[1]
                or int(info_message.height) != color_bgr.shape[0]
            ):
                raise ValueError(
                    "Synchronized CameraInfo resolution does not match color image: "
                    f"{info_message.width}x{info_message.height} vs "
                    f"{color_bgr.shape[1]}x{color_bgr.shape[0]}"
                )
            intrinsics = {
                "fx": float(info_message.k[0]),
                "fy": float(info_message.k[4]),
                "ppx": float(info_message.k[2]),
                "ppy": float(info_message.k[5]),
            }
        except Exception as exc:
            self._node.get_logger().warning(f"RGB-D conversion failed: {exc}")
            return

        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            motion_score = float("inf")
            self._stable_frame_count = 0
        else:
            motion_score = float(
                cv2.mean(cv2.absdiff(gray, self._previous_gray))[0]
            )
            self._stable_frame_count = (
                self._stable_frame_count + 1
                if motion_score < self._motion_threshold
                else 0
            )
        self._previous_gray = gray

        self._synchronized_frame_count += 1
        if self._synchronized_frame_count <= self._warmup_frames:
            return
        stamp = color_message.header.stamp
        stamp_key = (int(stamp.sec), int(stamp.nanosec))
        self._latest = (
            stamp_key,
            RGBDFrame(
                color_bgr=color_bgr.copy(),
                depth_m=depth_m.copy(),
                intrinsics=intrinsics,
                stamp_sec=int(stamp.sec),
                stamp_nanosec=int(stamp.nanosec),
                motion_score=motion_score,
                stable_frame_count=self._stable_frame_count,
                is_stable=self._stable_frame_count >= self._stable_frames_required,
            ),
        )

    def capture(self) -> RGBDFrame | None:
        if self._latest is None or self._latest[0] == self._last_returned_stamp:
            return None
        self._last_returned_stamp = self._latest[0]
        return self._latest[1]


def valid_depth_values(
    values: np.ndarray,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[
        np.isfinite(values)
        & (values >= float(min_depth_m))
        & (values <= float(max_depth_m))
    ]


def median_depth_around_pixel(
    depth_image_m: np.ndarray,
    u: float,
    v: float,
    window_size: int = 7,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
) -> tuple[float | None, int]:
    """Return a robust aligned depth at a pixel and the valid sample count."""
    depth = np.asarray(depth_image_m)
    if depth.ndim != 2:
        raise ValueError("depth_image_m must be a 2D array in meters")
    size = max(1, int(window_size))
    size += int(size % 2 == 0)
    half = size // 2
    x = int(round(float(u)))
    y = int(round(float(v)))
    x0, x1 = max(0, x - half), min(depth.shape[1], x + half + 1)
    y0, y1 = max(0, y - half), min(depth.shape[0], y + half + 1)
    if x0 >= x1 or y0 >= y1:
        return None, 0
    values = valid_depth_values(depth[y0:y1, x0:x1], min_depth_m, max_depth_m)
    if values.size == 0:
        return None, 0
    return float(np.median(values)), int(values.size)


def deproject_pixel_with_depth(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: Mapping[str, float] | Any,
) -> np.ndarray | None:
    """Deproject one aligned color pixel to camera optical XYZ in meters."""
    if depth_m is None or not math.isfinite(float(depth_m)) or depth_m <= 0.0:
        return None
    if isinstance(intrinsics, Mapping):
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        ppx = float(intrinsics["ppx"])
        ppy = float(intrinsics["ppy"])
    else:
        fx, fy = float(intrinsics.fx), float(intrinsics.fy)
        ppx, ppy = float(intrinsics.ppx), float(intrinsics.ppy)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera intrinsics fx and fy must be positive")
    z = float(depth_m)
    return np.array([(float(u) - ppx) * z / fx, (float(v) - ppy) * z / fy, z])


def pixel_to_camera_point(
    u: float,
    v: float,
    depth_image_m: np.ndarray,
    intrinsics: Mapping[str, float] | Any,
    window_size: int = 7,
    min_valid_depth_points: int = 1,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
) -> np.ndarray | None:
    """Return camera_color_optical_frame XYZ, or None for invalid depth."""
    depth_m, count = median_depth_around_pixel(
        depth_image_m, u, v, window_size, min_depth_m, max_depth_m
    )
    if depth_m is None or count < int(min_valid_depth_points):
        return None
    return deproject_pixel_with_depth(u, v, depth_m, intrinsics)


def _vector3(value: Any, field_name: str) -> np.ndarray:
    if isinstance(value, Mapping):
        keys = ("x", "y", "z")
        return np.array([float(value[key]) for key in keys], dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 3:
        raise ValueError(f"{field_name} must contain x, y, z")
    return array


def _quaternion4(value: Any, field_name: str) -> np.ndarray:
    if isinstance(value, Mapping):
        keys = ("x", "y", "z", "w")
        quaternion = np.array([float(value[key]) for key in keys])
    else:
        quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.size != 4:
        raise ValueError(f"{field_name} must contain x, y, z, w")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError(f"{field_name} must not be a zero quaternion")
    return quaternion / norm


def quaternion_to_rotation_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = _quaternion4(quaternion_xyzw, "quaternion")
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to normalized [x, y, z, w]."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quaternion = np.array(
                [0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[2, 1] - matrix[1, 2]) / scale]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quaternion = np.array(
                [(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale,
                 (matrix[0, 2] - matrix[2, 0]) / scale]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quaternion = np.array(
                [(matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
                 (matrix[1, 0] - matrix[0, 1]) / scale]
            )
    return quaternion / np.linalg.norm(quaternion)


def make_transform_matrix(
    translation_xyz: Sequence[float], quaternion_xyzw: Sequence[float]
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(quaternion_xyzw)
    transform[:3, 3] = _vector3(translation_xyz, "translation")
    return transform


def _validate_handeye_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("T_gripper_camera must be a 4x4 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("T_gripper_camera contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("T_gripper_camera has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("T_gripper_camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("T_gripper_camera rotation determinant is not +1")
    return matrix


def load_handeye_transform(handeye_yaml_path: str | Path) -> np.ndarray:
    """Load T_gripper_camera from NPZ, YAML, or JSON calibration output.

    Accepted keys are ``translation``/``translation_m`` and
    ``quaternion``/``quaternion_xyzw``. A 4x4 ``T_gripper_camera`` is also
    accepted. NPZ files must contain a 4x4 ``T_gripper_camera`` array.
    """
    path = Path(handeye_yaml_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Hand-eye calibration does not exist: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "T_gripper_camera" not in archive.files:
                raise ValueError(
                    f"Hand-eye NPZ lacks T_gripper_camera; keys={archive.files}"
                )
            return _validate_handeye_matrix(archive["T_gripper_camera"])

    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"Hand-eye YAML/JSON must contain a mapping: {path}")
    if "handeye" in data and isinstance(data["handeye"], Mapping):
        data = data["handeye"]
    if "T_gripper_camera" in data:
        return _validate_handeye_matrix(data["T_gripper_camera"])
    translation = data.get("translation", data.get("translation_m"))
    quaternion = data.get("quaternion", data.get("quaternion_xyzw"))
    if translation is None or quaternion is None:
        raise ValueError(
            "Hand-eye YAML requires translation plus quaternion (xyzw), or "
            "a T_gripper_camera 4x4 matrix"
        )
    return make_transform_matrix(
        _vector3(translation, "translation"),
        _quaternion4(quaternion, "quaternion"),
    )


def transform_point_camera_to_base(
    point_camera_m: Sequence[float],
    transform_base_gripper: np.ndarray,
    transform_gripper_camera: np.ndarray,
) -> np.ndarray:
    point = np.asarray(point_camera_m, dtype=np.float64).reshape(3)
    transform_base_camera = (
        np.asarray(transform_base_gripper, dtype=np.float64)
        @ np.asarray(transform_gripper_camera, dtype=np.float64)
    )
    return (transform_base_camera @ np.r_[point, 1.0])[:3]


def transform_points_camera_to_base(
    points_camera_m: Iterable[Sequence[float]],
    transform_base_gripper: np.ndarray,
    transform_gripper_camera: np.ndarray,
) -> np.ndarray:
    points = np.asarray(list(points_camera_m), dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    points = points.reshape(-1, 3)
    transform_base_camera = (
        np.asarray(transform_base_gripper, dtype=np.float64)
        @ np.asarray(transform_gripper_camera, dtype=np.float64)
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (transform_base_camera @ homogeneous.T).T[:, :3]


def quaternion_from_roll_pitch_yaw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def downward_grasp_quaternion(
    left_base_m: Sequence[float],
    right_base_m: Sequence[float],
    gripper_axis_mode: str = "x_parallel",
) -> np.ndarray:
    """Return a tool-down quaternion aligned to the base-XY grasp line.

    ``x_parallel`` aligns the end-effector X axis with left-to-right;
    ``y_parallel`` aligns its Y axis. The Panda hand/tool convention and which
    physical finger axis should follow this line must be verified on the real
    robot before descent.
    """
    left = np.asarray(left_base_m, dtype=np.float64).reshape(3)
    right = np.asarray(right_base_m, dtype=np.float64).reshape(3)
    delta = right[:2] - left[:2]
    if float(np.linalg.norm(delta)) < 1e-8:
        raise ValueError("Left and right grasp points are coincident in base XY")
    line_yaw = math.atan2(float(delta[1]), float(delta[0]))
    mode = str(gripper_axis_mode).strip().lower()
    if mode == "x_parallel":
        tool_yaw = line_yaw
    elif mode == "y_parallel":
        # With roll=pi below, tool +Y appears at yaw-90 deg in base XY.
        tool_yaw = line_yaw + math.pi / 2.0
    else:
        raise ValueError("gripper_axis_mode must be x_parallel or y_parallel")
    # RPY(pi, 0, yaw) points the conventional tool +Z axis downward.
    return quaternion_from_roll_pitch_yaw(math.pi, 0.0, tool_yaw)


def planar_object_quaternion(
    left_base_m: Sequence[float],
    right_base_m: Sequence[float],
    gripper_axis_mode: str = "x_parallel",
) -> np.ndarray:
    """Return an object-style yaw quaternion for force_control.

    ``force_control.get_downward_orientation`` converts this planar object
    orientation into its tool-down command. Publishing an already tool-down
    quaternion would apply that conversion twice.
    """
    left = np.asarray(left_base_m, dtype=np.float64).reshape(3)
    right = np.asarray(right_base_m, dtype=np.float64).reshape(3)
    delta = right[:2] - left[:2]
    if float(np.linalg.norm(delta)) < 1e-8:
        raise ValueError("Left and right grasp points are coincident in base XY")
    line_yaw = math.atan2(float(delta[1]), float(delta[0]))
    mode = str(gripper_axis_mode).strip().lower()
    if mode == "x_parallel":
        object_yaw = line_yaw
    elif mode == "y_parallel":
        object_yaw = line_yaw + math.pi / 2.0
    else:
        raise ValueError("gripper_axis_mode must be x_parallel or y_parallel")
    return quaternion_from_roll_pitch_yaw(0.0, 0.0, object_yaw)


def transform_message_to_matrix(transform_message: Any) -> np.ndarray:
    transform = transform_message.transform
    translation = transform.translation
    rotation = transform.rotation
    return make_transform_matrix(
        [translation.x, translation.y, translation.z],
        [rotation.x, rotation.y, rotation.z, rotation.w],
    )


class TFTransformProvider:
    """Read the latest T_base_gripper through TF2 without raising to callers."""

    def __init__(self, node: Any, base_frame: str, ee_frame: str) -> None:
        try:
            from tf2_ros import Buffer, TransformListener
        except ImportError as exc:
            raise RuntimeError("tf2_ros is required to run the ROS2 vision nodes") from exc
        self._node = node
        self.base_frame = str(base_frame)
        self.ee_frame = str(ee_frame)
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, node)

    def lookup_base_gripper(
        self,
        timeout_sec: float = 0.2,
        stamp_sec: int | None = None,
        stamp_nanosec: int | None = None,
    ) -> np.ndarray | None:
        """Look up T_base_gripper at the RGB-D acquisition time when provided."""
        try:
            from rclpy.duration import Duration
            from rclpy.time import Time

            query_time = (
                Time(
                    nanoseconds=(
                        int(stamp_sec) * 1_000_000_000 + int(stamp_nanosec or 0)
                    )
                )
                if stamp_sec is not None
                else Time()
            )
            message = self._buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                query_time,
                timeout=Duration(seconds=float(timeout_sec)),
            )
            return transform_message_to_matrix(message)
        except Exception as exc:  # TF failures are expected during startup.
            self._node.get_logger().warning(
                f"TF lookup {self.base_frame} <- {self.ee_frame} "
                f"at image stamp {stamp_sec}.{int(stamp_nanosec or 0):09d} failed: {exc}"
            )
            return None
