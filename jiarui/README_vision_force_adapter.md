# ROS2 Vision to Force-Control Adapter

This adapter turns the two vision notebooks into independent ROS2 nodes. It
does not modify or control `force_control.py`.

## Files and Topics

- `vision_common.py`: aligned RealSense RGB-D, depth deprojection, hand-eye
  loading, TF2 lookup, matrix transforms, part-name mapping, grasp quaternion.
- `global_vision_force_adapter.py`: global coarse part centers.
- `local_vision_force_adapter.py`: local left/right/midpoint refinement.
- `vision_config.yaml`: example ROS2 parameters.
- `/vision/global/<part_name>/pose`: global `geometry_msgs/msg/PoseStamped`.
- `/vision/local/<part_name>/pose`: local `geometry_msgs/msg/PoseStamped`.

Each part has its own topic, so `PoseStamped.header.frame_id` retains its
standard ROS meaning and equals the configured `base_frame` (`panda_link0` by
default). The topic prefixes are configurable with `topic_prefix`.

Published names are exactly:

```text
Base
Shaft_small
Shaft_large
Screw
Gear_small_handle
Gear_compound_handle
Gear_large_handle
hole_large
hole_small
hole_screw
```

The hole-to-part pairing is `hole_large` for `Shaft_large`, `hole_small` for
`Shaft_small`, and `hole_screw` for `Screw`.

## Transform Convention

The adapter uses column vectors and this exact chain:

```text
T_base_camera = T_base_gripper @ T_gripper_camera
p_base = T_base_camera @ p_camera
```

TF2 supplies the latest `T_base_gripper` by looking up
`base_frame <- ee_frame`. The YAML supplies the fixed hand-eye result
`T_gripper_camera`. No `CAMERA_HAND_OFFSET` or other fixed compensation is
applied. All translations, depth values, and output positions are meters.

Before robot motion, independently verify the transform direction by moving
the hand a known small distance and checking that a stationary object's
reported base position stays approximately fixed.

## Hand-Eye YAML

`handeye_yaml_path` accepts NPZ, YAML, and JSON calibration results. The
current configuration directly loads:

```text
/home/masais/Xinyu/cam_calibration/outputs/03_hand_eye/hand_eye_result.npz
```

The NPZ must contain a 4x4 `T_gripper_camera` array. A supported text format is:

```yaml
translation:
  x: -0.0515056834
  y: 0.0297196543
  z: 0.0599680742
quaternion:               # x, y, z, w
  x: 0.0
  y: 0.0
  z: 0.0
  w: 1.0
```

Replace the quaternion with the calibrated rotation; the identity above is
only a format example and is unsafe as a real calibration. The loader also
accepts this migration format:

```yaml
T_gripper_camera:
  - [r00, r01, r02, tx]
  - [r10, r11, r12, ty]
  - [r20, r21, r22, tz]
  - [0.0, 0.0, 0.0, 1.0]
```

The existing `hand_eye_result.json` uses the latter structure and can be
copied to a YAML file after confirming that its convention is
`p_gripper = T_gripper_camera @ p_camera`.

## ROS2 Setup

Run these files in the same ROS2 Python environment as `rclpy`, `tf2_ros`, and
`geometry_msgs`. That environment also needs `numpy`, `opencv-python`,
`pyyaml`, `cv_bridge`, `message_filters`, and `sensor_msgs`.

Start the RealSense ROS driver once, with depth aligned to color:

```bash
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true enable_sync:=true
```

The adapters default to `/camera/camera/color/image_raw`,
`/camera/camera/aligned_depth_to_color/image_raw`, and
`/camera/camera/color/camera_info`. Override the corresponding YAML parameters
if the camera namespace differs.

The files can be placed in a ROS2 Python package and registered as console
scripts. During initial testing they can also run directly:

```bash
source /opt/ros/humble/setup.bash
python3 Jiarui/global_vision_force_adapter.py --ros-args \
  --params-file Jiarui/vision_config.yaml
```

```bash
source /opt/ros/humble/setup.bash
python3 Jiarui/local_vision_force_adapter.py --ros-args \
  --params-file Jiarui/vision_config.yaml -p target_part:=Shaft_large
```

The RealSense ROS driver owns the device. Global and local adapters may both
subscribe to it; workflow logic can still enable them in separate stages.

## Stability and Publication Gates

The adapters do not publish the first detection after robot motion:

- Camera motion is evaluated at the synchronized RGB-D rate. Output stays
  blocked until 10 consecutive frames are below the configured motion score.
- TF2 is queried at the color-image timestamp, rather than using the latest
  end-effector transform.
- Several transformed detections must agree within `consensus_tolerance_m`.
  A large coordinate jump clears the pending consensus window.
- The Notebook-tuned detector requires 640x480 RGB-D; other resolutions are
  rejected instead of silently applying invalid pixel thresholds.
- Local detection is restricted to a configurable image-center ROI, matching
  the assumption that global motion has centered the requested target.
- Global low-confidence results are published by default because they are only
  used for coarse hover targeting, and the node reports `LOW-confidence` in the
  log. Low-confidence RGB-only local fallbacks are rejected by default. Enable
  local `publish_low_confidence` only for visualization, not robot descent.

## Stage 1: Global Node Only

1. Start the robot, MoveIt, robot-state publisher, and TF tree.
2. Confirm TF is available:

   ```bash
   ros2 run tf2_ros tf2_echo panda_link0 panda_hand
   ```

3. Start `realsense2_camera` with aligned depth, then start the global adapter.
4. Inspect a part message:

   ```bash
   ros2 topic echo /vision/global/Shaft_large/pose
   ```

5. Check the part name, plausible meter-scale XYZ, and stability while the
   robot is stationary. Do not command motion if TF or depth warnings occur.

## Stage 2: Global Move Test

1. Let the test/control adapter subscribe to the selected per-part global topic.
2. Use a large safe hover height and reduced speed.
3. Move only above the target and verify coarse XY alignment.
4. Keep automatic descent and force-control insertion disabled.

## Stage 3: Local Node Only

1. Keep `realsense2_camera` running; stop global only if the workflow requires it.
2. Park the robot above the selected target.
3. Start the local adapter, for example:

   ```bash
   python3 jiarui/local_vision_force_adapter.py --ros-args \
     --params-file jiarui/vision_config.yaml -p target_part:=Shaft_large
   ```

4. Inspect the refined pose:

   ```bash
   ros2 topic echo /vision/local/Shaft_large/pose
   ```

5. Confirm the logged left/right/midpoint pixels, camera XYZ, base XYZ, and
   quaternion. Invalid depth, failed TF, or missing grasp points suppresses
   publication instead of reusing stale data.

## Stage 4: Local Orientation Test

1. Move to a safe point above the local pose without descending.
2. Verify that the chosen tool axis is parallel to the physical grasp line.
3. If the physical finger convention is rotated by 90 degrees, change
   `gripper_axis_mode` between `x_parallel` and `y_parallel`.
4. The published orientation is already the tool-down grasp quaternion in the
   configured base frame. Conventional tool +Z points toward base -Z. Use it
   directly and do not call `force_control.get_downward_orientation()` again.
   Confirm this convention for the actual Panda tool/TCP.

## Stage 5: force_control Integration

After stages 1-4 pass, the control owner can replace `mock_poses` with cached
messages from the per-part global/local topics. Convert `Pose` to the
`Transform` shape currently returned by `get_object_pose`, or update that
method's internal cache while preserving its caller-facing result.

The adapters deliberately output `panda_link0` coordinates. `force_control`
currently sends MoveIt constraints in `world`. Before using a received pose,
the control integration must either verify that `world` and `panda_link0` are
coincident or transform the pose from `panda_link0` to `world`. Do not silently
interpret one frame as the other.

## Detector Boundaries

- Global: notebook-derived red gear/screw, black shaft, and silver base rules.
- Local base: three black holes inside the silver ROI; depth comes from the
  surrounding silver ring rather than the empty hole center. Select
  `hole_large`, `hole_small`, or `hole_screw` with `base_hole_target`. The
  semantic assignment follows the global detector: middle-u is screw; the two
  outer holes use ellipse-minor size to distinguish large from small.
- Local shaft: rotated rectangle and opposite long-edge midpoints.
- Local gear: two black markers first, depth top-surface or RGB geometry
  fallback otherwise.
- Local screw: top marker short axis first, rotated-rectangle fallback.

These remain rule-based detectors tuned to the current lighting and work area.
Use `debug_view:=true` only on a machine with a display, and retune/validate
under the exact true-robot lighting before enabling grasp descent.
