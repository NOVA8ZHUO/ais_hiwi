# ROS2 notebook-aligned vision adapters

真机分终端启动、Global/Local 阶段切换、固定观察位、抓取姿态和
rqt 调试窗口的完整步骤见 [OPERATION_GUIDE.md](OPERATION_GUIDE.md)。

This directory is a regenerated package whose only detection sources are:

- `../../global_rgb_part_detection.ipynb`
- `../../local_rgbd_grasp_refinement.ipynb`

The notebooks are read-only. Their recorded SHA-256 values are:

```text
5efe1218b3b6ed8c2b0e5440c678e969df769cbd4febb946f98cc2785bb85bef  global_rgb_part_detection.ipynb
2ac642cc27fbd050b79074d34672f3110f780fc8812ff9d8317c18733f96747f  local_rgbd_grasp_refinement.ipynb
```

See [NOTEBOOK_TO_ROS_MAPPING.md](NOTEBOOK_TO_ROS_MAPPING.md) for the cell/function mapping and [OLD_VS_NEW_DIFFS.md](OLD_VS_NEW_DIFFS.md) for the old-package audit.

## Files

- `global_notebook_detector.py`: mechanically extracted global constants and functions.
- `local_notebook_detector.py`: mechanically extracted local constants and functions in notebook execution order.
- `global_table_intersection_utils.py`: global notebook table-ray utility.
- `vision_common.py`: ROS RGB-D synchronization, units, TF, hand-eye, transforms and publication consensus.
- `global_vision_force_adapter.py`, `local_vision_force_adapter.py`: ROS wrappers only.
- `global_rgb_part_detection_node.py`, `local_rgbd_grasp_refinement_node.py`: entry points only.
- `vision_config.yaml`: complete node parameters.
- `test_notebook_alignment.py`: authoritative-cell versus exported-detector comparisons.

## Topics and names

Global publishes every supported result in a frame:

```text
/vision/global/Base/pose
/vision/global/Shaft_small/pose
/vision/global/Shaft_large/pose
/vision/global/Screw/pose
/vision/global/Gear_small_handle/pose
/vision/global/Gear_compound_handle/pose
/vision/global/Gear_large_handle/pose
/vision/global/hole_large/pose
/vision/global/hole_small/pose
/vision/global/hole_screw/pose
```

Local reads `target_part` each frame and publishes only that request. For `target_part:=Base`, `base_hole_target` selects `hole_large`, `hole_small` or `hole_screw`, and the corresponding local hole topic is published. Runtime switching with `ros2 param set` clears consensus before a new target can publish.

All pose messages use `geometry_msgs/msg/PoseStamped`. The header frame is `base_frame` (default `panda_link0`), and the stamp is the color-image acquisition stamp.

Debug arrays are published below `/vision/global/debug/<notebook_key>` and `/vision/local/debug/<notebook_key>`. `publication_gate` overlays the outer stability/TF/consensus state; `annotated` or `annotated_result` remains the raw single-frame notebook result.

## Tolerant outer publication gate

The notebook single-frame detectors are unchanged.  The ROS-only publication
gate tolerates unavoidable mount and lighting jitter with
`motion_threshold=5.0` and requires five stable frames.  Mild intermittent
motion, a missing detection, low confidence, or an invalid Local depth sample
does not erase earlier valid consensus samples; those frames simply contribute
nothing.  A target switch still clears Local consensus, and sustained motion
for 20 processed frames or a per-frame motion score of at least 15 clears all
stored consensus.  The spatial jump and tolerance checks remain active.

The corresponding debug text distinguishes
`mild image motion; consensus retained` from
`sustained/large motion; consensus reset`.  This change affects only when a ROS
pose may be published and does not alter any notebook mask, contour, threshold,
candidate, depth, geometry, classification, or fallback behavior.

## Coordinate chain and Franka pose

For every frame that can publish, TF2 queries:

```text
T_base_gripper = lookup(base_frame <- ee_frame, color_image.header.stamp)
T_base_camera = T_base_gripper @ T_gripper_camera
p_base = T_base_camera @ p_camera
```

`T_gripper_camera` means `p_gripper = T_gripper_camera @ p_camera` and is loaded from `handeye_path`. A failed timestamped lookup blocks that frame; no old TF is reused. All positions and translations are meters.

Global uses the notebook RGB-ray intersection with `base_z = table_z_m`. Local uses the exact depth source selected by the relevant notebook branch, including ring, component, mask, patch, neighbor or shifted-window medians. For non-Base local targets, the transformed `grasp_left -> grasp_right` line defines the published planar quaternion. Base holes have position only in the notebook and therefore publish identity orientation.

## RGB-D input

The node synchronizes all three messages with `message_filters`:

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

`16UC1`/`mono16` is multiplied by `0.001`; `32FC1` stays in meters. Color, depth and CameraInfo dimensions must all match. Resolution must be 640×480, and no resize is performed.

The notebook RealSense acquisition values are enforced at runtime through
the parameter services of the existing `realsense2_camera` process. The vision
node does not open the USB device a second time. It writes and reads back all 27
fixed values: RGB manual exposure/gain/white balance and image controls,
depth manual exposure/gain/emitter/laser/preset, spatial and temporal filters, and
disabled hole filling.

The two stages use separate notebook profiles:

| Parameter | Global notebook | Local notebook |
|---|---:|---:|
| `rgb_camera.exposure` | 260 | 220 |
| `rgb_camera.white_balance` | 3600 | 4600 |
| all other fixed RGB-D/filter values | Local value | Local value |

In particular, both stages keep spatial and temporal filtering enabled and hole
filling disabled. The Global adapter selects the Global profile in Python; the
Local adapter selects the Local profile. These values are not YAML-tunable
detector thresholds.

The defaults are deliberately fail-closed:

```yaml
realsense_camera_node: /camera/camera
enforce_fixed_camera_parameters: true
strict_camera_parameter_verification: true
camera_parameter_retry_sec: 2.0
```

No pose is published until every value is accepted and its readback equals the
notebook value. After successful verification, RGB-D warmup and motion stability
counting restart. If the camera node name is different, set
`realsense_camera_node` to the node shown by `ros2 node list`. If a camera model or
driver version does not expose one of the requested options, strict mode reports
the exact rejected parameter and keeps pose output blocked; do not disable strict
mode when notebook acquisition equivalence is required.

`emitter_always_on` is not sent because the authoritative notebook value is
`None`. Spatial and temporal filtering are enabled and the hole-filling filter is
explicitly disabled, matching the notebook.

## Run

### Independent stage runners

The recommended lifecycle entry points start RealSense first, wait for its
parameter service, start exactly one vision node, and close both process groups
on Ctrl-C or child failure:

```bash
python3 "ais_hiwi/jiarui/NEU FRANKA/run_global_vision_stage.py"
```

After Global poses have been cached and the robot is above the part, stop that
stage and start Local:

```bash
python3 "ais_hiwi/jiarui/NEU FRANKA/run_local_vision_stage.py" \
  --target-part Shaft_large --base-hole-target hole_large
```

The Local adapter can later be switched to `target_part=Base` and a semantic
hole without restarting the process. It clears its old consensus before any new
hole pose can publish.

For the current `visual_test_commander.py`, the Base-hole mapping is deliberately
limited to the three physical insertions:

```text
Shaft_small -> hole_small
Shaft_large -> hole_large
Screw       -> hole_screw
```

The commander caches both the part and Base during Global, uses the Local grasp
pose, carries the part over the cached Base, switches the running Local node to
`Base` plus the mapped hole, and waits for a fresh consensus hole pose. It no
longer adds a fixed camera/hand offset to Global base-frame coordinates. Physical
descent/release is disabled by default; first validate alignment above the hole:

```bash
python3 ais_hiwi/jiarui/visual_test_commander.py --ros-args \
  -p target_part:=Shaft_large \
  -p execute_assembly_descent:=false
```

Only after the above-hole test and force-control safety checks should physical
descent be explicitly enabled. Gears are rejected by this Base-hole test path;
their assembly targets are shafts, not Base holes.

### Manual launch

```bash
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true enable_sync:=true
```

The camera must be running before the vision node. Verification can be inspected
with:

```bash
ros2 param get /camera/camera rgb_camera.exposure
ros2 param get /camera/camera depth_module.exposure
ros2 param get /camera/camera spatial_filter.enable
ros2 param get /camera/camera temporal_filter.enable
ros2 param get /camera/camera hole_filling_filter.enable
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
python3 "ais_hiwi/jiarui/NEU FRANKA/global_rgb_part_detection_node.py" \
  --ros-args --params-file "ais_hiwi/jiarui/NEU FRANKA/vision_config.yaml"
```

```bash
source /opt/ros/humble/setup.bash
python3 "ais_hiwi/jiarui/NEU FRANKA/local_rgbd_grasp_refinement_node.py" \
  --ros-args --params-file "ais_hiwi/jiarui/NEU FRANKA/vision_config.yaml" \
  -p target_part:=Shaft_large
```

Checks:

```bash
ros2 topic echo /vision/global/Shaft_large/pose
ros2 topic echo /vision/local/Shaft_large/pose
ros2 param set /local_vision_force_adapter target_part Gear_large_handle
```

Move the robot while observing a stationary part. After the stability gate reopens, the base-frame position should remain approximately fixed. Stop if it follows the camera, which indicates a transform direction/time problem.

## Static alignment test

The test executes the authoritative notebook cell source directly in an isolated namespace, runs the exported module on the same paired frames, and recursively compares result dictionaries, contours, selected points, depth/source/confidence data, annotated images and every debug mask. Only `timestamp_sec` is excluded; numeric comparisons otherwise use an absolute tolerance of `1e-12`, and image/contour arrays require exact equality.

```bash
MPLCONFIGDIR=/tmp/mpl python3 \
  "ais_hiwi/jiarui/NEU FRANKA/test_notebook_alignment.py"
```

Pure import and syntax checks:

```bash
python3 -m py_compile \
  "ais_hiwi/jiarui/NEU FRANKA/vision_common.py" \
  "ais_hiwi/jiarui/NEU FRANKA/global_vision_force_adapter.py" \
  "ais_hiwi/jiarui/NEU FRANKA/global_rgb_part_detection_node.py" \
  "ais_hiwi/jiarui/NEU FRANKA/local_vision_force_adapter.py" \
  "ais_hiwi/jiarui/NEU FRANKA/local_rgbd_grasp_refinement_node.py"
```

ROS runtime topic/TF tests require a sourced ROS2 environment, an active RealSense driver and the real Franka TF tree; they cannot be replaced by mock poses.
