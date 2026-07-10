# NEU FRANKA Vision Nodes

This folder contains the Franka-ready vision packaging built from the validated
notebooks without modifying the notebooks themselves.

## Files

- `global_rgb_part_detector_core.py`
  - Extracted from `global_rgb_part_detection.ipynb`.
  - Contains the notebook detector constants and functions.
- `local_rgbd_grasp_refiner_core.py`
  - Extracted from `local_rgbd_grasp_refinement.ipynb`.
  - Contains the notebook local RGB-D refinement constants and functions.
- `global_rgb_part_detection_node.py`
  - ROS2 node entry point for global RGB part detection.
  - Uses `global_rgb_part_detector_core.py`.
  - Publishes 3D poses through robot-base table-plane intersection for coarse
    global localization. This keeps the existing global node behavior and is
    not a depth-median 3D reproduction.
- `local_rgbd_grasp_refinement_node.py`
  - ROS2 node entry point for local RGB-D grasp refinement.
  - Uses `local_rgbd_grasp_refiner_core.py`.
  - Passes live ROS `CameraInfo` intrinsics into the notebook core and prefers
    the core's `camera_points` XYZ output before any fallback deprojection.
- `camera_profiles.py`
  - Stores the tuned global/local RealSense RGB/RGB-D option targets.
- `extract_notebook_cores.py`
  - Regenerates the two core modules from the notebooks.

## Runtime Model

Do not run the core modules directly for robot integration. Run the node files.
The nodes read ROS image/depth/CameraInfo/TF data, call the extracted core, and
publish base-frame `PoseStamped` messages.

## Stability And Robust Filtering

Both nodes wait until image motion and `T_base_gripper` are stable before calling
the notebook core. Motion clears the robust buffers immediately; a single
detection miss does not.

Shared stability parameters:

- `wait_after_motion_sec` default `0.5`
- `min_stable_frames` default `8`
- `image_motion_threshold_px` default `1.5`
- `tf_motion_threshold_m` default `0.002`
- `tf_rotation_threshold_deg` default `1.0`
- `max_missed_frames` default `5`

Local robust filtering:

- Uses notebook core `camera_points` XYZ first, then falls back to node-side
  depth-window deprojection only if core XYZ is unavailable.
- Rejects jumps in left/right/midpoint or base-frame grasp yaw.
- Publishes median left/right/midpoint after `consensus_frames` valid frames.
- Keeps `gripper_axis_mode` behavior unchanged; final orientation still uses
  `downward_grasp_quaternion(left_base, right_base, gripper_axis_mode)`.

Global robust filtering:

- Maintains independent history per part and per base hole.
- Rejects only the jumping part/hole in the current frame.
- Publishes median table-intersection position after `per_part_consensus_frames`
  valid frames.

## Camera Settings

The camera profiles are packaged here, but these subscriber nodes do not own the
RealSense device. Apply the matching profile in the RealSense driver or launch
layer before each phase:

- global phase: `global_rgb`
- local phase: `local_rgbd`

The nodes still use live ROS `CameraInfo` for intrinsics. Do not hard-code
`fx/fy/cx/cy` unless you also lock the exact stream resolution/profile.
`camera_profiles.py` stores the tuned settings only; it does not apply them to
the RealSense driver automatically.

## Notebook Safety

The extractor reads:

- `global_rgb_part_detection.ipynb`
- `local_rgbd_grasp_refinement.ipynb`

It writes only into this folder.
global_rgb_part_detection_node.py
  -> GlobalVisionForceAdapter
  -> NotebookGlobalDetector
  -> global_core.detect_live_frame_bgr(...)
  -> pixel ray + table_z_m 求交
  -> publish /vision/global/<part_name>/pose
local_rgbd_grasp_refinement_node.py
  -> LocalVisionForceAdapter
  -> NotebookLocalRefiner
  -> local_core.run_local_refinement(rgb, depth_m, intrinsics, target_key)
  -> 使用 core 输出的 left/right/midpoint camera XYZ
  -> camera frame -> base frame
  -> downward_grasp_quaternion(left_base, right_base, gripper_axis_mode)
  -> publish /vision/local/<target>/pose
