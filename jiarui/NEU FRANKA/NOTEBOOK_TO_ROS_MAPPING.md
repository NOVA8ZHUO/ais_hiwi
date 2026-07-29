# Notebook-to-ROS mapping

The two notebooks are authoritative and are not modified. The exported modules record their SHA-256 hashes and preserve cell execution order.

## Global notebook

| Notebook cell / function | ROS file / function | Migration |
|---|---|---|
| cell 3 constants and `find_project_root` | `global_notebook_detector.py` same names | Exact constants; no YAML override |
| cell 5 `load_image`, `_auto_detect_white_table`, `create_work_area_mask` | same names | Exact; ROS calls the mask function on subscribed BGR |
| cell 7 segmentation and geometry helpers | same names | Exact |
| cell 9 `classify_red_part`, `_mask_distance_at_point`, `detect_red_parts` | same names | Exact |
| cell 11 shaft classification/scoring/deduplication | same names | Exact |
| cell 13 Base and three-hole detection | same names | Exact |
| cell 15 `draw_results`, `show_and_save_debug` | same names; ROS publishes returned debug arrays | Detection drawing exact; matplotlib UI is not run by ROS |
| cell 17 result serialization and offline `main` | same names | Exact helper code; offline main is not invoked by ROS |
| cell 22 `detect_live_frame_bgr` | same name; called by `GlobalPartDetector.detect` | Exact single-frame entry point |
| cells 23–24 table intersection | `global_table_intersection_utils.py`; adapter `_table_point` | Exact utility dependency and transform convention |
| cell 25 manual-pose link0 conversion | `GlobalVisionForceAdapter._process_frame` | ROS-only adaptation: fixed/manual pose replaced by image-time TF |
| cells 27–28 interactive diagnostics | README validation commands | Debug-only notebook cells; no detector behavior |

## Local notebook

| Notebook cell / function group | ROS file / function | Migration |
|---|---|---|
| cell 2 all thresholds/constants | `local_notebook_detector.py` same names | Exact; no YAML override |
| cells 4–5 RealSense capture/settings/filter helpers | same names plus ROS input documentation | Exact helpers retained; device ownership stays with RealSense ROS driver |
| cell 7 all depth/deprojection/top-surface helpers | same names | Exact |
| cell 9 RGB masks, contour helpers, failure result | same names | Exact |
| cell 11 `detect_base_holes_local` | same name | Exact full RGB/Hough/depth/CAD/refinement path |
| cell 13 legacy gear helpers/fallback and marker helpers | same names | Exact; retained because final cell calls these fallbacks |
| cell 15 original shaft detector | same name then captured as `_original_shaft_detector` | Exact notebook override semantics |
| cell 17 screw top-depth and marker paths | same names; final cell selects depth path | Exact notebook override semantics |
| cell 19 unified dispatcher and original drawing | same names | Exact; final drawing override follows cell 21 |
| cell 21 `compute_grasp_line_pose`, depth chains and final detectors | same names | Exact and executed after earlier definitions |
| `gear_small` branch | `_gear_small_square_grasp` | Exact center-square opposite-edge pair |
| `gear_large` branch | `_gear_large_square_cutout_grasp` | Exact square-to-cutout structure scoring |
| `gear_compound` branch | `_gear_compound_circle_grasp` | Exact center-circle, depth-anomaly and large-style-axis fallback |
| `shaft_small`, `shaft_large` | final `detect_shaft_grasp_local` + `_original_shaft_detector` | Exact; both use the same white segmentation, area and rectangularity limits. Small additionally uses a target-centred square ROI, near-square/border rejection, second-largest qualified candidate selection, and temporal selection between the two 90-degree-equivalent axes when the contour is nearly square; large retains the largest-area path. |
| `screw` | final `detect_screw_grasp_local` + `_detect_screw_grasp_depth_fallback` | Exact raw-depth top bar and centered short-axis line |
| `base` | final `detect_base_holes_local` wrapper | Exact holes plus image orientation metadata |
| cell 23 `save_local_outputs` | same name | Exact helper; import-time output-directory creation removed only to avoid side effects |
| cell 25 live UI | ROS synchronized callbacks and debug topics | ROS I/O adaptation; detector call is unchanged |
| cell 26 camera-to-link0 display | adapter `_base_geometry` | Manual pose replaced by image-time TF; same matrix direction |

## ROS-only files

- `vision_common.py`: synchronized RGB/depth/CameraInfo input, unit conversion, TF2, hand-eye loading, transforms, consensus and quaternions.
- `global_vision_force_adapter.py`: global result mapping, table intersection, per-name publication and debug publication.
- `local_vision_force_adapter.py`: target selection, exact notebook dispatch, point transform and publication.
- The two `*_node.py` files only import and start their adapters.

No notebook detector helper or active final branch is omitted. Notebook display loops, file-saving calls and manually pasted robot poses are not executed as ROS behavior; their functional ROS replacements are listed above.
