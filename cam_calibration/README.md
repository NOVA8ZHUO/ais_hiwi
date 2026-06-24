# Camera Calibration Workflow

Use the `xinyu_zhuo` Jupyter kernel and run the notebooks in order:

1. `01_depth_noise_fixed_camera.ipynb`
2. `02_charuco_depth_scale_calibration.ipynb`
3. `03_eye_in_hand_charuco_calibration.ipynb`
4. `04_robot_table_plane_system_validation.ipynb`
5. `05_part_task_specific_z_calibration.ipynb`

All paths are resolved from the Xinyu project root. Generated data is stored in
`cam_calibration/outputs`; cross-notebook results are stored in
`cam_calibration/outputs/shared`.

Before notebook 03, provide:

- `inputs/robot_poses.csv`

Notebook 03 detects the fixed ChArUco board in each image, solves the eye-in-hand
transform, and writes `config/hand_eye_result.json`.
It can directly read and save the RealSense 640x480 Color intrinsics when
`outputs/shared/camera_intrinsics.json` does not exist, so notebooks 01 and 02
are optional when the immediate goal is hand-eye calibration only.

Each CSV row must pair one image with the robot pose from the same stationary
sample. Notebook 03 also provides `save_hand_eye_sample(sample_id,
T_base_gripper)` for manual collection after the robot has stopped. Use 15–25
poses with substantial rotation around multiple axes; keep the ChArUco board
fixed throughout the experiment.

Before notebook 04, provide:

- `config/table_plane.json`, or select another plane input mode
- `inputs/rgbd_manifest.csv`

Before notebook 05, also provide:

- `config/part_heights.json`
- `inputs/current_robot_pose.json`

Depth arrays listed in `rgbd_manifest.csv` must already be aligned to the color
image. Raw `uint16` `.npy` files require a `depth_scale` column, normally
`0.001` for the existing project captures.

Install dependencies into the environment with:

```bash
conda run -n xinyu_zhuo python -m pip install -r cam_calibration/requirements.txt
conda run -n xinyu_zhuo python -m ipykernel install --user \
  --name xinyu_zhuo --display-name "Python (xinyu_zhuo)"
```

For mouse-based point selection in notebooks, run `%matplotlib widget` before
the preview/selection cells. If widget interaction is unavailable, select
`selection_mode="manual"` and enter pixel coordinates directly.
