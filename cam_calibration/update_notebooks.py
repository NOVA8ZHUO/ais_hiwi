from __future__ import annotations

import json
from pathlib import Path

import nbformat


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
KERNEL = {
    "display_name": "Python (xinyu_zhuo)",
    "language": "python",
    "name": "xinyu_zhuo",
}
LANGUAGE_INFO = {"name": "python", "version": "3.10.20"}


def replace(source: str, old: str, new: str, notebook: Path) -> str:
    if old not in source:
        raise RuntimeError(f"Expected text not found in {notebook.name}: {old[:80]!r}")
    return source.replace(old, new)


def update_code_cells(path: Path, replacements: list[tuple[str, str]]) -> None:
    notebook = nbformat.read(path, as_version=4)
    for old, new in replacements:
        found = False
        for cell in notebook.cells:
            if cell.cell_type == "code" and old in cell.source:
                cell.source = replace(cell.source, old, new, path)
                found = True
                break
        if not found:
            if any(
                cell.cell_type == "code" and new in cell.source
                for cell in notebook.cells
            ):
                continue
            raise RuntimeError(f"No code cell matched in {path.name}: {old[:80]!r}")
    notebook.metadata.kernelspec = KERNEL
    notebook.metadata.language_info = LANGUAGE_INFO
    nbformat.write(notebook, path)


def set_code_cell(path: Path, marker: str, source: str) -> None:
    notebook = nbformat.read(path, as_version=4)
    matches = [
        cell for cell in notebook.cells
        if cell.cell_type == "code" and marker in cell.source
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one cell containing {marker!r} in {path.name}, got {len(matches)}"
        )
    matches[0].source = source.strip()
    notebook.metadata.kernelspec = KERNEL
    notebook.metadata.language_info = LANGUAGE_INFO
    nbformat.write(notebook, path)


COMMON_PATHS = '''def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "cam_calibration").is_dir():
            return candidate
    raise RuntimeError("无法定位项目根目录；请从 Xinyu 项目内启动 Jupyter。")

PROJECT_ROOT = find_project_root()
CALIBRATION_ROOT = PROJECT_ROOT / "cam_calibration"
INPUT_DIR = CALIBRATION_ROOT / "inputs"
OUTPUT_ROOT = CALIBRATION_ROOT / "outputs"
SHARED_DIR = OUTPUT_ROOT / "shared"
CONFIG_DIR = CALIBRATION_ROOT / "config"
for directory in (INPUT_DIR, OUTPUT_ROOT, SHARED_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
'''


update_code_cells(
    HERE / "01_depth_noise_fixed_camera.ipynb",
    [
        (
            '''import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd''',
            '''import cv2
import numpy as np
import pandas as pd

# VS Code/Jupyter often starts with the non-interactive inline backend.
# Activate ipympl before importing pyplot so Block 4 can receive mouse clicks.
try:
    get_ipython().run_line_magic("matplotlib", "widget")
except Exception as exc:
    print("无法启用 Matplotlib widget，将在 Block 4 使用手动坐标:", exc)

import matplotlib.pyplot as plt''',
        ),
        (
            'width, height, fps = 640, 480, 30\nnum_frames = 100\npatch_size = 7\noutput_dir = Path("outputs/01_depth_noise")\noutput_dir.mkdir(parents=True, exist_ok=True)',
            COMMON_PATHS
            + '''
width, height, fps = 640, 480, 30
num_frames = 100
patch_size = 7
warmup_frames = 30
output_dir = OUTPUT_ROOT / "01_depth_noise"
output_dir.mkdir(parents=True, exist_ok=True)''',
        ),
        (
            '    for _ in range(30):\n        pipeline.wait_for_frames()',
            '    for _ in range(warmup_frames):\n        pipeline.wait_for_frames()',
        ),
        (
            'with open(output_dir / "camera_intrinsics.json", "w", encoding="utf-8") as f:\n    json.dump(intrinsics, f, indent=2, ensure_ascii=False)\nprint("结果已保存到", output_dir.resolve())',
            '''with open(output_dir / "camera_intrinsics.json", "w", encoding="utf-8") as f:
    json.dump(intrinsics, f, indent=2, ensure_ascii=False)
with open(SHARED_DIR / "camera_intrinsics.json", "w", encoding="utf-8") as f:
    json.dump(intrinsics, f, indent=2, ensure_ascii=False)
print("结果已保存到", output_dir.resolve())
print("共享内参已保存到", (SHARED_DIR / "camera_intrinsics.json").resolve())''',
        ),
    ],
)

set_code_cell(
    HERE / "04_robot_table_plane_system_validation.ipynb",
    "rgbd_samples = []",
    '''
def load_depth_file(path, depth_scale=None):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if depth_scale is None:
            raise ValueError("raw .npy depth 需要 depth_scale。")
        return arr.astype(np.float64) * float(depth_scale)
    data = np.load(path)
    if "depth_m" in data:
        return data["depth_m"].astype(np.float64)
    if "depth_raw" in data and depth_scale is not None:
        return data["depth_raw"].astype(np.float64) * float(depth_scale)
    raise KeyError("NPZ 需要 depth_m，或 depth_raw + depth_scale。")

def capture_aligned_rgbd():
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    aligner = rs.align(rs.stream.color)
    try:
        for _ in range(30):
            pipeline.wait_for_frames()
        frames = aligner.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("没有获得 aligned RGB-D 帧。")
        color = np.asanyarray(color_frame.get_data())
        raw = np.asanyarray(depth_frame.get_data())
        scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intr_dict = {
            "width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
            "coeffs": list(intr.coeffs),
        }
        return color, raw.astype(np.float64) * scale, intr_dict
    finally:
        pipeline.stop()

rgbd_samples = []
if rgbd_manifest_path.exists():
    manifest = pd.read_csv(rgbd_manifest_path, dtype={"sample_id": str})
    if manifest.empty:
        raise RuntimeError(
            f"{rgbd_manifest_path} 还是空模板。请连接机械臂后填写 RGB-D 样本。"
        )
    if "sample_id" not in manifest.columns:
        raise KeyError("rgbd_manifest.csv 必须包含 sample_id。")
    pose_by_id = {record["sample_id"]: record for record in robot_pose_records}
    missing_pose_ids = sorted(set(manifest["sample_id"]) - set(pose_by_id))
    if missing_pose_ids:
        raise KeyError(f"这些 RGB-D sample_id 缺少机器人位姿: {missing_pose_ids}")
    scale_hint = calibration.get("depth_scale") if "calibration" in globals() else None
    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        color_path = Path(row["color_path"])
        depth_path = Path(row["depth_path"])
        if not color_path.is_absolute():
            color_path = PROJECT_ROOT / color_path
        if not depth_path.is_absolute():
            depth_path = PROJECT_ROOT / depth_path
        color = cv2.imread(str(color_path)) if str(row.get("color_path", "")).strip() else None
        depth_scale_hint = (
            float(row["depth_scale"])
            if "depth_scale" in row and pd.notna(row["depth_scale"])
            else scale_hint
        )
        depth_m = load_depth_file(depth_path, depth_scale_hint)
        if color is not None and depth_m.shape != color.shape[:2]:
            raise ValueError(f"{sample_id}: depth 未与 color 同尺寸/对齐。")
        rgbd_samples.append({
            "sample_id": sample_id,
            "color": color,
            "depth_m": depth_m,
            "T_base_ee": pose_by_id[sample_id]["T_base_gripper"],
        })
    print("离线 RGB-D 样本:", len(rgbd_samples))
else:
    raise FileNotFoundError(f"缺少 RGB-D manifest: {rgbd_manifest_path}")
''',
)


update_code_cells(
    HERE / "02_charuco_depth_scale_calibration.ipynb",
    [
        (
            'width, height, fps = 640, 480, 30\npatch_size = 7\nmin_charuco_corners = 6\nedge_margin_px = 12\nmax_abs_initial_error_m = 0.08\noutput_dir = Path("outputs/02_charuco_calibration")\noutput_dir.mkdir(parents=True, exist_ok=True)',
            COMMON_PATHS
            + '''
width, height, fps = 640, 480, 30
patch_size = 7
min_charuco_corners = 6
edge_margin_px = 12
max_abs_initial_error_m = 0.08
warmup_frames = 30
output_dir = OUTPUT_ROOT / "02_charuco_calibration"
output_dir.mkdir(parents=True, exist_ok=True)''',
        ),
        (
            'for _ in range(30):\n    pipeline.wait_for_frames()',
            'for _ in range(warmup_frames):\n    pipeline.wait_for_frames()',
        ),
        (
            'shared_dir = Path("outputs/shared"); shared_dir.mkdir(parents=True, exist_ok=True)\nwith open(shared_dir / "depth_calibration_linear.json", "w", encoding="utf-8") as f:',
            'with open(SHARED_DIR / "depth_calibration_linear.json", "w", encoding="utf-8") as f:',
        ),
        (
            'print("校准参数已保存:", (shared_dir / "depth_calibration_linear.json").resolve())',
            '''with open(SHARED_DIR / "camera_intrinsics.json", "w", encoding="utf-8") as f:
    json.dump(intrinsics, f, indent=2, ensure_ascii=False)
print("校准参数已保存:", (SHARED_DIR / "depth_calibration_linear.json").resolve())''',
        ),
    ],
)

set_code_cell(
    HERE / "01_depth_noise_fixed_camera.ipynb",
    'selection_mode = "click"',
    '''
selection_mode = "click"  # "manual" 或 "click"
manual_px, manual_py = width // 2, height // 2
click_timeout_s = 60

if selection_mode == "manual":
    px, py = int(manual_px), int(manual_py)
elif selection_mode == "click":
    backend = plt.get_backend().lower()
    interactive_backend = any(
        name in backend for name in ("ipympl", "widget", "nbagg")
    )
    if not interactive_backend:
        print(
            f"当前后端 {plt.get_backend()!r} 不支持点击，"
            f"自动使用手动坐标 ({manual_px}, {manual_py})。"
        )
        px, py = int(manual_px), int(manual_py)
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(cv2.cvtColor(preview_color, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"请在 {click_timeout_s} 秒内点击一个平坦、无边缘的测试点"
        )
        ax.axis("off")
        fig.canvas.draw_idle()
        plt.show(block=False)

        clicked = plt.ginput(1, timeout=click_timeout_s, show_clicks=True)
        if not clicked:
            plt.close(fig)
            print(
                f"未收到点击，自动使用手动坐标 "
                f"({manual_px}, {manual_py})。"
            )
            px, py = int(manual_px), int(manual_py)
        else:
            px, py = map(lambda value: int(round(value)), clicked[0])
            plt.close(fig)
else:
    raise ValueError("selection_mode 只能是 'manual' 或 'click'。")

half = patch_size // 2
if not (half <= px < width - half and half <= py < height - half):
    raise ValueError("测试点太靠近图像边缘。")

selected_vis = preview_color.copy()
cv2.rectangle(
    selected_vis, (px-half, py-half), (px+half, py+half), (0, 255, 0), 2
)
cv2.drawMarker(
    selected_vis, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 16, 2
)
plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(selected_vis, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"Selected ({px}, {py}), patch={patch_size}x{patch_size}")
plt.show()
cv2.imwrite(str(output_dir / "selected_patch.png"), selected_vis)
print(f"已选择像素 ({px}, {py})。")
''',
)


update_code_cells(
    HERE / "04_robot_table_plane_system_validation.ipynb",
    [
        (
            'patch_size = 7\nsample_grid = (7, 7)\ngrid_fraction = (0.25, 0.75, 0.25, 0.75)  # xmin,xmax,ymin,ymax\noutput_dir = Path("outputs/03_system_validation")\noutput_dir.mkdir(parents=True, exist_ok=True)\n\n# 与 D:\\HIWI\\Hand_eye_calibration.ipynb 的默认输出路径对齐。\nhand_eye_result_path = Path(r"D:\\HIWI\\hand_eye_data\\hand_eye_result.json")\nintrinsics_path = Path(r"D:\\HIWI\\hand_eye_data\\camera_intrinsics.json")\nrobot_poses_path = Path(r"D:\\HIWI\\hand_eye_data\\robot_poses.csv")\n\ndepth_calibration_path = Path("outputs/shared/depth_calibration_linear.json")\nrgbd_manifest_path = Path("inputs/rgbd_manifest.csv")  # TODO: depth_path 与 pose sample_id 对齐',
            COMMON_PATHS
            + '''
patch_size = 7
sample_grid = (7, 7)
grid_fraction = (0.25, 0.75, 0.25, 0.75)  # xmin,xmax,ymin,ymax
output_dir = OUTPUT_ROOT / "04_system_validation"
output_dir.mkdir(parents=True, exist_ok=True)

hand_eye_result_path = CONFIG_DIR / "hand_eye_result.json"
intrinsics_path = SHARED_DIR / "camera_intrinsics.json"
robot_poses_path = INPUT_DIR / "robot_poses.csv"
depth_calibration_path = SHARED_DIR / "depth_calibration_linear.json"
rgbd_manifest_path = INPUT_DIR / "rgbd_manifest.csv"''',
        ),
        (
            '    print("警告：未找到参考 hand-eye JSON，当前使用手动输入。")',
            '''    if np.allclose(T_gripper_camera, np.eye(4)):
        raise FileNotFoundError(
            f"缺少真实手眼结果: {hand_eye_result_path}。"
            "请稍后填写 cam_calibration/config/hand_eye_result.json；禁止使用单位矩阵占位值继续验证。"
        )
    print("未找到 JSON，当前使用已明确填写的手动手眼矩阵。")''',
        ),
        (
            '''    if "T_gripper_camera" not in hand_eye_result:
        raise KeyError("hand_eye_result.json 缺少 T_gripper_camera。")
    T_gripper_camera = validate_transform(hand_eye_result["T_gripper_camera"], "T_gripper_camera")''',
            '''    if hand_eye_result.get("T_gripper_camera") is None:
        raise RuntimeError(
            f"{hand_eye_result_path} 还是模板；请连接机械臂并写入真实 4x4 T_gripper_camera。"
        )
    T_gripper_camera = validate_transform(
        hand_eye_result["T_gripper_camera"], "T_gripper_camera"
    )''',
        ),
        (
            'plane_mode = "fit_points"  # "direct" 或 "fit_points"',
            'plane_mode = "config"  # "config"、"direct" 或 "fit_points"',
        ),
        (
            'if plane_mode == "direct":\n    n_base, d = normalize_plane(n_base_direct, d_direct)\nelse:\n    n_base, d = fit_plane(tcp_contact_points)',
            '''plane_config_path = CONFIG_DIR / "table_plane.json"
if plane_mode == "config":
    if not plane_config_path.exists():
        raise FileNotFoundError(
            f"缺少工作台平面配置: {plane_config_path}。"
            "可填写该文件，或把 plane_mode 改为 direct/fit_points。"
        )
    plane_config = json.loads(plane_config_path.read_text(encoding="utf-8"))
    n_base, d = normalize_plane(plane_config["n_base"], plane_config["d"])
elif plane_mode == "direct":
    n_base, d = normalize_plane(n_base_direct, d_direct)
elif plane_mode == "fit_points":
    n_base, d = fit_plane(tcp_contact_points)''',
        ),
        (
            '    contact_residuals = tcp_contact_points @ n_base + d\n    plt.figure',
            '    contact_residuals = tcp_contact_points @ n_base + d\n    plt.figure',
        ),
        (
            'shared=Path("outputs/shared");shared.mkdir(parents=True,exist_ok=True)\n(shared/"system_geometry.json").write_text',
            '(SHARED_DIR/"system_geometry.json").write_text',
        ),
    ],
)


update_code_cells(
    HERE / "05_part_task_specific_z_calibration.ipynb",
    [
        (
            'patch_size=7\noutput_dir=Path("outputs/04_part_z_calibration")\noutput_dir.mkdir(parents=True,exist_ok=True)\nuse_depth_correction=True\nheight_sign=+1  # TODO: base +Z 向上用 +1；反之 -1\n\ndepth_calibration_path=Path("outputs/shared/depth_calibration_linear.json")\nsystem_geometry_path=Path("outputs/shared/system_geometry.json")\nhand_eye_result_path=Path(r"D:\\HIWI\\hand_eye_data\\hand_eye_result.json")\nintrinsics_path=Path(r"D:\\HIWI\\hand_eye_data\\camera_intrinsics.json")',
            COMMON_PATHS
            + '''
patch_size=7
output_dir=OUTPUT_ROOT / "05_part_z_calibration"
output_dir.mkdir(parents=True,exist_ok=True)
use_depth_correction=True
height_sign=+1  # base +Z 向上用 +1；反之 -1

depth_calibration_path=SHARED_DIR / "depth_calibration_linear.json"
system_geometry_path=SHARED_DIR / "system_geometry.json"
hand_eye_result_path=CONFIG_DIR / "hand_eye_result.json"
intrinsics_path=SHARED_DIR / "camera_intrinsics.json"
part_heights_path=CONFIG_DIR / "part_heights.json"''',
        ),
        (
            '    else:\n        T_gripper_camera=manual_T_gripper_camera\n    n_base=manual_n_base',
            '''    else:
        if np.allclose(manual_T_gripper_camera, np.eye(4)):
            raise FileNotFoundError(
                f"缺少真实手眼结果: {hand_eye_result_path}，且手动矩阵仍为单位矩阵。"
            )
        T_gripper_camera=manual_T_gripper_camera
    n_base=manual_n_base''',
        ),
        (
            '''    if hand_eye_result_path.exists():
        hand_eye=json.loads(hand_eye_result_path.read_text(encoding="utf-8"))
        T_gripper_camera=np.asarray(hand_eye["T_gripper_camera"],float)''',
            '''    if hand_eye_result_path.exists():
        hand_eye=json.loads(hand_eye_result_path.read_text(encoding="utf-8"))
        if hand_eye.get("T_gripper_camera") is None:
            raise RuntimeError(
                f"{hand_eye_result_path} 还是模板；请写入真实 4x4 T_gripper_camera。"
            )
        T_gripper_camera=np.asarray(hand_eye["T_gripper_camera"],float)''',
        ),
        (
            'part_heights={\n    "base": None,\n    "Gear_compound": None,\n    "Gear_large": None,\n    "Gear_small": None,\n    "Screw": None,\n    "Shaft_large": None,\n    "Shaft_small": None,\n}\nmissing=[name for name,value in part_heights.items() if value is None or value<=0]',
            '''if not part_heights_path.exists():
    raise FileNotFoundError(
        f"缺少零件高度配置: {part_heights_path}。请填写模板中的真实高度（米）。"
    )
part_heights=json.loads(part_heights_path.read_text(encoding="utf-8"))
required_parts={"base","Gear_compound","Gear_large","Gear_small","Screw","Shaft_large","Shaft_small"}
missing_keys=sorted(required_parts-set(part_heights))
if missing_keys:
    raise KeyError("零件高度配置缺少: "+", ".join(missing_keys))
missing=[name for name,value in part_heights.items() if value is None or value<=0]''',
        ),
        (
            'T_base_gripper=np.array([ # TODO: 当前帧机器人位姿，按行写成 4x4\n    [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]\n],dtype=float)',
            '''current_pose_path=INPUT_DIR / "current_robot_pose.json"
if not current_pose_path.exists():
    raise FileNotFoundError(
        f"缺少当前 RGB-D 帧同步的机器人位姿: {current_pose_path}。"
        "请连接机械臂后填写模板；禁止使用单位矩阵占位。"
    )
pose_payload=json.loads(current_pose_path.read_text(encoding="utf-8"))
T_base_gripper=np.asarray(pose_payload["T_base_gripper"],dtype=float)
if T_base_gripper.shape!=(4,4) or np.allclose(T_base_gripper,np.eye(4)):
    raise ValueError("T_base_gripper 必须是真实的非占位 4x4 位姿。")''',
        ),
    ],
)

set_code_cell(
    HERE / "05_part_task_specific_z_calibration.ipynb",
    'source_mode="file"',
    '''
source_mode="file" # "live" 或 "file"
color_path=PROJECT_ROOT / "realsense_pic/realsense_color_20260609_112227.png"
depth_path=PROJECT_ROOT / "realsense_pic/realsense_depth_raw_20260609_112227.npy"
offline_depth_scale_to_m=0.001

def capture_aligned():
    import pyrealsense2 as rs
    pipeline=rs.pipeline()
    config=rs.config()
    config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
    config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
    profile=pipeline.start(config)
    aligner=rs.align(rs.stream.color)
    try:
        for _ in range(30):
            pipeline.wait_for_frames()
        frames=aligner.process(pipeline.wait_for_frames())
        color_frame=frames.get_color_frame()
        depth_frame=frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("没有获得 aligned RGB-D 帧。")
        color=np.asanyarray(color_frame.get_data())
        raw=np.asanyarray(depth_frame.get_data())
        scale=float(profile.get_device().first_depth_sensor().get_depth_scale())
        return color,raw.astype(np.float64)*scale
    finally:
        pipeline.stop()

if source_mode=="live":
    color_image,depth_m=capture_aligned()
elif source_mode=="file":
    color_image=cv2.imread(str(color_path))
    if color_image is None:
        raise FileNotFoundError(color_path)
    loaded=np.load(depth_path)
    if isinstance(loaded,np.lib.npyio.NpzFile):
        if "depth_m" in loaded:
            depth_m=loaded["depth_m"].astype(float)
        elif "depth_raw" in loaded:
            depth_m=loaded["depth_raw"].astype(float)*offline_depth_scale_to_m
        else:
            raise KeyError("NPZ 必须包含 depth_m 或 depth_raw。")
    else:
        depth_m=loaded.astype(float)*offline_depth_scale_to_m
    if depth_m.shape != color_image.shape[:2]:
        raise ValueError("离线 depth 必须已经 align 到 color，且尺寸完全一致。")
else:
    raise ValueError("source_mode 只能是 live 或 file。")

plt.figure(figsize=(10,7))
plt.imshow(cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB))
plt.title("Current part RGB")
plt.axis("off")
plt.show()
''',
)


for path in sorted(HERE.glob("0*_*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    print(f"updated {path.name}: {len(notebook.cells)} cells")
