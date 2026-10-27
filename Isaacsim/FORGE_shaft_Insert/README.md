# FORGE Shaft Insertion

新增论文对齐插入任务：`FORGE-Shaft-Insert-Paper-v0`。奖励、阻抗控制、
随机化、与论文的差异及真机数据清单见 [FORGE_ALIGNMENT.md](docs/FORGE_ALIGNMENT.md)。
该入口现在使用24维 actor观测、48维 critic观测和5维残差动作；需要重新训练。
按真机参数持续闭合夹爪（20 N命令、35±5 mm宽度、20 mm/s闭合速度），
从hole_large上方开始基准下插，策略学习插入残差。
以下内容记录旧版 XY 对齐任务的验证结果，不代表新插入任务已通过相同验证。

# FORGE Franka Shaft Alignment — Reset Validation

The current milestone is the physical reset used before any PPO training.
It reproduces the grasp validated with `scripts/franka_diff_ik.py` and does not
insert, release the gripper, or use force observations yet.

The source USD is referenced rather than rebuilt.  `/Franka`, `/Parts/Shaft_large`,
and `/Parts/Base` are separate assets in every environment.  `Shaft_large` is an
independent dynamic rigid body with gravity enabled.  It is held only by the two
closed fingers through contact and friction; there is no fixed joint attaching it
to `panda_hand`.  Base is kinematic and stays at its original USD pose forever.

Exact reset:

- fingertip target: `(0.5, 0, 0.21) m`
- fingertip orientation: `(180, 0, 0) deg` XYZ Euler
- arm joints: the final values printed by the validated IK run
- finger state: measured contact-limited opening, with both drives commanded to
  `0.014 m` to maintain a physical squeeze without a cold-start contact impulse
- Shaft_large: the measured contact-settled pose near `(0.5, 0, 0.15) m`,
  independent and with gravity enabled
- Base root: `(0.5, 0, 0) m`, fixed
- hole_large: `(0.627, -0.00018226745, 0.05) m`, fixed

Task interface:

- task: `FORGE-Shaft-Insert-v0`
- observation: `[dx, dy, dz] = Shaft_large origin - hole_large origin`, world axes, metres
- action: accumulated Cartesian delta `[dx, dy, dz]`, clipped to `[-1, 1]`,
  scaled by `0.001 m`; zero action holds the previous Cartesian target
- success: XY norm `<= 0.25 mm`
- reset observation: approximately `[-0.127106, +0.000743, +0.099285] m`
- timeout: `8 s` (`240` policy steps at 30 Hz)
- reward: decrease in XY error + small action penalty + success bonus

All task geometry, thresholds, and frame offsets are centralized in
`TaskParametersCfg`. There is currently no reset randomization because the
physical reset must be validated first. In particular, Base is never randomized.

## One-time setup

The compatible source is kept separately at `/home/masais/Xinyu/Isaacsim/IsaacLab_2_3_2`; the original
`IsaacLab/` 3.0 checkout remains untouched. Activate the existing Python 3.11 / Isaac Sim 5.1 environment:

```bash
source /home/masais/miniconda3/etc/profile.d/conda.sh
conda activate Xinyu_isaacsim
cd /home/masais/Xinyu/Isaacsim/IsaacLab_2_3_2
./isaaclab.sh --install rsl_rl
./isaaclab.sh -p -m pip install -e ../FORGE_shaft_Insert
```

These installation steps are one-time only.

## Required validation order

Run these commands from `IsaacLab_2_3_2/` after setup. Do not start PPO until A--G all pass.

```bash
# A--C: registration, one-environment reset, observation units/frame (GUI)
./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/smoke_test.py --mode reset --num_envs 1 --keep_open

# D: zero action
./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/smoke_test.py --mode zero --num_envs 1 --steps 100

# D2: deterministic sign/frame check; action is -XY error
./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/smoke_test.py --mode directed --num_envs 1 --steps 200

# E: random action
./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/smoke_test.py --mode random --num_envs 1 --steps 300 --headless

# F: 32 parallel environments
./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/smoke_test.py --mode random --num_envs 32 --steps 300 --headless

# G: 128 parallel environments
./isaaclab.sh -p ../FORGE_shaft_Insert/scripts/smoke_test.py --mode random --num_envs 128 --steps 300 --headless
```

Check at each stage:

- A: the registered entry point is `isaaclab.envs:ManagerBasedRLEnv`.
- B: reset produces shape `(N, 3)` with finite values.
- C: reset observation is approximately `[-0.127106, +0.000743, +0.099285] m`.
  The GUI must show the shaft between both fingers at the original pickup point,
  while Base remains at the original USD pose.
- D: zero action keeps the gravity-enabled shaft physically held. The printed
  reset drift must stay below `3 mm`, and Base must never move.
- D2: the directed check repeatedly reaches the 0.25 mm XY threshold, proving frame and sign consistency.
- E: observations and rewards remain finite; termination and automatic reset counts are printed.
- F/G: the same checks pass without clone-path, sensor-shape, CUDA, or out-of-memory errors.

## PPO

Do not start PPO yet. The previous checkpoint was produced with the incorrect
moving-Base/fixed-shaft scene and is invalid for this task. First pass the GUI
reset and zero-action physical-grasp checks above; then proceed through D2--G.

## Validation status (Isaac Sim 5.1 / Isaac Lab 2.3.2)

Validated on the target machine:

- registration and one-environment reset: passed
- full 240-step zero-action episode: passed; Base stayed fixed, maximum
  Shaft_large world drift was `0.283 mm`, and maximum shaft-in-hand drift was
  `0.259 mm`
- directed Cartesian sign/scale test: passed; reached XY success and auto-reset
- one-environment 300-step random test: passed
- 32-environment 300-step random test: passed; 32/32 timeout auto-resets
- 128-environment 300-step random test: passed; 128/128 timeout auto-resets

The GUI appearance is intentionally left as a user-visible check with
`--keep_open`; PPO has not been started.


## Calibrated wrist camera

The camera-enabled diagnostic task is `FORGE-Shaft-Insert-Camera-v0`. It keeps
the PPO policy observation at `[dx, dy, dz]`; RGB/depth are available as scene
sensor data but are not yet policy inputs.

The hand-eye transform in `assets/franka.usda` is
`panda_hand <- camera_color_optical_frame`, copied from `daniilidis.json`.
The color intrinsics come from RealSense serial `317222075679`:

- requested calibration: 640 x 480 at 30 Hz
- `fx=605.3745`, `fy=604.3340`, `cx=318.1542`, `cy=255.3651`
- outputs: RGB and distance-to-image-plane depth

Run the GUI validation from `IsaacLab_2_3_2/`:

```bash
PYTHONUNBUFFERED=1 \
/home/masais/miniconda3/envs/Xinyu_isaacsim/bin/python \
../FORGE_shaft_Insert/scripts/camera_test.py \
--keep_open
```

The viewport automatically switches to the wrist camera and an RGB frame is
saved to `FORGE_shaft_Insert/outputs/wrist_camera_rgb.png`. The terminal prints
the camera prim path, world pose, requested/runtime intrinsic matrices, image
shapes, depth range, resolution, and frame rate.

Isaac Sim 5.1's USD pinhole camera cannot represent unequal `fx/fy` or an
off-center principal point. It therefore uses `f=604.8542` for both axes and
`(cx,cy)=(320,240)`. The script checks and reports this approximation instead
of claiming the measured matrix was represented exactly.
