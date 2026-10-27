"""Display one official reset, then keep physics paused for inspection."""
import argparse
import json
from pathlib import Path
import time
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--task', default='Isaac-Forge-ShaftLarge-CAD-Direct-v0')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--num_envs', type=int, default=1)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
launcher = AppLauncher(a)
app = launcher.app
env = None
try:
    import gymnasium as gym
    import isaaclab_tasks
    import forge_cad
    from isaaclab_tasks.utils import parse_env_cfg
    cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs)
    cfg.seed = a.seed
    cfg.viewer.eye = (1.0, 0.5, 0.55)
    cfg.viewer.lookat = (0.55, 0.0, 0.15)
    env = gym.make(a.task, cfg=cfg)
    env.reset()
    r = env.unwrapped
    target = r._fixed_asset.data.root_pos_w[0].detach().cpu().tolist()
    target[2] += 0.08
    r.sim.set_camera_view(eye=[target[0]+0.32, target[1]+0.32, target[2]+0.22], target=target)
    report = {
        'task': a.task, 'seed': a.seed, 'physics_paused': True,
        'shaft_tip_m': r.held_pos.detach().cpu().tolist(),
        'hole_entry_m': r.fixed_pos_obs_frame.detach().cpu().tolist(),
        'finger_joints_m': r.joint_pos[:, 7:].detach().cpu().tolist(),
        'shaft_mass_kg': r._held_asset.root_physx_view.get_masses().cpu().tolist(),
    }
    r.sim.pause()
    Path('reset_scene.json').write_text(json.dumps(report, indent=2))
    print('RESET_SCENE_READY: official reset completed; physics paused for GUI inspection.', flush=True)
    while app.is_running():
        if r.sim.is_playing():
            r.sim.pause()
        app.update()
        time.sleep(0.03)
except Exception:
    Path('failure.txt').write_text(traceback.format_exc())
    raise
finally:
    if env is not None:
        env.close()
    app.close()
