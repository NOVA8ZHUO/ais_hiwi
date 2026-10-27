# Copyright (c) 2026, FORGE Shaft Insert Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Run Isaac Lab 2.3.2's RSL-RL trainer with the FORGE task registered."""

import os
import sys
from pathlib import Path


isaaclab_root = Path(os.environ.get("ISAACLAB_ROOT", "/home/masais/Xinyu/Isaacsim/IsaacLab_2_3_2"))
trainer = isaaclab_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
if not trainer.is_file():
    raise FileNotFoundError(f"Isaac Lab 2.3.2 trainer not found: {trainer}")

# Preserve the upstream script's required launch order: AppLauncher creates the
# SimulationApp before any task/Omniverse imports.  Register this external task
# immediately after upstream isaaclab_tasks is imported and before Hydra reads
# the Gym entry points.
sys.path.insert(0, str(trainer.parent))
source = trainer.read_text(encoding="utf-8")
registration_anchor = "import isaaclab_tasks  # noqa: F401"
if registration_anchor not in source:
    raise RuntimeError("Isaac Lab trainer layout changed: task registration anchor not found.")
source = source.replace(
    registration_anchor,
    registration_anchor + "\nimport forge_shaft_insert  # noqa: F401",
    1,
)
# The installed recurrent PPO squeezes the environment axis when its size is 1.
# Keep at least two environments per mini-batch; do not patch site-packages.
batch_anchor = '    # set the environment seed'
if batch_anchor not in source:
    raise RuntimeError("Trainer configuration anchor changed.")
source = source.replace(batch_anchor,
    '    if agent_cfg.policy.class_name == "ActorCriticRecurrent":\n'
    '        batches = agent_cfg.algorithm.num_mini_batches\n'
    '        count = env_cfg.scene.num_envs\n'
    '        if batches < 1 or count % batches or count // batches < 2:\n'
    '            raise ValueError("Recurrent PPO requires divisible num_envs and at least 2 environments per mini-batch with this RSL-RL version. For 4 envs use agent.algorithm.num_mini_batches=2; for 128 envs use 4.")\n'
    + batch_anchor, 1)
# Archive exact completed-episode metrics alongside checkpoints.
metrics_anchor = '    env = gym.make(args_cli.task, cfg=env_cfg'
if metrics_anchor not in source:
    raise RuntimeError("Trainer environment creation anchor changed.")
source = source.replace(metrics_anchor,
    '    from forge_shaft_insert import experiment_archive\n'
    '    experiment_archive.begin(log_dir, argv=forge_original_argv)\n'
    '    if hasattr(env_cfg.task, "metrics_csv_path"):\n'
    '        env_cfg.task.metrics_csv_path = os.path.join(log_dir, "episode_metrics.csv")\n'
    + metrics_anchor, 1)
# Flush TensorBoard before exporting all scalar events.
flush_anchor = '    print(f"Training time:'
assert flush_anchor in source
source = source.replace(flush_anchor,
    '    if getattr(runner, "writer", None) is not None and hasattr(runner.writer, "flush"):\n'
    '        runner.writer.flush()\n' + flush_anchor, 1)
# SimulationApp.close may exit the process without unwinding outer finally blocks.
close_anchor = '    env.close()'
assert close_anchor in source
source = source.replace(close_anchor, close_anchor + '\n    experiment_archive.finish("completed")', 1)
namespace = {"__name__": "__main__", "__file__": str(trainer), "forge_original_argv": sys.argv.copy()}
archive_status = "completed"
try:
    exec(compile(source, str(trainer), "exec"), namespace)
except BaseException:
    archive_status = "interrupted_or_failed"
    raise
finally:
    archive = sys.modules.get("forge_shaft_insert.experiment_archive")
    if archive is not None:
        try:
            archive.finish(archive_status)
        except Exception as error:
            print(f"Experiment archive export failed: {error}", file=sys.stderr)
