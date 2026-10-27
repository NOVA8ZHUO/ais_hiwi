"""Official FORGE with geometry-dependent asset metadata only."""
from pathlib import Path
from isaaclab.utils import configclass
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg

ASSETS = Path(__file__).resolve().parents[1] / 'assets/cad'

@configclass
class ForgeCadCfg(ForgeTaskPegInsertCfg):
    def __post_init__(self):
        super().__post_init__()
        self.task.held_asset_cfg.usd_path = str(ASSETS / 'shaft_large.usda')
        self.task.held_asset_cfg.height = 0.083
        # Factory uses diameter for gripper opening: this is the grasp section.
        self.task.held_asset_cfg.diameter = 0.035
        self.task.held_asset_cfg.mass = 0.0175
        self.task.held_asset.spawn.usd_path = self.task.held_asset_cfg.usd_path
        self.task.held_asset.spawn.mass_props.mass = 0.0175
        self.task.fixed_asset_cfg.usd_path = str(ASSETS / 'base.usda')
        self.task.fixed_asset_cfg.height = 0.047
        self.task.fixed_asset_cfg.diameter = 0.026
        self.task.fixed_asset.spawn.usd_path = self.task.fixed_asset_cfg.usd_path
        # Preserve the official initial insertion-tip clearance with a 33 mm longer shaft.
        self.task.hand_init_pos = [0.0, 0.0, 0.047 + (0.083 - 0.050)]
