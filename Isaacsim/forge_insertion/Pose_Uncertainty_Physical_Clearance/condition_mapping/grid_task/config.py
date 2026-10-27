import os
from pathlib import Path
from isaaclab.utils import configclass
from paired.config import PairedCfg
@configclass
class GridCfg(PairedCfg):
 def __post_init__(self):
  super().__post_init__()
  c=float(os.environ['PG_CLEARANCE_MM'])
  assert c in [.25,.5,1.,2.]
  path=str(Path(__file__).resolve().parents[2]/'assets'/f'base_clearance_{c:g}mm.usda')
  self.task.fixed_asset_cfg.usd_path=path
  self.task.fixed_asset.spawn.usd_path=path
