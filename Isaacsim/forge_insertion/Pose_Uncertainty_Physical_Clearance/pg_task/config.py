"""Full randomized FORGE background; only XY estimate bias and bore geometry vary."""
from pathlib import Path
from copy import deepcopy
from isaaclab.utils import configclass
from isaaclab.sim import MultiUsdFileCfg
from forge_cad.config import ForgeCadCfg
ROOT=Path(__file__).resolve().parents[1]
CLEARANCES=[.25,.5,1.,2.]

def configure(cfg,p,g):
    cfg.obs_rand.fixed_asset_pos=[.001*p,.001*p,.001]
    # Required to preserve distinct PhysX shapes. Apply to ALL four groups.
    cfg.scene.replicate_physics=False
    cfg.scene.clone_in_fabric=False
    original=cfg.task.fixed_asset.spawn
    spawn=MultiUsdFileCfg()
    for key,value in vars(original).items():
        if key not in ['func','usd_path'] and hasattr(spawn,key):setattr(spawn,key,deepcopy(value))
    # Deterministic balanced domain mixture, NOT per-reset hot swapping.
    spawn.usd_path=[str(ROOT/'assets'/f'base_clearance_{c:g}mm.usda') for c in (CLEARANCES if g else [1.])]
    spawn.random_choice=False
    cfg.task.fixed_asset.spawn=spawn
    # Keep nominal task metadata; only meshes determine actual collision clearance.
    # Existing rewards/controller do not receive the selected clearance.

@configclass
class P0G0Cfg(ForgeCadCfg):
    def __post_init__(self):super().__post_init__();configure(self,0,0)
@configclass
class P1G0Cfg(ForgeCadCfg):
    def __post_init__(self):super().__post_init__();configure(self,1,0)
@configclass
class P0G1Cfg(ForgeCadCfg):
    def __post_init__(self):super().__post_init__();configure(self,0,1)
@configclass
class P1G1Cfg(ForgeCadCfg):
    def __post_init__(self):super().__post_init__();configure(self,1,1)
