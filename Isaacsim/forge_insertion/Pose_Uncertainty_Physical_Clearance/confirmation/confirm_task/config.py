from isaaclab.utils import configclass
from grid_task.config import GridCfg
@configclass
class ConfirmCfg(GridCfg):
 def __post_init__(self):
  super().__post_init__()
  self.scene.replicate_physics=False
  self.scene.clone_in_fabric=False
