"""Environment randomization ablation; all other CAD baseline fields inherited."""
from isaaclab.utils import configclass
from .cad_baseline import ForgeCadCfg

@configclass
class ForgeZeroCfg(ForgeCadCfg):
    def __post_init__(self):
        super().__post_init__()
        self.task.fixed_asset_init_pos_noise = [0.0]*3
        self.task.fixed_asset_init_orn_range_deg = 0.0
        self.task.hand_init_pos_noise = [0.0]*3
        self.task.hand_init_orn_noise = [0.0]*3
        self.task.held_asset_pos_noise = [0.0]*3
        self.obs_rand.fixed_asset_pos = [0.0]*3
        self.obs_rand.fingertip_pos = 0.0
        self.obs_rand.fingertip_rot_deg = 0.0
        self.obs_rand.ft_force = 0.0
        self.ctrl.task_prop_gains_noise_level = [0.0]*6
        self.ctrl.pos_threshold_noise_level = [0.0]*3
        self.ctrl.rot_threshold_noise_level = [0.0]*3
        self.ctrl.ema_factor_range = [0.0625, 0.0625]
        self.task.contact_penalty_threshold_range = [7.5, 7.5]
        self.events.object_scale_mass.params['mass_distribution_params'] = (0.0, 0.0)
        self.events.fixed_physics_material.params['static_friction_range'] = (0.75, 0.75)
        self.events.dead_zone_thresholds = None
