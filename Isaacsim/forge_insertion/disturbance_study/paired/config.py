import json,os
from isaaclab.utils import configclass
from forge_cad.config import ForgeCadCfg
from .events import mass_event,friction_event,dead_zone_event

@configclass
class PairedCfg(ForgeCadCfg):
    def __post_init__(self):
        super().__post_init__()
        active=set(json.loads(os.environ['FORGE_ACTIVE_FACTORS']))
        fields={
          'base_position':(self.task,'fixed_asset_init_pos_noise',[0.]*3),
          'base_yaw':(self.task,'fixed_asset_init_orn_range_deg',0.),
          'hand_position':(self.task,'hand_init_pos_noise',[0.]*3),
          'hand_orientation':(self.task,'hand_init_orn_noise',[0.]*3),
          'grasp_offset':(self.task,'held_asset_pos_noise',[0.]*3),
          'hole_observation':(self.obs_rand,'fixed_asset_pos',[0.]*3),
          'tip_position_noise':(self.obs_rand,'fingertip_pos',0.),
          'tip_rotation_noise':(self.obs_rand,'fingertip_rot_deg',0.),
          'force_noise':(self.obs_rand,'ft_force',0.),
          'gains':(self.ctrl,'task_prop_gains_noise_level',[0.]*6),
          'position_threshold':(self.ctrl,'pos_threshold_noise_level',[0.]*3),
          'rotation_threshold':(self.ctrl,'rot_threshold_noise_level',[0.]*3),
          'ema':(self.ctrl,'ema_factor_range',[.0625,.0625]),
          'contact_threshold':(self.task,'contact_penalty_threshold_range',[7.5,7.5]),
        }
        for key,(obj,field,value) in fields.items():
            if key not in active:setattr(obj,field,value)
        self.events.object_scale_mass.func=mass_event
        self.events.object_scale_mass.params={}
        self.events.fixed_physics_material.func=friction_event
        self.events.fixed_physics_material.params={}
        self.events.dead_zone_thresholds.func=dead_zone_event
        self.events.dead_zone_thresholds.params={}
