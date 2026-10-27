"""Only replace two hardcoded randomizations lacking configuration switches."""
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

class ForgeZeroEnv(ForgeEnv):
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Uniform [0, default_dead_zone] becomes its fixed midpoint.
        self.dead_zone_thresholds[:] = 0.5 * self.default_dead_zone
        # q and -q are identical rotations; remove random observation sign augmentation.
        self.flip_quats[:] = 1.0
