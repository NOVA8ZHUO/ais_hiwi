"""Deployable stage logic: TCP, estimated hole, contact force; no true shaft pose."""
import torch


class InsertionSupervisor:
    # SETTLE=0, SEARCH=1, PROBE=2, INSERT=3
    def __init__(self, count, device):
        self.mode = torch.zeros(count, device=device, dtype=torch.long)
        self.age = torch.zeros(count, device=device)
        self.cooldown = torch.zeros_like(self.age)
        self.contact_count = torch.zeros_like(self.mode)
        self.contact_seen = torch.zeros(count, device=device, dtype=torch.bool)
        self.progress_count = torch.zeros_like(self.mode)
        self.locked_xy = torch.zeros(count, 2, device=device)
        self.locked_yaw = torch.zeros(count, 1, device=device)
        self.anchor_z = torch.zeros_like(self.age)

    def reset(self, ids):
        for value in (self.mode, self.age, self.cooldown, self.contact_count,
                      self.progress_count, self.contact_seen, self.locked_xy, self.locked_yaw, self.anchor_z):
            value[ids] = 0

    def update(self, ready, position, hole_estimate, force, yaw, dt, task):
        self.mode[~ready] = 0
        start = ready & (self.mode == 0)
        self.mode[start] = 1
        self.anchor_z[start] = position[start, 2]
        self.cooldown = (self.cooldown-dt).clamp_min(0.)
        searching = self.mode == 1
        touching = force[:, 2] >= task.probe_contact_N
        self.contact_seen |= touching & ready
        self.contact_count = torch.where(searching & touching & (self.cooldown <= 0), self.contact_count+1, 0)
        # Estimated penetration can confirm free entry before reaching the bottom.
        free_entry = searching & (position[:, 2] < hole_estimate[:, 2]-task.free_entry_depth)
        free_entry &= (self.anchor_z-position[:, 2] > task.probe_progress)
        free_entry &= force[:, :2].norm(dim=-1) < task.entry_lateral_force_N
        probing = self.mode == 2
        progress = probing & (self.anchor_z-position[:, 2] >= task.probe_progress)
        progress &= force[:, :2].norm(dim=-1) < task.entry_lateral_force_N
        self.progress_count = torch.where(free_entry | progress, self.progress_count+1, 0)
        commit = (self.progress_count >= 2) & (searching | probing)
        begin_probe = searching & (self.contact_count >= 2) & ~commit
        lock = begin_probe | (commit & searching)
        self.locked_xy[lock] = position[lock, :2]
        self.locked_yaw[lock] = yaw[lock]
        self.anchor_z[begin_probe] = position[begin_probe, 2]
        self.mode[begin_probe] = 2
        self.age[begin_probe] = 0.
        self.mode[commit] = 3
        self.age[self.mode == 2] += dt
        # A rim contact without insertion progress must not freeze XY forever.
        stalled = (self.mode == 2) & (self.age >= task.probe_timeout_s)
        self.mode[stalled] = 1
        self.cooldown[stalled] = task.probe_cooldown_s
        self.contact_count[stalled] = 0
        self.progress_count[stalled] = 0
        self.anchor_z[stalled] = position[stalled, 2]

    def observation(self):
        return torch.nn.functional.one_hot(self.mode, num_classes=4).float()
