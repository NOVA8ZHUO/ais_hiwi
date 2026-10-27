"""Episode accounting, independent of Isaac Sim (SI units)."""
import torch


class EpisodeMetrics:
    def __init__(self, num_envs, max_steps, dt, device, stable_seconds=0.5):
        self.dt, self.stable_seconds = dt, stable_seconds
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.forces = torch.full((num_envs, max_steps), float('nan'), device=device)
        self.ever = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.final = self.ever.clone()
        self.streak = self.steps.clone()
        self.longest = self.steps.clone()
        self.saturated = torch.zeros(num_envs, device=device)
        self.active_steps = self.steps.clone()
        self.completed = 0
        self.totals = dict(success_any=0., success_final=0., success_stable=0.,
                           force_p95_N=0., force_peak_N=0., action_saturation=0.,
                           longest_success_s=0.)

    def sample(self, ids, success, force, actions, active):
        if ids.numel() == 0:
            return
        index = self.steps[ids]
        if (index >= self.forces.shape[1]).any():
            raise RuntimeError('Episode metrics exceeded configured episode length.')
        self.forces[ids, index] = force
        self.steps[ids] += 1
        self.ever[ids] |= success
        self.final[ids] = success
        self.streak[ids] = torch.where(success, self.streak[ids] + 1, 0)
        self.longest[ids] = torch.maximum(self.longest[ids], self.streak[ids])
        # Only the four motion commands; exclude the success prediction action.
        self.saturated[ids] += ((actions[:, :4].abs() >= .99).float().sum(-1) * active)
        self.active_steps[ids] += active.long()

    def finish(self, ids, reasons):
        if ids.numel() == 0:
            return []
        p95 = torch.nanquantile(self.forces[ids], .95, dim=1)
        peak = torch.nan_to_num(self.forces[ids], nan=0.).amax(dim=1)
        records = []
        for j, i in enumerate(ids.tolist()):
            if self.steps[i] == 0:
                continue
            active = int(self.active_steps[i])
            row = dict(env_id=i, steps=int(self.steps[i]), duration_s=float(self.steps[i])*self.dt,
                       success_any=int(self.ever[i]), success_final=int(self.final[i]),
                       success_stable=int(float(self.longest[i])*self.dt + 1e-9 >= self.stable_seconds),
                       longest_success_s=float(self.longest[i])*self.dt,
                       final_success_streak_s=float(self.streak[i])*self.dt,
                       force_p95_N=float(p95[j]), force_peak_N=float(peak[j]),
                       action_saturation=float(self.saturated[i])/(4*active) if active else 0.,
                       active_action_steps=active)
            row.update({name: int(value[j]) for name, value in reasons.items()})
            self.completed += 1
            row['episode_index'] = self.completed
            for key in self.totals:
                self.totals[key] += row[key]
            records.append(row)
        self.reset(ids)
        return records

    def reset(self, ids):
        for value in (self.steps, self.ever, self.final, self.streak, self.longest,
                      self.saturated, self.active_steps):
            value[ids] = 0
        self.forces[ids] = float('nan')

    def summary(self):
        return dict(completed_episodes=self.completed,
                    **{key: value/max(self.completed, 1) for key, value in self.totals.items()})
