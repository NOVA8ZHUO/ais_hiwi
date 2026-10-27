"""Capture terminal samples before ManagerBasedRLEnv automatically resets."""
import csv
from pathlib import Path
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply_inverse
from .episode_metrics import EpisodeMetrics
from .mdp import forge_paper as paper


class ForgeMetricsEnv(ManagerBasedRLEnv):
    def __init__(self, *args, **kwargs):
        self._metrics_in_step = False
        self._metrics_file = None
        self._trace_file = None
        super().__init__(*args, **kwargs)
        self.episode_metrics = EpisodeMetrics(self.num_envs, self.max_episode_length,
                                             self.step_dt, self.device)
        self._metrics_rows = []
        self._metric_log_values = {key: torch.zeros((), device=self.device)
                                   for key in self.episode_metrics.summary()}
        self._metrics_writer = None
        path = self.cfg.task.metrics_csv_path
        if path:
            target = Path(path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            self._metrics_file = target.open('w', newline='')
            self._trace_file = target.with_name(target.stem + '_control_trace.csv').open('w', newline='')
            self._trace_writer = None

    def _write_control_trace(self, ids):
        if self._trace_file is None:
            return
        ids = ids[ids < self.cfg.task.trace_num_envs]
        if ids.numel() == 0:
            return
        term = paper.action_term(self)
        tcp, hand_q, _, velocity = term.frame()
        tip, axis = paper.tip_and_axis(self)
        hole = paper.hole_target_position_w(self)
        local_axis = quat_apply_inverse(hand_q, axis)
        pitch = torch.rad2deg(torch.atan2(local_axis[:, 0], -local_axis[:, 2]))
        robot = self.scene['robot']
        fields = {'elapsed_s': (term.elapsed_steps*self.step_dt)[:, None],
                  'phase': paper.insertion_phase(self)[:, None],
                  'motion_enabled_before_action': self._metrics_active[:, None],
                  'shaft_error': tip-hole, 'pitch_hand_deg': pitch[:, None],
                  'tcp_error': tcp-hole, 'target_error': term.target-hole,
                  'reference_error': term.reference_position-hole,
                  'reference_velocity': term.reference_velocity, 'tcp_velocity': velocity[:, :3],
                  'command_force': term.commanded_force, 'contact_force': paper.contact_force(self),
                  'action': term.raw_actions, 'hand_quat': hand_q,
                  'joint_position': robot.data.joint_pos[:, term.joints],
                  'joint_velocity': robot.data.joint_vel[:, term.joints],
                  'width': paper.gripper_width(self)[:, None]}
        for name in self.termination_manager.active_terms:
            fields[name] = self.termination_manager.get_term(name)[:, None]
        header = ['control_step', 'env_id'] + [f'{name}_{j}' for name, value in fields.items() for j in range(value.shape[1])]
        if self._trace_writer is None:
            self._trace_writer = csv.writer(self._trace_file)
            self._trace_writer.writerow(header)
        values = torch.cat([value[ids].float() for value in fields.values()], dim=-1).detach().cpu().tolist()
        for env_id, values_row in zip(ids.tolist(), values):
            self._trace_writer.writerow([self.common_step_counter, env_id, *values_row])
        self._trace_file.flush()

    def _sample_metrics(self, ids):
        self.episode_metrics.sample(
            ids, paper.insertion_success(self)[ids], paper.contact_force(self)[ids].norm(dim=-1),
            paper.action_term(self).raw_actions[ids], self._metrics_active[ids])
        self._write_control_trace(ids)

    def _reset_idx(self, env_ids):
        if hasattr(self, 'episode_metrics'):
            if self._metrics_in_step:
                self._sample_metrics(env_ids)
                reasons = {name: self.termination_manager.get_term(name)[env_ids]
                           for name in self.termination_manager.active_terms}
                rows = self.episode_metrics.finish(env_ids, reasons)
                self._metrics_rows.extend(rows)
                if rows and self._metrics_file:
                    if self._metrics_writer is None:
                        self._metrics_writer = csv.DictWriter(self._metrics_file, fieldnames=list(rows[0]))
                        self._metrics_writer.writeheader()
                    self._metrics_writer.writerows(rows)
                    self._metrics_file.flush()
            else:
                # Explicit/manual reset discards an unfinished episode, not a failure.
                self.episode_metrics.reset(env_ids)
        super()._reset_idx(env_ids)

    def step(self, action):
        self._metrics_rows = []
        # Same readiness as process_actions, before its elapsed_steps increment.
        self._metrics_active = paper.insertion_enabled(self).clone()
        self._metrics_in_step = True
        try:
            obs, reward, terminated, truncated, info = super().step(action)
        finally:
            self._metrics_in_step = False
        ids = (~(terminated | truncated)).nonzero(as_tuple=False).flatten()
        self._sample_metrics(ids)
        # Use a fresh dictionary: the runner retains each step's log dictionary.
        info['log'] = dict(info.get('log', {}))
        for key, value in self.episode_metrics.summary().items():
            # RSL averages step logs within a rollout. Shared scalar storage
            # intentionally supplies the latest cumulative snapshot to every
            # retained entry, so counts remain exact rather than fractional.
            self._metric_log_values[key].fill_(value)
            info['log']['Metrics_cumulative/' + key] = self._metric_log_values[key]
        info['episode_metrics'] = self._metrics_rows
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._trace_file:
            self._trace_file.close()
            self._trace_file = None
        if self._metrics_file:
            self._metrics_file.close()
            self._metrics_file = None
        super().close()
