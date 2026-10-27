"""CPU checks for pilot update arithmetic, command construction and quota filtering."""
import importlib.util
from pathlib import Path
import tempfile
import csv

spec=importlib.util.spec_from_file_location('pilot',Path(__file__).with_name('run_pilot200.py'))
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
assert p.checkpoints()==[70,120,218]
assert [i-19+1 for i in p.checkpoints()]==[52,102,200]
train, evaluations=p.commands('/python','unique_run',Path('/run'))
assert train[train.index('--max_iterations')+1]=='200'
assert train[train.index('--num_envs')+1]=='128'
assert '--resume' in train and '--headless' in train
assert 'agent.algorithm.num_mini_batches=4' in train
assert 'agent.save_interval=10' in train
for i,cmd in zip(p.checkpoints(),evaluations):
    assert cmd[cmd.index('--checkpoint')+1]==f'/run/model_{i}.pt'
    assert cmd[cmd.index('--num_envs')+1]=='32'
    assert cmd[cmd.index('--episodes_per_env')+1]=='4'
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    fields=['env_id','physics_peak_speed_0','releasing_0','elapsed_s_0','shaft_axis_world_2','time_out_0']
    with (root/'episode_metrics_control_trace.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(5):
            w.writerow(dict(env_id=0,physics_peak_speed_0=.02 if i<4 else 99,releasing_0=0,elapsed_s_0=20,shaft_axis_world_2=1,time_out_0=1))
    (root/'evaluation_episodes.csv').write_text('success_final,final_depth_mm\n0,10\n1,47\n0,0\n0,24\n')
    stats=p.motion_stats(root)
    assert stats['max_physics_tcp_speed_mm_s']==20
    assert stats['max_gripped_tilt_deg']==0
    assert stats['shallow_stalls_2_to_25mm']==2
print('PILOT200_TEST_PASS')
