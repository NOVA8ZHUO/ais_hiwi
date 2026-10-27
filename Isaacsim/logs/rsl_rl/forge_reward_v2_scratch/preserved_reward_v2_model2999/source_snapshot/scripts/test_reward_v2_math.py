import ast,torch
from pathlib import Path
from types import SimpleNamespace as S
s=Path('FORGE_shaft_Insert/forge_shaft_insert/mdp/forge_paper.py').read_text();node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='insertion_reward_v2')
t=S(elapsed_steps=torch.zeros(2,dtype=torch.long),reward_stamp=torch.full((2,),-1),reward_best_depth=torch.zeros(2),reward_anchor_depth=torch.zeros(2),reward_stall_age=torch.zeros(2),reward_values=torch.zeros(2,4),releasing=torch.zeros(2,dtype=torch.bool))
pos=torch.tensor([[0.,0.,-.01],[.02,0.,-.01]]);force=torch.tensor([[4.,0.,0.],[0.,0.,0.]])
e=S(cfg=S(task=S(insertion_region_radius=.013,insertion_depth=.047,success_depth=.025)),step_dt=1/15)
ns=dict(torch=torch,action_term=lambda e:t,tip_and_axis=lambda e:(pos,None),hole_target_position_w=lambda e:torch.zeros(2,3),insertion_enabled=lambda e:torch.ones(2,dtype=torch.bool),contact_force=lambda e:force)
exec(compile(ast.Module(body=[node],type_ignores=[]),'<reward>','exec'),ns);f=ns['insertion_reward_v2']
assert torch.allclose(f(e,0),torch.tensor([10.,0.]));assert f(e,3)[1]==1
assert f(e,0)[0]==10 # repeated queries must not consume reward
for _ in range(20):t.elapsed_steps+=1;f(e,0)
assert f(e,1)[0]==1 and f(e,2)[0]==2
pos[0,2]=-.005;t.elapsed_steps+=1;assert f(e,0)[0]==0
pos[0,2]=-.01;t.elapsed_steps+=1;assert f(e,0)[0]==0 # no oscillation farming
pos[0,2]=-.012;t.elapsed_steps+=1;assert abs(f(e,0)[0]-2)<1e-4;assert f(e,1)[0]==0
t.releasing[:]=True;t.elapsed_steps+=1;assert sum(f(e,j).abs().sum() for j in range(4))==0
print('PASS: geometric gate, no-contact outside, per-step cache, no repeat-depth reward, stall grace, progress reset, release exclusion')
