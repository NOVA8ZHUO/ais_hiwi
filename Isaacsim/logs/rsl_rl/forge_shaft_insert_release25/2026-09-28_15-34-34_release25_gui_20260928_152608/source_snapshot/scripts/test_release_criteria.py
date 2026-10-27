"""Exercise actual geometric/termination functions without launching Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import torch

path = Path(__file__).parents[1] / 'forge_shaft_insert/mdp/forge_paper.py'
source = ast.parse(path.read_text())
names = {'insertion_success', 'fully_seated', 'release_complete', 'invalid_grasp_width'}
functions = ast.Module(body=[node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
tip = torch.tensor([[0., 0., -.0249], [.003, 0., -.025], [0., 0., -.047], [.0131, 0., -.030], [0., 0., -.058]])
axis = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
term = NS(releasing=torch.tensor([False, True, True, False, False]), release_age=torch.tensor([0., 1., 2., 0., 0.]), elapsed_steps=torch.full((5,), 60))
env = NS(step_dt=1/15, cfg=NS(task=NS(success_depth=.025, insertion_depth=.047,
    insertion_region_radius=.013, success_xy_tolerance=.0005, depth_tolerance=.001,
    success_axis_cos=.996194698, release_observe_s=2., gripper_check_delay_s=1.5)))
scope = dict(torch=torch, tip_and_axis=lambda env: (tip, axis), hole_target_position_w=lambda env: torch.zeros_like(tip),
             action_term=lambda env: term, grasp_width_valid=lambda env: torch.zeros(5, dtype=torch.bool))
exec(compile(functions, str(path), 'exec'), scope)
# Depth success deliberately accepts 3 mm lateral error and large tilt; full seating does not.
assert scope['insertion_success'](env).tolist() == [False, True, True, False, False]
assert scope['fully_seated'](env).tolist() == [False, False, True, False, False]
assert scope['release_complete'](env).tolist() == [False, False, True, False, False]
# Opening fingers is allowed only in the explicit release phase.
assert scope['invalid_grasp_width'](env).tolist() == [True, False, False, True, True]
print('RELEASE_CRITERIA_TEST_PASS')
