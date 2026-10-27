"""Only replace random draws in three pinned official methods; retain their math."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LAB=ROOT.parents[1]/'IsaacLab_2_3_2'
DIRECT=LAB/'source/isaaclab_tasks/isaaclab_tasks/direct'
SPECS=[('factory/factory_env.py','randomize_initial_state',['base_position','base_yaw','hole_observation','hand_position','hand_orientation','grasp_offset']),
 ('forge/forge_env.py','_reset_idx',['ema','contact_threshold','dead_zone','quaternion_sign']),
 ('forge/forge_env.py','_compute_intermediate_values',['tip_position_noise','rotation_axis','rotation_angle','force_noise'])]
patches={}
class Rewrite(ast.NodeTransformer):
 def __init__(self,keys):self.keys=iter(keys);self.count=0;self.gains=iter(['gains','position_threshold','rotation_threshold'])
 def visit_Call(self,node):
  if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch' and node.func.attr in ['rand','randn']:
   key=next(self.keys);self.count+=1
   return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_sample',ctx=ast.Load()),args=[ast.Constant(key),ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[]),*node.args],keywords=node.keywords),node)
  if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='forge_utils' and node.func.attr=='get_random_prop_gains':
   return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_paired_gains',ctx=ast.Load()),args=[*node.args,ast.Constant(next(self.gains))],keywords=[]),node)
  if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Call) and isinstance(node.func.value.func,ast.Name) and node.func.value.func.id=='super':
   node.func.value=ast.Name(id='FactoryEnv',ctx=ast.Load());node.args.insert(0,ast.Name(id='self',ctx=ast.Load()))
  return self.generic_visit(node)
for filename,method,keys in SPECS:
 src=(DIRECT/filename).read_text();tree=ast.parse(src)
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name in ['FactoryEnv','ForgeEnv'])
 fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==method)
 rw=Rewrite(keys);fn=rw.visit(fn);assert rw.count==len(keys)
 if method=='_compute_intermediate_values':
  fn.body.insert(0,ast.parse('self._select_observation_slot()').body[0])
 ast.fix_missing_locations(fn)
 patches[method]={'source_file':filename,'original_file_sha256':hashlib.sha256(src.encode()).hexdigest(),'method':ast.unparse(fn),'draw_keys':keys}
(ROOT/'paired/adapter.json').write_text(json.dumps(patches,indent=2))
for method,d in patches.items():(ROOT/'paired'/('adapted'+method+'.py.txt')).write_text(d['method']+'\n')
print('Generated adapters for',list(patches))
