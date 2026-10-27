"""Developer utility: freeze source once after validation. Never invoked by run.sh."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;LAB=PROJECT.parent/'IsaacLab_2_3_2'
ckpt=next((PROJECT/'0randomization/runs/20261013T160124Z_train').rglob('last_Forge_ep_200_rew__*.pth'))
random_ckpt=Path(json.loads((PROJECT/'experiments/forge_cad_baseline/pipeline_status.json').read_text())['checkpoint'])
assert 'ep_200' in random_ckpt.name and random_ckpt.is_file()
files=list(ROOT.glob('*.py'))+list((ROOT/'paired').glob('*.py'))+[ROOT/'paired/adapter.json',PROJECT/'forge_cad/config.py',ckpt,random_ckpt]
files+=list((PROJECT/'assets/cad').glob('*.usda'))
for component in ['factory','forge']:
 files+=list((LAB/'source/isaaclab_tasks/isaaclab_tasks/direct'/component).rglob('*.py'))
files+=[LAB/'source/isaaclab_tasks/isaaclab_tasks/direct/forge/agents/rl_games_ppo_cfg.yaml']
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'lock.json').write_text(json.dumps({'checkpoint':str(ckpt),'checkpoint_sha256':h(ckpt),'models':{'zero':{'checkpoint':str(ckpt),'checkpoint_sha256':h(ckpt)},'randomized':{'checkpoint':str(random_ckpt),'checkpoint_sha256':h(random_ckpt)}},'sha256':{str(p):h(p) for p in files}},indent=2))
