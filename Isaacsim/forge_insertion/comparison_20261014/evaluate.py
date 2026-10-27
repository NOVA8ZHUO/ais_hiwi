import json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
LAB=PROJECT.parent/'IsaacLab_2_3_2'
random_ckpt=Path(json.loads((PROJECT/'experiments/forge_cad_baseline/pipeline_status.json').read_text())['checkpoint'])
zero_run=PROJECT/'0randomization/runs/20261013T160124Z_train'
zero_ckpt=next(zero_run.rglob('last_Forge_ep_200_rew__*.pth'))
env=os.environ.copy()
env['PYTHONPATH']=os.pathsep.join(map(str,[PROJECT,PROJECT/'0randomization']+[LAB/'source'/x for x in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']]))
env['PYTHONUNBUFFERED']='1'
for name,task,ckpt in [
    ('zero_on_zero','Isaac-Forge-ShaftLarge-ZeroRand-Direct-v0',zero_ckpt),
    ('zero_on_random','Isaac-Forge-ShaftLarge-CAD-Direct-v0',zero_ckpt),
    ('random_on_zero','Isaac-Forge-ShaftLarge-ZeroRand-Direct-v0',random_ckpt)]:
    run=ROOT/name;run.mkdir(exist_ok=True)
    cmd=[sys.executable,str(ROOT/'rollout.py'),'--task',task,'--seed','0','--num_envs','128','--episodes','128','--checkpoint',str(ckpt),'--headless']
    (run/'command.json').write_text(json.dumps(cmd,indent=2))
    print('START',name,flush=True)
    with (run/'console.log').open('w') as log:
        result=subprocess.run(cmd,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode or not (run/'evaluation.json').exists() or (run/'failure.txt').exists():
        raise SystemExit('Evaluation failed: '+name)
    report=json.loads((run/'evaluation.json').read_text())
    print('DONE',name,report['success_rate_final'],flush=True)
