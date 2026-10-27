from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
LAB=PROJECT.parent/'IsaacLab_2_3_2'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--clearances',nargs='+',type=float,choices=[.25,.5,1.,2.],default=[1.,.5,.25,2.]);args=parser.parse_args()
 with (ROOT/'validation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONPATH']=os.pathsep.join(map(str,[PROJECT]+[LAB/'source'/x for x in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']]))
  outcomes=[]
  for clearance in args.clearances:
   for pose in [1,0]:
    case=f'P{pose}_clearance_{clearance:g}mm';out=ROOT/'validation'/case;out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(ROOT/'validate_runtime.py'),'--clearance',str(clearance),'--pose',str(pose),'--num_envs','4','--headless']
    (out/'command.json').write_text(json.dumps(cmd))
    print('START',case,flush=True)
    with (out/'console.log').open('w') as log:
     try:code=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
     except subprocess.TimeoutExpired:code=-1
    p=out/'validation.json';data=json.loads(p.read_text()) if p.exists() else {'status':'failed','error':'No report; inspect console.log'}
    outcomes.append(dict(case=case,exit_code=code,**data));(ROOT/'validation_summary.json').write_text(json.dumps(outcomes,indent=2))
    print('DONE',case,data['status'],flush=True)
  outcomes=[]
  for clearance in [1.,.5,.25,2.]:
   for pose in [1,0]:
    case=f'P{pose}_clearance_{clearance:g}mm';path=ROOT/'validation'/case/'validation.json'
    data=json.loads(path.read_text()) if path.exists() else {'status':'pending'}
    outcomes.append(dict(case=case,exit_code=0 if data['status']=='passed' else -1,**data))
  (ROOT/'validation_summary.json').write_text(json.dumps(outcomes,indent=2))
  lines=['# 正式训练前验证结果','','本报告只包含资产/配置/reset/碰撞验证，不是训练或策略成功率。','', '| 条件 | 状态 | 说明 |','|---|---|---|']
  for row in outcomes:lines.append(f"| {row['case']} | {row['status']} | [详细报告](validation/{row['case']}/validation.json) |")
  passed=all(x['status']=='passed' and x['exit_code']==0 for x in outcomes)
  lines+=['',('全部运行时检查通过；训练分布和实测间隙仍须在正式训练方案中固定。' if passed else '尚未通过正式训练前验证。失败项见各报告；不得将未通过的间隙直接用于正式训练。'),'','其他随机化均继承完整随机化 CAD 基线。G1 的回合级几何采样和四组正式训练尚未实现/启动。']
  lines+=['','## 近壁碰撞探针（mm）','','| 条件 | 自由位置偏移 | 自由位置最大位移 | 穿墙位置偏移 | 最小排斥位移 |','|---|---:|---:|---:|---:|']
  for row in outcomes:
   d=row.get('small_clearance_contact_check')
   if d:lines.append('| '+row['case']+' | '+' | '.join(f'{1000*d[k]:.5f}' for k in ['free_offset_m','free_displacement_max_m','overlap_offset_m','overlap_displacement_min_m'])+' |')
  lines+=['','探针只覆盖离散初始姿态，不代表连续插入策略成功率，也不能确定 SDF 误差上限。未将旧模型成功率转移到新几何。']
  (ROOT/'VALIDATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
