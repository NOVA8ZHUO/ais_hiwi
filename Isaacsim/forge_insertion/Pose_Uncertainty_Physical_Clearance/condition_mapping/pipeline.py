import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parents[1]/'disturbance_study'
sys.path.insert(0,str(OLD))
from evaluate import run_case,verify_lock,require_preflight
from design import ALL
from samples import generate
import model_comparison as stats
stats.ROOT=ROOT
CLEARANCES=[.25,.5,1.,2.];ERRORS=[0.,.5,1.,2.,4.];SEEDS=list(range(400,405))
def condition(c,e):return f'c{c:g}mm_e{e:g}mm'
def save(path,data):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 verify_lock();require_preflight()
 for p,h in json.loads((ROOT/'manifest.json').read_text())['sha256'].items():assert sha(Path(p))==h,p

def check_bias(path,radius):
 import numpy as np
 state=json.loads((path/'initial_state.json').read_text())
 r=np.linalg.norm(np.asarray(state['hole_estimate_bias_m'])[:,:2],axis=1)*1000
 np.testing.assert_allclose(r,radius,rtol=1e-5,atol=1e-5)
 assert sorted(state['active_factors'])==sorted(ALL)

def report():
 import numpy as np
 rows=[];lines=['# 现有两个第 200 轮模型：工况摸底结果','','固定 XY 误差幅值、随机方向；Z 孔位误差及其他全部扰动保留。每格每模型 640 回合。95% CI 为按 seed/环境配对分层 bootstrap。不是训练随机化因果消融，也不是实机成功率。','', '| 直径间隙 mm | XY 误差 mm | 零随机化模型成功率 | 完整随机化模型成功率 | 提升 pp [95% CI] |','|---|---|---|---|---|']
 for c in CLEARANCES:
  for e in ERRORS:
   name=condition(c,e)
   for seed in SEEDS:
    for folder in ['results','results_randomized']:check_bias(ROOT/folder/name/f'seed_{seed}',e)
   row=stats.condition_stats(name,SEEDS);row.update(clearance_mm=c,error_mm=e);rows.append(row)
   g=row['success_gain'];lo,hi=g['ci95']
   lines.append(f"| {c:g} | {e:g} | {row['zero_success']:.2%} | {row['randomized_success']:.2%} | {100*g['estimate']:.2f} [{100*lo:.2f}, {100*hi:.2f}] |")
 lines+=['','## 失败表现','','| 间隙 / 误差 / 模型 | 曾成功未保持 | 最终深度 mm | 横向误差 mm | 平均腕力 N | 峰值力 P95 N |','|---|---:|---:|---:|---:|---:|']
 for row in rows:
  for model,d in row['diagnostics'].items():
   lines.append(f"| {row['clearance_mm']:g} / {row['error_mm']:g} / {model} | {d['lost_success_count']} | {d['mean_depth_mm']:.2f} | {d['mean_xy_error_mm']:.2f} | {d['mean_force_N']:.2f} | {d['p95_episode_peak_force_N']:.2f} |")
 lines+=['','使用未修改的原 FORGE 成功判据。小间隙下原固定横向容差不等于几何可装配证明，须结合深度/碰撞及后续完整插入验证解释。4 mm 是压力测试幅值；高斯训练分布无硬边界，不能称为严格分布外。仅一个训练 seed/模型，各格区间未多重比较校正；100% 的经验 bootstrap 退化不代表总体无失败。']
 (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');save(ROOT/'summary.json',rows)
 # Optional heatmaps do not gate completed numerical reports.
 try:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(12,4))
  for ax,key,title in zip(axes,['zero_success','randomized_success'],['Zero-randomization model','Full-randomization model']):
   z=np.array([r[key] for r in rows]).reshape(4,5)*100
   im=ax.imshow(z,vmin=0,vmax=100,aspect='auto');ax.set_xticks(range(5),ERRORS);ax.set_yticks(range(4),CLEARANCES);ax.set_xlabel('XY error radius (mm)');ax.set_ylabel('Diametral clearance (mm)');ax.set_title(title)
   for i in range(4):
    for j in range(5):ax.text(j,i,f'{z[i,j]:.1f}',ha='center',va='center',color='white' if z[i,j]<50 else 'black')
  fig.colorbar(im,ax=axes,label='Final success (%)');fig.savefig(ROOT/'success_heatmaps.png',dpi=180,bbox_inches='tight');plt.close(fig)
 except ImportError:pass

def run():
 with (ROOT/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=dict(status='preflight',pid=os.getpid(),completed=0,total=200,updated=time.time());save(ROOT/'status.json',state)
  try:
   verify()
   for model,c,e in [('zero',1.,0.),('randomized',.25,4.)]:
    os.environ['PG_CLEARANCE_MM']=str(c);os.environ['PG_ERROR_MM']=str(e)
    run_case(condition(c,e),ALL,450,n=4,model=model,area=str(ROOT/'preflight'/model))
    check_bias(ROOT/'preflight'/model/condition(c,e)/'seed_450',e)
   for seed in SEEDS:generate(ROOT/'samples'/f'n128_seed{seed}.npz',seed,128)
   durations=[];count=0
   for c in CLEARANCES:
    for e in ERRORS:
     os.environ['PG_CLEARANCE_MM']=str(c);os.environ['PG_ERROR_MM']=str(e)
     for seed in SEEDS:
      for model in ['zero','randomized']:
       seconds=sum(durations[-20:])/len(durations[-20:]) if durations else 100.
       state.update(status='running',current=dict(clearance_mm=c,error_mm=e,seed=seed,model=model),completed=count,updated=time.time(),estimated_remaining_seconds=(200-count)*seconds)
       save(ROOT/'status.json',state)
       print(f'PROGRESS {count}/200 ETA remaining {(200-count)*seconds/3600:.2f} h {state["current"]}',flush=True)
       run_case(condition(c,e),ALL,seed,model=model)
       folder='results' if model=='zero' else 'results_randomized';path=ROOT/folder/condition(c,e)/f'seed_{seed}'
       check_bias(path,e);durations.append(json.loads((path/'done.json').read_text())['seconds']);count+=1
       state.update(completed=count,updated=time.time());save(ROOT/'status.json',state)
   state.update(status='reporting');save(ROOT/'status.json',state);report()
   state.update(status='complete',updated=time.time());save(ROOT/'status.json',state)
  except BaseException as exc:
   state.update(status='failed',error=repr(exc),updated=time.time());save(ROOT/'status.json',state);raise
if __name__=='__main__':run()
