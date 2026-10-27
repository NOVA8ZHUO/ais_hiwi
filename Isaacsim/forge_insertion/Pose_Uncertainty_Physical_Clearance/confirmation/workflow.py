"""Detached follower: validate final checkpoints, run paired grid, report interactions."""
import fcntl,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;STUDY=ROOT.parent;OLD=STUDY.parent/'disturbance_study'
sys.path.insert(0,str(OLD))
from design import ALL
from samples import generate
from evaluate import run_case
import numpy as np
GROUPS=['P0G0','P1G0','P0G1','P1G1'];TRAIN=[0,1,2];SEEDS=list(range(600,605));CS=[.25,.5,1.,2.];ES=[0.,.5,1.,2.,3.,4.]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def status(**kw):
 p=ROOT/'status.json';s=json.loads(p.read_text()) if p.exists() else {};s.update(kw,updated=time.time(),pid=os.getpid());save(p,s)
def model(g,t):return f'{g}_seed{t}'
def case(c,e):return f'c{c:g}mm_e{e:g}mm'
def path(g,t,c,e,s):return ROOT/'results'/model(g,t)/case(c,e)/f'seed_{s}'
def verify_sources():
 for p,h in json.loads((ROOT/'manifest.json').read_text())['sha256'].items():assert sha(p)==h,p

def completed_models():
 manifest=sha(STUDY/'training_manifest.json');models={}
 for t in TRAIN:
  for g in GROUPS:
   done=STUDY/'training'/g/f'seed_{t}'/'done.json'
   if not done.exists():return None
   d=json.loads(done.read_text());assert d['manifest_sha256']==manifest
   # RL-Games saves both periodic and max-epoch snapshots at epoch 200.
   # The audit verifies model + observation normalization tensors are identical.
   audit=json.loads((STUDY/'training_analysis/checkpoint_audit.json').read_text())[model(g,t)]
   assert audit['all_checkpoints']==d['checkpoints']
   assert audit['both_epoch']==200 and audit['policy_and_normalization_tensors_equal']
   for file,digest in d['checkpoints'].items():assert sha(file)==digest
   p=audit['checkpoint'];h=audit['checkpoint_sha256'];assert d['checkpoints'][p]==h
   models[model(g,t)]={'checkpoint':p,'checkpoint_sha256':h}
 return {'models':models,'training_manifest_sha256':manifest}

def validate_job(folder,c,e,seed,n):
 d=json.loads((folder/'evaluation.json').read_text());done=json.loads((folder/'done.json').read_text());init=json.loads((folder/'initial_state.json').read_text())
 assert sha(folder/'evaluation.json')==done['evaluation_sha256']
 assert done['identity']['lock_sha256']==sha(ROOT/'models.json')
 assert d['episodes']==n and d['seed']==seed
 assert d['sample_sha256']==sha(ROOT/'samples'/f'n{n}_seed{seed}.npz')==done['identity']['sample_sha256']
 selected=json.loads((ROOT/'models.json').read_text())['models'][done['identity']['model']]
 assert done['identity']['checkpoint_sha256']==selected['checkpoint_sha256']
 assert d['clearance_mm']==c and d['xy_error_radius_mm']==e
 assert sorted(d['active_factors'])==sorted(ALL)
 rows=sorted(d['rows'],key=lambda x:x['env_id']);assert [r['env_id'] for r in rows]==list(range(n))
 assert all(r['steps']==d['policy_sample_count'] for r in rows)
 assert abs(np.mean([r['success_final'] for r in rows])-d['success_rate_final'])<1e-12
 np.testing.assert_allclose(np.linalg.norm(np.asarray(init['hole_estimate_bias_m'])[:,:2],axis=1)*1000,e,rtol=1e-5,atol=1e-5)
 return d

def intervals(v,reps=5000):
 # Crossed paired bootstrap: training seeds and shared evaluation seeds/envs.
 # Preserve group pairing; the SAME eval indices apply to every resampled model.
 rng=np.random.default_rng(601204);T,K,N,_=v.shape;boot=[]
 for _ in range(reps):
  ti=rng.integers(T,size=T);ki=rng.integers(K,size=K);ni=rng.integers(N,size=(K,N))
  boot.append(v[ti[:,None,None],ki[None,:,None],ni[None,:,:],:].mean(axis=(0,1,2)))
 return {'mean':v.mean(axis=(0,1,2)).tolist(),'ci95':np.quantile(boot,[.025,.975],axis=0).tolist()}
def metrics(v):
 s00,s10,s01,s11=[v[...,i] for i in range(4)]
 return np.stack([s00,s10,s01,s11,s10-s00,s01-s00,s11-s01,s11-s10,s11-s10-s01+s00],axis=-1)

def report():
 import re
 results=[];tapes={};lengths={};total=0
 for c in CS:
  for e in ES:
   v=np.zeros((3,5,128,4));diag={g:[] for g in GROUPS}
   for ti,t in enumerate(TRAIN):
    for ki,s in enumerate(SEEDS):
     for gi,g in enumerate(GROUPS):
      d=validate_job(path(g,t,c,e,s),c,e,s,128)
      assert tapes.setdefault(s,d['sample_sha256'])==d['sample_sha256']
      assert lengths.setdefault(s,d['policy_sample_count'])==d['policy_sample_count']
      rows=sorted(d['rows'],key=lambda x:x['env_id']);v[ti,ki,:,gi]=[r['success_final'] for r in rows];diag[g]+=rows;total+=128
   result={'clearance_mm':c,'error_mm':e,'rho':e/(c/2),'per_training_seed_success':v.mean(axis=(1,2)).tolist(),'columns':GROUPS+['S10-S00','S01-S00','S11-S01','S11-S10','interaction'],'statistics':intervals(metrics(v)),'diagnostics':{}}
   for g,rows in diag.items():
    times=[r['first_success_seconds'] for r in rows if r['first_success_seconds'] is not None]
    result['diagnostics'][g]={'ever_success_lost_count':sum(r['success_any'] and not r['success_final'] for r in rows),'mean_depth_mm':float(np.mean([r['final_insertion_depth_m']*1000 for r in rows])),'mean_xy_mm':float(np.mean([r['final_xy_error_m']*1000 for r in rows])),'mean_force_N':float(np.mean([r['force_mean_N'] for r in rows])),'peak_force_p95_N':float(np.percentile([r['force_max_N'] for r in rows],95)),'mean_first_success_seconds_conditional_on_ever_success':float(np.mean(times)) if times else None,'never_success_count':len(rows)-len(times)}
   results.append(result)
 assert total==184320
 save(ROOT/'summary.json',results)
 lines=['# 四组训练：统一网格评估与交互分析','','所有1440个正式评估任务通过文件、checkpoint、配对样本与回合完整性检查。每格每组3个训练seed×5个评估seed×128回合。下面是三个训练seed的等权平均，不是挑选best seed。','', '区间为5000次配对交叉分层bootstrap：共同抽取训练seed，再共同抽取评估seed与环境，保留四组和配对测试关系。仅3个训练seed，不确定性估计有限；各格比较未做多重校正，逐格显著性作为探索性证据。','', '| 间隙mm | 误差mm | rho | P0G0 | P1G0 | P0G1 | P1G1 | I pp [95% CI] |','|---|---|---|---|---|---|---|---|']
 for r in results:
  st=r['statistics'];m=st['mean'];lo,hi=st['ci95']
  lines.append(f"| {r['clearance_mm']:g} | {r['error_mm']:g} | {r['rho']:g} | "+' | '.join(f'{100*x:.2f}%' for x in m[:4])+f' | {100*m[8]:.2f} [{100*lo[8]:.2f}, {100*hi[8]:.2f}] |')
 lines+=['','## 各项收益（百分点，95%区间）','','| 间隙/误差 mm | S10−S00 | S01−S00 | S11−S01 | S11−S10 |','|---|---|---|---|---|']
 for r in results:
  m=r['statistics']['mean'];lo,hi=r['statistics']['ci95'];lines.append(f"| {r['clearance_mm']:g}/{r['error_mm']:g} | "+' | '.join(f'{100*m[i]:.2f} [{100*lo[i]:.2f},{100*hi[i]:.2f}]' for i in range(4,8))+' |')
 lines+=['','## 逐格解释','']
 for r in results:
  m=r['statistics']['mean'];lo,hi=r['statistics']['ci95'];interpret='正交互区间高于零' if lo[8]>0 else ('负交互区间低于零' if hi[8]<0 else '交互方向尚不确定')
  joint='联合组优于两个单独组的区间均高于零' if lo[6]>0 and lo[7]>0 else '尚未同时确认联合组优于两个单独组'
  lines.append(f"- 间隙{r['clearance_mm']:g}mm / 误差{r['error_mm']:g}mm：{interpret}；{joint}。")
 lines+=['','## 辅助指标','','| 间隙/误差/组 | 曾成功未保持 | 深度mm | XY误差mm | 平均力N | 峰值力P95 N | 首次成功时间s（仅曾成功回合） | 从未成功回合 |','|---|---|---|---|---|---|---|---|']
 for r in results:
  for g,d in r['diagnostics'].items():lines.append(f"| {r['clearance_mm']:g}/{r['error_mm']:g}/{g} | {d['ever_success_lost_count']} | {d['mean_depth_mm']:.2f} | {d['mean_xy_mm']:.2f} | {d['mean_force_N']:.2f} | {d['peak_force_p95_N']:.2f} | {d['mean_first_success_seconds_conditional_on_ever_success']} | {d['never_success_count']} |")
 lines+=['','负交互不等于联合组更差，需结合联合组相对两个单独组的绝对表现和区间；接近100%时存在天花板效应。全成功数据的经验bootstrap可能退化。原FORGE成功判据未改，额外完整几何装配判据尚未认证，因此这些图表不证明无穿透或实机成功。第7维策略成功预测不作评估标签。0mm只关闭XY孔位误差，其他随机化包括Z偏置保留。','', '每训练seed成功率和全部统计数据见 summary.json；逐回合记录见 results/。']
 provenance=json.loads((ROOT/'models.json').read_text())
 lines+=['','## 训练完成记录','','每组使用3个训练seed、128个环境和200轮预算。评估只使用第200轮权重，不按评估结果挑选checkpoint。训练回报不等于装配成功率。','', '| 组别 | 训练seed | 日志最后轮次 | 最后记录的best reward | checkpoint | SHA256 |','|---|---|---|---|---|---|']
 for t in TRAIN:
  for g in GROUPS:
   info=provenance['models'][model(g,t)]
   logpath=STUDY/'training'/g/f'seed_{t}'/'console.log'
   log=logpath.read_text(errors='replace') if logpath.exists() else ''
   epochs=re.findall(r'epoch:\s*(\d+)/200',log)
   rewards=re.findall(r'saving next best rewards:\s*\[([^\]]+)\]',log)
   lines.append(f"| {g} | {t} | {epochs[-1] if epochs else '日志未提取'} | {rewards[-1] if rewards else '日志未提取'} | `{info['checkpoint']}` | `{info['checkpoint_sha256']}` |")
 lines+=['','## 图表','','![四组成功率](success_heatmaps.png)','','![随机化收益](gain_heatmaps.png)','','![成功率交互](interaction_heatmap.png)','','来源锁定文件：manifest.json、models.json、integrity.json。训练原始日志与TensorBoard记录保存在上一级training目录。']
 (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 save(ROOT/'integrity.json',{'status':'passed','jobs':1440,'episodes':total,'models_sha256':sha(ROOT/'models.json')})
 try:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  for filename,indices,titles,vmin,vmax,cmap in [('success_heatmaps.png',range(4),GROUPS,0,100,'viridis'),('gain_heatmaps.png',range(4,8),['S10-S00','S01-S00','S11-S01','S11-S10'],-100,100,'coolwarm'),('interaction_heatmap.png',[8],['Interaction I'],-100,100,'coolwarm')]:
   fig,axs=plt.subplots(1,len(indices),figsize=(5*len(indices),4),squeeze=False)
   for ax,i,title in zip(axs[0],indices,titles):
    z=np.array([r['statistics']['mean'][i] for r in results]).reshape(4,6)*100
    im=ax.imshow(z,vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto');ax.set_xticks(range(6),ES);ax.set_yticks(range(4),CS);ax.set_xlabel('XY error (mm)');ax.set_ylabel('Clearance (mm)');ax.set_title(title)
    for x in range(4):
     for y in range(6):ax.text(y,x,f'{z[x,y]:.1f}',ha='center',va='center',fontsize=8)
   fig.colorbar(im,ax=axs.ravel().tolist());fig.savefig(ROOT/filename,dpi=180,bbox_inches='tight');plt.close(fig)
 except ImportError:(ROOT/'plot_warning.txt').write_text('matplotlib unavailable; numerical and Markdown results completed.')

def run():
 with (ROOT/'workflow.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   verify_sources();status(state='waiting_for_training',completed=0,total=1440,error=None)
   while True:
    models=completed_models()
    if models:break
    p=STUDY/'training_status.json';s=json.loads(p.read_text()) if p.exists() else {}
    if s.get('pid'):
     try:os.kill(int(s['pid']),0)
     except ProcessLookupError:raise RuntimeError('Training queue is no longer running; final checkpoints incomplete')
    if s.get('state')=='failed':raise RuntimeError('Training failed; downstream evaluation stopped: '+str(s.get('error')))
    status(state='waiting_for_training',training_state=s.get('state'),training_group=s.get('group'),training_seed=s.get('seed'),training_finished=len(list((STUDY/'training').glob('P*/seed_*/done.json'))))
    time.sleep(30)
   verify_sources()
   if (ROOT/'models.json').exists():assert json.loads((ROOT/'models.json').read_text())==models
   else:save(ROOT/'models.json',models)
   # Small-scale runtime gate on each of the twelve actual checkpoints.
   os.environ['PG_CLEARANCE_MM']='1';os.environ['PG_ERROR_MM']='3'
   for t in TRAIN:
    for g in GROUPS:
     name=model(g,t);status(state='preflight',current=name)
     run_case(case(1,3),ALL,650,n=4,model=name,area=str(ROOT/'preflight'/name))
     validate_job(ROOT/'preflight'/name/case(1,3)/'seed_650',1,3,650,4)
   for seed in SEEDS:generate(ROOT/'samples'/f'n128_seed{seed}.npz',seed,128)
   count=0;durations=[]
   for c in CS:
    for e in ES:
     os.environ['PG_CLEARANCE_MM']=str(c);os.environ['PG_ERROR_MM']=str(e)
     for s in SEEDS:
      for t in TRAIN:
       for g in GROUPS:
        name=model(g,t);rate=float(np.mean(durations[-30:])) if durations else 100.
        status(state='evaluating',completed=count,current={'clearance':c,'error':e,'eval_seed':s,'model':name},remaining_seconds=(1440-count)*rate)
        print(f'EVAL {count+1}/1440 {name} {case(c,e)} seed={s}',flush=True)
        run_case(case(c,e),ALL,s,model=name,area=str(ROOT/'results'/name))
        folder=path(g,t,c,e,s);validate_job(folder,c,e,s,128)
        durations.append(json.loads((folder/'done.json').read_text())['seconds']);count+=1
        status(completed=count)
   status(state='reporting');report();status(state='complete',completed=1440,remaining_seconds=0)
  except BaseException as exc:
   status(state='failed',error=repr(exc));raise
if __name__=='__main__':run()
