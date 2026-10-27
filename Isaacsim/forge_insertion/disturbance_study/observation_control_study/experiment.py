"""Fixed-budget paired observation/control follow-up. No training."""
import argparse, fcntl, hashlib, json, os, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
from design import ALL, GROUPS, LABELS
from evaluator import run_case, verify_lock, require_preflight
import model_comparison as comparison
import numpy as np
comparison.ROOT=HERE
PHASES={'screen':list(range(200,205)), 'confirmation':list(range(300,305))}
FACTORS=GROUPS['observation']+GROUPS['control']

def conditions():
    out={'zero':[], 'full':ALL}
    for prefix,items in [('group',{k:GROUPS[k] for k in ['observation','control']}),('factor',{k:[k] for k in FACTORS})]:
        for name,active in items.items():
            out[f'{prefix}_on_{name}']=active
            out[f'{prefix}_off_{name}']=[x for x in ALL if x not in active]
    return out

def plan():
    return [dict(phase=phase,model=model,condition=c,active=sorted(active),seed=seed)
            for phase,seeds in PHASES.items() for c,active in conditions().items()
            for seed in seeds for model in ['zero','randomized']]

def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify():
    verify_lock(); require_preflight()
    lock=json.loads((HERE/'manifest.json').read_text())
    for file,h in lock['sha256'].items():
        if digest(file)!=h:raise RuntimeError('实验代码或来源已变化，不能混用结果: '+file)
    assert lock['jobs']==plan()

def jobdir(job):
    return HERE/('results' if job['model']=='zero' else 'results_randomized')/job['condition']/f"seed_{job['seed']}"

def estimate_seconds():
    ds=[json.loads(p.read_text())['seconds'] for folder in ['results','results_randomized'] for p in (PARENT/folder).glob('*/seed_*/done.json')]
    return float(np.mean(ds)) if ds else 96.5

def bootstrap_interval(values,alpha=.05):
    # Paired hierarchical bootstrap: same seed/env indices for every contrast.
    rng=np.random.default_rng(912873);s,n=values.shape[:2];boot=[]
    for _ in range(10000):
        si=rng.integers(s,size=s);ei=rng.integers(n,size=(s,n))
        boot.append(values[si[:,None],ei].mean(axis=(0,1)))
    return {'estimate':values.mean(axis=(0,1)).tolist(),'interval':np.quantile(boot,[alpha/2,1-alpha/2],axis=0).tolist()}

def report():
    # Validate every job and pair before reporting. Also check tapes across conditions.
    tapes={};lock=verify_lock()
    for j in plan():
        d=jobdir(j);done=json.loads((d/'done.json').read_text());data=json.loads((d/'evaluation.json').read_text());identity=done['identity']
        assert done['evaluation_sha256']==digest(d/'evaluation.json')
        assert identity['checkpoint_sha256']==lock['models'][j['model']]['checkpoint_sha256']
        assert identity['lock_sha256']==digest(PARENT/'lock.json')
        assert identity['condition']==j['condition'] and identity['model']==j['model'] and identity['seed']==j['seed']
        assert identity['active']==j['active'] and sorted(data['active_factors'])==j['active']
        assert data['episodes']==128 and all(r['steps']==data['policy_sample_count'] for r in data['rows'])
        assert data['seed']==j['seed'] and identity['sample_sha256']==data['sample_sha256']==digest(HERE/'samples'/f"n128_seed{j['seed']}.npz")
        assert tapes.setdefault(j['seed'],data['sample_sha256'])==data['sample_sha256']
    results={};lines=['# 观测与控制单项补充实验：最终结果','', '仅在全部 480 个任务完成并通过配对完整性检查后生成。实验方案见 [EXPERIMENT.md](EXPERIMENT.md)。', '',
        '所有差值单位均为百分点。初筛与确认分开报告，不合并后再选择结论。95% 区间为配对分层 bootstrap；确认阶段九项主要比较另列 Bonferroni 校正区间（近似家族置信度 95%）。','']
    fmt=lambda x:f'{100*x:.2f}'
    for phase,seeds in PHASES.items():
        stats=[comparison.condition_stats(c,seeds) for c in conditions()]
        z,_=comparison.paired('zero',seeds);f,_=comparison.paired('full',seeds)
        lines += [f'## {"初筛" if phase=="screen" else "独立确认"}：seeds {seeds}', '', '| 条件 | 零随机化成功率 % | 随机化成功率 % | 提升 pp [95% CI] |','|---|---:|---:|---|']
        for row in stats:
            gain=row['success_gain'];lo,hi=gain['ci95']
            lines.append(f"| {row['condition']} | {fmt(row['zero_success'])} | {fmt(row['randomized_success'])} | {fmt(gain['estimate'])} [{fmt(lo)}, {fmt(hi)}] |")
        summaries=[]
        lines+=['','### 单项敏感性及抗扰改善','', 'D=无扰动成功率−单独开启成功率；R=单独关闭成功率−完整随机化成功率；T=(D+R)/2。主要改善量 ΔD=D零−D随机，正值表示随机化模型对该扰动的开启损失较小。T 不是失败贡献占比。','', '| 单项 | D零 | D随机 | R零 | R随机 | ΔD [95% CI] | ΔD 校正区间（仅确认） |','|---|---:|---:|---:|---:|---|---|']
        for factor in FACTORS:
            on,_=comparison.paired('factor_on_'+factor,seeds);off,_=comparison.paired('factor_off_'+factor,seeds)
            drop=z-on;recovery=off-f;score=(drop+recovery)/2
            values=np.stack([drop[:,:,0],drop[:,:,1],recovery[:,:,0],recovery[:,:,1],score[:,:,0],score[:,:,1],drop[:,:,0]-drop[:,:,1],recovery[:,:,0]-recovery[:,:,1]],axis=-1)
            ci=bootstrap_interval(values);adjusted=bootstrap_interval(values[:,:,6],alpha=.05/len(FACTORS)) if phase=='confirmation' else None
            summaries.append({'factor':factor,'columns':['D_zero','D_randomized','R_zero','R_randomized','T_zero','T_randomized','delta_D','delta_R'],'statistics':ci,'delta_D_familywise_interval':adjusted})
            m=ci['estimate'];lo,hi=ci['interval'];adj='—' if adjusted is None else '['+', '.join(fmt(x) for x in adjusted['interval'])+']'
            lines.append(f"| {LABELS[factor]} | {fmt(m[0])} | {fmt(m[1])} | {fmt(m[2])} | {fmt(m[3])} | {fmt(m[6])} [{fmt(lo[6])}, {fmt(hi[6])}] | {adj} |")
        lines+=['','### 辅助指标','', '| 条件 / 模型 | 曾成功未保持（回合） | 最终深度 mm | 最终横向误差 mm | 平均腕力 N | 回合峰值力 P95 N |','|---|---:|---:|---:|---:|---:|']
        for row in stats:
            for model,d in row['diagnostics'].items():
                lines.append(f"| {row['condition']} / {model} | {d['lost_success_count']} | {d['mean_depth_mm']:.2f} | {d['mean_xy_error_mm']:.2f} | {d['mean_force_N']:.2f} | {d['p95_episode_peak_force_N']:.2f} |")
        results[phase]={'conditions':stats,'factors':summaries}
    lines+=['','## 根据独立确认集得到的结论','']
    confirmed=results['confirmation']['factors']
    for group in ['observation','control']:
        ranking=sorted([x for x in confirmed if x['factor'] in GROUPS[group]],key=lambda x:-x['statistics']['estimate'][4])
        lines.append(f"- {LABELS[group]}：按零随机化模型 T 点估计排序为 "+'、'.join(LABELS[x['factor']] for x in ranking)+'。排序不等于已证明各项之间存在显著差异。')
    for x in confirmed:
        lo,hi=x['delta_D_familywise_interval']['interval']
        finding='支持随机化模型的开启损失更小' if lo>0 else ('支持随机化模型的开启损失更大' if hi<0 else '尚不能确认开启损失有差异；不代表两者等效')
        lines.append(f"- {LABELS[x['factor']]}：{finding}。")
    lines+=['', '仅比较两个固定 checkpoint，在当前扰动分布和固定值下解释。ON/OFF 背景不同，不能将各项分数相加归因为全部失败，也不能据此证明哪种训练随机化最有价值。全成功/全失败时经验 bootstrap 可能退化为零宽区间，不表示总体确定性；有限样本、五个评估 seed 和单个训练 seed 仍限制推断。均值腕力下降不代表峰值下降。']
    save(HERE/'summary.json',results)
    tmp=HERE/'RESULTS.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(HERE/'RESULTS.md')

def pipeline():
    fd=os.environ.pop('STUDY_START_LOCK_FD',None)
    with (os.fdopen(int(fd),'a') if fd is not None else (HERE/'pipeline.lock').open('a')) as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('本实验已经运行。')
        state={'status':'validating','pid':os.getpid(),'completed':0,'total':len(plan()),'updated':time.time()}
        save(HERE/'status.json',state)
        try:
            verify();jobs=plan();durations=[];fallback=estimate_seconds()
            # Generate every sample tape before any simulation, never reuse historical seeds.
            from samples import generate
            for seed in sum(PHASES.values(),[]):generate(HERE/'samples'/f'n128_seed{seed}.npz',seed,128)
            for i,j in enumerate(jobs):
                state.update(status='running',completed=i,current=j,updated=time.time(),seconds_per_job=float(np.mean(durations[-30:])) if durations else fallback)
                save(HERE/'status.json',state)
                remain=(len(jobs)-i)*state['seconds_per_job']
                print(f"PROGRESS {i}/{len(jobs)} ({100*i/len(jobs):.1f}%) remaining~{remain/3600:.2f}h ETA={datetime.fromtimestamp(time.time()+remain).isoformat(timespec='seconds')} {j}",flush=True)
                run_case(j['condition'],j['active'],j['seed'],model=j['model'])
                durations.append(json.loads((jobdir(j)/'done.json').read_text())['seconds'])
                state.update(completed=i+1,updated=time.time());save(HERE/'status.json',state)
            state.update(status='reporting',updated=time.time());save(HERE/'status.json',state)
            report();state.update(status='complete',updated=time.time());save(HERE/'status.json',state)
            print('COMPLETE 480/480; final report: '+str(HERE/'RESULTS.md'),flush=True)
        except BaseException as exc:
            state.update(status='failed',error=repr(exc),updated=time.time());save(HERE/'status.json',state)
            raise

def progress():
    path=HERE/'status.json'
    state=json.loads(path.read_text()) if path.exists() else {'status':'not_started','completed':0,'total':480}
    done=state['completed'];total=state['total'];status=state['status']
    if status in ['running','validating','reporting']:
        try:os.kill(state['pid'],0)
        except ProcessLookupError:status='stopped (进程已退出，查看 pipeline.log)'
    rate=state.get('seconds_per_job',estimate_seconds());remaining=max(0,(total-done)*rate-max(0,time.time()-state.get('updated',time.time())))
    print(f"[{('#'*int(30*done/total)).ljust(30,'-')}] {done}/{total} ({100*done/total:.1f}%)\n状态: {status}\n已完成回合: {done*128:,}/{total*128:,}")
    if status in ['running','not_started']:
        print(f"预计剩余: {remaining/3600:.2f} 小时；预计完成: {datetime.fromtimestamp(time.time()+remaining).astimezone().isoformat(timespec='seconds')}（按近期任务耗时，非保证）")
    if 'current' in state:print('当前任务:',state['current'])
    if 'error' in state:print('错误:',state['error'])
    print('日志:',HERE/'pipeline.log');print('最终结果:',HERE/'RESULTS.md')

def start():
    verify()
    with (HERE/'pipeline.lock').open('a') as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('本实验已运行，请使用 progress.sh 查看。')
        with (HERE/'pipeline.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(HERE/'experiment.py'),'run'],cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(guard.fileno(),),env={**os.environ,'STUDY_START_LOCK_FD':str(guard.fileno())})
        (HERE/'pipeline.pid').write_text(str(child.pid)+'\n')
    time.sleep(2)
    if child.poll() is not None:raise SystemExit('启动失败，请查看 pipeline.log')
    print(f'已后台启动 PID={child.pid}，可以关闭 terminal。\n进度: bash {HERE}/progress.sh\n日志: tail -f {HERE}/pipeline.log')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','run','progress','check']);args=parser.parse_args()
    {'start':start,'run':pipeline,'progress':progress,'check':lambda:(verify(),print(f'Validated: {len(plan())} jobs / {128*len(plan())} episodes; estimated {estimate_seconds()*len(plan())/3600:.1f} h'))}[args.action]()
