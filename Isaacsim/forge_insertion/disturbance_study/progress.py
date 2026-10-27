#!/usr/bin/env python3
"""Read-only terminal dashboard. Does not import runner or change experiment locks."""
import argparse,json,math,statistics,sys,time
from pathlib import Path
from design import GROUPS,LABELS
ROOT=Path(__file__).resolve().parent

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError,OSError):return default

def bar(done,total,width=28):
    fraction=min(1,max(0,done/max(total,1)))
    count=int(width*fraction)
    return '['+'█'*count+'░'*(width-count)+f'] {100*fraction:5.1f}%'

def duration(seconds):
    minutes=max(0,math.ceil(seconds/60));h,m=divmod(minutes,60)
    return f'{h}小时{m:02d}分' if h else f'{m}分钟'

def records(folder):
    out=[]
    for p in (ROOT/folder).glob('*/seed_*/done.json'):
        d=read(p)
        if d and 'identity' in d:
            out.append((d['identity']['condition'],d['identity']['seed'],d.get('seconds',0),p))
    return out

def next_budget(summary,seen,active):
    value=max(5,5*math.ceil((seen+1)/5)) if seen>=0 else 5
    if summary:
        value=max(value,len(summary['seeds']))
        if active and not summary['clear_leader'] and len(summary['seeds'])<15:
            value=max(value,len(summary['seeds'])+5)
    return min(15,value)

def render():
    state=read(ROOT/'status.json',{})
    stage=state.get('stage','not_started');status=state.get('status','not_started')
    zero=records('results');random=records('results_randomized');completed=len(zero)+len(random)
    g=read(ROOT/'group_screen.json');f=read(ROOT/'factor_screen.json')
    group_seen=max([s for c,s,_,_ in zero if s<100 and (c.startswith('group_') or c in ['zero','full'])]+[-1])
    factor_seen=max([s for c,s,_,_ in zero if s<100 and c.startswith('factor_')]+[-1])
    if stage=='group_screen':group_seen=max(group_seen,state.get('seed',0))
    if stage=='factor_screen':factor_seen=max(factor_seen,state.get('seed',0))
    group_done=stage in ['factor_screen','held_out_confirmation','randomized_model_replay','model_comparison_report','complete']
    factor_done=stage in ['held_out_confirmation','randomized_model_replay','model_comparison_report','complete']
    gmin=len(g['seeds']) if group_done and g else next_budget(g,group_seen,stage=='group_screen')
    fmin=len(f['seeds']) if factor_done and f else next_budget(f,factor_seen,stage=='factor_screen')
    gmax=gmin if group_done else 15;fmax=fmin if factor_done else 15
    selected=state.get('selected_group')
    kmin=kmax=len(GROUPS[selected]) if selected in GROUPS else 0
    if not kmin:kmin,kmax=min(map(len,GROUPS.values())),max(map(len,GROUPS.values()))
    # Baselines are shared by the two screening levels; confirmation has 10 cases x 5 seeds.
    budget=lambda gs,fs,k:2*(10*gs+2*max(gs,fs)+2*k*fs+50)
    low,high=budget(gmin,fmin,kmin),budget(gmax,fmax,kmax)
    plan=read(ROOT/'randomized_replay_plan.json')
    if plan is not None:low=high=2*len(plan)
    if status=='complete':low=high=completed
    recent=sorted(zero+random,key=lambda r:r[3].stat().st_mtime)[-20:]
    times=[r[2] for r in recent if r[2]>0]
    rate=statistics.median(times) if times else read(ROOT/'runtime_estimate.json',{}).get('measured_seconds_per_128_episode_job',100)
    title={'group_screen':'① 零随机化模型：类别筛选','factor_screen':'② 零随机化模型：单项细分',
      'held_out_confirmation':'③ 零随机化模型：独立确认','randomized_model_replay':'④ 原随机化模型：配对重放',
      'model_comparison_report':'⑤ 生成两模型报告','complete':'全部完成','not_started':'尚未启动'}.get(stage,stage)
    lines=['FORGE 两模型扰动实验进度  '+time.strftime('%Y-%m-%d %H:%M:%S'),
      f'状态：{status}    当前阶段：{title}',
      '',f'总体（按当前最少预算） {bar(completed,low)}',
      f'已完成 {completed} 个任务：零随机化模型 {len(zero)}，原随机化模型 {len(random)}',
      f'总任务预算：{low}' + (f'～{high}（自动追加样本，尚未定案）' if low!=high else '（已确定）'),
      f'剩余任务：{max(0,low-completed)}'+(f'～{max(0,high-completed)}' if low!=high else ''),
      f'预计剩余：{duration(max(0,low-completed)*rate)}'+(f'～{duration(max(0,high-completed)*rate)}' if low!=high else ''),
      f'估速依据：最近 {len(times)} 个任务耗时中位数 {rate:.1f} 秒/任务' if times else f'估速依据：预检 {rate:.1f} 秒/任务',
      '（剩余时间为粗估；未扣除当前任务已运行时间，不含异常重试/停机。）','']
    if stage=='group_screen':
        count=sum(s<gmin and (c in ['zero','full'] or c.startswith('group_')) for c,s,_,_ in zero)
        lines.append(f'类别筛选当前预算 {bar(count,12*gmin)}  {count}/{12*gmin}，{gmin} seeds')
    elif stage=='factor_screen':
        count=sum(s<fmin and c.startswith('factor_') for c,s,_,_ in zero)
        lines.append(f'单项筛选 {bar(count,2*kmin*fmin)}  {count}/{2*kmin*fmin}（不含共享基准）')
    elif stage=='held_out_confirmation':
        count=sum(s>=100 for _,s,_,_ in zero)
        lines.append(f'独立确认 {bar(count,50)}  {count}/50')
    elif stage=='randomized_model_replay' and plan:
        lines.append(f'第二模型重放 {bar(len(random),len(plan))}  {len(random)}/{len(plan)}')
    condition=state.get('condition');seed=state.get('seed');model=state.get('model','zero')
    if condition is not None and status!='complete':
        folder='results_randomized' if model=='randomized' else 'results'
        job=ROOT/folder/condition/f'seed_{seed}'
        lines.append(f'当前任务：model={model}  condition={condition}  seed={seed}')
        log=job/'console.log'
        if log.exists():
            # Only read the tail; Isaac Sim startup logs can be large.
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-8000));tail=stream.read().decode(errors='replace').splitlines()
            rollouts=[line for line in tail if line.startswith('Rollout step=')]
            lines.append('仿真：'+(rollouts[-1] if rollouts else '初始化/加载场景中'))
    if status=='failed':
        lines+=['','实验已失败停止；上面的剩余量是待完成工作，不是仍在倒计时。',str(state.get('error',''))[-1000:]]
    if status=='running':
        # A dead pipeline can leave a stale running status. Inspect the saved PID command.
        pidfile=ROOT/'pipeline.pid'
        try:
            pid=int(pidfile.read_text());cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'disturbance_study/pipeline.py' not in cmd:lines.append('警告：未找到对应主进程，状态可能已过期。')
        except (OSError,ValueError):lines.append('提示：无法确认主进程存活，请检查 pipeline.log。')
    lines+=['','进度条按已完成的128回合任务计算；追加预算时百分比可能回退。',
      'Ctrl+C 只关闭本监视器，不会停止后台实验。']
    return '\n'.join(lines)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');p.add_argument('--interval',type=float,default=2)
    args=p.parse_args()
    if args.interval<=0:p.error('--interval must be positive')
    try:
        while True:
            if not args.once and sys.stdout.isatty():print('\033[2J\033[H',end='')
            print(render(),flush=True)
            if args.once:break
            time.sleep(args.interval)
    except KeyboardInterrupt:pass
