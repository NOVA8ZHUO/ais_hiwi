"""Replay exactly the first model's jobs, then compare on paired seeds/envs."""
import csv,json
from pathlib import Path
import numpy as np
from runner import ROOT,run_case,verify_lock,digest
from design import LABELS

def replay_randomized(progress):
    lock=verify_lock();jobs=[]
    for path in sorted((ROOT/'results').glob('*/seed_*/done.json')):
        done=json.loads(path.read_text());identity=done['identity']
        assert identity['model']=='zero'
        assert identity['checkpoint_sha256']==lock['models']['zero']['checkpoint_sha256']
        assert identity['lock_sha256']==digest(ROOT/'lock.json')
        assert done['evaluation_sha256']==digest(path.with_name('evaluation.json'))
        jobs.append({'condition':identity['condition'],'active':identity['active'],'seed':identity['seed'],
                     'n':identity['num_envs'],'source_evaluation_sha256':done['evaluation_sha256']})
    assert jobs,'No completed first-model conditions'
    plan=ROOT/'randomized_replay_plan.json'
    if plan.exists():assert json.loads(plan.read_text())==jobs,'First-model job set changed during replay'
    else:plan.write_text(json.dumps(jobs,indent=2))
    for i,job in enumerate(jobs):
        progress(model='randomized',condition=job['condition'],seed=job['seed'],replay_job=i+1,replay_total=len(jobs))
        run_case(job['condition'],job['active'],job['seed'],n=job['n'],model='randomized')

def paired(condition,seeds):
    """Reject missing/mismatched pairs instead of silently comparing unmatched rows."""
    arrays=[];diagnostics={'zero':[],'randomized':[]}
    for seed in seeds:
        pair=[];fingerprints=[];factors=[];counts=[]
        for model,folder in [('zero','results'),('randomized','results_randomized')]:
            path=ROOT/folder/condition/f'seed_{seed}/evaluation.json'
            data=json.loads(path.read_text());done=json.loads(path.with_name('done.json').read_text())
            assert done['evaluation_sha256']==digest(path)
            assert done['identity']['model']==model
            assert data['seed']==seed and done['identity']['condition']==condition
            assert data['sample_sha256']==done['identity']['sample_sha256']
            rows=sorted(data['rows'],key=lambda r:r['env_id'])
            assert [r['env_id'] for r in rows]==list(range(data['episodes']))
            fingerprints.append(data['sample_sha256']);factors.append(data['active_factors']);counts.append(data['policy_sample_count'])
            pair.append([float(r['success_final']) for r in rows]);diagnostics[model]+=rows
        assert fingerprints[0]==fingerprints[1],'Different environmental/policy sample tapes'
        assert factors[0]==factors[1],'Different perturbation conditions'
        assert counts[0]==counts[1],'Different rollout lengths'
        arrays.append(np.stack(pair,axis=-1))
    return np.stack(arrays),diagnostics # seed,env,model

def interval(values,replicates=2000):
    values=np.asarray(values);s,n=values.shape[:2]
    rng=np.random.default_rng(798214);boot=[]
    for _ in range(replicates):
        si=rng.integers(s,size=s);ei=rng.integers(n,size=(s,n))
        boot.append(values[si[:,None],ei].mean(axis=(0,1)))
    return {'estimate':np.mean(values,axis=(0,1)).tolist(),'ci95':np.quantile(boot,[.025,.975],axis=0).tolist()}

def condition_stats(condition,seeds):
    v,diag=paired(condition,seeds)
    result={'condition':condition,'seeds':seeds,'episodes_per_model':v.shape[0]*v.shape[1],
            'zero_success':float(v[:,:,0].mean()),'randomized_success':float(v[:,:,1].mean()),
            'success_gain':interval(v[:,:,1]-v[:,:,0]),'diagnostics':{}}
    for model,rows in diag.items():
        result['diagnostics'][model]={
            'lost_success_count':sum(r['success_any'] and not r['success_final'] for r in rows),
            'mean_depth_mm':float(np.mean([r['final_insertion_depth_m']*1000 for r in rows])),
            'mean_xy_error_mm':float(np.mean([r['final_xy_error_m']*1000 for r in rows])),
            'mean_force_N':float(np.mean([r['force_mean_N'] for r in rows])),
            'p95_episode_peak_force_N':float(np.percentile([r['force_max_N'] for r in rows],95))}
    return result

def sensitivity(prefix,name,seeds):
    z,_=paired('zero',seeds);f,_=paired('full',seeds)
    on,_=paired(prefix+'_on_'+name,seeds);off,_=paired(prefix+'_off_'+name,seeds)
    drop=z-on;recovery=off-f
    # Positive = randomized training reduces this fixed model's sensitivity.
    improvement=np.stack([drop[:,:,0]-drop[:,:,1],recovery[:,:,0]-recovery[:,:,1],
         (drop[:,:,0]+recovery[:,:,0]-drop[:,:,1]-recovery[:,:,1])/2],axis=-1)
    return {'name':name,'seeds':seeds,'zero_enable_drop':float(drop[:,:,0].mean()),
      'randomized_enable_drop':float(drop[:,:,1].mean()),'zero_disable_recovery':float(recovery[:,:,0].mean()),
      'randomized_disable_recovery':float(recovery[:,:,1].mean()),'improvement':interval(improvement)}

def write_comparison():
    plan=json.loads((ROOT/'randomized_replay_plan.json').read_text())
    report={'conditions':{},'sensitivity':{}}
    for phase,predicate in [('screen',lambda s:s<100),('confirmation',lambda s:s>=100)]:
        cases={}
        for job in plan:
            if predicate(job['seed']):cases.setdefault(job['condition'],[]).append(job['seed'])
        report['conditions'][phase]=[condition_stats(c,sorted(seeds)) for c,seeds in sorted(cases.items())]
        summaries=[]
        for c,seeds in sorted(cases.items()):
            for prefix in ['group','factor']:
                if c.startswith(prefix+'_on_'):
                    name=c[len(prefix+'_on_'):]
                    assert set(seeds)==set(cases[prefix+'_off_'+name])
                    summaries.append({'level':prefix,**sensitivity(prefix,name,sorted(seeds))})
        report['sensitivity'][phase]=summaries
    (ROOT/'model_comparison.json').write_text(json.dumps(report,indent=2))
    fmt=lambda d:f"{100*d['estimate']:.2f} [{100*d['ci95'][0]:.2f}, {100*d['ci95'][1]:.2f}]"
    lines=['# 随机化训练提升了哪些方面的抗扰能力？','',
      '固定比较两版第200轮最终模型；第二阶段完整重放第一阶段实际完成的全部条件与seed，不单独筛选随机化模型的测试集。环境样本和策略标准正态样本逐条件配对。',
      '## 如何解读','',
      '- 成功率增益 = 随机化模型成功率 − 零随机化模型成功率。正值表示该测试条件下表现更好。',
      '- 开启下降改善 = 零随机化模型的开启下降 − 随机化模型的开启下降；它扣除了两模型在固定基准上的能力差异。',
      '- 关闭恢复减少 = 零随机化模型的关闭恢复 − 随机化模型的关闭恢复；正值表示随机化模型更少依赖移除该扰动。这个量受其他扰动和天花板效应影响，不能单独解释。',
      '- 敏感性分数减少 = 上述两项改善的平均值。正值表示在这两个测试背景下敏感性降低，不是可相加的贡献比例。',
      '- 所有差值单位为百分点（pp）；95%配对分层bootstrap先重采样seed，再共同重采样env_id，2000次。正的区间下界表示该区间口径下支持提升；跨零则尚不明确。未作全体多重比较校正。',
      '- 初筛使用零随机化模型选择条件，不能把筛选结果当成无偏的全体排名；独立确认seed单列。单项仅覆盖第一阶段选中的类别，未覆盖的类别不能声称没有提升。',
      '- 这是两个已训练模型的对比，各自只有一个训练seed；不能证明跨训练seed的因果效应，也不能拆出某项训练随机化单独带来的收益。','']
    # Prefer held-out confirmation for an automatic, appropriately limited conclusion.
    confirmed=report['sensitivity']['confirmation']
    for level,label in [('group','类别'),('factor','单项')]:
        subset=[d for d in confirmed if d['level']==level]
        supported=[d for d in subset if d['improvement']['ci95'][0][0]>0]
        lines += [f"独立确认中，开启扰动造成的成功率下降有改善证据的{label}："+
            ('、'.join(LABELS.get(d['name'],d['name']) for d in supported) if supported else '尚无区间下界为正的结果')+'。']
    for phase,title in [('screen','筛选样本'),('confirmation','独立确认样本')]:
        lines += ['',f'## {title}：同条件成功率','',
         '| 条件 | seeds | N/模型 | 零随机化模型 | 随机化模型 | 提升 pp [95% CI] |','|---|---|---:|---:|---:|---:|']
        for d in report['conditions'][phase]:
            lines.append(f"| {d['condition']} | {d['seeds']} | {d['episodes_per_model']} | {d['zero_success']:.2%} | {d['randomized_success']:.2%} | {fmt(d['success_gain'])} |")
        lines += ['',f'## {title}：对扰动的敏感性变化','',
         '| 层级/扰动 | 开启下降：零模型 pp | 开启下降：随机化模型 pp | 开启下降改善 pp [95% CI] | 关闭恢复减少 pp [95% CI] | 分数减少 pp [95% CI] |',
         '|---|---:|---:|---:|---:|---:|']
        for d in report['sensitivity'][phase]:
            x=d['improvement'];parts=[fmt({'estimate':x['estimate'][i],'ci95':[x['ci95'][0][i],x['ci95'][1][i]]}) for i in range(3)]
            lines.append(f"| {d['level']}/{LABELS.get(d['name'],d['name'])} | {100*d['zero_enable_drop']:.2f} | {100*d['randomized_enable_drop']:.2f} | {' | '.join(parts)} |")
        lines += ['',f'## {title}：失败表现（零模型 → 随机化模型）','',
         '| 条件 | 曾成功但未保持 次 | 平均深度 mm | 平均XY误差 mm | 平均腕力 N | 回合峰值腕力P95 N |',
         '|---|---:|---:|---:|---:|---:|']
        for d in report['conditions'][phase]:
            a=d['diagnostics']['zero'];b=d['diagnostics']['randomized']
            values=[f'{a[k]:.2f} → {b[k]:.2f}' for k in ['mean_depth_mm','mean_xy_error_mm','mean_force_N','p95_episode_peak_force_N']]
            lines.append(f"| {d['condition']} | {a['lost_success_count']} → {b['lost_success_count']} | {' | '.join(values)} |")
    lines+=['','腕力是控制频率下平滑腕部力模长，不是原始高频接触峰值。全部模型指纹见 lock.json，重放任务清单见 randomized_replay_plan.json，逐回合数据见 model_comparison_episodes.csv。']
    (ROOT/'MODEL_COMPARISON.md').write_text('\n'.join(lines)+'\n')
    with (ROOT/'model_comparison_episodes.csv').open('w') as f:
        writer=None
        for model,folder in [('zero','results'),('randomized','results_randomized')]:
            for job in plan:
                d=json.loads((ROOT/folder/job['condition']/f"seed_{job['seed']}"/'evaluation.json').read_text())
                for row in d['rows']:
                    record={'model':model,'condition':job['condition'],'seed':job['seed'],**row}
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(record));writer.writeheader()
                    writer.writerow(record)
    main=ROOT/'RESULTS.md'
    marker='\n## 两模型抗扰能力对比\n'
    text=main.read_text().split(marker)[0]
    main.write_text(text+marker+'\n第二阶段已完成，详见 [MODEL_COMPARISON.md](MODEL_COMPARISON.md)。同条件成功率、敏感性改善及失败表现均使用配对样本比较。\n')
