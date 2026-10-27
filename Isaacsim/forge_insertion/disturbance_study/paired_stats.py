"""Paired hierarchical bootstrap: resample seeds, then env IDs jointly across contrasts."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def outcomes(condition,seeds):
    arrays=[]
    for seed in seeds:
        d=json.loads((ROOT/'results'/condition/f'seed_{seed}/evaluation.json').read_text())
        arrays.append([int(r['success_final']) for r in sorted(d['rows'],key=lambda r:r['env_id'])])
    return np.asarray(arrays,dtype=float)

def compare(items,seeds,prefix,replicates=2000):
    zero=outcomes('zero',seeds);full=outcomes('full',seeds)
    contrasts=[]
    for name in items:
        on=outcomes(prefix+'_on_'+name,seeds);off=outcomes(prefix+'_off_'+name,seeds)
        contrasts.append(np.stack([zero-on,off-full,(zero-on+off-full)/2],axis=-1))
    values=np.stack(contrasts,axis=-1) # seed,env,contrast,item
    s,n=zero.shape;rng=np.random.default_rng(67123)
    boot=np.empty((replicates,3,len(items)))
    for b in range(replicates):
        si=rng.integers(s,size=s);ei=rng.integers(n,size=(s,n))
        boot[b]=values[si[:,None],ei].mean(axis=(0,1))
    means=values.mean(axis=(0,1));rank=np.argsort(-means[2])
    result=[]
    for i,name in enumerate(items):
        def stat(j):return {'estimate':float(means[j,i]),'ci95':np.quantile(boot[:,j,i],[.025,.975]).tolist()}
        result.append({'name':name,'enable_drop':stat(0),'disable_recovery':stat(1),'score':stat(2),
          'on_success':float(outcomes(prefix+'_on_'+name,seeds).mean()),'off_success':float(outcomes(prefix+'_off_'+name,seeds).mean())})
    gap=boot[:,2,rank[0]]-boot[:,2,rank[1]] if len(rank)>1 else np.ones(replicates)
    gap_mean=float(means[2,rank[0]]-means[2,rank[1]]) if len(rank)>1 else 1.
    return {'seeds':list(seeds),'zero_success':float(zero.mean()),'full_success':float(full.mean()),'items':result,
       'ranking':[items[i] for i in rank],'top_gap':{'estimate':gap_mean,'ci95':np.quantile(gap,[.025,.975]).tolist()},
       'clear_leader':bool(gap_mean>=.05 and np.quantile(gap,.025)>0),
       'bootstrap':'95% paired hierarchical percentile; seed then env; exploratory during adaptive selection'}
