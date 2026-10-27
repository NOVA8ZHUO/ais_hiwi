from pathlib import Path
import json,math,re,csv
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
runs={'Randomized':ROOT.parent/'experiments/forge_cad_baseline/logs/20261013T115201_931054Z_train','Zero randomization':ROOT.parent/'0randomization/runs/20261013T160124Z_train'}
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
tags=['rewards/iter','successes/iter','logs_rew_curr_success/iter','losses/a_loss']
report={};allrows=[]
for label,run in runs.items():
    e=EventAccumulator(str(next(run.rglob('summaries'))),size_guidance={'scalars':0});e.Reload()
    series={k:e.Scalars(k) for k in e.Tags()['scalars']}
    print(label,'tags:',list(series))
    d={'run':str(run),'metrics':{},'nonfinite_scalars':sum(not math.isfinite(x.value) for v in series.values() for x in v)}
    log=(run/'console.log').read_text(errors='replace')
    d['training_time_seconds']=re.findall(r'Training time: ([\d.]+)',log)
    for tag in ['rewards/iter','successes/iter','logs_rew_curr_success/iter','losses/a_loss','losses/c_loss','losses/cval_loss']:
        if tag not in series:continue
        v=series[tag];vals=[x.value for x in v]
        d['metrics'][tag]={'last':vals[-1],'last20_mean':float(np.mean(vals[-20:])), 'last20_min':min(vals[-20:]),'max':max(vals)}
        if tag=='successes/iter':
            for threshold in [.9,.95,.99]:
                windows=[v[i].step for i in range(9,len(v)) if np.mean(vals[i-9:i+1])>=threshold]
                d['metrics'][tag]['first_10record_mean_ge_'+str(threshold)]=windows[0] if windows else None
            d['success_by_20epochs']=[{'start':i+1,'end':i+20,'mean':float(np.mean([x.value for x in v if i<x.step<=i+20]))} for i in range(0,200,20)]
    for k,v in series.items():
        allrows.extend({'model':label,'tag':k,'step':x.step,'value':x.value} for x in v)
    for ax,tag in zip(axes.flat,tags):
        if tag in series:
            v=series[tag];ax.plot([x.step for x in v],[x.value for x in v],label=label,alpha=.85)
        ax.set_title(tag);ax.set_xlabel('Epoch' if tag.endswith('/iter') else 'Logged step');ax.grid(alpha=.2);ax.legend()
    report[label]=d
fig.savefig(ROOT/'training_comparison.png',dpi=160)
(ROOT/'training_summary.json').write_text(json.dumps(report,indent=2))
with (ROOT/'training_scalars.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['model','tag','step','value']);w.writeheader();w.writerows(allrows)
print(json.dumps(report,indent=2))
