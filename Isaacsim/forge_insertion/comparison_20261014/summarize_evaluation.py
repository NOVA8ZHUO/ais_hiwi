from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parent
original=ROOT.parent/'experiments/forge_cad_baseline/logs/20261013T160114_110218Z_evaluate/evaluation.json'
paths={'random_on_random':original,**{n:ROOT/n/'evaluation.json' for n in ['zero_on_zero','zero_on_random','random_on_zero']}}
output=[]
for name,path in paths.items():
 if not path.exists():continue
 d=json.loads(path.read_text());rows=d['rows'];first=[r['first_success_seconds'] for r in rows if r['first_success_seconds'] is not None]
 item={'case':name,'n':len(rows),'success_final_count':sum(r['success_final'] for r in rows),
  'success_any_count':sum(r['success_any'] for r in rows),'reward_mean':d['mean_episode_reward'],
  'first_success_seconds_mean_success_only':float(np.mean(first)) if first else None,
  'episode_force_mean_N':float(np.mean([r['force_mean_N'] for r in rows])),
  'episode_peak_force_mean_N':float(np.mean([r['force_max_N'] for r in rows])),
  'episode_peak_force_p95_N':float(np.percentile([r['force_max_N'] for r in rows],95)),
  'episode_peak_force_max_N':max(r['force_max_N'] for r in rows),
  'final_depth_mean_mm':float(np.mean([r['final_insertion_depth_m']*1000 for r in rows])),
  'final_xy_error_mean_mm':float(np.mean([r['final_xy_error_m']*1000 for r in rows])),
  'episodes_exceeding_50mm_depth':d['cad_geometry']['episodes_exceeding_50mm_depth'],
  'report':str(path)}
 output.append(item)
(ROOT/'evaluation_summary.json').write_text(json.dumps(output,indent=2))
with (ROOT/'evaluation_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
print(json.dumps(output,indent=2))
