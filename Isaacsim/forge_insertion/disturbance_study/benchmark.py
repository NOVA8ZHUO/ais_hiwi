"""One standard-size validation job, excluded from formal results."""
import json
from runner import ROOT,run_case
from design import ALL
run_case('full',ALL,4321,n=128,area='benchmark')
p=ROOT/'benchmark/full/seed_4321/done.json'
d=json.loads(p.read_text())
(ROOT/'runtime_estimate.json').write_text(json.dumps({'measured_seconds_per_128_episode_job':d['seconds'],
 'normal_jobs':[130,160],'maximum_jobs':[290,380],
 'normal_hours':[d['seconds']*n/3600 for n in [130,160]],
 'maximum_hours':[d['seconds']*n/3600 for n in [290,380]]},indent=2))
print((ROOT/'runtime_estimate.json').read_text(),flush=True)
