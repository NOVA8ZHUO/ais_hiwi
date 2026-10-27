import json,time
from datetime import datetime
from pathlib import Path
s=json.loads(Path(__file__).with_name('status.json').read_text());n=s['completed'];total=s['total']
print(f"[{'#'*int(30*n/total):<30}] {n}/{total} ({100*n/total:.1f}%) {s['status']}")
print('Current:',s.get('current','small-scale preflight'))
if s['status']=='running':
 remain=max(0,s.get('estimated_remaining_seconds',0)-(time.time()-s['updated']))
 print(f'Remaining ~{remain/3600:.2f} h; ETA {datetime.fromtimestamp(time.time()+remain).astimezone().isoformat(timespec="seconds")}')
if 'error' in s:print(s['error'])
