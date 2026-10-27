from pathlib import Path
import json,re,time
from datetime import datetime
ROOT=Path(__file__).resolve().parent
p=ROOT/'training_status.json'
if not p.exists():print('No training status yet');raise SystemExit()
s=json.loads(p.read_text());completed=len(list((ROOT/'training').glob('P*/seed_*/done.json')));epoch=0
console=Path(s.get('console','/nonexistent'))
if s['state']=='training' and console.exists():
 text=console.read_text(errors='replace')[-200000:];found=re.findall(r'epoch:\s*(\d+)',text)
 if found:epoch=int(found[-1])
print(f"State: {s['state']} | group={s.get('group')} seed={s.get('seed')}\nFinished runs: {completed}/12 | current epoch: {epoch}/200")
n=completed*200+epoch
print('['+('#'*int(30*n/2400)).ljust(30,'-')+f'] {n}/2400 scheduled epochs ({100*n/2400:.1f}%)')
if epoch>=5:
 elapsed=time.time()-s['current_run_started_unix'];remain=(2400-n)*elapsed/epoch
 print(f'Approx remaining {remain/3600:.1f} h; ETA {datetime.fromtimestamp(time.time()+remain).astimezone().isoformat(timespec="minutes")} (includes current initialization overhead; groups may run at different speeds)')
if s.get('error') and s['state']=='failed':print(s['error'])
if s['state']=='training':print('Current log:',console)
