from pathlib import Path
import json,time,datetime,subprocess,sys
ROOT=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(ROOT.parent/'training_progress.py')],check=False)
p=ROOT/'status.json'
if not p.exists():print('评估队列尚未启动');raise SystemExit
s=json.loads(p.read_text());n=s.get('completed',0);total=s.get('total',1440)
print('\n后续评估阶段:',s.get('state'),'PID:',s.get('pid'))
print('['+'#'*int(30*n/total)+'-'*(30-int(30*n/total))+f'] {n}/{total} ({100*n/total:.1f}%)')
print('当前任务:',s.get('current','等待全部训练完成'))
if s.get('state')=='evaluating':
 sec=max(0,s.get('remaining_seconds',0)-(time.time()-s['updated']))
 print(f'预计评估剩余 {sec/3600:.1f} 小时；预计完成:',datetime.datetime.fromtimestamp(time.time()+sec).astimezone().isoformat(timespec='minutes'))
else:print('等待期间不估计正式评估完成时刻；开始评估后按实测速度更新。')
if s.get('error'):print('失败原因:',s['error'])
print('最终报告:',ROOT/'RESULTS.md')
