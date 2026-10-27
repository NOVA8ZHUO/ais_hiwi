import fcntl,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent
with (r/'pipeline.lock').open('a') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Worker already running; see status.json')
with (r/'pipeline.log').open('a') as log:
 child=subprocess.Popen([sys.executable,'-u',str(r/'pipeline.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'pipeline.pid').write_text(str(child.pid))
time.sleep(3)
if child.poll() is not None:raise SystemExit('Worker exited; see pipeline.log')
print(f'Started PID={child.pid}. Progress: {r}/status.json')
