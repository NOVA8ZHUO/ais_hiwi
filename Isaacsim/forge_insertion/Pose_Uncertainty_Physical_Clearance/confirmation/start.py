"""Attach a resumable downstream queue without restarting training."""
from pathlib import Path
import subprocess,sys,time,fcntl
ROOT=Path(__file__).resolve().parent
with (ROOT/'workflow.lock').open('a') as guard:
 try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Confirmation queue already running; no duplicate started.')
with (ROOT/'queue.log').open('a') as log:
 child=subprocess.Popen([sys.executable,'-u',str(ROOT/'workflow.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
if child.poll() is not None:raise SystemExit('Queue exited; inspect confirmation/queue.log')
print(f'Confirmation queue attached PID={child.pid}. Waiting for all 12 training checkpoints.')
