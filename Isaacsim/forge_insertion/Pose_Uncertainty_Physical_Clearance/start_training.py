from pathlib import Path
import subprocess,sys,time,fcntl
ROOT=Path(__file__).resolve().parent
(ROOT/'training').mkdir(exist_ok=True)
with (ROOT/'training/queue.lock').open('a') as guard:
 try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Queue already running')
with (ROOT/'training_queue.log').open('a') as log:
 child=subprocess.Popen([sys.executable,'-u',str(ROOT/'train_queue.py'),'--execute'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(ROOT/'training_queue.pid').write_text(str(child.pid))
time.sleep(3)
if child.poll() is not None:raise SystemExit('Exited; inspect training_queue.log')
print(f'Training pipeline started PID={child.pid}; includes all four preflight gates.')
