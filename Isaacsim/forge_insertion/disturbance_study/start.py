"""Detach one resumable pipeline; terminal may be closed after this command."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
with (ROOT/'pipeline.lock').open('a') as f:
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('An experiment is already running. See status.json / pipeline.log.')
from runner import verify_lock,require_preflight
verify_lock()
require_preflight()
with (ROOT/'pipeline.log').open('a') as log:
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'pipeline.py'),'--resume'],cwd=ROOT,
         stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(ROOT/'pipeline.pid').write_text(str(child.pid)+'\n')
time.sleep(1)
if child.poll() is not None:raise SystemExit('Pipeline exited immediately. Inspect pipeline.log.')
print(f'Experiment started, PID={child.pid}. You may close this terminal.\nProgress: {ROOT}/status.json\nLog: {ROOT}/pipeline.log\nFinal report: {ROOT}/RESULTS.md')
