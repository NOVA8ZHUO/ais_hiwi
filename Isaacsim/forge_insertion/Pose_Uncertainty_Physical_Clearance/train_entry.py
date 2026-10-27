import runpy,sys
import pg_task
script=sys.argv.pop(1)
runpy.run_path(script,run_name='__main__')
