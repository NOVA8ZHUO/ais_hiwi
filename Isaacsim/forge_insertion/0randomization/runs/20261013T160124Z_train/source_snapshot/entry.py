import runpy
import sys
import forge_zero
script = sys.argv.pop(1)
runpy.run_path(script, run_name='__main__')
