"""Register the independent CAD task, then execute the unchanged official entrypoint."""
import runpy
import sys
import forge_cad  # registration has no simulator imports
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
