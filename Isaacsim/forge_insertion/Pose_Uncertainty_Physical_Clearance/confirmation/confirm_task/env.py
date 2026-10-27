from grid_task.env import GridEnv
from pg_task.env import PGEnv
class ConfirmEnv(GridEnv):
 def _setup_scene(self):PGEnv._setup_scene(self)
