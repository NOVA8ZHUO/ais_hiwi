"""Scene-only adaptation: preserve heterogeneous assets already spawned per environment."""
import inspect,textwrap
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

# InteractiveScene already creates all env transforms when replicate_physics=False.
# Factory's final clone overwrites the heterogeneous meshes with env_0's mesh.
# Compile a private copy removing ONLY that clone; keep the installed setup otherwise.
source=textwrap.dedent(inspect.getsource(FactoryEnv._setup_scene))
line='self.scene.clone_environments(copy_from_source=False)'
assert source.count(line)==1,'Upstream setup changed; re-audit heterogeneous scene'
source=source.replace(line,'pass  # env transforms and per-object clones already exist')
namespace=FactoryEnv._setup_scene.__globals__.copy()
exec(compile(source,__file__,'exec'),namespace)
_setup_heterogeneous=namespace['_setup_scene']

class PGEnv(ForgeEnv):
    def _setup_scene(self):
        _setup_heterogeneous(self)
        self.scene.filter_collisions(global_prim_paths=['/World/ground'])
