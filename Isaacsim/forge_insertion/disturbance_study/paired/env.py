import hashlib,json,os
from pathlib import Path
import torch
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from samples import Tape
from design import RESET_SLOTS,IK_ATTEMPTS

class PairedEnv(ForgeEnv):
    def __init__(self,*args,**kwargs):
        self.tape=Tape(os.environ['FORGE_SAMPLE_FILE'])
        self.active=set(json.loads(os.environ['FORGE_ACTIVE_FACTORS']))
        self.eval_step=-1;self._resetting=True;self._reset_obs_count=0;self._obs_slot=0
        super().__init__(*args,**kwargs)

    def _select_observation_slot(self):
        if self._resetting:
            assert self._reset_obs_count<RESET_SLOTS,'Reset exceeded pre-generated observation budget'
            self._obs_slot=self._reset_obs_count;self._reset_obs_count+=1
        else:self._obs_slot=RESET_SLOTS+self.eval_step+1

    def _sample(self,key,context,shape,**kwargs):
        device=kwargs.get('device','cpu');v=self.tape.tensor(key,device)
        if key in ['hand_position','hand_orientation']:
            attempt=context['ik_attempt'];assert attempt<IK_ATTEMPTS,'IK retry budget exhausted'
            v=v[attempt,context['bad_envs'].to(device)]
        elif key in ['base_position','base_yaw']:
            v=v[context['env_ids'].to(device)]
        elif key in ['tip_position_noise','rotation_axis','rotation_angle','force_noise']:
            v=v[self._obs_slot]
            if key=='rotation_angle':v=v[:,0]
        elif key=='dead_zone':
            v=v[0] if key in self.active else torch.full_like(v[0],.5)
        elif key=='quaternion_sign' and key not in self.active:v=torch.zeros_like(v)
        expected=(shape,) if isinstance(shape,int) else tuple(shape)
        assert tuple(v.shape)==expected,(key,v.shape,expected)
        return v.to(dtype=kwargs.get('dtype',torch.float32)).clone()

    def _paired_gains(self,default_values,noise_levels,num_envs,device,key):
        magnitude=self.tape.tensor(key+'_magnitude',device)
        direction=self.tape.tensor(key+'_direction',device)>.5
        multiplier=1.+magnitude*torch.tensor(noise_levels,device=device)
        return default_values*torch.where(direction,1./multiplier,multiplier)

    def _reset_idx(self,env_ids):
        self._resetting=True;self._reset_obs_count=0
        self._paired_reset(env_ids)
        self._resetting=False

# Compile copies in private globals. No global monkeypatch of Isaac Lab or original tasks.
for name,data in json.loads(Path(__file__).with_name('adapter.json').read_text()).items():
    owner=FactoryEnv if name=='randomize_initial_state' else ForgeEnv
    original=getattr(owner,name)
    assert hashlib.sha256(Path(original.__code__.co_filename).read_bytes()).hexdigest()==data['original_file_sha256']
    namespace=original.__globals__.copy();namespace['FactoryEnv']=FactoryEnv
    exec(compile(data['method'],str(Path(__file__).with_name('adapted'+name+'.py.txt')),'exec'),namespace)
    setattr(PairedEnv,'_paired_reset' if name=='_reset_idx' else name,namespace[name])
