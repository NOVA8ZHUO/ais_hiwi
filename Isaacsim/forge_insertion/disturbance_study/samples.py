"""Condition-independent, named NumPy streams; samples persisted before simulation."""
import hashlib,json
from pathlib import Path
import numpy as np
from design import N,STEPS,RESET_SLOTS,IK_ATTEMPTS

def seed_for(seed,key):
    return int.from_bytes(hashlib.sha256(f'forge-paired-v1:{seed}:{key}'.encode()).digest()[:8],'little')
def generate(path,seed,n=N):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['seed']==seed and meta['num_envs']==n
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
        return
    arrays={}
    def draw(key,shape,kind='uniform'):
        rng=np.random.default_rng(seed_for(seed,key))
        arrays[key]=(rng.standard_normal(shape) if kind=='normal' else rng.random(shape)).astype('float32')
    for key in ['base_position','base_yaw','grasp_offset']:
        draw(key,(n,3))
    draw('hole_observation',(n,3),'normal')
    for key in ['hand_position','hand_orientation']:draw(key,(IK_ATTEMPTS,n,3))
    for key in ['gains','position_threshold','rotation_threshold']:
        d=6 if key=='gains' else 3
        draw(key+'_magnitude',(n,d));draw(key+'_direction',(n,d))
    draw('ema',(n,1));draw('contact_threshold',(n,));draw('quaternion_sign',(n,))
    draw('dead_zone',(8,n,6));draw('mass',(n,1))
    draw('friction_buckets',(128,3));draw('friction_ids',(n,16))
    for key,d in [('tip_position_noise',3),('rotation_axis',3),('rotation_angle',1),('force_noise',3)]:
        draw(key,(RESET_SLOTS+STEPS+2,n,d),'normal')
    draw('policy',(STEPS,n,7),'normal')
    np.savez_compressed(path,**arrays)
    path.with_suffix('.json').write_text(json.dumps({'seed':seed,'num_envs':n,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'stream_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},'shapes':{k:list(v.shape) for k,v in arrays.items()}},indent=2))

class Tape:
    def __init__(self,path):
        self.path=Path(path);self.meta=json.loads(self.path.with_suffix('.json').read_text())
        assert hashlib.sha256(self.path.read_bytes()).hexdigest()==self.meta['sha256']
        with np.load(path) as f:self.data={k:f[k] for k in f.files}
        self.tensors={}
    def tensor(self,key,device='cpu'):
        import torch
        cache=(key,str(device))
        if cache not in self.tensors:self.tensors[cache]=torch.from_numpy(self.data[key]).to(device)
        return self.tensors[cache]
