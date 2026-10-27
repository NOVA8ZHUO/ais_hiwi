"""CPU inference only. No robot connection or actuator commands."""
from pathlib import Path
import torch

class Policy:
    def __init__(self, model_dir, seed=0):
        self.actor=torch.jit.load(str(Path(model_dir)/'policy_actor.ts'),map_location='cpu').eval()
        self.rng=torch.Generator(device='cpu').manual_seed(seed)
        self.reset()

    def reset(self):
        self.h=torch.zeros(2,1,1024)
        self.c=torch.zeros_like(self.h)

    @torch.inference_mode()
    def step(self, observation, *, mode):
        obs=torch.as_tensor(observation,dtype=torch.float32).reshape(1,24)
        if not torch.isfinite(obs).all():raise ValueError('Observation contains NaN/Inf')
        if mode=='stochastic':eps=torch.randn(1,7,generator=self.rng)
        elif mode=='mean':eps=torch.zeros(1,7)
        else:raise ValueError('Explicit mode required: stochastic or mean')
        action,mu,sigma,h,c=self.actor(obs,self.h,self.c,eps)
        if not all(torch.isfinite(x).all() for x in (action,mu,sigma,h,c)):
            raise ValueError('Non-finite policy output')
        self.h,self.c=h,c
        return action[0].clone()
