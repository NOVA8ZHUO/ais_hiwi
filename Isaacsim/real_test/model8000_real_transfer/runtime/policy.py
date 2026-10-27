"""CPU/GPU mean-action inference only. No robot commands are sent."""
from pathlib import Path
import torch
from torch import nn

class PortableActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean',torch.zeros(1,29))
        self.register_buffer('std',torch.ones(1,29))
        self.rnn=nn.LSTM(29,128,1)
        self.actor=nn.Sequential(nn.Linear(128,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,5))
    def forward(self,obs:torch.Tensor,h:torch.Tensor,c:torch.Tensor):
        x=(obs-self.mean)/(self.std+0.01)
        y,(hn,cn)=self.rnn(x.unsqueeze(0),(h,c))
        raw=self.actor(y.squeeze(0))
        return raw,torch.clamp(raw,-1.,1.),hn,cn

class Policy:
    def __init__(self,device='cpu'):
        self.device=torch.device(device)
        self.model=torch.jit.load(str(Path(__file__).resolve().parents[1]/'policy_actor.ts'),map_location=self.device).eval()
        self.reset()
    def reset(self):
        self.h=torch.zeros(1,1,128,device=self.device)
        self.c=torch.zeros_like(self.h)
    @torch.inference_mode()
    def step(self,observation):
        obs=torch.as_tensor(observation,dtype=torch.float32,device=self.device).reshape(1,29)
        if not torch.isfinite(obs).all():raise ValueError('Non-finite observation; do not issue robot action')
        raw,action,h,c=self.model(obs,self.h,self.c)
        if not all(torch.isfinite(x).all() for x in (raw,action,h,c)):
            raise ValueError('Non-finite policy output; do not issue robot action')
        self.h,self.c=h,c
        return action[0].cpu().tolist()

if __name__=='__main__':
    p=Policy();print('Inference only, synthetic observation:',p.step([0.]*29))
