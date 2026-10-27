"""Actor only; input already follows FORGE observation semantics, before normalization."""
import torch
from torch import nn

class PortableActor(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.register_buffer('mean',reference.running_mean_std.running_mean.float().clone())
        self.register_buffer('variance',reference.running_mean_std.running_var.float().clone())
        self.epsilon=float(reference.running_mean_std.epsilon)
        net=reference.a2c_network
        self.rnn=net.rnn.rnn
        self.layer_norm=net.layer_norm
        self.mlp=net.actor_mlp
        self.mu=net.mu
        self.logstd=net.sigma

    def forward(self, observation: torch.Tensor, h: torch.Tensor, c: torch.Tensor, epsilon: torch.Tensor):
        x=((observation-self.mean)/torch.sqrt(self.variance+self.epsilon)).clamp(-5.,5.)
        x,(hn,cn)=self.rnn(x.unsqueeze(0),(h,c))
        x=self.mlp(self.layer_norm(x.squeeze(0)))
        mu=self.mu(x)
        sigma=torch.exp(self.logstd(x))
        action=(mu+sigma*epsilon).clamp(-1.,1.)
        return action,mu,sigma,hn,cn
