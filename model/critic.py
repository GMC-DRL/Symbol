import torch
import torch.nn as nn
from .utils import MLP

class Critic(nn.Module):
    def __init__(self,opts) -> None:
        super().__init__()
        self.opts=opts
        self.input_dim=opts.fea_dim
        self.output_dim=opts.value_dim
        if self.opts.fea_mode=='xy':
            net_config = [{'in': self.input_dim, 'out': 16, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 16, 'out': self.output_dim, 'drop_out': 0, 'activation': 'None'}]
            self.value_net=MLP(net_config)
        else:
            self.value_net=nn.Linear(self.input_dim,self.output_dim)

    # return baseline value detach & baseling value
    def forward(self,x):
        # depend on diff
        # x=x[:,:self.opts.fea_dim]
        if self.opts.fea_mode=='xy':
            baseline_val=self.value_net(x).mean(-2)
        else:
            baseline_val=self.value_net(x)

        return baseline_val.detach().squeeze(),baseline_val.squeeze()