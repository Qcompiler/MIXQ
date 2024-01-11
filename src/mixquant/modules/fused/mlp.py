import torch.nn as nn

import torch
from mixquant.Cache import MixLibCache, MLPCache





class MixLlamaMLP(nn.Module):

    def __init__(
        self,
        gate_proj,
        down_proj,
        up_proj,
        MixGemmCache = None
    ):
        super().__init__()



        self.down_proj_ = down_proj
        self.gate_proj_ = gate_proj
        self.up_proj_ = up_proj
        self.out_features = down_proj.out_features
        self.MLPCache = MLPCache()
        
 
    def forward(self, x):
 

        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])

        up_output = self.up_proj_(x, self.MLPCache)
        gate_output = self.gate_proj_.forward_without_preconditionFusedSilu(x, self.MLPCache)
        
        gate_output *= up_output
         

        y = self.down_proj_(gate_output, self.MLPCache, True)
 
        return y.reshape(out_shape)
    
 