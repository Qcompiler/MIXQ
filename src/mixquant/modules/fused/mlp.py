import torch.nn as nn

import torch
from mixquant.Cache import MixLibCache





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
        self.MixGemmCache = None
        

        self.down_proj_.SetSigma(1.0)
 
    def forward(self, x):
        if self.MixGemmCache is None:
            if len(x.shape) == 3:
                self.MixGemmCache = MixLibCache(x.shape[0] * x.shape[1])
            else:
                self.MixGemmCache = MixLibCache(x.shape[0])

 
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.reshape(-1, x.shape[-1])

        up_output = self.up_proj_(x, self.MixGemmCache)
        gate_output = self.gate_proj_.forward_without_preconditionFusedSilu(x, self.MixGemmCache)
        
        gate_output *= up_output
         

        y = self.down_proj_(gate_output, self.MixGemmCache, True)
 
        return y.reshape(out_shape)
    
 