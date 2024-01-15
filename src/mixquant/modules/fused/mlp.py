import torch.nn as nn

import torch
from mixquant.Cache import MixLibCache, MLPCache




import time
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
        #start = time.time()
        up_output = self.up_proj_(x, self.MLPCache)

        #print("up time %.8f"%(time.time() - start))
        #start = time.time()
        gate_output = self.gate_proj_.forward_without_preconditionFusedSilu(x, self.MLPCache)
        
        #print("gate_output time",time.time() - start)
        gate_output *= up_output
         
        #start = time.time()
        y = self.down_proj_(gate_output, self.MLPCache, True)
        #print("down time",time.time() - start)
 
        return y.reshape(out_shape)
    
 