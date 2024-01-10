import torch
from torch import nn
import mixlib
 

class FasterTransformerRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight.cuda().to(torch.float16)
        self.variance_epsilon = eps

    @torch.no_grad()
    def forward(self, x):


        output = torch.empty_like(x)

        mixlib.layernorm_forward_cuda(x, self.weight, output, self.variance_epsilon)
        return output 
