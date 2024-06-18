import sys

sys.path.append('/home/chenyidong/SC/MixQ/src')
import torch
torch.manual_seed(0)
from mixquant.Cache import MixLibCache
cache = MixLibCache()
import mixlib
M = 32
N = 12288
inputs = torch.randn((M,N),dtype=torch.float16,device='cuda')

import time
for i in range(10):
    x_scale = torch.max(inputs.abs(),dim=1)[0] / 127.0
    q_xcache = mixlib.Int8quantize(inputs,x_scale)

start = time.time()
repeat = 100
for i in range(repeat):
    x_scale = torch.max(inputs.abs(),dim=1)[0] / 127.0
    q_xquant = mixlib.Int8quantize(inputs,x_scale)

print("ave time of origion = %.8f"%((time.time()-start)/repeat))



start = time.time()
repeat = 100
for i in range(repeat):
    q_xquant_new = mixlib.FindRowScale(inputs,cache.x_scale,M,N)

print("ave time of opt = %.8f"%((time.time()-start)/repeat))


 
print(    ) 
print(  torch.sum( torch.max(inputs.abs(),dim=1)[0].squeeze(0)[0:M] /127.0  - cache.x_scale[0:M].squeeze(0).T ))
 

# print(q_xquant[0:10])
# print(q_xquant_new[0:10])

 