import sys

sys.path.append('/home/chenyidong/SC/MixQ/src')
import torch
torch.manual_seed(0)
from mixquant.Cache import MixLibCache
cache = MixLibCache()
import mixlib
import time
M = 32
N = 12288
inputs = torch.randn((M,N),dtype=torch.float16,device='cuda')
inputs2 = torch.clone(inputs)
ind = torch.as_tensor([2,5,8,9],dtype=torch.int32,device='cuda')




def run1(ind,inputs):
    activation_outliers_return = mixlib.ExtractOutliersAndSetToZeros(ind,inputs)
    
    for i in range(10):
        x_scale = torch.max(inputs.abs(),dim=1)[0] / 127.0
        q_xcache = mixlib.Int8quantize(inputs,x_scale)

    start = time.time()
    repeat = 100
    for i in range(repeat):
        activation_outliers = mixlib.ExtractOutliersAndSetToZeros(ind,inputs)
        x_scale = torch.max(inputs.abs(),dim=1)[0] / 127.0
        q_xquant = mixlib.Int8quantize(inputs,x_scale)
    

    print("ave time of origion = %.8f"%((time.time()-start)/repeat))
    return q_xquant, activation_outliers_return


q_xquant, activation_outliers = run1(ind,inputs)

q_xquant_new, activation_outliers_new = mixlib.FindRowScaleFusedExtracOutliers(inputs2,cache.x_scale,ind,len(ind),M,N)
start = time.time()
repeat = 100
for i in range(repeat):
    _ , _ = mixlib.FindRowScaleFusedExtracOutliers(inputs,cache.x_scale,ind,len(ind),M,N)

print("ave time of opt = %.8f"%((time.time()-start)/repeat))


 
print(q_xquant) 
print(q_xquant_new)
print(activation_outliers)
print(activation_outliers_new)
#print(  torch.sum( torch.max(inputs.abs(),dim=1)[0].squeeze(0)[0:M] /127.0  - cache.x_scale[0:M].squeeze(0).T ))
 

# print(q_xquant[0:10])
# print(q_xquant_new[0:10])

 