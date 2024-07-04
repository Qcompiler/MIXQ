 
import torch
import torch.nn as nn
import sys
import mixlib
import numpy as np
from torch.nn import functional as F
from vllm import _custom_ops as ops
import scipy

from torch import Tensor
def two_compl(x: Tensor, bits: int) -> Tensor:
    return torch.where(x < 0, 2 ** bits + x, x)
def pack_to_i4(X: Tensor):

    X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
    X_i4 = X_i8[:, 0::2] | (X_i8[:, 1::2] << 4)
    return X_i4

def unpack_int8_to_int4(weight,ind):
    assert weight.dim() == 2
    return mixlib.unpack_int4_to_fp16(weight,ind)
 

from EETQ import quant_weights,w8_a16_gemm
class MixLinear_GEMM(nn.Module):
    def __init__(self, in_features, out_features, bias, dev,  bit, 
            weight_only = False, cache = None, fp_features_num = 256, name = None):
        super().__init__()
        
 
        self.in_features = in_features
        self.out_features = out_features
        self.bit = bit
        self.name = name  
 


        if weight_only is False:
            self.register_buffer('scale_col', torch.empty((1,out_features), dtype=torch.float16, device=dev,requires_grad=False))

            if bit == 8:
                self.ind = torch.zeros((0,),dtype=torch.int32, device=dev)    
                self.register_buffer('q_weight', torch.empty((out_features,in_features), dtype=torch.int8, device=dev,requires_grad=False))
                self.weight_cache = None
            
            if bit == 4:
                self.fp_features_num = fp_features_num
                self.register_buffer('q_weight', torch.empty((out_features, (in_features )//2),
                                                     dtype=torch.uint8, device=dev,requires_grad=False))
                self.register_buffer('weight_cache', torch.empty((out_features, fp_features_num),
                                                                            device=dev,
                                                                            dtype=torch.float16, 
                                                                            requires_grad=False))
                self.register_buffer('ind', torch.empty(
                    (fp_features_num), dtype=torch.int32,device=dev, requires_grad=False)) 

        else:

            self.register_buffer('q_weight', torch.empty((in_features,out_features), dtype=torch.int8, device=dev,requires_grad=False))
            self.register_buffer('scale_col', torch.empty((out_features), dtype=torch.float16, device=dev,requires_grad=False))
        

        if bias:

            self.register_buffer('bias', torch.empty((out_features), dtype=torch.float16, device=dev,requires_grad=False))
        else:
            self.bias = None
        self.cnt = 0
        self.forward_without_precondition_len = -1

        self.cache = cache
        self.weight_only = weight_only


        self.add_outliers = True

         
        if cache is not None:
            self.sigma = torch.ones((1, 1),dtype=torch.float16, requires_grad=False,
                                            device = dev)
            self.sigma[0] = cache.sigma

        self.arch = torch.cuda.get_device_capability()[0]

        self.scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        self.scale_b = torch.ones((out_features, 1), device="cuda", dtype=torch.float32)
        self.use_old_kernel = True
        self.cnt_debug = 0
    @classmethod
    def from_linear(cls, linear, bit, weight_only=False, init_only=False,cache=None, 
                    layer_scales= None, dev = 'cuda', name = None):


        quant_linear = cls(linear.in_features, linear.out_features, linear.bias is not None, 
                           dev, bit=bit, weight_only=weight_only,
                           cache=cache, name = name)
   

        if init_only is True: 
            return quant_linear   
             
        if weight_only is True:
            int8_weight_cpu = torch.t(linear.weight.data).contiguous().cpu()
            int8_weight, scales = quant_weights(int8_weight_cpu, torch.int8, False)
            quant_linear.q_weight.copy_ (int8_weight)
            quant_linear.scale_col.copy_(scales.half())

        else:


            if bit == 8:

                scale =   (torch.max(torch.abs(linear.weight.data), dim=1)[0].unsqueeze(1) / (
                                127)).to(torch.float16).reshape((1,linear.out_features))
                quant_linear.scale_col.copy_(scale)
                tmp = linear.weight.data.cuda()
                tmp /= quant_linear.scale_col.T
                tmp = tmp.round().to(torch.int8)
                quant_linear.q_weight.copy_(tmp)   
            else:


                assert layer_scales is not None
                fp_features = quant_linear.fp_features_num
                linear.ind = torch.sort(layer_scales)[1][-fp_features:]
                 
      
                tmp = linear.weight.data.cuda()                
                quant_linear.weight_cache.copy_(tmp[:, linear.ind].to(tmp.dtype).cuda())  



                tmp = linear.weight.data.cuda()
                tmp[:, linear.ind] = 0

                scale =   (torch.max(torch.abs(tmp), dim=1)[0].unsqueeze(1) / (10)).to(torch.float16).reshape((1,linear.out_features))
                quant_linear.scale_col.copy_(scale)
                tmp /= quant_linear.scale_col.T

                tmp = torch.clamp(tmp.round(), -8, 7)  
                tmp = pack_to_i4(tmp.to(torch.int8).cpu())
                quant_linear.q_weight.copy_(tmp.cuda()) 
                quant_linear.ind.copy_(linear.ind.cuda().to(torch.int32))
 


        if linear.bias is not None:
            quant_linear.bias.copy_(linear.bias.half())

        return quant_linear
    
 
         

    
    @torch.no_grad()
    def FindOutliers(self,Activation, sigma = None):

        if sigma is None:
            sigma = self.sigma
        
        tmp = torch.unique(torch.where((  Activation.abs() > sigma ))[1])
        return tmp.to(torch.int32)


    @torch.no_grad()
    def forward(self, x, cache = None, unfused = False, bench_gemm = False):

        if cache is  None:
            cache = self.cache




  
        cache.shape = x.shape[:-1] + (self.out_features, )

        inputs = x.reshape(-1, x.shape[-1])
        M =  inputs.shape[0]


        # record the outliers in H100, please remove the code when inference
        dev = int(torch.cuda.get_device_properties(0).major)
        if dev >= 9:
            unfused = True 
            # record the outliers in H100, please remove the code when inference
            
            #print(self.name)
            if   1 : 
                if 1:
                    # act = torch.max(torch.abs(inputs), dim=0)[0].cpu().numpy()

                    # print(act)
                    # scipy.io.savemat("/home/dataset/tmp/tmp/" +  str(self.cnt_debug)  + self.name + ".mat", {"data" : act})  
                    # self.cnt_debug += 1
                    #exit(0)
                    if self.name is not None:
                        
                        if "down" in self.name:
                            sigma = 50
                            if self.cnt_debug <= 50:
                                sigma = max(self.cnt_debug / 2, 6 )
                            self.cnt_debug += 1
                            ind = self.FindOutliers(inputs, sigma).cpu().numpy()
                        elif "up"  in self.name :
                            sigma = 6
                            # if self.cnt_debug <= 50:
                            #     sigma = 4   
                            self.cnt_debug += 1
                            ind = self.FindOutliers(inputs, sigma).cpu().numpy()
                        elif  "dense"  in self.name :
                            sigma = 6
                            if self.cnt_debug <= 50:
                                sigma = 4.5 
                            self.cnt_debug += 1
                            ind = self.FindOutliers(inputs, sigma).cpu().numpy()

                        else:
                            
                            ind = self.FindOutliers(inputs).cpu().numpy()
                        name = self.name
                        c = ""  
                        f = open("/home/dataset/tmp/llama13b/" + name + ".csv", "a+")
                        if len(ind):
                            
                            
                            
                            for j in ind:
                                c = c + str(j) + ","
                        else:
                            c = "0,"
                        f.writelines(c+"\n")
                        f.close()
                    
            else:
                pass
            
        if self.weight_only is True:

            y =  w8_a16_gemm(inputs, self.q_weight, self.scale_col)

            if self.bias is not None:
                y += self.bias
            return y.reshape(cache.shape)
 
    

        if unfused :
            if self.ind.shape[0]:
                cache.activation_outliers = mixlib.ExtractOutliersAndSetToZeros(self.ind, inputs)
                
            if self.use_old_kernel:
                cache.q_xcache = mixlib.FindRowScale(inputs,cache.x_scale, M, self.in_features ,self.bit)
            else:
                # for sm90 we use the new kernel
                # to do !! fuse the following kernel
                cache.x_scale =   (torch.max(torch.abs(inputs), dim=1)[0].unsqueeze(1) / (
                            127)).to(torch.float16).reshape((M,1))

                tmp /= inputs.scale
                cache.q_xcache = tmp.round().to(torch.int8) 

        cache.ind = self.ind

 
        if self.add_outliers:
            if cache.x_scale[0:M].max() > self.sigma / ((  2 ** (self.bit - 1) - 1  )  )  :
                 
                ind = self.FindOutliers(inputs)

                activation_outliers = mixlib.ExtractOutliersAndSetToZeros(ind,inputs)
                if self.bit == 8:
                    weight_cache = self.q_weight[:,ind].to(torch.float16) *  self.scale_col.T
                else:
                    w = unpack_int8_to_int4(self.q_weight, ind)
                    weight_cache = w *  self.scale_col.T
 
                if self.ind.shape[0] == 0:
                    cache.activation_outliers = activation_outliers
                    self.weight_cache =  weight_cache
                else:
                    cache.activation_outliers =  torch.hstack((cache.activation_outliers,activation_outliers))
                    self.weight_cache =  torch.hstack((self.weight_cache,weight_cache))
                self.ind = torch.hstack((self.ind,ind))
                cache.ind = self.ind
                cache.q_xcache = mixlib.FindRowScale(inputs,cache.x_scale, M, self.in_features ,self.bit)

                # if not self.arch == 9:
                #     cache.q_xcache = mixlib.FindRowScale(inputs,cache.x_scale, M, self.in_features ,self.bit)
                # else:
                #     cache.x_scale =   (torch.max(torch.abs(inputs), dim=1)[0].unsqueeze(1) / (
                #                 127)).to(torch.float32).reshape((M,1))

                #     tmp = inputs / cache.x_scale.to(torch.float16)
                #     cache.q_xcache = tmp.round().to(torch.int8)
                
                
            self.cnt += 1
            if self.cnt >= self.cache.stop or self.ind.shape[0] > 256:
                self.add_outliers = False
             

 

        
 


            # if self.ind.shape[0]:
            #     # get the same result with bench gemm = false 
            #     outliers_fp16 = F.linear(cache.activation_outliers ,  self.weight_cache)
            #     y1 =   ops.cutlass_scaled_mm(
            #                 cache.q_xcache,
            #                 self.q_weight.T,
            #                 out_dtype=torch.float16,
            #                 scale_a=cache.x_scale,
            #                 scale_b=self.scale_b,
            #     )
            #     y1 += outliers_fp16

            # else:
            #     y1 =   ops.cutlass_scaled_mm(
            #                 cache.q_xcache,
            #                 self.q_weight.T,
            #                 out_dtype=torch.float16,
            #                 scale_a=cache.x_scale,
            #                 scale_b=self.scale_b,
            #     )
            # return y1
        
        if self.arch == 9:
            
            y = mixlib.gemm(cache.q_xcache,self.q_weight,M, self.out_features, self.in_features)
            if self.ind.shape[0]:

     
                outliers_fp16 = F.linear(cache.activation_outliers ,  self.weight_cache)
                y1 = mixlib.dequantizeInt8(y, cache.x_scale, self.scale_col, outliers_fp16, 8, M, self.out_features)
 
                
                
            else:
                y1 = mixlib.dequantizeInt8(y, cache.x_scale, self.scale_col, self.cache.zeros, 8, M, self.out_features)
                
            
            return y1.reshape(cache.shape)
        else:

            if self.ind.shape[0]:
                
                outliers_fp16 = F.linear(cache.activation_outliers ,  self.weight_cache)
                if self.bit == 8:
                    
                    y1 = mixlib.int8FusedDequantize(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            outliers_fp16,
                                                            M,self.out_features,self.in_features)  
                if self.bit == 4:
                     
                    y1 = mixlib.int4FusedDequantize(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            outliers_fp16,
                                                            M,self.out_features, (self.in_features ) // 2)                      
            else:
                if self.bit == 8:    
                    y1 = mixlib.int8FusedDequantize(cache.q_xcache, 
                                                    self.q_weight, 
                                                    cache.x_scale,
                                                    self.scale_col,
                                                    self.cache.zeros,
                                                    M,self.out_features,self.in_features)  

                    
                if self.bit == 4:  
                      
                    y1 = mixlib.int4FusedDequantize(cache.q_xcache, 
                                                    self.q_weight, 
                                                    cache.x_scale,
                                                    self.scale_col,
                                                    self.cache.zeros,
                                                    M,self.out_features,(self.in_features )// 2) 
        if self.bias is not None:
            y1 += self.bias
        
        #print(self.ind.shape[0])
 
        return y1.reshape(cache.shape)

    @torch.no_grad()
    def forward_without_preconditionFusedSilu(self, x, cache):
        

        inputs = x.reshape(-1, x.shape[-1])
        M =  inputs.shape[0]


        if not self.forward_without_precondition_len ==  cache.ind.shape[0]:
            self.ind = cache.ind
            if self.ind.shape[0]:
                if self.bit == 8:
                    self.weight_cache = self.q_weight[:,self.ind].to(torch.float16) 
                    self.weight_cache *=  self.scale_col.T
                # add new fp weight
                if self.bit == 4:
                    self.weight_cache = self.weight_cache

                self.forward_without_precondition_len = self.ind.shape[0]

 
 
      
        if self.arch == 9:
            y = mixlib.gemm(cache.q_xcache,self.q_weight,M, self.out_features, self.in_features)
            if self.ind.shape[0]:
                #outliers_fp16 = torch.mm( cache.activation_outliers ,  self.weight_cache.T)
                outliers_fp16 = F.linear(cache.activation_outliers ,  self.weight_cache)
                y1 = mixlib.dequantizeInt8Silu(y, cache.x_scale, self.scale_col, outliers_fp16, 8, M, self.out_features)
                
            else:
                y1 = mixlib.dequantizeInt8Silu(y, cache.x_scale, self.scale_col, self.cache.zeros, 8, M, self.out_features)
                

        else:    
            if self.bit == 8:        
                if self.ind.shape[0]:
 
                    outliers_fp16 = F.linear(cache.activation_outliers ,  self.weight_cache)
                
                    
                    y1 = mixlib.int8FusedDequantizeSilu(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            outliers_fp16,
                                                            M,self.out_features,self.in_features)  
                    
                else:

                    y1 = mixlib.int8FusedDequantizeSilu(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            self.cache.zeros,
                                                            M,self.out_features,self.in_features )  

            if self.bit == 4:        
                if self.ind.shape[0]:


                    outliers_fp16 = F.linear(cache.activation_outliers ,  self.weight_cache)
                    
                    y1 = mixlib.int4FusedDequantizeSilu(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            outliers_fp16,
                                                            M,self.out_features,
                                                            (self.in_features )// 2)  
                    
                else:
 
                    raise RuntimeError("int4 mod should have outliers !")

        if self.bias is not None:
            y1 += self.bias

        
        return y1.reshape(cache.shape)
    


