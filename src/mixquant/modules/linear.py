 
import torch
import torch.nn as nn
import sys
import mixlib

 
 
from EETQ import quant_weights, preprocess_weights, w8_a16_gemm

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
 


class MixLinear_GEMM(nn.Module):
    def __init__(self, in_features, out_features, bias, dev,  bit, 
            weight_only = False, cache = None, fp_features_num = 256):
        super().__init__()
        
 
        self.in_features = in_features
        self.out_features = out_features
        self.bit = bit
 


        if weight_only is False:
            self.register_buffer('scale_col', torch.empty((1,out_features), dtype=torch.float16, device=dev,requires_grad=False))

            if bit == 8:

                self.register_buffer('q_weight', torch.empty((out_features,in_features), dtype=torch.int8, device=dev,requires_grad=False))
            if bit == 4:
                self.fp_features_num = fp_features_num
                self.register_buffer('q_weight', torch.empty((out_features, (in_features )//2),
                                                     dtype=torch.uint8, device=dev,requires_grad=False))
                self.register_buffer('fp_weight', torch.empty((out_features, fp_features_num),
                                                                            device=dev,
                                                                            dtype=torch.float16, 
                                                                            requires_grad=False))
                self.register_buffer('fp_indices', torch.empty(
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
        self.quant = True
        self.cache = cache
        self.weight_only = weight_only


        self.ind = None
        self.add_outliers = True

        if cache is not None:
            self.sigma = torch.ones((1, 1),dtype=torch.float16, requires_grad=False,
                                            device = cache.sigma.device)
            self.sigma[0] = cache.sigma
        self.weight_cache = None
        self.arch = torch.cuda.get_device_capability()[0]

    @classmethod
    def from_linear(cls, linear, bit, weight_only=False, init_only=False,cache=None, layer_scales= None):


        quant_linear = cls(linear.in_features, linear.out_features, linear.bias is not None, 
                           linear.weight.device,bit=bit, weight_only=weight_only,cache=cache)
   

        if init_only is True: 
            return quant_linear   
             
        if weight_only is True:
            int8_weight_cpu = torch.t(linear.weight.data).contiguous().cpu()
            int8_weight, scales = quant_weights(int8_weight_cpu, torch.int8, False)
            quant_linear.q_weight.copy_ (int8_weight)
            quant_linear.scale_col.copy_(scales.half())

        else:

 





            if bit == 8:


                #self.scale_col = self.scale.repeat((1,self.N))
                
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
                linear.fp_indices = torch.sort(layer_scales)[1][-fp_features:]
                 
      
                tmp = linear.weight.data.cuda()                
                quant_linear.fp_weight.copy_(tmp[:, linear.fp_indices].to(tmp.dtype).cuda())  



                tmp = linear.weight.data.cuda()
                tmp[:, linear.fp_indices] = 0

                scale =   (torch.max(torch.abs(tmp), dim=1)[0].unsqueeze(1) / (10)).to(torch.float16).reshape((1,linear.out_features))
                quant_linear.scale_col.copy_(scale)
                tmp /= quant_linear.scale_col.T

                tmp = torch.clamp(tmp.round(), -8, 7)  
                tmp = pack_to_i4(tmp.to(torch.int8).cpu())
                quant_linear.q_weight.copy_(tmp.cuda()) 
                quant_linear.fp_indices.copy_(linear.fp_indices.cuda().to(torch.int32))
 


        if linear.bias is not None:
            quant_linear.bias.copy_(linear.bias.half())

        return quant_linear
    

    @torch.no_grad()
    def SetSigma(self,sigma):
        self.sigma[0] = sigma
         

    
    @torch.no_grad()
    def FindOutliers(self,Activation):

        
        tmp = torch.unique(torch.where((  Activation.abs() > self.sigma ))[1])
        return tmp.to(torch.int32)
    @torch.no_grad()
    def ExtractFP16weight(self):
        if len(self.ind) == 0:
            return
        assert self.ind is not None
        assert self.weight_cache is  None
        if self.bit == 8:
            self.weight_cache = self.q_weight[:,self.ind].to(torch.float16)
            self.weight_cache *=  self.scale_col.T
            self.q_weight[:,self.ind] *= 0
        if self.bit == 4:
            self.weight_cache = self.fp_weight  

 
 
        return self.weight_cache

    @torch.no_grad()
    def forward(self, x, cache = None, weight_only = False):
        #memory = torch.cuda.memory_allocated()/1024/1024
        #print("start forward",memory)
        #print(self.weight_only)



        #torch.cuda.set_stream(cache.stream)
        if cache is  None:
            cache = self.cache

 
        cache.shape = x.shape[:-1] + (self.out_features, )

 
        inputs = x.reshape(-1, x.shape[-1])
 



        M =  inputs.shape[0]
        


        if self.weight_only is True:

            y =  w8_a16_gemm(inputs, self.q_weight, self.scale_col)

            if self.bias is not None:
                y += self.bias
            return y.reshape(cache.shape)
        if self.cache.is_prefill:
            weight_cache = self.q_weight.to(torch.float16) *  self.scale_col.T            
            y = torch.mm( inputs ,weight_cache.T  )      
            return y.reshape(cache.shape)
     


 
        if self.ind is None:
            if self.bit == 8:
                self.ind = self.FindOutliers(inputs)
            else:
                self.ind = self.fp_indices
            cache.ind = self.ind
 
            self.weight_cache = self.ExtractFP16weight()  
 
        if len(self.ind):  
            cache.activation_outliers = mixlib.ExtractOutliersAndSetToZeros(self.ind,inputs)

            outliers_fp16 = torch.mm( cache.activation_outliers ,  self.weight_cache.T)

 
        cache.q_xcache = mixlib.FindRowScale(inputs,cache.x_scale, M, self.in_features ,self.bit)


 

        if self.add_outliers:
            if cache.x_scale[0:M].max() > self.sigma / ((  2 ** (self.bit - 1) - 1  )  )  :
                 
                ind = torch.unique(torch.where((  inputs.abs() > self.sigma ))[1])
                ind = ind.to(torch.int32)
                activation_outliers = mixlib.ExtractOutliersAndSetToZeros(ind,inputs)
                if self.bit == 8:
                    weight_cache = self.q_weight[:,ind].to(torch.float16) *  self.scale_col.T
                else:
                    w = unpack_int8_to_int4(self.q_weight, ind)
                    weight_cache = w *  self.scale_col.T
 
                if len(self.ind) == 0:
                    cache.activation_outliers = activation_outliers
                    self.weight_cache =  weight_cache
                else:
                    cache.activation_outliers =  torch.hstack((cache.activation_outliers,activation_outliers))
                    self.weight_cache =  torch.hstack((self.weight_cache,weight_cache))
                self.ind = torch.hstack((self.ind,ind))
                cache.ind = self.ind



                cache.q_xcache = mixlib.FindRowScale(inputs,cache.x_scale, M, self.in_features ,self.bit)
                outliers_fp16 = torch.mm( cache.activation_outliers ,  self.weight_cache.T) 
                
            self.cnt += 1
            if self.cnt >= self.cache.stop or len(self.ind) > 256:
                self.add_outliers = False
             
            #print("after add outliers",torch.cuda.memory_allocated()/1024/1024 - memory)

 

        
        
        
        if self.arch == 9:
            y = mixlib.gemm(cache.q_xcache,self.q_weight,M, self.out_features, self.in_features)
            if len(self.ind):
                
                y1 = mixlib.dequantizeInt8(y, cache.x_scale, self.scale_col, outliers_fp16, 8, M, self.out_features)
                
            else:
                y1 = mixlib.dequantizeInt8(y, cache.x_scale, self.scale_col, self.cache.zeros, 8, M, self.out_features)
                

        else:

            if len(self.ind):
                

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
        
        #print(len(self.ind))
 
        return y1.reshape(cache.shape)

    @torch.no_grad()
    def forward_without_preconditionFusedSilu(self, x, cache):
        

        inputs = x.reshape(-1, x.shape[-1])
        M =  inputs.shape[0]
        assert M == cache.shape[0]

        if self.cache.is_prefill:
            weight_cache = self.q_weight.to(torch.float16) *  self.scale_col.T
            y = torch.mm( inputs ,weight_cache.T  )      
            return y.reshape(cache.shape)
        

      
        if self.weight_only is True:

            y =  w8_a16_gemm(inputs, self.q_weight, self.scale_col)

            if self.bias is not None:
                y += self.bias
            return y.reshape(cache.shape)



        if not self.forward_without_precondition_len == len(cache.ind):
            self.ind = cache.ind
            if len(self.ind):
                if self.bit == 8:
                    self.weight_cache = self.q_weight[:,self.ind].to(torch.float16) 
                    self.weight_cache *=  self.scale_col.T
                if self.bit == 4:
                    self.weight_cache = self.fp_weight

                self.forward_without_precondition_len = len(self.ind)

 
 
      
        if self.arch == 9:
            y = mixlib.gemm(cache.q_xcache,self.q_weight,M, self.out_features, self.in_features)
            if len(self.ind):
                outliers_fp16 = torch.mm( cache.activation_outliers ,  self.weight_cache.T)
                y1 = mixlib.dequantizeInt8Silu(y, cache.x_scale, self.scale_col, outliers_fp16, 8, M, self.out_features)
                
            else:
                y1 = mixlib.dequantizeInt8Silu(y, cache.x_scale, self.scale_col, self.cache.zeros, 8, M, self.out_features)
                

        else:    
            if self.bit == 8:        
                if len(self.ind):
 
                    
                    outliers_fp16 = torch.mm( cache.activation_outliers,  self.weight_cache.T)
                
                    
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
                if len(self.ind):


                    outliers_fp16 = torch.mm( cache.activation_outliers,  
                    self.weight_cache.T)
                    
                    y1 = mixlib.int4FusedDequantizeSilu(cache.q_xcache, 
                                                            self.q_weight, 
                                                            cache.x_scale,
                                                            self.scale_col,
                                                            outliers_fp16,
                                                            M,self.out_features,
                                                            (self.in_features )// 2)  
                    
                else:
 
                    raise RuntimeError("int4 mod should have outliers !")
        #print("after gemm",torch.cuda.memory_allocated()/1024/1024 - memory)
        if self.bias is not None:
            y1 += self.bias


        return y1.reshape(cache.shape)
    
