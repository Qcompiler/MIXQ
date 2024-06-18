/*

Adapted from NVIDIA FasterTransformer:
https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/layernorm_kernels.cu

*/

#include <torch/extension.h>
#include <cuda_fp16.h>
#include "reduction.cuh"
#include "layernorm.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.h"
#include "util.h"
#include "int4.h"
static inline __device__ float to_float(half src)
{
    return __half2float(src);
}

static inline __device__ float to_float(float src)
{
    return src;
}

template<typename T>
__global__ void generalT5LayerNorm(
    const T* __restrict input, const T* __restrict gamma, T* output, const float layernorm_eps, int m, int n)
{
    // layernorm module in the T5 style No bias and no subtraction of mean.
    const int tid = threadIdx.x;

    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = to_float(__ldg(&input[blockIdx.x * n + i]));
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((to_float(input[blockIdx.x * n + i]) * s_variance) * to_float(__ldg(&gamma[i])));
    }
}


template<typename T>
void invokeGeneralT5LayerNorm(T*           out,
                              const T*     input,
                              const T*     gamma,
                              // const T*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0) {
        block.x = 1024;
    }

    block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision*/
    generalT5LayerNorm<T><<<grid, block>>>(input, gamma, out, layernorm_eps, m, n);  // For gpt-3
}

template void invokeGeneralT5LayerNorm(half*           out,
                              const half*     input,
                              const half*     gamma,
                              // const half*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n);

template void invokeGeneralT5LayerNorm(float*           out,
                              const float*     input,
                              const float*     gamma,
                              // const half*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n);



// input b, n, c
void layernorm_forward_cuda(
    torch::Tensor _input,
    torch::Tensor _gamma,
    torch::Tensor _out,
    float eps)
{
    int m = _input.size(0) * _input.size(1);
    int n = _input.size(2);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_input));

    auto input = reinterpret_cast<half*>(_input.data_ptr<at::Half>());
    auto gamma = reinterpret_cast<half*>(_gamma.data_ptr<at::Half>());
    auto out = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

    invokeGeneralT5LayerNorm(out, input, gamma, eps, m, n);
}




template<typename T, int size>
__global__ void generalT5LayerNorm_extract_outliers(
    const T* __restrict input, 
    const T* __restrict gamma, 
    T* output, const float layernorm_eps, 
    int m, int n , T* outliers, int *ind, int len_ind, int8_t * output_i8,  half * scale)
{
    // layernorm module in the T5 style No bias and no subtraction of mean.
    const int tid = threadIdx.x;
    const int bid = blockIdx.x ;

    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = to_float(__ldg(&input[blockIdx.x * n + i]));
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((to_float(input[blockIdx.x * n + i]) * s_variance)
             * to_float(__ldg(&gamma[i])));
    }

    __syncthreads();

    half *outliers_ =  outliers + blockIdx.x * len_ind;
    half *output_ = output + blockIdx.x * n;
    for (int i = tid; i < len_ind; i += blockDim.x) {
        int location = ind[i];
        outliers_[  i ] = output_[  location];
        output_[location] = 0.0;
         
    }   
    __syncthreads();

    half *start = output_ ;
    __shared__ half sdata[256];
    sdata[tid] = __habs(output_[tid]); 
    for (int i = tid + size; i < n; i += size)
        sdata[tid] = __hmax ( __habs(start[i]),  sdata[tid] ); 
    __syncthreads();


    // do reduction in shared mem
    for (unsigned int s= blockDim.x/2; s >= 1; s >>=1 ) {
        if (tid < s) {
            sdata[tid] =  __hmax ( __habs(sdata[tid + s]),  sdata[tid]);
        }
        __syncthreads();
    }

    half  max = sdata[0];
    // write result for this block to global mem
    //if (tid < 32) warpReduce(sdata, tid);

    __syncthreads();

    half quant_scales = __hdiv( max, 127.0);
    if (tid == 0){
        scale[bid] = quant_scales;
    }
    // quant
    int8_t * d_out = output_i8 + bid * n;
    for (int i = tid ; i < n; i += size)
        d_out[i] =  static_cast<int8_t>(__half2int_rn( __hdiv( start[i], quant_scales ) ))  ; 
    __syncthreads(); 


}

template<typename T, int size>
__global__ void generalT5LayerNorm_extract_outliers_int4(
    const T* __restrict input, 
    const T* __restrict gamma, 
    T* output, const float layernorm_eps, 
    int m, int n , T* outliers, int *ind, int len_ind, Int4Storage * output_i4,  half * scale, int colsDst)
{
    // layernorm module in the T5 style No bias and no subtraction of mean.
    const int tid = threadIdx.x;
    const int bid = blockIdx.x ;

    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = to_float(__ldg(&input[blockIdx.x * n + i]));
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<T>((to_float(input[blockIdx.x * n + i]) * s_variance)
             * to_float(__ldg(&gamma[i])));
    }

    __syncthreads();

    half *outliers_ =  outliers + blockIdx.x * len_ind;
    half *output_ = output + blockIdx.x * n;
    for (int i = tid; i < len_ind; i += blockDim.x) {
        int location = ind[i];
        outliers_[  i ] = output_[  location];
        output_[location] = 0.0;
         
    }   
    __syncthreads();

    half *start = output_ ;
    __shared__ half sdata[size];
    sdata[tid] = __habs(output_[tid]); 
    for (int i = tid + size; i < n; i += size)
        sdata[tid] = __hmax ( __habs(start[i]),  sdata[tid] ); 
    __syncthreads();


    // do reduction in shared mem
    for (unsigned int s= blockDim.x/2; s >= 1; s >>=1 ) {
        if (tid < s) {
            sdata[tid] =  __hmax ( __habs(sdata[tid + s]),  sdata[tid]);
        }
        __syncthreads();
    }

    half  max = sdata[0];
    // write result for this block to global mem
    //if (tid < 32) warpReduce(sdata, tid);

    __syncthreads();

    half quant_scales = __hdiv( max, 7.0);
    if (tid == 0){
        scale[bid] = quant_scales;
    }


    Int4Storage storage;
    memset(&storage, 0, sizeof(storage));
    Int4Storage * d_out = output_i4 + blockIdx.x * colsDst;
    // quant
    for (int i = tid ; i < colsDst; i += size){
        __half data1  = __hdiv( start[2*i + 0], quant_scales );
        __half data2  = __hdiv( start[2*i + 1], quant_scales );
        Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), 0}.set(
            __half2int_rn(data1));
        
        Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), 1}.set(
            __half2int_rn(data2));            
        d_out[i] =  storage ; 
    }


}
template<typename T>
void invokeGeneralT5LayerNorm_extract_outliers(T* out,
                              const T*     input,
                              const T*     gamma,
                              // const T*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n, 
                              T* outliers, 
                              int * ind, int len_ind, int8_t *output_i8, half *scales)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0) {
        block.x = 1024;
    }

    block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision*/
    generalT5LayerNorm_extract_outliers<T,1024/4><<<grid, block, 1024, at::cuda::getCurrentCUDAStream()>>>
    (input, gamma, out, layernorm_eps, m, n, outliers, ind, len_ind, output_i8, scales );  
}
std::vector<torch::Tensor>  layernorm_forward_cuda_extract_outliers(torch::Tensor &_input, torch::Tensor &_gamma, 
                torch::Tensor &_out, float eps, torch::Tensor &_ind,  torch::Tensor &scaleRow){


    int m = _input.size(0) * _input.size(1);
    int n = _input.size(2);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_input));

    auto input = reinterpret_cast<half*>(_input.data_ptr<at::Half>());
    auto gamma = reinterpret_cast<half*>(_gamma.data_ptr<at::Half>());
    auto out = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
    int* ind = reinterpret_cast<int*>(_ind.data_ptr<int32_t>());

    int len_ind = _ind.size(0);
 
    auto options = torch::TensorOptions().dtype(torch::kF16).device(_input.device());

    torch::Tensor _outliers = torch::zeros(
                {m, len_ind}, options);


    auto options_int8 = torch::TensorOptions().dtype(torch::kInt8).device(_input.device());
    torch::Tensor quant_out = torch::zeros(
                {m, n}, options_int8);

    invokeGeneralT5LayerNorm_extract_outliers(out, input, gamma, eps, m, n, 
                    reinterpret_cast<half*>(_outliers.data_ptr<at::Half>()), ind, len_ind,
                    (int8_t *)quant_out.data_ptr<int8_t>(),(half *)scaleRow.data_ptr<torch::Half>()
                    );
    return  { _outliers, quant_out};
}




template<typename T>
void invokeGeneralT5LayerNorm_extract_outliers_int4(T* out,
                              const T*     input,
                              const T*     gamma,
                              // const T*     beta,
                              const float  layernorm_eps,
                              const int    m,
                              const int    n, 
                              T* outliers, 
                              int * ind, int len_ind, Int4Storage *output_i4, half *scales, int colsDst)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
        Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0) {
        block.x = 1024;
    }

    block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision*/
    generalT5LayerNorm_extract_outliers_int4<T,1024/4><<<grid, block>>>
    (input, gamma, out, layernorm_eps, m, n, outliers, ind, len_ind, output_i4, scales,colsDst );  
}

std::vector<torch::Tensor>  layernorm_forward_cuda_extract_outliers_int4(torch::Tensor &_input, torch::Tensor &_gamma, 
                torch::Tensor &_out, float eps, torch::Tensor &_ind,  torch::Tensor &scaleRow){


    int m = _input.size(0) * _input.size(1);
    int n = _input.size(2);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_input));

    auto input = reinterpret_cast<half*>(_input.data_ptr<at::Half>());
    auto gamma = reinterpret_cast<half*>(_gamma.data_ptr<at::Half>());
    auto out = reinterpret_cast<half*>(_out.data_ptr<at::Half>());
    int* ind = reinterpret_cast<int*>(_ind.data_ptr<int32_t>());

    int len_ind = _ind.size(0);
 
    auto options = torch::TensorOptions().dtype(torch::kF16).device(_input.device());

    torch::Tensor _outliers = torch::zeros(
                {m, len_ind}, options);


     unsigned colsDst = (n - 1) / 2 + 1;
            assert (colsDst *2 == n);
            auto options_int4 = torch::TensorOptions().dtype
            (TorchDtypeDispatcher<Int4Storage>::value).device(_input.device());

    auto quant_out = torch::zeros(
        {m, colsDst }, options_int4);

    invokeGeneralT5LayerNorm_extract_outliers_int4(out, input, gamma, eps, m, n, 
                    reinterpret_cast<half*>(_outliers.data_ptr<at::Half>()), ind, len_ind,
                    quant_out.data_ptr<Int4Storage>(),(half *)scaleRow.data_ptr<torch::Half>(), colsDst
                    );
    return  { _outliers, quant_out};
}



