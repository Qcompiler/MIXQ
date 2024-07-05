
#include "cub/cub.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fpA_intB_gemm_wrapper.h"
#include "fpA_intB_gemm.h"
#include "cutlass_preprocessors.h"
#include "cuda_utils.h"
#include "weightOnlyBatchedGemv/enabled.h"
#include "weightOnlyBatchedGemv/kernelLauncher.h"


#include <vector>

namespace ft = fastertransformer;

int getWorkspaceSize(const int m, const int n, const int k)
{
    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    const int max_grid_m = (m + 31) / 32;
    const int max_grid_n = (n + 127) / 128;
    const int split_k_limit = 7;
    // We need 4 bytes per block in the worst case. We launch split_k_limit in z dim.
    return max_grid_m * max_grid_n * split_k_limit * 4;
}



void w8_a16_gemm_forward_cuda(const half* input, const  int8_t * weight,
                                      const  half *scale, half* output, 
                                      int m, int n, int k,
                                      cudaStream_t stream)
{
    
    
    
    const ft::half *input_ptr = reinterpret_cast<const ft::half *>(input);
    const uint8_t *weight_ptr = reinterpret_cast<const uint8_t *>(weight);
    const ft::half *scale_ptr = reinterpret_cast<const ft::half *>(scale);
    ft::half *output_ptr = reinterpret_cast<ft::half *>(output);
    // const int max_size = std::max(n, k);
    // size_t workspace_size = getWorkspaceSize(m, max_size, max_size);
    // void *ptr = nullptr;
    // char *workspace_ptr = workspace_size > 0 ? (char *)cudaMalloc((void **)&ptr, workspace_size) : nullptr;
    const bool use_cuda_kernel = m <= SMALL_M_FAST_PATH;
    // const bool use_cuda_kernel = false; 
   
    if(use_cuda_kernel){
        tensorrt_llm::kernels::WeightOnlyActivationType weight_only_act_type = tensorrt_llm::kernels::WeightOnlyActivationType::FP16;
        tensorrt_llm::kernels::WeightOnlyQuantType weight_only_quant_type = tensorrt_llm::kernels::WeightOnlyQuantType::Int8b;
        tensorrt_llm::kernels::WeightOnlyParams params{weight_ptr, 
        reinterpret_cast<const uint8_t *>(scale), nullptr,
            reinterpret_cast<const half *>(input), nullptr, nullptr, 
            reinterpret_cast<half *>(output), m, n, k, 0, weight_only_quant_type,
            tensorrt_llm::kernels::WeightOnlyType::PerChannel,
            tensorrt_llm::kernels::WeightOnlyActivationFunctionType::Identity, weight_only_act_type};
        tensorrt_llm::kernels::weight_only_batched_gemv_launcher(params, stream);
    }
    else
        ft::gemm_fp16_int(
            input_ptr,
            weight_ptr,
            scale_ptr,
            output_ptr,
            m, n, k,
            nullptr,
            0,
            stream);
     
}


void w8_a16_gemm_forward_cuda_(const half* input,const int8_t * weight,
                                        const half*  scale,
                                         half*  output,
                                        const int M,
                                        const int N,
                                        const int K,
                                        cudaStream_t stream)
{

    const ft::half *input_ptr = reinterpret_cast<const ft::half *>(input);
    const uint8_t *weight_ptr = reinterpret_cast<const uint8_t *>(weight);
    const ft::half *scale_ptr = reinterpret_cast<const ft::half *>(scale);
    ft::half *output_ptr = reinterpret_cast<ft::half *>(output);

    ft::gemm_fp16_int(
        input_ptr,
        weight_ptr,
        scale_ptr,
        output_ptr,
        M,N,K,
        nullptr,
        0,
        stream);
     
}