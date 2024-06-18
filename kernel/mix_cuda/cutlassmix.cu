
#include <torch/extension.h>
#include "c10/cuda/CUDAStream.h"
#include "cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#include "stdlib.h"
#include <chrono>
#include "iostream"
#include <string>

int  get_workspace_size(int m, int k){


    auto mixed_gemm_runner = fastertransformer::CutlassFpAIntBGemmRunner<half, uint8_t,  cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>();

    int mixgemm_max_size=std::max(m,k);
    int mixgemm_workspace_size_bytes=mixed_gemm_runner.getWorkspaceSize(m, mixgemm_max_size, mixgemm_max_size);
    return mixgemm_workspace_size_bytes;


}
torch::Tensor MixGemmCutlass(
    const torch::Tensor& ind,
    size_t lenid,
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2,  //int8
    const torch::Tensor& scales,  //fp16
    const torch::Tensor& mixgemmworkspace 
    ){

    // 我们先假设离群点的数量是2的幂次
    int m = (int) mat1.sizes()[0];
    int n = (int) mat2.sizes()[0];
    int k = (int) mat1.sizes()[1];

   auto options = torch::TensorOptions().dtype(torch::kFloat16).device(mat1.device());  
   at::Tensor output = torch::zeros({m, n}, options);

    half* d_c_half = (half *)output.data_ptr<at::Half>();
    half* d_a_half = (half *)mat1.data_ptr<at::Half>();
    

    uint8_t* d_b_int = (uint8_t *)mat2.data_ptr();
    half* d_b_scale = (half *)scales.data_ptr<at::Half>();


    char* mixgemm_workspace_data = (char *)mixgemmworkspace.data_ptr();
    int mixgemm_workspace_size_bytes = (int) mixgemmworkspace.sizes()[0];

    //cudaMalloc(&mixgemm_workspace_data, mixgemm_workspace_size_bytes);
    at::cuda::CUDAStream myStream = at::cuda::getStreamFromPool();

    at::cuda::setCurrentCUDAStream(myStream);
    cudaStream_t stream = myStream.stream();
    auto mixed_gemm_runner = fastertransformer::CutlassFpAIntBGemmRunner<half, uint8_t,  cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>();
    
    


    
 

    mixed_gemm_runner.gemm(
        reinterpret_cast<const half*>(d_a_half),
        reinterpret_cast<const uint8_t*>(d_b_int),
        reinterpret_cast<const half*>(d_b_scale),
        reinterpret_cast<half*>(d_c_half),
        m,
        n,
        k,
        k,
        mixgemm_workspace_data,
        mixgemm_workspace_size_bytes,
        stream
    );
    return output;

}




