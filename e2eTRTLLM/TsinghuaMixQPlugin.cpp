/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "TsinghuaMixQPlugin.h"

#include <numeric>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdio.h>

#define CUBLAS_CHECK(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                           \
        {                                                                                                              \
            fprintf(stderr, "cuBLAS error at %s:%d : %d\n", __FILE__, __LINE__, status);                               \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

void gemm(
    const int8_t * mat1,
    const int8_t * mat2, int *mat3, int m, int n, int k,cublasHandle_t handle, cudaStream_t stream) {
 

  static int64_t _beta = 0;
  static  int64_t _alpha = 1;

  auto beta_ptr = (void*)&_beta;
  auto alpha_ptr = (void*)&_alpha;

  auto input_ptr = (void*)mat3;
  auto mat1_ptr = (void*)mat1;
  auto mat2_ptr = (void*)mat2;
    //cublasHandle_t handle; 
   
 

  (cublasGemmEx(
       handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      alpha_ptr,
      mat2_ptr,
      CUDA_R_8I,
      k,
      mat1_ptr,
      CUDA_R_8I,
      k,
      beta_ptr,
      input_ptr,
      CUDA_R_32I,
      n,
      CUBLAS_COMPUTE_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

 

}


void gemmfp16add(
    const half * mat1,
    const half * mat2, half *mat3, int m, int n, int k,cublasHandle_t handle, cudaStream_t stream) {
 

  static float _beta = 1.0;
  static  float _alpha = 1.0;

  auto beta_ptr = (void*)&_beta;
  auto alpha_ptr = (void*)&_alpha;

  auto input_ptr = (void*)mat3;
  auto mat1_ptr = (void*)mat1;
  auto mat2_ptr = (void*)mat2;
 
 
    
  (cublasGemmEx(
       handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      alpha_ptr,
      mat2_ptr,
      CUDA_R_16F,
      k,
      mat1_ptr,
      CUDA_R_16F,
      k,
      beta_ptr,
      input_ptr,
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

 

}

void gemmfp16(
    const half * mat1,
    const half * mat2, half *mat3, int m, int n, int k, cublasHandle_t handle, cudaStream_t stream) {
 

  static float _beta = 0.0;
  static  float _alpha = 1.0;

  auto beta_ptr = (void*)&_beta;
  auto alpha_ptr = (void*)&_alpha;

  auto input_ptr = (void*)mat3;
  auto mat1_ptr = (void*)mat1;
  auto mat2_ptr = (void*)mat2;

    
  (cublasGemmEx(
       handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      alpha_ptr,
      mat2_ptr,
      CUDA_R_16F,
      k,
      mat1_ptr,
      CUDA_R_16F,
      k,
      beta_ptr,
      input_ptr,
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  

}
// Import a generated header to use generated triton kernels.
extern "C"
{
// #include "aot/fmha_kernel_fp16.h"
// #include "aot/fmha_kernel_fp32.h"

}
#include "kernel/int8FusedDequantizeCUDA.h"
#include "weightonlykernel/fpA_intB_gemm_wrapper.h"
#include <cstring>
#include <cuda_fp16.h>
#include <iostream>
#include <string>

using namespace nvinfer1;
using openai_triton::plugin::MixQPluginCreator;
using openai_triton::plugin::MixQPlugin;

static char const* TRITON_FLASH_ATTENTION_PLUGIN_VERSION{"1"};
static char const* TRITON_FLASH_ATTENTION_PLUGIN_NAME{"MixQ"};
PluginFieldCollection MixQPluginCreator::mFC{};
std::vector<PluginField> MixQPluginCreator::mPluginAttributes;

namespace openai_triton::plugin
{

// Write values into buffer
template <typename T>
void writeArg(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
void readArg(char const*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

std::uintptr_t constexpr kCudaMemAlign = 128;

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    if (addr % kCudaMemAlign)
    {
        addr += kCudaMemAlign - addr % kCudaMemAlign;
    }
    return (int8_t*) addr;
}

MixQPlugin::MixQPlugin(
    int m, int n, int k)
    : mm(m)
    , mn(n)
    , mk(k)
  
{
}

// Parameterized constructor
MixQPlugin::MixQPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    readArg(d, mm);
    readArg(d, mn);
    readArg(d, mk);
 
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* MixQPlugin::clone() const noexcept
{
    auto* plugin = new MixQPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs MixQPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output shape.
    //   output tensor [batchSize, seqLen, mNumHeads, head_size]
    assert(outputIndex == 0);
     int const nbDimsA = inputs[0].nbDims;
        int const nbDimsB = inputs[1].nbDims;
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[1].d[0]->getConstantValue());
        
        return ret;
}

bool MixQPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{


    

    //printf(" pos =%d --------------------",pos);
    switch (pos)
    {

    case 0:
        // activation
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        // weights
        // Weights stored in checkpoint must have int8 type
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        // scales channels
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    case 3:
        // fp weight
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    case 4:
        //  ind 
        // std:: cout << inOut[pos].type << std::endl;
        // std:: cout << nvinfer1::DataType::kINT32 << std::endl;
       
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    // 增加3个

    case 5:
 
       
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    case 6:
 
    
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    // case 7:
 
    
    //     return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;


    case 7:
    
        // out
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        // Never should be here
        assert(false);
        return false;
    }


}




void MixQPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1] ;
    int const maxN = in[1].max.d[0];
    int const minK = in[0].min.d[in[0].min.nbDims - 1] ;
    int const minN = in[1].min.d[0];

    assert(minN == maxN );
    assert(minK == maxK );

 
    // printf("maxM = %d\n",maxM);
    // printf("maxN = %d\n",maxN);
    // printf("maxK = %d\n",maxK);

    //  int8 quant + scale factor + fp16 weight + grand
    m_workspaceMaxSize =    maxM * maxK * sizeof(int8_t) +  maxM * sizeof(half) 
    +  maxK * maxN * sizeof(half)  ;
    int m_workspaceMaxSize2 =  maxM * maxN * sizeof(half) * 8; // for awq
    if ( m_workspaceMaxSize2 > m_workspaceMaxSize)
        m_workspaceMaxSize = m_workspaceMaxSize2;
    //m_workspaceMaxSize *= 2;
}

size_t MixQPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    // Set workspace size if needed. In this example, we need for L and m buffers.
    
    // int const numBuffers = 1;
    // size_t workspaces[numBuffers];
    // workspaces[0] = sizeof(half) * mm * mn;

 
    // size_t total = 0;
    // for (int i = 0; i < numBuffers; i++)
    // {
    //     total += workspaces[i];
    //     if (workspaces[i] % kCudaMemAlign)
    //     {
    //         total += kCudaMemAlign - (workspaces[i] % kCudaMemAlign);
    //     }
    // }
    // printf("total is  %d %d %d", mm, mn, workspaces[0]);
    // return total * 2;
    //printf("m_workspaceMaxSize = %d \n",m_workspaceMaxSize);
    return m_workspaceMaxSize;
}





int MixQPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, 
    void* workspace,
    cudaStream_t stream)
{
  
   int M = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        M *= inputDesc[0].dims.d[ii];
    }
    
    int K = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];


    int N = inputDesc[1].dims.d[0];
 
 
    int res = 0;

    half * Out = reinterpret_cast<half *>(outputs[0]);

    half* actPtr = reinterpret_cast<half*>(workspace);
 
    
    int8_t* int8_out = reinterpret_cast<int8_t*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), 0));

    const size_t bufSize_int8_out = sizeof(int8_t) * (M) *  K  ;
    half* scale_a = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(int8_out), 
    bufSize_int8_out));


    half* fp_activation = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(scale_a), 
     sizeof(half) * (M)  ));


     int* int_tmp = reinterpret_cast<int*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(scale_a), 
     sizeof(half) * (M)  ));

    // half* grand = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(fp_weight), 
    // sizeof(half) * (N) * K  ));




    const half * A = reinterpret_cast<half const*>(inputs[0]);

    const int8_t * W = reinterpret_cast<int8_t const*>(inputs[1]);
    


    const half * scale_b = reinterpret_cast<half const* >(inputs[2]);

    // outliers
    const half * fp_weight = reinterpret_cast<half const*>(inputs[3]);
    const int * ind  = reinterpret_cast<int const*>(inputs[4]);
    
    const int8_t *    q_weight = reinterpret_cast<int8_t const*>(inputs[5]);  // int weight // 采用fp16存
    const half *   scaling_factors  = reinterpret_cast<half const*>(inputs[6]); // scaling factors
 
    // std::vector<int> out(128);


    


    // // Launch a cuda kernel generated by Triton AoT.
    // 

    

        //         FILE *fp;
        //     half *tmp = (half *) malloc( sizeof(half) * M * N);
        //    cudaMemcpy(tmp, actPtr, sizeof(half) * M * N,  cudaMemcpyDeviceToHost);
        //     fp = fopen ("init.csv", "w");
        //     for (int i =0; i < 10; ++i){
        //         for (int j = 0; j < 20; ++j){
                        
        //                     fprintf (fp, "%4f\t",   float(  tmp[j + i * N] ));
        //                 }
        //                 fprintf (fp, "\n");
                        
                        
        //         }
        //     fclose(fp);
       


    if (M > 4){
        // prefill
        //printf("M N K is %d %d %d\n",M,N,K);
        
        

        // // print_half(fp_weight,10);
        // // exit(0);


 

        // check outliers of A
        // FILE *fp;

        // half *tmp = (half *) malloc( sizeof(half) * M * K);

        // cudaMemcpy(tmp, A, sizeof(half) * M * K,  cudaMemcpyDeviceToHost);

        // fp = fopen ("A1.csv", "w");
        // for (int i =0; i < M; ++i){
        //     for (int j = 0; j < K; ++j){
                    
        //                 fprintf (fp, "%4f\t",    float(tmp[j + i * K]));
        //             }
        //             fprintf (fp, "\n");
                    
                    
        //     }
        // fclose(fp);
        // ExtractOutliersAndSetToZeros(M, K, A, fp_activation, ind, num_ind, stream);

        // fp = fopen ("A2.csv", "w");
        // for (int i =0; i < M; ++i){
        //     for (int j = 0; j < K; ++j){
                    
        //                 fprintf (fp, "%4f\t",   float(  tmp[j + i * K] ));
        //             }
        //             fprintf (fp, "\n");
                    
                    
        //     }
        // fclose(fp);
        // exit(0);


        const int num_ind = 128;
        ExtractOutliersAndSetToZeros(M, K, A, fp_activation, ind, num_ind, stream);
        cublasSetStream(handle,stream);
        gemmfp16(fp_activation,fp_weight,Out, M, N, num_ind, handle, stream);
        int8quant(M, K, A, int8_out, scale_a, stream);

        //cuda
        //gemm(int8_out, W, int_tmp, M, N, K, stream);
        //dequantizationCUDA(Out, int_tmp, scale_a, scale_b, M, N, stream); 
        // half *tmp = (half *) malloc( sizeof(half) * M * K);
        // cudaMemcpy(tmp, A, sizeof(half) * M * K,  cudaMemcpyDeviceToHost);
        int8FusedDequantizeCUDA(int8_out, W, scale_a,
                                scale_b, Out, Out, M, N, K, 
                                reinterpret_cast<char*>(workspace),
                                stream);


        //     FILE *fp;
        //     half *tmp = (half *) malloc( sizeof(half) * M * N);
        //   cudaMemcpy(tmp, Out, sizeof(half) * M * N,  cudaMemcpyDeviceToHost);
        //     fp = fopen ("grand.csv", "w");
        //     for (int i =0; i < 10; ++i){
        //         for (int j = 0; j < 20; ++j){
                        
        //                     fprintf (fp, "%4f\t",   float(  tmp[j + i * N] ));
        //                 }
        //                 fprintf (fp, "\n");
                        
                        
        //         }
        //     fclose(fp);
        //     exit(0);

        // bool flag = 0;
        // for (int i =0; i < M; ++i){
        //     for (int j = 0; j < K; ++j){
                    
        //                 float aaa =  float(tmp[j + i * K]);
        //                 if (aaa > 6.0){
        //                     printf("not!!! outliers in %d\n",j);

        //                     flag = 1;
        //                 }
        //             }
                     
                    
                    
        //     }
         
        // if (flag){
        //     int8dequant(N, K, fp_activation, W, scale_b, stream);
        //     gemmfp16add(A,fp_activation,Out, M, N, K, stream);
        //     }
        //     else{
        //     int8quant(M, K, A, int8_out, scale_a, stream);
        //     gemm(int8_out, W, int_tmp, M, N, K, stream);
        //     dequantizationCUDA(Out, int_tmp, scale_a, scale_b, M, N, stream);

        //     // int8FusedDequantizeCUDA(int8_out, W, scale_a,
        //     //                         scale_b,
        //     //                         Out, Out, M, N, K, stream);
        // }

        
        // int8quant(M, K, A, int8_out, scale_a, stream);
        // int8FusedDequantizeCUDA(int8_out, W, scale_a,
        //                         scale_b,
        //                         Out, Out, M, N, K, stream);

        
        // float alpha = -1.0;
        // half result = -1.0;
        // half norm = -1.0;
        // cublasHandle_t handle; 
        // cublasCreate(&handle);
        // cublasSetStream(handle,stream);

        // CUBLAS_CHECK( cublasNrm2Ex(handle, M * N, Out, CUDA_R_16F, 1,
        //      &norm, CUDA_R_16F, CUDA_R_32F));

        // CUBLAS_CHECK (cublasAxpyEx(handle, M * N, &alpha, CUDA_R_32F, Out, CUDA_R_16F, 1,  fp_weight, CUDA_R_16F, 1, CUDA_R_32F));
        // CUBLAS_CHECK( cublasNrm2Ex(handle, M * N   , fp_weight, CUDA_R_16F, 1, &result, CUDA_R_16F, CUDA_R_32F));
        
        // float re = (float)result / (float) norm;

        // if (re > 0.05)
        // printf("relative error = %.8f\n", (float)result / (float) norm);
 
        // cublasDestroy(handle);
        // printf("input A!!\n");
        // std::vector<int8_t> a(128);
        // cudaMemcpy(a.data(),int8_out, sizeof(int8_t) * 128,cudaMemcpyDeviceToHost);
        // for (int i = 0 ; i < 10; ++i)
        //     printf("%d \t", a[i]);
        // printf("A done!\n");
        // printf("\n");
        
        // printf("input W!!\n");
        // std::vector<int8_t> w(128);
        // cudaMemcpy(w.data(),W, sizeof(int8_t) * 128,cudaMemcpyDeviceToHost);
        // for (int i = 0 ; i < 10; ++i)
        //     printf("%d \t", w[i]);
        // printf("W done!\n");
        // printf("\n");

    }
    else
    {
        // decode

        // gemm_forward_cuda(
        //     M,
        //     N,
        //     K,
        //     Out,
        //     A, //activation
        //     int4_weight,  // int4 weight // 采用fp16存
        //     scaling_factors, // scaling factors
        //     zeros,
        //     stream,
        //     actPtr
        //     );
     
        w8_a16_gemm_forward_cuda(A, q_weight,
                                scaling_factors,
                                Out,
                                M, 
                                N, 
                                K,
                                stream);
        //            FILE *fp;
        //     half *tmp = (half *) malloc( sizeof(half) * M * N);
        //   cudaMemcpy(tmp, actPtr, sizeof(half) * M * N,  cudaMemcpyDeviceToHost);
        //     fp = fopen ("awq.csv", "w");
        //     for (int i =0; i < 10; ++i){
        //         for (int j = 0; j < 20; ++j){
                        
        //                     fprintf (fp, "%4f\t",   float(  tmp[j + i * N] ));
        //                 }
        //                 fprintf (fp, "\n");
                        
                        
        //         }
        //     fclose(fp);
        //     exit(0);
        //            FILE *fp;
        //     half *tmp = (half *) malloc( sizeof(half) * M * K);
        //    cudaMemcpy(tmp, A, sizeof(half) * M * K,  cudaMemcpyDeviceToHost);
        //     fp = fopen ("Input.csv", "w");
        //     for (int i = 0; i < M; ++i){
        //         for (int j = 0; j < K; ++j){
                        
        //                     fprintf (fp, "%4f\t",   float(  tmp[j + i * K] ));
        //                 }
        //                 fprintf (fp, "\n");
                        
                        
        //         }
        //     fclose(fp);exit(0);


        // ------------------------
        //     FILE *fp;
        //     half *tmp = (half *) malloc( sizeof(half) * M * N);
        //    cudaMemcpy(tmp, scaling_factors, sizeof(half) * M * N,  cudaMemcpyDeviceToHost);
        //     fp = fopen ("scalings.csv", "w");
        //     for (int i = 0; i < 2; ++i){
        //         for (int j = 0; j < 12288; ++j){
                        
        //                     fprintf (fp, "%4f\t",   float(  tmp[j + i * N] ));
        //                 }
        //                 fprintf (fp, "\n");
                        
                        
        //         }
        //     fclose(fp);

        //     int *tmp2 = (int *) malloc( sizeof(int) * 10 * N);
        //     cudaMemcpy(tmp2, int4_weight, sizeof(int) * 10 * N,  cudaMemcpyDeviceToHost);
        //     fp = fopen ("int4_weight.csv", "w");
        //     for (int i = 0; i < 1; ++i){
        //         for (int j = 0; j < 20; ++j){
                        
        //                     fprintf (fp, "%d\t",   int(  tmp2[j + i * N] ));
        //                 }
        //                 fprintf (fp, "\n");
                        
                        
        //         }
        //     fclose(fp);


        //     int *tmp3 = (int *) malloc( sizeof(int)   * N);
        //     cudaMemcpy(tmp3, zeros, sizeof(int)   * N,  cudaMemcpyDeviceToHost);
        //     fp = fopen ("zeros.csv", "w");
        //     for (int i = 0; i < 1; ++i){
        //         for (int j = 0; j < 20; ++j){
                        
        //                     fprintf (fp, "%d\t",   int(  tmp3[j + i * N] ));
        //                 }
        //                 fprintf (fp, "\n");
                        
                        
        //         }
        //     fclose(fp);
        //     exit(0);

        // const int num_ind = 16;
        // ExtractOutliersAndSetToZeros(M, K, A, fp_activation, ind, num_ind, stream);

        // gemmfp16(fp_activation,fp_weight,Out, M, N, num_ind, handle, stream);
        // int8quant(M, K, A, int8_out, scale_a, stream);
        // int8FusedDequantizeCUDA(int8_out, W, scale_a,
        //                         scale_b,
        //                         Out, Out, M, N, K, reinterpret_cast<char*>(workspace),stream);

 
                    
    }

     // if (std::is_same<T, floalue)
    // {
    //     res = fmha_d64_fp32_default(stream, reinterpret_cast<CUdeviceptr>(Out), reinterpret_cast<CUdeviceptr>(L),
    //         reinterpret_cast<CUdeviceptr>(M), reinterpret_cast<CUdeviceptr>(Q), reinterpret_cast<CUdeviceptr>(K),
    //         reinterpret_cast<CUdeviceptr>(V), mSoftmaxScale, batchSize, mNumHeads, seqLen);
    // }
    // else
    // {
    //     res = fmha_d64_fp16_default(stream, reinterpret_cast<CUdeviceptr>(Out), reinterpret_cast<CUdeviceptr>(L),
    //         reinterpret_cast<CUdeviceptr>(M), reinterpret_cast<CUdeviceptr>(Q), reinterpret_cast<CUdeviceptr>(K),
    //         reinterpret_cast<CUdeviceptr>(V), mSoftmaxScale, batchSize, mNumHeads, seqLen);
    // }
    
    
    return res;
}

int MixQPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    {
        return enqueueImpl(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }

    return 1;
}

// IPluginV2Ext Methods
nvinfer1::DataType MixQPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return nvinfer1::DataType::kHALF;
}

// IPluginV2 Methods

char const* MixQPlugin::getPluginType() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_NAME;
}

char const* MixQPlugin::getPluginVersion() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_VERSION;
}

int MixQPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int MixQPlugin::initialize() noexcept
{
    // Load kernels generated by Triton AoT.
    // load_fmha_d64_fp32();
    // load_fmha_d64_fp16();
    cublasCreate(&handle);
    return 0;
}

void MixQPlugin::terminate() noexcept
{
    // Unload kernels generated by Triton AoT.
    // unload_fmha_d64_fp32();
    // unload_fmha_d64_fp16();
}

size_t MixQPlugin::getSerializationSize() const noexcept
{
    return sizeof(mm) + sizeof(mn) + sizeof(mk) ;
}

void MixQPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    writeArg(d, mm);
    writeArg(d, mn);
    writeArg(d, mk);

}

// bool MixQPlugin::supportsFormatCombination(
//     int pos, nvinfer1::PluginTensorDesc const* inOut, 
//     int nbInputs, int nbOutputs) noexcept
// {
//     switch (pos)
//     {
//     case 0:
//         // activation
//         return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
//     case 1:
//         // weights
//         // Weights stored in checkpoint must have int8 type
//         return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
//     case 2:
//         // scales channels
//         return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
//     case 3:
//         // scales tokens
//         return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
//     case 4:
//         // out
//         return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
//     default:
//         // Never should be here
//         assert(false);
//         return false;
//     }
// }

void MixQPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void MixQPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* MixQPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

MixQPluginCreator::MixQPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("mm", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("mn", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("mk", nullptr, PluginFieldType::kINT32, -1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* MixQPluginCreator::getPluginName() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_NAME;
}

char const* MixQPluginCreator::getPluginVersion() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_VERSION;
}

PluginFieldCollection const* MixQPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* MixQPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int m = 0;
    int n = 0;
    int k = 0;
   
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "m"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            m = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "n"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            n = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }else if (!strcmp(attrName, "k"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            k = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }

    }
    try
    {
        auto* obj = new MixQPlugin(m, n, k);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    return nullptr;
}

IPluginV2* MixQPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call MixQPlugin::destroy()
    try
    {
        auto* obj = new MixQPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    return nullptr;
}

void MixQPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* MixQPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} 
