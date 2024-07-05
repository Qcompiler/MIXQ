
#include <vector>

#define SMALL_M_FAST_PATH 4

#include<cuda_fp16.h>
#include<stdint.h>
void w8_a16_gemm_forward_cuda(const half* input, const  int8_t * weight,
                                      const  half *scale, half* output,
                                       int M, int N, int K,
                                       cudaStream_t stream);

void w8_a16_gemm_forward_cuda_(const half* input,const int8_t * weight,
                                        const half*  scale,
                                         half*  output,
                                        const int M,
                                        const int N,
                                        const int K,
                                        cudaStream_t stream);