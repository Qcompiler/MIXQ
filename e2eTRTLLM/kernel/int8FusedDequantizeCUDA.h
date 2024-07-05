
#include <cuda_fp16.h>
void int8FusedDequantizeCUDA(const int8_t *A,
                             const int8_t *B,
                             const half *scale_row,
                             const half *scale_col,
                             half *y, half *D, int M, int N, int K,
                             char *,
                             cudaStream_t);

void int8quant(int rows, int cols, const half * src, int8_t *output, 
        half *scale,  cudaStream_t);

void print_half(const half *A, int M);

void int8dequant(int rows, int cols,  half * output, const int8_t *src,
         const half *scale, cudaStream_t);


void ExtractOutliersAndSetToZeros(int, int, const half * A, half *fp_A, const int *ind, int n, cudaStream_t);

void print_int( const  int *A, int M);


void dequantizationCUDA(half * out, const int * x,
                                 const half * scaleRow,
                                 const half * scaleCol, int M, int N, cudaStream_t);


void gemm_forward_cuda(
    int M,
    int N,
    int K,
    half *out_feats,
    const half * in_feats, //activation
    const int * kernel,  // int4 weight
    const half * scaling_factors, // scaling factors
    const int * zeros,
    cudaStream_t stream,
    half * cache
    );