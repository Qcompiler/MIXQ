#include <iostream>
#include <stdexcept>
#include "common.h"

 
#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2  // BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4  // BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 16  // BLOCK_COLS / MMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / MMA_M

#define WARP_ROW_TILES 8  // WARP_COLS / MMA_N
#define WARP_COL_TILES 4  // WARP_ROWS / MMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / MMA_K

#define CHUNK_LINE_BYTES 64          // CHUNK_K * MMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * sizeof(int4) / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 32  // CHUNK_K * MMA_K

#define C_SMEM_STRIDE 128  // BLOCK_COLS
#define C_SMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16


#define MMA_M 16
#define MMA_N 8
#define MMA_K 16




#include <cublas_v2.h>
#include <cublasLt.h>
#include "ops.cuh"
#include<iostream>


 

template <int FORMATB, int DTYPE_OUT, int SCALE_ROWS> int 
igemmlt(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, int *C,
 float *row_scale, int lda, int ldb, int ldc)
{
#ifdef NO_CUBLASLT
  std::cout << "" << std::endl;;
  std::cout << "=============================================" << std::endl;;
  std::cout << "ERROR: Your GPU does not support Int8 Matmul!" << std::endl;;
  std::cout << "=============================================" << std::endl;;
  std::cout << "" << std::endl;;
  assert(false);

	return 0;
#else


    int has_error = 0;
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasOperation_t opT = CUBLAS_OP_T;
    cublasLtPointerMode_t alphaVec = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
    cublasLtOrder_t col32 = CUBLASLT_ORDER_COL32;
    cublasLtOrder_t col_turing = CUBLASLT_ORDER_COL4_4R2_8C;
    cublasLtOrder_t col_ampere = CUBLASLT_ORDER_COL32_2R_4R4;

    has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, lda));
    has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, n, k, ldb));

    has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
    if(FORMATB == COL_TURING)
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_turing, sizeof(col_turing)));
    else
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col_ampere, sizeof(col_ampere)));

    if(DTYPE_OUT == 32)
    {
      has_error |= checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
      has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, m, n, ldc));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
      int alpha = 1, beta = 0;
      has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc,&alpha, A, Adesc, B, Bdesc, &beta, (int32_t*)C, Cdesc, (int32_t*)C, Cdesc, NULL, NULL, 0, 0));
    }
    else
    {
      has_error |= checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F));
      has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_8I, m, n, ldc));
      has_error |= checkCublasStatus(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &col32, sizeof(col32)));
      if(!SCALE_ROWS)
      {
        float alpha = 1.0f, beta = 0.0f;
        has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc,&alpha, A, Adesc, B, Bdesc, &beta, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, NULL, NULL, 0, 0));
      }
      else
      {
        has_error |= checkCublasStatus(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &alphaVec, sizeof(alphaVec)));
        has_error |= checkCublasStatus(cublasLtMatmul(ltHandle, matmulDesc, row_scale, A, Adesc, B, Bdesc, NULL, (int8_t*)C, Cdesc, (int8_t*)C, Cdesc, NULL, NULL, 0, 0));
      }
    }


    if (Cdesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) has_error |= checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc) has_error |= checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
    if(has_error == 1)
      printf("error detected");

    return has_error;
#endif
}


template int igemmlt<COL_AMPERE, 32, 0>(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, int *C, float *row_scale, int lda, int ldb, int ldc);


 int igemmlt_ampere_32(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, int *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt<COL_AMPERE, 32, 0>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }

int cigemmlt_ampere_32(long long contextinput, int m, int n, int k, 
    const long long  A_, const long long  B_, long long  C_, 
    const long long row_scale_, int lda, int ldb, int ldc)
	{ 
    float  *row_scale = reinterpret_cast<float*> (row_scale_);
    Context *context = reinterpret_cast<Context*> (contextinput);
    int8_t  *A = reinterpret_cast<int8_t*> (A_);
    int8_t  *B = reinterpret_cast<int8_t*> (B_);
    int  *C = reinterpret_cast<int*> (C_);

    return igemmlt_ampere_32((cublasLtHandle_t) context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); 
    }


#include <torch/extension.h>
torch::Tensor linear_a8_w8_o32_with_scaling(long long contextinput, int M, int N, int K, 
     torch::Tensor  A_,  torch::Tensor  B_)
	{ 
    static int alpha = 1;
    static int beta = 0;
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A_.device());

    at::Tensor _out_feats = torch::empty({M, N}, options);

    cublasHandle_t handle = reinterpret_cast<cublasHandle_t> (contextinput);
    int8_t*  d_A = reinterpret_cast<int8_t *>(A_.data_ptr<int8_t>());
    int8_t*  d_B = reinterpret_cast<int8_t *>(B_.data_ptr<int8_t>());
    int*  d_C = reinterpret_cast<int *>(_out_feats.data_ptr<int>());

    // void * d_A =  reinterpret_cast<void *> (d_A_);
    // void * d_B =  reinterpret_cast<void *> (d_B_);
    // void * d_C =  reinterpret_cast<void *> (d_C_);

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_8I, K, d_A,
                                          CUDA_R_8I, K, &beta, d_C, CUDA_R_32I, N, CUBLAS_COMPUTE_32I,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                              
    return _out_feats;

}

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
torch::Tensor gemm(
    const torch::Tensor& mat1,
    const torch::Tensor& mat2, int m, int n, int k) {
 

  static int64_t _beta = 0;
  static  int64_t _alpha = 1;
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(mat1.device());

  at::Tensor input = torch::empty({m, n}, options);
  
  auto beta_ptr = (void*)&_beta;
  auto alpha_ptr = (void*)&_alpha;

  auto input_ptr = (void*)input.data_ptr<int32_t>();
  auto mat1_ptr = (void*)mat1.data_ptr<int8_t>();
  auto mat2_ptr = (void*)mat2.data_ptr<int8_t>();

  (cublasGemmEx(
       at::cuda::getCurrentCUDABlasHandle(),
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

  return input;
}


void gemmColumn(
    const torch::Tensor& mat1, //列优先
    const torch::Tensor& mat2, int m, int n, int k,
    const torch::Tensor& output) {
 

  static int64_t _beta = 0;
  static  int64_t _alpha = 1;

  
  auto beta_ptr = (void*)&_beta;
  auto alpha_ptr = (void*)&_alpha;

  auto input_ptr = (void*)output.data_ptr<int32_t>();
  auto mat1_ptr = (void*)mat1.data_ptr<int8_t>();
  auto mat2_ptr = (void*)mat2.data_ptr<int8_t>();

  (cublasGemmEx(
       at::cuda::getCurrentCUDABlasHandle(),
      CUBLAS_OP_T,
      CUBLAS_OP_T,
      n,
      m,
      k,
      alpha_ptr,
      mat2_ptr,
      CUDA_R_8I,
      k,
      mat1_ptr,
      CUDA_R_8I,
      m,
      beta_ptr,
      input_ptr,
      CUDA_R_32I,
      n,
      CUBLAS_COMPUTE_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  return ;
}

__global__ void set_value(half* input, half* mat1, half*mat2){

  for (int i = 0 ;i < 4; ++ i){
    input[i] = (half)i;
  }
}


#define WARP_REPEAT_M 2



 

// op3 按照列 计算四次，需要额外的寄存器，不需要额外的共享内存
__global__ void mmaNaiveKernelop3(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K, const int* ind, size_t lenind) {
    const size_t K_tiles =  lenind / MMA_K; // 这儿应该是不能超过K个 你如不足16个 那么后面的16个就不能用向量化的方式导入了

    const size_t warp_row = blockIdx.y * MMA_M * WARP_REPEAT_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[WARP_REPEAT_M*MMA_M][MMA_N];

    __shared__ int ind_smem[2][8];

    const int lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[WARP_REPEAT_M][2];
    for (int i = 0; i < WARP_REPEAT_M; ++i){
        RC[i][0]  = 0;
        RC[i][1]  = 0;
    }


// #pragma unroll
    half * targetA = nullptr;
    half * targetB = nullptr;
    const half * sourceB = nullptr;
    const half * sourceA = nullptr;
    if ( K_tiles ){
        targetA = &A_smem[lane_id / 2][0] + (lane_id % 2) * 8;
        sourceA =  A + (warp_row + lane_id / 2) * K;
        // 一个int4 16字节
        // 需要写八个 half (每个 2 字节)
        if (lane_id < MMA_N * 2){
            targetB = &B_smem[lane_id / 2][0] + (lane_id % 2) * 8;
            sourceB =  B + (warp_col + lane_id / 2) * K;
        }
    }
    // 一次放128个
     

    for (size_t i = 0; i < K_tiles; ++i) {
        // 所有线程为偶数的用相同的 ind 
        // 实际上我们只需要16个线程来把ind 写入shared

        ind_smem[(lane_id % 2)][lane_id/4] =  ind  [i * MMA_K + (lane_id % 2) * 8 + lane_id/4 ];
        __syncthreads();

        for (int kk = 0 ; kk < WARP_REPEAT_M ; ++kk ){


            const half * sourceA_ = sourceA + kk * MMA_M * K;
            for (int j = 0 ; j < 8 ; ++j ){
                int id =  ind_smem[(lane_id % 2)][j];
                targetA[j] = sourceA_[id] ;
                if (lane_id < MMA_N * 2)
                    targetB[j] = sourceB[ id ] ;
            }
            
            __syncthreads();

            uint32_t RA[4] = {0,0,0,0};
            uint32_t RB[2] = {0,0};

            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
            LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

            HMMA16816(RC[kk][0], RC[kk][1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[kk][0], RC[kk][1]);

            __syncthreads();
        }
    }


    if ( lenind % MMA_K )
    {

        // 把shared memory 设置成 0
        half *tmp1 = (half *)((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2);
            for (int i = 0 ; i < 8; ++i)  tmp1[i] = 0.0;

        if (lane_id < MMA_N * 2){
            half *tmp2 =  (half *)((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2);
            for (int i = 0 ; i < 8; ++i)  tmp2[i] = 0.0;
        }

        int stride = lenind - K_tiles * MMA_K;

        int ptrl =  min ( (stride - (lane_id % 2) * 8), 8) ; 

        // 最后一个tile 也太难处理了把。。。
        // int4：4 个 int 16字节， 需要写8个half （每个half 两个字节）


        ind_smem[(lane_id % 2)][lane_id/4] =  ind  [K_tiles * MMA_K + (lane_id % 2) * 8 + lane_id/4 ];
        __syncthreads();


        for (int kk = 0 ; kk < WARP_REPEAT_M ; ++kk )
        {
            if (ptrl){
                targetA = &A_smem[lane_id / 2][0]  + (lane_id % 2) * 8;
                sourceA = &A[(kk * MMA_M + warp_row + lane_id / 2) * K] ;
                if (lane_id < MMA_N * 2){
                    targetB = &B_smem[lane_id / 2][0] + (lane_id % 2) * 8;
                    sourceB = &B[ (warp_col + lane_id / 2) * K] ; 
                }
                // int id =  K_tiles * MMA_K + (lane_id % 2) * 8  ;
                for (int j = 0 ;j < ptrl; ++j){ 
                    int loc = ind_smem[lane_id % 2][ j];  
                    targetA[j] = sourceA[ loc];           
                    if (lane_id < MMA_N * 2) {    
                        targetB[j] = sourceB[loc]; 
                    }         
                }
            }
            __syncthreads();
            uint32_t RA[4] = {0,0,0,0};
            uint32_t RB[2] = {0,0};
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
            LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

            HMMA16816(RC[kk][0], RC[kk][1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[kk][0], RC[kk][1]);

            __syncthreads();  
        }
        
    }

    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0][0];
    *((uint32_t *)(&C_smem[ lane_id / 4 + 8][0]) + lane_id % 4) = RC[0][1];
    *((uint32_t *)(&C_smem[ MMA_M + lane_id / 4][0]) + lane_id % 4) = RC[1][0];
    *((uint32_t *)(&C_smem[ MMA_M + lane_id / 4 + 8][0]) + lane_id % 4) = RC[1][1];

    __syncthreads();

     if (lane_id < MMA_M) {


       *((double2 *)(&C[( warp_row + lane_id) * N + warp_col])) = *reinterpret_cast<double2*>(C_smem[ lane_id]);
       *((double2 *)(&C[( MMA_M + warp_row + lane_id) * N + warp_col])) = *reinterpret_cast<double2*>(C_smem[ MMA_M + lane_id]);

     }
}

 
 
 
// num warp = 4 的时候 总共的warp数量变成了16
template<int MMA, int NUM_WARP, int REPEATK>
__device__ __forceinline__ void loadsparse_v3_squre_warp( half * shmd, const int len_shmd, const half* global,  
        const int * ind,  const int lenind, 
        int idx, int K){

    int startcol =  idx / ( 2 * NUM_WARP  ); 
    int startrow = (idx % ( 2 * NUM_WARP  ) ) *   ( (MMA/2));    
    int col = ind[startcol ];
    const half *global_tmp = global + col +  (startrow ) * K;
    half *shmd_tmp = shmd + startrow * (MMA_K * REPEATK) + startcol;
    idx = 0;
    for (int i = 0; i <  ((MMA/2)  ); i++) {
        if (idx < len_shmd) {
            shmd_tmp[idx] = global_tmp[ i * K];
            idx += (MMA_K * REPEATK ) ;
        }
    }
}
template<int NUM_WARP, int REPEATK>
__global__ void mmaNaiveKernelSqureWarp(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K, const int* ind, size_t lenind) {
    const size_t K_tiles =  lenind / (MMA_K * REPEATK); // 这儿应该是不能超过K个 你如不足16个 那么后面的16个就不能用向量化的方式导入了

    const size_t warp_row = blockIdx.y * MMA_M * NUM_WARP;
    const size_t warp_col = blockIdx.x * MMA_N * NUM_WARP;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_smem[MMA_M * NUM_WARP][MMA_K * REPEATK];
    __shared__ half B_smem[MMA_N * NUM_WARP][MMA_K * REPEATK];
 
    __shared__ int ind_smem[MMA_K * REPEATK];

    const int tid =  threadIdx.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    // uint32_t RC[REPEATK][2]  ;

    // for (int i = 0 ; i < REPEATK ; ++i){
    //     RC[i][0] = 0;
    //     RC[i][1] = 0;
    // }

    uint32_t RC[2] = {0,0};

     
    // 计算
    uint32_t RA[REPEATK][4];
    uint32_t RB[REPEATK][2];
    for (size_t i = 0; i < K_tiles; ++i) {
        // 所有线程为偶数的用相同的 ind 
        // 实际上我们只需要16个线程来把ind 写入shared
        if (lane_id < MMA_K * REPEATK)
            ind_smem[lane_id] =  ind  [  i * MMA_K * REPEATK +  lane_id];
        
        __syncthreads();
        // 调度

        loadsparse_v3_squre_warp<MMA_M,NUM_WARP,REPEATK>( &A_smem[0][0], MMA_M * MMA_K * NUM_WARP * REPEATK, 
                A + (warp_row) * K,  
                ind_smem,  MMA_K * REPEATK, 
                tid,  K);
        loadsparse_v3_squre_warp<MMA_N,NUM_WARP,REPEATK>( &B_smem[0][0], MMA_N * MMA_K * NUM_WARP * REPEATK, 
                B + (warp_col) * K,  
                ind_smem,  MMA_K * REPEATK, 
                tid,  K);
        
        __syncthreads();

        for (int r = 0 ; r < REPEATK; ++r){
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[ (warp_id / NUM_WARP) * MMA_M + lane_id % 16][(lane_id / 16) * 8 + r * MMA_K]);
            LDMATRIX_X4(RA[r][0], RA[r][1], RA[r][2], RA[r][3], A_smem_lane_addr);


            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[ (warp_id % NUM_WARP) * MMA_N + lane_id % 8][((lane_id / 8) % 2) * 8 + r * MMA_K]);
            LDMATRIX_X2(RB[r][0], RB[r][1], B_smem_lane_addr);
        }
        __syncthreads();


        for (int r = 0 ; r < REPEATK; ++r){
            HMMA16816(RC[0], RC[1], RA[r][0], RA[r][1], RA[r][2], RA[r][3], RB[r][0], RB[r][1], RC[0], RC[1]);
        }
        __syncthreads();
        
    }

    half * C_temp = C + warp_row * N + warp_col;
    for (int w = 0 ; w < NUM_WARP; ++w){
        
        *((uint32_t *)(&C_temp[ (lane_id / 4 + (warp_id / NUM_WARP) * MMA_M) * N + (warp_id % NUM_WARP) * MMA_N]) + lane_id % 4) = RC[0];
        *((uint32_t *)(&C_temp[ (lane_id / 4 + 8 + (warp_id / NUM_WARP) * MMA_M) * N + (warp_id % NUM_WARP) * MMA_N]) + lane_id % 4) = RC[1];
    }
 

 
}

 
template <int NUM_WARP>
void mmaNaiveSqureWarp(half *A, half *B, half *C, size_t M, size_t N, size_t K, const int* ind, size_t lenind) {
  
    // 优化1 :  每一个warp   算 16 个矩阵的乘法
    dim3 block(WARP_SIZE * NUM_WARP * NUM_WARP);
    dim3 grid(div_ceil(N, MMA_N * NUM_WARP), div_ceil(M, MMA_M * NUM_WARP));


    mmaNaiveKernelSqureWarp<NUM_WARP,NUM_WARP><<<grid, block>>>(A, B, C, M, N, K, ind, lenind);

}


void mmaNaiveop3(half *A, half *B, half *C, size_t M, size_t N, size_t K, const int* ind, size_t lenind) {
  
    // 优化1 :  每一个 block   算 4倍多的矩阵乘法
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M/WARP_REPEAT_M, MMA_M));
    mmaNaiveKernelop3<<<grid, block>>>(A, B, C, M, N, K, ind, lenind);
}




#include<cuda_fp16.h>
// totally dense compute
torch::Tensor Mixgemm(
    const torch::Tensor& ind,
    size_t lenid,
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2  //fp16
    ){
  auto m = mat1.sizes()[0];
  auto n = mat2.sizes()[0];
  auto k = mat1.sizes()[1];

  //std::cout <<" k is" << k<<  std::endl;

  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(mat1.device());  
  at::Tensor input = torch::zeros({m, n}, options);

  half* input_ptr = (half *)input.data_ptr<at::Half>();
  half* mat1_ptr = (half *)mat1.data_ptr<at::Half>();
  half* mat2_ptr = (half *)mat2.data_ptr<at::Half>();
  int* ind_ptr = (int *)ind.data_ptr<int>();

    

//   if (lenid < 8){
//         // need to be optimized
//         mmaNaiveop3(mat1_ptr, mat2_ptr, input_ptr, m, n, k, ind_ptr, lenid);
//   }
//   else {
//         mmaNaiveREPEATK<4,2>  (mat1_ptr, mat2_ptr, input_ptr, m, n, k, ind_ptr, lenid);
//   }
  //mmaNaiveREPEATK<4,2>  (mat1_ptr, mat2_ptr, input_ptr, m, n, k, ind_ptr, lenid);  
  //mmaNaiveSqureWarp<2>(mat1_ptr, mat2_ptr, input_ptr, m, n, k, ind_ptr, lenid);
   
  return input;
}











#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void myHGEMMAlignedV1(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; bk++) {
        FLOAT4(s_a[load_a_smem_m    ][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr        ]);
        FLOAT4(s_a[load_a_smem_m + 1][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr +     K]);
        FLOAT4(s_b[load_b_smem_k    ][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr        ]);
        FLOAT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr +     N]);
        FLOAT4(s_b[load_b_smem_k + 2][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 2 * N]);
        FLOAT4(s_b[load_b_smem_k + 3][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr + 3 * N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64     ][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64     ][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[ 0][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[ 0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[ 0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[ 0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
    }
}


const int MI = 128;
// const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

__device__ void loadSmemA(half *smem, half *A, int M, int K, int ko)
{
    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = A[(by * 128 + row) * K + ko * KI + col];
    }
}

__device__ void loadSmemB(half *smem, half *B, int N, int K, int ko)
{
    // load 128 * 32
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 64; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = B[(bx * 128 + row) * K + ko * KI + col];
    }
}

__device__ void loadSmemC(float *smem, half *C, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = (float)(C[(by * 128 + row) * N + bx * 128 + col]);
    }
}

__device__ void storeSmemC(half *C, float *smem, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        (C[(by * 128 + row) * N + bx * 128 + col]) = (half)smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16];
    }
}

__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> *frag, half *smem, int ki)
{
    // load 64x16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        int row = tz * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> *frag, half *smem, int ki)
{
    // load 64x16
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i)
    {
        int row = ty * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void storeAccum(float *ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> *frag)
{
    // store 64x64
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int row = tz * 64 + i * 16;
            int col = ty * 64 + j * 16;
            // laoyut: [8, 8, 16, 16]
            nvcuda::wmma::store_matrix_sync(ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16), frag[i * 4 + j], 
            16, nvcuda::wmma::mem_row_major);
        }
    }
}

__global__ void matmul(half *A, half *B, half *C, int M, int N, int K, float alpha, float beta)
{
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    half *SA = reinterpret_cast<half *>(shared_storage);
    half *SB = reinterpret_cast<half *>(shared_storage + MI * KI * sizeof(half));
    float *SC = reinterpret_cast<float *>(shared_storage);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> FragA[MII / wmmaM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragB[NII / wmmaN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[MII / wmmaM * NII / wmmaN];

    for (int mii = 0; mii < MII / wmmaM; mii += 1)
    {
        for (int nii = 0; nii < NII / wmmaN; nii += 1)
        {
            nvcuda::wmma::fill_fragment(Accum[mii * (NII / wmmaN) + nii], 0.0);
        }
    }
    for (int ko = 0; ko < K / KI; ko += 1)
    {
        loadSmemA(SA, A, M, K, ko);
        loadSmemB(SB, B, N, K, ko);
        __syncthreads();
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA, ki);
            loadFragB(FragB, SB, ki);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
                }
            }
        }
    }
    storeAccum(SC, Accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
}


#define MAX(a, b) (a) > (b) ? (a) : (b)
torch::Tensor Mixgemmalligned(
    const torch::Tensor& ind,
    size_t lenid,
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2  //fp16
    ){
  auto m = mat1.sizes()[0];
  auto n = mat2.sizes()[0];
  auto k = mat1.sizes()[1];

  //std::cout <<" k is" << k<<  std::endl;

  auto options = torch::TensorOptions().dtype(torch::kFloat16).device(mat1.device());  
  at::Tensor input = torch::zeros({m, n}, options);

  half* input_ptr = (half *)input.data_ptr<at::Half>();
  half* mat1_ptr = (half *)mat1.data_ptr<at::Half>();
  half* mat2_ptr = (half *)mat2.data_ptr<at::Half>();
  int* ind_ptr = (int *)ind.data_ptr<int>();

//   const int BM = 128, BN = 256;
//   dim3 blockDim(256);
//   int BX = (n + BN - 1) / BN;
//   int BY = (m + BM - 1) / BM;
//   dim3 gridDim(BX, BY);
//   myHGEMMAlignedV1<<<gridDim, blockDim>>>(mat1_ptr, mat2_ptr, input_ptr, m, n, k);  
    int STAGES = 1;
    int smem_size = MAX(STAGES * 128 * 32 * 2 * 2, 128 * 128 * 4);
    if (smem_size >= (48 << 10))
    {
        cudaFuncSetAttribute(matmul,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size);
    }
  dim3 dimBlock(32, 2 * STAGES , 2);
  dim3 dimGrid(n / 128, m / 128);
  matmul<<<dimGrid, dimBlock, smem_size, nullptr>>>(mat1_ptr, mat2_ptr, input_ptr, m, n, k, 1.0, 0.0);
  return input;
}


template <typename SrcType, typename DstType, int M, int N, int K>
__device__  void 
convertFragment(const nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, SrcType, nvcuda::wmma::col_major>& input,
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, DstType, nvcuda::wmma::col_major> &output, float scale){

    // 执行类型转换
    #pragma unroll
    for (int i = 0; i < output.num_elements; ++i) {
        output.x[i] =  1;
    }

    return ;
}
__device__ void loadFragBint8tofp16(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, int8_t, 
        nvcuda::wmma::col_major> *fragint8,
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, 
        nvcuda::wmma::col_major> *fragfp16, 
        int8_t *smem, int ki,float scale)
{
    // load 64x16
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i)
    {
        int row = ty * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(fragint8[i], (smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16)), 16);
    }
    for (int i = 0; i < 4; ++i)
          convertFragment<int8_t, half, wmmaM, wmmaN, wmmaK>(fragint8[i],fragfp16[i],scale);



}
__device__ void loadSmemB(int8_t *smem, int8_t *B, int N, int K, int ko)
{
    // load 128 * 32
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 64; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = B[(bx * 128 + row) * K + ko * KI + col];
    }
}
__global__ void  matmulfp16int8(half *A, int8_t *B, half *C, int M, int N, int K, float alpha, float beta, float scale)
{
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    half *SA = reinterpret_cast<half *>(shared_storage);
    int8_t *SB = reinterpret_cast<int8_t *>(shared_storage + MI * KI * sizeof(half));
    float *SC = reinterpret_cast<float *>(shared_storage);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> FragA[MII / wmmaM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragBfp16[NII / wmmaN];  // 4个
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, int8_t, nvcuda::wmma::col_major> FragBint8[NII / wmmaN];  // 4个


    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[MII / wmmaM * NII / wmmaN];

    for (int mii = 0; mii < MII / wmmaM; mii += 1)
    {
        for (int nii = 0; nii < NII / wmmaN; nii += 1)
        {
            nvcuda::wmma::fill_fragment(Accum[mii * (NII / wmmaN) + nii], 0.0);
        }
    }
    for (int ko = 0; ko < K / KI; ko += 1)
    {
        loadSmemA(SA, A, M, K, ko);
        loadSmemB(SB, B, N, K, ko);
        __syncthreads();
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA, ki);
            loadFragBint8tofp16(FragBint8,FragBfp16, SB, ki,scale);
            for (int mii = 0; mii < MII / wmmaM; mii += 1)
            {
                for (int nii = 0; nii < NII / wmmaN; nii += 1)
                {
                    // 16x16x16 for each wmma
                    nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], 
                    FragBfp16[nii], Accum[mii * (NII / wmmaN) + nii]);
                }
            }
        }
    }
    storeAccum(SC, Accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
}
torch::Tensor Mixgemmallignedfp16int8(
    const torch::Tensor& ind,
    size_t lenid,
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2,  //int8
   const torch::Tensor& scaletensor  // float
    ){
    auto m = mat1.sizes()[0];
    auto n = mat2.sizes()[0];
    auto k = mat1.sizes()[1];
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(mat1.device());  
    at::Tensor input = torch::zeros({m, n}, options);

    half* input_ptr = (half *)input.data_ptr<at::Half>();
    half* mat1_ptr = (half *)mat1.data_ptr<at::Half>(); //  fp16
    //half* mat2_ptr = (half *)mat2.data_ptr<at::Half>();
    int8_t*  mat2_ptr = reinterpret_cast<int8_t *>(mat2.data_ptr<int8_t>());
    int* ind_ptr = (int *)ind.data_ptr<int>();

    int STAGES = 1;
    int smem_size = MAX(STAGES * 128 * 32 * 2 * 2, 128 * 128 * 4);
    if (smem_size >= (48 << 10))
    {
        cudaFuncSetAttribute(matmulfp16int8,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size);
    }
  dim3 dimBlock(32, 2 * STAGES , 2);
  dim3 dimGrid(n / 128, m / 128);

  float*  scale_ptr = reinterpret_cast<float *>(scaletensor.data_ptr<float>());
  float scale = scale_ptr[0];
  //std::cout << scaletensor[0] << std::endl;  
  std::cout << scale << std::endl;
  matmulfp16int8<<<dimGrid, dimBlock, smem_size, at::cuda::getCurrentCUDAStream()>>>
  (mat1_ptr, mat2_ptr, input_ptr, m, n, k, 1.0, 0.0, scale);
  return input;
}



__global__ void elementWiseMultiplyKernel(const int32_t* y, const float x_scale,
     const float *scale, const int m, const int n, half * out, half * outliers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float tmp = (x_scale ) * ( scale[0]);
    for (int i = idx; i < m * n; i += stride) {
        out[i] = (half) ((static_cast<float>(y[i]) / 127.0) * tmp + static_cast<float>(outliers[i]));
    }
}
__global__ void elementWiseMultiplyKernelNoOutliers(const int32_t* y, const float x_scale,
     const float *scale, const int m, const int n, half * out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float tmp = (x_scale ) * ( scale[0]);
    for (int i = idx; i < m * n; i += stride) {
        out[i] = (half) ((static_cast<float>(y[i]) / 127.0) * tmp );
    }
}

torch::Tensor  elementWiseMultiply(const torch::Tensor & y, 
        const int m,
        const int n,
        const float x_scale, 
        const torch::Tensor  &scale,
        const torch::Tensor & outliers) {
        // Get the size of the tensor

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(y.device());  
    at::Tensor output = torch::zeros({m, n}, options);    
    int32_t * d_y = (int32_t *)y.data_ptr();
    half * out = (half *)output.data_ptr();
    half * d_outliers = (half *)outliers.data_ptr();

    float * d_scale = (float *)scale.data_ptr();
    // Launch CUDA kernel
    int blockSize = 256;
    int numBlocks = (m * n + blockSize - 1) / blockSize;

    elementWiseMultiplyKernel<<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>
    (d_y, x_scale, d_scale, m, n, out, d_outliers);
    return output;

}



__global__ void elementWiseQuantInt8_Kernel(const half* inputs, const half * x_scale,
   const int m, const int n, int8_t * out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float tmp =  static_cast<float>(x_scale[0]);
 
    for (int i = idx; i < m * n; i += stride) {
        out[i] = static_cast<int8_t>( rintf  ( (static_cast<float>(inputs[i])  * 127.0) / tmp ) );
    }
}

// q_x = (inputs / x_scale).round().to(torch.int8)
torch::Tensor  elementWiseQuantInt8(torch::Tensor & input, 
        torch::Tensor  &x_scale) {
        // Get the size of the tensor
    int m = input.size(0);
    int n = input.size(1);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(input.device());  
    at::Tensor output = torch::zeros({m, n}, options);    
    half * d_input = (half *)input.data_ptr();
    half * d_x_scale = (half *)x_scale.data_ptr();
    int8_t * d_output = (int8_t *)output.data_ptr();

 
    int blockSize = 256;
    int numBlocks = (m * n + blockSize - 1) / blockSize;

    elementWiseQuantInt8_Kernel<<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>
    (d_input, d_x_scale, m, n, d_output);
    return output;

}


torch::Tensor  elementWiseMultiplyNoOurliers(torch::Tensor & y, 
    int m, int n,
        const float x_scale, 
        torch::Tensor  &scale) {
        // Get the size of the tensor

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(y.device());  
    at::Tensor output = torch::zeros({m, n}, options);    
    int32_t * d_y = (int32_t *)y.data_ptr();
    half * out = (half *)output.data_ptr();

    float * d_scale = (float *)scale.data_ptr();
    // Launch CUDA kernel
    int blockSize = 256;
    int numBlocks = (m * n + blockSize - 1) / blockSize;

    elementWiseMultiplyKernelNoOutliers<<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>
    (d_y, x_scale, d_scale, m, n, out);
    return output;
    
 
}



void  elementWiseQuantInt8CStyle(torch::Tensor & input, 
        int m, int n,
        torch::Tensor  &x_scale, torch::Tensor & output) {
        // Get the size of the tensor

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(input.device());  

    half * d_input = (half *)input.data_ptr();
    half * d_x_scale = (half *)x_scale.data_ptr();
    int8_t * d_output = (int8_t *)output.data_ptr();

 
    int blockSize = 256;
    int numBlocks = (m * n + blockSize - 1) / blockSize;

    elementWiseQuantInt8_Kernel<<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>
    (d_input, d_x_scale, m, n, d_output);
    return ;

}

template< int N, int blockNUM>
__global__ void ExtractOutliers_Kernel_row_order
    ( half *d_input,  half * d_weight, 
    const int m, const int n, const int k, 
    half * d_activatetion_output,  half *  d_weight_fp16, 
    const int64_t *ind, const int lenind){
    
    int col_id = blockIdx.y;
    int tid = threadIdx.x;
    int row_id = blockIdx.x * N + tid;

    int col = ind[col_id];
    half *source_d_input = d_input + col + row_id * k;
    half *source_d_weight = d_weight + col + row_id * k;
    half *target_d_activatetion = d_activatetion_output + col_id * m + row_id;
    half *target_d_weight_fp16 = d_weight_fp16 + col_id * n+ row_id; 

    int min_ = min(m,n);
    for (int i = row_id ; i < min_; i+= (N*blockNUM)){
        
        target_d_activatetion[ 0 ] = source_d_input[  0 ];
        source_d_input[0] = 0.0;
        target_d_activatetion +=  (N*blockNUM);
        source_d_input += k * (N*blockNUM);
        target_d_weight_fp16 [ 0 ] = source_d_weight[ 0 ];
        target_d_weight_fp16 += (N*blockNUM);
        source_d_weight += k * (N*blockNUM);            


    }
    if (n > m){
        // 先处理 m 部分
        target_d_weight_fp16 = d_weight_fp16 + col_id * n + (m + row_id); 
        source_d_weight = d_weight + col + (m + row_id)  * k;
        for (int i = m + row_id ; i < n; i+= (N*blockNUM)){
            target_d_weight_fp16 [ 0 ] = source_d_weight[ 0 ];
            target_d_weight_fp16 += (N*blockNUM);
            source_d_weight += k * (N*blockNUM);            
        }
        //
    }else{
        half *target_d_activatetion = d_activatetion_output + col_id * m + (n + row_id);
        half *source_d_input = d_input + col + (n + row_id) * k;

        for (int i = n + row_id ; i < m; i+= (N*blockNUM)){
            target_d_activatetion [ 0 ] = source_d_input[ 0 ];
            source_d_input[ 0 ] = 0.0;
            target_d_activatetion += (N*blockNUM);
            source_d_input += k * (N*blockNUM);            
        }

    }

}
void  ExtractOutliers( torch::Tensor & input, 
        torch::Tensor & weight, 
        torch::Tensor & activatetion_fp16, 
        torch::Tensor & weight_fp16, 
        torch::Tensor & ind) {
        // Get the size of the tensor
    int m = input.size(0);
    int k = input.size(1);
    int n = weight.size(0);
    int lenind = ind.size(0);

    half * d_input = (half *)input.data_ptr();
    half * d_weight = (half *)weight.data_ptr();
    half * d_activatetion_output = (half *)activatetion_fp16.data_ptr();
    half * d_weight_fp16 = (half *)weight_fp16.data_ptr();
    int64_t *d_ind= (int64_t *)ind.data_ptr();
 
    const int blockSize = 128;
    const int NumberOfblocksx = 16; 
    
    dim3 numBlocks(NumberOfblocksx, lenind);

    //   each outliers use 4 blocks, sice the row is 4096  
    //   4096 / 4 = 1024 
    //   each threads computes
    //
    //  每一个列启动4个block
    //  128个线程
    // 一次可以处理512元素
    // 如果是4096 需要处理8次
    ExtractOutliers_Kernel_row_order<blockSize,NumberOfblocksx><<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>
    (d_input, d_weight, m, n, k, d_activatetion_output, d_weight_fp16, d_ind, lenind);
    return ;

}



__inline__ __device__ half warpReduceMax(half val) 
{
    const unsigned int FULL_MASK = 0xffffffff;

    for (int mask = warpSize / 2; mask > 0; mask /= 2) 
    {
        val = __hmax (__shfl_xor_sync(FULL_MASK, val, mask), val);
    }
    
    return val;
}

template <int NUM_WARP>
__global__  void FindOutliers_kernel(int64_t *ind, int *lenind, half *input, half *weight, int m, int n, int k, half * sig,
        half *maxvec, half *outliersOutput, half* weightOutput ){

    // 一个warp 处理一列

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int start_col = blockIdx.x * NUM_WARP + warp_id;
    int start_row = lane_id * 4;


    half sig_ = sig[0];

    
    half *start = input +  start_col * m ;
    __shared__ int flag[4];
    for (int i = 0 ; i < 4; ++i){
        flag[i] = 0;
    }

    __syncthreads();

 

    half max = 0.0;

    for (int i = start_row; i < m ; i+=  (WARP_SIZE) * 4  ){
        for (int j = 0; j < 4; ++j){
            if ( __habs(start[i + j]) > sig_){
                flag[warp_id] = 1;
                break;
            }
            if (__habs(start[i + j]) > max){
                max = __habs(start[i + j]); 
            }

        }
        if (flag[warp_id]){
            
            break;
        }

    }

    __syncthreads();
    if (!flag[warp_id]){
        max = warpReduceMax(max);
        ind[start_col] = 0;
    }else{
        if (lane_id == 0){
            ind[start_col] = atomicAdd(lenind,1) + 1;
        }

    }
    
    __syncthreads();
    for (int k = 0; k < NUM_WARP; ++k)
        if (flag[k])
        {
            int start_col = blockIdx.x * NUM_WARP + k;
            // 提取离群点
            int col = ind[start_col] - 1;
            half *start = input +  start_col * m ;
            half *outliersOutput_ = outliersOutput + col * m;   
            int  start_row = tid * 4;   
            for (int i = start_row; i < m ; i+=  (WARP_SIZE * NUM_WARP) * 4  ){
                // for (int j = 0; j < 4; ++j){
                //     outliersOutput_[i + j] = start[i + j];
                // }
                *(int64_t *)(outliersOutput_ + i) =  *(int64_t *)(start + i);
                *(int64_t *)(start + i) = 0;
            }
            half *start_weight = weight +  start_col * n ;
            half *weightOutput_ = weightOutput + col * n;  
            for (int i = start_row; i < n ; i+=  (WARP_SIZE * NUM_WARP) * 4  ){
                // for (int j = 0; j < 4; ++j){
                //     weightOutput_[i + j] = start_weight[i + j];
                // }
                 *(int64_t *)(weightOutput_ + i) =  *(int64_t *)(start_weight + i);
            }
    }
    if (lane_id == 0)
        maxvec[start_col] = max;


}

torch::Tensor FindOutliers(
    torch::Tensor & ind,
    torch::Tensor & input,
    torch::Tensor & weight,
    int m, int n, int k,
    torch::Tensor & sig, // half
    torch::Tensor & maxvec, // half
    torch::Tensor & input_out,
    torch::Tensor & weight_out    
){


    const int blockSize = 128;
    const int NumberOfblocksx = (k + 4 - 1) / 4; // 一个warp 处理一列
    
    dim3 numBlocks(NumberOfblocksx);

    assert ( m % 4 == 0);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input_out.device());  
    at::Tensor lenind_ = torch::zeros({1, 1}, options);    

    FindOutliers_kernel<4><<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        (int64_t *)ind.data_ptr(),
        (int *)lenind_.data_ptr(),
        (half *)input.data_ptr(),
        (half *)weight.data_ptr(),
        m,
        n,
        k,
        (half *)sig.data_ptr(),
        (half *)maxvec.data_ptr(),
        (half *)input_out.data_ptr(),
        (half *)weight_out.data_ptr()      
    );

    return lenind_;
}






 
__global__  void FindOutliersAndSetToZeros_kernel(const int *ind,  half *input, 
        half *outliersOutput, int m, int k, int len){
 

    int tid = threadIdx.x;
 
    int start_col = blockIdx.x ;
 
    if (start_col > len)
        return ;

  
 
 
    int col = ind[start_col];
    half *start = input +  col ;
    half *outliersOutput_ = outliersOutput + start_col;   
 
    for (int i = tid; i < m ; i+=  128  ){
        outliersOutput_[ i * len ] = start[ i * k ] ;
        start[ i * k ] = 0.0;
    }
 
 


}
torch::Tensor ExtractOutliersAndSetToZeros(
    torch::Tensor & ind,
    torch::Tensor & input
 
){

    int len = ind.size(0);
    
    const int blockSize = 128;
    const int NumberOfblocksx = len; // 一个warp 处理一列
    int m = input.size(0);
    int k = input.size(1);


    dim3 numBlocks(NumberOfblocksx);


    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(input.device());  
    at::Tensor output = torch::zeros({m, len}, options);    
   

    FindOutliersAndSetToZeros_kernel<<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        (int *)ind.data_ptr(),
        (half *)input.data_ptr(),
        (half *)output.data_ptr(),
        m,
        k,
        len
    );

    return output;
}
// __global__ void GetScaleFactor_kernel(int *ind,
//         half *input,
//         half *max_factor,
//         int m,
//         int n){
        
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;

//     half * start = input + bid * n;
//     __shared__ half out[256];
//     out[tid] = 0;
//     __syncthreads();


//     for (int k = tid; k < n ; k+=256){

//     }

// }


// 找到每一行的最大值，跳过ind
// torch::Tensor GetScaleFactor(
//     torch::Tensor & ind,
//     torch::Tensor & input,
//     int m, 
//     int n  
// ){


//     const int blockSize = 256;
//     const int NumberOfblocksx = m; // 一个  block 处理一行
    
//     dim3 numBlocks(NumberOfblocksx);

//     assert ( m % 4 == 0);
//     auto options = torch::TensorOptions().dtype(torch::kFloat16).device(input.device());  
//     at::Tensor max_factor = torch::zeros({m, 1}, options);    

//     GetScaleFactor_kernel<4><<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
//         (int *)ind.data_ptr(),
//         (half *)input.data_ptr(),
//         (half *)max_factor.data_ptr(),
//         m,
//         n
//     );

//     return max_factor;
// }

// torch::Tensor ExtractOutliers_to_cache(
//     torch::Tensor & ind,
//     torch::Tensor & input,
//     int m, int n, int k,
//     torch::Tensor & sig, // half
//     torch::Tensor & maxvec, // half
//     torch::Tensor & input_out,
//     torch::Tensor & weight_out    
// ){


//     const int blockSize = 128;
//     const int NumberOfblocksx = (k + 4 - 1) / 4; // 一个warp 处理一列
    
//     dim3 numBlocks(NumberOfblocksx);

//     assert ( m % 4 == 0);
//     auto options = torch::TensorOptions().dtype(torch::kInt32).device(input_out.device());  
//     at::Tensor lenind_ = torch::zeros({1, 1}, options);    

//     ExtractOutliers_kernel<4><<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
//         (int64_t *)ind.data_ptr(),
//         (int *)lenind_.data_ptr(),
//         (half *)input.data_ptr(),
//         (half *)weight.data_ptr(),
//         m,
//         n,
//         k,
//         (half *)sig.data_ptr(),
//         (half *)maxvec.data_ptr(),
//         (half *)input_out.data_ptr(),
//         (half *)weight_out.data_ptr()      
//     );

//     return lenind_;
// }

template <int NUM_WARP, int EACH_TID_DO>
__global__  void FindOutliers_kernelRow(int64_t *ind, int *lenind, half *input, half *weight, int m, int n, int K, half * sig,
        half *maxvec, half *outliersOutput, half* weightOutput ){

    // 一个warp 处理一列

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int start_col = blockIdx.x * NUM_WARP + warp_id;
    int start_row = lane_id * EACH_TID_DO;


    half sig_ = sig[0];

    
    half *start = input +  start_col;
    __shared__ int flag[NUM_WARP];
    for (int i = 0 ; i < NUM_WARP; ++i){
        flag[i] = 0;
    }

    __syncthreads();

 

    half max = 0.0;

    for (int i = start_row; i < m ; i+= (WARP_SIZE * EACH_TID_DO)   ){

        for (int j = 0; j < EACH_TID_DO; ++j){
            half tmp =  __habs(start[ (i + j) * K]);
            if ( tmp > sig_){
                flag[warp_id] = 1;
                break;
            }
            if ( tmp > max){
                max = tmp; 
            }

        }
        if (flag[warp_id]){
            
            break;
        }

    }

    __syncthreads();
    if (!flag[warp_id]){
        max = warpReduceMax(max);
        ind[start_col] = 0;
    }else{
        if (lane_id == 0){
            ind[start_col] = atomicAdd(lenind,1) + 1;
        }

    }
    
    __syncthreads();
    for (int k = 0; k < NUM_WARP; ++k)
        if (flag[k])
        {
            int start_col = blockIdx.x * NUM_WARP + k;
            // 提取离群点
            int col = ind[start_col] - 1;
            half *start = input +  start_col  ;
            half *outliersOutput_ = outliersOutput + col * m;   
            int  start_row = tid ;   
            for (int i = start_row; i < m ; i+=  (WARP_SIZE * NUM_WARP)   ){
                outliersOutput_[i] = start[(i) * K];
                start[(i) * K] = 0;

            }
            half *start_weight = weight +  start_col * n ;
            half *weightOutput_ = weightOutput + col * n;  
            for (int i = start_row * EACH_TID_DO; i < n ; i+=  (WARP_SIZE * NUM_WARP) * EACH_TID_DO ){
                // for (int j = 0; j < 4; ++j){
                //     weightOutput_[i + j] = start_weight[i + j];
                // }
                 *(int64_t *)(weightOutput_ + i) =  *(int64_t *)(start_weight + i);
            }
    }
    if (lane_id == 0)
        maxvec[start_col] = max;


}

void FindOutliersRow(
    torch::Tensor & ind,
    torch::Tensor & lenind_,
    torch::Tensor & input,
    torch::Tensor & weight,
    int m, int n, int k,
    torch::Tensor & sig, // half
    torch::Tensor & maxvec, // half
    torch::Tensor & input_out,
    torch::Tensor & weight_out    
){


    const int blockSize = 128;
    const int NumberOfblocksx = (k + 4 - 1) / 4; // 一个warp 处理一列
    
    dim3 numBlocks(NumberOfblocksx);

    assert ( m % 4 == 0);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input_out.device());  

    cudaMemset((int *)lenind_.data_ptr(),0,sizeof(int));
    FindOutliers_kernelRow<4,4><<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        (int64_t *)ind.data_ptr(),
        (int *)lenind_.data_ptr(),
        (half *)input.data_ptr(),
        (half *)weight.data_ptr(),
        m,
        n,
        k,
        (half *)sig.data_ptr(),
        (half *)maxvec.data_ptr(),
        (half *)input_out.data_ptr(),
        (half *)weight_out.data_ptr()      
    );

    return ;
}







__global__ void elementWiseAddKernel(const float* y, 
     const int m, const int n, half * out, half * outliers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < m * n; i += stride) {

        out[i] =  half( y[i] + float(outliers[i]));
    }
}
torch::Tensor  elementWiseAdd(const torch::Tensor & y, 
        const int m,
        const int n,
        const torch::Tensor & outliers) {
        // Get the size of the tensor

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(y.device());  
    at::Tensor output = torch::zeros({m, n}, options);   

    float * d_y = (float *)y.data_ptr();
    half * out = (half *)output.data_ptr();
    half * d_outliers = (half *)outliers.data_ptr();

    // Launch CUDA kernel
    int blockSize = 256;
    int numBlocks = (m * n + blockSize - 1) / blockSize;

    elementWiseAddKernel<<<numBlocks, blockSize, 0, at::cuda::getCurrentCUDAStream().stream()>>>
    (d_y, m, n, out, d_outliers);
    return output;

}





 

 
 


__global__ void int8QuantizationCUDAKernel(
    int8_t *__restrict__ dst, const torch::Half *__restrict__ scale,
    const torch::Half *__restrict__ src, const unsigned rows,
    const unsigned cols) {
  const unsigned row = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned col = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= rows || col >= cols) {
    return;
  }
  const unsigned id = col + row * cols;
  dst[id] = __half2int_rn(__hdiv(src[id], scale[row]));
}
torch::Tensor int8QuantizationCUDA(const torch::Tensor &src,
                                   const torch::Tensor &scale) {
  torch::checkSameGPU("quantize", {src, "src", 0}, {scale, "scale", 1});
  torch::checkSize("quantize", torch::TensorArg{scale, "scale", 1}, 0,
                   src.size(0));
  unsigned rows = src.size(0);
  unsigned cols = src.size(1);
    
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(src.device());
  auto dst = torch::empty(
      {rows, cols}, options);
  dim3 block{std::min<unsigned>(cols, 32), std::min<unsigned>(rows, 16)};
  dim3 grid{(cols - 1) / block.x + 1, (rows - 1) / block.y + 1};
  int8QuantizationCUDAKernel<<<grid, block>>>(
      dst.data_ptr<int8_t>(), scale.data_ptr<torch::Half>(),
      src.data_ptr<torch::Half>(), rows, cols);
  return dst;
}


 

torch::Tensor Int8quantize(const torch::Tensor &src, const torch::Tensor &scale) {
  torch::checkAllContiguous("quantize", {{src, "src", 0}, {scale, "scale", 1}});
  torch::checkDeviceType("quantize", {src, scale}, at::DeviceType::CUDA);
  return int8QuantizationCUDA(src, scale);

}





#include "symmetric/gemm/device/gemm_dequant.h"
#include "symmetric/symmetric_internal.h"

torch::Tensor int8FusedDequantizeCUDA(
                                      const torch::Tensor &A,
                                      const torch::Tensor &B,
                                      const torch::Tensor &scale_row,
                                      const torch::Tensor &scale_col,
                                      const torch::Tensor &y, int M, int N, int K) {
  torch::checkAllSameGPU("int8FusedDequantize", {{A, "A", 0},
                                                 {B, "B", 1},
                                                 {scale_row, "scale_row", 2},
                                                 {scale_col, "scale_col", 3},
                                                 {y, "y", 4}});

  auto D = torch::empty({M, N}, torch::dtype(torch::kF16).device(A.device()));

  using Gemm = cutlass::gemm::device::symmetric::GemmDequant<
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      cutlass::half_t,                 // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {(cutlass::half_t *)y.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)D.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_col.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_row.data_ptr<torch::Half>(), M},
      Gemm::ElementC(1)};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return D;
}


torch::Tensor int8FusedDequantize(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &y,  int M, int N, int K) {
  torch::checkAllContiguous("int8FusedDequantize", {{A, "A", 0},
                                                    {B, "B", 1},
                                                    {scale_row, "scale_row", 2},
                                                    {scale_col, "scale_col", 3},
                                                    {y, "y", 4}});
  torch::checkDeviceType("int8FusedDequantize", {A, B, scale_row, scale_col, y},
                         at::DeviceType::CUDA);
  return int8FusedDequantizeCUDA(A, B, scale_row, scale_col, y, M, N, K);
}




#include "symmetric/gemm/device/gemm_dequantsilu.h"
torch::Tensor int8FusedDequantizeSiluCUDA(
                                      const torch::Tensor &A,
                                      const torch::Tensor &B,
                                      const torch::Tensor &scale_row,
                                      const torch::Tensor &scale_col,
                                      const torch::Tensor &y,  int M, int N, int K) {
  torch::checkAllSameGPU("int8FusedDequantizeSiluCUDA", {{A, "A", 0},
                                                 {B, "B", 1},
                                                 {scale_row, "scale_row", 2},
                                                 {scale_col, "scale_col", 3},
                                                 {y, "y", 4}});

  auto D = torch::empty({M, N}, torch::dtype(torch::kF16).device(A.device()));

  using Gemm = cutlass::gemm::device::symmetric::GemmDequantSilu<
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      cutlass::half_t,                 // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {(cutlass::half_t *)y.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)D.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_col.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_row.data_ptr<torch::Half>(), M},
      Gemm::ElementC(1)};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return D;
}
torch::Tensor int8FusedDequantizeSilu(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &y,  int M, int N, int K) {
  torch::checkAllContiguous("int8FusedDequantizeSilu", {{A, "A", 0},
                                                    {B, "B", 1},
                                                    {scale_row, "scale_row", 2},
                                                    {scale_col, "scale_col", 3},
                                                    {y, "y", 4}});
  torch::checkDeviceType("int8FusedDequantizeSilu", {A, B, scale_row, scale_col, y},
                         at::DeviceType::CUDA);
  return int8FusedDequantizeSiluCUDA(A, B, scale_row, scale_col, y,  M, N, K);
}




template <typename T>
__device__ __half convertToHalf(T value) {
  return __int2half_rn(static_cast<int>(value));
}

template <>
__device__ __half convertToHalf(torch::Half value) {
  return (__half)value;
}

template <typename T>
__global__ void dequantizationKernel(torch::Half *__restrict__ out,
                                     const T *__restrict__ x,
                                     const torch::Half *__restrict__ scaleRow,
                                     const torch::Half *__restrict__ scaleCol,
                                     const torch::Half *__restrict__ y,
                                     const unsigned rows, const unsigned cols) {
  const unsigned row = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned col = threadIdx.x + blockIdx.x * blockDim.x;
  if (col >= cols) {
    return;
  }

  if (row >= rows) {
    return;
  }

  float xElement =  static_cast<float>(x[col + row * cols]);

  out[col + row * cols] =
      __hadd(   __float2half( ( xElement * __half2float(scaleRow[row])) * __half2float(scaleCol[col]) ) ,
      y[col + row * cols]);
}


torch::Tensor dequantizationCUDA(const torch::Tensor &x,
                                 const torch::Tensor &scaleRow,
                                 const torch::Tensor &scaleCol,
                                 const torch::Tensor &y, int M, int N) {

  unsigned rows = x.size(0);
  unsigned cols = x.size(1);
    

  auto out = torch::empty({M, N}, torch::dtype(torch::kF16).device(x.device()));

  //auto out = torch::empty_like(y);
  dim3 block{std::min<unsigned>(cols, 16),
             std::min<unsigned>((rows - 1) + 1, 16)};
  dim3 grid{(cols - 1) / block.x + 1, (rows - 1) / block.y + 1};
  dequantizationKernel<<<grid, block>>>(
      out.data_ptr<torch::Half>(), x.data_ptr<int>(),
      scaleRow.data_ptr<torch::Half>(), scaleCol.data_ptr<torch::Half>(),
      y.data_ptr<torch::Half>(), rows, cols);
  return out;
}


torch::Tensor dequantizeInt8(const torch::Tensor &x, const torch::Tensor &scaleRow,
                         const torch::Tensor &scaleCol, const torch::Tensor &y,
                         const int bits, int M, int N) {


    return dequantizationCUDA(x, scaleRow, scaleCol, y, M, N);

}



__forceinline__ __host__ __device__ float silu(float x)
{
    return x / (1.f + expf(-x));
}
template <typename T>
__global__ void dequantizationKernelSilu(torch::Half *__restrict__ out,
                                     const T *__restrict__ x,
                                     const torch::Half *__restrict__ scaleRow,
                                     const torch::Half *__restrict__ scaleCol,
                                     const torch::Half *__restrict__ y,
                                     const unsigned rows, const unsigned cols) {
  const unsigned row = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned col = threadIdx.x + blockIdx.x * blockDim.x;
  if (col >= cols) {
    return;
  }

  if (row >= rows) {
    return;
  }

  float xElement =  static_cast<float>(x[col + row * cols]);
  float tmp  = silu(  ( xElement * __half2float(scaleRow[row])) * __half2float(scaleCol[col])   +  __half2float(y[col + row * cols]) );
  out[col + row * cols] =  __float2half(tmp);
  
}
torch::Tensor dequantizationCUDASilu(const torch::Tensor &x,
                                 const torch::Tensor &scaleRow,
                                 const torch::Tensor &scaleCol,
                                 const torch::Tensor &y, int M, int N) {

  unsigned rows = x.size(0);
  unsigned cols = x.size(1);
    

  auto out = torch::empty({M, N}, torch::dtype(torch::kF16).device(x.device()));

  //auto out = torch::empty_like(y);
  dim3 block{std::min<unsigned>(cols, 16),
             std::min<unsigned>((rows - 1) + 1, 16)};
  dim3 grid{(cols - 1) / block.x + 1, (rows - 1) / block.y + 1};
  dequantizationKernelSilu<<<grid, block>>>(
      out.data_ptr<torch::Half>(), x.data_ptr<int>(),
      scaleRow.data_ptr<torch::Half>(), scaleCol.data_ptr<torch::Half>(),
      y.data_ptr<torch::Half>(), rows, cols);
  return out;
}


torch::Tensor dequantizeInt8Silu(const torch::Tensor &x, const torch::Tensor &scaleRow,
                         const torch::Tensor &scaleCol, const torch::Tensor &y,
                         const int bits, int M, int N) {


    return dequantizationCUDASilu(x, scaleRow, scaleCol, y, M, N);

}

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_types.h>


torch::Tensor linear_a8_w8_b32_o32(torch::Tensor &input,  // INT8
                                   torch::Tensor &weight, // INT8
                                   torch::Tensor &cache){
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());  
  at::Tensor out = torch::zeros({M, N}, options);    

//   using ElementOutput = int32_t;
//   using ElementAccumulator = int32_t;
//   using ElementComputeEpilogue = int32_t;
//   using ElementInputA = int8_t; // <- data type of elements in input matrix A
//   using ElementInputB = int8_t; // <- data type of elements in input matrix B

//   // The code section below describes matrix layout of input and output
//   // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
//   // for Matrix C
//   using LayoutInputA = cutlass::layout::RowMajor;
//   using LayoutInputB = cutlass::layout::ColumnMajor;
//   using LayoutOutput = cutlass::layout::RowMajor;


//   using Gemm = cutlass::gemm::device::Gemm<
//       int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
//       ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
//       cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
//       cutlass::gemm::GemmShape<256, 128, 64>,
//       cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
//       cutlass::epilogue::thread::LinearCombination<
//           ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
//           ElementAccumulator, ElementComputeEpilogue,
//           cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
//       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;


//   auto input_size = cutlass::MatrixCoord(M, K);
//   auto weight_size = cutlass::MatrixCoord(K, N);
//   auto output_size = cutlass::MatrixCoord(M, N);

//   auto device = input.device();
//   // use the broadcasted bias as the output

//   // constexpr int kSparse = Gemm::kSparse;
//   // How many elements of A are covered per ElementE
//   // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
//   // The size of individual meta data
//   // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
//   cutlass::gemm::GemmCoord problem_size(M, N, K);

//   cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
//       input.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
//   cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
//       weight.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
//   cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
//       out.data_ptr<int32_t>(), LayoutOutput::packed(output_size));

//   // Initialize alpha and beta for dot product computation
//   ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

//   typename Gemm::Arguments arguments{
//       problem_size, // <- problem size of matrix multiplication
//       input_ref,    // <- reference to matrix A on device
//       weight_ref,   // <- reference to matrix B on device
//       out_ref,      // <- reference to matrix C on device
//       out_ref,      // <- reference to matrix D on device
//       {alpha},      1};
//   Gemm gemm_op;



//   // Check the problem size is supported or not
//   cutlass::Status status = gemm_op.can_implement(arguments);
//   if (status != cutlass::Status::kSuccess) {
//     throw std::runtime_error("cutlass cannot implement");
//   }

//   // Initialize CUTLASS kernel with arguments and workspace pointer
//   status = gemm_op.initialize(arguments, cache.data_ptr<int8_t>());
//   if (status != cutlass::Status::kSuccess) {
//     throw std::runtime_error("cutlass cannot initialize");
//   }

//   status = gemm_op();
//   if (status != cutlass::Status::kSuccess) {
//     throw std::runtime_error("cutlass cannot run");
//   }

  return out;
}



__device__ void warpReduce(volatile half* cache, unsigned int tid){

    cache[tid] =  __hmax ( __habs(cache[tid + 32]),  cache[tid]);
    __syncthreads();
    cache[tid] =  __hmax ( __habs(cache[tid + 16]),  cache[tid]);
    __syncthreads();
    cache[tid] =  __hmax ( __habs(cache[tid + 8]),  cache[tid]);
    __syncthreads();
    cache[tid] =  __hmax ( __habs(cache[tid + 4]),  cache[tid]);
    __syncthreads();
    cache[tid] =  __hmax ( __habs(cache[tid + 2]),  cache[tid]);
    __syncthreads();
    cache[tid] =  __hmax ( __habs(cache[tid + 1]),  cache[tid]);
    __syncthreads();
}

template<int size>
__global__ void FindRowScaleKernel(int8_t * output, const half * d_in, half * scale, int rows, int cols){

    __shared__ half sdata[size];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x ;
    if (bid > rows)
        return ;
    const  __half *start = d_in + bid * cols;
    int8_t * d_out = output + bid * cols;
    sdata[tid] = __habs(start[tid]); 
    for (int i = tid + size; i < cols; i += size)
        sdata[tid] = __hmax ( __habs(start[i]),  sdata[tid] ); 
    __syncthreads();


    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s >= 1; s >>=1 ) {
        if (tid < s) {
            sdata[tid] =  __hmax ( __habs(sdata[tid + s]),  sdata[tid]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    //if (tid < 32) warpReduce(sdata, tid);

    __syncthreads();

    half quant_scales = __hdiv( sdata[0], 127.0);
    if (tid == 0){
        scale[bid] = quant_scales;
    }
    // quant
    for (int i = tid ; i < cols; i += size)
        d_out[i] =  static_cast<int8_t>(__half2int_rn( __hdiv( start[i], quant_scales ) ))  ; 
    __syncthreads();    

}

torch::Tensor FindRowScale(  const torch::Tensor &x,  torch::Tensor &scaleRow,
                         int rows, int cols) {


  auto options = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
  auto quant_out = torch::zeros(
      {rows, cols}, options);
    dim3 block(32);
    dim3 grid(rows, 1);
    FindRowScaleKernel<32><<<grid, block>>>(
        (int8_t *)quant_out.data_ptr<int8_t>(),
        (half *)x.data_ptr<torch::Half>(), (half *)scaleRow.data_ptr<torch::Half>(),
         rows, cols);
    return quant_out;

}