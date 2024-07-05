#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "symmetric/gemm/device/gemm_dequant.h"
#include "int8FusedDequantizeCUDA.h"
#include <cutlass/cutlass.h>
#include <cutlass/half.h>


// col 应该是 4096
// rows 应该是 12288
template<int size, typename T>
__global__ void DequantKernel(const T * input,  half * output, const half * scale,
     int rows, int cols){



    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x ;
    if (bid > rows)
        return ;

    const T *start = input + bid * cols;
     __half  * d_out = output + bid * cols;
    half quant_scales = scale[bid];

    // quant
    for (int i = tid ; i < cols; i += size){

 
        half tmp_ = (half) start[i];
        d_out[i] =  static_cast<half>(( __hmul( tmp_, quant_scales ) ))  ; 
    }
    __syncthreads();    

}

void int8dequant(int rows, int cols,  half * output, const int8_t *src,const  half *scale,
        cudaStream_t stream)
{


    dim3 block(256);
    dim3 grid(rows, 1);
    DequantKernel<256, int8_t><<<grid, block,0, stream>>>(
                src,
                output, scale,
                rows, cols);

}

void int8dequant(int rows, int cols,  half * output, const half *src,const  half *scale,
            cudaStream_t stream)
{


    dim3 block(256);
    dim3 grid(rows, 1);
    DequantKernel<256, half><<<grid, block,0, stream>>>(
                src,
                output, scale,
                rows, cols);

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
    for (int i = tid ; i < cols; i += size)
        d_out[i] =  static_cast<int8_t>(__half2int_rn( __hdiv( start[i], quant_scales ) ))  ; 
    __syncthreads();    

}

__global__ void PrintKernel(const half *A, int M){

    int tid = threadIdx.x;
    float tmp = __half2float(A[tid]);
 
    printf("A %d is %.6f\t",tid, tmp);
}
void print_half(const half *A, int M){
    dim3 block(M);
    dim3 grid(1, 1);
    PrintKernel<<<grid, block>>>(
                A, M);

}

__global__ void PrintKernelint( const  int *A, int M){

    int tid = threadIdx.x;
    int tmp =  (A[tid]);
 
    printf("A %d is %d\t",tid, tmp);
}
void print_int(  const int *A, int M){
    dim3 block(M);
    dim3 grid(1, 1);
    PrintKernelint<<<grid, block>>>(
                A, M);

}

void int8quant(int rows, int cols, const half * src, int8_t *output, 
        half *scale, cudaStream_t stream){


    dim3 block(256);
    dim3 grid(rows, 1);
    FindRowScaleKernel<256><<<grid, block, 0, stream>>>(
                output,
                src, scale,
                rows, cols);

}
void int8FusedDequantizeCUDA(const int8_t *A,
                             const int8_t *B,
                             const half *scale_row,
                             const half *scale_col,
                             half *y, half *D, 
                             int M, int N, int K,
                             char * workspace,
                             cudaStream_t stream) {

 
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
  //cutlass::Status status = gemmOp(stream);


  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {(const int8_t *)A, K},
      {(const int8_t *)B, K},
      {(const cutlass::half_t *)y, N},
      {(cutlass::half_t *)D, N},
      {(const cutlass::half_t *)scale_col, N},
      {(const cutlass::half_t *)scale_row, M},
      Gemm::ElementC(1)};

  gemmOp.initialize(arguments, workspace, stream);
  //status = gemmOp(arguments);
  gemmOp.run(stream);
 
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
        // start[ i * k ] = 0.0;
    }
 
 


}
void ExtractOutliersAndSetToZeros(int M, int N, const half * A, half *fp_A, 
        const int *ind, const int len, cudaStream_t stream){


    const int blockSize = 128;
 

    half * tmp = const_cast<half*>(A);
    dim3 numBlocks(len);        
    FindOutliersAndSetToZeros_kernel<<<numBlocks, blockSize, 0, 
            stream>>>(
            ind,
            tmp,
            fp_A,
            M,
            N,
            len
        );

}



template <typename T>
__device__ __half convertToHalf(T value) {
  return __int2half_rn(static_cast<int>(value));
}

template <>
__device__ __half convertToHalf(half value) {
  return (__half)value;
}

template <typename T>
__global__ void dequantizationKernel(half *__restrict__ out,
                                     const T *__restrict__ x,
                                     const half *__restrict__ scaleRow,
                                     const half *__restrict__ scaleCol,
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
      out[col + row * cols]);
}



void dequantizationCUDA(half * out, const int * x,
                                 const half * scaleRow,
                                 const half * scaleCol, int M, int N, cudaStream_t s) {

  unsigned rows = M;
  unsigned cols = N;
    


  //auto out = torch::empty_like(y);
  dim3 block{std::min<unsigned>(cols, 16),
             std::min<unsigned>((rows - 1) + 1, 16)};
  dim3 grid{(cols - 1) / block.x + 1, (rows - 1) / block.y + 1};
  dequantizationKernel<<<grid, block, 0, s>>>(
      out, x ,
      scaleRow , scaleCol, rows, cols);
  
}





__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source)
{
    uint4 result;

    uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
    static constexpr uint32_t TOP_MASK              = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
    // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
    // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
    // elt_67 to fp16 without having to shift them to the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
    // immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[0])
                    : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[1])
                    : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[2])
                    : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[3])
                    : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
    // half2 ctor. In this case, I chose performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    // static constexpr uint32_t NEG_72 = 0xd480d480;
    // Haotian: Let's use {-64, -64}.
    static constexpr uint32_t NEG_64 = 0xd400d400;

    // Finally, we construct the output numbers.
    // Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    // Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    return result;
}

// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

__device__ __forceinline__ int make_divisible(int c, int divisor){
  return (c + divisor - 1) / divisor;
}

__global__ void __launch_bounds__(64) gemm_forward_4bit_cuda_m16n128k32(int G,
 int split_k_iters, const half* __restrict__ A, const int* __restrict__ B,
 const half* __restrict__ scaling_factors, const int* __restrict__ zeros,
  int M, int IC, int OC, half* __restrict__ C) 
{
  static constexpr uint32_t ZERO = 0x0;
  float C_warp[32];
  __shared__ half A_shared[16 * (32 + 8)];
  __shared__ half B_shared[32 * (128 + 8)];

  int j_factors1 = ((OC + 128 - 1) / 128);
  int blockIdx_x = 0;
  int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
  int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

  half A_shared_warp[8];
  half B_shared_warp[32];
  for (int j_0_4_init = 0; j_0_4_init < 4; ++j_0_4_init) {
    for (int i = 0; i < 8; ++i) {
      C_warp[(j_0_4_init * 8) + i] = 0.0;
    }
  }

  static constexpr int row_stride_warp = 32 * 8 / 32;
  static constexpr int row_stride = 2 * 32 * 8 / 128;
  bool ld_zero_flag = (threadIdx.y * 32 + threadIdx.x) * 8 < 128;
  // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
  bool ld_A_flag = (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) < M;     // threadIdx.y is warp_id
  // bool wb_C_flag = (threadIdx.x / 4) < M;

  const half* A_ptr = A 
                + (((int)blockIdx_y) / j_factors1 * 16 + (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) * IC
                + (((int)threadIdx.x) % (32 / 8)) * 8;
  
  const int* B_ptr = B
            + ((int)threadIdx.y) * (OC / 8) * 2
            + (((int)threadIdx.x) / (128 / 8)) * (OC / 8)
            + (((int)blockIdx_y) % j_factors1) * (128 / 8)
            + (((int)threadIdx.x) % (128 / 8)) * 1;
// Why * 1 in the above line?
                        
  half* A_shared_ptr = A_shared 
                    + ((int)threadIdx.y) * row_stride_warp * (32 + 8) 
                    + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
                    + (((int)threadIdx.x) % (32 / 8) ) * 8;

  half* B_shared_ptr = B_shared
                    + ((int)threadIdx.y) * (row_stride / 2) * (128 + 8)
                    + (((int)threadIdx.x) / (128 / 8)) * (128 + 8)
                    + (((int)threadIdx.x) % (128 / 8)) * 8;
  
  const int* zeros_ptr = zeros
                + (((int)blockIdx_y) % j_factors1) * (128 / 8)
                + ((int)threadIdx.x) % (128 / 8);
  
  const half* scaling_factors_ptr = scaling_factors
                            + (((int)blockIdx_y) % j_factors1) * (128) 
                            + (((int)threadIdx.x) % (128 / 8)) * 8;

  half* C_ptr = C 
              + static_cast<long long>(blockIdx_z) * M * OC        // blockIdz.x -> split_k dim
              + (((int)blockIdx_y) % j_factors1) * 128
              + ((int)threadIdx.y) * 64
              + (((int)threadIdx.x) % 4) * 2;

  // preload s.f. and zeros
  int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
  if ((k_bound - 1) * split_k_iters * 32 + blockIdx_z * 32 >= IC) k_bound -= 1;
  for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
    int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
    __syncthreads();
    // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
    if (ld_A_flag)
    {
      *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
    }
    else
    {
      *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
    }

    // for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
    uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / G * (OC / 8));
    uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
    uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / G * (OC));
    /*
    if (blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 0 && threadIdx.y == 0){
      printf("%x %x %x %x %x %x %x %x\n", B_loaded_scale.x, B_loaded_scale.y, B_loaded_scale.z, B_loaded_scale.w, B_loaded_zero.x, B_loaded_zero.y, B_loaded_zero.z, B_loaded_zero.w);
    }
    */
    // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
    const int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

    for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0) {

      // B: 32 x 136 (128+8) float16
      // each warp: 32 x 4
      // each thr: read 32 bit -> convert to 8xFP16 (a UINT4) -> scale and minus zero -> WB UINT4
      // *(uint4*)(B_shared + ((((ax0_ax1_fused_0 * 544) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(B + ((((((k_0_0 * 163840) + (ax0_ax1_fused_0 * 20480)) + (((int)threadIdx.y) * 10240)) + ((((int)threadIdx.x) >> 4) * 5120)) + (((int)blockIdx_y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
      // row stride in shared memory: (NWARPS * 32 * 8 / cta_N) 
      uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
      uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
      //uint4 B_loaded_zero = *(uint4*)(zeros_shared + (threadIdx.x % (cta_N / 8)) * 8);

      // uint4 B_loaded_scale = *(uint4*)(scaling_factors_shared + (threadIdx.x % (cta_N / 8)) * 8);
      // - zero and * scale
      // TODO (Haotian): can save 4 assembly instructions if sormulate as deq = q * scale - zero * scale.
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
      asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
      asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
      /*
      if (ax0_ax1_fused_0 == 0 && blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 17 && threadIdx.y == 0){
        printf("[x] %X %X %X %X\n", B_loaded_fp16.x, B_loaded_fp16.y, B_loaded_fp16.z, B_loaded_fp16.w);
      }
      */

      // write back
      *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (128 + 8)) = B_loaded_fp16;
    }
    __syncthreads();

    for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
      {
        unsigned int addr;
        asm volatile(
          "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
          : "=r"(addr)
          : "l"((void *)((&(A_shared[(k_0_1 * 16)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
        );


        asm volatile(
          "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
          "{%0, %1, %2, %3}, [%4];\n"
          : "=r"(((unsigned *)(A_shared_warp + 0))[0]), "=r"(((unsigned *)(A_shared_warp + 0))[1]), "=r"(((unsigned *)(A_shared_warp + 0))[2]), "=r"(((unsigned *)(A_shared_warp + 0))[3])
          : "r"(addr)
        );
      }

      for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
        {
          unsigned int addr;
          asm volatile(
            "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
            : "=r"(addr)
            : "l"((void *)((&(B_shared[(((k_0_1 * 2176) + (((int)threadIdx.y) * 64)) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 136) + ((((int)threadIdx.x) >> 4) * 8))))
          );
          asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[3])
            : "r"(addr)
          );
        }
      }
      for (int j_0_4 = 0; j_0_4 < 4; ++j_0_4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
        {
          asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }

        {
          asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
            :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
#else
        {
          asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "=f"(((float *)(C_warp + (j_0_4 * 8)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[0]), "f"(((float *)(C_warp + (j_0_4 * 8)))[1]), "f"(((float *)(C_warp + (j_0_4 * 8)))[2]), "f"(((float *)(C_warp + (j_0_4 * 8)))[3]));
        }

        {
          asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            :  "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "=f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3])
            : "r"(((unsigned *)(A_shared_warp + 0))[0]), "r"(((unsigned *)(A_shared_warp + 0))[1]), "r"(((unsigned *)(A_shared_warp + 0))[2]), "r"(((unsigned *)(A_shared_warp + 0))[3]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[2]), "f"(((float *)(C_warp + ((j_0_4 * 8) + 4)))[3]));
        }
#endif
      }
    }
  }

// TODO: Shang: Hoist loop invariance.
  for (int ax1_0_1 = 0; ax1_0_1 < 4; ++ax1_0_1) {
    for (int local_id = 0; local_id < 8; ++local_id) {
      int row_offset = (((int)blockIdx_y) / j_factors1) * 16 + ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;
      if (row_offset < M)
      {
        *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
      }
    }
  }
}

__global__ void sum_all(const half * source, half *out, int M, int N){

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int col = blockIdx.y * 128 + tid;
 
  assert ( col < N);
  assert ( row < M);

  const half * src =  source + (row * N  + col) ; //先要定位到 row 和 col 然后求和！！！
  half *dest = out + ( row * N + col);
  half tmp = 0.0;
  for (int i = 0 ; i < 8; ++i){
      tmp = __hadd(src[ ( M * N  ) * i ], tmp);
  }
  dest[0] = tmp;
}
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
    half * tmp
    ){

    int num_in_feats = M;
    int num_in_channels = K;
    const int group_size = 128;
    const int split_k_iters = 8;
    int num_out_feats = M;
    int num_out_channels = N;  //12288 N
 
    int j_factors1 = num_out_channels / 128 / 1;
    dim3 num_blocks((num_out_feats + 16 - 1) / 16 * j_factors1 * split_k_iters);
    // threadIdx.x: 32
    // threadIdx.y: i_factors[2] * j_factors[2]
    dim3 threads_per_block(32, 2);
    gemm_forward_4bit_cuda_m16n128k32<<<num_blocks, threads_per_block, 0, stream>>>(
        group_size, split_k_iters, in_feats, kernel, scaling_factors, zeros, num_in_feats,
         num_in_channels, num_out_channels, tmp);

    assert ( M < 65536 );
    assert ( (N % 128) == 0 );
    dim3 num_blocks_sum( M, ( N / 128 ) );
    sum_all<<<num_blocks_sum, 128, 0, stream>>>(tmp, out_feats, M, N);

}