#include <pybind11/pybind11.h>

#include<vector>
#include "ops.cuh"


int cigemmlt_ampere_32(long long context, int m, int n, int k, 
        long long  A, long long B, long long C, 
        const long long row_scale, int lda, int ldb, int ldc);

int testread(long long context, int m, int n, int k, 
        long long A){
    
    int8_t * data_ = reinterpret_cast<int8_t*> (A);
    printf("m=%d\n",m);
    std::vector<int8_t> data(m*n);
    cudaMemcpy(data.data(),data_,sizeof(int8_t)*m*n,cudaMemcpyDeviceToHost);
    for (int i =0; i < 100; ++i){
        int tmp = (int) data[i];
        printf("%d\t",tmp);
    }


}

long long get_context(){ return (long long) new Context(); }
long long get_cublas_handle(){     
    
    cublasHandle_t handle = nullptr;
    (cublasCreate(&handle));
    (cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    return (long long) handle;}

#include <torch/extension.h>
torch::Tensor linear_a8_w8_o32_with_scaling(long long contextinput, int M, int N, int K, 
     torch::Tensor  A_,  torch::Tensor  B_);
// void linear_a8_w8_o32_with_scaling_(long long contextinput, int M, int N, int K, 
//      long long  A_,  long long  B_, long long  C_);
// void linear_a8_w8_o32_with_scaling(long long contextinput, int M, int N, int K, 
//      long long  A_,  long long  B_, long long  C_){
// 	linear_a8_w8_o32_with_scaling_( contextinput,  M,  N,  K, 
//         A_,     B_,    C_);
// }

torch::Tensor gemm(
    const torch::Tensor& mat1,
    const torch::Tensor& mat2, int m, int n, int k) ;


// ind: the target col of mat1, also the target row of mat1
torch::Tensor Mixgemm(
    const torch::Tensor& ind,
    size_t lenid,
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2  //fp16
    );


torch::Tensor MixgemmBase(
    const torch::Tensor& ind,
    size_t lenid,
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2  //fp16
    );

void MixPermuted(
    const torch::Tensor& ind,
    size_t lenid,
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2,  //fp16
    const torch::Tensor& output
    );
torch::Tensor MixAsyncStage4(
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2  //fp16
    );

torch::Tensor MixGemmCutlass(
    const torch::Tensor& ind,
    size_t lenid,
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2,  //int8
    const torch::Tensor& scales,  //fp16
    const torch::Tensor& mixgemmworkspace 
    );


int  get_workspace_size(int m, int k);
torch::Tensor Mixgemmalligned(
    const torch::Tensor& ind,
    size_t lenid,
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2  //fp16
    );
torch::Tensor Mixgemmallignedfp16int8(
    const torch::Tensor& ind,
    size_t lenid,
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2,  //int8
    const torch::Tensor& scaletensor  // float
    );

torch::Tensor  elementWiseMultiply(const torch::Tensor & y, 
        const int m,
        const int n,
        const float x_scale, 
        const torch::Tensor  &scale,
        const torch::Tensor & outliers);

torch::Tensor  elementWiseQuantInt8(torch::Tensor & input, 
        torch::Tensor  &x_scale);

torch::Tensor  elementWiseMultiplyNoOurliers(torch::Tensor & y, 
    int m, int n,
        const float x_scale, 
        torch::Tensor  &scale) ;
void  elementWiseQuantInt8CStyle(torch::Tensor & input, int m, int n, 
        torch::Tensor  &x_scale, torch::Tensor & output);

void  ExtractOutliers( torch::Tensor & input, 
        torch::Tensor & weight, 
        torch::Tensor & activatetion_fp16, 
        torch::Tensor & weight_fp16, 
        torch::Tensor & ind);


torch::Tensor FindOutliers(
    torch::Tensor & ind,
    torch::Tensor & input,
    torch::Tensor & weight,
    int m, int n, int k,
    torch::Tensor & sig, // half
    torch::Tensor & maxvec, // half
    torch::Tensor & input_out,
    torch::Tensor & weight_out    
);
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
);


void gemmColumn(
    const torch::Tensor& mat1, //列优先
    const torch::Tensor& mat2, int m, int n, int k,
    const torch::Tensor& output);

torch::Tensor  elementWiseAdd(const torch::Tensor & y, 
        const int m,
        const int n,
        const torch::Tensor & outliers);






torch::Tensor Int8quantize(const torch::Tensor &src, const torch::Tensor &scale);
torch::Tensor ExtractOutliersAndSetToZeros(
    torch::Tensor & ind,
    torch::Tensor & input

);
torch::Tensor int8FusedDequantize(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &y, int m, int n, int k);


torch::Tensor int8FusedDequantizeSilu(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &y, int m, int n, int k);



torch::Tensor int4FusedDequantize(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &y, int m, int n, int k);
torch::Tensor int4FusedDequantizeSilu(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &scale_row,
                                  const torch::Tensor &scale_col,
                                  const torch::Tensor &y, int m, int n, int k);

torch::Tensor dequantizeInt8(const torch::Tensor &x, const torch::Tensor &scaleRow,
                         const torch::Tensor &scaleCol, const torch::Tensor &y,
                         const int bits, int M, int N) ;


torch::Tensor linear_a8_w8_b32_o32(torch::Tensor &input,  // INT8
                                   torch::Tensor &weight, // INT8
                                   torch::Tensor &cache);
torch::Tensor dequantizeInt8Silu(const torch::Tensor &x, const torch::Tensor &scaleRow,
                         const torch::Tensor &scaleCol, const torch::Tensor &y,
                         const int bits, int M, int N);

torch::Tensor FindRowScale(  const torch::Tensor &x,  torch::Tensor &scaleRow,
                         int rows, int cols, int bit) ;


torch::Tensor packInt4ToFp16(const torch::Tensor & weight, 
                            const torch::Tensor & scale,
                            const torch::Tensor & ind);

torch::Tensor unpack_int4_to_fp16(const torch::Tensor & weight, 
                            const torch::Tensor & ind);



std::vector<torch::Tensor>
 FindRowScaleFusedExtracOutliers(  torch::Tensor &x,  torch::Tensor &scaleRow,
                         const torch::Tensor & ind,  int len_ind,
                         int rows, int cols) ;



torch::Tensor aint4FusedDequantize(
    const torch::Tensor &A, const torch::Tensor &B,
    const torch::Tensor &scale_row, const torch::Tensor &scale_col,
    const float shift_value, const torch::Tensor &zero_row,
    const torch::Tensor &w_reduced, const torch::Tensor &y);
torch::Tensor aint4FusedDequantizeSilu(
    const torch::Tensor &A, const torch::Tensor &B,
    const torch::Tensor &scale_row, const torch::Tensor &scale_col,
    const float shift_value, const torch::Tensor &zero_row,
    const torch::Tensor &w_reduced, const torch::Tensor &y);

void layernorm_forward_cuda(torch::Tensor _input, torch::Tensor _gamma, torch::Tensor _out, float eps);
std::vector<torch::Tensor>  layernorm_forward_cuda_extract_outliers(torch::Tensor &_input, torch::Tensor &_gamma, 
                torch::Tensor &_out, float eps, torch::Tensor &_ind,  torch::Tensor &scaleRow);
std::vector<torch::Tensor>  layernorm_forward_cuda_extract_outliers_int4(torch::Tensor &_input, torch::Tensor &_gamma, 
                torch::Tensor &_out, float eps, torch::Tensor &_ind,  torch::Tensor &scaleRow);


torch::Tensor MixgemmDenseFusedequantSM90(
    const torch::Tensor& mat1, //fp16
    const torch::Tensor& mat2,  //fp16
    const torch::Tensor& scale_a, //fp16
    const torch::Tensor& scale_b,  //fp16
    const torch::Tensor& C // int32
    );

void cutlass_scaled_mm_dq_sm90(torch::Tensor &out, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cigemmlt_ampere_32", &cigemmlt_ampere_32, "cigemmlt_ampere_32");
    m.def("get_context", &get_context, "get_context");
    m.def("testread", &testread, "testread");
    m.def("linear_a8_w8_o32_with_scaling", &linear_a8_w8_o32_with_scaling, "linear_a8_w8_o32_with_scaling");
    m.def("get_cublas_handle", &get_cublas_handle, "get_cublas_handle");
    m.def("gemm", &gemm, "gemm");  // int8 gemm
    m.def("Mixgemm", &Mixgemm, "Mixgemm");
     

    m.def("MixAsyncStage4", &MixAsyncStage4, "MixAsyncStage4");
    m.def("Mixgemmalligned", &Mixgemmalligned, "Mixgemmalligned");
    m.def("Mixgemmallignedfp16int8", &Mixgemmallignedfp16int8, "Mixgemmallignedfp16int8");
    m.def("MixgemmDenseFusedequantSM90", &MixgemmDenseFusedequantSM90, "MixgemmDenseFusedequantSM90");
    m.def("cutlass_scaled_mm_dq_sm90", &cutlass_scaled_mm_dq_sm90, "cutlass_scaled_mm_dq_sm90");
    
    //m.def("MixGemmCutlass", &MixGemmCutlass, "MixGemmCutlass");
    //m.def("get_workspace_size", &get_workspace_size, "get_workspace_size");


    m.def("elementWiseMultiply", &elementWiseMultiply, "elementWiseMultiply");
    m.def("elementWiseMultiplyNoOurliers", &elementWiseMultiplyNoOurliers, "elementWiseMultiplyNoOurliers");
    m.def("elementWiseQuantInt8", &elementWiseQuantInt8, "elementWiseQuantInt8");
    m.def("elementWiseQuantInt8CStyle", &elementWiseQuantInt8CStyle, "elementWiseQuantInt8CStyle");
    m.def("ExtractOutliers", &ExtractOutliers, "ExtractOutliers");
    m.def("FindOutliers", &FindOutliers, "FindOutliers");
    m.def("packInt4ToFp16", &packInt4ToFp16, "packInt4ToFp16");
    
    m.def("FindOutliersRow", &FindOutliersRow, "FindOutliersRow");
    m.def("ExtractOutliersAndSetToZeros", &ExtractOutliersAndSetToZeros, "ExtractOutliersAndSetToZeros");


    m.def("gemmColumn", &gemmColumn, "gemmColumn");
    m.def("elementWiseAdd", &elementWiseAdd, "elementWiseAdd");




    m.def("Int8quantize", &Int8quantize, "Int8quantize");
    m.def("int8FusedDequantize", &int8FusedDequantize,
        "int8FusedDequantize");
    m.def("int4FusedDequantize", &int4FusedDequantize,
        "int4FusedDequantize");
    m.def("aint4FusedDequantize", &aint4FusedDequantize,
        "aint4FusedDequantize");

    m.def("int8FusedDequantizeSilu", &int8FusedDequantizeSilu,
        "int8FusedDequantizeSilu");
    m.def("int4FusedDequantizeSilu", &int4FusedDequantizeSilu,
        "int4FusedDequantizeSilu");      
    m.def("aint4FusedDequantizeSilu", &aint4FusedDequantizeSilu,
        "aint4FusedDequantizeSilu");  
    

    m.def("layernorm_forward_cuda", &layernorm_forward_cuda,
        "layernorm_forward_cuda");
    m.def("layernorm_forward_cuda_extract_outliers", &layernorm_forward_cuda_extract_outliers,
        "layernorm_forward_cuda_extract_outliers");
    m.def("layernorm_forward_cuda_extract_outliers_int4", &layernorm_forward_cuda_extract_outliers_int4,
        "layernorm_forward_cuda_extract_outliers_int4");
        

    m.def("dequantizeInt8", &dequantizeInt8,
        "dequantizeInt8");
    m.def("dequantizeInt8Silu", &dequantizeInt8Silu,
        "dequantizeInt8Silu");
    m.def("unpack_int4_to_fp16", &unpack_int4_to_fp16,
        "unpack_int4_to_fp16");


    m.def("linear_a8_w8_b32_o32", &linear_a8_w8_b32_o32,
        "linear_a8_w8_b32_o32");       

    m.def("FindRowScale", &FindRowScale,
        "FindRowScale");  
    m.def("FindRowScaleFusedExtracOutliers", &FindRowScaleFusedExtracOutliers,
        "FindRowScaleFusedExtracOutliers");  
}