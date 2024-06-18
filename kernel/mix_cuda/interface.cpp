
#include "ops.cuh"
#include <pybind11/pybind11.h>



 int igemmlt_ampere_32(cublasLtHandle_t ltHandle, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ 
		return igemmlt<COL_AMPERE, 32, 0>(ltHandle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); 
		
}

int cigemmlt_ampere_32(Context *context, int m, int n, int k, const int8_t *A, const int8_t *B, void *C, float *row_scale, int lda, int ldb, int ldc)
	{ return igemmlt_ampere_32((cublasLtHandle_t) context->m_handle, m, n, k, A, B, C, row_scale, lda, ldb, ldc); }


void linear_a8_w8_o32_with_scaling_(long long contextinput, int M, int N, int K, 
     long long  A_,  long long  B_, long long  C_);
void linear_a8_w8_o32_with_scaling(long long contextinput, int M, int N, int K, 
     long long  A_,  long long  B_, long long  C_){
	linear_a8_w8_o32_with_scaling_( contextinput,  M,  N,  K, 
        A_,     B_,    C_);
}

PYBIND11_MODULE(csrlib, m) {
    m.def("igemmlt_ampere_32", &igemmlt_ampere_32, "igemmlt_ampere_32");
     m.def("linear_a8_w8_o32_with_scaling", &linear_a8_w8_o32_with_scaling, "linear_a8_w8_o32_with_scaling");


}
