#include <torch/extension.h>

void layernorm_forward_cuda(torch::Tensor _input, torch::Tensor _gamma, torch::Tensor _out, float eps);
std::vector<torch::Tensor>  layernorm_forward_cuda_extract_outliers(torch::Tensor &_input, torch::Tensor &_gamma, 
                torch::Tensor &_out, float eps, torch::Tensor &_ind,  torch::Tensor &scaleRow);


std::vector<torch::Tensor>  layernorm_forward_cuda_extract_outliers_int4(torch::Tensor &_input, torch::Tensor &_gamma, 
                torch::Tensor &_out, float eps, torch::Tensor &_ind,  torch::Tensor &scaleRow);


                