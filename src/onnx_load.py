import onnx
import torch
 
import onnxruntime

DEVICE_NAME = 'cuda'
DEVICE_INDEX = 0
# 创建一个InferenceSession的实例，并将模型的地址传递给该实例
sess = onnxruntime.InferenceSession('onnxmodel.onnx')
 
def create_session(model: str) -> onnxruntime.InferenceSession:
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    return onnxruntime.InferenceSession(model, providers=providers)
def run_with_torch_tensors_on_device(x: torch.Tensor, y: torch.Tensor, 
                                      torch_type: torch.dtype = torch.float16) -> torch.Tensor:
    session = create_session("test.onnx.pb")

    binding = session.io_binding()

    x_tensor = x.contiguous()
    y_tensor = y.contiguous()

    binding.bind_input(
        name='x',
        device_type=DEVICE_NAME,
        device_id=DEVICE_INDEX,
        element_type=np_type,
        shape=tuple(x_tensor.shape),
        buffer_ptr=x_tensor.data_ptr(),
        )
 

    ## Allocate the PyTorch tensor for the model output
    z_tensor = torch.empty(x_tensor.shape, dtype=torch_type, device=DEVICE).contiguous()
    binding.bind_output(
        name='z',
        device_type=DEVICE_NAME,
        device_id=DEVICE_INDEX,
        element_type=np_type,
        shape=tuple(z_tensor.shape),
        buffer_ptr=z_tensor.data_ptr(),
    )

    session.run_with_iobinding(binding)

    return z_tensor
 
                        
 