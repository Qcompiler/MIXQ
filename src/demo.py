import torch

from torch import Tensor
def two_compl(x: Tensor, bits: int) -> Tensor:
    return torch.where(x < 0, 2 ** bits + x, x)
def pack_to_i4(X: Tensor):

    X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
    X_i4 = X_i8[:, 0::2] | (X_i8[:, 1::2] << 4)
    return X_i4


B, M = 4, 4
a = torch.randint(-8, 7, (B, M), dtype=torch.int8).cuda()

print(a)
qa = pack_to_i4(a)
print(qa)

import mixlib
n = torch.as_tensor([0,1,3],dtype=torch.int32).cuda()

print(n.shape)
b = mixlib.unpack_int4_to_fp16(qa,n)

print(b)