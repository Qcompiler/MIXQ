import torch
import time
import argparse
import numpy as np
import sys
sys.path.append('/home/chenyidong/quant/QUIK/experiments')
from qlinear import MixedQLinear, Linear8bit, Linear4bit
fp_features_num = 128
model_sizes = [   (4096, 10240)]
import sys
sys.path.append('/home/chenyidong/SC/MixQ/src')
from mixquant.modules.linear import MixLinear_GEMM
from mixquant.Cache import MixLibCache

def benchmark(args):
    global model_sizes
    cache = MixLibCache(args.input_size)
    input_size = args.input_size
    for (feature_dim_out,feature_dim_in) in model_sizes:
        for dtype in [torch.float16]:
            x = torch.randn((input_size, feature_dim_in),dtype=dtype).cuda() / 1.3

            def run_benchmark(module):
                num_bench_steps = 100
                for i in range(20):
                    out = module(x)
                start_time = time.perf_counter()
                torch.cuda.synchronize()
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
                for i in range(num_bench_steps):
                    out = module(x)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStop()
                return (end_time - start_time) * 1000 / num_bench_steps
            baseline_mod = torch.nn.Linear(feature_dim_in, feature_dim_out, bias=False).cuda().to(dtype)
            baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data, low=-2, high=1).to(dtype)


            mix_mod =  MixLinear_GEMM.from_linear(baseline_mod,False,False,cache)
            
            fp_indices = torch.randperm(feature_dim_in)[:fp_features_num]
            s_w = torch.ones((feature_dim_out, 1), dtype=dtype, device='cuda')
            int8_mod = MixedQLinear.from_float(baseline_mod,
                                               baseline_mod.weight.data,
                                               s_w, shared_input=None,
                                               fp_indices=fp_indices, bits=8).cuda()
            print(f"{dtype}. Sizes: {baseline_mod.weight.shape}")

            times = []
            for i in range(10):
                times.append(run_benchmark(mix_mod))
            print(f"mix time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms")
            



            times = []
            for i in range(10):
                times.append(run_benchmark(baseline_mod))
            print(f"FP16 time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms")


            times = []
            for i in range(10):
                times.append(run_benchmark(int8_mod))
            print(f"int8 mod time: {np.mean(times):.3f} +- {1.96 * np.std(times):.3f}ms")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-size', type=int,
        help='Size of the input sequence',
        default=512,
    )
    parser.add_argument(
        '--profile', help='Do profile',
        action='store_true',
    )
    args = parser.parse_args()
    benchmark(args)
