import torch



class MixLibCache:
    def __init__(self, inputdim, max_outliers = 12288*3, max_batch=1024,max_weight_dim_N=12288,max_activatetion_dim=12288):
        self.device = 'cuda'

        self.sigma = torch.zeros((1,1),dtype=torch.float16).to('cuda')  
        self.zeros = torch.zeros((512,12288),dtype=torch.float16).to('cuda')    
        self.sigma[0] = 6
        
        self.x_scale = None
        self.ind = None
        self.shape = None
        self.activation_outliers = None

    def do_bench_cudagraph(self, fn):
        if torch.cuda.current_stream() == torch.cuda.default_stream():
            raise RuntimeError("Cannot capture graph in default stream. Please use side stream in benchmark code.")
        # warmup
        for i in range(10):
            fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()


        return g
    


class MLPCache:
    def __init__(self):
        self.device = 'cuda'
        self.x_scale = None
        self.ind = None
        self.shape = None
        self.activation_outliers = None

