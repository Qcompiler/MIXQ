import numpy as np
import torch
import pandas as pd
a1 = np.genfromtxt("A1.csv")
a1 = torch.as_tensor(a1)
print(a1.abs().max())