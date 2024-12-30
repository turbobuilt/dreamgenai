import glob
import json
import os
import torch
from PIL import Image
import shutil

import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from img_one_net_loader import create_loader
from torch.utils.tensorboard import SummaryWriter
import os
import math
import time


# vmap to apply fully connected to each row
def fc_row(row, param_1, bias_1):
    return torch.matmul(param_1, row) + bias_1
fc_row_vmap = torch.vmap(fc_row)
#map batch dimension
def fc_row_batched(x, param_1, bias_1):
    return fc_row_vmap(x, param_1, bias_1)
fc_row_batched_vmap = torch.vmap(fc_row_batched, in_dims=(0, None, None))

# operates fully connected first on sqrt, then on all
class SparseResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.root = int(math.sqrt(dim))
        self.param_1 = nn.Parameter(torch.randn([self.root, self.root*4, self.root]))
        self.bias_1 = nn.Parameter(torch.randn([self.root, self.root*4]))
        self.param_2 = nn.Parameter(torch.randn([self.root, self.root, self.root*4]))
        self.bias_2 = nn.Parameter(torch.randn([self.root, self.root]))
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        shape = x.shape
        r = x
        x = fc_row_batched_vmap(x.reshape(-1, self.root, self.root), self.param_1, self.bias_1)
        x = self.act(x)
        x = fc_row_batched_vmap(x, self.param_2, self.bias_2)
        x = self.act(x)
        x = x.reshape(-1, shape[1])
        x = self.norm(x)
        x = self.act(self.linear(x))
        return x + r
    
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm_1 = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        r = self.norm_1(x)
        x = self.act(self.fc1(x))
        return x + r
    
def benchmark_model(model, input_tensor, iterations=1000):
    start_time = time.time()
    for _ in range(iterations):
        _ = model(input_tensor)
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    dim = 4096
    batch_size = 2
    iterations = 10000

    # Benchmark ResidualBlock
    model = ResidualBlock(dim)
    print("num params residual", sum(p.numel() for p in model.parameters()))
    x = torch.randn(batch_size, dim)
    time_taken = benchmark_model(model, x, iterations)
    print(f"ResidualBlock took {time_taken:.4f} seconds for {iterations} iterations")

    # Benchmark SparseResidualBlock
    model = SparseResidualBlock(dim)
    print("num params sparse", sum(p.numel() for p in model.parameters()))
    x = torch.randn(batch_size, dim)
    time_taken = benchmark_model(model, x, iterations)
    print(f"SparseResidualBlock took {time_taken:.4f} seconds for {iterations} iterations")