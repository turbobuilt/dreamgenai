from torch import nn
import torch
import math
import torch.nn.functional as F

def two_largest_factors(n):
    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return (i, n // i)
        
print(two_largest_factors(262144+1024))

class SquareNet(nn.Module):
    def __init__(self, dim,):
        super().__init__()
        self.dim = dim
        # assert dim**.5 != round(dim**.5), "dim must be a perfect square"
        a_dim, b_dim = two_largest_factors(dim)
        self.a_dim = a_dim
        self.b_dim = b_dim

        self.sqrt_dim = int(dim**.5)
        self.linear_1 = nn.Linear(a_dim, b_dim, dtype=torch.bfloat16)
        self.norm_1 = nn.LayerNorm(b_dim, dtype=torch.bfloat16)
        self.linear_2 = nn.Linear(b_dim, a_dim, dtype=torch.bfloat16)
        self.norm_2 = nn.LayerNorm(a_dim, dtype=torch.bfloat16)
        self.dropout = nn.Dropout(.01)

    def forward(self, src: torch.Tensor):
        shape = src.shape
        src = src.reshape(*shape[:-1], self.b_dim, self.a_dim)
        src = self.linear_1(src)
        src = F.gelu(src)
        src = self.norm_1(src).transpose(-1,-2)
        src = self.dropout(src)
        return self.norm_2(F.gelu(self.linear_2(src))).reshape(shape)
        

