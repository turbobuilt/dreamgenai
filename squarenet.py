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
    def __init__(self, dim, device=None, out_dim=None):
        super().__init__()
        self.dim = dim
        # assert dim**.5 != round(dim**.5), "dim must be a perfect square"
        a_dim, b_dim = two_largest_factors(dim)
        self.a_dim = a_dim
        self.b_dim = b_dim

        # if out_dim == None:
        #     self.a_out_dim = self.a_dim
        # else:
        #     difference = out_dim - dim
        #     b_dim_difference = difference*.5 // a_dim
        #     self.b_dim = b_dim + b_dim_difference
        #     a_dim_difference = difference*.5 // b_dim
        #     self.a_dim_out = a_dim + a_dim_difference

        self.sqrt_dim = int(dim**.5)
        self.linear_1 = nn.Linear(a_dim, b_dim, dtype=torch.bfloat16, device=device)
        self.norm_1 = nn.LayerNorm(b_dim, dtype=torch.bfloat16, device=device)
        self.linear_2 = nn.Linear(b_dim, a_dim, dtype=torch.bfloat16, device=device)
        self.norm_2 = nn.LayerNorm(a_dim, dtype=torch.bfloat16, device=device)
        self.dropout = nn.Dropout(.01)

    def forward(self, src: torch.Tensor):
        shape = src.shape
        src = src.reshape(*shape[:-1], self.b_dim, self.a_dim)
        src = self.linear_1(src)
        src = F.gelu(src)
        src = self.norm_1(src).transpose(-1,-2)
        src = self.dropout(src)
        return self.norm_2(F.gelu(self.linear_2(src))).reshape(shape)
        

from torch import nn
import torch
import math
import torch.nn.functional as F

def two_largest_factors(n):
    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return (i, n // i)
        
print(two_largest_factors(262144+1024))

import torch
import torch.nn as nn

class GroupLinear(nn.Module):
    def __init__(self, num_groups, in_features, out_features, dtype, device):
        super(GroupLinear, self).__init__()
        self.weights = nn.Parameter(torch.empty(num_groups, in_features, out_features, dtype=torch.bfloat16, device=device))
        self.biases = nn.Parameter(torch.empty(num_groups, 1, out_features, dtype=torch.bfloat16, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, x):
        # x shape: (batch_size, num_groups, in_features)
        # weights shape: (num_groups, in_features, out_features)
        # biases shape: (num_groups, 1, out_features)
        # Output should be of shape: (batch_size, num_groups, out_features)
        batch_size, num_groups, _ = x.size()
        x = x.view(batch_size*num_groups, -1)
        weights = self.weights.view(num_groups, -1).t()
        output = torch.addmm(self.biases.view(-1), x, weights)
        output = output.view(batch_size, num_groups, -1)
        return output

class CompactCustomLinearLayer(nn.Module):
    def __init__(self, num_features, output_features, num_items, dtype, device):
        super().__init__()
        self.weights = nn.Parameter(torch.empty((num_items, num_features, output_features), device=device, dtype=dtype).uniform_())
        self.biases = nn.Parameter(torch.zeros(num_items, output_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, x):
        # Assuming x shape: (batch_size, num_items, num_features)
        return torch.matmul(x, self.weights) + self.biases
    
import torch
import torch.nn as nn
import torch.nn.functional as F



def apply_linear_per_group(weight, bias, input):
    # print("weight shape", weight.shape, "bias shape", bias.shape, "input shape", input.shape)
    return F.linear(input, weight, bias)
apply_linear_per_group_vmap = torch.vmap(torch.vmap(apply_linear_per_group, in_dims=(0, 0, 0)), in_dims=(None, None,0))

class GroupLinearVmap(nn.Module):
    def __init__(self, num_groups, in_features, out_features, dtype, device):
        super(GroupLinearVmap, self).__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.randn(num_groups, out_features, in_features, device=device, dtype=dtype))
        self.biases = nn.Parameter(torch.randn(num_groups, out_features, device=device, dtype=dtype))

    def forward(self, x):
        out = apply_linear_per_group_vmap(self.weights, self.biases, x)
        return out

class SquareNetHighMemory(nn.Module):
    def __init__(self, dim, i, device=None):
        super().__init__()
        self.dim = dim
        # assert dim**.5 != round(dim**.5), "dim must be a perfect square"
        a_dim, b_dim = two_largest_factors(dim)
        self.a_dim = a_dim
        self.b_dim = b_dim

        # if out_dim == None:
        #     self.a_out_dim = self.a_dim
        # else:
        #     difference = out_dim - dim
        #     b_dim_difference = difference*.5 // a_dim
        #     self.b_dim = b_dim + b_dim_difference
        #     a_dim_difference = difference*.5 // b_dim
        #     self.a_dim_out = a_dim + a_dim_difference4
        print("initing", i)
        self.sqrt_dim = int(dim**.5)
        self.conv_1 = GroupLinearVmap(a_dim, b_dim, b_dim, dtype=torch.bfloat16, device=device)
        self.norm_1 = nn.GroupNorm(a_dim, a_dim, dtype=torch.bfloat16, device=device)
        self.conv_2 = GroupLinearVmap(b_dim, a_dim, a_dim, dtype=torch.bfloat16, device=device)
        self.norm_2 = nn.GroupNorm(b_dim, b_dim, dtype=torch.bfloat16, device=device)
        self.dropout = nn.Dropout(.01)

    def forward(self, src: torch.Tensor):
        shape = src.shape
        src = src.reshape(*shape[:-1], self.a_dim, self.b_dim)
        src = self.conv_1(src)
        src = F.gelu(src)
        src = self.norm_1(src).transpose(-1,-2)
        src = self.dropout(src)
        return self.norm_2(F.gelu(self.conv_2(src))).reshape(shape)
        

