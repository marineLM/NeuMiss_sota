import torch
from torch import nn
from torch import Tensor
import math


class Mask(nn.Module):
    __constants__ = ['mask']
    mask: Tensor

    def __init__(self, input: Tensor):
        super(Mask, self).__init__()
        self.mask = torch.isnan(input)

    def forward(self, input: Tensor) -> Tensor:
        return ~self.mask*input


class SkipConnection(nn.Module):
    __constants__ = ['mask']
    value: Tensor

    def __init__(self, value: Tensor):
        super(SkipConnection, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return input + self.value


class NeuMissBlockBase(nn.Module):

    def __init__(self, n_features, depth):
        super().__init__()
        self.depth = depth
        self.W = nn.Parameter(torch.empty(n_features, n_features, dtype=torch.float))
        # self.Wc = nn.Parameter(torch.empty(n_features, n_features, dtype=torch.float))
        self.mu = nn.Parameter(torch.empty(n_features, dtype=torch.float))

    def forward(self, x):
        m = torch.isnan(x)
        x = torch.nan_to_num(x)
        h = x - ~m*self.mu
        h_res = x - ~m*self.mu

        for _ in range(self.depth):
            h = torch.matmul(h, self.W)*~m
            h += h_res

        return h

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))


class NeuMissBlock(nn.Module):

    def __init__(self, n_features, depth):
        super().__init__()
        self.depth = depth
        self.mu = nn.Parameter(torch.empty(n_features, dtype=torch.float))
        self.linear = nn.Linear(n_features, n_features, bias=False)

    def forward(self, x):
        mask = Mask(x)
        x = torch.nan_to_num(x)
        h = x - mask(self.mu)
        skip = SkipConnection(h)

        # One Neumann iteration
        layer = [
            self.linear,
            mask,
            skip,
        ]

        # Neumann block
        layers = nn.Sequential(*(layer*self.depth))
        
        return layers(h)

    # def reset_parameters(self) -> None:
    #     nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
