import torch
from torch import nn
from torch import Tensor
import math


class Mask(nn.Module):
    __constants__ = ['mask']
    mask: Tensor

    def __init__(self, mask: bool = False):
        super(Mask, self).__init__()
        self.mask = mask

    def forward(self, input: Tensor) -> Tensor:
        return ~self.mask*input


class NeuMissBlockBase(nn.Module):

    def __init__(self, n_features, depth):
        super().__init__()
        self.depth = depth
        self.W = nn.Parameter(torch.empty(n_features, n_features, dtype=torch.double))
        # self.Wc = nn.Parameter(torch.empty(n_features, n_features, dtype=torch.double))
        self.mu = nn.Parameter(torch.empty(n_features, dtype=torch.double))

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
        # self.W = nn.Parameter(torch.empty(n_features, n_features, dtype=torch.double))
        # self.Wc = nn.Parameter(torch.empty(n_features, n_features, dtype=torch.double))
        self.mu = nn.Parameter(torch.empty(n_features, dtype=torch.double))

        self.linear = nn.Linear(n_features, n_features, bias=False)

    def forward(self, x):
        # x = x.double()
        m = torch.isnan(x)
        x = torch.nan_to_num(x)
        h = x - ~m*self.mu
        h_res = x - ~m*self.mu

        # nn.Linear(n_features, n_features, bias=False)

        for _ in range(self.depth):
            print(h.float())
            print(self.linear.weight)
            h = self.linear(h.float())*~m
            # h = torch.matmul(h, self.W)*~m
            h += h_res

        return h

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

# nn.Linear
nn.ReLU
