import torch
from torch import nn
import math 
import pytest
from neumiss.NeuMissBlock import NeuMissBlock, NeuMissBlockBase


@pytest.mark.parametrize('n_features', [10])
@pytest.mark.parametrize('depth', [1])
def test_neumissblock(n_features, depth):
    # m = NeuMissBlock(n_features, depth)

    W = nn.Parameter(torch.empty(n_features, n_features, dtype=torch.float))

    torch.manual_seed(0)
    # nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    x = torch.rand(n_features)

    # print(W)
    # print(x)

    # for net in [NeuMissBlock, NeuMissBlockBase]:
    m = NeuMissBlock(n_features, depth)
    # m.linear.weight = W
    torch.nn.init.xavier_uniform(m.linear.weight)
    # m.W = nn.Parameter(torch.zeros_like(m.W))
    # m.mu = nn.Parameter(torch.zeros_like(m.mu))
    # m.reset_parameters()
    # print(x)
    y1 = m.forward(x)
    print(y1)
