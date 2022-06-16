import torch
from torch import nn
import math
import pytest
from neumiss.NeuMissBlock import NeuMissBlock, NeuMissBlockBase
import numpy as np

@pytest.mark.parametrize('n_features', [2, 10])
@pytest.mark.parametrize('depth', [1, 3])
def test_neumissblock(n_features, depth):
    # m = NeuMissBlock(n_features, depth)


    rng = np.random.RandomState(0)
    _W = rng.uniform(size=(n_features, n_features))
    # W = nn.Parameter(torch.empty(n_features, n_features, dtype=torch.float))
    # print(W)
    # return

    torch.manual_seed(0)
    # nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    x = np.array(torch.rand(n_features))

    mask = rng.binomial(1, 0.5, size=(n_features))
    np.putmask(x, mask, np.nan)
    print(mask)
    print(x)
    x = torch.Tensor(x)

    m = NeuMissBlock(n_features, depth)
    with torch.no_grad():
        W = nn.Parameter(torch.tensor(_W, dtype=torch.float))
        m.linear.weight = W
        m.mu = nn.Parameter(torch.zeros_like(m.mu))
    y1 = m.forward(x)
    print(y1)

    m3 = NeuMissBlock(n_features, depth, dtype=torch.double)
    with torch.no_grad():
        W = nn.Parameter(torch.tensor(_W, dtype=torch.double))
        m3.linear.weight = W
        m3.mu = nn.Parameter(torch.zeros_like(m.mu))
    y3 = m3.forward(x)
    print(y3)


    m2 = NeuMissBlockBase(n_features, depth)
    m2.W = nn.Parameter(torch.tensor(_W.T, dtype=torch.float))
    # m2.W = nn.Parameter(torch.zeros_like(m2.W, dtype=torch.float))
    m2.mu = nn.Parameter(torch.zeros_like(m.mu, dtype=torch.float))
    y2 = m2.forward(x)
    print(y2)


    assert torch.allclose(y1.double(), y2.double())
    assert torch.allclose(m2.W.T.double(), m.linear.weight.double())

    # a1 = torch.tensor(np.pi, dtype=torch.float)
    # a2 = torch.tensor(np.pi, dtype=torch.double)


    assert torch.allclose(y1, y3.float())

    print(m3)
