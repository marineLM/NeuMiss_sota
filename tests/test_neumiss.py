import math

import numpy as np
import pytest
import torch
from neumiss.NeuMissBlock import NeuMissBlock, NeuMissMLP
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.utils.data import DataLoader


class NeuMissBlockRef(nn.Module):
    """Reference NeuMissBlock."""

    def __init__(self, n_features, depth):
        super().__init__()
        self.depth = depth
        self.W = nn.Parameter(torch.empty(n_features, n_features, dtype=torch.float))
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


@pytest.mark.parametrize('n_features', [2, 10])
@pytest.mark.parametrize('depth', [1, 3])
def test_neumissblock_old_vs_new(n_features, depth):
    rng = np.random.RandomState(0)
    _W = rng.uniform(size=(n_features, n_features))

    x = rng.normal(size=n_features)
    mask = rng.binomial(1, 0.5, size=n_features)
    np.putmask(x, mask, np.nan)
    x = torch.Tensor(x)

    m = NeuMissBlock(n_features, depth)
    W = nn.Parameter(torch.tensor(_W, dtype=torch.float))
    m.linear.weight = W
    m.mu = nn.Parameter(torch.zeros_like(m.mu))
    y1 = m.forward(x)

    m2 = NeuMissBlockRef(n_features, depth)
    m2.W = nn.Parameter(torch.tensor(_W.T, dtype=torch.float))
    m2.mu = nn.Parameter(torch.zeros_like(m.mu, dtype=torch.float))
    y2 = m2.forward(x)

    assert torch.allclose(y1.double(), y2.double())
    assert torch.allclose(m2.W.T.double(), m.linear.weight.double())


@pytest.mark.parametrize('n_features', [2, 10])
@pytest.mark.parametrize('depth', [1, 3])
def test_neumissblock_float_vs_double(n_features, depth):
    rng = np.random.RandomState(0)
    _W = rng.uniform(size=(n_features, n_features))

    x = rng.normal(size=n_features)
    mask = rng.binomial(1, 0.5, size=n_features)
    np.putmask(x, mask, np.nan)
    x = torch.Tensor(x)

    m = NeuMissBlock(n_features, depth, dtype=torch.float)
    W = nn.Parameter(torch.tensor(_W, dtype=torch.float))
    m.linear.weight = W
    m.mu = nn.Parameter(torch.zeros_like(m.mu))
    y1 = m.forward(x)

    m = NeuMissBlock(n_features, depth, dtype=torch.double)
    W = nn.Parameter(torch.tensor(_W, dtype=torch.double))
    m.linear.weight = W
    m.mu = nn.Parameter(torch.zeros_like(m.mu))
    y2 = m.forward(x)

    assert torch.allclose(y1, y2.float())
    # assert torch.allclose(m2.W.T.double(), m.linear.weight.double())


@pytest.mark.parametrize('n_features', [2, 10])
@pytest.mark.parametrize('depth', [1, 3])
@pytest.mark.parametrize('link', ['linear', 'probit'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('net', ['neumiss_block', 'neumiss_mlp'])
def test_training(n_features, depth, link, dtype, net):
    from datamiss import MCARDataset
    n_epochs = 2

    # Dataset
    n_samples = 1000
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    beta = np.ones(n_features + 1)
    ds = MCARDataset(n_samples, mean, cov, link=link, beta=beta,
                     missing_rate=0.5, snr=10, dtype=dtype)

    # Network
    if net == 'neumiss_block':
        neumiss_block = NeuMissBlock(n_features, depth, dtype=dtype)
        model = nn.Sequential(neumiss_block,
                              nn.Linear(n_features, 1, bias=True, dtype=dtype))

    elif net == 'neumiss_mlp':
        model = NeuMissMLP(n_features, neumiss_depth=depth, mlp_depth=1,
                           dtype=dtype)

    train_loader = DataLoader(ds, batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)

    # TRAIN LOOP
    model.train()
    _loss = binary_cross_entropy_with_logits if ds.is_classif() else mse_loss
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch}')
        for x, y in train_loader:
            print(x.dtype)
            y_hat = torch.squeeze(model(x))
            loss = _loss(y_hat, y)
            print('train loss: ', loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


@pytest.mark.parametrize('n_features', [1, 2, 10])
@pytest.mark.parametrize('neumiss_depth', [0, 1, 2])
@pytest.mark.parametrize('mlp_depth', [0, 1, 2])
def test_neumiss_mlp(n_features, neumiss_depth, mlp_depth):
    rng = np.random.RandomState(0)
    x = rng.normal(size=n_features)
    mask = rng.binomial(1, 0.5, size=n_features)
    np.putmask(x, mask, np.nan)
    x = torch.Tensor(x)

    m = NeuMissMLP(n_features, neumiss_depth, mlp_depth, dtype=torch.float)
    y1 = m.forward(x)

    if mlp_depth == 0:
        assert y1.shape == (n_features,)
    else:
        assert y1.shape == (1,)
