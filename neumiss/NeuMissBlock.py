import torch
from torch import nn
from torch import Tensor
from torch.types import _dtype


class Mask(nn.Module):
    __constants__ = ['mask']
    mask: Tensor

    def __init__(self, input: Tensor):
        super(Mask, self).__init__()
        self.mask = torch.isnan(input)

    def forward(self, input: Tensor) -> Tensor:
        return ~self.mask*input


class SkipConnection(nn.Module):
    __constants__ = ['value']
    value: Tensor

    def __init__(self, value: Tensor):
        super(SkipConnection, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return input + self.value


class NeuMissBlock(nn.Module):
    """Implement the NeuMiss block from "What’s a good imputation to predict
    with missing values?" by Marine Le Morvan, Julie Josse, Erwan Scornet,
    Gael Varoquaux."""

    def __init__(self, n_features: int, depth: int,
                 dtype: _dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs and outputs of the NeuMiss block.
        depth : int
            Number of layers (Neumann iterations) in the NeuMiss block.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.

        """
        super().__init__()
        self.depth = depth
        self.dtype = dtype
        self.mu = nn.Parameter(torch.empty(n_features, dtype=dtype))
        self.linear = nn.Linear(n_features, n_features, bias=False)
        # self.linear.weight = nn.Parameter(self.linear.weight.type(dtype))
        # self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = x.type(self.dtype)  # Cast tensor to appropriate dtype
        mask = Mask(x)  # Initialize mask non-linearity
        x = torch.nan_to_num(x)  # Fill missing values with 0
        h = x - mask(self.mu)  # Subtract masked parameter mu
        skip = SkipConnection(h)  # Initialize skip connection with this value

        layer = [self.linear, mask, skip]  # One Neumann iteration
        layers = nn.Sequential(*(layer*self.depth))  # Neumann block

        return layers(h)

    def reset_parameters(self) -> None:
        torch.manual_seed(0)
        nn.init.normal_(self.mu)
        # W = self.linear.weight
        torch.manual_seed(0)
        nn.init.normal_(self.linear.weight.float())
        # nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        print('Wb', self.linear.weight)
        self.linear.weight = nn.Parameter(self.linear.weight.type(self.dtype))
        print('Wa', self.linear.weight)

        # print('Wdtyped', self.linear.weight.type(self.dtype))
        # self.mu = nn.Parameter(self.mu.type(self.dtype))
        # self.linear.weight = nn.Parameter(self.linear.weight.type(self.dtype))

    def extra_repr(self) -> str:
        return 'depth={}'.format(self.depth)
