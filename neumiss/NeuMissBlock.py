import torch
from torch import Tensor, nn
from torch.nn import Linear, Parameter, ReLU, Sequential
from torch.types import _dtype


class Mask(nn.Module):
    """A mask non-linearity."""
    mask: Tensor

    def __init__(self, input: Tensor):
        super(Mask, self).__init__()
        self.mask = torch.isnan(input)

    def forward(self, input: Tensor) -> Tensor:
        return ~self.mask*input


class SkipConnection(nn.Module):
    """A skip connection operation."""
    value: Tensor

    def __init__(self, value: Tensor):
        super(SkipConnection, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return input + self.value


class NeuMissBlock(nn.Module):
    """The NeuMiss block from "What’s a good imputation to predict with
    missing values?" by Marine Le Morvan, Julie Josse, Erwan Scornet,
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
        self.mu = Parameter(torch.empty(n_features, dtype=dtype))
        self.linear = Linear(n_features, n_features, bias=False, dtype=dtype)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = x.type(self.dtype)  # Cast tensor to appropriate dtype
        mask = Mask(x)  # Initialize mask non-linearity
        x = torch.nan_to_num(x)  # Fill missing values with 0
        h = x - mask(self.mu)  # Subtract masked parameter mu
        skip = SkipConnection(h)  # Initialize skip connection with this value

        layer = [self.linear, mask, skip]  # One Neumann iteration
        layers = Sequential(*(layer*self.depth))  # Neumann block

        return layers(h)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mu)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def extra_repr(self) -> str:
        return 'depth={}'.format(self.depth)


class NeuMissMLP(nn.Module):
    """A NeuMiss block followed by a MLP."""

    def __init__(self, n_features: int, neumiss_depth: int, mlp_depth: int,
                 mlp_width: int = None, dtype: _dtype = torch.float) -> None:
        """
        Parameters
        ----------
        n_features : int
            Dimension of inputs.
        neumiss_depth : int
            Number of layers in the NeuMiss block.
        mlp_depth : int
            Number of hidden layers in the MLP.
        mlp_width : int
            Width of the MLP. If None take mlp_width=n_features. Default: None.
        dtype : _dtype
            Pytorch dtype for the parameters. Default: torch.float.

        """
        super().__init__()
        self.n_features = n_features
        self.neumiss_depth = neumiss_depth
        self.mlp_depth = mlp_depth
        self.dtype = dtype
        mlp_width = n_features if mlp_width is None else mlp_width
        self.mlp_width = mlp_width

        if mlp_width != n_features and mlp_depth == 0:
            raise ValueError(f'mlp_width should be equal to n_features when '
                             f'mlp_depth=0. Got mlp_width={mlp_width} != '
                             f'n_features={n_features}.')

        b = int(mlp_depth >= 1)
        self.layers = Sequential(
            NeuMissBlock(n_features, neumiss_depth, dtype),
            *[Linear(n_features, mlp_width, dtype=dtype), ReLU()]*b,
            *[Linear(mlp_width, mlp_width, dtype=dtype), ReLU()]*b*(mlp_depth-1),
            *[Linear(mlp_width, 1, dtype=dtype)],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
