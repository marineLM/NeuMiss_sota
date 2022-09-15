import numpy as np
from math import sqrt
import torch
from torch.utils.data import TensorDataset
from sklearn.utils import check_random_state


def get_example_dataset():

    rng = check_random_state(0)

    # Size of data
    n = 10000
    p = 10

    # Parameters of Gaussian data
    B = rng.randn(p, p//2)
    cov = B.dot(B.T) + np.diag(rng.uniform(low=0.01, high=0.1, size=p))

    mean = rng.randn(p)

    # Generate Gaussian data
    X = rng.multivariate_normal(
        mean=mean, cov=cov, size=n, check_valid='raise'
        )

    # Generate y
    beta = np.repeat(1., p + 1)
    var = beta[1:].dot(cov).dot(beta[1:])
    beta[1:] *= 1/sqrt(var)
    y = X.dot(beta[1:]) + beta[0]

    snr = 10
    noise = rng.normal(loc=0, scale=sqrt(np.var(y)/snr), size=n)
    y += noise

    # Add missing values
    missing_rate = 0.5
    ber = rng.rand(n, p)
    mask = ber < missing_rate
    np.putmask(X, mask, np.nan)

    # train/val/test split
    n_train = int(0.8*n)
    n_val = int(0.1*n)
    n_test = int(0.1*n)

    X_train = X[0:n_train]
    y_train = y[0:n_train]
    X_val = X[n_train:(n_train+n_val)]
    y_val = y[n_train:(n_train+n_val)]
    X_test = X[(n_train+n_val):]
    y_test = y[(n_train+n_val):(n_train+n_val+n_test)]

    # Convert to PyTorch TensorDataset
    ds_train = TensorDataset(
                torch.tensor(X_train, dtype=torch.float),
                torch.tensor(y_train, dtype=torch.float)
            )
    ds_val = TensorDataset(
        torch.tensor(X_val, dtype=torch.float),
        torch.tensor(y_val, dtype=torch.float)
    )
    ds_test = TensorDataset(
        torch.tensor(X_test, dtype=torch.float),
        torch.tensor(y_test, dtype=torch.float)
        )

    return ds_train, ds_val, ds_test
