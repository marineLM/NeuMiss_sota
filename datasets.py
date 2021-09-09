"""Simulated Gaussian datasets with MCAR, MAR or GSM missing values."""
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset
from sklearn.utils import check_random_state
from scipy.stats import norm


class BaseDataset(ABC, Dataset):
    """Abstract dataset class."""

    def __init__(self, n_samples, mean, cov, link, beta, curvature=None,
                 snr=None, X_model='gaussian', random_state=None) -> None:
        """
        Parameters
        ----------
        n_samples : int
        mean : np.array
        cov : np.array
        link : str
            The model to use to generate the outcome y.
            'logit' or 'probit' for classifications.
            'linear', 'square' or 'stairs' for regressions.
        beta : np.array
        curvature : float
        snr : float
        X_model : str
            The model to use to generate the data X.
            'gaussian' for Gaussian data.
        random_state : int
        """
        super().__init__()
        self.n_samples = n_samples
        self.X_model = X_model
        self.mean = mean
        self.cov = cov
        self.link = link
        self.beta = beta
        self.curvature = curvature
        self.snr = snr

        self.random_state = random_state
        self.rng = check_random_state(random_state)

        # Generate data
        self.X = self._generate_X()

        # Generate outcome from data
        self.y = self._generate_y(self.X)

        # Generate missing values in the data
        self.M = self.generate_mask()
        self.X = np.putmask(self.X, self.M, np.nan)

    def _generate_X(self):
        if self.X_model == 'gaussian':
            if not all(self.mean, self.cov):
                raise ValueError('mean or cov is None.')

            X = self.rng.multivariate_normal(
                mean=self.mean, cov=self.cov,
                size=self.n_samples, check_valid='raise')

        else:
            raise ValueError(f'Unknown X model "{self.X_model}.')

        return X

    def _generate_y(self, X):
        if self.link not in self.available_links():
            raise ValueError(
                f'Unknown link "{self.link}".'
                f'Supported: {self.available_links()}.'
                )

        dot_product = X.dot(self.beta[1:]) + self.beta[0]

        if self.is_classif():
            if self.link == 'logit':
                link_fn = Logit()

            elif self.link == 'probit':
                link_fn = Probit()

            else:
                raise ValueError(f'Unknown link "{self.link}"')

            # Y ~ Binomial(link(<X,Beta>))
            probas = link_fn(dot_product)
            y = self.rng.binomial(p=probas, n=1)

        else:

            if self.link == 'linear':
                y = dot_product

            elif self.link == 'square':
                y = self.curvature*(dot_product-1)**2

            elif self.link == 'stairs':
                y = dot_product - 1
                for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
                    tmp = np.sqrt(np.pi/8)*self.curvature*(dot_product + b)
                    y += a*norm.cdf(tmp)

            else:
                raise ValueError(f'Unknown link "{self.link}"')

            sigma2_noise = np.var(y)/self.snr
            noise = self.rng.normal(
                loc=0, scale=np.sqrt(sigma2_noise), size=self.n_samples)
            y += noise

        return y

    @staticmethod
    def available_links():
        return ['logit', 'probit', 'linear', 'square', 'stairs']

    def is_classif(self):
        return self.link in ['logit', 'probit']

    def is_regression(self):
        return not self.is_classif()

    @abstractmethod
    def _generate_mask(self):
        # Function to be implemented by child classes
        pass

    def __getitem__(self, index):
        """Return a sample and its outcome from the dataset."""
        return self.X[index, :], self.y[index]

    def __len__(self):
        """Return the number of samples of the dataset."""
        return self.X.shape[0]


class Probit():

    def __call__(self, x):
        return norm.cdf(x)


class Logit():

    def __call__(self, x):
        return np.divide(1, 1 + np.exp(-x))
