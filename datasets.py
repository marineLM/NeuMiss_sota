"""Simulate Gaussian datasets with MCAR, MAR or MNAR missing values."""
from abc import ABC, abstractmethod
from amputation import (Probit, Sigmoid, MCAR, MAR_logistic, MNAR_logistic,
                        MNAR_logistic_uniform, MNAR_PSM, MNAR_GSM)

import numpy as np
from torch.utils.data import Dataset
from sklearn.utils import check_random_state


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

        # Check attributes
        self._check_attributes()
        self.rng = check_random_state(random_state)

        # Generate data
        self.X = self._generate_X()

        # Generate outcome from data
        self.y = self._generate_y(self.X)

        # Generate missing values in the data
        self.M = self._generate_mask()
        np.putmask(self.X, self.M, np.nan)

    def _generate_X(self):
        if self.X_model == 'gaussian':
            X = self.rng.multivariate_normal(
                mean=self.mean, cov=self.cov,
                size=self.n_samples, check_valid='raise')

        return X

    def _generate_y(self, X):
        dot_product = X.dot(self.beta[1:]) + self.beta[0]

        if self.is_classif():
            if self.link == 'logit':
                link_fn = Sigmoid()

            elif self.link == 'probit':
                link_fn = Probit()

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
                    y += a*Probit()(tmp)

            sigma2_noise = np.var(y)/self.snr
            noise = self.rng.normal(
                loc=0, scale=np.sqrt(sigma2_noise), size=self.n_samples)
            y += noise

        return y

    def _check_attributes(self):
        # Check attributes for the data generation
        if self.X_model not in ['gaussian']:
            raise ValueError(f'Unknown X model "{self.X_model}.')

        if self.X_model == 'gaussian' and self.mean is None:
            raise ValueError('mean is None.')

        if self.X_model == 'gaussian' and self.cov is None:
            raise ValueError('cov is None.')

        # Check attributes for the outcome generation
        available_links = ['logit', 'probit', 'linear', 'square', 'stairs']
        if self.link not in available_links:
            raise ValueError(f'Unknown link "{self.link}". '
                             f'Supported: {available_links}.')

        if self.link in ['square', 'stairs'] and self.curvature is None:
            raise ValueError('curvature is None.')

        if self.is_regression() and self.snr is None:
            raise ValueError('snr is None')

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


class CompleteDataset(BaseDataset):
    """Generate a Dataset without missing values."""

    def _generate_mask(self):
        return np.zeros_like(self.X)


class MCARDataset(BaseDataset):
    """Generate a dataset with MCAR missing values."""

    def __init__(self, *args, missing_rate, **kwargs) -> None:
        self.missing_rate = missing_rate
        super().__init__(*args, **kwargs)

    def _generate_mask(self):
        return MCAR(self.X, self.missing_rate, self.rng)


class MARDataset(BaseDataset):
    """Generate a dataset with MAR missing values."""

    def __init__(self, *args, missing_rate, p_obs, model='logistic',
                 **kwargs) -> None:
        self.missing_rate = missing_rate
        self.p_obs = p_obs
        self.model = model
        super().__init__(*args, **kwargs)

    def _check_attributes(self):
        available_models = ['logistic']
        if self.model not in available_models:
            raise ValueError(f'Unknown link "{self.model}". '
                             f'Supported: {available_models}.')
        super()._check_attributes()

    def _generate_mask(self):
        if self.model == 'logistic':
            return MAR_logistic(self.X, self.missing_rate,
                                self.p_obs, self.rng)


class MNARDataset(BaseDataset):
    """Generate a dataset with MNAR missing values."""

    def __init__(self, *args, missing_rate=None, p_params=None, model='logistic',
                 k=None, sigma2_tilde=None, lbd=None, c=None, **kwargs):
        self.missing_rate = missing_rate
        self.p_params = p_params
        self.model = model
        self.k = k
        self.sigma2_tilde = sigma2_tilde
        self.lbd = lbd
        self.c = c
        super().__init__(*args, **kwargs)

    def _check_attributes(self):
        available_models = ['logistic', 'logistic_uniform', 'GSM', 'PSM']
        if self.model not in available_models:
            raise ValueError(f'Unknown link "{self.model}". '
                             f'Supported: {available_models}.')

        if self.model == 'logistic' and self.missing_rate is None:
            raise ValueError('missing_rate is None.')

        if self.model == 'logistic_uniform':
            if self.missing_rate is None:
                raise ValueError('missing_rate is None.')
            if self.p_params is None:
                raise ValueError('p_params is None.')

        if self.model == 'GSM':
            if self.k is None:
                raise ValueError('k is None.')
            if self.sigma2_tilde is None:
                raise ValueError('sigma2_tilde is None.')
            if self.mean is None:
                raise ValueError('mean is None.')
            if self.cov is None:
                raise ValueError('cov is None.')

        if self.model == 'PSM':
            if self.lbd is None:
                raise ValueError('lbd is None.')
            if self.c is None:
                raise ValueError('c is None.')

        super()._check_attributes()

    def _generate_mask(self):
        if self.model == 'logistic':
            return MNAR_logistic(self.X, self.missing_rate, self.rng)

        elif self.model == 'logistic_uniform':
            return MNAR_logistic_uniform(self.X, self.missing_rate,
                                         self.p_params, self.rng)

        elif self.model == 'GSM':
            mu_tilde = self.mean + self.k*np.sqrt(np.diagonal(self.cov))
            return MNAR_GSM(self.X, mu_tilde, self.sigma2_tilde, self.rng)

        elif self.model == 'PSM':
            return MNAR_PSM(self.X, self.lbd, self.c, self.rng)
