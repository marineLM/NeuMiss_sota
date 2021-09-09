"""Simulate Gaussian datasets with MCAR, MAR or MNAR missing values."""
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset
from sklearn.utils import check_random_state
from scipy.stats import norm
from scipy.optimize import fsolve


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
            if self.mean is None or self.cov is None:
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
                link_fn = Sigmoid()

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
                if self.curvature is None:
                    raise ValueError('curvature is None.')

                y = self.curvature*(dot_product-1)**2

            elif self.link == 'stairs':
                if self.curvature is None:
                    raise ValueError('curvature is None.')

                y = dot_product - 1
                for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
                    tmp = np.sqrt(np.pi/8)*self.curvature*(dot_product + b)
                    y += a*norm.cdf(tmp)

            else:
                raise ValueError(f'Unknown link "{self.link}"')

            if self.snr is None:
                raise ValueError('snr is None')

            sigma2_noise = np.var(y)/self.snr
            noise = self.rng.normal(
                loc=0, scale=np.sqrt(sigma2_noise), size=self.n_samples)
            y += noise

        return y

    def _check_attributes(self):
        pass

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
        return self.rng.binomial(n=1, p=self.missing_rate, size=self.X.shape)


class MARDataset(BaseDataset):
    """Generate a dataset with MAR missing values."""

    def __init__(self, *args, missing_rate, p_obs, model='logistic',
                 **kwargs) -> None:
        self.missing_rate = missing_rate
        self.p_obs = p_obs
        self.model = model
        super().__init__(*args, **kwargs)

    def _generate_mask_logistic(self):
        n, d = self.X.shape

        # number of variables that will have no missing values
        # (at least one variable)
        d_obs = max(int(self.p_obs * d), 1)
        # number of variables that will have missing values
        d_na = d - d_obs

        # Sample variables that will all be observed, and those with missing values
        idxs_obs = self.rng.choice(d, d_obs, replace=False)
        idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

        # Other variables will have NA proportions that depend on those observed
        # variables, through a logistic model. The parameters of this logistic
        # model are random, and adapted to the scale of each variable.
        # var = np.var(X, axis=0)
        # coeffs = rng.randn(d_obs, d_na)/np.sqrt(var[idxs_obs, None])

        mu = self.X.mean(axis=0)
        cov = (self.X - mu).T.dot(self.X - mu)/n
        cov_obs = cov[np.ix_(idxs_obs, idxs_obs)]
        coeffs = self.rng.randn(d_obs, d_na)
        v = np.array([coeffs[:, j].dot(cov_obs).dot(
            coeffs[:, j]) for j in range(d_na)])
        steepness = self.rng.uniform(low=0.1, high=0.5, size=d_na)
        coeffs /= steepness*np.sqrt(v)

        # Rescale the sigmoid to have a desired amount of missing values
        # ps = sigmoid(X[:, idxs_obs].dot(coeffs) + intercepts)
        # ps /= (ps.mean(0) / p)

        # Move the intercept to have the desired amount of missing values
        intercepts = np.zeros((d_na))
        for j in range(d_na):
            w = coeffs[:, j]

            def f(b):
                s = Sigmoid()(self.X[:, idxs_obs].dot(w) + b) - self.missing_rate
                return s.mean()

            res = fsolve(f, x0=0)
            intercepts[j] = res[0]

        M = np.zeros_like(self.X)
        ps = Sigmoid()(self.X[:, idxs_obs].dot(coeffs) + intercepts)
        M[:, idxs_nas] = self.rng.binomial(n=1, p=ps)

        return M

    def _generate_mask(self):
        if self.model == 'logistic':
            M = self._generate_mask_logistic()

        else:
            raise ValueError(f'Unknown MAR model {self.logistic}')

        return M


class Probit():

    def __call__(self, x):
        return norm.cdf(x)


class Sigmoid():

    def __call__(self, x):
        return np.divide(1, 1 + np.exp(-x))
