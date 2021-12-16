"""Simulate Gaussian datasets with MCAR, MAR or MNAR missing values."""
import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch import randperm
from torch._utils import _accumulate
from torch.utils.data import TensorDataset

from .amputation import (MCAR, MNAR_GSM, MNAR_PSM, MAR_logistic, MNAR_logistic,
                         MNAR_logistic_uniform, Probit, Sigmoid, Square,
                         Stairs)


class BaseDataset(ABC, TensorDataset):
    """Abstract dataset class."""

    def __init__(self, n_samples, mean, cov, link, beta, curvature=None,
                 snr=None, X_model='gaussian', random_state=None, _data=None):
        """
        Parameters
        ----------
        n_samples : int
        mean : np.array
            Mean of the Gaussian data.
        cov : np.array
            Covariance of the Gaussian data.
        link : str
            The model to use to generate the outcome y.
            'logit' or 'probit' for classifications.
            'linear', 'square' or 'stairs' for regressions.
        beta : np.array
            Parameter to generate the outcome.
        curvature : float
            For square and stairs links.
        snr : float
            Signal to noise ratio for regression.
        X_model : str
            The model to use to generate the data X.
            'gaussian' for Gaussian data.
        random_state : int
        _data : tuple (X, y)
            To set the X and y variables.
        """
        self.n_samples = n_samples
        self.X_model = X_model
        self.mean = np.atleast_1d(mean)
        self.cov = np.atleast_2d(cov)
        self.link = link
        self.beta = np.atleast_1d(beta)
        self.curvature = curvature
        self.snr = snr
        self.random_state = random_state

        # Check attributes
        self._check_attributes()
        self.rng = check_random_state(random_state)

        if _data is None:
            # Generate data
            self.X = self._generate_X()

            # Generate outcome from data
            self.y = self._generate_y(self.X)

            # Generate missing values in the data
            self.M = self._generate_mask()
            np.putmask(self.X, self.M, np.nan)

        else:
            self.X, self.y = _data
            self.M = np.isnan(self.X)
            self.n_samples = self.X.shape[0]

        # Create a TensorDataset
        super().__init__(torch.from_numpy(self.X), torch.from_numpy(self.y))

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

    def _generate_X(self):
        if self.X_model == 'gaussian':
            X = self.rng.multivariate_normal(
                mean=self.mean, cov=self.cov,
                size=self.n_samples, check_valid='raise')

        return X

    def _generate_y(self, X):
        dot_product = X.dot(self.beta[1:]) + self.beta[0]
        link_fn = get_link_function(self.link, curvature=self.curvature)

        if self.is_classif():
            # Y ~ Binomial(link(<X,Beta>))
            probas = link_fn(dot_product)
            y = self.rng.binomial(p=probas, n=1)

        else:
            y = link_fn(dot_product)
            sigma2_noise = np.var(y)/self.snr
            noise = self.rng.normal(
                loc=0, scale=np.sqrt(sigma2_noise), size=self.n_samples)
            y += noise

        return y

    def random_split(self, lengths, random_state):
        """Split dataset in datasets of same class, based on random_split of
        PyTorch."""
        if sum(lengths) != len(self):  # type: ignore[arg-type]
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        generator = torch.Generator().manual_seed(random_state)
        indices = randperm(sum(lengths), generator=generator).tolist()

        def new_dataset(indices):
            X = self.X[indices, :]
            y = self.y[indices]
            params = self.get_data_params()
            params.pop('mv_mechanism', None)
            params['n_samples'] = len(indices)
            return self.__class__(**params, _data=(X, y))

        return [new_dataset(indices[offset - length:offset]) for
                (offset, length) in zip(_accumulate(lengths), lengths)]

    def is_classif(self):
        return self.link in ['logit', 'probit']

    def is_regression(self):
        return not self.is_classif()

    @abstractmethod
    def _generate_mask(self):
        # Function to be implemented by child classes
        pass

    def get_data_params(self):
        """Retrieve the parameters used to generate the data."""
        return {
            'n_samples': self.n_samples,
            'X_model': self.X_model,
            'mean': self.mean,
            'cov': self.cov,
            'link': self.link,
            'beta': self.beta,
            'curvature': self.curvature,
            'snr': self.snr,
            'random_state': self.random_state,
        }

    def _check_equal_params(self, other):
        return (
            self.X_model == other.X_model and
            np.allclose(self.mean, other.mean) and
            np.allclose(self.cov, other.cov) and
            np.allclose(self.beta, other.beta) and
            self.link == other.link and
            self.curvature == other.curvature and
            self.snr == other.snr and
            (self.random_state is None) == (other.random_state is None)
        )

    def __add__(self, other):
        if not self._check_equal_params(other):
            raise ValueError('Adding two datasets that have different parameters')

        X = np.concatenate([self.X, other.X], axis=0)
        y = np.concatenate([self.y, other.y], axis=0)

        return _MixedDataset(
            n_samples=X.shape[0],
            mean=self.mean,
            cov=self.cov,
            link=self.link,
            beta=self.beta,
            curvature=self.curvature,
            snr=self.snr,
            X_model=self.X_model,
            random_state=self.random_state,
            _data=(X, y),
        )


class _MixedDataset(BaseDataset):
    """Dataset with mixed parameters."""

    def _generate_mask(self):
        raise ValueError('Cannot generate mask in a _MixedDataset')

    def get_data_params(self):
        return dict(super().get_data_params(), **{'mv_mechanism': 'mixed'})


class CompleteDataset(BaseDataset):
    """Generate a Dataset without missing values."""

    def _generate_mask(self):
        return np.zeros_like(self.X)

    def get_data_params(self):
        return dict(super().get_data_params(), **{'mv_mechanism': 'complete'})


class MCARDataset(BaseDataset):
    """Generate a dataset with MCAR missing values."""

    def __init__(
        self,
        n_samples,
        mean,
        cov,
        link,
        beta,
        missing_rate,
        curvature=None,
        snr=None,
        X_model='gaussian',
        random_state=None,
        _data=None
    ):
        """
        Parameters
        ----------
        missing_rate : float
            Proportion of missing values to generate for variables which will
            have missing values.
        """
        self.missing_rate = missing_rate
        super().__init__(
            n_samples=n_samples,
            mean=mean,
            cov=cov,
            link=link,
            beta=beta,
            curvature=curvature,
            snr=snr,
            X_model=X_model,
            random_state=random_state,
            _data=_data,
        )

    def _generate_mask(self):
        return MCAR(self.X, self.missing_rate, self.rng)

    def get_data_params(self):
        return dict(**super().get_data_params(), **{
            'mv_mechanism': 'MCAR',
            'missing_rate': self.missing_rate,
        })


class MARDataset(BaseDataset):
    """Generate a dataset with MAR missing values."""

    def __init__(
            self,
            n_samples,
            mean,
            cov,
            link,
            beta,
            missing_rate,
            p_obs,
            model='logistic',
            curvature=None,
            snr=None,
            X_model='gaussian',
            random_state=None,
            _data=None,
    ):
        """
        Parameters
        ----------
        missing_rate : float
            Proportion of missing values to generate for variables which will
            have missing values.
        p_obs : float
            Proportion of variables with *no* missing values that will be used
            for the logistic masking model.
        model : str
            Model to generate the mask. Available: "logistic".
        """
        self.missing_rate = missing_rate
        self.p_obs = p_obs
        self.model = model
        super().__init__(
            n_samples=n_samples,
            mean=mean,
            cov=cov,
            link=link,
            beta=beta,
            curvature=curvature,
            snr=snr,
            X_model=X_model,
            random_state=random_state,
            _data=_data,
        )

    def _check_attributes(self):
        available_models = ['logistic']
        if self.model not in available_models:
            raise ValueError(f'Unknown model "{self.model}". '
                             f'Supported: {available_models}.')
        super()._check_attributes()

    def _generate_mask(self):
        if self.model == 'logistic':
            return MAR_logistic(self.X, self.missing_rate,
                                self.p_obs, self.rng)

    def get_data_params(self):
        return dict(super().get_data_params(), **{
            'mv_mechanism': 'MAR',
            'missing_rate': self.missing_rate,
            'p_obs': self.p_obs,
            'model': self.model,
        })


class MNARDataset(BaseDataset):
    """Generate a dataset with MNAR missing values."""

    def __init__(
            self,
            n_samples,
            mean,
            cov,
            link,
            beta,
            missing_rate=None,
            p_params=None,
            model='logistic',
            k=None,
            sigma2_tilde=None,
            lbd=None,
            c=None,
            curvature=None,
            snr=None,
            X_model='gaussian',
            random_state=None,
            _data=None,
    ):
        """
        Parameters
        ----------
        missing_rate : float
            Proportion of missing values to generate for variables which will
            have missing values.
        p_params : float
            Proportion of variables that will be used for the logistic masking
            model.
        model : str
            Model to generate the mask. Available: "logistic",
            "logistic_uniform", "GSM", "PSM".
        k : float
            Used to compute mu_tilde for GSM.
        sigma2_tilde : float
            Used for GSM.
        lbd : float
            Used for PSM.
        c : float
            Used for PSM.
        """
        self.missing_rate = missing_rate
        self.p_params = p_params
        self.model = model
        self.k = k
        self.sigma2_tilde = sigma2_tilde
        self.lbd = lbd
        self.c = c
        super().__init__(
            n_samples=n_samples,
            mean=mean,
            cov=cov,
            link=link,
            beta=beta,
            curvature=curvature,
            snr=snr,
            X_model=X_model,
            random_state=random_state,
            _data=_data,
        )

    def _check_attributes(self):
        available_models = ['logistic', 'logistic_uniform', 'GSM', 'PSM']
        if self.model not in available_models:
            raise ValueError(f'Unknown model "{self.model}". '
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

    def get_data_params(self):
        return dict(super().get_data_params(), **{
            'mv_mechanism': 'MNAR',
            'missing_rate': self.missing_rate,
            'p_params': self.p_params,
            'model': self.model,
            'k': self.k,
            'sigma2_tilde': self.sigma2_tilde,
            'lbd': self.lbd,
            'c': self.c,
        })


def get_link_function(link, curvature=None):
    # Regression links
    if link == 'linear':
        return lambda x: x

    if link == 'square':
        if curvature is None:
            raise ValueError('curvature is None')
        return Square(curvature)

    if link == 'stairs':
        if curvature is None:
            raise ValueError('curvature is None')
        return Stairs(curvature)

    # Classification links
    if link == 'logit':
        return Sigmoid()

    if link == 'probit':
        return Probit()

    raise ValueError(f'Unknown link "{link}".')
