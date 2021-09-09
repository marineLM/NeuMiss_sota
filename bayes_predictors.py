"""Bayes predictors for supervised learning with missing values."""
from abc import ABC, abstractmethod
import numpy as np

from datasets import CompleteDataset, MCARDataset, MNARDataset, get_link_function
from amputation import Sigmoid, Probit, Square, Stairs


class BaseBayesPredictor(ABC):
    """Abstract Bayes predictor."""

    def __init__(self,
                 X_model='gaussian',
                 mean=None,
                 cov=None,
                 link=None,
                 beta=None,
                 curvature=None,
                 model=None,
                 k=None,
                 sigma2_tilde=None,
                 lbd=None,
                 c=None,
                 compute_probas=None,
                 **kwargs,
                 ) -> None:
        """
        Parameters
        ----------
        dataset : BaseDataset object
        compute_probas : bool
            Whether to compute probas for classifications or just labels.
        """
        self.X_model = X_model
        self.mean = mean
        self.cov = cov
        self.link = link
        self.beta = beta
        self.curvature = curvature
        self.model = model
        self.k = k
        self.sigma2_tilde = sigma2_tilde
        self.lbd = lbd
        self.c = c
        self.compute_probas = compute_probas
        self._check_attributes()

    def _check_attributes(self):
        if not self.X_model == 'gaussian':
            raise NotImplementedError('Non-Gaussian data not supported.')

        if self.compute_probas and self.is_regression:
            raise ValueError(f'No probability to compute in regression. '
                             f'Got compute_probas={self.compute_probas} '
                             f'but link={self.link}.')

        # available_mv_mechanisms = ['complete', 'MCAR', 'MAR', 'MNAR']
        # if self.mv_mechanism not in available_mv_mechanisms:
        #     raise ValueError(f'Unknown mechanism "{self.mv_mechanism}". '
        #                      f'Supported: {available_mv_mechanisms}.')

        # if self.mv_mechanism == 'MNAR' and self.model != 'GSM':
        #     raise NotImplementedError(
        #         f'Non-GSM mechanism not supported for MNAR. '
        #         f'Got "{self.model}"')

    def fit(self):
        return self

    @abstractmethod
    def predict(self):
        pass

    def is_classif(self):
        return self.link in ['logit', 'probit']

    def is_regression(self):
        return not self.is_classif()


class CompleteBayesPredictor(BaseBayesPredictor):
    """Bayes predictor on complete data."""

    def __init__(self, beta, **kwargs) -> None:
        super().__init__(beta=beta, **kwargs)

    def _check_attributes(self):

        super()._check_attributes()

    def predict(self, X):
        X = np.atleast_2d(X)
        nu = np.inner(np.c_[np.ones(X.shape[0]), X], self.beta)

        if self.is_classif():
            y_pred = nu > 0

            if self.compute_probas:
                link_fn = get_link_function(self.link)
                y_prob = link_fn(nu)
                return y_pred, y_prob

        else:
            link_fn = get_link_function(self.link, curvature=self.curvature)
            y_pred = link_fn(nu)

        return y_pred
