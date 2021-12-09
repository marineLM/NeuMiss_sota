"""Bayes predictors for supervised learning with missing values."""
import os
import warnings
from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader

from .amputation import Probit
from .datasets import get_link_function


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
                 mv_mechanism=None,
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
        self.mv_mechanism = mv_mechanism
        self._check_attributes()

    def _check_attributes(self):
        if not self.X_model == 'gaussian':
            raise NotImplementedError('Non-Gaussian data not supported.')
        else:
            if self.mean is None:
                raise ValueError('mean is None.')
            if self.cov is None:
                raise ValueError('cov is None.')

        available_links = ['logit', 'probit', 'linear', 'square', 'stairs']
        if self.link not in available_links:
            raise ValueError(f'Unknown link "{self.link}". '
                             f'Supported: {available_links}.')

        if self.compute_probas and self.is_regression():
            raise ValueError(f'No probability to compute in regression. '
                             f'Got compute_probas={self.compute_probas} '
                             f'but link={self.link}.')

    def fit(self, X, y):
        return self

    @abstractmethod
    def predict(self, X):
        pass

    def predict_from_dataset(self, dataset, batch_size=10000):
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=8,
                            multiprocessing_context='fork')
        y_pred = [self.predict(x) for x, _ in loader]
        return np.concatenate(y_pred, axis=0)

    def is_classif(self):
        return self.link in ['logit', 'probit']

    def is_regression(self):
        return not self.is_classif()

    def _unpack_predict_result(self, r):
        if self.compute_probas:  # unzip
            y_pred, y_prob = zip(*r)
            y_pred = np.array(y_pred)
            y_prob = np.array(y_prob)
            return y_pred, y_prob

        y_pred = np.array(r)
        return y_pred


class CompleteBayesPredictor(BaseBayesPredictor):
    """Bayes predictor on complete data."""

    def __init__(self, beta, **kwargs) -> None:
        super().__init__(beta=beta, **kwargs)

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


class MARBayesPredictor(BaseBayesPredictor):
    """Bayes predictor on MAR and MCAR data."""

    def __init__(self, beta, X_model='gaussian', mean=None, cov=None,
                 order0=None, **kwargs):
        self.order0 = order0
        super().__init__(beta=beta, X_model=X_model, mean=mean, cov=cov, **kwargs)

    def _check_attributes(self):
        if self.link == 'logit':
            raise NotImplementedError('No MAR Bayes predictor for logit yet.')
        super()._check_attributes()

    def predict(self, X):
        X = np.atleast_2d(X)

        def predict_one(x):
            mis = np.where(np.isnan(x))[0]
            obs = np.where(~np.isnan(x))[0]

            dot_product = self.beta[0]
            if len(mis) > 0:  # At least one variable missing
                dot_product += self.beta[mis + 1].dot(self.mean[mis])

            if len(obs) > 0:  # At least one variable observed
                dot_product += self.beta[obs + 1].dot(x[obs])

            # At least one variable missing and one observed
            if len(obs) * len(mis) > 0:
                cov_obs = self.cov[np.ix_(obs, obs)]
                cov_obs_inv = np.linalg.inv(cov_obs)
                cov_misobs = self.cov[np.ix_(mis, obs)]

                dot_product += self.beta[mis + 1].dot(cov_misobs).dot(
                    cov_obs_inv).dot(x[obs] - self.mean[obs])

            if self.link == 'linear':
                pred = dot_product

            elif self.link in ['square', 'stairs']:
                link_fn = get_link_function(self.link,
                                            curvature=self.curvature)

                if len(mis) > 0:
                    cov_mismis = self.cov[np.ix_(mis, mis)]
                    cov_mis_obs = cov_mismis

                    if len(obs) > 0:
                        cov_mis_obs -= cov_misobs.dot(
                            cov_obs_inv).dot(cov_misobs.T)

                    var_Tmis = self.beta[mis + 1].dot(
                        cov_mis_obs).dot(self.beta[mis + 1])

                else:
                    var_Tmis = 0

                if self.link == 'square':
                    pred0 = link_fn(dot_product)
                    pred = pred0 + self.curvature*var_Tmis

                elif self.link == 'stairs':
                    pred0 = link_fn(dot_product)
                    pred = dot_product - 1
                    for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
                        pred += a*Probit()((dot_product + b)/np.sqrt(
                            1/(np.pi/8*self.curvature**2) + var_Tmis))

                if self.order0:
                    pred = pred0

            elif self.link == 'probit':
                link_fn = get_link_function(self.link)
                nu = dot_product
                pred = nu > 0

                if self.compute_probas:
                    if len(mis) > 0:  # If at least one variable is missing
                        cov_mismis = self.cov[np.ix_(mis, mis)]
                        cov_misgobs = cov_mismis
                        if len(obs) > 0:  # If at least one variable is observed
                            cov_misgobs = cov_misgobs - cov_misobs.dot(cov_obs_inv).dot(cov_misobs.T)
                        s2 = self.beta[mis + 1].dot(cov_misgobs).dot(self.beta[mis + 1])
                    else:
                        s2 = 0
                    prob = link_fn(nu/np.sqrt(1 + s2))

            if self.compute_probas:
                return pred, prob

            return pred

        r = [predict_one(x) for x in X]
        return self._unpack_predict_result(r)


class MNARBayesPredictor(BaseBayesPredictor):
    """Bayes predictor on MNAR data."""

    def __init__(self, beta, X_model='gaussian', mean=None, cov=None,
                 order0=None, k=None, sigma2_tilde=None, **kwargs):
        self.order0 = order0
        super().__init__(beta=beta, X_model=X_model, mean=mean, cov=cov,
                         k=k, sigma2_tilde=sigma2_tilde, **kwargs)

    def _check_attributes(self):
        super()._check_attributes()
        if self.mv_mechanism != 'MNAR':
            warnings.warn(
                f'You use a MNAR Bayes predictor on {self.mv_mechanism} data.')
        if self.model is None:
            raise ValueError('model is None.')
        if self.link == 'logit':
            raise NotImplementedError('No MNAR Bayes predictor for logit yet.')
        if self.model != 'GSM':
            raise NotImplementedError(
                f'No MNAR Bayes predictor for "{self.model}" model yet.')
        if self.model == 'probit' and self.compute_probas:
            raise NotImplementedError('No probas for probit yet.')

    def predict(self, X):
        X = np.atleast_2d(X)
        mu_tilde = self.mean + self.k*np.sqrt(np.diag(self.cov))

        def predict_one(x):
            mis = np.where(np.isnan(x))[0]
            obs = np.where(~np.isnan(x))[0]

            D_mis_inv = np.diag(1/self.sigma2_tilde[mis])

            cov_misobs = self.cov[np.ix_(mis, obs)]
            cov_obs_inv = np.linalg.inv(self.cov[np.ix_(obs, obs)])
            cov_mis = self.cov[np.ix_(mis, mis)]

            mu_mis_obs = self.mean[mis] + cov_misobs.dot(cov_obs_inv).dot(
                x[obs] - self.mean[obs])
            cov_mis_obs = cov_mis - cov_misobs.dot(cov_obs_inv).dot(
                cov_misobs.T)
            cov_mis_obs_inv = np.linalg.inv(cov_mis_obs)

            S = np.linalg.inv(D_mis_inv + cov_mis_obs_inv)
            s = S.dot(D_mis_inv.dot(mu_tilde[mis]) +
                      cov_mis_obs_inv.dot(mu_mis_obs))

            dot_product = self.beta[0] + self.beta[obs + 1].dot(x[obs]) + \
                self.beta[mis + 1].dot(s)

            if self.link == 'linear':
                pred = dot_product

            elif self.link in ['square', 'stairs']:
                link_fn = get_link_function(self.link,
                                            curvature=self.curvature)
                var_Tmis = self.beta[mis + 1].dot(S).dot(self.beta[mis + 1])

                if self.link == 'square':
                    pred0 = link_fn(dot_product)
                    pred = pred0 + self.curvature*var_Tmis

                elif self.link == 'stairs':
                    pred0 = link_fn(dot_product)
                    pred = dot_product - 1
                    for a, b in zip([2, -4, 2], [-0.8, -1, -1.2]):
                        pred += a*Probit()((dot_product + b)/np.sqrt(
                            1/(np.pi/8*self.curvature**2) + var_Tmis))

                if self.order0:
                    pred = pred0

            elif self.link == 'probit':
                link_fn = get_link_function(self.link)
                nu = dot_product
                pred = nu > 0

                if self.compute_probas:
                    raise NotImplementedError('No proba for probit yet.')

            return pred

        r = [predict_one(x) for x in X]
        return self._unpack_predict_result(r)
