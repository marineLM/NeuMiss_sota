import unittest
from contextlib import nullcontext

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from neumiss.datasets import (CompleteDataset, MARDataset, MCARDataset,
                              MNARDataset)


def check_link(self, link, snr, curvature):
    if link in ['linear'] and snr is None:
        cm = self.assertRaises(ValueError)
    elif link in ['square', 'stairs'] and None in [snr, curvature]:
        cm = self.assertRaises(ValueError)
    else:
        cm = nullcontext()

    return cm


class TestCompleteDataset(unittest.TestCase):

    @given(
        link=st.sampled_from(['linear', 'square', 'stairs', 'logit', 'probit']),
        snr=st.one_of(st.none(), st.floats(min_value=0)),
        curvature=st.one_of(st.none(), st.floats(min_value=0)),
    )
    def test_links(self, link, snr, curvature):
        cm = check_link(self, link, snr, curvature)
        with cm:
            CompleteDataset(n_samples=1, mean=1, cov=1, link=link, beta=[1, 1],
                            X_model='gaussian', snr=snr, curvature=curvature,
                            random_state=0)


class TestMCARDataset(unittest.TestCase):

    @given(
        link=st.sampled_from(['linear', 'square', 'stairs', 'logit', 'probit']),
        snr=st.one_of(st.none(), st.floats(min_value=0)),
        curvature=st.one_of(st.none(), st.floats(min_value=0)),
        missing_rate=st.floats(),
    )
    def test_links(self, link, snr, curvature, missing_rate):
        cm = check_link(self, link, snr, curvature)
        with cm:
            MCARDataset(n_samples=1, mean=1, cov=1, link=link, beta=[1, 1],
                        X_model='gaussian', snr=snr, curvature=curvature,
                        missing_rate=missing_rate, random_state=0)


class TestMARDataset(unittest.TestCase):

    @given(
        link=st.sampled_from(['linear', 'square', 'stairs', 'logit', 'probit']),
        snr=st.one_of(st.none(), st.floats(min_value=0)),
        curvature=st.one_of(st.none(), st.floats(min_value=0)),
        missing_rate=st.floats(),
        p_obs=st.floats(0, 1),
        model=st.sampled_from(['logistic']),
    )
    def test_links(self, link, snr, curvature, missing_rate, p_obs, model):
        cm = check_link(self, link, snr, curvature)
        with cm:
            MARDataset(n_samples=1, mean=[1, 1], cov=np.eye(2), link=link,
                       beta=[1, 1, 1], X_model='gaussian', snr=snr,
                       curvature=curvature, missing_rate=missing_rate,
                       p_obs=p_obs, model=model, random_state=0)


class TestMNARDataset(unittest.TestCase):

    @given(
        link=st.sampled_from(['linear', 'square', 'stairs', 'logit', 'probit']),
        snr=st.one_of(st.none(), st.floats(min_value=0)),
        curvature=st.one_of(st.none(), st.floats(min_value=0)),
        missing_rate=st.floats(),
        p_obs=st.floats(0, 1),
        model=st.sampled_from(['logistic', 'logistic_uniform', 'PSM', 'GSM']),
    )
    def test_links(self, link, snr, curvature, missing_rate, p_obs, model):
        cm = check_link(self, link, snr, curvature)
        p_params = None
        lbd = None
        c = None
        k = None
        sigma2_tilde = None

        if model == 'logistic_uniform':
            p_params = 0.5
        if model == 'PSM':
            lbd = 1
            c = [1, 1]
        if model == 'GSM':
            k = 2
            sigma2_tilde = [1, 1]

        with cm:
            MNARDataset(n_samples=1, mean=[1, 1], cov=np.eye(2), link=link,
                        beta=[1, 1, 1], X_model='gaussian', snr=snr,
                        curvature=curvature, missing_rate=missing_rate,
                        p_params=p_params, lbd=lbd, c=c, k=k,
                        sigma2_tilde=sigma2_tilde, model=model, random_state=0)
