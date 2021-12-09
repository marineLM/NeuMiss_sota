import unittest
import numpy as np
from scipy.stats import norm

from neumiss import MARBayesPredictor


class TestMARBayes(unittest.TestCase):

    def test_2D_no_correlation_x1_missing(x):
        beta = np.array([1, 2, 4])
        mean = np.array([1, 2])
        cov = np.array([[1, 0],
                        [0, 2]])
        X_complete = np.array([-1., -2.])

        # First value missing
        M = np.array([1, 0])
        X = np.where(M, np.nan, X_complete)

        predictor = MARBayesPredictor(beta, X_model='gaussian', mean=mean,
                                      cov=cov, link='probit',
                                      compute_probas=True)

        y_pred, y_prob = predictor.predict(X)

        assert y_prob.item() == norm.cdf(-np.sqrt(5))
        assert y_pred.item() == 0

    def test_2D_no_correlation_x2_missing(x):
        beta = np.array([1, 2, 4])
        mean = np.array([1, 2])
        cov = np.array([[1, 0],
                        [0, 2]])
        X_complete = np.array([-1., -2.])

        # Second value missing
        M = np.array([0, 1])
        X = np.where(M, np.nan, X_complete)

        predictor = MARBayesPredictor(beta, X_model='gaussian', mean=mean,
                                      cov=cov, link='probit',
                                      compute_probas=True)

        y_pred, y_prob = predictor.predict(X)

        assert y_prob.item() == norm.cdf(7/np.sqrt(33))
        assert y_pred.item() == 1

    def test_2D_no_correlation_no_missing(x):
        beta = np.array([1, 2, 4])
        mean = np.array([1, 2])
        cov = np.array([[1, 0],
                        [0, 2]])
        X_complete = np.array([-1., -2.])

        # No missing value
        predictor = MARBayesPredictor(beta, X_model='gaussian', mean=mean,
                                      cov=cov, link='probit',
                                      compute_probas=True)

        y_pred, y_prob = predictor.predict(X_complete)

        assert y_prob.item() == norm.cdf(-9)
        assert y_pred.item() == 0

    def test_2D_no_correlation_both_missing(x):
        beta = np.array([1, 2, 4])
        mean = np.array([1, 2])
        cov = np.array([[1, 0],
                        [0, 2]])
        X_complete = np.array([-1., -2.])

        # No missing value
        M = np.array([1, 1])
        X = np.where(M, np.nan, X_complete)
        predictor = MARBayesPredictor(beta, X_model='gaussian', mean=mean,
                                      cov=cov, link='probit',
                                      compute_probas=True)

        y_pred, y_prob = predictor.predict(X)

        assert y_prob.item() == norm.cdf(11/np.sqrt(37))
        assert y_pred.item() == 1

    def test_2D_correlation_x1_missing(x):
        beta = np.array([1, 2, 4])
        mean = np.array([1, 2])
        cov = np.array([[3, 1],
                        [2, 4]])
        X_complete = np.array([-1., -2.])

        # First value missing
        M = np.array([1, 0])
        X = np.where(M, np.nan, X_complete)

        predictor = MARBayesPredictor(beta, X_model='gaussian', mean=mean,
                                      cov=cov, link='probit',
                                      compute_probas=True)

        y_pred, y_prob = predictor.predict(X)

        assert y_prob.item() == norm.cdf(-7/np.sqrt(12))
        assert y_pred.item() == 0

    def test_2D_correlation_x2_missing(x):
        beta = np.array([1, 2, 4])
        mean = np.array([1, 2])
        cov = np.array([[3, 1],
                        [2, 4]])
        X_complete = np.array([-1., -2.])

        # Second value missing
        M = np.array([0, 1])
        X = np.where(M, np.nan, X_complete)

        predictor = MARBayesPredictor(beta, X_model='gaussian', mean=mean,
                                      cov=cov, link='probit',
                                      compute_probas=True)

        y_pred, y_prob = predictor.predict(X)

        assert y_prob.item() == norm.cdf((7 - 16/3)/np.sqrt(1 + 128/3))
        assert y_pred.item() == 1

    def test_2D_correlation_no_missing(x):
        beta = np.array([1, 2, 4])
        mean = np.array([1, 2])
        cov = np.array([[3, 1],
                        [2, 4]])
        X_complete = np.array([-1., -2.])

        # No missing value
        predictor = MARBayesPredictor(beta, X_model='gaussian', mean=mean,
                                      cov=cov, link='probit',
                                      compute_probas=True)

        y_pred, y_prob = predictor.predict(X_complete)

        assert y_prob.item() == norm.cdf(-9)
        assert y_pred.item() == 0

    def test_2D_correlation_both_missing(x):
        beta = np.array([1, 2, 4])
        mean = np.array([1, 2])
        cov = np.array([[3, 1],
                        [2, 4]])
        X_complete = np.array([-1., -2.])

        # No missing value
        M = np.array([1, 1])
        X = np.where(M, np.nan, X_complete)
        predictor = MARBayesPredictor(beta, X_model='gaussian', mean=mean,
                                      cov=cov, link='probit',
                                      compute_probas=True)

        y_pred, y_prob = predictor.predict(X)

        assert y_prob.item() == norm.cdf(11/np.sqrt(101))
        assert y_pred.item() == 1
