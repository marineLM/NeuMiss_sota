import unittest

import numpy as np
from datamiss import CompleteDataset, MCARDataset
from neumiss.NeuMiss_lightning import NeuMissClassifier, NeuMissRegressor


class TestNeuMissClassifier(unittest.TestCase):

    def test_train_complete(self):
        ds = CompleteDataset(n_samples=10, mean=1, cov=1, link='logit',
                             beta=[1, 1], random_state=0)

        ds_train, ds_test = ds.random_split([8, 2], random_state=0)
        neumiss = NeuMissClassifier(1, mode='shared_accelerated', depth=1,
                                    random_state=0, max_epochs=1, mlp_depth=0,
                                    init_type='normal', batch_size=1,
                                    weight_decay=1e-4, lr=0.01,
                                    residual_connection=True,
                                    early_stopping=True, optimizer='adam',
                                    sched_factor=0.2, sched_patience=1,
                                    sched_threshold=1e-4, stopping_lr=1e-5,
                                    logger=False)
        neumiss.fit_from_dataset(ds_train, percent_val=0.2)
        neumiss.test_from_dataset(ds_test)

    def test_train_mcar(self):
        ds = MCARDataset(n_samples=10, mean=1, cov=1, link='logit',
                         beta=[1, 1], random_state=0, missing_rate=0.5)

        ds_train, ds_test = ds.random_split([8, 2], random_state=0)

        mask = 1 - np.isnan(ds_train.X)
        mu_hat = np.nanmean(ds_train.X, axis=0)
        X_train_centered = np.nan_to_num(ds_train.X-mu_hat)
        Sigma_hat = X_train_centered.T.dot(X_train_centered)
        den = mask.T.dot(mask)
        Sigma_hat = Sigma_hat/(den-1)
        L_hat = np.linalg.norm(Sigma_hat, ord=2)

        neumiss = NeuMissClassifier(1, mode='shared_accelerated', depth=1,
                                    random_state=0, max_epochs=1, mlp_depth=0,
                                    init_type='custom_normal', batch_size=1,
                                    weight_decay=1e-4, lr=0.01,
                                    residual_connection=True,
                                    early_stopping=True, optimizer='adam',
                                    sched_factor=0.2, sched_patience=1,
                                    sched_threshold=1e-4, Sigma=Sigma_hat,
                                    mu=mu_hat, L=L_hat, stopping_lr=1e-5,
                                    logger=False)
        neumiss.fit_from_dataset(ds_train, percent_val=0.2)
        neumiss.test_from_dataset(ds_test)


class TestNeuMissRegressor(unittest.TestCase):

    def test_train_complete(self):
        ds = CompleteDataset(n_samples=10, mean=1, cov=1, link='linear',
                             beta=[1, 1], random_state=0, snr=10)

        ds_train, ds_test = ds.random_split([8, 2], random_state=0)
        neumiss = NeuMissRegressor(1, mode='shared_accelerated', depth=1,
                                   random_state=0, max_epochs=1, mlp_depth=0,
                                   init_type='normal', batch_size=2,
                                   weight_decay=1e-4, lr=0.01,
                                   residual_connection=True,
                                   early_stopping=True, optimizer='adam',
                                   sched_factor=0.2, sched_patience=1,
                                   sched_threshold=1e-4, stopping_lr=1e-5,
                                   logger=False)
        neumiss.fit_from_dataset(ds_train, percent_val=0.5)
        neumiss.test_from_dataset(ds_test)

    def test_train_mcar(self):
        ds = MCARDataset(n_samples=10, mean=1, cov=1, link='linear', snr=10,
                         beta=[1, 1], random_state=0, missing_rate=0.5)

        ds_train, ds_test = ds.random_split([8, 2], random_state=0)

        mask = 1 - np.isnan(ds_train.X)
        mu_hat = np.nanmean(ds_train.X, axis=0)
        X_train_centered = np.nan_to_num(ds_train.X-mu_hat)
        Sigma_hat = X_train_centered.T.dot(X_train_centered)
        den = mask.T.dot(mask)
        Sigma_hat = Sigma_hat/(den-1)
        L_hat = np.linalg.norm(Sigma_hat, ord=2)

        neumiss = NeuMissRegressor(1, mode='shared_accelerated', depth=1,
                                   random_state=0, max_epochs=1, mlp_depth=0,
                                   init_type='custom_normal', batch_size=2,
                                   weight_decay=1e-4, lr=0.01,
                                   residual_connection=True,
                                   early_stopping=True, optimizer='adam',
                                   sched_factor=0.2, sched_patience=1,
                                   sched_threshold=1e-4, Sigma=Sigma_hat,
                                   mu=mu_hat, L=L_hat, stopping_lr=1e-5,
                                   logger=False)
        neumiss.fit_from_dataset(ds_train, percent_val=0.5)
        neumiss.test_from_dataset(ds_test)
