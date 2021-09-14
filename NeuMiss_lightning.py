'''Implements Neumann with the posibility to do batch learning'''

import math
import numpy as np
from sklearn.base import BaseEstimator
from abc import ABC

import torch.nn as nn
import torch
from torch.utils.data.sampler import RandomSampler
# from torch.utils.data.dataset import TensorDataset
from torchmetrics import Accuracy, R2Score
from torch.nn.modules.loss import BCELoss
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from pytorchtools import EarlyStopping
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import Tensor
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning import seed_everything


class NeuMiss(pl.LightningModule):
    def __init__(self, n_features, mode, depth, residual_connection=False,
                 mlp_depth=0, width_factor=1, init_type='normal',
                 add_mask=False, Sigma=None, mu=None, beta=None,
                 beta0=None, L=None, tmu=None, tsigma=None,
                 coefs=None, optimizer='adam', lr=1e-3, weight_decay=1e-4,
                 classif=False, random_state=0):
        super().__init__()
        self.n_features = n_features
        self.mode = mode
        self.depth = depth
        self.residual_connection = residual_connection
        self.mlp_depth = mlp_depth
        self.width_factor = width_factor
        self.relu = nn.ReLU()
        self.add_mask = add_mask

        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.classif = classif
        self.random_state = random_state

        self.loss = nn.BCEWithLogitsLoss() if classif else nn.MSELoss()
        self.score = Accuracy() if classif else R2Score()

        self._check_attributes()

        seed_everything(self.random_state, workers=True)

        # NeuMiss block
        # -------------
        if 'analytical' not in self.mode:

            # Create the parameters of the network
            if init_type != 'custom_normal':
                if self.mode == 'baseline':
                    l_W = [torch.empty(n_features, n_features,
                           dtype=torch.double)
                           for _ in range(self.depth)]
                else:
                    W = torch.empty(n_features, n_features, dtype=torch.double)
                Wc = torch.empty(n_features, n_features, dtype=torch.double)
                mu = torch.empty(n_features, dtype=torch.double)
            else:
                if Sigma is None or mu is None or L is None:
                    raise ValueError('With custom' +
                                     'initialisation, Sigma, mu and L' +
                                     'must be specified.')
                Sigma = torch.as_tensor(Sigma, dtype=torch.double)
                W = torch.eye(n_features, dtype=torch.double) - Sigma*2/L
                Wc = Sigma*2/L
                mu = torch.as_tensor(mu, dtype=torch.double)

            if self.mlp_depth > 0:
                beta = torch.empty(width_factor*n_features, dtype=torch.double)
            elif add_mask:
                beta = torch.empty(2*n_features, dtype=torch.double)
            else:
                beta = torch.empty(n_features, dtype=torch.double)
            b = torch.empty(1, dtype=torch.double)
            coefs = torch.ones(self.depth+1, dtype=torch.double)

            # Initialize randomly the parameters of the network
            if init_type == 'normal':
                if self.mode == 'baseline':
                    for W in l_W:
                        nn.init.xavier_normal_(W, gain=0.5)
                else:
                    nn.init.xavier_normal_(W, gain=0.5)
                nn.init.xavier_normal_(Wc)

                tmp = math.sqrt(2/(beta.shape[0]+1))
                nn.init.normal_(beta, mean=0, std=tmp)

            elif init_type == 'uniform':
                if self.mode == 'baseline':
                    for W in l_W:
                        nn.init.xavier_uniform_(W, gain=0.5)
                else:
                    nn.init.xavier_uniform_(W, gain=0.5)
                nn.init.xavier_uniform_(Wc)

                tmp = math.sqrt(2*6/(beta.shape[0]+1))
                nn.init.uniform_(beta, -tmp, tmp)

            elif init_type == 'custom_normal':
                tmp = math.sqrt(2/(beta.shape[0]+1))
                nn.init.normal_(beta, mean=0, std=tmp)

            nn.init.normal_(mu)
            nn.init.zeros_(b)

            # Make tensors learnable parameters
            if self.mode == 'baseline':
                self.l_W = [torch.nn.Parameter(W) for W in l_W]
                for i, W in enumerate(self.l_W):
                    self.register_parameter('W_{}'.format(i), W)
            else:
                self.W = torch.nn.Parameter(W)
            self.Wc = torch.nn.Parameter(Wc)
            self.beta = torch.nn.Parameter(beta)
            self.mu = torch.nn.Parameter(mu)
            self.b = torch.nn.Parameter(b)
            self.coefs = torch.nn.Parameter(coefs)

            if mode != 'shared_accelerated':
                self.coefs.requires_grad = False

        else:
            # If analytical mode, initialize parameters to their ground truth
            # values
            if Sigma is None or mu is None or L is None:
                raise ValueError('In analytical mode, Sigma , mu and L' +
                                 'must be specified.')
            Sigma = torch.as_tensor(Sigma, dtype=torch.double)
            if 'accelerated' in self.mode:
                self.W = torch.eye(
                    n_features, dtype=torch.double) - Sigma*1/L
                self.Wc = Sigma
            else:
                self.W = torch.eye(
                    n_features, dtype=torch.double) - Sigma*2/L
                self.Wc = Sigma*2/L

            self.mu = torch.as_tensor(mu, dtype=torch.double)

            # if self.mlp_depth > 0:
            beta = torch.empty(1*n_features, dtype=torch.double)
            b = torch.empty(1, dtype=torch.double)
            if init_type == 'normal':
                nn.init.normal_(beta)
                nn.init.normal_(b)
            elif init_type == 'uniform':
                bound = 1 / math.sqrt(n_features)
                nn.init.uniform_(beta, -bound, bound)
                nn.init.normal_(b)
            self.beta = torch.nn.Parameter(beta)
            self.b = torch.nn.Parameter(b)
            # else:
            #     if beta is None or beta0 is None:
            #         raise ValueError('In analytical mode, beta and beta0' +
            #                          'must be specified.')
            #     self.beta = torch.as_tensor(beta, dtype=torch.double)
            #     self.b = torch.as_tensor(beta0, dtype=torch.double)

            if 'GSM' in self.mode:
                if (tsigma is None or tmu is None):
                    raise ValueError('In GSM analytical mode, tsigma and tmu' +
                                     'must be specified')
                self.tmu = torch.as_tensor(tmu, dtype=torch.double)
                self.tsigma = torch.as_tensor(tsigma, dtype=torch.double)
                self.Sigma = torch.as_tensor(Sigma, dtype=torch.double)
                # alpha = torch.empty(1, dtype=torch.double) # for analytical GSM
                # nn.init.normal_(alpha)
                # self.alpha = torch.nn.Parameter(alpha)
                self.alpha = torch.ones(1, dtype=torch.double)*0.55

            if 'accelerated' in self.mode:
                self.coefs = torch.as_tensor(coefs, dtype=torch.double)
            else:
                self.coefs = torch.ones(self.depth+1, dtype=torch.double)

        # MLP after the NeuMiss block
        # ---------------------------
        # Create the parameters for the MLP added after the NeuMiss block
        width = width_factor*n_features
        if self.add_mask:
            n_input = 2*n_features
        else:
            n_input = n_features
        l_W_mlp = [torch.empty(n_input, width, dtype=torch.double)]
        for _ in range(mlp_depth - 1):
            l_W_mlp.append(torch.empty(width, width, dtype=torch.double))
        l_b_mlp = [torch.empty(width, dtype=torch.double)
                   for _ in range(mlp_depth)]

        # Initialize randomly the parameters of the MLP
        if init_type in ['normal', 'custom_normal']:
            for W in l_W_mlp:
                nn.init.xavier_normal_(W, gain=math.sqrt(2))

        elif init_type == 'uniform':
            for W in l_W_mlp:
                nn.init.xavier_uniform_(W, gain=math.sqrt(2))

        for b_mlp in l_b_mlp:
            nn.init.zeros_(b_mlp)


        # Make tensors learnable parameters
        self.l_W_mlp = [torch.nn.Parameter(W) for W in l_W_mlp]
        for i, W in enumerate(self.l_W_mlp):
            self.register_parameter('W_mlp_{}'.format(i), W)
        self.l_b_mlp = [torch.nn.Parameter(b) for b in l_b_mlp]
        for i, b in enumerate(self.l_b_mlp):
            self.register_parameter('b_mlp_{}'.format(i), b)

    def _check_attributes(self):
        if self.optimizer not in ['adam', 'sgd']:
            raise ValueError(f'Unknown optimizer "{self.optimizer}."')

    def forward(self, x, m, phase='train'):
        """
        Parameters:
        ----------
        x: tensor, shape (batch_size, n_features)
            The input data imputed by 0.
        m: tensor, shape (batch_size, n_features)
            The missingness indicator (0 if observed and 1 if missing).
        """

        if 'analytical_GSM' in self.mode:
            self.mu_prime = (
                self.mu +
                self.alpha*torch.matmul(self.tmu/self.tsigma, self.Sigma)
                )
            h0 = x + m*self.mu_prime
            h = x - ~m*self.mu_prime
            h_res = x - ~m*self.mu_prime
        else:
            h0 = x + m*self.mu
            h = x - ~m*self.mu
            h_res = x - ~m*self.mu

        h = h*self.coefs[0]

        for i in range(self.depth):
            if self.mode == 'baseline':
                self.W = self.l_W[i]
            h = torch.matmul(h, self.W)*~m
            is_baseline_and_init = (i == 0) and (self.mode == 'baseline')
            if self.residual_connection and not is_baseline_and_init:
                h += h_res*self.coefs[i+1]

        # h = torch.matmul(h, self.Wc)*m + h0
        if self.add_mask:
            h = torch.cat((h, m), 1)

        if self.mlp_depth > 0:
            for W, b in zip(self.l_W_mlp, self.l_b_mlp):
                h = torch.matmul(h, W) + b
                h = self.relu(h)

        y = torch.matmul(h, self.beta)

        y = y + self.b

        return y

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay)

        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                        weight_decay=self.weight_decay)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                      patience=10, threshold=1e-4)

        return {
            'optimizer': optimizer,
            'lr_scheduler':  scheduler,
            'monitor': 'val_loss',
        }

    def _step(self, batch, batch_idx, step_name, prog_bar):
        """Compute loss for one step. For now the step is common to training,
        validation and testing. But each can have separate code in the
        functions below."""
        x, y = batch
        x = torch.nan_to_num(x)
        m = torch.isnan(x)
        y_hat = self(x, m)
        loss = self.loss(y_hat, y.double())
        score = self.score(y_hat, y)
        self.log(f'{step_name}_loss', loss, prog_bar=prog_bar)
        self.log(f'{step_name}_score', score, prog_bar=prog_bar)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch, batch_idx, 'train', prog_bar=False)

    def validation_step(self, val_batch, batch_idx):
        return self._step(val_batch, batch_idx, 'val', prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        return self._step(test_batch, batch_idx, 'test', prog_bar=True)


class BaseNeuMiss(BaseEstimator, NeuMiss):
    """The NeuMiss neural network.
    Use NeuMissRegressor and NeuMissClassifier."""

    def __init__(self, n_features, mode, depth, classif, max_epochs=1000,
                 batch_size=100, early_stopping=True,
                 residual_connection=False,
                 mlp_depth=0, width_factor=1, init_type='normal',
                 add_mask=False, Sigma=None, mu=None, beta=None,
                 beta0=None, L=None, tmu=None, tsigma=None,
                 coefs=None, optimizer='adam', lr=1e-3, weight_decay=1e-4,
                 random_state=None):
        """The NeuMiss neural network.

        Parameters
        ----------
        mode : str
            One of:
            * 'baseline': The weight matrices for the Neumann iteration are not
            shared.
            * 'shared': The weight matrices for the Neumann iteration are shared.
            * 'shared_accelerated': The weight matrices for the Neumann iterations
            are shared and one corefficient per residual connection can be learned
            for acceleration.
            * 'analytical_MAR_accelerated', 'analytical_GSM_accelerated',
            'analytical_MAR', 'analytical_GSM': The weights of the Neumann block
            are set to their ground truth values, only the MLP block is learned.
            The accelerated version uses the values of coefs passed as arguments
            while in the non accelerated version, the coefs are set to 1.
        depth : int
            The number of Neumann iterations.
        n_epochs : int
            The maximum number of epochs.
        batch_size : int
        lr : float
            The learning rate.
        weight_decay : float
            The weight decay parameter.
        early_stopping : boolean
            If True, early stopping is used based on the validaton set, with a
            patience of 15 epochs.
        optimizer : srt
            One of `sgd`or `adam`.
        residual_connection : boolean
            If True, the residual connection of the Neumann network are
            implemented.
        mlp_depth : int
            The depth of the MLP stacked on top of the Neuman iterations.
        width_factor : int
            The width of the MLP stacked on top of the NeuMiss layer is calculated
            as width_factor times n_features.
        init_type : str
            The type of initialization for the parameters. Either 'normal',
            'uniform', or 'custom_normal'. If 'custom_normal', the values provided
            for the parameter `Sigma`, `mu`, `L` (and `coefs` if accelerated) are
            used to initialise the Neumann block.
        add_mask : boolean
            If True, the mask is concatenated to the output of the NeuMiss block.
        verbose : boolean

        """
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        super().__init__(n_features=n_features,
                         mode=mode,
                         depth=depth,
                         residual_connection=residual_connection,
                         mlp_depth=mlp_depth,
                         width_factor=width_factor,
                         init_type=init_type,
                         add_mask=add_mask,
                         Sigma=Sigma,
                         mu=mu,
                         beta=beta,
                         beta0=beta0,
                         L=L,
                         tmu=tmu,
                         tsigma=tsigma,
                         coefs=coefs,
                         optimizer=optimizer,
                         lr=lr,
                         weight_decay=weight_decay,
                         classif=classif,
                         random_state=random_state,
                         )

    def fit(self, X, y, percent_val=0.1):
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return self.fit_from_dataset(dataset, percent_val=percent_val)

    def fit_from_dataset(self, dataset, percent_val=0.1):
        # Seed for reproducibility
        seed_everything(self.random_state, workers=True)

        n_val = int(percent_val*len(dataset))
        n_train = len(dataset) - n_val
        ds_train, ds_val = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(ds_train, batch_size=self.batch_size,
                                  shuffle=True)
        val_loader = DataLoader(ds_val, batch_size=self.batch_size)

        lr_monitor_callback = LearningRateMonitor(logging_interval='step')
        callbacks = [lr_monitor_callback]

        early_stop_callback = EarlyStopping(monitor='val_loss')
        if self.early_stopping:
            callbacks.append(early_stop_callback)

        trainer = pl.Trainer(deterministic=True, max_epochs=self.max_epochs,
                             callbacks=callbacks)
        trainer.fit(self, train_loader, val_loader)

        return self


class NeuMissRegressor(BaseNeuMiss):
    """NeuMiss regressor."""

    def __init__(self, n_features, mode, depth, max_epochs=1000,
                 batch_size=100, early_stopping=True,
                 residual_connection=False,
                 mlp_depth=0, width_factor=1, init_type='normal',
                 add_mask=False, Sigma=None, mu=None, beta=None,
                 beta0=None, L=None, tmu=None, tsigma=None,
                 coefs=None, optimizer='adam', lr=1e-3, weight_decay=1e-4,
                 random_state=0):
        super().__init__(n_features=n_features,
                         mode=mode,
                         depth=depth,
                         max_epochs=max_epochs,
                         batch_size=batch_size,
                         early_stopping=early_stopping,
                         residual_connection=residual_connection,
                         mlp_depth=mlp_depth,
                         width_factor=width_factor,
                         init_type=init_type,
                         add_mask=add_mask,
                         Sigma=Sigma,
                         mu=mu,
                         beta=beta,
                         beta0=beta0,
                         L=L,
                         tmu=tmu,
                         tsigma=tsigma,
                         coefs=coefs,
                         optimizer=optimizer,
                         lr=lr,
                         weight_decay=weight_decay,
                         classif=False,
                         random_state=random_state,
                         )


class NeuMissClassifier(BaseNeuMiss):
    """NeuMiss classifier."""

    def __init__(self, n_features, mode, depth, max_epochs=1000,
                 batch_size=100, early_stopping=True,
                 residual_connection=False,
                 mlp_depth=0, width_factor=1, init_type='normal',
                 add_mask=False, Sigma=None, mu=None, beta=None,
                 beta0=None, L=None, tmu=None, tsigma=None,
                 coefs=None, optimizer='adam', lr=1e-3, weight_decay=1e-4,
                 random_state=0):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        super().__init__(n_features=n_features,
                         mode=mode,
                         depth=depth,
                         residual_connection=residual_connection,
                         mlp_depth=mlp_depth,
                         width_factor=width_factor,
                         init_type=init_type,
                         add_mask=add_mask,
                         Sigma=Sigma,
                         mu=mu,
                         beta=beta,
                         beta0=beta0,
                         L=L,
                         tmu=tmu,
                         tsigma=tsigma,
                         coefs=coefs,
                         optimizer=optimizer,
                         lr=lr,
                         weight_decay=weight_decay,
                         classif=True,
                         random_state=random_state,
                         )

