'''Implements Neumann with the posibility to do batch learning'''

import math
import numpy as np
from sklearn.base import BaseEstimator

import torch.nn as nn
import torch
from torchmetrics import Accuracy, R2Score
from torch.nn.modules.loss import BCELoss
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from pytorchtools import EarlyStopping
import pytorch_lightning as pl


class Neumiss(pl.LightningModule):
    def __init__(self, n_features, mode, depth, residual_connection=False,
                 mlp_depth=0, width_factor=1, init_type='normal',
                 add_mask=False, Sigma_gt=None, mu_gt=None, beta_gt=None,
                 beta0_gt=None, L_gt=None, tmu_gt=None, tsigma_gt=None,
                 coefs=None, optimizer='adam', lr=1e-3, weight_decay=1e-4,
                 classif=False):
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

        self.loss = nn.BCEWithLogitsLoss() if classif else nn.MSELoss()
        self.score = Accuracy() if classif else R2Score()

        self._check_attributes()

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
                if Sigma_gt is None or mu_gt is None or L_gt is None:
                    raise ValueError('With custom' +
                                     'initialisation, Sigma, mu and L' +
                                     'must be specified.')
                Sigma_gt = torch.as_tensor(Sigma_gt, dtype=torch.double)
                W = torch.eye(n_features, dtype=torch.double) - Sigma_gt*2/L_gt
                Wc = Sigma_gt*2/L_gt
                mu = torch.as_tensor(mu_gt, dtype=torch.double)

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
            if Sigma_gt is None or mu_gt is None or L_gt is None:
                raise ValueError('In analytical mode, Sigma , mu and L' +
                                 'must be specified.')
            Sigma_gt = torch.as_tensor(Sigma_gt, dtype=torch.double)
            if 'accelerated' in self.mode:
                self.W = torch.eye(
                    n_features, dtype=torch.double) - Sigma_gt*1/L_gt
                self.Wc = Sigma_gt
            else:
                self.W = torch.eye(
                    n_features, dtype=torch.double) - Sigma_gt*2/L_gt
                self.Wc = Sigma_gt*2/L_gt

            self.mu = torch.as_tensor(mu_gt, dtype=torch.double)

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
            #     if beta_gt is None or beta0_gt is None:
            #         raise ValueError('In analytical mode, beta and beta0' +
            #                          'must be specified.')
            #     self.beta = torch.as_tensor(beta_gt, dtype=torch.double)
            #     self.b = torch.as_tensor(beta0_gt, dtype=torch.double)

            if 'GSM' in self.mode:
                if (tsigma_gt is None or tmu_gt is None):
                    raise ValueError('In GSM analytical mode, tsigma and tmu' +
                                     'must be specified')
                self.tmu = torch.as_tensor(tmu_gt, dtype=torch.double)
                self.tsigma = torch.as_tensor(tsigma_gt, dtype=torch.double)
                self.Sigma = torch.as_tensor(Sigma_gt, dtype=torch.double)
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

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = torch.nan_to_num(x)
        m = torch.isnan(x)
        y_hat = self(x, m)
        loss = self.loss(y_hat, y.double())
        score = self.score(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_score', score)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = torch.nan_to_num(x)
        m = torch.isnan(x)
        y_hat = self(x, m)
        loss = self.loss(y_hat, y.double())
        score = self.score(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_score', score, prog_bar=True)
        return loss
