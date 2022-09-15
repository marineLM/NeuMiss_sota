import numpy as np
import torch
import torch.optim as optim
from torchmetrics import Accuracy, R2Score
from torch.nn import BCELoss, MSELoss
from pytorchtools import EarlyStopping


def train_model(model, criterion, train_loader, val_loader, optimizer,
                scheduler, early_stopping, n_epochs, lr_threshold,
                device='cpu', trial=None):
    if early_stopping:
        early_stop_obj = EarlyStopping(verbose=1)

    for i_epoch in range(n_epochs):
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            y_hat = model(bx)
            loss = criterion(y_hat, by)
            loss.backward()
            # Take gradient step
            optimizer.step()

        # Evaluate the validation loss
        val_loss = eval_model_loss(model, val_loader, criterion, device=device)
        if trial is not None:
            trial.report(val_loss, step=i_epoch)

        # Check wether to stop optimization
        if early_stopping:
            early_stop_obj(val_loss, model)
            if early_stop_obj.early_stop:
                print("Early stopping")
                break

        lr = optimizer.param_groups[0]['lr']
        if lr < lr_threshold:
            print("Learning rate threshold crossed: stopping")
            break

        scheduler.step(val_loss)

    # load the last checkpoint with the best model
    if early_stopping and early_stop_obj.early_stop:
        model.load_state_dict(early_stop_obj.checkpoint)


def compute_pred(model, data_loader, classif):
    y_pred = [model(x) for x, _ in data_loader]
    y_pred = torch.cat(y_pred, axis=0)
    if classif:
        return torch.sigmoid(y_pred).detach().numpy()
    else:
        return y_pred.detach().numpy()


def compute_preds(model, train_loader, val_loader, test_loader, classif):
    pred = {}
    splits = ['train', 'val', 'test']
    data_loaders = [train_loader, val_loader, test_loader]
    model.eval()
    for split, data_loader in zip(splits, data_loaders):
        pred[split] = compute_pred(model, data_loader, classif)
    model.train()
    return pred


def eval_model_metric(model, data_loader, metric):
    model.eval()
    with torch.no_grad():
        for bx, by in data_loader:
            y_hat = model(bx)
            m = metric(y_hat, by)
    model.train()
    m = metric.compute()
    return m.item()


def eval_model_loss(model, data_loader, criterion, device='cpu'):
    loss = 0
    n_tot = 0
    model = model.to(device)
    with torch.no_grad():
        for bx, by in data_loader:
            bx = bx.to(device)
            by = by.to(device)
            y_hat = model(bx)
            n_batch = bx.size(0)
            loss += criterion(y_hat, by).item()*n_batch
            n_tot += n_batch
    return loss/n_tot


def compute_accuracy(y_scores, y_labels):
    y_scores = torch.from_numpy(np.array(y_scores))
    y_labels = torch.from_numpy(np.array(y_labels))

    metric = {
        'acc': Accuracy()(y_scores, y_labels).item(),
    }

    return metric


def compute_classif_metrics(y_scores, y_labels):
    y_scores = torch.from_numpy(y_scores).float()
    y_labels = torch.from_numpy(y_labels)

    # if y_scores.ndim == 2 and y_scores.shape[1] == 1:
    #     y_scores = y_scores.flatten()

    metrics = {
        'acc': Accuracy()(y_scores, y_labels).item(),
        'bce': BCELoss()(y_scores, y_labels.float()).item()
    }

    return metrics


def compute_regression_metrics(y_scores, y_labels):
    y_scores = torch.from_numpy(np.array(y_scores))
    y_labels = torch.from_numpy(np.array(y_labels))

    # if y_scores.ndim == 2 and y_scores.shape[1] == 1:
    #     y_scores = y_scores.flatten()

    metrics = {
        'r2': R2Score()(y_scores, y_labels).item(),
        'mse': MSELoss()(y_labels, y_scores).item()
    }

    return metrics


def get_optimizer_by_group(model, optim_hyperparams):
    wd = optim_hyperparams.pop('weight_decay')

    group_wd = []
    group_no_wd = []
    for name, param in model.named_parameters():
        if name == 'layers.0.mu':
            group_no_wd.append(param)
        else:
            group_wd.append(param)

    optimizer = optim.Adam(
            [{'params': group_wd, 'weight_decay': wd},
                {'params': group_no_wd, 'weight_decay': 0}],
            **optim_hyperparams
        )

    return optimizer
