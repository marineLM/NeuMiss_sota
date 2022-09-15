import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from neumiss import NeuMissMLP
from generate_example_dataset import get_example_dataset
from utils import get_optimizer_by_group, train_model, compute_preds, \
    compute_regression_metrics

# Generate synthetic data
ds_train, ds_val, ds_test = get_example_dataset()

p = ds_train.tensors[0].shape[1]  # n_features

train_loader = DataLoader(ds_train, batch_size=200, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=200)
test_loader = DataLoader(ds_test, batch_size=200)

# Create model
model = NeuMissMLP(n_features=p, neumiss_depth=10, mlp_depth=0, mlp_width=p)

# Initialize optimizer
optim_hyperparams = {'weight_decay': 0, 'lr': 1e-3}
optimizer = get_optimizer_by_group(model, optim_hyperparams)

# Initialize scheduler
sched_hyperparams = {'factor': 0.2, 'patience': 10, 'threshold': 1e-4}
scheduler = ReduceLROnPlateau(optimizer, mode='min', **sched_hyperparams)

criterion = nn.MSELoss()
train_model(model, criterion, train_loader, val_loader, optimizer,
            scheduler, early_stopping=False, n_epochs=500, lr_threshold=1e-6)

# Compute prediction score
train_loader = DataLoader(ds_train, batch_size=200, shuffle=False)
pred = compute_preds(model, train_loader, val_loader, test_loader,
                     classif=False)

res = {}
splits = ['train', 'val', 'test']
preds = [pred[split] for split in splits]
y_labels = [ds_train.tensors[1], ds_val.tensors[1], ds_test.tensors[1]]

for split, pred, y_label in zip(splits, preds, y_labels):
    res_split = compute_regression_metrics(pred, y_label)
    for metric, value in res_split.items():
        res[f'{metric}_{split}'] = value

print(res)
