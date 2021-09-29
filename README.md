# NeuMiss SOTA

## Install
Clone this repository and install it with pip in the environment of your choice (venv, conda...):
```bash
pip install .
```

Import the neumiss package from your code:

```python
import neumiss
```

## Examples

### Generate a dataset

```python
import numpy as np
from neumiss.datasets import MCARDataset, MARDataset, MNARDataset

n = 10  # samples
p = 5  # features

# Parameters of Gaussian data
mean = np.ones(p)
cov = np.eye(p)

# Parameters of data generation
beta = np.ones(p + 1)
beta[0] = -5

ds = MCARDataset(n, mean, cov, 'linear', beta, snr=10, random_state=0, missing_rate=0.5)

print(ds.X)
print(ds.y)
```

### Split a dataset and iteate over it with a DataLoader

```python
from torch.utils.data import DataLoader

# Split a dataset into train, val and test sets
train_ds, val_ds, test_ds = ds.random_split([5, 3, 2], random_state=0)

# Use dataloaders
train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

# Iterate over dataloaders
for train_features, train_labels in train_loader:
    print(train_features, train_labels)

# Access data parameters of a dataset
print(ds.get_data_params())
```

### Predict with the optimal predictors

```python
import torch
from torchmetrics import R2Score
from neumiss import CompleteBayesPredictor, MARBayesPredictor, MNARBayesPredictor

predictor = MARBayesPredictor(**ds.get_data_params())
y_pred = predictor.predict_from_dataset(ds)
y_test = torch.from_numpy(ds.y)
y_pred = torch.from_numpy(y_pred)
acc = R2Score()(y_pred, y_test).item()
print(acc)
```
