"""Example of use of Datasets classes."""
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split

from neumiss import CompleteDataset, MARDataset, MCARDataset, MNARDataset

# Size of data
n = 10  # samples
p = 5  # features

# Parameters of Gaussian data
mean = np.ones(p)
cov = np.eye(p)

# Parameters of data generation
beta = np.ones(p + 1)
beta[0] = -5

# Links available for y
ds = CompleteDataset(n, mean, cov, 'linear', beta, snr=10, random_state=0)  # Linear
ds = CompleteDataset(n, mean, cov, 'square', beta, snr=10, curvature=0.5, random_state=0)  # Square
ds = CompleteDataset(n, mean, cov, 'stairs', beta, snr=10, curvature=0.5, random_state=0)  # Stairs
ds = CompleteDataset(n, mean, cov, 'logit', beta, random_state=0)  # Logit
ds = CompleteDataset(n, mean, cov, 'probit', beta, random_state=0)  # Probit

# MCAR dataset
ds = MCARDataset(n, mean, cov, 'logit', beta, missing_rate=0.5, random_state=0)

# MAR dataset
ds = MARDataset(n, mean, cov, 'logit', beta, model='logistic', missing_rate=0.5, p_obs=0.5, random_state=0)  # logistic

# MNAR datasets
ds = MNARDataset(n, mean, cov, 'logit', beta, model='logistic', missing_rate=0.5, random_state=0)  # logistic
ds = MNARDataset(n, mean, cov, 'logit', beta, model='logistic_uniform', missing_rate=0.5, p_params=0.5, random_state=0)  # logistic uniform
ds = MNARDataset(n, mean, cov, 'logit', beta, model='PSM', lbd=1, c=np.ones(p), random_state=0)  # Probit self-masking
ds = MNARDataset(n, mean, cov, 'logit', beta, model='GSM', k=2, sigma2_tilde=np.ones(p), random_state=0)  # Gaussian self-masking

# Iterate over samples of a dataset
print('All samples in dataset:')
for x, y in ds:
    print(f'x={x}, y={y}')

# Subsample a dataset
print('Samples of a subsampled dataset:')
sub_samples_idx = [0, 3, 5, 9]
sub_ds = Subset(ds, sub_samples_idx)
for x, y in sub_ds:
    print(f'x={x}, y={y}')

# Create subsampled datasets
print('Subsampled datasets:')
n_sizes = [3, 5, 9]
for n_size in n_sizes:
    sub_ds = Subset(ds, range(n_size))
    print(f'Subdataset size: {len(sub_ds)}')

# Split a dataset into train, val and test sets
train_ds, val_ds, test_ds = random_split(ds, [5, 3, 2])

# Use dataloaders
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=2, shuffle=True)

# Iterate over dataloaders
for train_features, train_labels in train_loader:
    print(train_features, train_labels)

# Access data parameters of a dataset
print(ds.get_data_params())
