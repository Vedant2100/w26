# %% [markdown]
# # CIFAR10 Data Augmentation Experiments with ResNet18
# This assignment explores the benefits of different data augmentation techniques (Mixup, Cutout, Standard) 
# on the learning performance of a ResNet18 model trained on a subset of CIFAR10.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset

# %% [markdown]
# ## 0. Data Preparation
# Loading CIFAR10 and sampling 10,000 training examples (1,000 per class uniformly at random).
# Normalizing features (pixels) to have zero mean and unit variance for each channel per image.

# %%
dataset = load_dataset("cifar10")
np.random.seed(42)
torch.manual_seed(42)

def preprocess_cifar10(split):
    images = np.array(split["img"]) # (N, 32, 32, 3)
    labels = np.array(split["label"])
    
    # Reshape to (N, 3, 32, 32)
    images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    
    # Normalize per image, per channel: zero mean, unit variance
    # mean: (N, 3, 1, 1), std: (N, 3, 1, 1)
    mean = images.mean(axis=(2, 3), keepdims=True)
    std = images.std(axis=(2, 3), keepdims=True)
    std[std == 0] = 1.0 # avoid division by zero
    images = (images - mean) / std
    
    return images, labels

X_train_full, y_train_full = preprocess_cifar10(dataset["train"])
X_test, y_test = preprocess_cifar10(dataset["test"])

# Sample 1,000 examples per class
train_idxs = []
for i in range(10):
    class_idxs = np.where(y_train_full == i)[0]
    sampled_idxs = np.random.choice(class_idxs, 1000, replace=False)
    train_idxs.append(sampled_idxs)
train_idxs = np.concatenate(train_idxs)
np.random.shuffle(train_idxs)

X_train = X_train_full[train_idxs]
y_train = y_train_full[train_idxs]

print(f"Training set size: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test set size: {X_test.shape}, Labels: {y_test.shape}")

# %% [markdown]
# ## 1. Model and Training Loop Setup
# Using ResNet18 (non-pretrained) as the base model.
# Training for 10 epochs with Adam optimizer and learning rate 0.001.

# %%
def get_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train_one_epoch(model, loader, optimizer, criterion, device, augmentation_fn=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply augmentation if provided
        if augmentation_fn:
            inputs.requires_grad = True # some augmentations might need this if they involve grad computation (not here but good practice)
            inputs, targets = augmentation_fn(inputs, targets)
            
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # If targets are soft (indices/mixup), use a compatible loss or handle it
        if targets.dim() > 1: # Mixup targets
            loss = criterion(outputs, targets)
        else: # Standard targets
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Accuracy is only well-defined for hard labels in this simple loop
    epoch_acc = 100. * correct / total if total > 0 else 0
    return running_loss / len(loader), epoch_acc

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def run_experiment(augmentation_fn=None, epochs=10, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = get_resnet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # For soft targets (mixup), we need soft cross entropy
    def criterion(outputs, targets):
        if targets.dim() > 1: # Soft labels
            return torch.mean(torch.sum(-targets * F.log_softmax(outputs, dim=1), dim=1))
        return F.cross_entropy(outputs, targets)
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)
    
    # We also need a "clean" training loader for reporting training accuracy after the epoch
    # because augmentation might change labels (mixup) or hide data
    report_loader = DataLoader(train_ds, batch_size=100, shuffle=False)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, augmentation_fn)
        train_acc = evaluate(model, report_loader, device)
        test_acc = evaluate(model, test_loader, device)
        
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
    return history

# %% [markdown]
# ## 2. Data Augmentation Implementations

# %%
def mixup_augmentation(inputs, targets, alpha=0.2):
    """
    Mixup: x = lambda * x1 + (1 - lambda) * x2
           y = lambda * y1 + (1 - lambda) * y2
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)

    mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
    
    # Convert targets to one-hot for mixing
    y_a = F.one_hot(targets, 10).float()
    y_b = F.one_hot(targets[index], 10).float()
    mixed_y = lam * y_a + (1 - lam) * y_b
    
    return mixed_x, mixed_y

def cutout_augmentation(inputs, K=16):
    """
    Cutout: With 50% probability, set a square mask of size KxK to zero.
    """
    outputs = inputs.clone()
    batch_size, channels, h, w = outputs.shape
    
    for i in range(batch_size):
        if np.random.rand() > 0.5:
            # Pick a random pixel as the center
            center_h = np.random.randint(0, h)
            center_w = np.random.randint(0, w)
            
            # Mask boundaries
            y1 = np.clip(center_h - K // 2, 0, h)
            y2 = np.clip(center_h + K // 2, 0, h)
            x1 = np.clip(center_w - K // 2, 0, w)
            x2 = np.clip(center_w + K // 2, 0, w)
            
            outputs[i, :, y1:y2, x1:x2] = 0
            
    return outputs

def standard_augmentation(inputs, K=4):
    """
    Standard: Random shifts (up to K) and 50% horizontal flip.
    """
    batch_size, channels, h, w = inputs.shape
    outputs = inputs.clone()
    
    # Apply horizontal flip
    flip = torch.rand(batch_size) > 0.5
    outputs[flip] = torch.flip(outputs[flip], dims=[3])
    
    # Apply random shifts
    # Using padding and cropping for shifts as suggested in the problem
    padded = F.pad(outputs, (K, K, K, K), mode='constant', value=0)
    final_outputs = torch.zeros_like(outputs)
    
    for i in range(batch_size):
        k1 = np.random.randint(0, 2*K + 1) # shift vertical
        k2 = np.random.randint(0, 2*K + 1) # shift horizontal
        final_outputs[i] = padded[i, :, k1:k1+h, k2:k2+w]
        
    return final_outputs

# Helpers for specific experiment calls
def get_mixup_fn(alpha):
    return lambda x, y: mixup_augmentation(x, y, alpha)

def cutout_fn(x, y):
    return cutout_augmentation(x, K=16), y

def standard_fn(x, y):
    return standard_augmentation(x, K=4), y

def combined_fn(alpha):
    def augment(x, y):
        # 1. Standard
        x_aug, y_aug = standard_fn(x, y)
        # 2. Cutout
        x_aug, y_aug = cutout_fn(x_aug, y_aug)
        # 3. Mixup
        return mixup_augmentation(x_aug, y_aug, alpha)
    return augment

# %% [markdown]
# ## 3. Experiments

# %% [markdown]
# ### Task 1: No Augmentation
# %%
print("Task 1: No Augmentation")
history_none = run_experiment(augmentation_fn=None)

# %% [markdown]
# ### Task 2: Mixup Augmentation ($\alpha=0.2, 0.4$)
# %%
print("\nTask 2: Mixup (alpha=0.2)")
history_mixup_02 = run_experiment(augmentation_fn=get_mixup_fn(0.2))

print("\nTask 2: Mixup (alpha=0.4)")
history_mixup_04 = run_experiment(augmentation_fn=get_mixup_fn(0.4))

# %% [markdown]
# ### Task 3: Cutout Augmentation ($K=16$)
# %%
print("\nTask 3: Cutout (K=16)")
history_cutout = run_experiment(augmentation_fn=cutout_fn)

# %% [markdown]
# ### Task 4: Standard Augmentation ($K=4$)
# %%
print("\nTask 4: Standard (K=4)")
history_standard = run_experiment(augmentation_fn=standard_fn)

# %% [markdown]
# ### Task 5: Combined Augmentation
# %%
# Choose alpha with higher test accuracy from Task 2
best_alpha = 0.2 if history_mixup_02['test_acc'][-1] > history_mixup_04['test_acc'][-1] else 0.4
print(f"\nTask 5: Combined (Standard + Cutout + Mixup alpha={best_alpha})")
history_combined = run_experiment(augmentation_fn=combined_fn(best_alpha))

# %% [markdown]
# ## 4. Results and Plots

# %%
def plot_results(histories, titles):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Training Loss
    for i, hist in enumerate(histories):
        axes[0].plot(hist['train_loss'], label=titles[i])
    axes[0].set_title('Training Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Training Accuracy
    for i, hist in enumerate(histories):
        axes[1].plot(hist['train_acc'], label=titles[i])
    axes[1].set_title('Training Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    
    # Test Accuracy
    for i, hist in enumerate(histories):
        axes[2].plot(hist['test_acc'], label=titles[i])
    axes[2].set_title('Test Accuracy over Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

histories = [
    history_none, 
    history_mixup_02, 
    history_mixup_04, 
    history_cutout, 
    history_standard, 
    history_combined
]
titles = [
    'None', 
    'Mixup 0.2', 
    'Mixup 0.4', 
    'Cutout', 
    'Standard', 
    'Combined'
]

plot_results(histories, titles)

# Recording final test accuracy
for title, hist in zip(titles, histories):
    print(f"Final Test Accuracy for {title}: {hist['test_acc'][-1]:.2f}%")
