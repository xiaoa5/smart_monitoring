# Path2 Phase B: Probabilistic LSTM Implementation

## Overview

This module implements a **probabilistic LSTM** that predicts 3D object positions as **Gaussian distributions** from multi-camera bbox sequences.

## Key Features

### 1. Multi-Camera Input
- Accepts bbox sequences from **1-3+ cameras**
- Handles **missing observations** (occlusions, tracking failures)
- Adds **noise augmentation** to simulate YOLO detection errors

### 2. Probabilistic Output
- Outputs **Gaussian distribution**: `N(μ, Σ)`
  - `μ`: Predicted 3D position `[x, y, z]`
  - `Σ`: Uncertainty (covariance matrix)
- **Heteroscedastic uncertainty**: Learns when to be uncertain
  - More cameras → Lower uncertainty
  - Occlusions → Higher uncertainty
  - Erratic motion → Higher uncertainty

### 3. Architecture

```
Input: Multi-camera bboxes
  ↓
[Multi-Camera Attention Fusion]
  • Camera-specific embeddings
  • Self-attention over cameras
  • Masked attention (handles missing observations)
  ↓
[LSTM Temporal Modeling]
  • 2-layer LSTM
  • Hidden dim: 128
  • Learns motion patterns
  ↓
[Dual Output Heads]
  • mean_head → [x, y, z]
  • logvar_head → [log(σ²_x), log(σ²_y), log(σ²_z)]
  ↓
Output: μ ± σ (Gaussian distribution)
```

### 4. Training

**Loss Function**: Gaussian Negative Log-Likelihood (NLL)

```
L = 0.5 * (log(σ²) + (y - μ)²/σ²)
```

This loss encourages:
1. **Accurate predictions**: Minimize `(y - μ)²`
2. **Calibrated uncertainty**: Penalize overconfident predictions

**Optimizer**: Adam with learning rate decay

## Module Components

### `ProbabilisticLSTMTracker`
Main model class.

**Input**:
- `bbox_seq`: `[batch, seq_len, num_cameras, 4]` - YOLO bboxes
- `camera_ids`: `[batch, seq_len, num_cameras]` - camera indices
- `mask`: `[batch, seq_len, num_cameras]` - True for missing observations

**Output**:
- `mean`: `[batch, seq_len, 3]` - predicted positions
- `logvar`: `[batch, seq_len, 3]` - log-variance (uncertainty)

### `MultiCameraAttention`
Attention-based fusion module.

**Features**:
- Camera-specific embeddings
- Masked self-attention
- Handles variable number of cameras

### `MultiCameraDataset`
Dataset loader for PyBullet data.

**Features**:
- Loads from Phase 1 JSON files
- Creates sequences with sliding window
- Adds noise augmentation
- Simulates missing observations

### `ProbabilisticTrainer`
Training infrastructure.

**Features**:
- Gaussian NLL loss
- Learning rate scheduling
- Gradient clipping
- Validation metrics (MAE, uncertainty)

## Usage

### 1. Training

```python
from path2_probabilistic_lstm import (
    LSTMConfig, ProbabilisticLSTMTracker,
    MultiCameraDataset, ProbabilisticTrainer
)
import torch
from torch.utils.data import DataLoader, random_split

# Configuration
config = LSTMConfig(
    seq_len=10,
    max_cameras=4,
    num_epochs=50,
    batch_size=32
)

# Load dataset
dataset = MultiCameraDataset(
    json_dir='output/data',
    seq_len=config.seq_len,
    max_cameras=config.max_cameras,
    add_noise=True,  # Data augmentation
    noise_std=0.02,
    missing_prob=0.1
)

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
)

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False
)

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ProbabilisticLSTMTracker(config)

# Train
trainer = ProbabilisticTrainer(model, config, device)
history = trainer.train(train_loader, val_loader, config.num_epochs)
```

### 2. Inference

```python
# Get test sample from validation dataset
sample = val_dataset[0]

# Prepare input data (add batch dimension)
bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)
camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
mask = sample['mask'].unsqueeze(0).to(device)
ground_truth = sample['pos_3d_seq'].cpu().numpy()

# Predict distribution
mean, std = model.predict_distribution(
    bbox_seq, camera_ids, mask
)

print(f"Predicted position: {mean[-1]}")
print(f"Uncertainty (std): {std[-1]}")
print(f"Ground truth: {ground_truth[-1]}")

# 95% confidence interval
lower_bound = mean[-1] - 1.96 * std[-1]
upper_bound = mean[-1] + 1.96 * std[-1]
print(f"95% CI: [{lower_bound}, {upper_bound}]")
```

### 3. Uncertainty Analysis

```python
# Compare uncertainty with different camera counts
import torch

# Get a sample
sample = val_dataset[0]
bbox_seq_full = sample['bbox_seq'].unsqueeze(0).to(device)  # [1, seq_len, 4, 4]
camera_ids_full = sample['camera_ids'].unsqueeze(0).to(device)
mask_full = sample['mask'].unsqueeze(0).to(device)

# Scenario 1: All 4 cameras
mean_4, std_4 = model.predict_distribution(
    bbox_seq_full, camera_ids_full, mask_full
)

# Scenario 2: Only 1 camera (mask out cameras 1, 2, 3)
mask_1cam = mask_full.clone()
mask_1cam[:, :, 1:] = True  # Mask cameras 1-3
mean_1, std_1 = model.predict_distribution(
    bbox_seq_full, camera_ids_full, mask_1cam
)

print(f"Uncertainty (4 cameras): {std_4[-1].mean():.4f} m")
print(f"Uncertainty (1 camera):  {std_1[-1].mean():.4f} m")
print(f"Increase ratio: {std_1[-1].mean() / std_4[-1].mean():.2f}x")

# Expected: std_1 > std_4 (more cameras → less uncertainty)
```

## Colab Testing (Phase B)

Upload `path2_probabilistic_lstm.py` to Colab and run the test cells in `Path2_Probabilistic_LSTM_Colab_Test.ipynb`.

### Expected Results

**Test 1: Multi-camera fusion**
- ✓ Model accepts variable number of cameras
- ✓ Handles missing observations with mask

**Test 2: Uncertainty learning**
- ✓ More cameras → Lower uncertainty
- ✓ Training loss decreases
- ✓ Validation MAE < 0.1 meters

**Test 3: Prediction quality**
- ✓ Predicted trajectory follows ground truth
- ✓ Uncertainty increases during occlusions
- ✓ Uncertainty bands contain ground truth

## Configuration

See `LSTMConfig` dataclass for all configurable parameters:

```python
@dataclass
class LSTMConfig:
    # Model architecture
    input_dim: int = 4          # YOLO bbox dimensions
    hidden_dim: int = 128       # LSTM hidden size
    num_layers: int = 2         # LSTM layers
    output_dim: int = 3         # 3D position

    # Attention
    attention_heads: int = 4
    attention_dim: int = 64

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50

    # Data
    seq_len: int = 10           # Sequence length
    max_cameras: int = 4        # Max cameras

    # Uncertainty
    min_logvar: float = -10.0   # Prevents collapse
    max_logvar: float = 10.0    # Prevents explosion
```

## Next Steps

After Phase B is validated in Colab:
1. **Phase C**: Implement constraint-based Bayesian update
2. **Phase D**: Integrate full pipeline

## References

- **Heteroscedastic Uncertainty**: Kendall & Gal (2017) - "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
- **Multi-View Fusion**: Attention mechanisms for multi-camera tracking
- **LSTM for Tracking**: Recurrent networks for temporal modeling
