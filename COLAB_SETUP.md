# Path2 Colab Setup Guide

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```python
# Install required packages
!pip install -q pybullet==3.2.7 torch scipy matplotlib

# Clone repository (if not already done)
# !git clone https://github.com/xiaoa5/smart_monitoring.git
# %cd smart_monitoring
```

### Step 2: Generate Test Data

```python
# Quick test data generation (50 frames, ~30 seconds)
!python generate_test_data.py
```

**Output:**
```
✓ Test dataset generation complete!

Generated files:
  output/data/cam_0.json
  output/data/cam_1.json
  output/data/cam_2.json
  output/data/cam_3.json
```

### Step 3: Test Phase B (Probabilistic LSTM)

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
    num_epochs=10,  # Quick test: 10 epochs
    batch_size=8
)

# Load dataset
dataset = MultiCameraDataset(
    json_dir='output/data',
    seq_len=config.seq_len,
    max_cameras=config.max_cameras,
    add_noise=True
)

print(f"✓ Loaded {len(dataset)} sequences")

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Create and train model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ProbabilisticLSTMTracker(config).to(device)
trainer = ProbabilisticTrainer(model, config, device)

print(f"\n{'='*60}")
print("Training Probabilistic LSTM")
print(f"{'='*60}")

history = trainer.train(train_loader, val_loader, num_epochs=10)

print(f"\n✓ Phase B training complete!")
```

---

## Full Pipeline (All Phases)

### Option A: Quick Integrated Test

```python
from path2_integrated import IntegratedTracker, IntegratedConfig

# Configure
config = IntegratedConfig(
    data_dir='output/data',
    constraint_type='circle',
    num_epochs=10,  # Quick test
    batch_size=8
)

# Run full pipeline
tracker = IntegratedTracker(config)
results = tracker.run_full_pipeline()

# View results
from IPython.display import Image
Image('output/integrated/scenario_comparison.png')
```

### Option B: Step-by-Step Testing

```python
# 1. Load data
tracker.load_data()
print(f"✓ Loaded {len(tracker.dataset)} sequences")

# 2. Infer constraint
tracker.infer_constraint()
print(f"✓ Inferred {tracker.constraint.__class__.__name__}")

# 3. Train LSTM
history = tracker.train()
print(f"✓ Training complete")

# 4. Evaluate scenarios
results = tracker.evaluate_scenarios()
for scenario, metrics in results.items():
    print(f"{scenario}: MAE={metrics['mae']:.4f}m")

# 5. Visualize
tracker.visualize_results(results)
```

---

## Troubleshooting

### Error: "No camera JSON files found"

**Solution:** Run data generation first:
```python
!python generate_test_data.py
```

### Error: "IndexError: list index out of range"

**Cause:** No data files in `output/data/`

**Solution:** Generate data:
```python
!python generate_test_data.py
```

### Error: "ModuleNotFoundError: No module named 'pybullet'"

**Solution:** Install dependencies:
```python
!pip install pybullet==3.2.7 torch scipy matplotlib
```

---

## Data Generation Options

### Quick Test (Recommended for Colab)
```python
# 50 frames, ~30 seconds
!python generate_test_data.py
```

### Full Dataset (Local/High-Memory Only)
```python
# 500 frames, ~5 minutes, better quality
!python path2_phase1_2_verified.py
```

---

## Expected Results

After training and evaluation, you should see:

```
Scenario Comparison:
  4cam_no_constraint:     MAE=0.02m
  2cam_no_constraint:     MAE=0.05m
  1cam_no_constraint:     MAE=0.15m
  1cam_with_constraint:   MAE=0.03m  ← 80% improvement!
```

**Key insight:** Constraint makes 1-camera competitive with 4-camera!

---

## Visualization

```python
# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_losses'], label='Train')
plt.plot(history['val_losses'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
# Scenario comparison
plt.bar(range(len(results)), [results[s]['mae'] for s in results])
plt.xticks(range(len(results)), results.keys(), rotation=45)
plt.ylabel('MAE (m)')
plt.title('Scenario Comparison')

plt.tight_layout()
plt.show()
```

---

## Phase-by-Phase Testing

### Test Phase A (Data Generation)
```python
!python generate_test_data.py

# Verify
import json
with open('output/data/cam_0.json', 'r') as f:
    data = json.load(f)
print(f"✓ Camera 0: {len(data)} frames")
print(f"✓ Objects: {[obj['name'] for obj in data[0]['objects']]}")
```

### Test Phase B (Probabilistic LSTM)
```python
from path2_probabilistic_lstm import ProbabilisticLSTMTracker, LSTMConfig

config = LSTMConfig()
model = ProbabilisticLSTMTracker(config)
print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward pass
import torch
bbox_seq = torch.randn(1, 10, 4, 4)
camera_ids = torch.zeros(1, 10, 4, dtype=torch.long)
mask = torch.zeros(1, 10, 4, dtype=torch.bool)

mean, logvar = model(bbox_seq, camera_ids, mask)
print(f"✓ Forward pass: mean={mean.shape}, logvar={logvar.shape}")
```

### Test Phase C (Constraints)
```python
from path2_constraints import CircleConstraint, GaussianDistribution
import numpy as np

# Create constraint
circle = CircleConstraint(center=[0, 0, 0.5], radius=1.5)

# Create prior
prior = GaussianDistribution(
    mean=np.array([1.2, 0.8, 0.5]),
    cov=np.diag([0.1, 0.1, 0.05])**2
)

# Apply constraint
posterior = circle.constrain(prior, constraint_std_radial=0.01)

print(f"Prior uncertainty:     {prior.std}")
print(f"Posterior uncertainty: {posterior.std}")
print(f"✓ Reduction: {(1 - posterior.std.prod() / prior.std.prod())*100:.1f}%")
```

### Test Phase D (Integrated)
```python
from path2_integrated import IntegratedTracker, IntegratedConfig

config = IntegratedConfig(num_epochs=5)  # Quick test
tracker = IntegratedTracker(config)
results = tracker.run_full_pipeline()

print(f"✓ Pipeline complete!")
for scenario, metrics in results.items():
    print(f"  {scenario}: {metrics['mae']:.4f}m")
```

---

## Memory Management (for Colab Free Tier)

```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Clear variables
import gc
gc.collect()

# Use smaller batch size
config.batch_size = 4  # Instead of 32

# Use fewer epochs
config.num_epochs = 10  # Instead of 50
```

---

## Saving Results

```python
# Save trained model
torch.save(model.state_dict(), 'lstm_model.pt')

# Save results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Download from Colab
from google.colab import files
files.download('lstm_model.pt')
files.download('results.json')
```

---

## Complete Colab Notebook Template

```python
# ==============================================================================
# Path2 Complete Testing - Colab Template
# ==============================================================================

# 1. SETUP
print("="*60)
print("Step 1: Installing Dependencies")
print("="*60)
!pip install -q pybullet==3.2.7 torch scipy matplotlib

# 2. GENERATE DATA
print("\n" + "="*60)
print("Step 2: Generating Test Data")
print("="*60)
!python generate_test_data.py

# 3. TRAIN & EVALUATE
print("\n" + "="*60)
print("Step 3: Training & Evaluation")
print("="*60)

from path2_integrated import IntegratedTracker, IntegratedConfig

config = IntegratedConfig(
    data_dir='output/data',
    constraint_type='circle',
    num_epochs=10,
    batch_size=8
)

tracker = IntegratedTracker(config)
results = tracker.run_full_pipeline()

# 4. VISUALIZE
print("\n" + "="*60)
print("Step 4: Results")
print("="*60)

for scenario, metrics in results.items():
    print(f"\n{scenario}:")
    print(f"  MAE:  {metrics['mae']:.4f} m")
    print(f"  RMSE: {metrics['rmse']:.4f} m")
    print(f"  Std:  {metrics['avg_std']:.4f} m")
    print(f"  Cal:  {metrics['calibration']:.1f}%")

# Show plot
from IPython.display import Image
display(Image('output/integrated/scenario_comparison.png'))

print("\n✓ All tests complete!")
```

---

## Next Steps

After successful testing:
1. Try full dataset: `!python path2_phase1_2_verified.py`
2. Experiment with different constraints (line, spline)
3. Test with different camera counts
4. Integrate with your own YOLO detector
5. Deploy to production

---

For detailed documentation, see:
- `PATH2_COMPLETE_README.md` - Complete guide
- `PATH2_PHASE_B_README.md` - LSTM details
- `PATH2_PHASE_C_README.md` - Constraint details
- `PATH2_PHASE_D_README.md` - Pipeline details
