# Path 2 Implementation: Motion Sequence Generator + LSTM Tracker

## 📋 Overview

This implementation covers **Stage 1 and Stage 2** of Path 2 from the future development plan:

- **Stage 1**: Motion Sequence Generator - 连续多目标轨迹生成
- **Stage 2**: LSTM-Based Multi-Object Tracker - LSTM时序跟踪器

## 🎯 Objectives

### Stage 1: Motion Sequence Generator
- ✅ Generate continuous multi-object trajectories
- ✅ Support multiple motion patterns (linear, circular, random walk)
- ✅ Multi-camera synchronized data generation
- ✅ Automatic occlusion estimation
- ✅ Export in standardized JSON format

### Stage 2: LSTM-Based Multi-Object Tracker
- ✅ Temporal prediction of bounding boxes
- ✅ Multi-step sequence prediction
- ✅ Auto-regressive forecasting
- ✅ Training/validation pipeline
- ✅ Model checkpointing

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage 1: Motion Generator                 │
│  ┌───────────┐    ┌──────────┐    ┌─────────────────┐      │
│  │ PyBullet  │───▶│ Multiple │───▶│  Multi-Camera   │      │
│  │   Scene   │    │ Objects  │    │   Rendering     │      │
│  └───────────┘    └──────────┘    └─────────────────┘      │
│                           │                                   │
│                           ▼                                   │
│                  ┌─────────────────┐                         │
│                  │ Motion Sequence │                         │
│                  │  JSON Export    │                         │
│                  └─────────────────┘                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Stage 2: LSTM Tracker                     │
│  ┌───────────┐    ┌──────────┐    ┌─────────────────┐      │
│  │  Dataset  │───▶│   LSTM   │───▶│   Prediction    │      │
│  │ Generator │    │  Network │    │  (Multi-step)   │      │
│  └───────────┘    └──────────┘    └─────────────────┘      │
│                           │                                   │
│                           ▼                                   │
│                  ┌─────────────────┐                         │
│                  │ Model Checkpoint│                         │
│                  │   (.pth file)   │                         │
│                  └─────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Install dependencies
pip install pybullet numpy torch opencv-python matplotlib pyyaml --break-system-packages

# Or use requirements.txt
pip install -r requirements.txt --break-system-packages
```

## 🚀 Quick Start

### Run Complete Pipeline

```bash
python path2_stage1_2_implementation.py
```

This will:
1. Generate motion sequences with 3 objects (10 seconds @ 30fps)
2. Export data to `./path2_output/stage1/motion_sequence.json`
3. Train LSTM tracker on generated sequences
4. Save best model to `./path2_output/stage2/best_lstm_tracker.pth`

### Custom Usage

#### Stage 1: Generate Custom Motion Sequences

```python
from path2_stage1_2_implementation import MotionSequenceGenerator, MotionType

# Create generator
generator = MotionSequenceGenerator(
    scene_size=(10.0, 10.0),
    num_cameras=4,
    fps=30,
    output_dir="./my_sequences"
)

# Add objects with different motion patterns
generator.add_object(
    obj_id=1,
    start_pos=[1.0, 1.0, 0.5],
    motion_type=MotionType.LINEAR,
    velocity=[0.5, 0.3, 0.0]
)

generator.add_object(
    obj_id=2,
    start_pos=[5.0, 5.0, 0.5],
    motion_type=MotionType.CIRCULAR,
    center=[5.0, 5.0, 0.5],
    radius=2.0,
    angular_velocity=0.5
)

# Generate sequences
frame_data = generator.generate_sequence(
    duration=20.0,
    save_json=True
)

generator.cleanup()
```

#### Stage 2: Train LSTM Tracker

```python
from path2_stage1_2_implementation import (
    LSTMTracker, 
    LSTMTrackerTrainer, 
    TrackingDataset
)
from torch.utils.data import DataLoader

# Create dataset
dataset = TrackingDataset(
    json_file="./my_sequences/motion_sequence.json",
    sequence_length=10,
    prediction_horizon=5
)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Create and train model
model = LSTMTracker(
    input_size=4,
    hidden_size=128,
    num_layers=2,
    output_size=4
)

trainer = LSTMTrackerTrainer(model=model)
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    save_path="./my_model.pth"
)
```

## 📊 Data Format

### Motion Sequence JSON Format

```json
{
  "frame": 120,
  "timestamp": 4.0,
  "camera_id": 0,
  "objects": [
    {
      "id": 1,
      "pos_3d": [1.2, 0.4, 0.5],
      "bbox": [120.5, 80.2, 245.8, 380.1],
      "occlusion": 0.15,
      "velocity": [0.5, 0.3, 0.0],
      "motion_type": "linear"
    }
  ]
}
```

### Field Descriptions

- **frame**: Frame index (0-based)
- **timestamp**: Time in seconds
- **camera_id**: Camera identifier (0-3 for 4 cameras)
- **objects**: List of visible objects
  - **id**: Unique object ID
  - **pos_3d**: 3D position in world coordinates [x, y, z]
  - **bbox**: 2D bounding box [x1, y1, x2, y2] in image coordinates
  - **occlusion**: Occlusion level (0.0 = fully visible, 1.0 = fully occluded)
  - **velocity**: 3D velocity vector [vx, vy, vz]
  - **motion_type**: Type of motion ("linear", "circular", "random_walk", "stationary")

## 🧠 Model Architecture

### LSTM Tracker

```
Input: (batch, seq_len=10, input_size=4)  # 10 frames of bbox history
    ↓
LSTM (hidden_size=128, num_layers=2, dropout=0.2)
    ↓
FC Layer (128 → 64)
    ↓
ReLU + Dropout
    ↓
FC Layer (64 → 4)
    ↓
Output: (batch, output_size=4)  # Predicted next bbox

For multi-step prediction:
Auto-regressive loop for N steps
```

### Training Details

- **Loss Function**: MSE Loss
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 16
- **Default Epochs**: 30-50
- **Sequence Length**: 10 frames input
- **Prediction Horizon**: 5 frames output

## 📈 Performance Metrics

### Stage 1 Output
- **Frames Generated**: ~300-1200 frames (10-40 seconds)
- **Objects per Frame**: 1-5 objects
- **Cameras**: 4 synchronized viewpoints
- **Frame Rate**: 30 FPS

### Stage 2 Training
- **Dataset Size**: Depends on sequence duration
- **Training Samples**: ~50-500 per object
- **Training Time**: ~1-5 minutes (CPU), ~30s (GPU)
- **Model Size**: ~2-5 MB

## 🔧 Configuration

### Motion Types

1. **LINEAR**: Straight-line motion
   ```python
   velocity=[vx, vy, vz]  # meters per second
   ```

2. **CIRCULAR**: Circular motion around a center
   ```python
   center=[cx, cy, cz]
   radius=2.0  # meters
   angular_velocity=0.5  # radians per second
   ```

3. **RANDOM_WALK**: Brownian-like random motion
   ```python
   step_size=0.1  # meters per step
   ```

4. **STATIONARY**: No motion
   ```python
   # No additional parameters
   ```

### Camera Configuration

Default 4-camera setup:
- **South**: [scene_x/2, -2, 3]
- **North**: [scene_x/2, scene_y+2, 3]
- **West**: [-2, scene_y/2, 3]
- **East**: [scene_x+2, scene_y/2, 3]

All cameras look at scene center: [scene_x/2, scene_y/2, 0]

## 🎯 Next Steps (Future Enhancements)

### Immediate (Weeks 1-2)
- [ ] Add visualization tools for motion sequences
- [ ] Implement trajectory smoothing
- [ ] Add more motion patterns (zigzag, figure-8, etc.)
- [ ] Improve occlusion calculation with raycasting

### Short-term (Weeks 3-4)
- [ ] **Stage 3**: ReID appearance variation simulation
- [ ] Integrate with YOLO detection pipeline
- [ ] Add Hungarian matching for data association
- [ ] Implement online tracking evaluation

### Medium-term (Months 2-3)
- [ ] **Stage 4**: Complete integrated tracking system
- [ ] Add Kalman filter fusion with LSTM predictions
- [ ] Multi-camera fusion and world-space tracking
- [ ] Real-time performance optimization

## 🐛 Known Limitations

1. **Simplified Projection**: Current 3D→2D projection is simplified. For production, use proper camera matrices.

2. **Occlusion Estimation**: Occlusion is currently randomized. Should use depth-based raycasting.

3. **No Visual Rendering**: Stage 1 generates metadata but doesn't render RGB images (can be added).

4. **Fixed Object Size**: All objects are same size (0.6m cubes). Should support variable sizes.

5. **No Collision Handling**: Objects can pass through each other. Add collision detection if needed.

## 📚 References

### Related Files
- `Two-Dimension_Dual-Leg_System_Plan_v2.md` - Overall system architecture
- `future_plan_dual_path.md` - Detailed Path 1 & 2 roadmap
- `Project_Status_Summary_Report.md` - Current progress report

### Key Technologies
- **PyBullet**: Physics simulation and multi-view rendering
- **PyTorch**: LSTM model and training pipeline
- **LSTM**: Temporal sequence modeling
- **Multi-Object Tracking (MOT)**: Core tracking algorithms

## 📞 Support

For questions or issues:
1. Check the documentation in `/mnt/project/`
2. Review the code comments in `path2_stage1_2_implementation.py`
3. Examine the output JSON to understand data format

## ✅ Validation Checklist

- [x] Stage 1: Motion sequence generation works
- [x] Stage 1: Multi-camera synchronized data
- [x] Stage 1: JSON export with correct format
- [x] Stage 2: Dataset creation from JSON
- [x] Stage 2: LSTM model training pipeline
- [x] Stage 2: Multi-step prediction capability
- [x] Stage 2: Model checkpointing
- [ ] Stage 3: ReID integration (next phase)
- [ ] Stage 4: Full tracking system (future)

---

**Status**: ✅ Ready for Testing  
**Last Updated**: 2025-11-14  
**Version**: 1.0
