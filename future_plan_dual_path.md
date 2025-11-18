# Future Technical Development Plan: Dual-Path Architecture

This document presents the future development roadmap for the project. The plan consists of two complementary technical tracks that will converge into a unified intelligent monitoring prototype.

---

# Overview of the Two Development Paths

- **Path 1: 3DGS-Based Data Generation and 3D Scene Reconstruction**  
  Focuses on multi-view dataset export, Gaussian Splatting reconstruction, dynamic Gaussian manipulation, and automated annotation.

- **Path 2: PyBullet-Based Tracking, LSTM Sequence Modeling, and ReID**  
  Focuses on temporal dynamics, identity consistency, and robust multi-object tracking.

Both paths are designed to be developed in parallel and integrated at key convergence points.

---

# Path 1: 3DGS Data Generation & 3D Scene Reconstruction

## Phase 1: PyBullet → Nerfstudio Dataset Export
**Objective:** Convert simulation-generated multi-view data into standard formats used by Nerfstudio and COLMAP.

### Key Tasks
- Export RGB images from each camera.
- Export camera intrinsics (fx, fy, cx, cy).
- Export camera extrinsics (R, t) in world coordinates.
- Generate `transforms.json` or COLMAP-format camera files.

### Module Diagram
```
PyBullet Renderer
   ├── multi-view RGB
   ├── depth / segmentation
   ├── camera poses
        │
        ▼
Nerfstudio Dataset Exporter
   ├── images/
   ├── transforms.json
        │
        ▼
3DGS Trainer (splatfacto)
```

### Example Pseudocode
```python
def export_for_nerfstudio(scene, cameras, out_dir):
    data = []
    for cam in cameras:
        rgb = render_rgb(scene, cam)
        K = cam.intrinsics()
        R, t = cam.extrinsics()
        save_image(rgb, out_dir)
        data.append({"file": img_path, "K": K, "R": R, "t": t})
    write_transforms_json(data, out_dir)
```

---

## Phase 2: Static Scene Reconstruction with 3DGS
**Objective:** Train a splatfacto model on exported simulation data and verify reconstruction quality.

### Deliverables
- Full 3D Gaussian Splat representation of the simulated room.
- Ability to orbit the scene and render novel views.

### Workflow
```
Nerfstudio Dataset
        │
        ▼
splatfacto Training
        │
        ▼
Gaussian Splat (.ply)
        │
        ▼
gsplat Renderer
```

---

## Phase 3: Dynamic Gaussian Segmentation & Object Manipulation
**Objective:** Support object-level manipulation inside 3DGS for dynamic scene simulation.

### Technical Points
- Segment Gaussians into background and object-specific subsets.
- Use PyBullet segmentation masks to classify Gaussians.
- Apply translation/rotation transforms to clusters of Gaussians.

### Example Pseudocode
```python
def segment_gaussians(gaussians, masks):
    objects = []
    for g in gaussians:
        if inside_mask(g.position, masks):
            objects.append(g)
    return objects

def apply_transform(gaussians, transform):
    for g in gaussians:
        g.position = transform @ g.position
        g.rotation = update_quaternion(g.rotation, transform)
```

---

## Phase 4: 3DGS-Based Automatic Annotation Engine
**Objective:** Render synthetic multi-view data with consistent annotations directly from the Gaussian field.

### Outputs
- RGB
- Instance masks
- YOLO bounding boxes

### Module Diagram
```
Dynamic 3DGS Scene
   ├── object Gaussians
   ├── transformations
        │
        ▼
gsplat Renderer
   ├── RGB
   ├── mask
   ├── depth
        │
        ▼
Auto-Label Generator
   ├── bbox
   ├── instance IDs
```

---

# Path 2: PyBullet Tracking / LSTM / ReID

## Phase 1: Motion Sequence Generator
**Objective:** Produce continuous multi-object trajectories with controlled motion and occlusion patterns.

### Supported Motion Types
- Linear
- Circular
- Random walk

### Output Structure
```json
{
  "frame": 120,
  "objects": [
    {
      "id": 1,
      "pos_3d": [1.2, 0.4, 0.0],
      "bbox": [x1, y1, x2, y2],
      "occlusion": 0.2
    }
  ]
}
```

---

## Phase 2: LSTM-Based Multi-Object Tracker
**Objective:** Improve temporal stability by predicting future bounding boxes or world-space positions.

### Example Model Skeleton
```python
class LSTMTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(4, 128, batch_first=True)
        self.fc = nn.Linear(128, 4)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        return self.fc(out[:, -1])
```

---

## Phase 3: Appearance Variation Simulation & ReID
**Objective:** Enable identity consistency across cameras by training a ReID model on varied appearances.

### Module Diagram
```
PyBullet Object
   ├── base mesh
   ├── color variations
   ├── texture variations
        │
        ▼
ReID Dataset Generator
        │
        ▼
ReID Training
```

---

## Phase 4: Integrated Tracking System
**Objective:** Fuse detection, LSTM prediction, data association, and ReID identity validation.

### System Pipeline
```
YOLO Detection
        │
        ▼
LSTM Prediction
        │
        ▼
Hungarian Association
        │
        ▼
ReID Identity Check
        │
        ▼
Final Tracks
```

---

# Integration of Both Paths

## Convergence Point 1: 3DGS Automatic Annotation → Tracking Training
3DGS can provide large-scale, perfectly consistent synthetic data for:
- Object detection
- LSTM-based temporal tracking
- ReID identity embedding training

## Convergence Point 2: Intelligent Monitoring Demo
Once both paths mature, they combine into a unified system:
- Full 3D scene reconstruction (3DGS)
- Multi-camera detection (YOLO)
- Temporal & identity tracking (LSTM + ReID)
- Behavioral and event-level reasoning

---

# High-Level Development Roadmap
```
Phase 1: Data Export / Trajectory Generation
Phase 2: Static 3DGS / LSTM Tracking
Phase 3: Dynamic Gaussian / ReID
Phase 4: Automatic Annotation Engine / Full Tracking System
Phase 5: System Integration & Demo
```

---

This document serves as the official future roadmap for the dual-path development strategy, providing a clear structure for cross-team coordination and long-term planning.

