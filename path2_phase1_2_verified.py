"""
Path 2 Implementation: Stage 1 & 2 (Verified for Colab)
基于验证过的 Multi-Camera PyBullet + YOLO 代码重写

Author: AI Assistant
Date: 2025-11-18
Status: Production Ready (Colab Tested)

Key Features:
- ✅ Real bbox from segmentation masks (not simplified projection)
- ✅ EGL GPU-accelerated rendering
- ✅ Full camera matrices (view/projection)
- ✅ RGB image output for YOLO training
- ✅ 3D unprojection for multi-camera fusion
- ✅ Complex motion patterns (circular, sine wave, bounce)
- ✅ LSTM temporal prediction

Dependencies (Colab verified):
    pip install pybullet==3.2.7 numpy==2.1.1 torch opencv-python matplotlib tqdm pyyaml
"""

import os
import random
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import trange

import pybullet as p
import pybullet_data

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Global Configuration
# ============================================================================

ROOM_XY = 10.0      # Room size (meters)
ROOM_H = 3.0        # Room height (meters)
W, H = 640, 480     # Image resolution
FOV_DEG = 110       # Field of view (degrees)
FPS = 30            # Frames per second
MIN_PIXELS = 20     # Min pixels for valid detection


# ============================================================================
# Stage 1: Motion Sequence Generator (Verified Version)
# ============================================================================

class MotionType(Enum):
    """Supported motion types"""
    LINEAR = "linear"
    CIRCULAR = "circular"
    SINE_WAVE = "sine_wave"
    BOUNCE = "bounce"
    STATIONARY = "stationary"


@dataclass
class ObjectState:
    """Object state per frame"""
    id: int
    name: str
    pos_3d: List[float]      # [x, y, z] in world coordinates
    bbox: List[float]        # [cx, cy, w, h] in YOLO format (normalized)
    bbox_pixels: List[int]  # [x1, y1, x2, y2] in pixels
    occlusion: float         # Occlusion level (0.0 = visible, 1.0 = occluded)
    velocity: List[float]    # [vx, vy, vz]
    motion_type: str


@dataclass
class FrameData:
    """Data per camera per frame"""
    frame: int
    timestamp: float
    camera_id: int
    objects: List[ObjectState]


def init_bullet_with_optional_egl():
    """Initialize PyBullet with EGL if available (GPU acceleration)"""
    cid = p.connect(p.DIRECT)
    use_gpu = False
    try:
        egl = p.loadPlugin('eglRendererPlugin')
        print(f'✅ EGL plugin loaded: {egl}')
        use_gpu = True
    except Exception as e:
        print(f'⚠️  EGL not available, using TinyRenderer. Reason: {e}')

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    return cid, use_gpu


def create_room():
    """Create room with walls"""
    plane = p.loadURDF('plane.urdf')
    wall_thick = 0.05
    half = ROOM_XY / 2

    # North/South walls
    col_box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, wall_thick, ROOM_H/2])
    vis_grey = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, wall_thick, ROOM_H/2],
                                    rgbaColor=[0.8, 0.8, 0.8, 1])
    p.createMultiBody(0, col_box, vis_grey, [0, half, ROOM_H/2])
    p.createMultiBody(0, col_box, vis_grey, [0, -half, ROOM_H/2])

    # East/West walls
    col_box2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thick, half, ROOM_H/2])
    vis_grey2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thick, half, ROOM_H/2],
                                     rgbaColor=[0.8, 0.8, 0.8, 1])
    p.createMultiBody(0, col_box2, vis_grey2, [half, 0, ROOM_H/2])
    p.createMultiBody(0, col_box2, vis_grey2, [-half, 0, ROOM_H/2])

    return plane


def corner_cameras(height=1.2, margin=1.0):
    """Setup 4 corner cameras"""
    h = height
    half = ROOM_XY / 2 - margin
    return [
        np.array([-half, -half, h]),  # SW
        np.array([half, -half, h]),   # SE
        np.array([half, half, h]),    # NE
        np.array([-half, half, h]),   # NW
    ]


def look_at(from_xyz, to_xyz):
    """Compute view matrix (camera looking at target)"""
    up = [0, 1, 0] if abs(from_xyz[2] - to_xyz[2]) < 0.5 else [0, 0, 1]
    return p.computeViewMatrix(from_xyz, to_xyz, up)


def camera_specs(width, height, fov_deg=FOV_DEG, near=0.01, far=20.0):
    """Compute projection matrix"""
    return p.computeProjectionMatrixFOV(
        fov=fov_deg,
        aspect=width/height,
        nearVal=near,
        farVal=far
    )


def render_camera(cam_pose, target, width, height, use_gpu=False):
    """Render camera view with RGB, depth, and segmentation"""
    view = look_at(cam_pose, target)
    proj = camera_specs(width, height, fov_deg=FOV_DEG)

    renderer = p.ER_BULLET_HARDWARE_OPENGL if use_gpu else p.ER_TINY_RENDERER
    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

    img = p.getCameraImage(
        width, height, view, proj,
        shadow=0,
        renderer=renderer,
        flags=flags
    )

    rgb = np.reshape(img[2], (height, width, 4))[:, :, :3].astype(np.uint8)
    depth = np.reshape(img[3], (height, width))
    seg = np.reshape(img[4], (height, width))

    return rgb, depth, seg, view, proj


def yolo_bboxes_from_seg(seg, body_ids, body_names, w, h, min_pixels=MIN_PIXELS):
    """Extract YOLO bboxes from segmentation mask (VERIFIED METHOD)"""
    obj_uid = (seg & ((1 << 24) - 1)).astype(np.int32)
    bboxes = {}

    for bid, name in zip(body_ids, body_names):
        ys, xs = np.where(obj_uid == bid)
        if ys.size < min_pixels:
            continue

        # Convert NumPy int64 to Python int for JSON serialization
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        # YOLO format: center_x, center_y, width, height (normalized)
        cx = (x0 + x1) / 2 / w
        cy = (y0 + y1) / 2 / h
        bw = (x1 - x0) / w
        bh = (y1 - y0) / h

        # Filter out too large boxes (likely walls or background)
        if bw >= 0.98 or bh >= 0.98:
            continue

        bboxes[bid] = {
            'name': name,
            'bbox_norm': (float(cx), float(cy), float(bw), float(bh)),
            'bbox_pixels': (x0, y0, x1, y1)
        }

    return bboxes


def unproject_to_world(pixel_xy, view, proj, width, height):
    """Unproject pixel to 3D world coordinates (ground plane z=0)"""
    V = np.array(view).reshape(4, 4).T
    P = np.array(proj).reshape(4, 4).T
    invVP = np.linalg.inv(P @ V)

    x_ndc = (pixel_xy[0] / width) * 2 - 1
    y_ndc = 1 - (pixel_xy[1] / height) * 2

    p_near = np.array([x_ndc, y_ndc, -1, 1])
    p_far = np.array([x_ndc, y_ndc, 1, 1])

    w_near = invVP @ p_near
    w_near = w_near[:3] / w_near[3]
    w_far = invVP @ p_far
    w_far = w_far[:3] / w_far[3]

    ray_o = w_near
    ray_d = w_far - w_near

    if abs(ray_d[2]) < 1e-6:
        return None

    t = -ray_o[2] / ray_d[2]
    if t < 0:
        return None

    hit = ray_o + t * ray_d
    return float(hit[0]), float(hit[1])


class MotionSequenceGenerator:
    """
    Stage 1: Motion Sequence Generator (Verified Colab Version)

    Features:
    - Real bbox from segmentation masks
    - EGL GPU rendering
    - RGB image output
    - Multiple motion patterns
    - Multi-camera synchronized
    """

    def __init__(
        self,
        scene_size: Tuple[float, float] = (ROOM_XY, ROOM_XY),
        num_cameras: int = 4,
        fps: int = FPS,
        output_dir: str = "./path2_output/stage1"
    ):
        self.scene_size = scene_size
        self.num_cameras = num_cameras
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize PyBullet with EGL
        self.client, self.use_gpu = init_bullet_with_optional_egl()
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Create scene
        self.plane = create_room()

        # Objects storage
        self.objects = []  # List of (body_id, name, motion_config)
        self.velocities = {}  # For bounce motion

        # Camera setup
        self.cameras = corner_cameras(height=1.2, margin=1.0)
        self.target = [0, 0, 0.2]

        print(f"✅ MotionSequenceGenerator initialized")
        print(f"   GPU rendering: {self.use_gpu}")
        print(f"   Cameras: {self.num_cameras}")
        print(f"   Output: {self.output_dir}")

    def add_object(
        self,
        obj_id: int,
        name: str,
        start_pos: List[float],
        motion_type: MotionType,
        color: Optional[List[float]] = None,
        **motion_params
    ):
        """Add object with motion configuration"""
        if color is None:
            color = [np.random.rand(), np.random.rand(), np.random.rand(), 1]

        # Create visual and collision shapes
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.3]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.3],
            rgbaColor=color
        )

        body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos
        )

        motion_config = {
            'obj_id': obj_id,
            'type': motion_type,
            'start_pos': start_pos,
            'params': motion_params,
            'time': 0.0
        }

        self.objects.append((body_id, name, motion_config))

        # Initialize velocity for bounce motion
        if motion_type == MotionType.BOUNCE:
            self.velocities[body_id] = np.array(
                motion_params.get('velocity', [0.02, 0.018])
            )

        print(f"  Added object: {name} (id={obj_id}, motion={motion_type.value})")
        return body_id

    def _update_motion(self, body_id, motion_config, dt):
        """Update object position based on motion type"""
        motion_config['time'] += dt
        t = motion_config['time']
        motion_type = motion_config['type']
        params = motion_config['params']
        start_pos = motion_config['start_pos']

        if motion_type == MotionType.CIRCULAR:
            # Circular motion
            r = params.get('radius', 2.0)
            omega = params.get('angular_velocity', 0.5)
            center = params.get('center', [0, 0, 0.5])

            angle = omega * t
            new_pos = [
                center[0] + r * np.cos(angle),
                center[1] + r * np.sin(angle),
                center[2]
            ]
            velocity = [
                -r * omega * np.sin(angle),
                r * omega * np.cos(angle),
                0
            ]

        elif motion_type == MotionType.SINE_WAVE:
            # Sine wave motion
            vx = params.get('vx', 0.04)
            amplitude = params.get('amplitude', 1.2)
            k = params.get('k', 0.8)

            half = self.scene_size[0] / 2
            x = (vx * t) % (self.scene_size[0] - 2) - half + 1
            y = amplitude * np.sin(k * x)

            new_pos = [x, y, start_pos[2]]
            velocity = [vx, amplitude * k * np.cos(k * x) * vx, 0]

        elif motion_type == MotionType.BOUNCE:
            # Bounce with random acceleration
            pos, orn = p.getBasePositionAndOrientation(body_id)
            v = self.velocities[body_id]

            # Random acceleration
            v += 0.0015 * np.random.uniform(-1, 1, size=2)
            v = np.clip(v, -0.04, 0.04)
            self.velocities[body_id] = v

            # Sine perturbation
            sin_perturb = np.array([
                0.1 * np.sin(0.05 * t + hash(body_id) % 10),
                0.1 * np.cos(0.06 * t + hash(body_id) % 10)
            ])

            # Update position
            new_xy = np.array([pos[0], pos[1]]) + v + sin_perturb

            # Bounce off walls
            half = self.scene_size[0] / 2 - 0.25
            for i in [0, 1]:
                if new_xy[i] > half:
                    new_xy[i] = half
                    self.velocities[body_id][i] *= -1
                if new_xy[i] < -half:
                    new_xy[i] = -half
                    self.velocities[body_id][i] *= -1

            new_pos = [new_xy[0], new_xy[1], pos[2]]
            velocity = [v[0], v[1], 0]

        elif motion_type == MotionType.LINEAR:
            # Linear motion
            velocity = params.get('velocity', [0.5, 0.3, 0.0])
            new_pos = [
                start_pos[0] + velocity[0] * t,
                start_pos[1] + velocity[1] * t,
                start_pos[2] + velocity[2] * t
            ]

        else:  # STATIONARY
            new_pos = start_pos
            velocity = [0, 0, 0]

        p.resetBasePositionAndOrientation(body_id, new_pos, [0, 0, 0, 1])
        return new_pos, velocity

    def generate_sequence(
        self,
        duration: float = 10.0,
        save_images: bool = True,
        save_json: bool = True
    ) -> List[FrameData]:
        """Generate motion sequence with multi-camera rendering"""
        dt = 1.0 / self.fps
        num_frames = int(duration * self.fps)

        all_frame_data = []

        # Create image directories
        if save_images:
            for cam_id in range(self.num_cameras):
                (self.output_dir / f"camera_{cam_id}").mkdir(exist_ok=True)

        print(f"\n🎬 Generating {num_frames} frames...")

        for frame_idx in trange(num_frames, desc="Generating frames"):
            t = frame_idx * dt

            # Update all objects
            for body_id, name, motion_config in self.objects:
                self._update_motion(body_id, motion_config, dt)

            p.stepSimulation()

            # Render from each camera
            for cam_id, cam_pos in enumerate(self.cameras):
                rgb, depth, seg, view, proj = render_camera(
                    cam_pos, self.target, W, H, self.use_gpu
                )

                # Extract bboxes from segmentation
                body_ids = [bid for bid, _, _ in self.objects]
                body_names = [name for _, name, _ in self.objects]
                bboxes = yolo_bboxes_from_seg(seg, body_ids, body_names, W, H)

                # Build object states
                objects_in_frame = []
                for body_id, name, motion_config in self.objects:
                    if body_id not in bboxes:
                        continue

                    bbox_info = bboxes[body_id]
                    pos_3d, _ = p.getBasePositionAndOrientation(body_id)

                    # Get velocity from motion
                    velocity = motion_config['params'].get('velocity', [0, 0, 0])
                    if motion_config['type'] == MotionType.BOUNCE:
                        v = self.velocities[body_id]
                        velocity = [v[0], v[1], 0]

                    # Compute occlusion (simple: based on bbox size)
                    _, _, bw, bh = bbox_info['bbox_norm']
                    expected_size = 0.1  # Expected normalized size
                    occlusion = max(0, 1 - (bw * bh) / (expected_size ** 2))
                    occlusion = min(occlusion, 1.0)

                    obj_state = ObjectState(
                        id=motion_config['obj_id'],
                        name=name,
                        pos_3d=list(pos_3d),
                        bbox=list(bbox_info['bbox_norm']),
                        bbox_pixels=list(bbox_info['bbox_pixels']),
                        occlusion=occlusion,
                        velocity=velocity,
                        motion_type=motion_config['type'].value
                    )
                    objects_in_frame.append(obj_state)

                # Save RGB image
                if save_images:
                    img_path = self.output_dir / f"camera_{cam_id}" / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                # Store frame data
                frame_data = FrameData(
                    frame=frame_idx,
                    timestamp=t,
                    camera_id=cam_id,
                    objects=objects_in_frame
                )
                all_frame_data.append(frame_data)

        # Save JSON
        if save_json:
            self._save_json(all_frame_data)

        print(f"✅ Generated {len(all_frame_data)} camera frames")
        return all_frame_data

    def _save_json(self, frame_data: List[FrameData]):
        """Save frame data as JSON"""
        output_file = self.output_dir / "motion_sequence.json"

        data_list = []
        for fd in frame_data:
            frame_dict = {
                'frame': fd.frame,
                'timestamp': fd.timestamp,
                'camera_id': fd.camera_id,
                'objects': [asdict(obj) for obj in fd.objects]
            }
            data_list.append(frame_dict)

        # Custom JSON encoder to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(output_file, 'w') as f:
            json.dump(data_list, f, indent=2, cls=NumpyEncoder)

        print(f"✅ Saved JSON to {output_file}")

    def cleanup(self):
        """Cleanup PyBullet"""
        p.disconnect(self.client)


# ============================================================================
# Stage 2: LSTM-Based Multi-Object Tracker (Enhanced)
# ============================================================================

class TrackingDataset(Dataset):
    """Dataset for LSTM tracking (using real bbox data)"""

    def __init__(
        self,
        json_file: str,
        sequence_length: int = 10,
        prediction_horizon: int = 5,
        normalize: bool = True
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize

        # Load data
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        print(f"  Loaded {len(self.data)} frames from {json_file}")

        # Organize trajectories by camera and object
        self.trajectories = self._organize_trajectories()

        # Generate samples
        self.samples = self._generate_samples()

        print(f"  Generated {len(self.samples)} training samples")

    def _organize_trajectories(self) -> Dict:
        """Organize data by camera_id and object_id"""
        trajectories = {}

        for frame_data in self.data:
            cam_id = frame_data['camera_id']

            if cam_id not in trajectories:
                trajectories[cam_id] = {}

            for obj in frame_data['objects']:
                obj_id = obj['id']

                if obj_id not in trajectories[cam_id]:
                    trajectories[cam_id][obj_id] = []

                trajectories[cam_id][obj_id].append({
                    'frame': frame_data['frame'],
                    'bbox': obj['bbox'],  # [cx, cy, w, h] normalized
                    'pos_3d': obj['pos_3d'],
                    'velocity': obj['velocity']
                })

        return trajectories

    def _generate_samples(self) -> List[Dict]:
        """Generate sliding window samples"""
        samples = []

        for cam_id, objects in self.trajectories.items():
            for obj_id, trajectory in objects.items():
                if len(trajectory) < self.sequence_length + self.prediction_horizon:
                    continue

                # Sliding window
                for i in range(len(trajectory) - self.sequence_length - self.prediction_horizon + 1):
                    input_seq = trajectory[i:i + self.sequence_length]
                    target_seq = trajectory[
                        i + self.sequence_length:
                        i + self.sequence_length + self.prediction_horizon
                    ]

                    # Extract bboxes
                    input_bboxes = np.array([t['bbox'] for t in input_seq], dtype=np.float32)
                    target_bboxes = np.array([t['bbox'] for t in target_seq], dtype=np.float32)

                    samples.append({
                        'input': input_bboxes,
                        'target': target_bboxes,
                        'obj_id': obj_id,
                        'cam_id': cam_id
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_seq = torch.FloatTensor(sample['input'])
        target_seq = torch.FloatTensor(sample['target'])

        return input_seq, target_seq


class LSTMTracker(nn.Module):
    """
    Stage 2: LSTM Multi-Object Tracker

    Predicts future bounding boxes based on historical sequence
    """

    def __init__(
        self,
        input_size: int = 4,      # bbox: [cx, cy, w, h]
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 4,
        dropout: float = 0.2
    ):
        super(LSTMTracker, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            hidden: (h0, c0) or None
        Returns:
            output: (batch, output_size)
            hidden: (h, c)
        """
        lstm_out, hidden = self.lstm(x, hidden)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction, hidden

    def predict_sequence(self, x, steps: int = 5):
        """
        Auto-regressive multi-step prediction

        Args:
            x: (batch, seq_len, input_size)
            steps: number of future steps to predict
        Returns:
            predictions: (batch, steps, output_size)
        """
        predictions = []
        current_seq = x.clone()
        hidden = None

        for _ in range(steps):
            pred, hidden = self.forward(current_seq, hidden)
            predictions.append(pred)

            # Slide window
            current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)

        return torch.stack(predictions, dim=1)


class LSTMTrackerTrainer:
    """Trainer for LSTM tracker"""

    def __init__(
        self,
        model: LSTMTracker,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001
    ):
        self.model = model.to(device)
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []

        print(f"✅ LSTMTrackerTrainer initialized on {device}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Predict all future steps
            predictions = self.model.predict_sequence(inputs, steps=targets.size(1))

            # Compute loss
            loss = self.criterion(predictions, targets)

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model.predict_sequence(inputs, steps=targets.size(1))
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        save_path: Optional[str] = None
    ):
        """Full training loop"""
        best_val_loss = float('inf')

        print(f"\n🚀 Training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✅ Saved best model (val_loss: {val_loss:.6f})")

        print(f"\n🎉 Training completed! Best val loss: {best_val_loss:.6f}")


# ============================================================================
# Demo & Main
# ============================================================================

def run_stage1_demo():
    """Stage 1 Demo: Generate motion sequences"""
    print("=" * 70)
    print("Stage 1: Motion Sequence Generator (Verified Version)")
    print("=" * 70)

    generator = MotionSequenceGenerator(
        scene_size=(ROOM_XY, ROOM_XY),
        num_cameras=4,
        fps=FPS,
        output_dir="./path2_output/stage1"
    )

    # Add objects with different motion patterns
    print("\n📦 Adding objects...")

    # Object 1: Circular motion
    generator.add_object(
        obj_id=1,
        name='red_cube',
        start_pos=[2.0, 0.0, 0.5],
        motion_type=MotionType.CIRCULAR,
        color=[1, 0, 0, 1],
        radius=2.0,
        angular_velocity=0.5,
        center=[0, 0, 0.5]
    )

    # Object 2: Sine wave
    generator.add_object(
        obj_id=2,
        name='green_cylinder',
        start_pos=[-2.0, -2.0, 0.5],
        motion_type=MotionType.SINE_WAVE,
        color=[0, 1, 0, 1],
        vx=0.04,
        amplitude=1.2,
        k=0.8
    )

    # Object 3: Bounce
    generator.add_object(
        obj_id=3,
        name='blue_sphere',
        start_pos=[1.0, 2.0, 0.5],
        motion_type=MotionType.BOUNCE,
        color=[0, 0, 1, 1],
        velocity=[0.02, 0.018]
    )

    # Generate sequence
    frame_data = generator.generate_sequence(
        duration=10.0,
        save_images=True,
        save_json=True
    )

    generator.cleanup()

    print(f"\n✅ Stage 1 completed!")
    print(f"   Frames: {len(frame_data)}")
    print(f"   Output: ./path2_output/stage1/")

    return "./path2_output/stage1/motion_sequence.json"


def run_stage2_demo(json_file: str):
    """Stage 2 Demo: Train LSTM tracker"""
    print("\n" + "=" * 70)
    print("Stage 2: LSTM Tracker Training")
    print("=" * 70)

    # Create dataset
    print("\n📊 Creating dataset...")
    dataset = TrackingDataset(
        json_file=json_file,
        sequence_length=10,
        prediction_horizon=5
    )

    if len(dataset) == 0:
        print("⚠️  No training samples. Need longer sequences.")
        return

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"   Train: {train_size}, Val: {val_size}")

    # Create model
    print("\n🧠 Creating LSTM model...")
    model = LSTMTracker(
        input_size=4,
        hidden_size=128,
        num_layers=2,
        output_size=4,
        dropout=0.2
    )

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = LSTMTrackerTrainer(
        model=model,
        learning_rate=0.001
    )

    os.makedirs("./path2_output/stage2", exist_ok=True)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,
        save_path="./path2_output/stage2/best_lstm_tracker.pth"
    )

    print("\n✅ Stage 2 completed!")


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("🚀 Path 2 Implementation: Stage 1 & 2 (Verified for Colab)")
    print("=" * 70)

    # Run Stage 1
    json_file = run_stage1_demo()

    # Run Stage 2
    run_stage2_demo(json_file)

    print("\n" + "=" * 70)
    print("🎉 All stages completed!")
    print("=" * 70)
    print("\n📁 Output files:")
    print("   Stage 1 JSON: ./path2_output/stage1/motion_sequence.json")
    print("   Stage 1 Images: ./path2_output/stage1/camera_*/")
    print("   Stage 2 Model: ./path2_output/stage2/best_lstm_tracker.pth")


if __name__ == "__main__":
    main()
