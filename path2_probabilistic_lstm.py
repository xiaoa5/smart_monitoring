#!/usr/bin/env python3
"""
Path2 Phase B: Probabilistic LSTM for Multi-Camera 3D Tracking
==============================================================

This module implements a probabilistic LSTM that:
1. Takes multi-camera bbox sequences as input (1-3+ cameras)
2. Outputs 3D position as Gaussian distribution N(μ, Σ)
3. Learns heteroscedastic uncertainty (data-dependent variance)
4. Handles missing observations and noisy measurements

Architecture:
    Input: [N_cameras, Seq_len, 4] bbox sequences (YOLO format)
    Attention Fusion: Multi-camera feature fusion
    LSTM: Temporal modeling
    Output Heads:
        - mean_head → [x, y, z] (predicted position)
        - logvar_head → [σ²_x, σ²_y, σ²_z] (uncertainty)

Training:
    Loss: Gaussian Negative Log-Likelihood (NLL)
    L = 0.5 * (log(2π) + log(σ²) + (y - μ)²/σ²)

Author: Claude
Date: 2025-11-18
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LSTMConfig:
    """Configuration for Probabilistic LSTM"""
    # Model architecture
    input_dim: int = 4  # YOLO bbox: [cx, cy, w, h]
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 3  # 3D position: [x, y, z]
    dropout: float = 0.1

    # Attention fusion
    attention_heads: int = 4
    attention_dim: int = 64

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50
    weight_decay: float = 1e-5

    # Data
    seq_len: int = 10  # Input sequence length
    max_cameras: int = 4  # Maximum number of cameras

    # Uncertainty
    min_logvar: float = -10.0  # Minimum log-variance (prevents collapse)
    max_logvar: float = 10.0   # Maximum log-variance (prevents explosion)

    # Noise simulation (for data augmentation)
    bbox_noise_std: float = 0.02  # Standard deviation for bbox noise
    missing_prob: float = 0.1  # Probability of missing observation


# ============================================================================
# Multi-Camera Attention Fusion
# ============================================================================

class MultiCameraAttention(nn.Module):
    """
    Attention-based fusion of multi-camera bbox features.

    When multiple cameras observe the same object, we need to fuse
    their observations. Attention allows the model to:
    1. Weight cameras by reliability (e.g., closer cameras get higher weight)
    2. Handle missing observations (masked attention)
    3. Learn camera-specific biases
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Camera-specific embedding (learns camera characteristics)
        self.camera_embed = nn.Embedding(10, hidden_dim)  # Support up to 10 cameras

        # Bbox feature projection
        self.bbox_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,
                bbox_features: torch.Tensor,
                camera_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            bbox_features: [batch, num_cameras, input_dim]
            camera_ids: [batch, num_cameras] - camera indices
            mask: [batch, num_cameras] - True for missing observations

        Returns:
            fused_features: [batch, hidden_dim]
        """
        batch_size, num_cams, _ = bbox_features.shape

        # Project bbox features
        bbox_feats = self.bbox_proj(bbox_features)  # [batch, num_cams, hidden]

        # Add camera-specific embeddings
        cam_embeds = self.camera_embed(camera_ids)  # [batch, num_cams, hidden]
        features = bbox_feats + cam_embeds

        # Self-attention over cameras
        # mask: True values are masked out (not attended to)
        attn_output, attn_weights = self.attention(
            query=features,
            key=features,
            value=features,
            key_padding_mask=mask,
            need_weights=True
        )

        # Average pool over cameras (masked average)
        if mask is not None:
            # Set masked positions to 0
            attn_output = attn_output.masked_fill(mask.unsqueeze(-1), 0.0)
            # Count valid cameras
            valid_counts = (~mask).sum(dim=1, keepdim=True).float()  # [batch, 1]
            valid_counts = torch.clamp(valid_counts, min=1.0)  # Avoid division by 0
            fused = attn_output.sum(dim=1) / valid_counts  # [batch, hidden]
        else:
            fused = attn_output.mean(dim=1)  # [batch, hidden]

        # Output projection
        output = self.output_proj(fused)

        return output


# ============================================================================
# Probabilistic LSTM Tracker
# ============================================================================

class ProbabilisticLSTMTracker(nn.Module):
    """
    Probabilistic LSTM for 3D object tracking from multi-camera bboxes.

    Key features:
    1. Multi-camera fusion with attention
    2. Temporal modeling with LSTM
    3. Outputs Gaussian distribution: N(μ, Σ)
    4. Heteroscedastic uncertainty (learns when to be uncertain)

    Example:
        - Many cameras, clear view → Low uncertainty
        - One camera, occlusion → High uncertainty
        - Erratic motion → High uncertainty
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        # Multi-camera attention fusion
        self.camera_fusion = MultiCameraAttention(
            input_dim=config.input_dim,
            hidden_dim=config.attention_dim,
            num_heads=config.attention_heads
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=config.attention_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0
        )

        # Output heads
        self.mean_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

        self.logvar_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )

    def forward(self,
                bbox_seq: torch.Tensor,
                camera_ids_seq: torch.Tensor,
                mask_seq: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            bbox_seq: [batch, seq_len, num_cameras, 4]
            camera_ids_seq: [batch, seq_len, num_cameras]
            mask_seq: [batch, seq_len, num_cameras] - True for missing

        Returns:
            mean: [batch, seq_len, 3] - predicted 3D positions
            logvar: [batch, seq_len, 3] - log-variance (uncertainty)
        """
        batch_size, seq_len, num_cams, _ = bbox_seq.shape

        # Fuse multi-camera observations at each timestep
        fused_features = []
        for t in range(seq_len):
            bbox_t = bbox_seq[:, t, :, :]  # [batch, num_cams, 4]
            cam_ids_t = camera_ids_seq[:, t, :]  # [batch, num_cams]
            mask_t = mask_seq[:, t, :] if mask_seq is not None else None

            fused_t = self.camera_fusion(bbox_t, cam_ids_t, mask_t)
            fused_features.append(fused_t)

        # Stack to create sequence
        fused_seq = torch.stack(fused_features, dim=1)  # [batch, seq_len, hidden]

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(fused_seq)  # [batch, seq_len, hidden_dim]

        # Output heads
        mean = self.mean_head(lstm_out)  # [batch, seq_len, 3]
        logvar = self.logvar_head(lstm_out)  # [batch, seq_len, 3]

        # Clamp log-variance to prevent numerical instability
        logvar = torch.clamp(
            logvar,
            min=self.config.min_logvar,
            max=self.config.max_logvar
        )

        return mean, logvar

    def predict_distribution(self,
                            bbox_seq: torch.Tensor,
                            camera_ids_seq: torch.Tensor,
                            mask_seq: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict 3D position distribution.

        Returns:
            mean: [seq_len, 3] - predicted positions
            std: [seq_len, 3] - standard deviations
        """
        self.eval()
        with torch.no_grad():
            mean, logvar = self.forward(bbox_seq, camera_ids_seq, mask_seq)
            std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))

        return mean[0].cpu().numpy(), std[0].cpu().numpy()


# ============================================================================
# Gaussian Negative Log-Likelihood Loss
# ============================================================================

def gaussian_nll_loss(pred_mean: torch.Tensor,
                     pred_logvar: torch.Tensor,
                     target: torch.Tensor,
                     reduction: str = 'mean') -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood Loss.

    For a Gaussian distribution N(μ, σ²), the NLL is:
        L = 0.5 * (log(2π) + log(σ²) + (y - μ)²/σ²)

    This loss function encourages the model to:
    1. Predict accurate mean (minimize (y - μ)²)
    2. Calibrate uncertainty (penalize overconfident predictions)

    Args:
        pred_mean: [batch, seq_len, 3] - predicted mean
        pred_logvar: [batch, seq_len, 3] - predicted log-variance
        target: [batch, seq_len, 3] - ground truth
        reduction: 'mean' or 'sum'

    Returns:
        loss: scalar
    """
    # Compute squared error
    squared_error = (target - pred_mean) ** 2

    # Compute NLL
    # L = 0.5 * (log(2π) + log(σ²) + (y-μ)²/σ²)
    # Since log(2π) is constant, we can drop it
    nll = 0.5 * (pred_logvar + squared_error / torch.exp(pred_logvar))

    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll


# ============================================================================
# Dataset
# ============================================================================

class MultiCameraDataset(Dataset):
    """
    Dataset for multi-camera 3D tracking.

    Loads data from Path2 Phase 1 JSON files and organizes into
    sequences for LSTM training.
    """

    def __init__(self,
                 json_dir: str,
                 seq_len: int = 10,
                 max_cameras: int = 4,
                 add_noise: bool = False,
                 noise_std: float = 0.02,
                 missing_prob: float = 0.1):
        """
        Args:
            json_dir: Directory containing camera JSON files
            seq_len: Sequence length for LSTM
            max_cameras: Maximum number of cameras to use
            add_noise: Whether to add noise to bboxes (data augmentation)
            noise_std: Standard deviation of bbox noise
            missing_prob: Probability of randomly dropping observations
        """
        self.json_dir = json_dir
        self.seq_len = seq_len
        self.max_cameras = max_cameras
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.missing_prob = missing_prob

        # Load all camera data
        self.camera_data = self._load_camera_data()

        # Organize into sequences
        self.sequences = self._create_sequences()

    def _load_camera_data(self) -> Dict[int, List[Dict]]:
        """Load data from all camera JSON files."""
        camera_data = {}

        # Find all camera JSON files
        json_files = [f for f in os.listdir(self.json_dir)
                     if f.startswith('cam_') and f.endswith('.json')]

        for json_file in json_files:
            # Extract camera ID
            cam_id = int(json_file.split('_')[1].split('.')[0])

            # Load JSON
            json_path = os.path.join(self.json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            camera_data[cam_id] = data

        return camera_data

    def _create_sequences(self) -> List[Dict]:
        """
        Create sequences for training.

        Each sequence contains:
        - bbox_seq: [seq_len, max_cameras, 4]
        - pos_3d_seq: [seq_len, 3]
        - camera_ids: [seq_len, max_cameras]
        - mask: [seq_len, max_cameras] (True for missing)
        """
        sequences = []

        # Get number of frames (assume all cameras have same length)
        cam_ids = list(self.camera_data.keys())
        num_frames = len(self.camera_data[cam_ids[0]])

        # Get object IDs from first frame
        first_frame = self.camera_data[cam_ids[0]][0]
        object_ids = [obj['id'] for obj in first_frame['objects']]

        # Create sequences for each object
        for obj_id in object_ids:
            # Extract sequences with sliding window
            for start_idx in range(0, num_frames - self.seq_len + 1, self.seq_len // 2):
                seq = self._extract_sequence(obj_id, start_idx)
                if seq is not None:
                    sequences.append(seq)

        return sequences

    def _extract_sequence(self, obj_id: int, start_idx: int) -> Optional[Dict]:
        """Extract a single sequence for given object."""
        seq_len = self.seq_len
        max_cams = self.max_cameras

        bbox_seq = np.zeros((seq_len, max_cams, 4), dtype=np.float32)
        pos_3d_seq = np.zeros((seq_len, 3), dtype=np.float32)
        camera_ids = np.zeros((seq_len, max_cams), dtype=np.int64)
        mask = np.ones((seq_len, max_cams), dtype=bool)  # True = missing

        cam_ids = sorted(list(self.camera_data.keys()))[:max_cams]

        for t in range(seq_len):
            frame_idx = start_idx + t

            # For each camera
            for c, cam_id in enumerate(cam_ids):
                camera_ids[t, c] = cam_id

                # Get frame data
                frame = self.camera_data[cam_id][frame_idx]

                # Find object in frame
                obj = None
                for o in frame['objects']:
                    if o['id'] == obj_id:
                        obj = o
                        break

                if obj is not None:
                    # Get bbox
                    bbox = obj['bbox']  # [cx, cy, w, h]
                    bbox_seq[t, c] = bbox

                    # Get 3D position (same for all cameras)
                    pos_3d_seq[t] = obj['pos_3d']

                    # Mark as valid
                    mask[t, c] = False

        # Check if we have at least one valid observation per timestep
        valid_per_timestep = (~mask).sum(axis=1)
        if (valid_per_timestep > 0).all():
            return {
                'bbox_seq': bbox_seq,
                'pos_3d_seq': pos_3d_seq,
                'camera_ids': camera_ids,
                'mask': mask,
                'obj_id': obj_id
            }
        else:
            return None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]

        bbox_seq = seq['bbox_seq'].copy()
        pos_3d_seq = seq['pos_3d_seq'].copy()
        camera_ids = seq['camera_ids'].copy()
        mask = seq['mask'].copy()

        # Add noise (data augmentation)
        if self.add_noise:
            # Add Gaussian noise to bboxes
            noise = np.random.randn(*bbox_seq.shape) * self.noise_std
            bbox_seq = bbox_seq + noise

            # Randomly drop observations
            drop_mask = np.random.rand(*mask.shape) < self.missing_prob
            mask = mask | drop_mask  # Combine masks

        return {
            'bbox_seq': torch.from_numpy(bbox_seq),
            'pos_3d_seq': torch.from_numpy(pos_3d_seq),
            'camera_ids': torch.from_numpy(camera_ids),
            'mask': torch.from_numpy(mask)
        }


# ============================================================================
# Trainer
# ============================================================================

class ProbabilisticTrainer:
    """Trainer for Probabilistic LSTM."""

    def __init__(self,
                 model: ProbabilisticLSTMTracker,
                 config: LSTMConfig,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Metrics
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc='Training'):
            bbox_seq = batch['bbox_seq'].to(self.device)
            pos_3d_seq = batch['pos_3d_seq'].to(self.device)
            camera_ids = batch['camera_ids'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward
            pred_mean, pred_logvar = self.model(bbox_seq, camera_ids, mask)

            # Loss
            loss = gaussian_nll_loss(pred_mean, pred_logvar, pos_3d_seq)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_uncertainty = 0.0

        for batch in val_loader:
            bbox_seq = batch['bbox_seq'].to(self.device)
            pos_3d_seq = batch['pos_3d_seq'].to(self.device)
            camera_ids = batch['camera_ids'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward
            pred_mean, pred_logvar = self.model(bbox_seq, camera_ids, mask)

            # Loss
            loss = gaussian_nll_loss(pred_mean, pred_logvar, pos_3d_seq)
            total_loss += loss.item()

            # Mean Absolute Error
            mae = torch.abs(pred_mean - pos_3d_seq).mean()
            total_mae += mae.item()

            # Average uncertainty (σ)
            std = torch.exp(0.5 * pred_logvar)
            total_uncertainty += std.mean().item()

        metrics = {
            'loss': total_loss / len(val_loader),
            'mae': total_mae / len(val_loader),
            'uncertainty': total_uncertainty / len(val_loader)
        }

        return metrics

    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int) -> Dict[str, List[float]]:
        """Full training loop."""
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])

            # Print metrics
            print(f"\nTrain Loss: {train_loss:.6f}")
            print(f"Val Loss:   {val_metrics['loss']:.6f}")
            print(f"Val MAE:    {val_metrics['mae']:.6f}")
            print(f"Val Std:    {val_metrics['uncertainty']:.6f}")

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                print(f"✓ New best model (Val Loss: {best_val_loss:.6f})")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == '__main__':
    print("Path2 Phase B: Probabilistic LSTM")
    print("="*60)

    # Configuration
    config = LSTMConfig(
        seq_len=10,
        max_cameras=4,
        num_epochs=50
    )

    print(f"Device: {torch.cuda.is_available() and 'cuda' or 'cpu'}")
    print(f"Sequence Length: {config.seq_len}")
    print(f"Max Cameras: {config.max_cameras}")
    print(f"Hidden Dim: {config.hidden_dim}")

    # Create model
    model = ProbabilisticLSTMTracker(config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {num_params:,}")

    # Test forward pass
    batch_size = 4
    bbox_seq = torch.randn(batch_size, config.seq_len, config.max_cameras, 4)
    camera_ids = torch.randint(0, 4, (batch_size, config.seq_len, config.max_cameras))
    mask = torch.rand(batch_size, config.seq_len, config.max_cameras) > 0.8  # 20% missing

    mean, logvar = model(bbox_seq, camera_ids, mask)
    std = torch.exp(0.5 * logvar)

    print(f"\n{'='*60}")
    print("Forward Pass Test")
    print(f"{'='*60}")
    print(f"Input shape:  {list(bbox_seq.shape)}")
    print(f"Mean shape:   {list(mean.shape)}")
    print(f"Logvar shape: {list(logvar.shape)}")
    print(f"\nPredicted position (first sample, last timestep):")
    print(f"  Mean: {mean[0, -1].detach().numpy()}")
    print(f"  Std:  {std[0, -1].detach().numpy()}")

    print(f"\n{'='*60}")
    print("✓ Phase B implementation complete!")
    print("Ready for Colab testing.")
    print(f"{'='*60}")
