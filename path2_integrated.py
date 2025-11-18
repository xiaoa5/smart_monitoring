#!/usr/bin/env python3
"""
Path2 Phase D: Integrated End-to-End Pipeline
==============================================

This module integrates all Path2 components into a complete
3D tracking system:

Pipeline:
    Multi-Camera Data (Phase 1)
           ↓
    Probabilistic LSTM (Phase B)
           ↓
    Bayesian Constraint (Phase C)
           ↓
    Refined 3D Tracking

Features:
- End-to-end training and inference
- Adaptive constraint weighting
- Multi-scenario comparison
- Comprehensive visualization
- Performance metrics

Author: Claude
Date: 2025-11-18
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import Phase B and C components
from path2_probabilistic_lstm import (
    LSTMConfig,
    ProbabilisticLSTMTracker,
    MultiCameraDataset,
    ProbabilisticTrainer
)

from path2_constraints import (
    GaussianDistribution,
    TrajectoryConstraint,
    CircleConstraint,
    LineConstraint,
    SplineConstraint
)


# ============================================================================
# Integrated Tracker Configuration
# ============================================================================

@dataclass
class IntegratedConfig:
    """Configuration for integrated tracking system."""
    # Data
    data_dir: str = 'output/data'
    seq_len: int = 10
    max_cameras: int = 4

    # LSTM (Phase B)
    lstm_config: LSTMConfig = field(default_factory=LSTMConfig)

    # Constraint (Phase C)
    use_constraint: bool = True
    constraint_type: str = 'circle'  # 'circle', 'line', 'spline'
    constraint_std_radial: float = 0.01  # 1cm radial constraint

    # Adaptive constraint weighting
    adaptive_weighting: bool = True
    min_cameras_for_no_constraint: int = 3  # Don't use constraint if ≥3 cameras

    # Training
    train_split: float = 0.8
    num_epochs: int = 50
    batch_size: int = 32

    # Evaluation
    eval_scenarios: List[str] = field(default_factory=lambda: [
        '4cam_no_constraint',
        '2cam_no_constraint',
        '1cam_no_constraint',
        '1cam_with_constraint'
    ])

    # Output
    output_dir: str = 'output/integrated'
    save_models: bool = True
    save_plots: bool = True


# ============================================================================
# Integrated Tracker
# ============================================================================

class IntegratedTracker:
    """
    Integrated 3D tracker combining LSTM and constraints.

    This class manages the full pipeline from data loading to
    refined predictions.
    """

    def __init__(self, config: IntegratedConfig):
        """
        Args:
            config: Configuration for integrated system
        """
        self.config = config

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # LSTM model
        self.lstm = ProbabilisticLSTMTracker(config.lstm_config).to(self.device)

        # Constraint (to be set based on data)
        self.constraint: Optional[TrajectoryConstraint] = None

        # Dataset
        self.dataset: Optional[MultiCameraDataset] = None
        self.train_dataset = None
        self.val_dataset = None

        # Trainer
        self.trainer: Optional[ProbabilisticTrainer] = None

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def load_data(self):
        """Load multi-camera dataset."""
        print("\n" + "="*60)
        print("Loading Data")
        print("="*60)

        self.dataset = MultiCameraDataset(
            json_dir=self.config.data_dir,
            seq_len=self.config.seq_len,
            max_cameras=self.config.max_cameras,
            add_noise=True,
            noise_std=0.02,
            missing_prob=0.1
        )

        print(f"Total sequences: {len(self.dataset)}")

        # Train/val split
        train_size = int(self.config.train_split * len(self.dataset))
        val_size = len(self.dataset) - train_size

        from torch.utils.data import random_split
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )

        print(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def infer_constraint(self):
        """
        Infer trajectory constraint from data.

        This analyzes the ground truth trajectories to automatically
        detect the constraint type (circle, line, etc.).
        """
        print("\n" + "="*60)
        print("Inferring Trajectory Constraint")
        print("="*60)

        # Get all 3D positions from dataset
        positions = []
        for i in range(len(self.dataset)):
            seq = self.dataset[i]
            positions.append(seq['pos_3d_seq'])

        positions = np.concatenate(positions, axis=0)  # [N, 3]

        print(f"Analyzing {len(positions)} position samples...")

        # Detect constraint type
        if self.config.constraint_type == 'circle':
            self.constraint = self._fit_circle_constraint(positions)
        elif self.config.constraint_type == 'line':
            self.constraint = self._fit_line_constraint(positions)
        elif self.config.constraint_type == 'spline':
            self.constraint = self._fit_spline_constraint(positions)
        else:
            raise ValueError(f"Unknown constraint type: {self.config.constraint_type}")

        # Evaluate constraint fit
        distances = np.array([self.constraint.distance(p) for p in positions])
        mean_dist = distances.mean()
        max_dist = distances.max()

        print(f"Constraint type: {self.config.constraint_type}")
        print(f"Mean distance to constraint: {mean_dist:.4f} m")
        print(f"Max distance to constraint:  {max_dist:.4f} m")

        if mean_dist > 0.1:
            print("⚠ Warning: Poor constraint fit! Consider different constraint type.")

    def _fit_circle_constraint(self, positions: np.ndarray) -> CircleConstraint:
        """Fit circle to positions using least squares."""
        # Assume positions are in XY plane
        xy = positions[:, :2]  # [N, 2]
        z_mean = positions[:, 2].mean()

        # Fit circle: (x - cx)² + (y - cy)² = r²
        # Linear least squares approximation
        A = np.column_stack([xy, np.ones(len(xy))])
        b = (xy**2).sum(axis=1)

        # Solve: [cx, cy, c] = (A^T A)^(-1) A^T b
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        cx = params[0] / 2
        cy = params[1] / 2
        r = np.sqrt(params[2] + cx**2 + cy**2)

        center = np.array([cx, cy, z_mean])
        print(f"  Circle center: {center}")
        print(f"  Circle radius: {r:.4f} m")

        return CircleConstraint(
            center=center,
            radius=r,
            normal=np.array([0, 0, 1])
        )

    def _fit_line_constraint(self, positions: np.ndarray) -> LineConstraint:
        """Fit line to positions using PCA."""
        # Mean center
        mean_pos = positions.mean(axis=0)

        # PCA to find principal direction
        centered = positions - mean_pos
        num_points = max(len(centered), 1)
        cov = centered.T @ centered / num_points
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Principal direction (largest eigenvalue)
        direction = eigenvectors[:, -1]

        print(f"  Line point: {mean_pos}")
        print(f"  Line direction: {direction}")

        return LineConstraint(
            point=mean_pos,
            direction=direction
        )

    def _fit_spline_constraint(self, positions: np.ndarray) -> SplineConstraint:
        """Fit spline to positions."""
        # Sort positions along trajectory (heuristic: by angle)
        center = positions.mean(axis=0)
        angles = np.arctan2(
            positions[:, 1] - center[1],
            positions[:, 0] - center[0]
        )
        sorted_indices = np.argsort(angles)
        sorted_positions = positions[sorted_indices]

        # Sample control points
        num_control_points = 8
        indices = np.linspace(0, len(sorted_positions) - 1, num_control_points, dtype=int)
        control_points = sorted_positions[indices]

        print(f"  Spline control points: {len(control_points)}")

        return SplineConstraint(control_points)

    def train(self):
        """Train probabilistic LSTM."""
        print("\n" + "="*60)
        print("Training Probabilistic LSTM")
        print("="*60)

        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        self.trainer = ProbabilisticTrainer(
            self.lstm,
            self.config.lstm_config,
            self.device
        )

        history = self.trainer.train(
            train_loader,
            val_loader,
            self.config.num_epochs
        )

        # Save model
        if self.config.save_models:
            model_path = os.path.join(self.config.output_dir, 'lstm_model.pt')
            torch.save(self.lstm.state_dict(), model_path)
            print(f"\n✓ Model saved: {model_path}")

        return history

    def predict(self,
                bbox_seq: torch.Tensor,
                camera_ids: torch.Tensor,
                mask: torch.Tensor,
                use_constraint: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction with optional constraint.

        Args:
            bbox_seq: [1, seq_len, num_cameras, 4]
            camera_ids: [1, seq_len, num_cameras]
            mask: [1, seq_len, num_cameras]
            use_constraint: Whether to apply constraint

        Returns:
            mean: [seq_len, 3] - predicted positions
            std: [seq_len, 3] - standard deviations
        """
        self.lstm.eval()
        with torch.no_grad():
            # LSTM prediction
            pred_mean, pred_logvar = self.lstm(bbox_seq, camera_ids, mask)
            pred_std = torch.exp(0.5 * pred_logvar)

            mean = pred_mean[0].cpu().numpy()
            std = pred_std[0].cpu().numpy()

        # Apply constraint if requested
        if use_constraint and self.constraint is not None:
            refined_mean = []
            refined_std = []

            for t in range(mean.shape[0]):
                # Create Gaussian distribution
                prior = GaussianDistribution(
                    mean=mean[t],
                    cov=np.diag(std[t]**2)
                )

                # Apply constraint
                posterior = self.constraint.constrain(
                    prior,
                    constraint_std_radial=self.config.constraint_std_radial
                )

                refined_mean.append(posterior.mean)
                refined_std.append(posterior.std)

            mean = np.array(refined_mean)
            std = np.array(refined_std)

        return mean, std

    def evaluate_scenarios(self) -> Dict[str, Dict]:
        """
        Evaluate different tracking scenarios.

        Scenarios:
        1. 4 cameras, no constraint
        2. 2 cameras, no constraint
        3. 1 camera, no constraint
        4. 1 camera, with constraint

        Returns:
            results: Dictionary of scenario results
        """
        print("\n" + "="*60)
        print("Evaluating Scenarios")
        print("="*60)

        from torch.utils.data import DataLoader

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False
        )

        results = {}

        for scenario in self.config.eval_scenarios:
            print(f"\nScenario: {scenario}")
            print("-" * 40)

            # Parse scenario
            num_cams = int(scenario[0])  # e.g., '4' from '4cam_...'
            use_constraint = 'with_constraint' in scenario

            # Evaluate
            metrics = self._evaluate_scenario(
                val_loader,
                num_cameras=num_cams,
                use_constraint=use_constraint
            )

            results[scenario] = metrics

            # Print metrics
            print(f"  MAE:         {metrics['mae']:.4f} m")
            print(f"  RMSE:        {metrics['rmse']:.4f} m")
            print(f"  Avg Std:     {metrics['avg_std']:.4f} m")
            print(f"  Calibration: {metrics['calibration']:.2f}%")

        return results

    def _evaluate_scenario(self,
                          val_loader,
                          num_cameras: int,
                          use_constraint: bool) -> Dict:
        """Evaluate a single scenario."""
        maes = []
        rmses = []
        stds = []
        calibrations = []

        for batch in val_loader:
            bbox_seq = batch['bbox_seq'].to(self.device)
            pos_3d_seq = batch['pos_3d_seq'].numpy()[0]
            camera_ids = batch['camera_ids'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Simulate limited cameras by masking
            if num_cameras < bbox_seq.shape[2]:
                # Keep only first num_cameras cameras
                mask[:, :, num_cameras:] = True

            # Predict
            pred_mean, pred_std = self.predict(
                bbox_seq,
                camera_ids,
                mask,
                use_constraint=use_constraint
            )

            # Compute metrics
            error = np.abs(pred_mean - pos_3d_seq)
            mae = error.mean()
            rmse = np.sqrt((error**2).mean())
            avg_std = pred_std.mean()

            # Calibration: % of ground truth within 2σ
            within_2sigma = np.abs(pred_mean - pos_3d_seq) < 2 * pred_std
            calibration = within_2sigma.mean() * 100

            maes.append(mae)
            rmses.append(rmse)
            stds.append(avg_std)
            calibrations.append(calibration)

        return {
            'mae': np.mean(maes),
            'rmse': np.mean(rmses),
            'avg_std': np.mean(stds),
            'calibration': np.mean(calibrations)
        }

    def visualize_results(self, results: Dict[str, Dict]):
        """Visualize scenario comparison."""
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)

        scenarios = list(results.keys())
        metrics = ['mae', 'rmse', 'avg_std', 'calibration']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            values = [results[s][metric] for s in scenarios]

            # Bar plot
            bars = ax.bar(range(len(scenarios)), values)

            # Color code
            for i, bar in enumerate(bars):
                if 'with_constraint' in scenarios[i]:
                    bar.set_color('green')
                else:
                    bar.set_color('blue')

            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if self.config.save_plots:
            plot_path = os.path.join(self.config.output_dir, 'scenario_comparison.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved: {plot_path}")

        plt.close()

    def run_full_pipeline(self):
        """Run complete integrated pipeline."""
        print("\n" + "="*80)
        print("  Path2 Integrated Pipeline - End-to-End Tracking")
        print("="*80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Infer constraint
        if self.config.use_constraint:
            self.infer_constraint()

        # Step 3: Train LSTM
        history = self.train()

        # Step 4: Evaluate scenarios
        results = self.evaluate_scenarios()

        # Step 5: Visualize
        self.visualize_results(results)

        # Summary
        print("\n" + "="*80)
        print("  Pipeline Complete!")
        print("="*80)

        print("\nBest Scenario:")
        best_scenario = min(results.keys(), key=lambda s: results[s]['mae'])
        print(f"  {best_scenario}")
        print(f"  MAE: {results[best_scenario]['mae']:.4f} m")

        print("\nConstraint Impact:")
        # Compare 1cam without vs with constraint
        if '1cam_no_constraint' in results and '1cam_with_constraint' in results:
            mae_without = results['1cam_no_constraint']['mae']
            mae_with = results['1cam_with_constraint']['mae']
            improvement = (mae_without - mae_with) / mae_without * 100
            print(f"  1-cam MAE without constraint: {mae_without:.4f} m")
            print(f"  1-cam MAE with constraint:    {mae_with:.4f} m")
            print(f"  Improvement: {improvement:.1f}%")

        return results


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function for integrated pipeline."""
    # Configuration
    config = IntegratedConfig(
        data_dir='output/data',
        seq_len=10,
        max_cameras=4,
        use_constraint=True,
        constraint_type='circle',
        num_epochs=30,  # Reduced for faster testing
        batch_size=32,
        output_dir='output/integrated'
    )

    # Create tracker
    tracker = IntegratedTracker(config)

    # Run pipeline
    results = tracker.run_full_pipeline()

    return tracker, results


if __name__ == '__main__':
    tracker, results = main()
