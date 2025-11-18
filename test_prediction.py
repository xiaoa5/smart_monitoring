#!/usr/bin/env python3
"""
Test script for Path2 Probabilistic LSTM prediction
Demonstrates how to:
1. Load the trained model
2. Get data from dataset
3. Make predictions
4. Visualize results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from path2_probabilistic_lstm import (
    ProbabilisticLSTMTracker,
    MultiCameraDataset,
    LSTMConfig
)


def visualize_prediction(mean, std, ground_truth=None):
    """
    Visualize predicted 3D trajectory with uncertainty.

    Args:
        mean: [seq_len, 3] - predicted positions
        std: [seq_len, 3] - standard deviations
        ground_truth: [seq_len, 3] - optional ground truth positions
    """
    seq_len = len(mean)
    time_steps = np.arange(seq_len)

    fig = plt.figure(figsize=(15, 10))

    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(mean[:, 0], mean[:, 1], mean[:, 2], 'b-', linewidth=2, label='Predicted')

    # Plot uncertainty as error bars
    for i in range(0, seq_len, max(1, seq_len // 10)):
        ax1.plot([mean[i, 0] - std[i, 0], mean[i, 0] + std[i, 0]],
                [mean[i, 1], mean[i, 1]],
                [mean[i, 2], mean[i, 2]], 'r-', alpha=0.3)

    if ground_truth is not None:
        ax1.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                'g--', linewidth=2, label='Ground Truth')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Prediction')
    ax1.legend()

    # X, Y, Z components over time
    dims = ['X', 'Y', 'Z']
    for i, dim in enumerate(dims):
        ax = fig.add_subplot(2, 3, i + 4)

        # Plot mean
        ax.plot(time_steps, mean[:, i], 'b-', linewidth=2, label='Predicted')

        # Plot uncertainty band
        ax.fill_between(time_steps,
                        mean[:, i] - std[:, i],
                        mean[:, i] + std[:, i],
                        alpha=0.3, color='blue', label='±1σ')

        if ground_truth is not None:
            ax.plot(time_steps, ground_truth[:, i], 'g--', linewidth=2, label='Ground Truth')

        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'{dim} Position (m)')
        ax.set_title(f'{dim} Component Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def test_single_prediction(model, dataset, sample_idx=0, device='cpu'):
    """
    Test prediction on a single sample from dataset.

    Args:
        model: Trained ProbabilisticLSTMTracker
        dataset: MultiCameraDataset
        sample_idx: Index of sample to test
        device: 'cuda' or 'cpu'

    Returns:
        mean, std, ground_truth
    """
    # Get sample from dataset
    sample = dataset[sample_idx]

    # Extract data
    bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)  # Add batch dim
    camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)
    ground_truth = sample['pos_3d_seq'].numpy()

    # Predict
    model.eval()
    model = model.to(device)
    mean, std = model.predict_distribution(bbox_seq, camera_ids, mask)

    return mean, std, ground_truth


def test_batch_prediction(model, val_loader, device='cpu'):
    """
    Test prediction on a batch from validation loader.

    Args:
        model: Trained ProbabilisticLSTMTracker
        val_loader: Validation DataLoader
        device: 'cuda' or 'cpu'

    Returns:
        List of (mean, std, ground_truth) tuples
    """
    model.eval()
    model = model.to(device)

    results = []

    # Get first batch
    batch = next(iter(val_loader))
    bbox_seq = batch['bbox_seq'].to(device)
    camera_ids = batch['camera_ids'].to(device)
    mask = batch['mask'].to(device)
    ground_truth = batch['pos_3d_seq'].cpu().numpy()

    # Predict
    with torch.no_grad():
        pred_mean, pred_logvar = model(bbox_seq, camera_ids, mask)
        pred_std = torch.exp(0.5 * pred_logvar)

    # Convert to numpy
    pred_mean = pred_mean.cpu().numpy()
    pred_std = pred_std.cpu().numpy()

    # Store results for each sample in batch
    for i in range(len(pred_mean)):
        results.append((pred_mean[i], pred_std[i], ground_truth[i]))

    return results


def calculate_metrics(mean, ground_truth):
    """
    Calculate prediction metrics.

    Args:
        mean: [seq_len, 3] - predicted positions
        ground_truth: [seq_len, 3] - ground truth positions

    Returns:
        dict with metrics
    """
    mae = np.abs(mean - ground_truth).mean()
    rmse = np.sqrt(((mean - ground_truth) ** 2).mean())
    max_error = np.abs(mean - ground_truth).max()

    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error
    }


if __name__ == '__main__':
    print("="*60)
    print("Path2 Probabilistic LSTM - Prediction Test")
    print("="*60)

    # Configuration
    config = LSTMConfig(
        seq_len=10,
        max_cameras=4
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = MultiCameraDataset(
            json_dir='output/data',
            seq_len=config.seq_len,
            max_cameras=config.max_cameras
        )
        print(f"✓ Loaded {len(dataset)} sequences")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease run data generation first:")
        print("  python path2_phase1_2_verified.py")
        exit(1)

    # Create model
    print("\nCreating model...")
    model = ProbabilisticLSTMTracker(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test prediction on single sample
    print("\n" + "="*60)
    print("Testing Single Sample Prediction")
    print("="*60)

    sample_idx = 0
    mean, std, ground_truth = test_single_prediction(model, dataset, sample_idx, device)

    print(f"\nSample {sample_idx}:")
    print(f"  Sequence length: {len(mean)}")
    print(f"  Predicted position (last step): {mean[-1]}")
    print(f"  Uncertainty (last step): {std[-1]}")
    print(f"  Ground truth (last step): {ground_truth[-1]}")

    # Calculate metrics
    metrics = calculate_metrics(mean, ground_truth)
    print(f"\nMetrics:")
    print(f"  MAE:  {metrics['mae']:.6f} m")
    print(f"  RMSE: {metrics['rmse']:.6f} m")
    print(f"  Max Error: {metrics['max_error']:.6f} m")

    # Visualize
    print("\nGenerating visualization...")
    visualize_prediction(mean, std, ground_truth)

    print("\n" + "="*60)
    print("✓ Prediction test complete!")
    print("="*60)
