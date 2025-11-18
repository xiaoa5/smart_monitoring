"""
Colab-friendly code snippet for testing predictions
Copy and paste this after training your model in Colab
"""

# ============================================================================
# Example 1: Get data from validation dataset
# ============================================================================

print("="*60)
print("Getting test data from dataset")
print("="*60)

# Get a single sample from validation dataset
sample = val_dataset[0]  # Get first sample

bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)  # [1, seq_len, num_cams, 4]
camera_ids = sample['camera_ids'].unsqueeze(0).to(device)  # [1, seq_len, num_cams]
mask = sample['mask'].unsqueeze(0).to(device)  # [1, seq_len, num_cams]
ground_truth = sample['pos_3d_seq'].cpu().numpy()  # [seq_len, 3]

print(f"✓ Sample loaded:")
print(f"  bbox_seq shape: {bbox_seq.shape}")
print(f"  camera_ids shape: {camera_ids.shape}")
print(f"  mask shape: {mask.shape}")
print(f"  ground_truth shape: {ground_truth.shape}")

# ============================================================================
# Example 2: Make prediction
# ============================================================================

print("\n" + "="*60)
print("Making prediction")
print("="*60)

# Predict distribution
mean, std = model.predict_distribution(bbox_seq, camera_ids, mask)

print(f"\n✓ Prediction complete:")
print(f"  mean shape: {mean.shape}")  # [seq_len, 3]
print(f"  std shape: {std.shape}")    # [seq_len, 3]

# Show results for last timestep
print(f"\nLast timestep results:")
print(f"  Predicted position: {mean[-1]}")
print(f"  Uncertainty (σ):    {std[-1]}")
print(f"  Ground truth:       {ground_truth[-1]}")
print(f"  Error:              {np.abs(mean[-1] - ground_truth[-1])}")

# ============================================================================
# Example 3: Calculate metrics
# ============================================================================

print("\n" + "="*60)
print("Calculating metrics")
print("="*60)

mae = np.abs(mean - ground_truth).mean()
rmse = np.sqrt(((mean - ground_truth) ** 2).mean())
max_error = np.abs(mean - ground_truth).max()

print(f"\nPrediction Metrics:")
print(f"  MAE (Mean Absolute Error): {mae:.6f} m")
print(f"  RMSE (Root Mean Square):   {rmse:.6f} m")
print(f"  Max Error:                 {max_error:.6f} m")

# ============================================================================
# Example 4: Visualize trajectory
# ============================================================================

print("\n" + "="*60)
print("Visualizing trajectory")
print("="*60)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 5))

# 3D trajectory
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(mean[:, 0], mean[:, 1], mean[:, 2], 'b-', linewidth=2, label='Predicted')
ax1.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
         'g--', linewidth=2, label='Ground Truth')
ax1.scatter(mean[-1, 0], mean[-1, 1], mean[-1, 2], c='blue', s=100, marker='o')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D Trajectory')
ax1.legend()

# X-Y plane
ax2 = fig.add_subplot(132)
ax2.plot(mean[:, 0], mean[:, 1], 'b-', linewidth=2, label='Predicted')
ax2.plot(ground_truth[:, 0], ground_truth[:, 1], 'g--', linewidth=2, label='Ground Truth')
ax2.scatter(mean[-1, 0], mean[-1, 1], c='blue', s=100, marker='o')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('X-Y Plane')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axis('equal')

# Uncertainty over time
ax3 = fig.add_subplot(133)
time_steps = np.arange(len(mean))
uncertainty_magnitude = np.linalg.norm(std, axis=1)
ax3.plot(time_steps, uncertainty_magnitude, 'r-', linewidth=2)
ax3.fill_between(time_steps, 0, uncertainty_magnitude, alpha=0.3, color='red')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Uncertainty (m)')
ax3.set_title('Prediction Uncertainty')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Visualization complete!")

# ============================================================================
# Example 5: Test on multiple samples
# ============================================================================

print("\n" + "="*60)
print("Testing on multiple samples")
print("="*60)

num_samples = min(5, len(val_dataset))
errors = []

for i in range(num_samples):
    sample = val_dataset[i]
    bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)
    camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
    mask = sample['mask'].unsqueeze(0).to(device)
    ground_truth = sample['pos_3d_seq'].cpu().numpy()

    mean, std = model.predict_distribution(bbox_seq, camera_ids, mask)

    mae = np.abs(mean - ground_truth).mean()
    errors.append(mae)

    print(f"  Sample {i}: MAE = {mae:.6f} m, Avg Uncertainty = {std.mean():.6f} m")

print(f"\nOverall Statistics:")
print(f"  Mean MAE: {np.mean(errors):.6f} m")
print(f"  Std MAE:  {np.std(errors):.6f} m")

print("\n" + "="*60)
print("✓ All tests complete!")
print("="*60)
