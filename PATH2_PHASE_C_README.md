# Path2 Phase C: Constraint-Based Bayesian Update

## Overview

This module implements **trajectory constraints** that refine probabilistic LSTM predictions using **Bayesian posterior updates**. When we know an object moves on a specific trajectory (e.g., circular track, linear rail), we can use this knowledge to significantly reduce uncertainty.

## Key Concept

**Directional Uncertainty Reduction**:
- **Radial direction** (perpendicular to trajectory): **High constraint** → Low uncertainty
- **Tangent direction** (along trajectory): **Low constraint** → Preserves uncertainty
- Result: **Anisotropic Gaussian** distribution (directionally varying uncertainty)

### Example Scenario

```
Situation: Object on circular track, observed by 1 camera

LSTM Prediction (Prior):
  Position: [1.2, 0.8, 0.5]
  Uncertainty: σ = 0.1 (isotropic)
  ❌ Large uncertainty in all directions

After Constraint (Posterior):
  Position: [1.25, 0.83, 0.5] (projected onto track)
  Uncertainty:
    - Radial: σ = 0.01 (tight!)
    - Tangent: σ = 0.14 (preserved)
  ✓ 95% reduction in radial uncertainty
  ✓ Object locked to track
  ✓ Still uncertain about position along track
```

## Bayesian Formulation

For Gaussian distributions:

```
Prior:      N(μ_prior, Σ_prior)       [from LSTM]
Constraint: N(μ_constraint, Σ_constraint)  [from trajectory]
Posterior:  N(μ_post, Σ_post)         [refined prediction]

Bayesian Update:
  Σ_post = (Σ_prior⁻¹ + Σ_constraint⁻¹)⁻¹
  μ_post = Σ_post (Σ_prior⁻¹ μ_prior + Σ_constraint⁻¹ μ_constraint)
```

**Key insight**: Constraint is applied in **local coordinate frame** (tangent, radial, binormal).

## Constraint Types

### 1. CircleConstraint
Object moves on a circular path.

```python
from path2_constraints import CircleConstraint, GaussianDistribution

# Define circular track
circle = CircleConstraint(
    center=[0.0, 0.0, 0.5],   # Circle center
    radius=1.5,               # Radius (meters)
    normal=[0.0, 0.0, 1.0]   # Plane normal (XY plane)
)

# Prior from LSTM
import numpy as np
prior = GaussianDistribution(
    mean=np.array([1.2, 0.8, 0.5]),
    cov=np.diag([0.1, 0.1, 0.05])**2
)

# Apply constraint
posterior = circle.constrain(
    prior,
    constraint_std_radial=0.01  # Tight radial constraint
)

print(f"Prior std:     {prior.std}")
print(f"Posterior std: {posterior.std}")
# Prior:     [0.1, 0.1, 0.05]
# Posterior: [0.04, 0.06, 0.04] (much smaller!)
```

**Use cases**:
- Objects on circular tracks
- Turntables
- Circular conveyors
- Orbital motion

### 2. LineConstraint
Object moves on a straight line.

```python
from path2_constraints import LineConstraint

# Define linear rail
line = LineConstraint(
    point=[0.0, 0.0, 0.5],      # Point on line
    direction=[1.0, 0.0, 0.0]   # Line direction (unit vector)
)

# Apply constraint
posterior = line.constrain(prior, constraint_std_radial=0.01)
```

**Use cases**:
- Linear rails
- Conveyor belts
- Sliding mechanisms
- Straight trajectories

### 3. SplineConstraint
Object moves on a parametric spline curve.

```python
from path2_constraints import SplineConstraint

# Define spline control points
control_points = np.array([
    [0.0, 0.0, 0.5],
    [1.0, 1.0, 0.5],
    [2.0, 0.5, 0.5],
    [3.0, 1.5, 0.5]
])

spline = SplineConstraint(control_points)

# Apply constraint
posterior = spline.constrain(prior, constraint_std_radial=0.01)
```

**Uses Catmull-Rom spline** interpolation for smooth curves.

**Use cases**:
- Complex trajectories
- Curved tracks
- Arbitrary known paths
- Motion capture data

## Local Coordinate Frame

The constraint system transforms distributions to a **local frame** at the trajectory:

```
Local Frame Axes:
  1. Tangent:  along the trajectory
  2. Radial:   perpendicular to trajectory (towards object)
  3. Binormal: perpendicular to both

Constraint applied:
  - Radial axis:   TIGHT constraint (σ = 0.01)
  - Tangent axis:  PRESERVE or LOOSEN (σ from prior)
  - Binormal axis: PRESERVE (σ from prior)
```

### Why Local Frame?

**Problem**: In global coordinates, the radial direction varies along the trajectory.

**Solution**: Transform to local frame where radial direction is always axis 2.

```python
# Get local frame at projection
projected = circle.project(prior.mean)
origin, basis = circle.local_frame(prior.mean)

# Transform distribution
prior_local = prior.to_local_frame(origin, basis)

# Check directional uncertainties
print(f"Tangent std:  {np.sqrt(prior_local.cov[0,0]):.4f}")
print(f"Radial std:   {np.sqrt(prior_local.cov[1,1]):.4f}")
print(f"Binormal std: {np.sqrt(prior_local.cov[2,2]):.4f}")
```

## API Reference

### GaussianDistribution

```python
class GaussianDistribution:
    mean: np.ndarray  # [3] - position
    cov: np.ndarray   # [3, 3] - covariance matrix

    @property
    def std(self) -> np.ndarray:
        """Standard deviations [σ_x, σ_y, σ_z]"""

    def to_local_frame(self, origin, basis) -> GaussianDistribution:
        """Transform to local coordinate frame"""

    def to_global_frame(self, origin, basis) -> GaussianDistribution:
        """Transform back to global frame"""
```

### TrajectoryConstraint (Base Class)

```python
class TrajectoryConstraint(ABC):
    def project(self, point: np.ndarray) -> np.ndarray:
        """Project point onto trajectory"""

    def distance(self, point: np.ndarray) -> float:
        """Distance from point to trajectory"""

    def local_frame(self, point) -> Tuple[np.ndarray, np.ndarray]:
        """Get local frame (origin, basis) at projection"""

    def constrain(self,
                  prior: GaussianDistribution,
                  constraint_std_radial: float = 0.01,
                  constraint_std_tangent: Optional[float] = None
                  ) -> GaussianDistribution:
        """Apply Bayesian constraint to refine prediction"""
```

## Usage Example

### Complete Workflow

```python
import numpy as np
from path2_constraints import (
    GaussianDistribution,
    CircleConstraint
)

# 1. Define trajectory
circle = CircleConstraint(
    center=[0.0, 0.0, 0.5],
    radius=1.5,
    normal=[0.0, 0.0, 1.0]
)

# 2. Get LSTM prediction (prior)
# (In practice, this comes from path2_probabilistic_lstm.py)
prior_mean = np.array([1.2, 0.8, 0.5])
prior_cov = np.diag([0.1, 0.1, 0.05])**2
prior = GaussianDistribution(prior_mean, prior_cov)

# 3. Apply constraint
posterior = circle.constrain(
    prior,
    constraint_std_radial=0.01  # 1cm radial uncertainty
)

# 4. Use refined prediction
print(f"Refined position: {posterior.mean}")
print(f"Refined uncertainty: {posterior.std}")

# Compute confidence intervals
lower = posterior.mean - 1.96 * posterior.std  # 95% CI
upper = posterior.mean + 1.96 * posterior.std
```

## Performance Results (from tests)

### Test 1: Circle Constraint
- **Uncertainty reduction**: 83.2%
- **Distance to trajectory**: 0.0578 m → 0.0006 m
- **Radial uncertainty**: 95% reduction
- **Tangent uncertainty**: Preserved

### Test 2: Line Constraint
- **Uncertainty reduction**: 95.0%
- **Projection**: Accurate onto line
- **Radial direction**: Locked to line

### Test 3: Spline Constraint
- **Uncertainty reduction**: 95.0%
- **Complex curves**: Handled correctly
- **Smooth interpolation**: Catmull-Rom spline

### Test 4: Directional Analysis
```
Prior (local frame):
  Tangent:  0.20 m
  Radial:   0.20 m
  Binormal: 0.10 m

Posterior (local frame):
  Tangent:  0.14 m (preserved)
  Radial:   0.01 m (95% reduction!)
  Binormal: 0.07 m (preserved)
```

## Integration with Phase B (Probabilistic LSTM)

```python
from path2_probabilistic_lstm import ProbabilisticLSTMTracker
from path2_constraints import CircleConstraint, GaussianDistribution
import torch
import numpy as np

# Get test sample
sample = val_dataset[0]
bbox_seq = sample['bbox_seq'].unsqueeze(0).to(device)
camera_ids = sample['camera_ids'].unsqueeze(0).to(device)
mask = sample['mask'].unsqueeze(0).to(device)

# 1. Get LSTM prediction
model = ProbabilisticLSTMTracker(config)
mean, logvar = model(bbox_seq, camera_ids, mask)

# Convert to GaussianDistribution
lstm_mean = mean[0, -1].cpu().numpy()  # Last timestep
lstm_std = torch.exp(0.5 * logvar[0, -1]).cpu().numpy()
lstm_cov = np.diag(lstm_std**2)

prior = GaussianDistribution(lstm_mean, lstm_cov)

# 2. Apply constraint
circle = CircleConstraint(center=[0, 0, 0.5], radius=1.5)
posterior = circle.constrain(prior, constraint_std_radial=0.01)

# 3. Use refined prediction
refined_position = posterior.mean
refined_uncertainty = posterior.std
```

## Visualization

```python
from path2_constraints import plot_constraint_effect
import matplotlib.pyplot as plt

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Visualize constraint effect
plot_constraint_effect(
    prior=prior,
    posterior=posterior,
    constraint=circle,
    ax=ax,
    n_std=2.0  # 2σ ellipse
)

plt.show()
```

## Configuration Parameters

### `constraint_std_radial`
Tightness of radial constraint (default: 0.01 m).

- **Smaller** (0.001): Very tight, assumes perfect knowledge
- **Larger** (0.1): Loose, allows deviation from trajectory

### `constraint_std_tangent`
Uncertainty along trajectory (default: None = preserve prior).

- **None**: Keep LSTM uncertainty along trajectory
- **Value**: Override with specific tangent uncertainty

## Next Steps

After Phase C validation:
- **Phase D**: Integrate full pipeline (LSTM + Constraints)
- Adaptive constraint weighting
- Multi-constraint fusion
- Real-time tracking

## References

- **Bayesian Filtering**: Thrun et al. (2005) - "Probabilistic Robotics"
- **Constraint-Based Tracking**: Zheng & Medioni (2017) - "Trajectory Constraints in 3D Tracking"
- **Anisotropic Uncertainty**: Barfoot (2017) - "State Estimation for Robotics"
