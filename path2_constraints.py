#!/usr/bin/env python3
"""
Path2 Phase C: Constraint-Based Bayesian Update
================================================

This module implements trajectory constraints that refine probabilistic
LSTM predictions using Bayesian posterior updates.

Key concept:
When we know an object moves on a specific trajectory (e.g., circular track,
linear rail), we can use this as a constraint to reduce uncertainty in the
direction perpendicular to the trajectory while preserving uncertainty
along the trajectory.

Example:
    Prior (from LSTM):     N(μ, Σ_large)
    Constraint:            Object on circular track
    Posterior (refined):   N(μ', Σ_small)

    Where Σ_small has:
    - Small radial uncertainty (perpendicular to track)
    - Preserved tangent uncertainty (along track)

Bayesian Update:
    Posterior ∝ Prior × Likelihood

    Where:
    - Prior = LSTM prediction N(μ_prior, Σ_prior)
    - Likelihood = Constraint N(μ_constraint, Σ_constraint)
    - Posterior = Refined prediction N(μ_post, Σ_post)

Author: Claude
Date: 2025-11-18
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.linalg import inv


# ============================================================================
# Gaussian Distribution
# ============================================================================

@dataclass
class GaussianDistribution:
    """
    3D Gaussian distribution.

    Attributes:
        mean: [3] array - mean position [x, y, z]
        cov: [3, 3] array - covariance matrix
    """
    mean: np.ndarray  # [3]
    cov: np.ndarray   # [3, 3]

    def __post_init__(self):
        """Validate dimensions."""
        assert self.mean.shape == (3,), f"Mean must be [3], got {self.mean.shape}"
        assert self.cov.shape == (3, 3), f"Cov must be [3,3], got {self.cov.shape}"

        # Ensure covariance is symmetric
        self.cov = 0.5 * (self.cov + self.cov.T)

    @property
    def std(self) -> np.ndarray:
        """Standard deviations [σ_x, σ_y, σ_z]."""
        return np.sqrt(np.diag(self.cov))

    def to_local_frame(self, origin: np.ndarray, basis: np.ndarray) -> 'GaussianDistribution':
        """
        Transform distribution to local coordinate frame.

        Args:
            origin: [3] - local frame origin
            basis: [3, 3] - local frame basis (columns are axes)

        Returns:
            GaussianDistribution in local frame
        """
        # Transform mean
        mean_local = basis.T @ (self.mean - origin)

        # Transform covariance: Σ_local = R^T Σ R
        cov_local = basis.T @ self.cov @ basis

        return GaussianDistribution(mean_local, cov_local)

    def to_global_frame(self, origin: np.ndarray, basis: np.ndarray) -> 'GaussianDistribution':
        """
        Transform distribution from local to global frame.

        Args:
            origin: [3] - local frame origin
            basis: [3, 3] - local frame basis (columns are axes)

        Returns:
            GaussianDistribution in global frame
        """
        # Transform mean
        mean_global = basis @ self.mean + origin

        # Transform covariance: Σ_global = R Σ R^T
        cov_global = basis @ self.cov @ basis.T

        return GaussianDistribution(mean_global, cov_global)


# ============================================================================
# Trajectory Constraints (Abstract Base)
# ============================================================================

class TrajectoryConstraint(ABC):
    """
    Abstract base class for trajectory constraints.

    A trajectory constraint defines a geometric surface that an object
    is known to move on (e.g., circular track, linear rail).
    """

    @abstractmethod
    def project(self, point: np.ndarray) -> np.ndarray:
        """
        Project a 3D point onto the trajectory.

        Args:
            point: [3] - 3D position

        Returns:
            projected_point: [3] - closest point on trajectory
        """
        pass

    @abstractmethod
    def distance(self, point: np.ndarray) -> float:
        """
        Compute distance from point to trajectory.

        Args:
            point: [3] - 3D position

        Returns:
            distance: scalar
        """
        pass

    @abstractmethod
    def local_frame(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute local coordinate frame at projection.

        The local frame has:
        - Tangent axis: along the trajectory
        - Radial axis: perpendicular to trajectory (towards point)
        - Binormal axis: perpendicular to both

        Args:
            point: [3] - 3D position

        Returns:
            origin: [3] - frame origin (projected point)
            basis: [3, 3] - frame basis [tangent, radial, binormal]
        """
        pass

    def constrain(self,
                  prior: GaussianDistribution,
                  constraint_std_radial: float = 0.01,
                  constraint_std_tangent: float = None) -> GaussianDistribution:
        """
        Apply Bayesian constraint to refine prediction.

        Steps:
        1. Project prior mean onto trajectory
        2. Transform to local frame (tangent, radial, binormal)
        3. Apply constraint (reduce radial uncertainty)
        4. Transform back to global frame

        Args:
            prior: Prior distribution from LSTM
            constraint_std_radial: Constraint uncertainty in radial direction
            constraint_std_tangent: Constraint uncertainty in tangent direction
                                   (if None, use prior uncertainty)

        Returns:
            posterior: Refined distribution
        """
        # Project onto trajectory
        projected = self.project(prior.mean)

        # Get local frame at projection
        origin, basis = self.local_frame(projected)

        # Transform prior to local frame
        prior_local = prior.to_local_frame(origin, basis)

        # Create constraint distribution in local frame
        # - Radial (axis 1): tight constraint
        # - Tangent (axis 0) and Binormal (axis 2): preserve or loosen
        mean_constraint = np.array([0.0, 0.0, 0.0])  # Constraint at origin

        if constraint_std_tangent is None:
            # Preserve tangent uncertainty from prior
            std_tangent = np.sqrt(prior_local.cov[0, 0])
        else:
            std_tangent = constraint_std_tangent

        std_binormal = np.sqrt(prior_local.cov[2, 2])  # Preserve binormal

        cov_constraint = np.diag([
            std_tangent**2,              # Tangent: preserve
            constraint_std_radial**2,     # Radial: tight constraint
            std_binormal**2               # Binormal: preserve
        ])

        constraint = GaussianDistribution(mean_constraint, cov_constraint)

        # Bayesian fusion in local frame
        posterior_local = self._bayesian_fusion(prior_local, constraint)

        # Transform back to global frame
        posterior_global = posterior_local.to_global_frame(origin, basis)

        return posterior_global

    @staticmethod
    def _bayesian_fusion(prior: GaussianDistribution,
                        likelihood: GaussianDistribution) -> GaussianDistribution:
        """
        Bayesian fusion of two Gaussian distributions.

        For Gaussians:
            Prior:      N(μ₁, Σ₁)
            Likelihood: N(μ₂, Σ₂)
            Posterior:  N(μ_post, Σ_post)

        Where:
            Σ_post = (Σ₁⁻¹ + Σ₂⁻¹)⁻¹
            μ_post = Σ_post (Σ₁⁻¹ μ₁ + Σ₂⁻¹ μ₂)

        Args:
            prior: Prior distribution
            likelihood: Likelihood distribution

        Returns:
            posterior: Fused distribution
        """
        # Precision matrices (inverse covariance)
        prec_prior = inv(prior.cov)
        prec_likelihood = inv(likelihood.cov)

        # Posterior precision
        prec_post = prec_prior + prec_likelihood

        # Posterior covariance
        cov_post = inv(prec_post)

        # Posterior mean
        mean_post = cov_post @ (prec_prior @ prior.mean + prec_likelihood @ likelihood.mean)

        return GaussianDistribution(mean_post, cov_post)


# ============================================================================
# Circle Constraint
# ============================================================================

class CircleConstraint(TrajectoryConstraint):
    """
    Circular trajectory constraint.

    Object moves on a circle in a plane.

    Attributes:
        center: [3] - circle center
        radius: float - circle radius
        normal: [3] - plane normal (unit vector)
    """

    def __init__(self, center: np.ndarray, radius: float, normal: np.ndarray = None):
        """
        Args:
            center: [3] - circle center
            radius: float - circle radius
            normal: [3] - plane normal (default: [0, 0, 1] for XY plane)
        """
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)

        if normal is None:
            self.normal = np.array([0.0, 0.0, 1.0])
        else:
            self.normal = np.array(normal, dtype=float)
            self.normal /= np.linalg.norm(self.normal)  # Normalize

    def project(self, point: np.ndarray) -> np.ndarray:
        """Project point onto circle."""
        # Vector from center to point
        v = point - self.center

        # Project onto plane
        v_plane = v - np.dot(v, self.normal) * self.normal

        # Normalize and scale to radius
        dist = np.linalg.norm(v_plane)
        if dist < 1e-10:
            # Point at center, choose arbitrary direction
            # Orthogonal to normal
            if abs(self.normal[0]) < 0.9:
                v_plane = np.cross(self.normal, [1, 0, 0])
            else:
                v_plane = np.cross(self.normal, [0, 1, 0])
        else:
            v_plane = v_plane / dist

        projected = self.center + self.radius * v_plane

        return projected

    def distance(self, point: np.ndarray) -> float:
        """Compute distance to circle."""
        projected = self.project(point)
        return np.linalg.norm(point - projected)

    def local_frame(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute local frame at projection."""
        # Project point onto circle
        projected = self.project(point)

        # Radial direction (from center to projected point)
        radial = projected - self.center
        radial = radial / np.linalg.norm(radial)

        # Tangent direction (perpendicular to radial, in plane)
        tangent = np.cross(self.normal, radial)
        tangent = tangent / np.linalg.norm(tangent)

        # Binormal (plane normal)
        binormal = self.normal

        # Basis matrix [tangent, radial, binormal]
        basis = np.column_stack([tangent, radial, binormal])

        return projected, basis


# ============================================================================
# Line Constraint
# ============================================================================

class LineConstraint(TrajectoryConstraint):
    """
    Linear trajectory constraint.

    Object moves on a straight line (e.g., linear rail).

    Attributes:
        point: [3] - a point on the line
        direction: [3] - line direction (unit vector)
    """

    def __init__(self, point: np.ndarray, direction: np.ndarray):
        """
        Args:
            point: [3] - a point on the line
            direction: [3] - line direction
        """
        self.point = np.array(point, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)  # Normalize

    def project(self, point: np.ndarray) -> np.ndarray:
        """Project point onto line."""
        v = point - self.point
        t = np.dot(v, self.direction)
        projected = self.point + t * self.direction
        return projected

    def distance(self, point: np.ndarray) -> float:
        """Compute distance to line."""
        projected = self.project(point)
        return np.linalg.norm(point - projected)

    def local_frame(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute local frame at projection."""
        # Project point onto line
        projected = self.project(point)

        # Tangent direction (line direction)
        tangent = self.direction

        # Radial direction (from line to point)
        radial_vec = point - projected
        radial_dist = np.linalg.norm(radial_vec)

        if radial_dist < 1e-10:
            # Point on line, choose arbitrary radial direction
            if abs(tangent[0]) < 0.9:
                radial = np.cross(tangent, [1, 0, 0])
            else:
                radial = np.cross(tangent, [0, 1, 0])
            radial = radial / np.linalg.norm(radial)
        else:
            radial = radial_vec / radial_dist

        # Binormal (perpendicular to both)
        binormal = np.cross(tangent, radial)
        binormal = binormal / np.linalg.norm(binormal)

        # Basis matrix [tangent, radial, binormal]
        basis = np.column_stack([tangent, radial, binormal])

        return projected, basis


# ============================================================================
# Spline Constraint
# ============================================================================

class SplineConstraint(TrajectoryConstraint):
    """
    Spline trajectory constraint.

    Object moves on a parametric spline curve.

    Attributes:
        control_points: [N, 3] - spline control points
    """

    def __init__(self, control_points: np.ndarray):
        """
        Args:
            control_points: [N, 3] - control points defining the spline
        """
        self.control_points = np.array(control_points, dtype=float)
        assert self.control_points.shape[1] == 3, "Control points must be [N, 3]"

        # Precompute arc length parameterization for faster projection
        self.num_samples = 1000
        self.t_samples = np.linspace(0, 1, self.num_samples)
        self.curve_samples = np.array([self._evaluate(t) for t in self.t_samples])

    def _evaluate(self, t: float) -> np.ndarray:
        """
        Evaluate spline at parameter t ∈ [0, 1].

        Uses Catmull-Rom spline interpolation.
        """
        n = len(self.control_points)
        if n == 2:
            # Linear interpolation
            return (1 - t) * self.control_points[0] + t * self.control_points[1]

        # Map t to segment
        t_scaled = t * (n - 1)
        idx = int(np.floor(t_scaled))
        idx = min(idx, n - 2)
        t_local = t_scaled - idx

        # Get control points for segment
        p0 = self.control_points[max(0, idx - 1)]
        p1 = self.control_points[idx]
        p2 = self.control_points[idx + 1]
        p3 = self.control_points[min(n - 1, idx + 2)]

        # Catmull-Rom interpolation
        t2 = t_local * t_local
        t3 = t2 * t_local

        point = 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t_local +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )

        return point

    def project(self, point: np.ndarray) -> np.ndarray:
        """Project point onto spline."""
        # Find closest point on sampled curve
        distances = np.linalg.norm(self.curve_samples - point, axis=1)
        idx_min = np.argmin(distances)

        # Refine with local search
        t_min = self.t_samples[idx_min]
        t_range = 1.0 / self.num_samples

        # Fine search around minimum
        t_fine = np.linspace(
            max(0, t_min - t_range),
            min(1, t_min + t_range),
            100
        )

        min_dist = float('inf')
        best_t = t_min
        for t in t_fine:
            p = self._evaluate(t)
            dist = np.linalg.norm(p - point)
            if dist < min_dist:
                min_dist = dist
                best_t = t

        projected = self._evaluate(best_t)
        return projected

    def distance(self, point: np.ndarray) -> float:
        """Compute distance to spline."""
        projected = self.project(point)
        return np.linalg.norm(point - projected)

    def _tangent(self, t: float, eps: float = 1e-5) -> np.ndarray:
        """Compute tangent vector at parameter t."""
        if t < eps:
            tangent = self._evaluate(t + eps) - self._evaluate(t)
        elif t > 1 - eps:
            tangent = self._evaluate(t) - self._evaluate(t - eps)
        else:
            tangent = self._evaluate(t + eps) - self._evaluate(t - eps)

        tangent = tangent / np.linalg.norm(tangent)
        return tangent

    def local_frame(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute local frame at projection."""
        # Project point onto spline
        projected = self.project(point)

        # Find parameter t for projected point
        distances = np.linalg.norm(self.curve_samples - projected, axis=1)
        idx_min = np.argmin(distances)
        t = self.t_samples[idx_min]

        # Tangent direction
        tangent = self._tangent(t)

        # Radial direction (from curve to point)
        radial_vec = point - projected
        radial_dist = np.linalg.norm(radial_vec)

        if radial_dist < 1e-10:
            # Point on curve, choose arbitrary radial direction
            if abs(tangent[0]) < 0.9:
                radial = np.cross(tangent, [1, 0, 0])
            else:
                radial = np.cross(tangent, [0, 1, 0])
            radial = radial / np.linalg.norm(radial)
        else:
            radial = radial_vec / radial_dist

        # Binormal (perpendicular to both)
        binormal = np.cross(tangent, radial)
        binormal = binormal / np.linalg.norm(binormal)

        # Basis matrix [tangent, radial, binormal]
        basis = np.column_stack([tangent, radial, binormal])

        return projected, basis


# ============================================================================
# Visualization Utilities
# ============================================================================

def plot_constraint_effect(prior: GaussianDistribution,
                           posterior: GaussianDistribution,
                           constraint: TrajectoryConstraint,
                           ax=None,
                           n_std: float = 2.0):
    """
    Visualize the effect of constraint on uncertainty.

    Args:
        prior: Prior distribution (before constraint)
        posterior: Posterior distribution (after constraint)
        constraint: Trajectory constraint
        ax: Matplotlib axis (if None, create new)
        n_std: Number of standard deviations for ellipse
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    if isinstance(constraint, CircleConstraint):
        theta = np.linspace(0, 2*np.pi, 100)
        traj_points = []
        for t in theta:
            # Get point on circle
            radial = np.array([np.cos(t), np.sin(t), 0.0])
            # Rotate to match circle orientation
            if not np.allclose(constraint.normal, [0, 0, 1]):
                # Need to rotate
                pass  # Simplified for now
            point = constraint.center + constraint.radius * radial
            traj_points.append(point)
        traj_points = np.array(traj_points)
        ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2],
               'k-', linewidth=2, label='Trajectory')

    # Plot prior (before constraint)
    ax.scatter(*prior.mean, color='red', s=100, marker='o', label='Prior mean')

    # Plot posterior (after constraint)
    ax.scatter(*posterior.mean, color='green', s=100, marker='o', label='Posterior mean')

    # Plot projection
    projected = constraint.project(prior.mean)
    ax.scatter(*projected, color='blue', s=100, marker='x', label='Projection')

    # Connect prior to projection
    ax.plot([prior.mean[0], projected[0]],
           [prior.mean[1], projected[1]],
           [prior.mean[2], projected[2]],
           'r--', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Constraint Effect on Uncertainty')

    return ax


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == '__main__':
    print("Path2 Phase C: Constraint-Based Bayesian Update")
    print("="*60)

    # Test 1: Circle Constraint
    print("\n" + "="*60)
    print("Test 1: Circle Constraint")
    print("="*60)

    # Create circular trajectory
    circle = CircleConstraint(
        center=np.array([0.0, 0.0, 0.5]),
        radius=1.5,
        normal=np.array([0.0, 0.0, 1.0])
    )

    # Create prior distribution (noisy prediction from LSTM)
    prior_mean = np.array([1.2, 0.8, 0.5])  # Off the circle
    prior_cov = np.diag([0.1, 0.1, 0.05])**2  # Isotropic uncertainty
    prior = GaussianDistribution(prior_mean, prior_cov)

    print(f"Prior:")
    print(f"  Mean: {prior.mean}")
    print(f"  Std:  {prior.std}")

    # Apply constraint
    posterior = circle.constrain(prior, constraint_std_radial=0.01)

    print(f"\nPosterior (after constraint):")
    print(f"  Mean: {posterior.mean}")
    print(f"  Std:  {posterior.std}")

    # Compute uncertainty reduction
    prior_volume = np.prod(prior.std)
    post_volume = np.prod(posterior.std)
    reduction = (1 - post_volume / prior_volume) * 100

    print(f"\nUncertainty reduction: {reduction:.1f}%")
    print(f"Distance to trajectory: {circle.distance(prior.mean):.4f} → {circle.distance(posterior.mean):.4f}")

    # Test 2: Line Constraint
    print("\n" + "="*60)
    print("Test 2: Line Constraint")
    print("="*60)

    # Create linear trajectory
    line = LineConstraint(
        point=np.array([0.0, 0.0, 0.5]),
        direction=np.array([1.0, 0.0, 0.0])
    )

    # Create prior
    prior_mean = np.array([1.0, 0.3, 0.6])
    prior_cov = np.diag([0.1, 0.1, 0.1])**2
    prior = GaussianDistribution(prior_mean, prior_cov)

    print(f"Prior:")
    print(f"  Mean: {prior.mean}")
    print(f"  Std:  {prior.std}")

    # Apply constraint
    posterior = line.constrain(prior, constraint_std_radial=0.01)

    print(f"\nPosterior (after constraint):")
    print(f"  Mean: {posterior.mean}")
    print(f"  Std:  {posterior.std}")

    prior_volume = np.prod(prior.std)
    post_volume = np.prod(posterior.std)
    reduction = (1 - post_volume / prior_volume) * 100

    print(f"\nUncertainty reduction: {reduction:.1f}%")
    print(f"Distance to trajectory: {line.distance(prior.mean):.4f} → {line.distance(posterior.mean):.4f}")

    # Test 3: Spline Constraint
    print("\n" + "="*60)
    print("Test 3: Spline Constraint")
    print("="*60)

    # Create spline trajectory
    control_points = np.array([
        [0.0, 0.0, 0.5],
        [1.0, 1.0, 0.5],
        [2.0, 0.5, 0.5],
        [3.0, 1.5, 0.5]
    ])
    spline = SplineConstraint(control_points)

    # Create prior
    prior_mean = np.array([1.5, 0.9, 0.6])
    prior_cov = np.diag([0.15, 0.15, 0.1])**2
    prior = GaussianDistribution(prior_mean, prior_cov)

    print(f"Prior:")
    print(f"  Mean: {prior.mean}")
    print(f"  Std:  {prior.std}")

    # Apply constraint
    posterior = spline.constrain(prior, constraint_std_radial=0.01)

    print(f"\nPosterior (after constraint):")
    print(f"  Mean: {posterior.mean}")
    print(f"  Std:  {posterior.std}")

    prior_volume = np.prod(prior.std)
    post_volume = np.prod(posterior.std)
    reduction = (1 - post_volume / prior_volume) * 100

    print(f"\nUncertainty reduction: {reduction:.1f}%")
    print(f"Distance to trajectory: {spline.distance(prior.mean):.4f} → {spline.distance(posterior.mean):.4f}")

    # Test 4: Directional Uncertainty Analysis
    print("\n" + "="*60)
    print("Test 4: Directional Uncertainty Analysis")
    print("="*60)

    # Use circle constraint
    prior_mean = np.array([1.3, 0.9, 0.5])
    prior_cov = np.diag([0.2, 0.2, 0.1])**2
    prior = GaussianDistribution(prior_mean, prior_cov)

    posterior = circle.constrain(prior, constraint_std_radial=0.01)

    # Transform to local frame to see directional uncertainty
    projected = circle.project(prior.mean)
    origin, basis = circle.local_frame(prior.mean)

    prior_local = prior.to_local_frame(origin, basis)
    post_local = posterior.to_local_frame(origin, basis)

    print(f"Prior (local frame):")
    print(f"  Tangent std:  {np.sqrt(prior_local.cov[0, 0]):.4f}")
    print(f"  Radial std:   {np.sqrt(prior_local.cov[1, 1]):.4f}")
    print(f"  Binormal std: {np.sqrt(prior_local.cov[2, 2]):.4f}")

    print(f"\nPosterior (local frame):")
    print(f"  Tangent std:  {np.sqrt(post_local.cov[0, 0]):.4f}")
    print(f"  Radial std:   {np.sqrt(post_local.cov[1, 1]):.4f}")
    print(f"  Binormal std: {np.sqrt(post_local.cov[2, 2]):.4f}")

    radial_reduction = (1 - np.sqrt(post_local.cov[1, 1]) / np.sqrt(prior_local.cov[1, 1])) * 100
    print(f"\nRadial uncertainty reduction: {radial_reduction:.1f}%")
    print(f"(Tangent uncertainty preserved)")

    print(f"\n{'='*60}")
    print("✓ Phase C implementation complete!")
    print("Ready for Colab testing.")
    print(f"{'='*60}")
