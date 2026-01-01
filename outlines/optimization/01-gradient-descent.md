# Chapter: Gradient Descent and Its Limitations

## Overview

This chapter establishes why vanilla gradient descent is suboptimal and motivates the need for more sophisticated optimization methods.

## Sections

### 1. The Gradient Descent Update

- Definition: θ_{t+1} = θ_t - η ∇L(θ_t)
- Interpretation as steepest descent in Euclidean space
- Convergence for convex functions: O(1/t) rate
- Learning rate selection: too large → divergence, too small → slow

### 2. The Loss Landscape

- Visualizing loss surfaces in 2D and high dimensions
- Local minima, saddle points, and plateaus
- Sharp vs flat minima and generalization
- The blessing of dimensionality: saddle points dominate in high-D

### 3. Curvature and the Hessian

- Second-order Taylor expansion: L(θ + δ) ≈ L(θ) + ∇L·δ + (1/2)δᵀHδ
- Hessian matrix H = ∂²L/∂θ∂θᵀ
- Eigenvalues and eigenvectors of H
- Principal curvatures and their meaning

### 4. Ill-Conditioning

- Condition number: κ = λ_max / λ_min
- Why ill-conditioning slows convergence: oscillation in steep directions
- Convergence rate depends on κ: O(κ log(1/ε)) iterations
- Neural networks are notoriously ill-conditioned (κ ~ 10⁶ or worse)

### 5. Why Gradients Point in Suboptimal Directions

- The gradient is steepest in *Euclidean* metric
- But parameter space isn't naturally Euclidean
- Preview: different metrics give different "steepest" directions
- The key insight: we need to precondition the gradient

### 6. A Geometric View

- Gradient descent as following level sets
- Elliptical level sets from quadratic approximation
- The ideal direction: toward the minimum, not perpendicular to level sets
- Preconditioning = reshaping the level sets to be circular

## Code

```python
# Demonstrate ill-conditioning effects
# Show how GD oscillates on ill-conditioned quadratics
# Visualize convergence with different condition numbers
```

## Key Equations

- Gradient descent: θ_{t+1} = θ_t - η ∇L(θ_t)
- Convergence bound: L(θ_t) - L* ≤ O(1/(ηt)) for convex, smooth L
- Condition number: κ(H) = ||H|| · ||H⁻¹||

## Exercises

1. Implement GD on a 2D quadratic with varying condition numbers. Plot trajectories.
2. Derive the optimal learning rate for a quadratic: η* = 2/(λ_max + λ_min)
3. Show that for a quadratic, GD converges in one step if H = I (identity Hessian)
4. Compute the Hessian of a simple neural network loss and examine its eigenspectrum

## Connections

- Forward: Momentum (Chapter 2) addresses oscillation
- Forward: Preconditioning (Chapters 4-7) addresses ill-conditioning
- Related: Loss landscape visualization in training dynamics chapter
