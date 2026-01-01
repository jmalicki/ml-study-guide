# Chapter: Momentum and Acceleration

## Overview

Momentum methods accelerate convergence by accumulating velocity, helping escape saddle points and reducing oscillation in ill-conditioned problems.

## Sections

### 1. Physical Intuition

- Heavy ball rolling down a loss surface
- Inertia smooths out oscillations
- Accumulating velocity in consistent directions
- Damping prevents overshooting

### 2. Classical Momentum (Polyak, 1964)

- Update rule:
  ```
  v_{t+1} = βv_t + ∇L(θ_t)
  θ_{t+1} = θ_t - ηv_{t+1}
  ```
- β ∈ [0, 1) as momentum coefficient (typically 0.9)
- Interpretation as exponential moving average of gradients
- Effective learning rate amplification: η/(1-β) in limit

### 3. Nesterov Accelerated Gradient (NAG)

- Key insight: look ahead before computing gradient
- Update rule:
  ```
  v_{t+1} = βv_t + ∇L(θ_t - ηβv_t)  # gradient at "lookahead" point
  θ_{t+1} = θ_t - ηv_{t+1}
  ```
- Equivalent formulation (more common in practice):
  ```
  v_{t+1} = βv_t + η∇L(θ_t)
  θ_{t+1} = θ_t - β v_{t+1} - η∇L(θ_t)
  ```
- Convergence: O(1/t²) vs O(1/t) for vanilla GD on convex problems

### 4. Why Momentum Helps

- Reduces oscillation in high-curvature directions
- Accelerates movement in low-curvature directions
- Escaping saddle points: momentum carries through flat regions
- Noise averaging: smooths stochastic gradients in SGD

### 5. Convergence Analysis

- For quadratics: optimal β approaches 1 as condition number increases
- Convergence rate: O(√κ) vs O(κ) for GD
- The "momentum gap": why Nesterov is faster
- Heavy ball vs Nesterov: when does lookahead help?

### 6. Momentum in Stochastic Settings

- SGD with momentum: the standard baseline
- Interaction between momentum and batch size
- Momentum and learning rate warmup
- Gradient noise and momentum's smoothing effect

### 7. Connection to Differential Equations

- GD as discretized gradient flow: dθ/dt = -∇L
- Momentum as second-order ODE: d²θ/dt² + γ dθ/dt = -∇L
- Nesterov as a specific discretization of accelerated flow
- Insights from continuous-time analysis

## Code

```python
class SGDMomentum:
    """SGD with classical momentum."""

class NesterovSGD:
    """SGD with Nesterov momentum."""

# Comparison on ill-conditioned quadratic
# Visualization of trajectories
# Saddle point escape demonstration
```

## Key Equations

- Classical momentum: v = βv + ∇L; θ = θ - ηv
- Nesterov: v = βv + ∇L(θ - ηβv); θ = θ - ηv
- Effective step: η_eff = η/(1-β) for consistent gradient direction

## Exercises

1. Implement both momentum variants and compare on Rosenbrock function
2. Derive the optimal β for a 2D quadratic with eigenvalues λ₁, λ₂
3. Show that momentum accumulates to η/(1-β) for constant gradients
4. Implement the continuous-time ODE and compare to discrete updates
5. Demonstrate saddle point escape with and without momentum

## Connections

- Back: Builds on GD limitations (Chapter 1)
- Forward: Adam combines momentum with adaptive rates (Chapter 3)
- Related: Momentum in natural gradient methods
