# Chapter: Natural Gradient and Information Geometry

## Overview

Natural gradient descent recognizes that neural network parameters define probability distributions, and the "correct" geometry is not Euclidean but the manifold of distributions with the Fisher metric.

## Sections

### 1. The Fundamental Problem

- Parameters θ live in ℝⁿ, but we care about the *functions* they represent
- Small change in θ can mean large change in function (and vice versa)
- Euclidean distance in θ-space doesn't reflect functional distance
- Need a metric that captures "how different are the predictions?"

### 2. Statistical Manifolds

- Neural network defines p(y|x, θ): a family of distributions
- θ parameterizes points on a *manifold* of distributions
- Different parameterizations = different coordinate systems
- Need a coordinate-invariant way to measure distances

### 3. KL Divergence as Distance

- Natural measure between distributions: KL divergence
- D_KL(p_θ || p_{θ+δ}) measures how different θ and θ+δ are *as distributions*
- Taylor expand KL divergence:
  ```
  D_KL(p_θ || p_{θ+δ}) ≈ (1/2) δᵀ F(θ) δ
  ```
- F(θ) is the Fisher Information Matrix

### 4. Fisher Information Matrix

- Definition:
  ```
  F_ij = E_{p(y|x,θ)}[∂log p/∂θ_i · ∂log p/∂θ_j]
  ```
- Equivalently: F = E[∇log p · (∇log p)ᵀ]
- F is the metric tensor on the statistical manifold
- F is always positive semi-definite
- F changes as θ changes (the manifold has curvature)

### 5. Natural Gradient

- Steepest descent direction in KL-divergence:
  ```
  θ_{t+1} = θ_t - η F⁻¹ ∇L
  ```
- ∇̃L = F⁻¹∇L is the *natural gradient*
- Invariant to parameterization: same update regardless of how we parameterize θ
- Amari (1998): foundational paper

### 6. Fisher = Expected Hessian

- For log-likelihood loss L = -log p(y|x, θ):
  ```
  E[H] = E[-∂²log p/∂θ∂θᵀ] = F
  ```
- Natural gradient ≈ Newton's method for maximum likelihood
- But: F is an *expectation*, H is for a specific sample
- F is always PSD, H might not be

### 7. Why Natural Gradient is Better

- **Invariance**: Same optimization path regardless of parameterization
- **Conditioning**: F captures the "right" geometry
- **Efficiency**: Faster convergence on ill-conditioned problems
- Empirically: 10-100x fewer iterations on some problems

### 8. The Computational Barrier

- F is n×n matrix (n = number of parameters)
- Computing F requires expectations over data
- Inverting F is O(n³)
- For 7B params: completely infeasible
- Must approximate: this is where K-FAC, Shampoo come in

### 9. Connection to Other Concepts

- **Mirror descent**: Natural gradient is mirror descent with KL as Bregman divergence
- **Trust regions**: Natural gradient automatically scales step size
- **Policy gradient in RL**: TRPO/PPO use Fisher to constrain policy updates
- **Adam as approximation**: diagonal of F ≈ E[g²] ≈ Adam's v

### 10. Exponential Families and Natural Parameters

- For exponential families, natural gradient has closed form
- Natural parameters vs mean parameters
- Connection to sufficient statistics
- Why softmax output layers are "natural"

## Code

```python
def compute_fisher_matrix(model, data_loader):
    """Compute empirical Fisher (expensive, for illustration)."""

def natural_gradient_step(model, loss, data):
    """One step of natural gradient descent."""

# Demonstrate on small model
# Show invariance to reparameterization
# Compare convergence: GD vs natural gradient
```

## Key Equations

- Fisher: F_ij = E[∂log p/∂θ_i · ∂log p/∂θ_j]
- Natural gradient: θ = θ - η F⁻¹ ∇L
- KL as local metric: D_KL(p_θ || p_{θ+δ}) ≈ (1/2) δᵀ F δ
- Fisher = expected Hessian: F = -E[∂²log p/∂θ²]

## Exercises

1. Compute the Fisher matrix for logistic regression (closed form)
2. Show that F is invariant to reparameterization (derive transformation rule)
3. Prove Fisher = expected Hessian for log-likelihood
4. Implement natural gradient on a 2-layer MLP, compare to Adam
5. Show that for Gaussian with known variance, F = (1/σ²)XᵀX

## Connections

- Back: Newton's method in distribution space (Chapter 4)
- Forward: K-FAC approximates F via Kronecker factors (Chapter 6)
- Forward: Muon uses different geometry entirely (Chapter 7)
- Related: TRPO/PPO in RL use Fisher-based trust regions
