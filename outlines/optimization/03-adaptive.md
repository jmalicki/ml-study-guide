# Chapter: Adaptive Learning Rates

## Overview

Adaptive methods assign different learning rates to different parameters based on gradient history, addressing ill-conditioning without explicit second-order computation.

## Sections

### 1. The Key Insight

- Different parameters need different learning rates
- Frequently updated parameters → smaller learning rates
- Rarely updated parameters → larger learning rates
- Use gradient history to infer appropriate scaling

### 2. AdaGrad (Duchi et al., 2011)

- Accumulate squared gradients: G_t = Σ_{i≤t} g_i ⊙ g_i
- Update: θ_{t+1} = θ_t - η · g_t / (√G_t + ε)
- Per-parameter learning rate: η / √(Σg²)
- **Strengths**: Excellent for sparse gradients (NLP, recommender systems)
- **Weakness**: Learning rate monotonically decreases → premature stopping

### 3. RMSProp (Hinton, unpublished)

- Exponential moving average instead of sum: v_t = βv_{t-1} + (1-β)g_t²
- Update: θ_{t+1} = θ_t - η · g_t / (√v_t + ε)
- Solves AdaGrad's decay problem
- β typically 0.99 (longer memory than momentum)
- Leaky average "forgets" old gradients

### 4. Adam (Kingma & Ba, 2015)

- Combine momentum (first moment) with RMSProp (second moment):
  ```
  m_t = β₁ m_{t-1} + (1-β₁) g_t      # First moment (momentum)
  v_t = β₂ v_{t-1} + (1-β₂) g_t²     # Second moment (RMSProp)
  ```
- Bias correction for initialization:
  ```
  m̂_t = m_t / (1 - β₁ᵗ)
  v̂_t = v_t / (1 - β₂ᵗ)
  ```
- Update: θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
- Default hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8

### 5. Adam's Convergence Issues

- Non-convergence examples (Reddi et al., 2018)
- The problem: second moment can decrease, causing large updates
- AMSGrad fix: v̂_t = max(v̂_{t-1}, v_t)
- In practice: Adam usually works fine, AMSGrad rarely needed

### 6. AdamW: Decoupled Weight Decay (Loshchilov & Hutter, 2019)

- L2 regularization ≠ weight decay in adaptive methods
- L2 in loss: ∇(L + λ||θ||²) = ∇L + 2λθ — gets scaled by 1/√v
- Decoupled: θ = θ - η(m̂/√v̂ + λθ) — weight decay not scaled
- **This is the standard for LLM training**
- Why it matters: proper regularization strength independent of gradient magnitude

### 7. Other Adam Variants

- **AdaFactor** (Shazeer & Stern, 2018): Memory-efficient, factored second moments
- **LAMB** (You et al., 2020): Layer-wise adaptive rates for large batch training
- **Lion** (Chen et al., 2023): Sign-based update, memory efficient
- **Sophia** (Liu et al., 2023): Diagonal Hessian estimation

### 8. Understanding Adaptive Methods Geometrically

- Diagonal preconditioning: P = diag(1/√v)
- Rescales each axis independently
- Approximates inverse diagonal Hessian
- Limitation: ignores correlations between parameters

## Code

```python
class AdaGrad:
    """AdaGrad optimizer."""

class RMSProp:
    """RMSProp optimizer."""

class Adam:
    """Adam optimizer with bias correction."""

class AdamW:
    """AdamW with decoupled weight decay."""

# Comparison on various loss surfaces
# Visualization of adaptive learning rates
```

## Key Equations

- AdaGrad: θ = θ - η·g/√(Σg²)
- RMSProp: v = βv + (1-β)g²; θ = θ - η·g/√v
- Adam: m = β₁m + (1-β₁)g; v = β₂v + (1-β₂)g²; θ = θ - η·m̂/√v̂
- AdamW: θ = θ - η·(m̂/√v̂ + λθ)

## Exercises

1. Implement AdaGrad, RMSProp, Adam, AdamW from scratch
2. Show AdaGrad's learning rate decay on a simple problem
3. Demonstrate the difference between L2 regularization and weight decay in Adam
4. Construct the non-convergence example for Adam (Reddi et al.)
5. Compare memory usage: Adam vs AdaFactor on a large model

## Connections

- Back: Uses momentum from Chapter 2
- Forward: Adam as diagonal approximation to natural gradient (Chapter 5)
- Forward: AdamW is the baseline for comparing Muon (Chapter 7)
