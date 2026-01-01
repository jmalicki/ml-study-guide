# Chapter: Second-Order Methods

## Overview

Second-order methods use curvature information (the Hessian) to determine optimal step directions, achieving faster convergence but at significant computational cost.

## Sections

### 1. Newton's Method

- Taylor expansion: L(θ + δ) ≈ L(θ) + ∇L·δ + (1/2)δᵀHδ
- Optimal step: set gradient of quadratic to zero
- Newton update: δ* = -H⁻¹∇L
- Update rule: θ_{t+1} = θ_t - H⁻¹∇L
- Quadratic convergence near optimum: error² each step

### 2. Why Newton's Method is Impractical

- Hessian has n² elements (n = number of parameters)
- For 7B parameter model: 7×10⁹ × 7×10⁹ = 4.9×10¹⁹ elements
- Matrix inversion is O(n³)
- Must recompute each step (Hessian changes)
- Not even close to feasible for deep learning

### 3. Gauss-Newton Method

- Specialized for least-squares: L = (1/2)||f(θ) - y||²
- Jacobian J = ∂f/∂θ (n_outputs × n_params)
- Hessian approximation: H ≈ JᵀJ (ignores second derivatives of f)
- Update: δ = -(JᵀJ)⁻¹Jᵀ(f - y)
- **Key insight**: JᵀJ is positive semi-definite (guaranteed descent)

### 4. When Gauss-Newton Equals Newton

- For linear models: f(θ) = Xθ, so J = X and JᵀJ is exact
- For generalized linear models near convergence
- For neural nets: approximation improves as residuals shrink
- Connection to natural gradient (coming in Chapter 5)

### 5. Levenberg-Marquardt

- Damped Gauss-Newton: (JᵀJ + λI)⁻¹ instead of (JᵀJ)⁻¹
- λ controls trust region size
- λ → 0: Newton step (aggressive)
- λ → ∞: Gradient descent step (conservative)
- Adaptive λ: increase if step fails, decrease if successful

### 6. Quasi-Newton Methods

- Idea: build Hessian approximation from gradient differences
- Secant equation: H_{k+1}(θ_{k+1} - θ_k) = ∇L_{k+1} - ∇L_k
- **BFGS**: rank-2 update to maintain positive definiteness
  ```
  H_{k+1} = H_k + (correction terms using gradient differences)
  ```
- **L-BFGS**: Limited memory version, stores only last m gradient pairs
- Still O(n²) for BFGS, O(mn) for L-BFGS

### 7. L-BFGS in Practice

- Works well for: small models, fine-tuning, full-batch training
- Fails for: large models, stochastic training, highly non-convex
- Why SGD+momentum beat L-BFGS: implicit regularization, noise helps
- Modern use: optimizer for small subproblems (e.g., line search)

### 8. Hessian-Free Optimization (Truncated Newton)

**Historical Significance**: This was a major step forward circa 2010-2012, showing that full Newton-like methods could work on deep networks before Adam dominated.

**Core Idea**: Never form H explicitly—only compute Hessian-vector products H·v

**Hessian-Vector Products via Autodiff**:
- Key identity: Hv = ∂/∂ε [∇L(θ + εv)]|_{ε=0}
- Compute with two backprop passes: O(n) cost, not O(n²)
- This makes "matrix-free" Newton feasible

**Conjugate Gradient (CG) for Newton Direction**:
- Newton step solves: H·δ = -∇L
- CG iteratively finds δ using only H·v products
- Each CG iteration: one Hessian-vector product
- After k iterations: δ lies in Krylov subspace span{∇L, H∇L, H²∇L, ...}

**Truncated CG**:
- Don't run CG to convergence (expensive)
- Stop after k iterations (typically 10-50)
- Early stopping provides implicit regularization
- Avoids negative curvature directions (important near saddles)

**Damping and Trust Regions**:
- Solve (H + λI)δ = -∇L instead of Hδ = -∇L
- λ ensures positive definiteness
- CG naturally handles this

**Martens (2010) "Deep Learning via Hessian-Free Optimization"**:
- First to train deep networks with full second-order method
- Used structural damping specific to neural nets
- Achieved state-of-the-art on speech recognition

**Why It Lost to Adam**:
- Hyperparameter sensitivity (CG iterations, damping)
- High per-step cost (multiple forward/backward passes)
- Adam's simplicity and robustness won in practice
- But: Ideas live on in K-FAC, Shampoo (approximating the same thing)

### 9. The Computational Hierarchy

| Method | Memory | Per-step cost | Convergence |
|--------|--------|---------------|-------------|
| GD | $O(n)$ | $O(n)$ | $O(\kappa)$ steps |
| Momentum | $O(n)$ | $O(n)$ | $O(\sqrt{\kappa})$ steps |
| Adam | $O(n)$ | $O(n)$ | $O(\sqrt{\kappa})$ (empirical) |
| L-BFGS | $O(mn)$ | $O(mn)$ | Superlinear |
| Hessian-Free | $O(n)$ | $O(kn)$ per step | ~Quadratic |
| Newton | $O(n^2)$ | $O(n^3)$ | Quadratic |

## Code

```python
class NewtonOptimizer:
    """Pure Newton (for tiny problems only)."""

class GaussNewton:
    """Gauss-Newton for least squares."""

class LBFGS:
    """L-BFGS implementation."""

def hessian_vector_product(loss_fn, params, v):
    """Efficient H·v via autodiff."""

class HessianFree:
    """Hessian-Free optimizer using truncated CG."""

def conjugate_gradient(hessian_vector_product, b, max_iters=50):
    """Truncated CG solver for Hx = b."""

# Demonstration on small problems
# Show quadratic convergence of Newton
# Compare L-BFGS to Adam on convex problem
# Visualize CG iterations building up the solution
```

## Key Equations

- Newton: θ = θ - H⁻¹∇L
- Gauss-Newton: θ = θ - (JᵀJ)⁻¹Jᵀr where r = f(θ) - y
- BFGS update: H_{k+1} = (I - ρsyᵀ)H_k(I - ρysᵀ) + ρssᵀ
- Hessian-vector: Hv = ∂/∂ε [∇L(θ + εv)]|_{ε=0}

## Exercises

1. Implement Newton's method for a 2D problem, show quadratic convergence
2. Derive the Gauss-Newton approximation from the full Hessian
3. Implement Hessian-vector products using PyTorch autograd
4. Compare L-BFGS vs Adam on logistic regression (convex, full-batch)
5. Show that Gauss-Newton = Newton for linear least squares
6. Implement truncated CG and train a small MLP with Hessian-Free
7. Visualize how CG iterations progressively find better directions in Krylov subspace

## Connections

- Back: Addresses ill-conditioning from Chapter 1
- Forward: Natural gradient is Newton in distribution space (Chapter 5)
- Forward: Shampoo approximates second-order (Chapter 6)
