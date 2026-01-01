# Chapter: Muon and Operator Geometry

## Overview

Muon represents a fundamentally different approach to optimization: instead of viewing parameters as points in a statistical manifold (Fisher geometry), it views weight matrices as *operators* and uses operator-norm geometry.

## Sections

### 1. Two Geometric Frameworks

| Aspect | Natural Gradient (Fisher) | Muon (Operator) |
|--------|---------------------------|-----------------|
| Parameters represent | Probability distributions | Linear operators |
| Distance metric | KL divergence | Operator norm (spectral) |
| Preconditioner | F⁻¹ (Fisher inverse) | Orthogonalization (UVᵀ) |
| Key papers | Amari (1998), K-FAC, Shampoo | Bernstein, Jordan et al. |

### 2. Weight Matrices as Operators

- Linear layer: y = Wx maps inputs to outputs
- How different is W vs W + ΔW as a *function*?
- Measure by how much the output changes for unit input
- This is the *operator norm*: ||ΔW||_{op} = max_{||x||=1} ||ΔWx||

### 3. Operator Norm Geometry

- Spectral norm: ||M||_{op} = σ_max(M) (largest singular value)
- Frobenius norm: ||M||_F = √(Σσ_i²) (all singular values)
- For gradient descent: want to control ||ΔW||_{op}, not ||ΔW||_F
- Problem: Raw gradient points in suboptimal direction for operator norm

### 4. The Stiefel Manifold

- Space of orthonormal matrices: {Q : QᵀQ = I}
- Orthogonal matrices have ||Q||_{op} = ||Q||_F = 1 (all singular values = 1)
- Key insight: Project gradient onto this space for optimal conditioning
- Nearest orthogonal matrix to G: argmin_{Q:QᵀQ=I} ||G - Q||_F

### 5. Polar Decomposition

- Any matrix G = UP where U is orthogonal, P is PSD
- U is the nearest orthogonal matrix to G
- U = G(GᵀG)^{-1/2} (requires matrix square root)
- This is expensive to compute directly

### 6. Newton-Schulz Iteration

- Fast iterative method for orthogonalization:
  ```
  X₀ = G / ||G||_F  (normalize for stability)
  X_{k+1} = X_k (3I - X_kᵀX_k) / 2
  ```
- Converges to orthogonal factor U quadratically
- Derivation: Newton's method for solving XᵀX = I
- 5 iterations typically sufficient

### 7. Why Newton-Schulz Works

- Define f(X) = XᵀX - I (want to find root)
- Newton update: X_{k+1} = X_k - [∂f/∂X]⁻¹ f(X_k)
- After simplification: X_{k+1} = X_k(3I - X_kᵀX_k)/2
- Quadratic convergence: ||X_k - U|| ≤ C · ||X_{k-1} - U||²
- GPU-friendly: just matrix multiplications

### 8. The Muon Algorithm

```
For each 2D weight matrix W with gradient G:
    1. Update momentum: M = β·M + G
    2. Orthogonalize: M_orth = NewtonSchulz(M)
    3. Scale appropriately for this layer
    4. Update: W = W - η · M_orth - λ · W  (weight decay)
```

### 9. Key Differences from Shampoo

- Shampoo: L^{-1/4} G R^{-1/4} (precondition left and right)
- Muon: NewtonSchulz(G) (orthogonalize directly)
- Shampoo: Approximates Fisher geometry
- Muon: Uses operator geometry directly
- Shampoo: Stores L, R matrices; Muon: No extra state beyond momentum

### 10. Why Muon Works

- Orthogonalized gradient has σ_max = σ_min = 1
- Perfect conditioning in operator norm sense
- All singular values contribute equally to update
- Avoids "rich get richer" dynamics of Adam

### 11. Limitations and Hybrid Training

- Only works for 2D weight matrices
- Doesn't apply to: embeddings, biases, layer norms, 1D weights
- **Solution**: Muon for weight matrices + AdamW for everything else
- In practice: ~80% of parameters use Muon, ~20% use AdamW

### 12. Scaling Muon

- **Moonlight results** (Liu et al., 2025): Muon scales to large LLMs
- Weight decay handling: Decoupled, applied after Muon update
- Learning rate: Different scale than Adam (typically larger)
- Batch size scaling: Similar to Adam (linear scaling rule)

### 13. Muon vs Natural Gradient: The Debate

- Some claim Muon ≈ natural gradient; this is misleading
- Natural gradient: Corrects for *statistical* geometry
- Muon: Corrects for *operator* geometry
- Different theoretical foundations, different algorithms
- Both outperform Adam, but for different reasons

## Code

```python
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize matrix via Newton-Schulz iteration."""
    X = G / G.norm()
    for _ in range(steps):
        X = X @ (3 * torch.eye(X.shape[1], device=X.device) - X.T @ X) / 2
    return X

class Muon:
    """
    Muon optimizer for weight matrices.
    Use with AdamW for other parameters.
    """

class MuonAdamW:
    """Combined optimizer: Muon for weights, AdamW for rest."""

# Demonstration of Newton-Schulz convergence
# Comparison: Muon vs AdamW vs Shampoo
# Visualization of orthogonalized gradients
```

## Key Equations

- Operator norm: ||M||_{op} = σ_max(M)
- Newton-Schulz: X_{k+1} = X_k(3I - X_kᵀX_k)/2
- Muon update: W = W - η · NS(M) - λW where M = βM + G
- Polar decomposition: G = U·P, U orthogonal, P = (GᵀG)^{1/2}

## Exercises

1. Implement Newton-Schulz and verify quadratic convergence
2. Show that Newton-Schulz derives from Newton's method on XᵀX = I
3. Compare singular value distributions: raw gradient vs orthogonalized
4. Implement full Muon optimizer with momentum
5. Train a small transformer with Muon vs AdamW, compare loss curves
6. Derive the Jacobian of Newton-Schulz iteration (for understanding backprop)

## Connections

- Back: Contrast with Fisher geometry (Chapter 5)
- Back: Contrast with Shampoo (Chapter 6)
- Forward: Learning rate schedules for Muon (Chapter 8)
- Related: Spectral normalization in GANs (similar operator-norm ideas)
