# Chapter: Practical Second-Order Methods

## Overview

This chapter covers practical approximations to second-order and natural gradient methods that are feasible for large-scale training: K-FAC, Shampoo, and SOAP.

## Sections

### 1. The Approximation Hierarchy

| Method | Approximation | Memory | Captures |
|--------|--------------|--------|----------|
| SGD | None (identity) | $O(n)$ | Nothing |
| Adam | Diagonal | $O(n)$ | Per-param variance |
| K-FAC | Block Kronecker | $O(\Sigma(m^2 + n^2))$ | Layer correlations |
| Shampoo | Full Kronecker | $O(\Sigma(m^2 + n^2))$ | Row/col correlations |
| Full Newton | Exact | $O(n^2)$ | Everything |

### 2. K-FAC: Kronecker-Factored Approximate Curvature

- Key insight: For layer with input a and gradient g:
  ```
  F ≈ E[aaᵀ] ⊗ E[ggᵀ] = A ⊗ G
  ```
- Kronecker product structure enables efficient inversion:
  ```
  (A ⊗ G)⁻¹ = A⁻¹ ⊗ G⁻¹
  ```
- Memory: O(m² + n²) instead of O(m²n²)
- Inversion: O(m³ + n³) instead of O(m³n³)

### 3. K-FAC Implementation Details

- Maintain running averages of A = E[aaᵀ] and G = E[ggᵀ]
- Damping: (A + λI)⁻¹ ⊗ (G + μI)⁻¹ for stability
- Inversion frequency: every 10-100 steps (amortize cost)
- Block-diagonal: treat each layer independently

### 4. Shampoo (Gupta et al., 2018)

- Different factorization: precondition rows and columns separately
- For weight matrix W ∈ ℝ^{m×n} with gradient G:
  ```
  L = E[GGᵀ]  (m×m, left preconditioner)
  R = E[GᵀG]  (n×n, right preconditioner)
  Update: W -= η · L^{-1/p} G R^{-1/p}
  ```
- Uses p=4 (fourth root) instead of p=2 (inverse square root)

### 5. Why Fourth Root?

- Square root: too aggressive, can destabilize
- Fourth root: better conditioning, more stable
- Derivation from tensor preconditioning theory
- Empirically: p=4 works best across many settings

### 6. Shampoo Implementation

- Eigendecomposition for matrix fourth root: L = VΛVᵀ → L^{-1/4} = VΛ^{-1/4}Vᵀ
- Expensive: O(m³ + n³) per layer
- Amortization: recompute every 100-1000 steps
- Grafting: use Shampoo direction with Adam learning rate

### 7. SOAP: Shampoo + Adam

- Observation: Shampoo works in eigenbasis of L and R
- SOAP: Run Adam in this rotated space
- Benefits:
  - More stable than pure Shampoo
  - Better learning rate adaptation
  - Adam handles dimensions Shampoo misses

### 8. Distributed Shampoo

- Challenge: Large L, R matrices for big layers
- Solution: Shard preconditioners across GPUs
- Communication: All-gather for matrix multiply
- Memory: Distributed storage of m² + n² elements

### 9. AlgoPerf Benchmark Results

- Setting: Standardized training benchmark (MLCommons)
- Shampoo: 28% faster wall-clock time than Adam
- Despite: Higher per-step cost
- Key: Fewer steps needed, amortized preconditioner cost

### 10. When to Use What

| Scenario | Recommended |
|----------|-------------|
| Limited memory | Adam/AdamW |
| Large batch, long training | Shampoo |
| RL / online learning | K-FAC |
| Very large models | Distributed Shampoo or Adam |
| Quick experiments | Adam (simplicity) |

## Code

```python
class KFAC:
    """K-FAC optimizer for neural networks."""

class ShampooOptimizer:
    """Shampoo with eigendecomposition."""

class SOAP:
    """SOAP: Shampoo + Adam hybrid."""

def matrix_power(M, p):
    """Compute M^p via eigendecomposition."""

# Benchmark on standard problems
# Memory usage comparison
# Convergence curves
```

## Key Equations

- K-FAC: F ≈ A ⊗ G where A = E[aaᵀ], G = E[ggᵀ]
- Shampoo: W -= η · L^{-1/4} G R^{-1/4}
- Matrix fourth root: M^{-1/4} = V Λ^{-1/4} Vᵀ (eigendecomposition)
- Kronecker inverse: (A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹

## Exercises

1. Implement K-FAC for a 2-layer MLP
2. Derive the Kronecker factorization for a linear layer's Fisher
3. Implement matrix fourth root via eigendecomposition
4. Compare Shampoo vs Adam on CIFAR-10: steps to accuracy, wall-clock time
5. Implement Shampoo with grafting (Adam learning rate)

## Connections

- Back: Approximates natural gradient (Chapter 5)
- Forward: Muon takes different approach (Chapter 7)
- Related: Distributed training chapter for scaling Shampoo
