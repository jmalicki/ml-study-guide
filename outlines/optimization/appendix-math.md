# Appendix: Mathematical Foundations for Optimization

## Overview

This appendix provides the mathematical background for understanding the geometric and algebraic structures underlying optimization algorithms.

## Sections

### 1. Riemannian Manifolds

- **Manifold**: A space that locally looks like ℝⁿ
- **Tangent space**: Linear approximation at each point
- **Riemannian metric**: Smoothly varying inner product on tangent spaces
  ```
  g_p(u, v) for u, v ∈ T_p M
  ```
- **Distance**: Integrate metric along curves
- **Euclidean space**: Special case where g = I everywhere

### 2. Geodesics and Exponential Map

- **Geodesic**: Shortest path between points on manifold
- Straight lines in Euclidean space, great circles on sphere
- **Exponential map**: exp_p(v) = endpoint of geodesic from p in direction v
- **Logarithmic map**: Inverse of exponential
- Optimization on manifolds: Move along geodesics, not straight lines

### 3. The Fisher Metric

- For statistical manifold M = {p_θ : θ ∈ Θ}
- Metric tensor:
  ```
  g_ij(θ) = E_{p_θ}[∂log p_θ/∂θ_i · ∂log p_θ/∂θ_j]
  ```
- This is the Fisher Information Matrix F(θ)
- Unique (up to scaling) metric invariant under sufficient statistics

### 4. Natural Gradient Derivation

- Want: Direction of steepest descent in KL divergence
- KL divergence: D_KL(p_θ || p_{θ+δ}) ≈ (1/2) δᵀ F δ (to second order)
- Steepest descent: minimize L(θ + δ) subject to δᵀFδ ≤ ε²
- Lagrangian: L(θ) + ∇Lᵀδ + λ(δᵀFδ - ε²)
- Solution: δ ∝ F⁻¹∇L

### 5. Matrix Calculus

**Vectorization**:
- vec(A): Stack columns of A into a vector
- For A ∈ ℝ^{m×n}: vec(A) ∈ ℝ^{mn}

**Kronecker Product**:
- A ⊗ B: Block matrix with A_ij B in position (i,j)
- Key identity: vec(ABC) = (Cᵀ ⊗ A) vec(B)
- For layers: vec(∂L/∂W) relates to input and output gradients

**Kronecker Properties**:
- (A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹
- (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD)
- eigenvalues(A ⊗ B) = {λ_i μ_j : λ_i ∈ eig(A), μ_j ∈ eig(B)}

### 6. Matrix Derivatives

**Scalar by Matrix**:
- ∂L/∂W where L is scalar, W is matrix
- Result has same shape as W

**Chain Rule**:
- For L(Y), Y = f(W):
  ```
  ∂L/∂W = Σ_ij (∂L/∂Y_ij)(∂Y_ij/∂W)
  ```

**Common Results**:
- ∂||AW||²/∂W = 2AᵀAW
- ∂trace(AᵀW)/∂W = A
- ∂log det(W)/∂W = W⁻ᵀ

### 7. Matrix Functions

**Matrix Square Root**:
- A^{1/2} satisfies A^{1/2} A^{1/2} = A
- For PSD A: Exists and is unique (PSD)
- Via eigendecomposition: A = VΛVᵀ → A^{1/2} = VΛ^{1/2}Vᵀ

**Matrix p-th Root**:
- A^{1/p}: Generalization to any p
- For positive A: A^{1/p} = V Λ^{1/p} Vᵀ

**Newton-Schulz Derivation**:
- Want: X such that XᵀX = I (orthogonal)
- Newton's method for f(X) = XᵀX - I
- Jacobian: ∂f/∂X (tensor, but simplifies)
- Update: X_{k+1} = X_k - [J]⁻¹ f(X_k) = X_k(3I - X_kᵀX_k)/2

### 8. Singular Value Decomposition

- Any A ∈ ℝ^{m×n}: A = UΣVᵀ
- U ∈ ℝ^{m×m} orthogonal (left singular vectors)
- Σ ∈ ℝ^{m×n} diagonal (singular values)
- V ∈ ℝ^{n×n} orthogonal (right singular vectors)

**Polar Decomposition**:
- A = QP where Q orthogonal, P PSD
- Q = UV^T (from SVD)
- P = VΣVᵀ = (AᵀA)^{1/2}

### 9. Convex Analysis

**Convexity**:
- f convex: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
- Equivalent: Hessian ∇²f ≽ 0 (PSD)

**Strong Convexity**:
- f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||²
- Guarantees unique minimum, faster convergence

**Smoothness**:
- ||∇f(x) - ∇f(y)|| ≤ L||x - y||
- Gradient doesn't change too fast
- Enables learning rate bounds: η ≤ 1/L

**Condition Number**:
- κ = L/μ (smoothness / strong convexity)
- GD convergence: O(κ log(1/ε))
- Accelerated: O(√κ log(1/ε))

### 10. Convergence Analysis

**Lyapunov Functions**:
- Find V(θ) that decreases each iteration
- V(θ_t) - V(θ_{t+1}) ≥ δ > 0 implies convergence

**Spectral Analysis**:
- For linear iteration θ_{t+1} = Mθ_t
- Convergence iff ρ(M) < 1 (spectral radius)
- Rate: ||θ_t|| ~ ρ(M)^t

**Stochastic Convergence**:
- Robbins-Monro conditions: Ση_t = ∞, Ση_t² < ∞
- Ensures reaching optimum while controlling variance

## Key Theorems

1. **Amari (1998)**: Natural gradient is covariant under reparameterization
2. **Martens (2014)**: K-FAC approximates natural gradient for neural networks
3. **Nesterov (1983)**: Accelerated gradient achieves optimal O(1/t²) for convex

## Further Reading

- Amari, "Natural Gradient Works Efficiently in Learning" (1998)
- Absil et al., "Optimization Algorithms on Matrix Manifolds" (2008)
- Martens & Grosse, "Optimizing Neural Networks with Kronecker-factored Approximate Curvature" (2015)
- Bernstein et al., "Old Optimizer, New Norm" (2023) — Muon foundations
