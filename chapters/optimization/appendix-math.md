# Appendix: Mathematical Foundations for Optimization

This appendix provides the mathematical background for the geometric and algebraic structures underlying optimization algorithms.

## Table of Contents

1. [Riemannian Manifolds](#riemannian-manifolds)
2. [Fisher Information Geometry](#fisher-information-geometry)
3. [Matrix Calculus](#matrix-calculus)
4. [Matrix Functions](#matrix-functions)
5. [Newton-Schulz Derivation](#newton-schulz-derivation)
6. [Convergence Analysis](#convergence-analysis)
7. [Key Theorems](#key-theorems)

---

## Riemannian Manifolds

A **manifold** is a space that locally looks like $\mathbb{R}^n$. Examples:
- A sphere $S^2$ (locally looks like $\mathbb{R}^2$)
- The space of probability distributions

A **Riemannian manifold** has a smoothly varying inner product $g_p(u, v)$ on each tangent space $T_p M$.

### Geodesics

A **geodesic** is the shortest path between points on a manifold:
- On $\mathbb{R}^n$: straight lines
- On a sphere: great circles

### Gradient on a Manifold

The Riemannian gradient $\nabla_M f$ satisfies:

$$g_p(\nabla_M f, v) = df(v) \quad \forall v \in T_p M$$

For Euclidean space with $g = I$: $\nabla_M f = \nabla f$

For Fisher metric with $g = F$: $\nabla_M f = F^{-1} \nabla f$ (natural gradient)

---

## Fisher Information Geometry

### Definition

For a parametric family $\{p_\theta : \theta \in \Theta\}$:

$$F_{ij}(\theta) = \mathbb{E}_{p_\theta}\left[\frac{\partial \log p}{\partial \theta_i} \cdot \frac{\partial \log p}{\partial \theta_j}\right]$$

### Fisher as Metric Tensor

$F$ defines a Riemannian metric on the parameter space:
- Infinitesimal distance: $ds^2 = d\theta^\top F d\theta$
- Equal to local KL divergence: $D_{KL}(p_\theta \| p_{\theta + d\theta}) = \frac{1}{2} d\theta^\top F d\theta$

### Fisher = Expected Hessian

For log-likelihood loss $L = -\log p(y|x, \theta)$:

$$F = \mathbb{E}[-\nabla^2 \log p] = \mathbb{E}[H]$$

**Proof sketch**: Use $\mathbb{E}[\nabla \log p] = 0$ and differentiate.

---

## Matrix Calculus

### Vectorization

For $A \in \mathbb{R}^{m \times n}$, $\text{vec}(A) \in \mathbb{R}^{mn}$ stacks columns.

### Kronecker Product

$(A \otimes B)_{(i,j),(k,l)} = A_{ik} B_{jl}$

Key identities:
- $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$
- $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$
- $\text{vec}(ABC) = (C^\top \otimes A) \text{vec}(B)$

### Matrix Derivatives

For scalar $L$ and matrix $W$:

$$\frac{\partial L}{\partial W_{ij}} = \text{gradient in standard coordinates}$$

Chain rule: If $L = L(Y)$ and $Y = f(W)$:

$$\frac{\partial L}{\partial W} = \sum_{kl} \frac{\partial L}{\partial Y_{kl}} \frac{\partial Y_{kl}}{\partial W}$$

---

## Matrix Functions

### Matrix Square Root

For symmetric positive definite $A$, $A^{1/2}$ satisfies $(A^{1/2})^2 = A$.

Computation via eigendecomposition:
$$A = V\Lambda V^\top \implies A^{1/2} = V\Lambda^{1/2}V^\top$$

### Matrix p-th Root

$$A^{1/p} = V\Lambda^{1/p}V^\top$$

Shampoo uses $p = 4$ (fourth root).

### Polar Decomposition

Any $M = UP$ where:
- $U$ is orthogonal (or unitary)
- $P = (M^\top M)^{1/2}$ is positive semi-definite

$U$ is the nearest orthogonal matrix to $M$:
$$U = \arg\min_{Q: Q^\top Q = I} \|M - Q\|_F$$

---

## Newton-Schulz Derivation

### Goal

Find orthogonal matrix $X$ satisfying $X^\top X = I$.

### Newton's Method

Define $f(X) = X^\top X - I$.

Newton update: $X_{k+1} = X_k - [Df]^{-1} f(X_k)$

### Computing the Differential

$f(X + H) - f(X) = X^\top H + H^\top X + H^\top H$

Linear part: $Df_X[H] = X^\top H + H^\top X$

### Solving for the Update

Newton step solves: $X^\top H + H^\top X = -(X^\top X - I)$

For $H = X \cdot A$ (symmetric $A$):
$$X^\top X A + A X^\top X = -(X^\top X - I)$$

Let $B = X^\top X$. Then $BA + AB = I - B$, solved by $A = (I - B)/(2I) = (I - B)/2$ when $B$ commutes.

### Final Formula

$$X_{k+1} = X_k + X_k \cdot \frac{I - X_k^\top X_k}{2} = X_k \cdot \frac{3I - X_k^\top X_k}{2}$$

### Convergence

**Quadratic convergence**: Near an orthogonal matrix,
$$\|X_{k+1} - U\| \leq C\|X_k - U\|^2$$

5 iterations typically sufficient for double precision.

---

## Convergence Analysis

### Gradient Descent on Quadratics

For $L(\theta) = \frac{1}{2}\theta^\top H \theta$ with $H = \text{diag}(\lambda_1, \ldots, \lambda_n)$:

Update: $\theta_{k+1} = (I - \eta H)\theta_k$

Convergence: $\theta_k = (I - \eta H)^k \theta_0$

For convergence, need $|1 - \eta \lambda_i| < 1$ for all $i$, giving $\eta < 2/\lambda_{\max}$.

Optimal $\eta = 2/(\lambda_{\max} + \lambda_{\min})$ gives rate $(\kappa-1)/(\kappa+1)$ where $\kappa = \lambda_{\max}/\lambda_{\min}$.

### Momentum Analysis

Heavy ball: $\theta_{k+1} = \theta_k - \eta \nabla L + \beta(\theta_k - \theta_{k-1})$

Optimal $\beta = ((\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1))^2$ gives rate $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1)$.

Improvement: $O(\kappa)$ iterations → $O(\sqrt{\kappa})$ iterations.

### Strong Convexity

$L$ is $\mu$-strongly convex if:
$$L(y) \geq L(x) + \nabla L(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2$$

Guarantees unique minimum and exponential convergence.

### Smoothness

$L$ has $L$-Lipschitz gradients if:
$$\|\nabla L(x) - \nabla L(y)\| \leq L\|x - y\|$$

Equivalent to Hessian bounded: $\|\nabla^2 L\| \leq L$.

---

## Key Theorems

### Amari (1998)

Natural gradient is **covariant**: the update direction is independent of parameterization.

### Nesterov (1983)

Accelerated gradient achieves optimal $O(1/t^2)$ rate for smooth convex functions, matching the lower bound.

### Martens & Grosse (2015)

K-FAC's Kronecker factorization $F \approx A \otimes G$ is exact for linear networks and approximately correct for deep networks.

---

## Further Reading

1. [Amari, S. "Natural Gradient Works Efficiently in Learning" (1998)](https://direct.mit.edu/neco/article/10/2/251/6143/Natural-Gradient-Works-Efficiently-in-Learning)

2. [Martens, J. & Grosse, R. "Optimizing Neural Networks with Kronecker-factored Approximate Curvature" (ICML 2015)](https://arxiv.org/abs/1503.05671)

3. [Gupta, V. et al. "Shampoo: Preconditioned Stochastic Tensor Optimization" (ICML 2018)](https://arxiv.org/abs/1802.09568)

4. [Bernstein, J. et al. "Old Optimizer, New Norm: An Anthology" (2023)](https://arxiv.org/abs/2312.08621) — Muon theoretical foundations

5. [Absil, P-A. et al. "Optimization Algorithms on Matrix Manifolds" (2008)](https://press.princeton.edu/absil)
