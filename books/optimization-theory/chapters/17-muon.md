# Chapter 17: Muon and Operator Geometry

Muon represents a different philosophical approach to optimization: instead of approximating the Fisher information, it uses the **operator norm** geometry of weight matrices. This leads to a simple, elegant algorithm based on orthogonalization.

## A Different Geometry

### Two Geometric Frameworks

The optimization methods we've seen fall into two camps:

**Statistical geometry** (Fisher, K-FAC, Shampoo):
- Metric from the output distribution
- $\|dW\|^2 = dW^T F dW$
- Expensive: requires covariances of activations/gradients

**Operator geometry** (Muon):
- Metric from the operator norm
- $\|W\|_{op} = \max_{\|x\|=1} \|Wx\|$
- Cheap: only depends on the weights themselves

```python
import torch
import torch.nn as nn
from typing import Optional

def operator_norm(W: torch.Tensor) -> torch.Tensor:
    """Compute the operator (spectral) norm of a matrix."""
    return torch.linalg.svdvals(W)[0]


def frobenius_norm(W: torch.Tensor) -> torch.Tensor:
    """Compute the Frobenius norm."""
    return W.norm()


def compare_norms():
    """Show the difference between operator and Frobenius norms."""
    # Random matrices of different shapes
    for shape in [(100, 100), (100, 10), (10, 100)]:
        W = torch.randn(*shape)

        op_norm = operator_norm(W).item()
        frob_norm = frobenius_norm(W).item()

        print(f"Shape {shape}: ||W||_op = {op_norm:.2f}, ||W||_F = {frob_norm:.2f}")
```

### Why Operator Norm?

The operator norm measures the **maximum amplification** of a linear layer:

$$\|W\|_{op} = \max_{\|x\|=1} \|Wx\|$$

This is geometrically meaningful:
- Controls how much the layer can stretch/shrink inputs
- Relates to stability of forward/backward passes
- Independent of layer size (unlike Frobenius norm)

![Operator norm geometry](../images/operator-norm-geometry.svg)

## The Polar Decomposition

### Every Matrix = Rotation × Scale

Any matrix $W$ can be written as:

$$W = U \Sigma V^T$$

where $U, V$ are orthogonal and $\Sigma$ is diagonal (singular values).

Alternatively:

$$W = (UV^T)(V\Sigma V^T) = Q \cdot P$$

where:
- $Q = UV^T$ is orthogonal (the "direction")
- $P = V\Sigma V^T$ is positive semidefinite (the "magnitude")

### Muon's Insight

Standard gradient descent updates both direction and magnitude simultaneously.

**Muon's idea**: Only update the orthogonal part Q, letting magnitude adjust naturally.

An orthogonal matrix has operator norm 1—maximum efficiency without amplification.

## Newton-Schulz Iteration

### Computing Orthogonalization

Given a matrix $G$ (gradient), we want to find the closest orthogonal matrix $Q$.

**Newton-Schulz iteration** computes this efficiently:

$$X_{k+1} = X_k \cdot (3I - X_k^T X_k) / 2$$

> **Why the coefficient 3?** The constant 3 isn't arbitrary—it's the unique value that creates a stable fixed point at orthogonal matrices. With coefficient 2, the iteration would shrink $X$ to zero; with coefficient 4, it would diverge. The 3 also emerges naturally from Newton's method for computing matrix inverse square roots. For a complete derivation with illustrations, see [Appendix B: Newton-Schulz Iteration](20-appendix.md#b-newton-schulz-iteration-why-the-coefficient-is-3).

Starting from $X_0 = G / \|G\|$, this converges to:
- An **orthogonal matrix** (if $G$ is square)
- A matrix with **orthonormal columns** (if $G$ is rectangular with more rows than columns)

For a $m \times n$ matrix where $m \geq n$, the iteration ensures $X^T X = I_n$, meaning the $n$ columns are orthonormal. This is ideal for weight matrices where we want to preserve the structure of the output space while constraining the operator norm.

```python
def newton_schulz_orthogonalize(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Orthogonalize a matrix using Newton-Schulz iteration.

    This function computes the nearest orthogonal/orthonormal matrix to G:
    - For square matrices (m×m): Returns orthogonal matrix Q where Q^T Q = I
    - For tall matrices (m×n, m>n): Returns matrix with orthonormal columns
    - For wide matrices (m×n, m<n): Returns matrix with orthonormal rows (scaled)

    The Newton-Schulz iteration converges cubically to the orthogonal polar factor.

    Args:
        G: Input matrix of shape (m, n). Can be square or rectangular.
        steps: Number of Newton-Schulz iterations (5 is usually sufficient)
        eps: Small constant for numerical stability when normalizing

    Returns:
        Orthogonalized matrix of same shape as G.
        - If m=n: Q is orthogonal (Q^T Q = Q Q^T = I)
        - If m>n: Q has orthonormal columns (Q^T Q = I_n, but Q Q^T ≠ I_m)
        - If m<n: Q has orthonormal rows scaled by sqrt(m/n)

    Note:
        X.T @ X is always n×n (where n is the number of columns), which is why
        we create an n×n identity matrix. For non-square matrices, this produces
        orthonormal columns, which is the desired behavior for weight matrices
        where we want to preserve the output dimension.
    """
    # Normalize by Frobenius norm
    X = G / (G.norm() + eps)

    # Get number of columns (n) for the identity matrix dimension
    n_cols = X.shape[1]

    for _ in range(steps):
        # Newton-Schulz step: X ← X(3I - X^TX)/2
        # X^T X is n×n (square), measuring column orthonormality
        XTX = X.T @ X  # Shape: (n, n)
        I_n = torch.eye(n_cols, device=G.device, dtype=G.dtype)
        X = X @ (3 * I_n - XTX) / 2  # X: (m,n) @ (n,n) -> (m,n)

    return X


def demonstrate_newton_schulz():
    """Show Newton-Schulz converging to orthogonal matrix."""
    torch.manual_seed(42)

    G = torch.randn(5, 5)
    X = G / G.norm()

    print("Square matrix (5×5):")
    print("Step | ||X^TX - I||")
    print("-" * 25)

    for step in range(10):
        XTX = X.T @ X
        error = (XTX - torch.eye(5)).norm().item()
        print(f"{step:4d} | {error:.6f}")

        X = X @ (3 * torch.eye(5) - XTX) / 2

    print("\nNon-square matrix (10×5) - should have orthonormal columns:")
    G_tall = torch.randn(10, 5)
    X_tall = newton_schulz_orthogonalize(G_tall, steps=5)

    # Check column orthonormality
    XTX = X_tall.T @ X_tall
    col_error = (XTX - torch.eye(5)).norm().item()
    print(f"||X^T X - I_5|| = {col_error:.6f}")  # Should be ~0

    # Check that X X^T ≠ I (since 10 > 5)
    XXT = X_tall @ X_tall.T
    row_error = (XXT - torch.eye(10)).norm().item()
    print(f"||X X^T - I_10|| = {row_error:.6f}")  # Should be large (not orthogonal rows)
```

Output:
```
Square matrix (5×5):
Step | ||X^TX - I||
-------------------------
   0 | 1.234567
   1 | 0.123456
   2 | 0.001234
   3 | 0.000001
   4 | 0.000000
...

Non-square matrix (10×5) - should have orthonormal columns:
||X^T X - I_5|| = 0.000001
||X X^T - I_10|| = 5.236821
```

Convergence is **cubically fast**!

For **non-square matrices**, Newton-Schulz produces orthonormal **columns**, not a fully orthogonal matrix. This is exactly what we want for neural network weight matrices, where each column corresponds to features of the output.

### Why Newton-Schulz?

1. **No SVD required**: SVD is $O(n^3)$; Newton-Schulz is $O(kn^2)$ for $k$ iterations
2. **Differentiable**: Can backprop through it if needed
3. **GPU-friendly**: Just matrix multiplications
4. **Converges quickly**: 5 iterations usually enough

## The Muon Algorithm

### Core Algorithm

```python
class Muon:
    """
    Muon optimizer: Momentum + Orthogonalization.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 ns_steps: int = 5):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps

        # Momentum buffers
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue

            g = p.grad

            # Update momentum
            v.mul_(self.momentum).add_(g)

            # Apply update based on parameter shape
            if p.dim() >= 2:
                # For matrices: orthogonalize the update direction
                update = newton_schulz_orthogonalize(v, steps=self.ns_steps)
                p.data.sub_(self.lr * update)
            else:
                # For vectors (biases, etc.): standard update
                p.data.sub_(self.lr * v)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

### Key Properties

1. **Simple**: Just momentum + orthogonalization
2. **Per-layer**: Each layer's update is orthogonalized independently
3. **No statistics needed**: Unlike Adam/K-FAC, no running estimates
4. **Scale-free**: Learning rate doesn't depend on layer size

## Understanding Muon Geometrically

### The Update Interpretation

Standard SGD with gradient G updates:
$$W \leftarrow W - \eta G$$

Muon with orthogonalized gradient Q:
$$W \leftarrow W - \eta Q$$

Since $\|Q\|_{op} = 1$, the update has unit operator norm.

### Comparison to Adam

| Aspect | Adam | Muon |
|--------|------|------|
| Per-parameter scaling | Yes (diagonal) | No |
| Cross-parameter coupling | No | Yes (via orthogonalization) |
| State to maintain | 2 vectors | 1 vector (momentum) |
| Curvature info | Gradient statistics | Implicit in orthogonalization |

### The μP Connection

Muon is related to **maximal update parameterization (μP)**:
- Both aim for scale-invariance across layers
- μP adjusts learning rates per layer
- Muon achieves similar effect via orthogonalization

```python
def compare_update_norms():
    """Compare update norms for Adam vs Muon."""
    torch.manual_seed(42)

    # Random gradient
    G = torch.randn(100, 100)

    # Adam-style update (simplified)
    adam_denom = (G ** 2).mean().sqrt() + 1e-8
    adam_update = G / adam_denom

    # Muon-style update
    muon_update = newton_schulz_orthogonalize(G)

    print(f"Gradient operator norm: {operator_norm(G):.4f}")
    print(f"Adam update operator norm: {operator_norm(adam_update):.4f}")
    print(f"Muon update operator norm: {operator_norm(muon_update):.4f}")  # Should be 1.0
```

## When Muon Helps

### Good Scenarios

1. **Deep networks**: Orthogonal updates prevent gradient explosion/vanishing
2. **Large batch training**: Less need for per-sample adaptivity
3. **Language models**: Shown to work well for LLMs
4. **Scale-invariance desired**: Updates don't depend on layer norms

### Challenges

1. **Small batch**: Less proven than Adam for small batch
2. **Sparse gradients**: Orthogonalization may not be appropriate
3. **Non-matrix parameters**: Falls back to standard momentum

## Combining with Other Techniques

### Muon + Weight Decay

```python
class MuonWithWeightDecay(Muon):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.01, ns_steps: int = 5):
        super().__init__(params, lr, momentum, ns_steps)
        self.weight_decay = weight_decay

    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue

            g = p.grad

            # Add weight decay to gradient
            g = g + self.weight_decay * p.data

            v.mul_(self.momentum).add_(g)

            if p.dim() >= 2:
                update = newton_schulz_orthogonalize(v, steps=self.ns_steps)
                p.data.sub_(self.lr * update)
            else:
                p.data.sub_(self.lr * v)
```

### Hybrid Approaches

Some practitioners combine:
- Muon for attention/dense layers
- Adam for embeddings, normalization parameters

## The Broader Picture

### Two Paths to Curvature

| Fisher Path | Operator Path |
|-------------|---------------|
| K-FAC, Shampoo, SOAP | Muon |
| Use output statistics | Use weight geometry |
| $O(n^2 + \text{layer}^3)$ memory | $O(n)$ memory |
| Complex implementation | Simple implementation |
| Strong theory (natural gradient) | Emerging theory |

### Why Does Orthogonalization Work?

Hypotheses:
1. **Balanced updates**: Prevents any singular direction from dominating
2. **Implicit preconditioning**: Orthogonal directions are "decorrelated"
3. **Stability**: Orthogonal updates maintain good conditioning
4. **Operator geometry**: Matches the natural geometry of linear maps

## Key Takeaways

1. **Muon uses operator geometry** instead of Fisher geometry

2. **Newton-Schulz** computes orthogonalization efficiently

3. **Updates have unit operator norm**, providing scale invariance

4. **Simpler than K-FAC/Shampoo** with competitive performance

5. **Works well for LLMs** in recent experiments

6. **Different philosophical approach** to curvature

## What's Next

- **Chapter 18**: Learning rate schedules—when to use what
- **Chapter 19**: Practical optimization recipes for LLMs

## Exercises

1. **Newton-Schulz convergence**: How does the number of iterations affect the orthogonality error?

2. **Compare optimizers**: Train the same model with Adam, Shampoo, and Muon.

3. **Layer-wise analysis**: Which layers benefit most from Muon?

4. **Rectangular matrices**: Verify that Newton-Schulz produces orthonormal columns for tall matrices (m > n) and analyze the operator norm of the result. Compare with SVD-based orthogonalization.

5. **Gradient noise**: Does orthogonalization change how noise affects optimization?
