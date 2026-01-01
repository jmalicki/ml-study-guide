# Appendix: Mathematical Foundations

This appendix provides reference material for the mathematical concepts used throughout the book.

## A. Linear Algebra Review

### Eigenvalue Decomposition

For a symmetric matrix $A \in \mathbb{R}^{n \times n}$:

$$A = Q \Lambda Q^T$$

where:
- $Q$ is orthogonal ($Q^T Q = I$)
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ contains eigenvalues
- $Aq_i = \lambda_i q_i$ for eigenvector $q_i$

**Properties**:
- All eigenvalues are real for symmetric matrices
- Eigenvectors are orthogonal
- $A$ is positive definite iff all $\lambda_i > 0$

### Singular Value Decomposition (SVD)

For any matrix $A \in \mathbb{R}^{m \times n}$:

$$A = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{m \times m}$ is orthogonal (left singular vectors)
- $V \in \mathbb{R}^{n \times n}$ is orthogonal (right singular vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with singular values $\sigma_i \geq 0$

**Key relationships**:
- Singular values: $\sigma_i = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(A A^T)}$
- Operator norm: $\|A\|_{op} = \sigma_1$ (largest singular value)
- Frobenius norm: $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$

### Kronecker Product

For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$:

$$A \otimes B = \begin{bmatrix} a_{11}B & \cdots & a_{1n}B \\ \vdots & \ddots & \vdots \\ a_{m1}B & \cdots & a_{mn}B \end{bmatrix} \in \mathbb{R}^{mp \times nq}$$

**Properties**:
- $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$
- $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$
- $\text{vec}(AXB) = (B^T \otimes A)\text{vec}(X)$

The last property is why Kronecker structure appears in K-FAC.

## B. Calculus Review

### Gradient

For $f: \mathbb{R}^n \to \mathbb{R}$:

$$\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

The gradient points in the direction of steepest ascent.

### Hessian

$$H = \nabla^2 f(x) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

**Properties**:
- Symmetric for smooth $f$: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$
- At a minimum: $H \succeq 0$ (positive semi-definite)
- Eigenvalues give curvature in each direction

### Jacobian

For $f: \mathbb{R}^n \to \mathbb{R}^m$:

$$J = \frac{\partial f}{\partial x} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

The Jacobian generalizes the gradient to vector-valued functions.

## C. Convexity and Smoothness

### Convexity

A function $f$ is **convex** if:

$$f(\alpha x + (1-\alpha) y) \leq \alpha f(x) + (1-\alpha) f(y), \quad \forall \alpha \in [0, 1]$$

**Equivalent conditions** (for differentiable $f$):
- $f(y) \geq f(x) + \nabla f(x)^T (y - x)$ (first-order)
- $\nabla^2 f(x) \succeq 0$ for all $x$ (second-order)

### Strong Convexity

$f$ is **μ-strongly convex** if:

$$f(y) \geq f(x) + \nabla f(x)^T (y - x) + \frac{\mu}{2}\|y - x\|^2$$

Equivalently: $\nabla^2 f(x) \succeq \mu I$ for all $x$.

### Smoothness

$f$ has **L-Lipschitz continuous gradients** if:

$$\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\|$$

Equivalently: $\nabla^2 f(x) \preceq L I$ for all $x$.

### Condition Number

For an L-smooth, μ-strongly convex function:

$$\kappa = \frac{L}{\mu}$$

The condition number measures the ratio of maximum to minimum curvature.

## D. Convergence Theory

### Gradient Descent Convergence

**For L-smooth convex functions** with $\eta = 1/L$:

$$f(x_T) - f(x^*) \leq \frac{L \|x_0 - x^*\|^2}{2T}$$

**For L-smooth, μ-strongly convex functions** with $\eta = 1/L$:

$$\|x_T - x^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^T \|x_0 - x^*\|^2$$

### Newton Convergence

For functions with Lipschitz Hessian, near the optimum:

$$\|x_{t+1} - x^*\| \leq C \|x_t - x^*\|^2$$

This is **quadratic convergence**—the error squares each step.

### Momentum Convergence

For L-smooth, μ-strongly convex functions with optimal momentum:

$$f(x_T) - f(x^*) \leq \left(1 - \frac{1}{\sqrt{\kappa}}\right)^T (f(x_0) - f(x^*))$$

This is the **optimal rate** for first-order methods.

## E. Information Theory

### KL Divergence

$$D_{KL}(P \| Q) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

**Properties**:
- $D_{KL}(P \| Q) \geq 0$
- $D_{KL}(P \| Q) = 0$ iff $P = Q$
- Not symmetric: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

### Fisher Information

$$F(\theta) = \mathbb{E}_{x \sim p_\theta}\left[\nabla_\theta \log p_\theta(x) \cdot \nabla_\theta \log p_\theta(x)^T\right]$$

**Equivalent form**:

$$F(\theta) = -\mathbb{E}_{x \sim p_\theta}\left[\nabla_\theta^2 \log p_\theta(x)\right]$$

**Local KL approximation**:

$$D_{KL}(p_\theta \| p_{\theta + d\theta}) \approx \frac{1}{2} d\theta^T F(\theta) d\theta$$

## F. Probability Distributions

### Gaussian Distribution

$$p(x) = \frac{1}{\sqrt{(2\pi)^n |\Sigma|}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$

**Fisher information for Gaussian mean**:

$$F = \Sigma^{-1}$$

### Categorical Distribution

For $p(y = k) = \pi_k$ where $\sum_k \pi_k = 1$:

$$\log p(y = k) = \log \pi_k$$

**Softmax parameterization**: $\pi_k = \frac{\exp(\theta_k)}{\sum_j \exp(\theta_j)}$

## G. Key Identities

### Matrix Calculus

$$\frac{\partial}{\partial X} \text{tr}(AX) = A^T$$

$$\frac{\partial}{\partial X} \text{tr}(X^T A X) = (A + A^T) X$$

$$\frac{\partial}{\partial x} x^T A x = (A + A^T) x$$

For symmetric $A$: $\frac{\partial}{\partial x} x^T A x = 2Ax$

### Woodbury Identity

$$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$$

Useful for low-rank updates to inverses.

### Sherman-Morrison

$$(A + uv^T)^{-1} = A^{-1} - \frac{A^{-1}uv^TA^{-1}}{1 + v^TA^{-1}u}$$

Special case of Woodbury for rank-1 updates.

## H. Numerical Stability

### Floating Point Precision

| Type | Bits | Exponent | Mantissa | Range | Epsilon |
|------|------|----------|----------|-------|---------|
| fp16 | 16 | 5 | 10 | ~6×10⁴ | ~10⁻³ |
| bf16 | 16 | 8 | 7 | ~10³⁸ | ~10⁻² |
| fp32 | 32 | 8 | 23 | ~10³⁸ | ~10⁻⁷ |

**Implications for optimization**:
- bf16: Good range, acceptable precision for gradients
- Loss scaling often needed for fp16
- Master weights in fp32 for stability

### Gradient Clipping

**Global norm clipping**:
$$g' = \begin{cases} g & \text{if } \|g\| \leq c \\ c \cdot g / \|g\| & \text{otherwise} \end{cases}$$

**Per-parameter clipping**:
$$g'_i = \text{clip}(g_i, -c, c)$$

Global norm is generally preferred—preserves relative magnitudes.

## I. Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Parameters |
| $L$ or $\ell$ | Loss function |
| $g$ or $\nabla L$ | Gradient |
| $H$ or $\nabla^2 L$ | Hessian |
| $F$ | Fisher information matrix |
| $J$ | Jacobian |
| $\eta$ | Learning rate |
| $\beta$ | Momentum coefficient |
| $\kappa$ | Condition number |
| $\lambda$ | Eigenvalue or regularization |
| $\sigma$ | Singular value |
| $\|\cdot\|$ | Euclidean norm |
| $\|\cdot\|_F$ | Frobenius norm |
| $\|\cdot\|_{op}$ | Operator (spectral) norm |
| $\otimes$ | Kronecker product |
| $\succeq$ | Positive semi-definite |
| $\mathbb{E}$ | Expectation |

## Further Reading

### Classical Optimization
- Nocedal & Wright, "Numerical Optimization"
- Boyd & Vandenberghe, "Convex Optimization"

### Deep Learning Optimization
- Bottou, Curtis, & Nocedal, "Optimization Methods for Large-Scale Machine Learning"
- Ruder, "An Overview of Gradient Descent Optimization Algorithms"

### Second-Order Methods
- Martens, "Deep Learning via Hessian-free Optimization"
- Martens & Grosse, "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"

### Natural Gradient
- Amari, "Natural Gradient Works Efficiently in Learning"
- Pascanu & Bengio, "Revisiting Natural Gradient for Deep Networks"

### Loss Landscape
- Li et al., "Visualizing the Loss Landscape of Neural Nets"
- Garipov et al., "Loss Surfaces, Mode Connectivity, and Fast Ensembling"

### Modern Optimizers
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW)
- Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam"
