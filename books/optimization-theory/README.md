# Optimization Theory for Deep Learning

A rigorous exploration of optimization in neural networks—from classical methods to modern algorithms, with deep dives into why optimization works at all.

## Why This Book?

Standard ML resources teach you *what* optimizers to use. This book explains *why* they work, covering:

- **Classical foundations**: Newton, Conjugate Gradient, L-BFGS—and why they break at scale
- **Loss landscape theory**: Why deep networks are optimizable despite non-convexity
- **The symmetry debate**: Permutation invariance, equivalent minima, and why the argument is incomplete
- **Modern algorithms**: From SGD to Muon, with the mathematical foundations to understand each

## Prerequisites

This book assumes familiarity with:
- Linear algebra (eigenvalues, matrix decompositions)
- Multivariable calculus (gradients, Hessians)
- Basic probability (expectations, distributions)
- PyTorch fundamentals

For a gentler introduction focused on practical usage, see the [ML Interview Study Guide](../ml-interview-guide/).

## Table of Contents

### Part I: Classical Optimization Theory
1. [Gradient Descent](chapters/01-gradient-descent.md) — Convergence, condition numbers, the baseline
2. [Newton's Method](chapters/02-newton.md) — Quadratic convergence, the Hessian ideal
3. [Conjugate Gradient](chapters/03-conjugate-gradient.md) — Krylov subspaces, solving Ax=b without forming A
4. [Quasi-Newton Methods](chapters/04-quasi-newton.md) — BFGS, L-BFGS, secant approximations
5. [Gauss-Newton](chapters/05-gauss-newton.md) — Nonlinear least squares, connection to Fisher
6. [Why These Break](chapters/06-why-these-break.md) — The deep learning scaling wall

### Part II: The Deep Learning Loss Landscape
7. [High-Dimensional Geometry](chapters/07-high-dimensional.md) — Why low-dimensional intuition fails
8. [Saddle Points](chapters/08-saddle-points.md) — Random matrix theory, critical point index
9. [Symmetry and Equivalent Minima](chapters/09-symmetry.md) — Permutation invariance and its limits
10. [Mode Connectivity](chapters/10-mode-connectivity.md) — Paths between solutions
11. [Why SGD Works](chapters/11-why-sgd-works.md) — Noise, implicit bias, escaping saddles

### Part III: Modern Methods
12. [Momentum and Acceleration](chapters/12-momentum.md) — Polyak, Nesterov, saddle escape
13. [Adaptive Learning Rates](chapters/13-adaptive.md) — AdaGrad through AdamW
14. [Hessian-Free Optimization](chapters/14-hessian-free.md) — CG for the Newton direction
15. [Natural Gradient](chapters/15-natural-gradient.md) — Fisher information geometry
16. [Practical Second-Order](chapters/16-practical-second-order.md) — K-FAC, Shampoo, SOAP
17. [Muon](chapters/17-muon.md) — Operator geometry and spectral norms

### Part IV: Practice
18. [Learning Rate Schedules](chapters/18-schedules.md) — Warmup, decay, WSD
19. [Practical Optimization](chapters/19-practical.md) — Debugging, hyperparameters, recipes
20. [Mathematical Foundations](chapters/20-appendix.md) — Reference material

## Visual Approach

This book emphasizes geometric intuition through extensive SVG illustrations. Abstract concepts like curvature, loss landscapes, and optimization trajectories are visualized to build understanding.

## License

MIT License
