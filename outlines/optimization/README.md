# Part: Optimization Theory

This part provides a comprehensive, mathematically rigorous treatment of optimization for deep learning, building from first principles to modern methods like Muon.

## Chapters

1. [Gradient Descent and Its Limitations](01-gradient-descent.md)
2. [Momentum and Acceleration](02-momentum.md)
3. [Adaptive Learning Rates](03-adaptive.md)
4. [Second-Order Methods](04-second-order.md)
5. [Natural Gradient and Information Geometry](05-natural-gradient.md)
6. [Practical Second-Order Methods](06-practical-second-order.md)
7. [Muon and Operator Geometry](07-muon.md)
8. [Learning Rate Schedules](08-schedules.md)
9. [Practical Optimization for LLMs](09-practical.md)

## Appendix

- [Mathematical Foundations](appendix-math.md)

## Philosophy

The goal is to explain the *why* behind each optimizer, not just the *what*. Each chapter should:

1. Motivate the problem being solved
2. Derive the solution from first principles where possible
3. Show connections to other methods
4. Provide runnable PyTorch implementations
5. Include exercises that test understanding
6. **Include SVG illustrations for geometric/mathematical concepts**

## Visual Approach

Optimization is inherently geometric. Every chapter should include SVG diagrams illustrating:

- Loss landscapes and contour plots
- Gradient directions vs optimal directions
- Manifold geometry (for natural gradient, Muon)
- Algorithm trajectories on example surfaces
- Convergence behavior visualizations
- Matrix structure diagrams (Kronecker, block-diagonal)

**Required illustrations per chapter:**

| Chapter | Key Illustrations Needed |
|---------|-------------------------|
| 01 GD | Ill-conditioned ellipse, oscillation, eigenvalue visualization |
| 02 Momentum | Heavy ball trajectory, Nesterov lookahead, saddle escape |
| 03 Adaptive | Per-parameter scaling, sparse gradient handling |
| 04 Second-Order | Newton step geometry, trust regions, CG iterations |
| 05 Natural Gradient | Statistical manifold, Fisher distortion, KL balls |
| 06 Practical | Kronecker structure, K-FAC factorization, Shampoo preconditioning |
| 07 Muon | Operator norm geometry, Stiefel manifold, Newton-Schulz convergence |
| 08 Schedules | Schedule curves (cosine, WSD, cyclical) |
| 09 Practical | Decision flowcharts, training curves |

## Key Themes

- **Geometry matters**: Parameter space is not Euclidean; different metrics give different "steepest descent" directions
- **Two geometric frameworks**: Fisher information geometry (natural gradient) vs operator norm geometry (Muon)
- **Approximation hierarchy**: Full Hessian → Kronecker → Block-diagonal → Diagonal
- **Theory to practice**: How theoretical insights become practical algorithms
