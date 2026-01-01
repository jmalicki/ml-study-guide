# Chapter 14: Hessian-Free Optimization

Hessian-free optimization uses conjugate gradient (Chapter 3) to compute Newton steps without forming the Hessian. It's a bridge between classical second-order methods and practical deep learning.

## The Core Idea

### Newton Without Inversion

Newton's method requires solving:

$$H \delta = -g$$

Hessian-free ([Martens, 2010](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)) uses CG to solve this iteratively using only Hessian-vector products:

$$Hv = \frac{\partial}{\partial t}[\nabla f(\theta + tv)]_{t=0} = \lim_{t \to 0} \frac{\nabla f(\theta + tv) - \nabla f(\theta)}{t}$$

This is the **directional derivative** of the gradient along direction $v$. See [Appendix C](20-appendix.md#directional-derivatives) for background on directional derivatives and finite difference approximations.

```python
import torch
import torch.nn as nn
from typing import Callable, Tuple, List

def hessian_vector_product(
    loss_fn: Callable[[], torch.Tensor],
    params: List[torch.Tensor],
    v: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Compute Hv using two backward passes.

    Args:
        loss_fn: Function that computes the loss
        params: List of parameters
        v: Vector to multiply with Hessian

    Returns:
        Hv as a list of tensors matching params shapes
    """
    # First backward: compute gradient
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Compute g^T v
    gv = sum((g * v_i).sum() for g, v_i in zip(grads, v))

    # Second backward: differentiate g^T v to get Hv
    Hv = torch.autograd.grad(gv, params)

    return list(Hv)


def conjugate_gradient_hvp(
    hvp_fn: Callable[[List[torch.Tensor]], List[torch.Tensor]],
    b: List[torch.Tensor],
    max_iter: int = 10,
    tol: float = 1e-6,
    damping: float = 1e-2
) -> List[torch.Tensor]:
    """
    Conjugate gradient using Hessian-vector products.

    Solves (H + damping * I) x = b

    Args:
        hvp_fn: Function computing Hessian-vector products
        b: Right-hand side (negative gradient)
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        damping: Tikhonov regularization

    Returns:
        Approximate solution x
    """
    # Initialize
    x = [torch.zeros_like(b_i) for b_i in b]
    r = [b_i.clone() for b_i in b]  # r = b - Ax, x=0 so r=b
    p = [r_i.clone() for r_i in r]

    def dot(a, b):
        return sum((a_i * b_i).sum() for a_i, b_i in zip(a, b))

    def add_scaled(a, b, scale):
        return [a_i + scale * b_i for a_i, b_i in zip(a, b)]

    rs_old = dot(r, r)

    for i in range(max_iter):
        # Compute (H + λI)p
        Hp = hvp_fn(p)
        Hp_damped = [Hp_i + damping * p_i for Hp_i, p_i in zip(Hp, p)]

        pHp = dot(p, Hp_damped)
        alpha = rs_old / (pHp + 1e-10)

        x = add_scaled(x, p, alpha)
        r = add_scaled(r, Hp_damped, -alpha)

        rs_new = dot(r, r)

        if rs_new.sqrt() < tol:
            break

        beta = rs_new / (rs_old + 1e-10)
        p = add_scaled(r, p, beta)

        rs_old = rs_new

    return x
```

## The Hessian-Free Algorithm

### Complete Algorithm

```python
class HessianFreeOptimizer:
    """
    Hessian-free optimization for neural networks.
    """
    def __init__(self, model: nn.Module, damping: float = 1.0,
                 cg_iters: int = 10):
        self.model = model
        self.params = list(model.parameters())
        self.damping = damping
        self.cg_iters = cg_iters

    def step(self, loss_fn: Callable[[], torch.Tensor]):
        """
        Take a Hessian-free optimization step.

        Args:
            loss_fn: Function that computes the loss
        """
        # Compute gradient
        loss = loss_fn()
        grads = torch.autograd.grad(loss, self.params, create_graph=True)
        grads = list(grads)

        # Define HVP function
        def hvp_fn(v):
            gv = sum((g * v_i).sum() for g, v_i in zip(grads, v))
            Hv = torch.autograd.grad(gv, self.params, retain_graph=True)
            return list(Hv)

        # Solve for Newton direction using CG
        neg_grad = [-g.detach() for g in grads]
        direction = conjugate_gradient_hvp(
            hvp_fn, neg_grad,
            max_iter=self.cg_iters,
            damping=self.damping
        )

        # Apply update
        with torch.no_grad():
            for p, d in zip(self.params, direction):
                p.add_(d)

        return loss.item()
```

### Key Components

1. **HVP via autodiff**: Compute $Hv$ in $O(n)$ time using two backward passes

2. **Truncated CG**: Only run 10-50 iterations, not full convergence

3. **Damping**: Add $\lambda I$ for numerical stability

4. **Gauss-Newton approximation** (optional): Use $J^TJ$ instead of full Hessian

## Gauss-Newton vs Full Hessian

### The GN Approximation

For least-squares losses, the Gauss-Newton matrix is often better than the full Hessian:

$$H_{GN} = J^T J \succeq 0 \quad \text{(always positive semi-definite)}$$
$$H_{full} = J^T J + \sum_i r_i \nabla^2 r_i \quad \text{(may have negative eigenvalues)}$$

```python
def gauss_newton_vector_product(
    model: nn.Module,
    loss_fn: Callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    v: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    Compute Gauss-Newton vector product: (J^T J) v

    This is always PSD, unlike the full Hessian.

    The computation has two parts:
    1. Jv (Jacobian-vector product): How outputs change when params move in direction v
    2. J^T(Jv) (vector-Jacobian product): Backprop the output change to get param change

    We use finite differences for Jv because PyTorch's autodiff is reverse-mode,
    which gives us J^T u (VJP) efficiently but not Jv (JVP) directly.
    See Appendix C for finite difference background.
    """
    # Forward pass
    outputs = model(inputs)
    params = list(model.parameters())

    # Step 1: Compute Jv using finite differences
    # Jv = d/dt[model(inputs; params + t*v)]|_{t=0}
    #    ≈ (model(inputs; params + ε*v) - model(inputs; params)) / ε
    eps = 1e-4
    with torch.no_grad():
        # Save original params
        original = [p.clone() for p in params]

        # Perturb params in direction v
        for p, v_i in zip(params, v):
            p.add_(eps * v_i)

        outputs_plus = model(inputs)

        # Restore original params
        for p, orig in zip(params, original):
            p.data = orig

    Jv = (outputs_plus - outputs) / eps  # Shape: same as outputs

    # Step 2: Compute J^T(Jv) using autodiff (reverse mode)
    # This is a standard VJP: backprop Jv through the model
    loss = (outputs * Jv.detach()).sum()
    JTJv = torch.autograd.grad(loss, params)

    return list(JTJv)
```

> **Note**: PyTorch 2.0+ provides `torch.func.jvp` for forward-mode autodiff, which can replace the finite difference approximation with an exact JVP. See the [PyTorch documentation](https://pytorch.org/docs/stable/func.api.html#torch.func.jvp).

## Practical Considerations

### Damping Strategy

The damping parameter $\lambda$ is critical:
- Too small: CG diverges, bad step
- Too large: Reverts to gradient descent

**Levenberg-Marquardt damping**: Adjust $\lambda$ based on how well the step works.

```python
class HFWithAdaptiveDamping:
    """Hessian-free with Levenberg-Marquardt style damping."""

    def __init__(self, model: nn.Module, damping: float = 1.0):
        self.model = model
        self.params = list(model.parameters())
        self.damping = damping

    def step(self, loss_fn: Callable) -> float:
        # Get current loss
        loss = loss_fn()
        current_loss = loss.item()

        # Compute Newton step with current damping
        grads = torch.autograd.grad(loss, self.params, create_graph=True)

        def hvp_fn(v):
            gv = sum((g * v_i).sum() for g, v_i in zip(grads, v))
            return list(torch.autograd.grad(gv, self.params, retain_graph=True))

        neg_grad = [-g.detach() for g in grads]
        direction = conjugate_gradient_hvp(
            hvp_fn, neg_grad, max_iter=20, damping=self.damping
        )

        # Save current parameters
        with torch.no_grad():
            saved = [p.clone() for p in self.params]

        # Try the step
        with torch.no_grad():
            for p, d in zip(self.params, direction):
                p.add_(d)

        # Evaluate new loss
        new_loss = loss_fn().item()

        # Compute predicted reduction
        with torch.no_grad():
            Hd = hvp_fn(direction)
            pred_reduction = -sum((g * d).sum() for g, d in zip(grads, direction))
            pred_reduction -= 0.5 * sum((Hd_i * d).sum() for Hd_i, d in zip(Hd, direction))

        actual_reduction = current_loss - new_loss
        rho = actual_reduction / (pred_reduction.item() + 1e-10)

        if rho < 0.25:
            # Bad step: increase damping, reject
            self.damping *= 2
            with torch.no_grad():
                for p, s in zip(self.params, saved):
                    p.data = s
            return current_loss
        else:
            # Good step: possibly decrease damping
            if rho > 0.75:
                self.damping = max(self.damping / 2, 1e-8)
            return new_loss
```

### CG Termination

When to stop CG:
1. **Fixed iterations**: Simple, common (10-50)
2. **Residual threshold**: Stop when $\|r\| < \epsilon$
3. **Negative curvature**: Stop if $p^T H p < 0$ (for non-GN)

### Cost Analysis

Per step:
- 1 forward pass
- 1 backward pass (gradient)
- k backward passes (HVPs, k = CG iterations)

Total: ~(k+1) backward passes per step, compared to 1 for SGD.

If k=20: Each HF step costs 20x an SGD step, but may make much more progress.

## When Hessian-Free Helps

### Good Scenarios

1. **Full-batch training**: HVP quality degrades with minibatches
2. **Well-conditioned problems**: CG converges fast
3. **Second-order structure is informative**: Curvature helps navigation

### Bad Scenarios

1. **Stochastic gradients**: HVPs are noisy
2. **Very ill-conditioned**: CG needs too many iterations
3. **Memory constrained**: Storing CG vectors is expensive

## Historical Importance

Martens demonstrated that ([Martens, 2010](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)):
- Hessian-free could train deep networks
- When others said "deep networks can't be trained"
- Showed that curvature information helps

This influenced:
- The deep learning renaissance
- Understanding of optimization difficulties
- Development of K-FAC and other methods

## Key Takeaways

1. **Hessian-free uses CG** to compute Newton steps without forming H

2. **HVPs cost $O(n)$** via autodiff—same as gradient

3. **Truncated CG** gives approximate Newton directions cheaply

4. **Damping is critical** for stability

5. **Gauss-Newton is often better** than full Hessian

6. **Stochastic gradients are problematic** for HVP quality

7. **Historically important** for understanding deep optimization

## What's Next

- **Chapter 15**: Natural gradient—using Fisher information geometry
- Chapter 15 connects the Gauss-Newton approximation to statistical geometry

## Exercises

1. **Implement HF**: Train a small network using Hessian-free. Compare to SGD.

2. **CG iteration study**: How does performance change with 5, 10, 20, 50 CG iterations?

3. **Damping sensitivity**: What happens with damping = 0.001, 0.1, 1.0, 10.0?

4. **Full vs GN**: Compare using full Hessian vs Gauss-Newton on a regression task.
