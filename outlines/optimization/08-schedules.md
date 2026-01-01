# Chapter: Learning Rate Schedules

## Overview

Learning rate schedules control how the learning rate changes during training. The right schedule can significantly impact both convergence speed and final performance.

## Sections

### 1. Why Schedules Matter

- Fixed learning rate: too high → diverge, too low → slow
- Early training: can use large learning rate (far from optimum)
- Late training: need small learning rate (fine-tuning near optimum)
- Schedules automate this transition

### 2. Learning Rate Warmup

- Problem: Large initial learning rate causes training instability
- Cause: Random initialization → large gradients → large updates → chaos
- Solution: Start with small η, gradually increase
- **Linear warmup**: η(t) = η_max · (t / T_warmup) for t < T_warmup
- Typical warmup: 1-5% of total training steps
- Especially important for: Adam (moment estimates need time to stabilize), large batches

### 3. Step Decay

- Simple schedule: divide η by factor every N epochs
- Example: η₀ = 0.1, multiply by 0.1 at epochs 30, 60, 90
- Pros: Simple, interpretable
- Cons: Discontinuous, requires tuning schedule points
- Historical: Popular in computer vision (ResNet papers)

### 4. Exponential Decay

- Continuous decay: η(t) = η₀ · γᵗ where γ < 1
- Or: η(t) = η₀ · exp(-λt)
- Smoother than step decay
- Problem: Decays too fast early, too slow late
- Less common in modern LLM training

### 5. Cosine Annealing

- Smooth decay following cosine curve:
  ```
  η(t) = η_min + (η_max - η_min) · (1 + cos(πt/T)) / 2
  ```
- Starts at η_max, ends at η_min (often 0 or η_max/10)
- Derived from simulated annealing theory
- **The standard for LLM training**

### 6. Cosine with Warm Restarts

- Multiple cosine cycles: "restart" to high learning rate
- Each cycle can have same or increasing period
- Motivation: Escape local minima, explore loss landscape
- Less common for LLMs (single long training run)

### 7. Warmup-Stable-Decay (WSD)

- Modern schedule for LLMs:
  ```
  Phase 1 (warmup): Linear increase from 0 to η_max
  Phase 2 (stable): Constant at η_max (majority of training)
  Phase 3 (decay): Cosine decay to η_min
  ```
- Typical split: 5% warmup, 80% stable, 15% decay
- Advantage: Most training at high learning rate
- Used by: LLaMA, many recent LLMs

### 8. Polynomial Decay

- η(t) = (η_max - η_min) · (1 - t/T)^p + η_min
- p = 1: Linear decay
- p = 2: Quadratic decay (faster initial, slower final)
- Less common but occasionally used

### 9. Cyclical Learning Rates

- Triangle wave between η_min and η_max
- Theory: Saddle point escape via high LR periods
- **Super-convergence**: Fast training with very high LR + cycles
- Less common for transformers (cosine/WSD work well)

### 10. Schedule-Free Optimization

- Recent research: Eliminate schedules entirely
- Methods:
  - Averaging: Keep running average of iterates
  - Primal averaging: D-Adaptation, Prodigy
- Advantages: No schedule tuning, works across learning rates
- Status: Promising but not yet standard

### 11. Hyperparameter Transfer

- Problem: Optimal η depends on model size, batch size, architecture
- **μP (maximal update parameterization)**: Scale η with model width
- **Learning rate transfer**: Find η on small model, scale to large
- Key insight: η_effective should be constant across scales

### 12. Batch Size and Learning Rate

- Linear scaling rule: η ∝ batch_size (within limits)
- Square root scaling: η ∝ √batch_size (more conservative)
- LARS/LAMB: Layer-wise scaling for very large batches
- Critical batch size: Above this, more data doesn't help per step

### 13. Schedule Interactions with Optimizers

- Adam: Less sensitive to schedule (adaptive rates help)
- SGD+momentum: Very sensitive to schedule
- Shampoo: Can use larger learning rates, shorter schedules
- Muon: Different learning rate scale (typically larger than Adam)

## Code

```python
class LRScheduler:
    """Base class for learning rate schedulers."""

class WarmupCosineScheduler(LRScheduler):
    """Warmup + cosine decay."""

class WSDScheduler(LRScheduler):
    """Warmup-Stable-Decay schedule."""

def linear_warmup(step, warmup_steps, max_lr):
    """Linear warmup function."""

def cosine_decay(step, total_steps, max_lr, min_lr=0):
    """Cosine decay function."""

# Visualization of different schedules
# Training comparison with different schedules
```

## Key Equations

- Linear warmup: η(t) = η_max · t / T_warmup
- Cosine decay: η(t) = η_min + (η_max - η_min) · (1 + cos(πt/T)) / 2
- WSD: Piecewise combination of warmup, constant, cosine
- Linear scaling: η' = η · (B' / B)

## Exercises

1. Implement warmup + cosine decay scheduler
2. Compare WSD vs pure cosine on a training run
3. Demonstrate the importance of warmup with large batch size
4. Implement the linear scaling rule and verify empirically
5. Train with cyclical LR, visualize loss landscape exploration

## Connections

- Back: Interacts with all optimizers (Chapters 2-7)
- Forward: Practical training combines schedule + optimizer (Chapter 9)
- Related: Scaling laws chapter (compute-optimal training)
