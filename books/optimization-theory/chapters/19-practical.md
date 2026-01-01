# Chapter 19: Practical Optimization for LLMs

This chapter synthesizes everything we've learned into practical recipes for training large language models.

## The Standard Recipe

### Components

Modern LLM training uses:

1. **Optimizer**: AdamW (or SOAP/Muon for advanced)
2. **Schedule**: WSD (warmup-stable-decay)
3. **Regularization**: Weight decay + dropout (sometimes)
4. **Precision**: Mixed precision (bf16 or fp16)

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

def create_optimizer_and_scheduler(
    model: nn.Module,
    num_training_steps: int,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    warmup_ratio: float = 0.01,
    min_lr_ratio: float = 0.1,
    betas: tuple = (0.9, 0.95),
):
    """
    Standard LLM optimizer setup.
    """
    # Separate weight decay parameters (no WD for biases, LayerNorm)
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'ln' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_groups, lr=learning_rate, betas=betas)

    # WSD schedule
    warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler
```

## Hyperparameter Guidelines

### Learning Rate

| Model Size | Typical LR |
|------------|-----------|
| < 1B | 3e-4 to 1e-3 |
| 1B - 10B | 1e-4 to 3e-4 |
| 10B - 100B | 5e-5 to 1e-4 |
| > 100B | 1e-5 to 5e-5 |

**Rule of thumb**: Larger models need smaller learning rates.

### Weight Decay

Typical values: 0.01 to 0.1

Higher weight decay for:
- Smaller models
- Less data
- Preventing memorization

```python
def weight_decay_heuristic(model_params: int, data_tokens: int) -> float:
    """Heuristic for weight decay selection."""
    # More data relative to parameters → less regularization needed
    ratio = data_tokens / model_params

    if ratio > 100:  # Data-rich regime
        return 0.01
    elif ratio > 10:
        return 0.05
    else:  # Data-limited regime
        return 0.1
```

### Batch Size

Batch size affects:
- Gradient noise (smaller batch = more noise)
- Convergence speed (larger batch = fewer steps)
- Hardware utilization (larger batch = better GPU usage)

```python
def effective_batch_size(micro_batch: int, gradient_accumulation: int,
                         num_gpus: int) -> int:
    """Calculate effective batch size."""
    return micro_batch * gradient_accumulation * num_gpus


def tokens_per_step(batch_size: int, sequence_length: int) -> int:
    """Tokens processed per optimization step."""
    return batch_size * sequence_length
```

## Common Issues and Solutions

### Issue 1: Loss Spikes

**Symptoms**: Sudden increase in loss during training

**Causes**:
- Learning rate too high
- Numerical instability
- Bad data batch
- Gradient explosion

**Solutions**:

```python
def handle_loss_spike(loss: float, prev_losses: list,
                      threshold_multiplier: float = 5.0) -> str:
    """Detect and suggest handling for loss spikes."""
    if len(prev_losses) < 10:
        return "continue"

    avg_loss = sum(prev_losses[-10:]) / 10

    if loss > threshold_multiplier * avg_loss:
        return "spike_detected"

    return "continue"


# Solutions in training loop:
# 1. Skip the update
# 2. Reduce learning rate
# 3. Increase gradient clipping
# 4. Check for NaN and skip
```

### Issue 2: Training Instability

**Symptoms**: Oscillating loss, NaN gradients

**Solutions**:

```python
def stability_checklist(model, optimizer, config):
    """Checklist for training stability."""
    checks = {
        'gradient_clipping': config.get('max_grad_norm', 1.0) is not None,
        'warmup': config.get('warmup_steps', 0) > 0,
        'weight_decay_exclusions': True,  # Check no WD on LayerNorm
        'bf16_or_fp16': config.get('dtype') in ['bf16', 'fp16'],
        'appropriate_lr': config['lr'] < 1e-3,
    }

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
```

### Issue 3: Slow Convergence

**Symptoms**: Loss decreases very slowly

**Causes**:
- Learning rate too low
- Poor initialization
- Suboptimal optimizer settings

**Solutions**:

```python
def diagnose_slow_convergence(loss_history: list, steps: int) -> dict:
    """Diagnose slow convergence."""
    if len(loss_history) < 100:
        return {'status': 'insufficient_data'}

    # Check loss decrease rate
    early_loss = sum(loss_history[:10]) / 10
    recent_loss = sum(loss_history[-10:]) / 10
    decrease = (early_loss - recent_loss) / early_loss

    suggestions = []

    if decrease < 0.1:  # Less than 10% decrease
        suggestions.append("Try increasing learning rate 2-5x")
        suggestions.append("Check if gradients are flowing (not zero)")
        suggestions.append("Verify data preprocessing")

    return {
        'early_loss': early_loss,
        'recent_loss': recent_loss,
        'decrease_pct': decrease * 100,
        'suggestions': suggestions
    }
```

## Monitoring Training

### Key Metrics

```python
class TrainingMonitor:
    """Monitor key training metrics."""

    def __init__(self):
        self.losses = []
        self.grad_norms = []
        self.learning_rates = []
        self.throughput = []  # tokens/second

    def log_step(self, loss: float, grad_norm: float, lr: float,
                 tokens: int, step_time: float):
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)
        self.learning_rates.append(lr)
        self.throughput.append(tokens / step_time)

    def summary(self, last_n: int = 100) -> dict:
        return {
            'avg_loss': sum(self.losses[-last_n:]) / min(len(self.losses), last_n),
            'avg_grad_norm': sum(self.grad_norms[-last_n:]) / min(len(self.grad_norms), last_n),
            'current_lr': self.learning_rates[-1] if self.learning_rates else 0,
            'avg_throughput': sum(self.throughput[-last_n:]) / min(len(self.throughput), last_n),
        }

    def detect_issues(self) -> list:
        issues = []

        if len(self.losses) > 100:
            # Check for loss plateau
            recent = self.losses[-50:]
            if max(recent) - min(recent) < 0.01:
                issues.append("Loss plateau detected")

            # Check for increasing loss
            if self.losses[-1] > self.losses[-50]:
                issues.append("Loss increasing")

        if len(self.grad_norms) > 10:
            # Check for exploding gradients
            if max(self.grad_norms[-10:]) > 100:
                issues.append("Very large gradients")

            # Check for vanishing gradients
            if max(self.grad_norms[-10:]) < 1e-6:
                issues.append("Very small gradients")

        return issues
```

### Gradient Statistics

```python
def compute_gradient_stats(model: nn.Module) -> dict:
    """Compute gradient statistics by layer."""
    stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            stats[name] = {
                'norm': g.norm().item(),
                'mean': g.mean().item(),
                'std': g.std().item(),
                'max': g.abs().max().item(),
                'has_nan': torch.isnan(g).any().item(),
            }

    return stats
```

## Debugging Training

### Common Debugging Steps

1. **Overfit a single batch**: Verify the model can memorize
2. **Check gradients flow**: Ensure no layers have zero gradients
3. **Visualize embeddings**: Are they learning meaningful representations?
4. **Compare to baselines**: Is loss reasonable for the task?

```python
def overfit_single_batch(model, batch, optimizer, steps: int = 100):
    """Test that model can memorize a single batch."""
    model.train()
    initial_loss = None

    for step in range(steps):
        optimizer.zero_grad()
        loss = compute_loss(model, batch)

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Reduction: {(1 - final_loss/initial_loss)*100:.1f}%")

    if final_loss > initial_loss * 0.1:
        print("WARNING: Model may not be learning properly")


def check_gradient_flow(model: nn.Module):
    """Check that gradients are flowing to all parameters."""
    zero_grad_params = []
    nan_grad_params = []

    for name, param in model.named_parameters():
        if param.grad is None:
            zero_grad_params.append(name)
        elif torch.isnan(param.grad).any():
            nan_grad_params.append(name)
        elif param.grad.abs().max() < 1e-10:
            zero_grad_params.append(name)

    if zero_grad_params:
        print(f"Zero gradients: {zero_grad_params[:5]}...")
    if nan_grad_params:
        print(f"NaN gradients: {nan_grad_params[:5]}...")

    return len(zero_grad_params) == 0 and len(nan_grad_params) == 0
```

## The Complete Training Loop

```python
def train_llm(
    model: nn.Module,
    train_dataloader,
    num_epochs: int,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.01,
    log_interval: int = 10,
    eval_interval: int = 500,
    eval_fn=None,
):
    """Complete LLM training loop."""
    num_training_steps = len(train_dataloader) * num_epochs

    optimizer, scheduler = create_optimizer_and_scheduler(
        model, num_training_steps, learning_rate, weight_decay, warmup_ratio
    )

    monitor = TrainingMonitor()
    global_step = 0

    for epoch in range(num_epochs):
        model.train()

        for batch in train_dataloader:
            start_time = time.time()

            # Forward pass
            optimizer.zero_grad()
            loss = compute_loss(model, batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )

            # Check for issues
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Step {global_step}: Invalid loss, skipping")
                continue

            # Optimizer step
            optimizer.step()
            scheduler.step()

            # Logging
            step_time = time.time() - start_time
            tokens = batch['input_ids'].numel()
            current_lr = scheduler.get_last_lr()[0]

            monitor.log_step(loss.item(), grad_norm.item(), current_lr,
                           tokens, step_time)

            if global_step % log_interval == 0:
                summary = monitor.summary()
                print(f"Step {global_step}: loss={summary['avg_loss']:.4f}, "
                      f"lr={summary['current_lr']:.2e}, "
                      f"tokens/s={summary['avg_throughput']:.0f}")

            # Evaluation
            if eval_fn and global_step % eval_interval == 0:
                eval_loss = eval_fn(model)
                print(f"Step {global_step}: eval_loss={eval_loss:.4f}")

            global_step += 1

    return model
```

## Key Takeaways

1. **AdamW + WSD** is the standard recipe

2. **Gradient clipping** is essential for stability

3. **Monitor everything**: Loss, gradients, learning rate

4. **Debug systematically**: Overfit first, then scale

5. **Hyperparameters depend on scale**: Bigger models need smaller LR

6. **Weight decay goes on weights only**, not biases/norms

## Exercises

1. **Implement the training loop**: Train a small transformer from scratch.

2. **Hyperparameter sweep**: Find optimal LR and weight decay for a task.

3. **Debug a broken training**: Intentionally break training and diagnose.

4. **Compare optimizers**: Run identical training with AdamW, SOAP, Muon.
