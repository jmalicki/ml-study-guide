# Chapter: Practical Optimization for LLMs

## Overview

This chapter synthesizes everything into practical guidance for training LLMs, including optimizer selection, hyperparameter tuning, debugging training issues, and scaling considerations.

## Sections

### 1. Optimizer Selection Decision Tree

```
Is it a weight matrix (2D, not embedding)?
├── Yes: Consider Muon
│   └── Is memory very constrained? → AdamW
│   └── Otherwise → Muon + AdamW hybrid
└── No: Use AdamW

Is training time more important than memory?
├── Yes, and batch is large: Consider Shampoo
└── No: Stick with AdamW
```

### 2. Recommended Configurations

**Standard LLM Training (GPT-style)**:
```python
optimizer = AdamW(
    lr=3e-4,           # or 6e-4 for small models
    betas=(0.9, 0.95), # slightly lower β₂ than default
    weight_decay=0.1,
    eps=1e-8
)
scheduler = WarmupCosineDecay(
    warmup_steps=2000,
    total_steps=100000,
    min_lr=3e-5
)
```

**With Muon**:
```python
muon_optimizer = Muon(
    weight_params,
    lr=0.02,           # Much higher than Adam
    momentum=0.95
)
adamw_optimizer = AdamW(
    other_params,
    lr=3e-4,
    weight_decay=0.1
)
```

### 3. Hyperparameter Sensitivity

| Hyperparameter | Sensitivity | Typical Range |
|----------------|-------------|---------------|
| Learning rate | Very high | 1e-4 to 1e-3 |
| Warmup steps | Medium | 0.5-5% of total |
| Weight decay | Medium | 0.01 to 0.3 |
| $\beta_1$ (momentum) | Low | 0.9 to 0.95 |
| $\beta_2$ (Adam) | Low-Medium | 0.95 to 0.999 |
| Batch size | High (with LR) | 256 to 4096 |

### 4. Gradient Clipping

- **Max norm clipping**: Scale gradient if ||g|| > threshold
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```
- Why: Prevents exploding gradients from bad batches
- Typical threshold: 1.0 for LLMs
- Always clip for large models
- Monitor gradient norm during training

### 5. Loss Spikes and Training Instabilities

**Symptoms**:
- Sudden increase in loss
- NaN gradients
- Oscillating metrics

**Common Causes**:
1. Learning rate too high
2. Bad data batch (corrupt, outlier)
3. Numerical overflow (especially FP16)
4. Architecture issues (missing LayerNorm, etc.)

**Solutions**:
1. Reduce learning rate
2. Increase gradient clipping threshold
3. Add data filtering/cleaning
4. Use BF16 instead of FP16
5. Check for architectural bugs

### 6. Debugging Checklist

```
□ Gradient norms reasonable? (not 0, not huge)
□ Learning rate schedule correct? (warmup happening)
□ Weight decay applied correctly? (decoupled)
□ Loss decreasing on training data? (sanity check)
□ Validation loss tracking? (overfitting detection)
□ Gradient clipping triggering? (too often = LR too high)
□ Memory usage stable? (no leaks)
□ Throughput stable? (no dataloader bottlenecks)
```

### 7. Optimizer State Sharding (ZeRO Stage 1)

- Adam state: 2× model size (momentum + variance)
- With 8 GPUs: Each GPU holds 1/8 of optimizer state
- Gather gradients, scatter states
- Memory reduction: 8× for optimizer state
- Always use for large model training

### 8. Mixed Precision Training

- Compute in FP16/BF16, accumulate in FP32
- Master weights in FP32 for optimizer
- Loss scaling for FP16 (not needed for BF16)
- **BF16 preferred**: Larger dynamic range, no loss scaling

### 9. μP: Maximal Update Parameterization

- Problem: Optimal hyperparameters change with model width
- Solution: Parameterize so that update magnitudes are stable
- Key rules:
  - Embedding LR scales as 1/width
  - Attention LR scales as 1/width
  - Output LR scales as 1/width
- Benefit: Tune on small model, transfer to large model

### 10. Batch Size Considerations

- **Gradient accumulation**: Simulate large batch with small memory
- **Effective batch size**: micro_batch × accumulation_steps × num_GPUs
- **Critical batch size**: Beyond this, diminishing returns
- Rule of thumb: 2M-4M tokens per batch for LLMs

### 11. Checkpointing Strategy

- Save optimizer state (not just model weights)
- Checkpoint frequency: Every 1000-5000 steps
- Keep multiple checkpoints (last N, best validation)
- Enable training resumption from any checkpoint
- Consider sharded checkpointing for large models

### 12. Common Mistakes

1. **Forgetting weight decay on embeddings** (should have it)
2. **Using L2 regularization instead of decoupled weight decay**
3. **Wrong learning rate for batch size** (not scaling properly)
4. **Ignoring warmup** (training fails early)
5. **Not monitoring gradient norms** (miss instabilities)
6. **Over-tuning on small models** (doesn't transfer)

### 13. Monitoring and Logging

Essential metrics to track:
- Loss (train and validation)
- Gradient norm (per-layer and global)
- Learning rate (verify schedule)
- Throughput (tokens/second)
- Memory usage (GPU utilization)

Warning signs:
- Gradient norm spikes
- Loss plateau with high gradient norm (stuck in saddle)
- Validation loss diverging from training (overfitting)

## Code

```python
def create_optimizer(model, config):
    """Create optimizer with proper parameter groups."""

def create_scheduler(optimizer, config):
    """Create learning rate scheduler."""

class TrainingMonitor:
    """Track and log training metrics."""

def diagnose_training_issue(metrics_history):
    """Automated diagnosis of common problems."""

# Full training loop with best practices
# Checkpoint save/load utilities
# Gradient norm monitoring
```

## Exercises

1. Implement a complete training loop with AdamW + WSD schedule
2. Add gradient clipping and monitoring, intentionally trigger instability
3. Implement ZeRO Stage 1 optimizer sharding (conceptual, with fake communication)
4. Compare training with and without warmup on a small model
5. Implement the linear scaling rule: double batch size, double learning rate

## Connections

- Synthesizes all previous chapters
- Related: Distributed training chapter (scaling to multiple GPUs)
- Related: Mixed precision in hardware chapter
