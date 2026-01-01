# Chapter 9: Practical Optimization for LLMs

This chapter synthesizes everything into practical guidance for training large language models.

## Table of Contents

1. [Optimizer Selection](#optimizer-selection)
2. [Recommended Configurations](#recommended-configurations)
3. [Hyperparameter Sensitivity](#hyperparameter-sensitivity)
4. [Gradient Clipping](#gradient-clipping)
5. [Debugging Training Issues](#debugging-training-issues)
6. [Common Mistakes](#common-mistakes)
7. [Monitoring](#monitoring)

---

## Optimizer Selection

### Decision Tree

```
Is it a 2D weight matrix (not embedding)?
├── Yes → Consider Muon
│   └── Memory very constrained? → AdamW
│   └── Otherwise → Muon + AdamW hybrid
└── No → AdamW

Is training time more important than memory?
├── Yes, with large batch → Consider Shampoo
└── No → AdamW
```

### Default Recommendation

**For most LLM training: AdamW with WSD schedule**

- Well-understood, robust
- Standard hyperparameters work across scales
- Easy to debug

---

## Recommended Configurations

### Standard LLM (GPT-style)

```python
optimizer = AdamW(
    model.parameters(),
    lr=3e-4,           # Scale with model size
    betas=(0.9, 0.95), # Slightly lower β₂
    weight_decay=0.1,
    eps=1e-8
)

scheduler = WSD(
    warmup_steps=2000,
    stable_ratio=0.8,
    min_lr_ratio=0.1
)
```

### With Muon

```python
# Separate parameter groups
weight_params = [p for n, p in model.named_parameters()
                 if 'weight' in n and p.ndim == 2
                 and 'embed' not in n]
other_params = [p for n, p in model.named_parameters()
                if p not in weight_params]

muon_opt = Muon(weight_params, lr=0.02, momentum=0.95)
adam_opt = AdamW(other_params, lr=3e-4, weight_decay=0.1)
```

---

## Hyperparameter Sensitivity

| Parameter | Sensitivity | Typical Range |
|-----------|-------------|---------------|
| Learning rate | Very High | 1e-4 to 1e-3 |
| Warmup steps | Medium | 0.5-5% of total |
| Weight decay | Medium | 0.01 to 0.3 |
| $\beta_1$ (momentum) | Low | 0.9 to 0.95 |
| $\beta_2$ (Adam) | Low-Medium | 0.95 to 0.999 |
| Batch size | High (with LR) | 256 to 4M tokens |

### Learning Rate Scaling

- **Model size**: Smaller models can use higher LR
- **Batch size**: Linear scaling rule (double batch → double LR)
- **μP**: Principled scaling with model width

---

## Gradient Clipping

**Always use gradient clipping for LLMs**:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Why It Matters

- Prevents exploding gradients from bad batches
- Essential for stability in long training runs
- Typical threshold: 1.0 for LLMs

---

## Debugging Training Issues

### Loss Spikes

**Symptoms**: Sudden loss increase, possibly recovery

**Causes**:
1. Bad data batch (corrupt, outlier)
2. Learning rate too high
3. Numerical overflow (especially FP16)

**Solutions**:
1. Add data filtering
2. Reduce learning rate
3. Use BF16 instead of FP16

### Loss Plateau

**Symptoms**: Loss stops decreasing but gradient norm is high

**Causes**:
1. Stuck near saddle point
2. Learning rate too low
3. Optimizer state corrupted

**Solutions**:
1. Increase learning rate briefly
2. Check optimizer initialization
3. Add noise (increase batch variation)

### Divergence

**Symptoms**: Loss goes to infinity or NaN

**Causes**:
1. Learning rate too high
2. Missing normalization
3. Numerical instability

**Solutions**:
1. Reduce learning rate by 10×
2. Check architecture (LayerNorm present?)
3. Switch to BF16

---

## Common Mistakes

1. **Using L2 regularization instead of weight decay in Adam**
   - Use AdamW with decoupled weight decay

2. **Wrong learning rate for batch size**
   - Use linear scaling rule

3. **Forgetting warmup**
   - Always use warmup, especially for large batches

4. **Not clipping gradients**
   - Always clip for LLM training

5. **Over-tuning on small models**
   - Hyperparameters often don't transfer to larger scales

6. **Ignoring gradient norm monitoring**
   - Plot gradient norms to catch issues early

---

## Monitoring

### Essential Metrics

```python
# Every step
log({
    'train/loss': loss,
    'train/lr': scheduler.get_lr(),
    'train/grad_norm': grad_norm,
})

# Every eval
log({
    'eval/loss': eval_loss,
    'eval/perplexity': math.exp(eval_loss),
})
```

### Warning Signs

| Metric | Warning Sign | Likely Issue |
|--------|-------------|--------------|
| Loss | Sudden spike | Bad batch or LR too high |
| Grad norm | Consistent spikes | Instability |
| Grad norm | Near zero | Stuck or vanishing gradients |
| Val loss | Increasing | Overfitting |

### Gradient Norm Distribution

Track per-layer gradient norms:
- Some layers should have larger gradients (output layer)
- Near-zero gradients in early layers = vanishing gradient
- Huge variance = instability

---

## Quick Reference

### Default AdamW Settings

```python
AdamW(
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)
```

### Default WSD Schedule

```python
warmup = 0.05 * total_steps
stable = 0.80 * total_steps
decay  = 0.15 * total_steps
```

### Gradient Clipping

```python
max_norm = 1.0
```

### Batch Size Guidelines

| Model Size | Batch Size (tokens) |
|-----------|-------------------|
| 125M | 256K - 512K |
| 1B | 1M - 2M |
| 7B | 2M - 4M |
| 70B | 4M - 8M |

---

## Connections

- **Previous**: [Learning Rate Schedules](08-schedules.md)
- **Appendix**: [Mathematical Foundations](appendix-math.md)
- **Related**: Distributed training chapter for scaling
