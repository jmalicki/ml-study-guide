# Chapter 9: The Transformer Block

The transformer block is the fundamental building unit of modern LLMs. Understanding its components - layer normalization, feed-forward networks, residual connections, and their arrangement - is essential for implementing and debugging transformer models. This chapter covers the architecture choices that make transformers trainable at scale.

## Table of Contents

1. [Overview](#overview)
2. [Layer Normalization](#layer-normalization)
   - [LayerNorm](#layernorm)
   - [RMSNorm](#rmsnorm)
   - [Implementation and Comparison](#implementation-and-comparison)
3. [Feed-Forward Networks](#feed-forward-networks)
   - [FFN Structure](#ffn-structure)
   - [Implementation](#ffn-implementation)
4. [Residual Connections](#residual-connections)
   - [Why Residuals Matter](#why-residuals-matter)
   - [Gradient Flow Analysis](#gradient-flow-analysis)
5. [Pre-Norm vs Post-Norm](#pre-norm-vs-post-norm)
   - [Post-Norm Architecture](#post-norm-architecture)
   - [Pre-Norm Architecture](#pre-norm-architecture)
   - [Why Pre-Norm is Standard](#why-pre-norm-is-standard)
6. [Complete Transformer Block Implementation](#complete-transformer-block-implementation)
7. [Visualization and Analysis](#visualization-and-analysis)
8. [Modern Variants](#modern-variants)
9. [Exercises](#exercises)

---

## Overview

A transformer block processes input sequences through two main sub-layers:

1. **Self-Attention Layer**: Captures relationships between tokens (covered in [Multi-Head Attention](04-multi-head-attention.md))
2. **Feed-Forward Network (FFN)**: Processes each position independently

Each sub-layer is wrapped with:
- **Residual connections**: Enable gradient flow in deep networks
- **Layer normalization**: Stabilize training

The arrangement of these components (pre-norm vs post-norm) significantly affects training stability and performance.

---

## Layer Normalization

Layer normalization normalizes activations across the feature dimension for each example independently. This is crucial for stable training of deep transformers.

### LayerNorm

**Original formulation** from [Ba et al., 2016](https://arxiv.org/abs/1607.06450):

For input $\mathbf{x} \in \mathbb{R}^d$:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

where:
- $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ (mean)
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$ (variance)
- $\gamma, \beta \in \mathbb{R}^d$ are learned scale and shift parameters
- $\epsilon$ is a small constant for numerical stability (typically $10^{-5}$ or $10^{-6}$)

**Key properties:**
- Normalizes across features (d), not batch dimension
- Each example normalized independently
- Invariant to scale and shift transformations

### RMSNorm

**Root Mean Square Normalization** from [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467), used in modern LLMs (LLaMA, GPT-4, etc.):

$$
\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}
$$

where:

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}
$$

**Differences from LayerNorm:**
- No mean centering (removes $\mu$)
- No bias term ($\beta$)
- ~10-15% faster computation
- Empirically similar or better performance

**Why RMSNorm works:**
- Re-centering contributes little to training stability
- Scale normalization is the key factor
- Simpler computation, fewer parameters

### Implementation and Comparison

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class LayerNorm(nn.Module):
    """Standard Layer Normalization with learned affine parameters."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale and shift
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            normalized: shape (batch, seq_len, d_model)
        """
        # Compute mean and variance across d_model dimension
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch, seq_len, 1)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """Root Mean Square Normalization - more efficient than LayerNorm."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Only learnable scale, no bias
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            normalized: shape (batch, seq_len, d_model)
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        return self.gamma * (x / rms)


# Optimized RMSNorm (closer to production implementations)
class RMSNormOptimized(nn.Module):
    """Optimized RMSNorm using rsqrt for better performance."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Using rsqrt is faster than 1/sqrt
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.gamma * norm


# Comparison
def compare_normalizations():
    """Compare LayerNorm and RMSNorm behavior."""
    d_model = 512
    batch_size = 2
    seq_len = 10

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize normalizations
    ln = LayerNorm(d_model)
    rms = RMSNorm(d_model)

    # Forward pass
    ln_out = ln(x)
    rms_out = rms(x)

    print("Input statistics:")
    print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
    print("\nLayerNorm output statistics:")
    print(f"  Mean: {ln_out.mean():.6f}, Std: {ln_out.std():.4f}")
    print("\nRMSNorm output statistics:")
    print(f"  Mean: {rms_out.mean():.4f}, Std: {rms_out.std():.4f}")

    # Check per-example statistics
    print("\nPer-example statistics (first example, first position):")
    print(f"  LayerNorm - Mean: {ln_out[0, 0].mean():.6f}, Std: {ln_out[0, 0].std():.4f}")
    print(f"  RMSNorm - Mean: {rms_out[0, 0].mean():.4f}, Std: {rms_out[0, 0].std():.4f}")

    # Benchmark
    import time

    x_cuda = x.cuda() if torch.cuda.is_available() else x
    ln_cuda = ln.cuda() if torch.cuda.is_available() else ln
    rms_cuda = rms.cuda() if torch.cuda.is_available() else rms

    # Warmup
    for _ in range(10):
        _ = ln_cuda(x_cuda)
        _ = rms_cuda(x_cuda)

    # Time LayerNorm
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = ln_cuda(x_cuda)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ln_time = time.time() - start

    # Time RMSNorm
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = rms_cuda(x_cuda)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rms_time = time.time() - start

    print(f"\nSpeed comparison (100 iterations):")
    print(f"  LayerNorm: {ln_time*1000:.2f}ms")
    print(f"  RMSNorm: {rms_time*1000:.2f}ms")
    print(f"  Speedup: {ln_time/rms_time:.2f}x")


if __name__ == "__main__":
    compare_normalizations()
```

**Output interpretation:**
- LayerNorm produces outputs with mean ≈ 0 (due to centering)
- RMSNorm preserves the mean but normalizes scale
- RMSNorm is 10-15% faster

---

## Feed-Forward Networks

The feed-forward network (FFN) processes each position independently with the same learned transformation.

### FFN Structure

Standard FFN consists of two linear transformations with a non-linear activation:

$$
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

where:
- $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ is the input
- $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
- $d_{\text{ff}}$ is the intermediate dimension (typically $4 \times d_{\text{model}}$)
- $\sigma$ is the activation function (see [Activation Functions](10-activation-functions.md))

**Standard choices:**
- Original Transformer: $d_{\text{ff}} = 4 \times d_{\text{model}}$, activation = ReLU
- Modern LLMs: $d_{\text{ff}} = \frac{8}{3} \times d_{\text{model}}$ or $3.5 \times d_{\text{model}}$, activation = SwiGLU/GeGLU

**Why is FFN needed?**
- Attention is linear (weighted sum), FFN adds non-linearity
- FFN processes each position independently (position-wise)
- Provides additional capacity and expressiveness
- Can be thought of as key-value memory (shown in research)

### FFN Implementation

```python
class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward intermediate dimension
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, seq_len, d_model)
        Returns:
            output: shape (batch, seq_len, d_model)
        """
        # First linear layer + activation
        hidden = self.activation(self.w1(x))  # (batch, seq_len, d_ff)

        # Dropout
        hidden = self.dropout(hidden)

        # Second linear layer
        output = self.w2(hidden)  # (batch, seq_len, d_model)

        return output


# Example usage
def test_ffn():
    d_model = 512
    d_ff = 2048  # 4x expansion
    batch_size = 2
    seq_len = 10

    ffn = FeedForward(d_model, d_ff, activation="gelu")

    x = torch.randn(batch_size, seq_len, d_model)
    output = ffn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"FFN parameters: {sum(p.numel() for p in ffn.parameters()):,}")
    # Parameters: d_model * d_ff + d_ff + d_ff * d_model + d_model
    #           = 2 * d_model * d_ff + d_ff + d_model
    #           = 2 * 512 * 2048 + 2048 + 512 = 2,099,712

if __name__ == "__main__":
    test_ffn()
```

**Parameter count:**
- FFN has ~$2 \times d_{\text{model}} \times d_{\text{ff}}$ parameters
- With $d_{\text{ff}} = 4 \times d_{\text{model}}$, FFN has $8 \times d_{\text{model}}^2$ parameters
- This is typically 2/3 of all transformer block parameters

---

## Residual Connections

Residual connections (skip connections) add the input of a sub-layer to its output:

$$
\mathbf{y} = \mathbf{x} + \text{SubLayer}(\mathbf{x})
$$

### Why Residuals Matter

**Training deep networks:**
Without residual connections, deep networks suffer from:
1. **Vanishing gradients**: Gradients shrink exponentially with depth
2. **Degradation**: Deeper networks perform worse than shallower ones

**How residuals help:**
- Create identity mappings for gradient flow
- Allow each layer to learn refinements rather than full transformations
- Enable training of 100+ layer transformers

**Mathematical insight:**

Consider a network with $L$ layers. During backpropagation:

Without residuals:
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \prod_{i=1}^{L-1} \frac{\partial \mathbf{x}_{i+1}}{\partial \mathbf{x}_i}
$$

With residuals ($\mathbf{x}_{i+1} = \mathbf{x}_i + F_i(\mathbf{x}_i)$):
$$
\frac{\partial \mathbf{x}_{i+1}}{\partial \mathbf{x}_i} = \mathbf{I} + \frac{\partial F_i(\mathbf{x}_i)}{\partial \mathbf{x}_i}
$$

The identity term $\mathbf{I}$ ensures gradients can flow directly backward.

### Gradient Flow Analysis

```python
def analyze_gradient_flow():
    """Demonstrate gradient flow with and without residual connections."""
    import torch.nn.functional as F

    # Simple network without residuals
    class WithoutResidual(nn.Module):
        def __init__(self, d_model, depth):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(depth)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = torch.tanh(layer(x))  # Non-linearity
            return x

    # Network with residuals
    class WithResidual(nn.Module):
        def __init__(self, d_model, depth):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(depth)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = x + torch.tanh(layer(x))  # Residual connection
            return x

    d_model = 128
    depth = 20
    batch_size = 4

    # Create models
    no_res = WithoutResidual(d_model, depth)
    with_res = WithResidual(d_model, depth)

    # Forward pass
    x = torch.randn(batch_size, d_model, requires_grad=True)

    # Without residual
    x_no_res = x.clone().detach().requires_grad_(True)
    out_no_res = no_res(x_no_res)
    loss_no_res = out_no_res.sum()
    loss_no_res.backward()

    # With residual
    x_with_res = x.clone().detach().requires_grad_(True)
    out_with_res = with_res(x_with_res)
    loss_with_res = out_with_res.sum()
    loss_with_res.backward()

    # Compare gradient magnitudes
    print("Gradient flow comparison:")
    print(f"Without residual - input gradient norm: {x_no_res.grad.norm():.6f}")
    print(f"With residual - input gradient norm: {x_with_res.grad.norm():.6f}")

    # Check layer gradients
    print("\nLayer gradient norms (first 5 layers):")
    for i in range(5):
        grad_no_res = no_res.layers[i].weight.grad.norm()
        grad_with_res = with_res.layers[i].weight.grad.norm()
        print(f"  Layer {i}: no_res={grad_no_res:.6f}, with_res={grad_with_res:.6f}")

if __name__ == "__main__":
    analyze_gradient_flow()
```

**Typical output:**
- Without residuals: gradients become very small (vanishing)
- With residuals: gradients remain stable across layers

---

## Pre-Norm vs Post-Norm

The arrangement of layer normalization relative to sub-layers significantly impacts training.

### Post-Norm Architecture

**Original Transformer** ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)):

```
x → [Self-Attention] → [Add & Norm] → [FFN] → [Add & Norm] → output
     ↑_______________↓                 ↑_______↓
```

Mathematically:
$$
\begin{align}
\mathbf{y}_1 &= \text{LayerNorm}(\mathbf{x} + \text{Attention}(\mathbf{x})) \\
\mathbf{y}_2 &= \text{LayerNorm}(\mathbf{y}_1 + \text{FFN}(\mathbf{y}_1))
\end{align}
$$

**Issues:**
- Gradients flow through normalization layers during backprop
- Harder to train very deep models (>12 layers)
- Often requires careful learning rate warmup

### Pre-Norm Architecture

**Modern standard** ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745)):

```
x → [Norm] → [Self-Attention] → [Add] → [Norm] → [FFN] → [Add] → output
     ↑_________________________↓          ↑_______________↓
```

Mathematically:
$$
\begin{align}
\mathbf{y}_1 &= \mathbf{x} + \text{Attention}(\text{LayerNorm}(\mathbf{x})) \\
\mathbf{y}_2 &= \mathbf{y}_1 + \text{FFN}(\text{LayerNorm}(\mathbf{y}_1))
\end{align}
$$

**Advantages:**
- Gradients flow directly through residual connections
- More stable training for deep models (100+ layers)
- Less sensitive to learning rate and initialization
- Standard in modern LLMs (GPT-3, LLaMA, PaLM, etc.)

### Why Pre-Norm is Standard

**Research findings:**

1. **Training stability** ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745)):
   - Pre-norm enables training without learning rate warmup
   - Supports larger learning rates
   - Better gradient flow in deep networks

2. **Performance** (empirical):
   - Pre-norm often matches or exceeds post-norm final performance
   - Especially beneficial for very deep models (>24 layers)

3. **Gradient analysis:**
   - Post-norm: gradient norm grows with depth
   - Pre-norm: gradient norm stays bounded

**Trade-offs:**
- Pre-norm: Better training, slightly worse representation power
- Post-norm: Theoretically stronger, harder to optimize
- Modern practice: Pre-norm + final norm after all blocks

```python
def compare_architectures():
    """Compare pre-norm and post-norm gradient flow."""

    class PostNormBlock(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            # Apply transformation, add residual, then normalize
            return self.norm(x + self.linear(x))

    class PreNormBlock(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            # Normalize, apply transformation, then add residual
            return x + self.linear(self.norm(x))

    d_model = 128
    depth = 24

    # Create deep networks
    post_norm = nn.Sequential(*[PostNormBlock(d_model) for _ in range(depth)])
    pre_norm = nn.Sequential(*[PreNormBlock(d_model) for _ in range(depth)])

    x = torch.randn(4, 10, d_model, requires_grad=True)

    # Test post-norm
    x_post = x.clone().detach().requires_grad_(True)
    out_post = post_norm(x_post)
    loss_post = out_post.mean()
    loss_post.backward()

    # Test pre-norm
    x_pre = x.clone().detach().requires_grad_(True)
    out_pre = pre_norm(x_pre)
    loss_pre = out_pre.mean()
    loss_pre.backward()

    print(f"Depth: {depth} layers")
    print(f"Post-norm input gradient norm: {x_post.grad.norm():.6f}")
    print(f"Pre-norm input gradient norm: {x_pre.grad.norm():.6f}")
    print(f"Pre-norm is more stable: {x_pre.grad.norm() > x_post.grad.norm()}")

if __name__ == "__main__":
    compare_architectures()
```

---

## Complete Transformer Block Implementation

Now we combine all components into a complete transformer block.

```python
from typing import Optional

class TransformerBlock(nn.Module):
    """
    Complete transformer block with pre-norm architecture.

    This is the standard building block of modern LLMs.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_type: str = "rmsnorm",
        activation: str = "gelu"
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward intermediate dimension
            dropout: Dropout probability
            norm_type: Type of normalization ('layernorm' or 'rmsnorm')
            activation: Activation function for FFN
        """
        super().__init__()

        # Normalization layers
        if norm_type == "layernorm":
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Self-attention (simplified - see Multi-Head Attention chapter)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask (batch, seq_len, seq_len) or (seq_len, seq_len)

        Returns:
            output: shape (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm
        # 1. Normalize
        x_norm = self.norm1(x)

        # 2. Self-attention
        attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=mask,
            need_weights=False
        )

        # 3. Residual connection
        x = x + self.dropout(attn_output)

        # Feed-forward with pre-norm
        # 1. Normalize
        x_norm = self.norm2(x)

        # 2. FFN
        ffn_output = self.ffn(x_norm)

        # 3. Residual connection
        x = x + self.dropout(ffn_output)

        return x


# Example: Stack multiple blocks
class TransformerStack(nn.Module):
    """Stack of transformer blocks forming a complete transformer."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_type: str = "rmsnorm"
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, norm_type)
            for _ in range(n_layers)
        ])

        # Final normalization (common in modern LLMs)
        if norm_type == "rmsnorm":
            self.final_norm = RMSNorm(d_model)
        else:
            self.final_norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: shape (batch, seq_len, d_model)
        """
        for block in self.blocks:
            x = block(x, mask)

        # Final normalization
        x = self.final_norm(x)

        return x


# Testing the complete implementation
def test_transformer_block():
    # Model configuration
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 6
    batch_size = 2
    seq_len = 10

    # Create model
    model = TransformerStack(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1,
        norm_type="rmsnorm"
    )

    # Create input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = model(x)

    print("Transformer Stack Test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Parameter breakdown
    block_params = sum(p.numel() for p in model.blocks[0].parameters())
    print(f"  Parameters per block: {block_params:,}")

if __name__ == "__main__":
    test_transformer_block()
```

---

## Visualization and Analysis

```python
def visualize_block_activations():
    """Visualize how activations change through a transformer block."""
    import matplotlib.pyplot as plt
    import numpy as np

    d_model = 128
    seq_len = 20
    batch_size = 1

    # Create a transformer block
    block = TransformerBlock(
        d_model=d_model,
        n_heads=4,
        d_ff=512,
        norm_type="rmsnorm"
    )
    block.eval()

    # Create input with some structure
    x = torch.randn(batch_size, seq_len, d_model)

    # Track activations
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    block.norm1.register_forward_hook(hook_fn('after_norm1'))
    block.self_attn.register_forward_hook(hook_fn('after_attn'))
    block.norm2.register_forward_hook(hook_fn('after_norm2'))
    block.ffn.register_forward_hook(hook_fn('after_ffn'))

    # Forward pass
    with torch.no_grad():
        output = block(x)

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Transformer Block Activations')

    def plot_activation(ax, tensor, title):
        # Plot heatmap of activations
        data = tensor[0].cpu().numpy()  # (seq_len, d_model)
        im = ax.imshow(data.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set_title(title)
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Feature Dimension')
        plt.colorbar(im, ax=ax)

    # Plot each stage
    plot_activation(axes[0, 0], x, 'Input')
    plot_activation(axes[0, 1], activations['after_norm1'], 'After Norm1')

    # For attention output, we need to get it differently
    plot_activation(axes[0, 2], output, 'After Attention + Residual')
    plot_activation(axes[1, 0], activations['after_norm2'], 'After Norm2')
    plot_activation(axes[1, 1], activations['after_ffn'], 'After FFN')
    plot_activation(axes[1, 2], output, 'Final Output')

    plt.tight_layout()
    plt.savefig('transformer_block_activations.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to 'transformer_block_activations.png'")


def analyze_parameter_distribution():
    """Analyze where parameters are distributed in a transformer block."""
    d_model = 768  # BERT-base size
    n_heads = 12
    d_ff = 3072  # 4x expansion

    block = TransformerBlock(d_model, n_heads, d_ff, norm_type="rmsnorm")

    # Count parameters by component
    param_counts = {}

    # Attention parameters
    attn_params = sum(p.numel() for p in block.self_attn.parameters())
    param_counts['Self-Attention'] = attn_params

    # FFN parameters
    ffn_params = sum(p.numel() for p in block.ffn.parameters())
    param_counts['Feed-Forward'] = ffn_params

    # Normalization parameters
    norm_params = sum(p.numel() for p in block.norm1.parameters()) + \
                  sum(p.numel() for p in block.norm2.parameters())
    param_counts['Normalization'] = norm_params

    total = sum(param_counts.values())

    print("Parameter Distribution in Transformer Block:")
    print(f"  Total: {total:,}")
    for name, count in param_counts.items():
        percentage = 100 * count / total
        print(f"  {name}: {count:,} ({percentage:.1f}%)")

    # Visualize
    plt.figure(figsize=(10, 6))
    components = list(param_counts.keys())
    counts = list(param_counts.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    plt.bar(components, counts, color=colors)
    plt.ylabel('Number of Parameters')
    plt.title(f'Transformer Block Parameter Distribution (d_model={d_model})')
    plt.xticks(rotation=15)

    # Add percentage labels
    for i, (component, count) in enumerate(zip(components, counts)):
        percentage = 100 * count / total
        plt.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('transformer_block_params.png', dpi=150, bbox_inches='tight')
    print("\nParameter distribution plot saved to 'transformer_block_params.png'")


if __name__ == "__main__":
    visualize_block_activations()
    analyze_parameter_distribution()
```

**Key insights:**
- FFN typically contains 60-70% of parameters
- Attention contains 30-40% of parameters
- Normalization is negligible (<1%)

---

## Modern Variants

### LLaMA Architecture

[LLaMA](https://arxiv.org/abs/2302.13971) and [LLaMA 2](https://arxiv.org/abs/2307.09288) use:
- **RMSNorm** instead of LayerNorm
- **SwiGLU** activation (see [Activation Functions](10-activation-functions.md))
- **Pre-normalization**
- **RoPE** positional embeddings (see [RoPE](08-rope.md))

```python
class LLaMABlock(nn.Module):
    """Transformer block following LLaMA architecture."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Self-attention (would include RoPE in practice)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # SwiGLU FFN (simplified)
        # Actual implementation: see Activation Functions chapter
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout, activation="gelu")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Pre-norm architecture
        # Attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h

        # FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x
```

### GPT-3 Architecture

GPT-3 uses:
- **LayerNorm** (not RMSNorm)
- **GELU** activation
- **Pre-normalization**
- **Learned positional embeddings**

### Differences in Modern LLMs

| Model | Normalization | FFN Activation | Position Encoding | $d_{ff}$ ratio |
|-------|--------------|----------------|-------------------|----------------|
| GPT-2/3 | LayerNorm | GELU | Learned | 4x |
| LLaMA 1/2/3 | RMSNorm | SwiGLU | RoPE | 8/3x |
| PaLM | RMSNorm | SwiGLU | RoPE | 4x |
| GPT-4 | RMSNorm (likely) | SwiGLU (likely) | RoPE (likely) | Unknown |

---

## Exercises

### Exercise 1: Implement Post-Norm Block

Implement a transformer block using post-norm architecture and compare training stability with pre-norm.

```python
def exercise1():
    """
    TODO: Implement PostNormTransformerBlock

    Requirements:
    - Use post-norm architecture (norm after residual)
    - Compare gradient flow with pre-norm version
    - Train a small model and observe convergence
    """
    pass
```

**Solution approach:**
- Swap order: apply sub-layer, add residual, then normalize
- Compare gradient norms at different depths
- Try training with and without learning rate warmup

### Exercise 2: Analyze FFN Expansion Ratio

Experiment with different $d_{ff}/d_{model}$ ratios and measure:
- Parameter count
- Training speed
- Model performance

```python
def exercise2():
    """
    TODO: Test FFN expansion ratios

    Test ratios: 1x, 2x, 4x, 8x
    Measure: params, FLOPs, performance on simple task
    """
    pass
```

**Expected findings:**
- Higher ratios: more parameters, slower, potentially better performance
- Diminishing returns beyond 4x
- Trade-off between capacity and efficiency

### Exercise 3: Implement Parallel Attention and FFN

Some recent models (PaLM, GPT-J) compute attention and FFN in parallel rather than sequentially.

```python
def exercise3():
    """
    TODO: Implement parallel transformer block

    Architecture:
        x_norm = norm(x)
        attn_out = attention(x_norm)
        ffn_out = ffn(x_norm)
        output = x + attn_out + ffn_out

    Compare:
    - Speed vs sequential
    - Memory usage
    - Performance
    """
    pass
```

**Hint:** Parallel version is faster but may sacrifice some performance.

### Exercise 4: Add Dropout Variations

Implement different dropout strategies:
- Standard dropout
- DropPath/Stochastic depth
- Attention dropout vs FFN dropout

```python
def exercise4():
    """
    TODO: Implement and compare dropout strategies

    Compare:
    - No dropout
    - Standard dropout
    - Stochastic depth (drop entire blocks)
    """
    pass
```

### Exercise 5: Gradient Norm Analysis

Track and visualize gradient norms through training:
- Per-layer gradient norms
- How they change with depth
- Effect of normalization type

```python
def exercise5():
    """
    TODO: Track gradient norms during training

    Plot:
    - Gradient norm vs layer depth
    - Gradient norm over training steps
    - Compare pre-norm vs post-norm
    """
    pass
```

---

## Summary

The transformer block is the fundamental building unit of modern LLMs:

**Key components:**
1. **Layer Normalization**: RMSNorm is standard (faster, similar performance)
2. **Feed-Forward Network**: Typically 4x expansion with non-linear activation
3. **Residual Connections**: Essential for gradient flow in deep networks
4. **Pre-Norm**: Standard in modern LLMs for training stability

**Important insights:**
- Pre-norm enables training very deep models (100+ layers)
- FFN contains ~2/3 of transformer parameters
- Residual connections create direct gradient paths
- RMSNorm is 10-15% faster than LayerNorm with similar performance

**Modern best practices:**
- Use pre-norm architecture
- Use RMSNorm for efficiency
- FFN expansion ratio of 4x (or 8/3x with SwiGLU)
- GELU or SwiGLU activation (see [Activation Functions](10-activation-functions.md))

In the next chapters, we'll explore:
- [Activation Functions](10-activation-functions.md): SwiGLU, GeGLU, and modern activations
- [Building a Complete Transformer](11-complete-transformer.md): Stacking blocks into full models

---

## References

1. **Attention is All You Need** - Vaswani et al., 2017
   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
   Original transformer with post-norm architecture

2. **Layer Normalization** - Ba et al., 2016
   [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
   Introduction of layer normalization

3. **Root Mean Square Layer Normalization** - Zhang & Sennrich, 2019
   [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
   RMSNorm formulation

4. **On Layer Normalization in the Transformer Architecture** - Xiong et al., 2020
   [https://arxiv.org/abs/2002.04745](https://arxiv.org/abs/2002.04745)
   Analysis of pre-norm vs post-norm

5. **Deep Residual Learning for Image Recognition** - He et al., 2015
   [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
   Introduction of residual connections

6. **LLaMA: Open and Efficient Foundation Language Models** - Touvron et al., 2023
   [https://arxiv.org/abs/2302.13971](https://arxiv.org/abs/2302.13971)
   LLaMA architecture with RMSNorm and SwiGLU

7. **LLaMA 2: Open Foundation and Fine-Tuned Chat Models** - Touvron et al., 2023
   [https://arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)
   LLaMA 2 improvements and training details

8. **GLU Variants Improve Transformer** - Shazeer, 2020
   [https://arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202)
   SwiGLU and gated linear units

9. **PaLM: Scaling Language Modeling with Pathways** - Chowdhery et al., 2022
   [https://arxiv.org/abs/2204.02311](https://arxiv.org/abs/2204.02311)
   PaLM architecture choices
