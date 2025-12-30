# Chapter 8: Rotary Position Embeddings (RoPE)

Rotary Position Embeddings (RoPE) have become the de facto standard for positional encoding in modern Large Language Models. Introduced in the RoFormer paper (Su et al., 2021), RoPE elegantly encodes absolute position information while maintaining relative position relationships through rotation matrices. This chapter explores the mathematical foundations, implementation details, and practical applications of RoPE.

## Table of Contents

1. [Introduction and Motivation](#introduction-and-motivation)
2. [The Problem with Absolute Position Encodings](#the-problem-with-absolute-position-encodings)
3. [Rotary Embeddings: Intuition](#rotary-embeddings-intuition)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Implementation in PyTorch](#implementation-in-pytorch)
6. [Benefits and Properties](#benefits-and-properties)
7. [RoPE in Modern LLMs](#rope-in-modern-llms)
8. [RoPE Scaling for Long Contexts](#rope-scaling-for-long-contexts)
9. [Advanced Topics](#advanced-topics)
10. [Exercises](#exercises)

---

## Introduction and Motivation

Transformers are inherently position-agnostic: without positional information, a transformer processes a sequence as a bag of words. As discussed in [Chapter 7: Positional Encodings](07-positional-encodings.md), various approaches have been developed to inject positional information.

**The ideal positional encoding should:**
1. Encode absolute position information
2. Maintain relative position relationships
3. Extrapolate to sequence lengths unseen during training
4. Be computationally efficient
5. Work well with attention mechanisms

RoPE achieves all these goals through a clever application of rotation matrices in the complex plane.

---

## The Problem with Absolute Position Encodings

### Learned Absolute Positional Embeddings

GPT-2 and GPT-3 use learned positional embeddings (see [Architecture Comparison](29-model-architectures.md)):

```python
import torch
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings (GPT-2 style)."""
    def __init__(self, max_position: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_position, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position embeddings to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            x + position embeddings
        """
        batch, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions)
```

**Limitations:**
1. **No extrapolation**: Cannot handle sequences longer than `max_position`
2. **No relative bias**: Position 5 and position 6 have no inherent relationship
3. **Inefficient**: Requires storing embeddings for every possible position

### Sinusoidal Position Encodings

The original Transformer (Vaswani et al., 2017) uses sinusoidal encodings:

$$
\begin{align}
\text{PE}(pos, 2i) &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
\text{PE}(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{align}
$$

**Limitations:**
1. **Added to inputs**: Positional information can be diluted through layers
2. **No relative bias in attention**: Doesn't directly encode relative distances
3. **Extrapolation issues**: Performance degrades on longer sequences

---

## Rotary Embeddings: Intuition

RoPE's key insight is to rotate the query and key vectors by an angle proportional to their position. This encoding:
- Gives each position a unique representation (absolute position)
- Makes relative positions fall out naturally from the rotation angle difference

### 2D Rotation Intuition

Consider two 2D vectors $\mathbf{q}$ and $\mathbf{k}$ at positions $m$ and $n$:

1. Rotate $\mathbf{q}$ by angle $m\theta$
2. Rotate $\mathbf{k}$ by angle $n\theta$
3. Their dot product now depends on $(m-n)\theta$ — the relative position!

```
Position 0:  ──→ (original)
Position 1:  ─↗  (rotated by θ)
Position 2:  ─↑  (rotated by 2θ)
Position 3:  ─↖  (rotated by 3θ)
```

The rotation matrix for angle $\theta$ in 2D is:

$$
\mathbf{R}(\theta) = \begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
$$

**Key property**: The attention score between positions $m$ and $n$ only depends on the relative position $m - n$:

$$
(\mathbf{R}(m\theta)\mathbf{q})^T (\mathbf{R}(n\theta)\mathbf{k}) = \mathbf{q}^T \mathbf{R}^T(m\theta) \mathbf{R}(n\theta) \mathbf{k} = \mathbf{q}^T \mathbf{R}((n-m)\theta) \mathbf{k}
$$

This is because rotation matrices satisfy: $\mathbf{R}^T(\alpha)\mathbf{R}(\beta) = \mathbf{R}(\beta - \alpha)$

---

## Mathematical Formulation

### Complex Number Representation

RoPE uses complex numbers to elegantly represent 2D rotations. A 2D vector $\mathbf{x} = (x_0, x_1)$ can be represented as a complex number:

$$
z = x_0 + ix_1
$$

Rotating by angle $\theta$ is simply multiplication by $e^{i\theta}$:

$$
z' = e^{i\theta} \cdot z = (\cos\theta + i\sin\theta)(x_0 + ix_1)
$$

Expanding:
$$
z' = (x_0\cos\theta - x_1\sin\theta) + i(x_0\sin\theta + x_1\cos\theta)
$$

This matches the 2D rotation matrix!

### Extending to High Dimensions

For a $d$-dimensional vector (where $d$ is even), we split it into $d/2$ pairs and apply different rotation frequencies to each pair.

Given query vector $\mathbf{q} \in \mathbb{R}^d$ at position $m$, the RoPE transformation is:

$$
\mathbf{f}_{\mathbf{q}}(\mathbf{q}, m) = \mathbf{R}_{\Theta,m}^d \mathbf{q}
$$

where $\mathbf{R}_{\Theta,m}^d$ is a block-diagonal rotation matrix:

$$
\mathbf{R}_{\Theta,m}^d = \begin{pmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots & 0 & 0 \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots & 0 & 0 \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos(m\theta_{d/2-1}) & -\sin(m\theta_{d/2-1}) \\
0 & 0 & 0 & 0 & \cdots & \sin(m\theta_{d/2-1}) & \cos(m\theta_{d/2-1})
\end{pmatrix}
$$

The rotation frequencies $\{\theta_i\}$ decrease geometrically:

$$
\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2-1
$$

This gives lower dimensions faster rotation (capturing fine-grained position) and higher dimensions slower rotation (capturing coarse-grained position).

### Why This Frequency Schedule?

The frequency schedule is inspired by sinusoidal positional encodings. For dimension pair $i$:
- Small $i$ (low dimensions): High frequency $\theta_i$ → distinguishes nearby positions
- Large $i$ (high dimensions): Low frequency $\theta_i$ → distinguishes distant positions

This multi-scale representation allows RoPE to capture both local and global positional relationships.

### Applied to Attention

In multi-head attention (see [Multi-Head Attention](04-multi-head-attention.md)), we apply RoPE to queries and keys before computing attention scores:

$$
\begin{align}
\mathbf{q}_m' &= \mathbf{R}_{\Theta,m}^d \mathbf{q}_m \\
\mathbf{k}_n' &= \mathbf{R}_{\Theta,n}^d \mathbf{k}_n \\
\text{score}(m, n) &= \frac{(\mathbf{q}_m')^T \mathbf{k}_n'}{\sqrt{d_k}}
\end{align}
$$

The attention score becomes:

$$
\text{score}(m, n) = \frac{\mathbf{q}_m^T \mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} \mathbf{k}_n}{\sqrt{d_k}} = \frac{\mathbf{q}_m^T \mathbf{R}_{\Theta, n-m}^d \mathbf{k}_n}{\sqrt{d_k}}
$$

**Key insight**: The score only depends on the relative position $n - m$, not the absolute positions!

---

## Implementation in PyTorch

### Basic RoPE Implementation

```python
import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotation matrices to query and key vectors based on their position.
    Each dimension pair is rotated by a different frequency.

    Args:
        dim: Dimension of the embedding (must be even)
        max_seq_len: Maximum sequence length (for precomputing)
        base: Base for the geometric progression (default: 10000)
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute rotation frequencies: theta_i = base^(-2i/d)
        # Shape: (dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute rotation matrices for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin for all positions up to seq_len.

        Args:
            seq_len: Sequence length to cache
        """
        # Position indices: [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len,)
        positions = torch.arange(seq_len).float()

        # Compute angles: position * theta_i for all positions and all i
        # Shape: (seq_len, dim/2)
        angles = torch.outer(positions, self.inv_freq)

        # Compute cos and sin
        # Shape: (seq_len, dim/2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Cache for reuse
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimensions.

        Rearranges [x0, x1, x2, x3, ...] to [-x1, x0, -x3, x2, ...]
        This implements the rotation in complex number form.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Rotated tensor of same shape
        """
        # Split into two halves
        x1 = x[..., : self.dim // 2]  # First half of pairs
        x2 = x[..., self.dim // 2 :]  # Second half of pairs

        # Interleave with negation: equivalent to rotation
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to queries and keys.

        Args:
            q: Query tensor of shape (batch, seq_len, n_heads, head_dim)
            k: Key tensor of shape (batch, seq_len, n_heads, head_dim)
            start_pos: Starting position (for incremental decoding)

        Returns:
            (rotated_q, rotated_k): Rotated query and key tensors
        """
        seq_len = q.shape[1]

        # Extend cache if needed
        if start_pos + seq_len > self.max_seq_len:
            self._build_cache(start_pos + seq_len)

        # Get cos and sin for current positions
        # Shape: (seq_len, dim/2)
        cos = self.cos_cached[start_pos : start_pos + seq_len]
        sin = self.sin_cached[start_pos : start_pos + seq_len]

        # Reshape for broadcasting: (1, seq_len, 1, dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Repeat each element twice to match full dimension
        # (1, seq_len, 1, dim/2) -> (1, seq_len, 1, dim)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        # Apply rotation using the formula:
        # rotate(x) = x * cos + rotate_half(x) * sin
        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin

        return q_rotated, k_rotated
```

### RoPE-Enhanced Attention

```python
class RoPEAttention(nn.Module):
    """Multi-head attention with RoPE positional encoding.

    Integrates RoPE with standard scaled dot-product attention.
    See [Multi-Head Attention](04-multi-head-attention.md) for base attention.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        rope_base: int = 10000
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Query, Key, Value projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(
            self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask (batch, seq_len, seq_len) or None
            start_pos: Starting position for incremental decoding

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x)  # (batch, seq_len, d_model)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape to (batch, seq_len, n_heads, head_dim)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q, k = self.rope(q, k, start_pos=start_pos)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        # (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        # (batch, n_heads, seq_len, head_dim)
        out = torch.matmul(attn, v)

        # Reshape and project
        # (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.w_o(out)
```

### Efficient RoPE with Complex Numbers

For better efficiency and numerical stability, we can implement RoPE using PyTorch's complex number support:

```python
class EfficientRoPE(nn.Module):
    """Efficient RoPE implementation using complex numbers.

    This implementation is more memory-efficient and numerically stable.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0

        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute complex exponentials
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)
        # e^(i*theta) = cos(theta) + i*sin(theta)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.register_buffer('freqs_cis', freqs_cis)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0
    ) -> torch.Tensor:
        """Apply RoPE using complex numbers.

        Args:
            x: Input tensor (batch, seq_len, n_heads, head_dim)
            start_pos: Starting position

        Returns:
            Rotated tensor of same shape
        """
        batch, seq_len, n_heads, head_dim = x.shape

        # Reshape x to complex: (..., head_dim/2)
        x_complex = torch.view_as_complex(
            x.float().reshape(batch, seq_len, n_heads, head_dim // 2, 2)
        )

        # Get frequencies for current positions
        freqs = self.freqs_cis[start_pos : start_pos + seq_len]
        freqs = freqs.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim/2)

        # Rotate by multiplying with e^(i*theta)
        x_rotated = x_complex * freqs

        # Convert back to real
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(batch, seq_len, n_heads, head_dim)

        return x_out.type_as(x)
```

---

## Benefits and Properties

### 1. Relative Position Bias

RoPE naturally encodes relative positions through rotation angles. The attention score between positions $m$ and $n$ only depends on $m - n$:

$$
\text{score}(m, n) \propto \mathbf{q}_m^T \mathbf{R}_{\Theta, n-m} \mathbf{k}_n
$$

This is superior to absolute positional encodings where position 10 and 11 have no inherent relationship.

### 2. Long-Range Decay

Due to the rotation, attention naturally decays with distance. For positions far apart, the rotation angle becomes large, leading to lower attention scores (on average).

### 3. Extrapolation to Longer Sequences

RoPE can generalize to sequences longer than those seen during training. The rotation angles are computed on-the-fly, so there's no hard limit like with learned embeddings.

**However**, there are limits to extrapolation (see [RoPE Scaling](#rope-scaling-for-long-contexts)).

### 4. Computational Efficiency

- **No learnable parameters**: Unlike learned embeddings, RoPE has zero parameters
- **Low memory**: Only stores $O(d)$ frequency coefficients
- **Fast computation**: Rotation is element-wise operations
- **Cache-friendly**: Precompute cos/sin for reuse

### 5. Translation Invariance

For any shift $\Delta$:
$$
\text{score}(m + \Delta, n + \Delta) = \text{score}(m, n)
$$

The model treats "word 5 attending to word 3" the same as "word 105 attending to word 103".

---

## RoPE in Modern LLMs

RoPE has become the standard positional encoding in modern LLMs. See [Architecture Comparison](29-model-architectures.md) for a comprehensive comparison.

### Models Using RoPE

| Model | RoPE Variant | Base Frequency | Max Context | Notes |
|-------|-------------|----------------|-------------|-------|
| LLaMA 1 | Standard | 10,000 | 2K | Original implementation |
| LLaMA 2 | Standard | 10,000 | 4K | Extended context |
| LLaMA 3 | Standard | 500,000 | 8K | Higher base for longer context |
| LLaMA 4 Scout | iRoPE | Variable | 10M | Interleaved RoPE/NoPE |
| Mistral 7B | Standard | 10,000 | 8K | With sliding window |
| Mixtral 8x7B | Standard | 10,000 | 32K | Extended |
| Qwen 2.5 | ABF Scaling | 1,000,000 | 128K | Attention Base Frequency |
| Qwen 3 | ABF Scaling | 1,000,000 | 128K | + QK-Norm |
| DeepSeek V3 | Standard | 10,000 | 128K | With MLA |
| Gemma 2 | Standard | 10,000 | 8K | Interleaved attention |

### LLaMA Implementation Details

LLaMA uses a slightly different frequency calculation:

```python
class LLaMARotaryEmbedding(nn.Module):
    """RoPE as implemented in LLaMA.

    Difference from standard RoPE: uses theta_i = base^(-2(i-1)/(dim-2))
    instead of base^(-2i/dim). This is a minor variation.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()

        # LLaMA-style frequency calculation
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

        # Build cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len).float()
        angles = torch.outer(positions, self.inv_freq)

        # LLaMA uses complex exponentials
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.register_buffer('freqs_cis', freqs_cis)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.shape[1]
        freqs = self.freqs_cis[start_pos : start_pos + seq_len]

        # Reshape x to pairs of dimensions
        x_complex = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        )

        # Apply rotation
        freqs = freqs.view(1, seq_len, 1, x_complex.shape[-1])
        x_rotated = x_complex * freqs

        # Back to real
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        return x_out.type_as(x)
```

---

## RoPE Scaling for Long Contexts

While RoPE can extrapolate beyond its training context, performance degrades significantly. Several techniques have been developed to extend RoPE to longer contexts.

See [Long Context Techniques](26-long-context.md) for comprehensive coverage of long-context methods.

### The Extrapolation Problem

When trained on sequences of length $L$ and tested on length $L' > L$:
- Position embeddings for positions $> L$ were never seen during training
- The model hasn't learned how to handle those rotation angles
- Attention patterns become unpredictable

### Position Interpolation (PI)

**Idea**: Instead of extrapolating to unseen angles, interpolate within seen angles by scaling down position indices.

$$
\theta'_m = \theta_{m \cdot L / L'}
$$

Effectively, we slow down the rotation to fit longer sequences into the trained range.

```python
class InterpolatedRoPE(nn.Module):
    """RoPE with Position Interpolation for longer contexts.

    Args:
        dim: Embedding dimension
        max_train_len: Maximum length seen during training
        max_infer_len: Maximum length for inference (can be longer)
        base: Base frequency
    """
    def __init__(
        self,
        dim: int,
        max_train_len: int = 2048,
        max_infer_len: int = 8192,
        base: int = 10000
    ):
        super().__init__()

        self.dim = dim
        self.max_train_len = max_train_len
        self.max_infer_len = max_infer_len
        self.scale = max_infer_len / max_train_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Build cache for maximum inference length
        self._build_cache(max_infer_len)

    def _build_cache(self, seq_len: int):
        # Scale positions to fit within training range
        positions = torch.arange(seq_len).float() / self.scale
        angles = torch.outer(positions, self.inv_freq)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)

    # forward() same as standard RoPE
```

**Pros**: Simple, requires minimal fine-tuning
**Cons**: Changes frequencies for ALL positions, including short ones

### NTK-Aware Scaling

**Neural Tangent Kernel (NTK)-aware scaling** adjusts base frequencies instead of positions:

$$
\theta_i' = \text{base}'^{-2i/d}, \quad \text{base}' = \text{base} \times \alpha^{d/(d-2)}
$$

where $\alpha$ is the extension ratio $L'/L$.

This preserves high-frequency components (important for local attention) while extending low-frequency components (for long-range).

```python
class NTKRoPE(nn.Module):
    """RoPE with NTK-aware scaling.

    Adjusts the base frequency to extend context while preserving
    high-frequency information for nearby positions.
    """
    def __init__(
        self,
        dim: int,
        max_train_len: int = 2048,
        max_infer_len: int = 8192,
        base: int = 10000
    ):
        super().__init__()

        self.dim = dim
        alpha = max_infer_len / max_train_len

        # Scale base frequency with NTK formula
        scaled_base = base * (alpha ** (dim / (dim - 2)))

        inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self._build_cache(max_infer_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len).float()
        angles = torch.outer(positions, self.inv_freq)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)
```

**Pros**: Better preserves local attention patterns
**Cons**: Still a heuristic, may require fine-tuning

### YaRN (Yet another RoPE extensioN)

YaRN combines multiple techniques:
1. **Frequency interpolation** for high-frequency components
2. **Frequency extrapolation** for low-frequency components
3. **Attention temperature scaling**

```python
class YaRNRoPE(nn.Module):
    """RoPE with YaRN scaling for extreme context extension.

    YaRN applies different scaling strategies to different frequency bands:
    - High frequencies: Interpolation (preserve local attention)
    - Low frequencies: Extrapolation (extend long-range)
    - Middle frequencies: Smooth transition

    Reference: https://arxiv.org/abs/2309.00071
    """
    def __init__(
        self,
        dim: int,
        max_train_len: int = 2048,
        max_infer_len: int = 32768,
        base: int = 10000,
        beta_fast: int = 32,
        beta_slow: int = 1
    ):
        super().__init__()

        self.dim = dim
        self.max_train_len = max_train_len
        self.max_infer_len = max_infer_len
        self.scale = max_infer_len / max_train_len

        # Frequency bands
        inv_freq_base = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # Determine which frequencies to interpolate vs extrapolate
        # High freq (small wavelength) -> interpolate
        # Low freq (large wavelength) -> extrapolate
        wavelengths = 2 * math.pi / inv_freq_base

        # Compute scaling factors per frequency
        freq_scales = torch.ones_like(inv_freq_base)

        for i, wavelength in enumerate(wavelengths):
            if wavelength < beta_fast:
                # High frequency: full interpolation
                freq_scales[i] = self.scale
            elif wavelength > beta_slow * self.scale:
                # Low frequency: no scaling (extrapolate)
                freq_scales[i] = 1.0
            else:
                # Middle: smooth interpolation
                alpha = (wavelength - beta_fast) / (beta_slow * self.scale - beta_fast)
                freq_scales[i] = 1.0 + alpha * (self.scale - 1.0)

        # Apply frequency-dependent scaling
        inv_freq = inv_freq_base * freq_scales
        self.register_buffer('inv_freq', inv_freq)

        # Attention temperature correction
        self.attn_scale = 0.1 * math.log(self.scale) + 1.0

        self._build_cache(max_infer_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len).float()
        angles = torch.outer(positions, self.inv_freq)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)

    def get_attention_scale(self) -> float:
        """Temperature scaling factor for attention scores."""
        return self.attn_scale
```

**Usage in attention:**
```python
# In attention computation:
scores = torch.matmul(q, k.transpose(-2, -1)) / (self.scale * rope.get_attention_scale())
```

### Comparison of Scaling Methods

| Method | Approach | Extension Ratio | Fine-tuning Required | Quality |
|--------|----------|-----------------|---------------------|---------|
| None (Extrapolation) | Direct use | 1.5-2x | No | Poor beyond 2x |
| Position Interpolation | Scale positions | 4-8x | Minimal | Good |
| NTK-Aware | Scale base frequency | 4-8x | Minimal | Better |
| YaRN | Mixed scaling | 16-32x | Recommended | Best |

### Real-World Examples

**Qwen 2.5** uses ABF (Attention Base Frequency) scaling:
- Base frequency: 1,000,000 (vs standard 10,000)
- Enables 128K context from 32K training
- Combined with Dynamic Context Awareness (DCA)

**LLaMA 3.1** uses position interpolation:
- Extended from 8K to 128K context
- Required continued pretraining on long sequences

---

## Advanced Topics

### RoPE with Grouped Query Attention

When using Grouped Query Attention (GQA, see [Multi-Head Attention](04-multi-head-attention.md)), RoPE is applied only to queries and keys, not values:

```python
class RoPEGroupedQueryAttention(nn.Module):
    """GQA with RoPE.

    RoPE is applied to Q and K, but not V (values don't need position info).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project
        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K only
        q, k = self.rope(q, k)

        # Expand K, V for groups
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)

        # Standard attention
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out)
```

### RoPE with Flash Attention

RoPE integrates seamlessly with Flash Attention (see [Flash Attention](12-flash-attention.md)):

```python
# Pseudo-code for RoPE + Flash Attention
def rope_flash_attention(q, k, v, rope):
    """Apply RoPE and Flash Attention.

    Flash Attention expects (batch, seq_len, n_heads, head_dim).
    RoPE is applied before Flash Attention.
    """
    # Apply RoPE
    q_rot, k_rot = rope(q, k)

    # Flash Attention (efficient implementation)
    # This is a simplified interface; real implementation is in CUDA
    from flash_attn import flash_attn_func

    out = flash_attn_func(
        q_rot, k_rot, v,
        causal=True,  # For autoregressive models
        softmax_scale=1.0 / math.sqrt(q.shape[-1])
    )

    return out
```

### Interleaved RoPE (iRoPE) - LLaMA 4

LLaMA 4 introduces iRoPE, which alternates between RoPE and NoPE (No Positional Encoding) layers:

```
Layer 0: RoPE + Chunked Attention
Layer 1: NoPE + Full Attention
Layer 2: NoPE + Full Attention
Layer 3: NoPE + Full Attention
Layer 4: RoPE + Chunked Attention
...
```

**Benefits**:
- NoPE layers handle long-range dependencies without position constraints
- RoPE layers provide position anchoring
- Enables 10M+ token contexts

```python
class iRoPEAttention(nn.Module):
    """Interleaved RoPE/NoPE attention (LLaMA 4 style).

    Every 4th layer uses RoPE with chunked attention.
    Other layers use no positional encoding with full attention.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layer_idx: int,
        chunk_size: int = 4096
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_rope = (layer_idx % 4 == 0)

        # Standard attention components
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(d_model // n_heads)
            self.chunk_size = chunk_size

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.w_v(x).view(batch, seq_len, self.n_heads, self.head_dim)

        if self.use_rope:
            # Apply RoPE and chunked attention
            q, k = self.rope(q, k)
            # Chunked attention implementation omitted for brevity

        # Standard attention computation
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out)
```

### RoPE for Multimodal Models

In vision-language models (see [Multimodality](27-multimodality.md)), RoPE can be applied differently:
- **1D RoPE** for text tokens (standard)
- **2D RoPE** for vision tokens (rows and columns)

```python
class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings for vision tokens.

    Applies separate RoPE for row and column positions.
    """
    def __init__(self, dim: int, max_height: int = 128, max_width: int = 128):
        super().__init__()
        assert dim % 4 == 0, "Dim must be divisible by 4 for 2D RoPE"

        # Split dimension: half for rows, half for columns
        self.rope_h = RotaryPositionalEmbedding(dim // 2, max_seq_len=max_height)
        self.rope_w = RotaryPositionalEmbedding(dim // 2, max_seq_len=max_width)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions_2d: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k: Query and key tensors (batch, n_tokens, n_heads, head_dim)
            positions_2d: (row_positions, col_positions) each (batch, n_tokens)

        Returns:
            (q_rotated, k_rotated)
        """
        # Split head_dim into row and column components
        q_h, q_w = torch.chunk(q, 2, dim=-1)
        k_h, k_w = torch.chunk(k, 2, dim=-1)

        # Apply RoPE separately for rows and columns
        q_h, k_h = self.rope_h(q_h, k_h)  # Based on row positions
        q_w, k_w = self.rope_w(q_w, k_w)  # Based on col positions

        # Concatenate back
        q_rotated = torch.cat([q_h, q_w], dim=-1)
        k_rotated = torch.cat([k_h, k_w], dim=-1)

        return q_rotated, k_rotated
```

---

## Summary

### Key Takeaways

1. **RoPE encodes position through rotation**: Each position is represented by a rotation angle, making relative positions emerge naturally.

2. **Multi-scale frequencies**: Different dimension pairs use different rotation frequencies, capturing both local and global positional relationships.

3. **Parameter-free and efficient**: No learnable parameters, low memory footprint, fast computation.

4. **Relative position bias**: Attention scores depend on relative positions $(m - n)$, not absolute positions.

5. **Extrapolation capability**: Can handle longer sequences than training length, though with degradation. Scaling techniques (NTK, YaRN) extend this further.

6. **Industry standard**: Used in LLaMA, Mistral, Qwen, DeepSeek, Gemma, and many other modern LLMs.

### When to Use RoPE

✅ **Use RoPE when:**
- Building a new LLM from scratch
- Need good extrapolation to longer contexts
- Want efficient positional encoding
- Following modern best practices

❌ **Consider alternatives when:**
- Working with very specific architectures (e.g., some multimodal models)
- Need absolute position information (rare)
- Context length is fixed and small (learned embeddings may be simpler)

### Comparison with Other Methods

| Property | Learned Absolute | Sinusoidal | ALiBi | RoPE |
|----------|-----------------|------------|-------|------|
| Extrapolation | ❌ Poor | ⚠️ Limited | ✅ Good | ✅ Good |
| Parameters | d_model × L | 0 | 0 | 0 |
| Relative bias | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| Memory | High | Low | Low | Low |
| Used in | GPT-2/3 | BERT | BLOOM | LLaMA, Mistral, most modern LLMs |

---

## Exercises

### Exercise 1: Implement Basic RoPE
Implement a simple RoPE module for a 4-dimensional embedding and verify that the attention score between two positions only depends on their relative distance.

<details>
<summary>Solution</summary>

```python
import torch
import math

def simple_rope_4d(q, k, positions):
    """RoPE for 4D vectors (2 pairs).

    Args:
        q, k: Vectors of shape (batch, seq_len, 4)
        positions: Position indices (seq_len,)

    Returns:
        (q_rotated, k_rotated)
    """
    dim = 4
    base = 10000

    # Frequencies for 2 pairs
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    # inv_freq = [1.0, 0.01]

    # Angles for each position
    angles = torch.outer(positions.float(), inv_freq)
    cos = torch.cos(angles)  # (seq_len, 2)
    sin = torch.sin(angles)

    # Expand for batch and pairs
    cos = cos.unsqueeze(0).repeat_interleave(2, dim=-1)  # (1, seq_len, 4)
    sin = sin.unsqueeze(0).repeat_interleave(2, dim=-1)

    # Rotation
    def rotate_half(x):
        x1, x2 = x[..., :2], x[..., 2:]
        return torch.cat([-x2[..., 1:2], x2[..., 0:1], -x1[..., 1:2], x1[..., 0:1]], dim=-1)

    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin

    return q_rotated, k_rotated

# Test
batch, seq_len, dim = 1, 5, 4
q = torch.randn(batch, seq_len, dim)
k = torch.randn(batch, seq_len, dim)
positions = torch.arange(seq_len)

q_rot, k_rot = simple_rope_4d(q, k, positions)

# Verify: score(m, n) only depends on (n - m)
for m in range(seq_len):
    for n in range(seq_len):
        score_mn = (q_rot[0, m] @ k_rot[0, n]).item()
        # Find another pair with same distance
        if m + 1 < seq_len and n + 1 < seq_len:
            score_m1_n1 = (q_rot[0, m+1] @ k_rot[0, n+1]).item()
            print(f"score({m}, {n}) = {score_mn:.4f}, score({m+1}, {n+1}) = {score_m1_n1:.4f}")
```
</details>

### Exercise 2: Visualize Rotation Angles
Create a visualization showing how different dimension pairs rotate at different speeds across positions.

<details>
<summary>Solution</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_rope_frequencies(dim=64, max_pos=100, base=10000):
    """Visualize RoPE rotation angles across positions."""

    # Compute frequencies
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    positions = np.arange(max_pos)

    # Angles for each dimension pair
    angles = np.outer(positions, inv_freq)  # (max_pos, dim/2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Angle vs position for different dimension pairs
    ax = axes[0, 0]
    for i in [0, dim//8, dim//4, dim//2-1]:
        ax.plot(positions, angles[:, i], label=f'Dim pair {i}')
    ax.set_xlabel('Position')
    ax.set_ylabel('Rotation Angle (radians)')
    ax.set_title('Rotation Angles vs Position')
    ax.legend()
    ax.grid(True)

    # Plot 2: Heatmap of all angles
    ax = axes[0, 1]
    im = ax.imshow(angles.T, aspect='auto', cmap='twilight',
                   extent=[0, max_pos, 0, dim//2])
    ax.set_xlabel('Position')
    ax.set_ylabel('Dimension Pair')
    ax.set_title('Rotation Angles Heatmap')
    plt.colorbar(im, ax=ax, label='Angle (radians)')

    # Plot 3: Frequency spectrum
    ax = axes[1, 0]
    ax.plot(inv_freq)
    ax.set_xlabel('Dimension Pair')
    ax.set_ylabel('Rotation Frequency')
    ax.set_title('RoPE Frequency Spectrum')
    ax.set_yscale('log')
    ax.grid(True)

    # Plot 4: Wavelengths
    ax = axes[1, 1]
    wavelengths = 2 * np.pi / inv_freq
    ax.plot(wavelengths)
    ax.set_xlabel('Dimension Pair')
    ax.set_ylabel('Wavelength (positions)')
    ax.set_title('RoPE Wavelengths')
    ax.set_yscale('log')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('rope_frequencies.png', dpi=150)
    plt.show()

visualize_rope_frequencies()
```
</details>

### Exercise 3: Measure Extrapolation Performance
Implement a test to measure how well RoPE extrapolates to sequences 2x, 4x, and 8x longer than training.

<details>
<summary>Solution</summary>

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def test_rope_extrapolation():
    """Test RoPE extrapolation to longer sequences."""

    dim = 64
    n_heads = 8
    head_dim = dim // n_heads
    train_len = 512

    # Create RoPE modules
    rope_standard = RotaryPositionalEmbedding(head_dim, max_seq_len=train_len)
    rope_interpolated = InterpolatedRoPE(
        head_dim, max_train_len=train_len, max_infer_len=train_len*8
    )
    rope_ntk = NTKRoPE(
        head_dim, max_train_len=train_len, max_infer_len=train_len*8
    )

    # Test on various lengths
    test_lengths = [train_len, train_len*2, train_len*4, train_len*8]

    results = {
        'standard': [],
        'interpolated': [],
        'ntk': []
    }

    for length in test_lengths:
        # Create random Q, K
        q = torch.randn(1, length, n_heads, head_dim)
        k = torch.randn(1, length, n_heads, head_dim)

        # Apply RoPE
        try:
            q_std, k_std = rope_standard(q, k)
            # Measure attention score variance (as proxy for quality)
            scores = torch.matmul(
                q_std.transpose(1, 2),
                k_std.transpose(1, 2).transpose(-2, -1)
            ) / math.sqrt(head_dim)
            results['standard'].append(scores.std().item())
        except:
            results['standard'].append(float('nan'))

        q_interp, k_interp = rope_interpolated(q, k)
        scores = torch.matmul(
            q_interp.transpose(1, 2),
            k_interp.transpose(1, 2).transpose(-2, -1)
        ) / math.sqrt(head_dim)
        results['interpolated'].append(scores.std().item())

        q_ntk, k_ntk = rope_ntk(q, k)
        scores = torch.matmul(
            q_ntk.transpose(1, 2),
            k_ntk.transpose(1, 2).transpose(-2, -1)
        ) / math.sqrt(head_dim)
        results['ntk'].append(scores.std().item())

    # Plot results
    plt.figure(figsize=(10, 6))
    x = [f'{l//train_len}x' for l in test_lengths]

    plt.plot(x, results['standard'], marker='o', label='Standard RoPE')
    plt.plot(x, results['interpolated'], marker='s', label='Position Interpolation')
    plt.plot(x, results['ntk'], marker='^', label='NTK Scaling')

    plt.xlabel('Sequence Length (relative to training)')
    plt.ylabel('Attention Score Std Dev')
    plt.title('RoPE Extrapolation Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('rope_extrapolation.png', dpi=150)
    plt.show()

test_rope_extrapolation()
```
</details>

### Exercise 4: Compare Memory Usage
Calculate and compare the memory usage of different positional encoding methods for a 7B parameter model with 32 layers, 32 heads, 128 head dimension, and 100K context length.

<details>
<summary>Solution</summary>

```python
def compare_memory_usage():
    """Compare memory usage of different positional encodings."""

    # Model configuration (LLaMA-style 7B)
    n_layers = 32
    n_heads = 32
    head_dim = 128
    d_model = n_heads * head_dim  # 4096
    max_seq_len = 100_000
    batch_size = 1

    bytes_per_param = 2  # FP16

    # 1. Learned absolute positional embeddings
    learned_params = max_seq_len * d_model
    learned_bytes = learned_params * bytes_per_param

    # 2. Sinusoidal (no parameters, but need to store the embeddings)
    sinusoidal_params = 0
    sinusoidal_runtime = max_seq_len * d_model * bytes_per_param  # Stored embeddings

    # 3. RoPE
    rope_params = 0
    rope_cache = max_seq_len * (d_model // 2) * 2 * 4  # cos and sin, FP32

    # 4. During inference: KV cache (affected by position encoding)
    # Standard: store full K, V
    kv_cache_standard = (
        2 *  # K and V
        n_layers *
        batch_size *
        max_seq_len *
        n_heads *
        head_dim *
        bytes_per_param
    )

    print("="*60)
    print("Memory Usage Comparison for 7B Model @ 100K Context")
    print("="*60)

    print("\nPositional Encoding Parameters:")
    print(f"  Learned Absolute:  {learned_bytes / 1e9:.2f} GB")
    print(f"  Sinusoidal:        {sinusoidal_params / 1e9:.2f} GB (no params)")
    print(f"  RoPE:              {rope_params / 1e9:.2f} GB (no params)")

    print("\nRuntime Memory (without KV cache):")
    print(f"  Learned Absolute:  {learned_bytes / 1e9:.2f} GB")
    print(f"  Sinusoidal:        {sinusoidal_runtime / 1e9:.2f} GB")
    print(f"  RoPE:              {rope_cache / 1e9:.2f} GB (cache)")

    print("\nKV Cache (for all methods):")
    print(f"  {kv_cache_standard / 1e9:.2f} GB")

    print("\nTotal Inference Memory:")
    print(f"  Learned Absolute:  {(learned_bytes + kv_cache_standard) / 1e9:.2f} GB")
    print(f"  Sinusoidal:        {(sinusoidal_runtime + kv_cache_standard) / 1e9:.2f} GB")
    print(f"  RoPE:              {(rope_cache + kv_cache_standard) / 1e9:.2f} GB")

    print("\n" + "="*60)
    print(f"RoPE Memory Saving vs Learned: {(learned_bytes - rope_cache) / 1e9:.2f} GB")
    print("="*60)

compare_memory_usage()
```
</details>

### Exercise 5: Implement and Test YaRN
Implement YaRN scaling and test it on a sequence 16x longer than training length.

<details>
<summary>Solution</summary>

```python
# Use the YaRNRoPE implementation from the chapter

def test_yarn():
    """Test YaRN on extended context."""

    dim = 128
    train_len = 2048
    test_len = 32768  # 16x longer

    # Create models
    rope_standard = RotaryPositionalEmbedding(dim, max_seq_len=test_len)
    rope_interpolated = InterpolatedRoPE(dim, train_len, test_len)
    rope_yarn = YaRNRoPE(dim, train_len, test_len)

    # Create test data
    batch, seq_len, n_heads = 1, test_len, 8
    q = torch.randn(batch, seq_len, n_heads, dim)
    k = torch.randn(batch, seq_len, n_heads, dim)

    print("Testing RoPE variants on 16x extended context...")

    # Standard (extrapolation)
    q_std, k_std = rope_standard(q, k)
    print(f"Standard RoPE: Q norm = {q_std.norm():.4f}")

    # Position Interpolation
    q_interp, k_interp = rope_interpolated(q, k)
    print(f"Position Interpolation: Q norm = {q_interp.norm():.4f}")

    # YaRN
    q_yarn, k_yarn = rope_yarn(q, k)
    print(f"YaRN: Q norm = {q_yarn.norm():.4f}")
    print(f"YaRN attention scale: {rope_yarn.get_attention_scale():.4f}")

    # Compare attention patterns at different distances
    print("\nAttention scores at different distances:")
    positions = [0, 1024, 2048, 4096, 8192, 16384]

    for i in range(len(positions) - 1):
        pos1, pos2 = positions[i], positions[i+1]

        # YaRN scores
        score = (q_yarn[0, pos1] @ k_yarn[0, pos2].T).mean()
        print(f"Distance {pos2-pos1:5d}: score = {score:.4f}")

test_yarn()
```
</details>

---

## References

### Primary Papers

1. **RoFormer: Enhanced Transformer with Rotary Position Embedding**
   Su, Jianlin, et al. (2021)
   https://arxiv.org/abs/2104.09864
   *The original RoPE paper. Essential reading.*

2. **Extending Context Window of Large Language Models via Position Interpolation**
   Chen, Shouyuan, et al. (2023)
   https://arxiv.org/abs/2306.15595
   *Position Interpolation method.*

3. **NTK-Aware Scaled RoPE**
   Reddit: /u/bloc97 (2023)
   https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
   *Community-driven discovery of NTK scaling.*

4. **YaRN: Efficient Context Window Extension of Large Language Models**
   Peng, Bowen, et al. (2023)
   https://arxiv.org/abs/2309.00071
   *State-of-the-art RoPE scaling method.*

### Implementation References

5. **LLaMA: Open and Efficient Foundation Language Models**
   Touvron, Hugo, et al. (2023)
   https://arxiv.org/abs/2302.13971
   *LLaMA uses RoPE as standard positional encoding.*

6. **Mistral 7B**
   Jiang, Albert Q., et al. (2023)
   https://arxiv.org/abs/2310.06825
   *RoPE with sliding window attention.*

7. **Qwen2.5 Technical Report**
   Qwen Team (2024)
   https://arxiv.org/abs/2412.15115
   *ABF scaling for extended context.*

8. **The Llama 4 Herd**
   Meta AI (2025)
   https://ai.meta.com/blog/llama-4-multimodal-intelligence/
   *Introduces iRoPE (interleaved RoPE/NoPE).*

### Related Chapters

- [Chapter 7: Positional Encodings](07-positional-encodings.md) - Other positional encoding methods
- [Chapter 4: Multi-Head Attention](04-multi-head-attention.md) - Attention mechanisms that use RoPE
- [Chapter 26: Long Context Techniques](26-long-context.md) - Advanced RoPE scaling and other long-context methods
- [Chapter 29: Architecture Comparison](29-model-architectures.md) - Which models use RoPE
- [Chapter 12: Flash Attention](12-flash-attention.md) - Efficient attention that works with RoPE

### Code Resources

- **llama.cpp**: Efficient C++ implementation with RoPE
  https://github.com/ggerganov/llama.cpp

- **transformers (Hugging Face)**: RoPE in LLaMA, Mistral, etc.
  https://github.com/huggingface/transformers

- **xFormers**: Optimized RoPE implementations
  https://github.com/facebookresearch/xformers

---

**Next Chapter**: [The Transformer Block](09-transformer-block.md) - Combine attention with normalization, feedforward, and residual connections.

**Previous Chapter**: [Positional Encodings](07-positional-encodings.md) - Other approaches to encoding position information.
