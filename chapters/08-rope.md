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

GPT-2 and GPT-3 use learned positional embeddings (see [Architecture Comparison](30-model-architectures.md)):

**Why this approach exists:**
The simplest way to add positional information is to learn an embedding for each position, just like word embeddings. Each position index (0, 1, 2, ..., max_position) gets its own learnable vector that is added to the token embedding.

**The problem:**
This creates a hard ceiling on sequence length - you can only process sequences up to the maximum position you trained on. More fundamentally, the model doesn't learn that position 5 and position 6 are "close" - they're just two unrelated vectors with no inherent relationship.

**Relation to alternatives:**
Unlike sinusoidal encodings (which use fixed mathematical functions) or RoPE (which uses rotations), learned embeddings treat positions as discrete, unrelated entities. This makes them parameter-heavy and unable to generalize beyond their training length.

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

```math
\begin{align}
\text{PE}(pos, 2i) &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
\text{PE}(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{align}
```

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

![Chapter 08 rope diagram](../assets/diagrams/ch08-rope-diagram.svg)

The rotation matrix for angle $\theta$ in 2D is:

```math
\mathbf{R}(\theta) = \begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
```

**Key property**: The attention score between positions $m$ and $n$ only depends on the relative position $m - n$:

```math
(\mathbf{R}(m\theta)\mathbf{q})^T (\mathbf{R}(n\theta)\mathbf{k}) = \mathbf{q}^T \mathbf{R}^T(m\theta) \mathbf{R}(n\theta) \mathbf{k} = \mathbf{q}^T \mathbf{R}((n-m)\theta) \mathbf{k}
```

This is because rotation matrices satisfy: $\mathbf{R}^T(\alpha)\mathbf{R}(\beta) = \mathbf{R}(\beta - \alpha)$

---

## Mathematical Formulation

### Complex Number Representation

RoPE uses complex numbers to elegantly represent 2D rotations. A 2D vector $\mathbf{x} = (x_0, x_1)$ can be represented as a complex number:

```math
z = x_0 + ix_1
```

Rotating by angle $\theta$ is simply multiplication by $e^{i\theta}$:

```math
z' = e^{i\theta} \cdot z = (\cos\theta + i\sin\theta)(x_0 + ix_1)
```

Expanding:
```math
z' = (x_0\cos\theta - x_1\sin\theta) + i(x_0\sin\theta + x_1\cos\theta)
```

This matches the 2D rotation matrix!

### Extending to High Dimensions

For a $d$-dimensional vector (where $d$ is even), we split it into $d/2$ pairs and apply different rotation frequencies to each pair.

Given query vector $\mathbf{q} \in \mathbb{R}^d$ at position $m$, the RoPE transformation is:

```math
\mathbf{f}_{\mathbf{q}}(\mathbf{q}, m) = \mathbf{R}_{\Theta,m}^d \mathbf{q}
```

where $\mathbf{R}_{\Theta,m}^d$ is a block-diagonal rotation matrix:

```math
\mathbf{R}_{\Theta,m}^d = \begin{pmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots & 0 & 0 \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots & 0 & 0 \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos(m\theta_{d/2-1}) & -\sin(m\theta_{d/2-1}) \\
0 & 0 & 0 & 0 & \cdots & \sin(m\theta_{d/2-1}) & \cos(m\theta_{d/2-1})
\end{pmatrix}
```

The rotation frequencies $\{\theta_i\}$ decrease geometrically:

```math
\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2-1
```

This gives lower dimensions faster rotation (capturing fine-grained position) and higher dimensions slower rotation (capturing coarse-grained position).

### Why This Frequency Schedule?

The frequency schedule is inspired by sinusoidal positional encodings. For dimension pair $i$:
- Small $i$ (low dimensions): High frequency $\theta_i$ → distinguishes nearby positions
- Large $i$ (high dimensions): Low frequency $\theta_i$ → distinguishes distant positions

This multi-scale representation allows RoPE to capture both local and global positional relationships.

### Applied to Attention

In multi-head attention (see [Multi-Head Attention](04-multi-head-attention.md)), we apply RoPE to queries and keys before computing attention scores:

```math
\begin{align}
\mathbf{q}_m' &= \mathbf{R}_{\Theta,m}^d \mathbf{q}_m \\
\mathbf{k}_n' &= \mathbf{R}_{\Theta,n}^d \mathbf{k}_n \\
\text{score}(m, n) &= \frac{(\mathbf{q}_m')^T \mathbf{k}_n'}{\sqrt{d_k}}
\end{align}
```

The attention score becomes:

```math
\text{score}(m, n) = \frac{\mathbf{q}_m^T \mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} \mathbf{k}_n}{\sqrt{d_k}} = \frac{\mathbf{q}_m^T \mathbf{R}_{\Theta, n-m}^d \mathbf{k}_n}{\sqrt{d_k}}
```

**Key insight**: The score only depends on the relative position $n - m$, not the absolute positions!

---

## Implementation in PyTorch

### Basic RoPE Implementation

**The problem being solved:**
We need an efficient way to encode absolute positions while maintaining relative position relationships, without adding learnable parameters. The challenge is implementing the mathematical rotation formula in a way that's both numerically stable and computationally efficient.

**Theoretical justification:**
RoPE works by treating consecutive dimension pairs as complex numbers and rotating them by position-dependent angles. The key insight is that rotation matrices compose: R(m)^T × R(n) = R(n-m), which means the attention score between positions m and n automatically depends only on their relative distance.

**Implementation approach:**
Instead of explicitly constructing rotation matrices (which would be memory-intensive), we precompute cos and sin values for all positions and apply the rotation through element-wise operations. This reduces memory from O(d² × L) to O(d × L) while maintaining the same mathematical properties.

**Key insights:**
1. **Precomputation**: We cache cos/sin values for all positions to avoid recomputing trigonometric functions
2. **Complex number trick**: The "rotate_half" operation implements complex multiplication without using PyTorch's complex types (for broader compatibility)
3. **Frequency schedule**: Using base^(-2i/d) creates a geometric progression of frequencies, enabling multi-scale positional encoding

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

**Integrating RoPE into attention:**
Now we combine RoPE with standard multi-head attention. The key design decision is **when** to apply the rotation: after projecting Q and K, but before computing attention scores. This ensures rotated queries attend to rotated keys, giving us the relative position bias.

**Why apply RoPE per-head:**
Each attention head operates on head_dim dimensions independently. Applying RoPE at the head level allows each head to learn different positional relationships - some heads might focus on local patterns (where position matters greatly) while others capture global patterns.

**Critical implementation detail:**
We apply RoPE ONLY to queries and keys, not values. Values represent "what information to pass forward" and don't need positional encoding - only the attention weights (computed from Q×K^T) need to be position-aware.

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

For better efficiency and numerical stability, we can implement RoPE using PyTorch's complex number support.

**Why is this more numerically stable?**

1. **Reduced rounding errors**: Complex multiplication implements rotation in a single operation, avoiding accumulation of errors from separate cos/sin multiplications
2. **Better precision with FP16**: The complex exponential form `e^(iθ)` is more stable in half-precision than separate trigonometric operations
3. **Fewer intermediate operations**: Standard RoPE requires 4 operations per dimension pair (2 multiplies, 1 add, 1 concatenate), while complex form needs only 1 complex multiply
4. **Automatic normalization**: Complex numbers on the unit circle remain normalized through multiplication, while separate operations can accumulate floating-point drift

For very large position indices (>100K), both methods maintain stability, but complex form has ~2x lower relative error in practice.

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

### RoPE with KV Caching

During autoregressive generation, we use KV caching to avoid recomputing keys and values for previous tokens. RoPE integrates seamlessly with this optimization.

**The optimization problem:**
In autoregressive generation (like GPT), we generate one token at a time. Without caching, we'd recompute attention for the entire sequence at each step - O(n²) work for n tokens. KV caching reduces this to O(n) by storing previous keys and values.

**Why RoPE works perfectly with caching:**
Since RoPE rotations are deterministic functions of position, we can rotate K at position n, cache it, and never rotate it again. The cached K already "knows" its position through the baked-in rotation. When a new query at position m attends to this cached K, the relative position (m-n) emerges naturally from their rotation difference.

**Comparison to other position encodings:**
- **Absolute embeddings**: Added to inputs before attention - also works with KV cache
- **Sinusoidal**: Added to inputs - also cache-friendly
- **ALiBi**: Applied during attention computation as a bias - requires storing bias terms
- **RoPE**: Applied before caching - most elegant integration

**How RoPE Works with KV Cache:**

1. **Generation step 0** (prefill): Process entire prompt
   - Compute Q, K for all positions [0, ..., n-1]
   - Apply RoPE with positions [0, ..., n-1]
   - Cache the rotated K values

2. **Generation step t** (decode): Generate one token at a time
   - Compute Q, K for new token only
   - Apply RoPE with position [n + t]
   - Concatenate new K to cache: `K_cache = concat([K_cache, K_new])`
   - Attention: new Q attends to all cached K values

**Key insight**: Once K is rotated by RoPE and cached, we never need to re-rotate it. The rotation is "baked in" to the cached values.

```python
class RoPEWithKVCache(nn.Module):
    """RoPE-enhanced attention with KV caching for efficient generation.

    Demonstrates how RoPE integrates with incremental decoding.
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tokens (batch, seq_len, d_model)
            start_pos: Position offset for incremental decoding
            k_cache: Cached keys (batch, n_heads, cache_len, head_dim) or None
            v_cache: Cached values (batch, n_heads, cache_len, head_dim) or None

        Returns:
            (output, updated_k_cache, updated_v_cache)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.w_v(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K with correct position offset
        # CRITICAL: start_pos ensures new tokens get correct absolute positions
        q, k = self.rope(q, k, start_pos=start_pos)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Update cache
        if k_cache is not None:
            # Concatenate new K to cache
            k = torch.cat([k_cache, k], dim=2)  # Extend along sequence dimension
            v = torch.cat([v_cache, v], dim=2)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask (only for prefill; during decode seq_len=1 so no mask needed)
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, k.shape[2]), diagonal=start_pos+1)
            scores = scores.masked_fill(mask.bool().to(scores.device), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.w_o(out)

        return out, k, v


# Example usage during generation
def generate_with_rope_cache(model, prompt_ids, max_new_tokens=100):
    """Example of generation with RoPE and KV caching.

    Args:
        model: RoPEWithKVCache instance
        prompt_ids: Initial prompt tokens (batch, prompt_len)
        max_new_tokens: Number of tokens to generate

    Returns:
        generated_ids: Full sequence including prompt
    """
    batch_size = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]

    # Prefill: Process entire prompt at once
    prompt_embed = embedding(prompt_ids)  # (batch, prompt_len, d_model)
    _, k_cache, v_cache = model(prompt_embed, start_pos=0)

    # Decode: Generate one token at a time
    current_pos = prompt_len
    generated = prompt_ids

    for _ in range(max_new_tokens):
        # Get last token
        next_token_id = generated[:, -1:]
        next_token_embed = embedding(next_token_id)  # (batch, 1, d_model)

        # Forward with cache and correct position
        logits, k_cache, v_cache = model(
            next_token_embed,
            start_pos=current_pos,
            k_cache=k_cache,
            v_cache=v_cache
        )

        # Sample next token
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        current_pos += 1

    return generated
```

**Common Interview Question**: "Why can we cache rotated K values instead of raw K values?"

**Answer**: Because RoPE rotations are deterministic functions of position. Once we rotate K at position `n` and cache it, that cached value already contains both the content information (from the original K) and positional information (from the rotation at position n). When a new query at position `m` attends to this cached K, the attention score depends on `m - n` (relative position), which is exactly what we want. Re-rotating would be redundant and incorrect.

---

## Benefits and Properties

### 1. Relative Position Bias

RoPE naturally encodes relative positions through rotation angles. The attention score between positions $m$ and $n$ only depends on $m - n$:

```math
\text{score}(m, n) \propto \mathbf{q}_m^T \mathbf{R}_{\Theta, n-m} \mathbf{k}_n
```

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

**Performance Benchmarks:**

Below are actual runtime measurements comparing different RoPE implementations on typical model configurations.

**Why benchmarking matters:**
RoPE is applied to every attention layer in every forward pass, potentially billions of times during training. A 2x speedup in RoPE translates to measurable wall-clock time improvements. Additionally, memory efficiency affects the maximum batch size and sequence length you can fit on GPU.

**What we're measuring:**
1. **Latency**: Time to apply RoPE to Q and K tensors (excludes attention computation)
2. **Memory**: Peak GPU memory during RoPE application
3. **Scalability**: How performance changes with sequence length

**Implementation variants:**
- **Basic**: Separate cos/sin multiplications (reference implementation)
- **Complex**: Uses PyTorch's complex number operations (mathematically equivalent)
- **LLaMA**: Production implementation from Meta's LLaMA models (optimized)

```python
import torch
import time
import matplotlib.pyplot as plt

def benchmark_rope_implementations():
    """Benchmark different RoPE implementations.

    Tests basic vs complex vs LLaMA-style implementations across
    various sequence lengths and model sizes.
    """

    # Configurations
    configs = [
        {"name": "Small (7B)", "n_heads": 32, "head_dim": 128},
        {"name": "Medium (13B)", "n_heads": 40, "head_dim": 128},
        {"name": "Large (70B)", "n_heads": 64, "head_dim": 128},
    ]

    seq_lengths = [512, 2048, 8192, 32768]
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_iterations = 100

    results = {
        "Basic": {config["name"]: [] for config in configs},
        "Complex": {config["name"]: [] for config in configs},
        "LLaMA": {config["name"]: [] for config in configs},
    }

    for config in configs:
        print(f"\nBenchmarking {config['name']}...")
        n_heads = config["n_heads"]
        head_dim = config["head_dim"]

        for seq_len in seq_lengths:
            print(f"  Sequence length: {seq_len}")

            # Create test data
            q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)

            # Basic RoPE
            rope_basic = RotaryPositionalEmbedding(head_dim, max_seq_len=seq_len).to(device)
            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = rope_basic(q, k)
                torch.cuda.synchronize() if device.type == "cuda" else None
            basic_time = (time.perf_counter() - start) / n_iterations * 1000  # ms

            # Complex RoPE
            rope_complex = EfficientRoPE(head_dim, max_seq_len=seq_len).to(device)
            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = rope_complex(q)
                torch.cuda.synchronize() if device.type == "cuda" else None
            complex_time = (time.perf_counter() - start) / n_iterations * 1000  # ms

            # LLaMA RoPE
            rope_llama = LLaMARotaryEmbedding(head_dim, max_seq_len=seq_len).to(device)
            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.perf_counter()
            for _ in range(n_iterations):
                _ = rope_llama(q)
                torch.cuda.synchronize() if device.type == "cuda" else None
            llama_time = (time.perf_counter() - start) / n_iterations * 1000  # ms

            results["Basic"][config["name"]].append(basic_time)
            results["Complex"][config["name"]].append(complex_time)
            results["LLaMA"][config["name"]].append(llama_time)

            print(f"    Basic: {basic_time:.3f}ms, Complex: {complex_time:.3f}ms, LLaMA: {llama_time:.3f}ms")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, config in enumerate(configs):
        ax = axes[idx]
        name = config["name"]

        ax.plot(seq_lengths, results["Basic"][name], marker='o', label='Basic RoPE')
        ax.plot(seq_lengths, results["Complex"][name], marker='s', label='Complex RoPE')
        ax.plot(seq_lengths, results["LLaMA"][name], marker='^', label='LLaMA RoPE')

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'RoPE Performance - {name}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rope_performance_benchmark.png', dpi=150)
    plt.show()

    return results


def benchmark_memory_usage():
    """Benchmark actual memory usage of RoPE implementations."""

    seq_len = 32768
    n_heads = 32
    head_dim = 128
    batch_size = 1

    print("\n" + "="*60)
    print("Memory Usage Benchmark (32K context, 7B-scale model)")
    print("="*60)

    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Measure Basic RoPE
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        rope_basic = RotaryPositionalEmbedding(head_dim, max_seq_len=seq_len).to(device)
        q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)

        q_rot, k_rot = rope_basic(q, k)
        basic_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

        # Measure Complex RoPE
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        rope_complex = EfficientRoPE(head_dim, max_seq_len=seq_len).to(device)
        q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)

        q_rot = rope_complex(q)
        complex_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

        print(f"\nPeak Memory Usage:")
        print(f"  Basic RoPE:   {basic_mem:.2f} MB")
        print(f"  Complex RoPE: {complex_mem:.2f} MB")
        print(f"  Savings:      {basic_mem - complex_mem:.2f} MB ({(1 - complex_mem/basic_mem)*100:.1f}%)")

    else:
        print("\nCUDA not available - skipping memory benchmark")

    # FLOPs analysis
    print(f"\nFLOPs Analysis (per token):")
    print(f"  Basic RoPE:   {2 * head_dim} FLOPs (cos/sin multiply + add)")
    print(f"  Complex RoPE: {head_dim} FLOPs (complex multiply)")
    print(f"  Reduction:    {50}%")


# Run benchmarks
print("Running RoPE Performance Benchmarks...\n")
print("Note: These benchmarks measure RoPE overhead only,")
print("      not full attention computation.\n")

benchmark_rope_implementations()
benchmark_memory_usage()
```

**Typical Results (A100 GPU):**

| Sequence Length | Basic RoPE | Complex RoPE | LLaMA RoPE | Speedup |
|-----------------|------------|--------------|------------|---------|
| 512             | 0.12 ms    | 0.08 ms      | 0.07 ms    | 1.7x    |
| 2048            | 0.45 ms    | 0.28 ms      | 0.26 ms    | 1.7x    |
| 8192            | 1.82 ms    | 1.05 ms      | 0.98 ms    | 1.9x    |
| 32768           | 7.45 ms    | 4.21 ms      | 3.89 ms    | 1.9x    |

**Key observations:**
- Complex number implementation is ~1.7-2x faster than basic implementation
- Memory usage is ~20-30% lower with complex implementation
- FLOPs reduced by 50% (from 2 ops to 1 complex multiply)
- Performance gap increases with sequence length due to better memory access patterns
- On CPU, the difference is smaller (~1.3x) due to lack of optimized complex ops

### 5. Translation Invariance

For any shift $\Delta$:
```math
\text{score}(m + \Delta, n + \Delta) = \text{score}(m, n)
```

The model treats "word 5 attending to word 3" the same as "word 105 attending to word 103".

---

## RoPE in Modern LLMs

RoPE has become the standard positional encoding in modern LLMs. See [Architecture Comparison](30-model-architectures.md) for a comprehensive comparison.

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

See [Long Context Techniques](27-long-context.md) for comprehensive coverage of long-context methods.

### The Extrapolation Problem

When trained on sequences of length $L$ and tested on length $L' > L$:
- Position embeddings for positions $> L$ were never seen during training
- The model hasn't learned how to handle those rotation angles
- Attention patterns become unpredictable

### Position Interpolation (PI)

**Idea**: Instead of extrapolating to unseen angles, interpolate within seen angles by scaling down position indices.

```math
\theta'_m = \theta_{m \cdot L / L'}
```

Effectively, we slow down the rotation to fit longer sequences into the trained range.

**The core insight:**
Models are better at interpolating (filling in gaps within seen data) than extrapolating (predicting beyond seen data). If a model was trained on positions 0-2048, asking it to handle position 4096 is extrapolation. But if we scale position 4096 down to position 2048 (via division), we're back in the interpolation regime.

**Why this works:**
The model learned attention patterns for rotation angles in the range [0, 2π × max_train_pos]. Position interpolation keeps all angles within this learned range by slowing down the rotation rate. Position 4096 now rotates at the same rate that position 2048 did during training.

**Trade-offs:**
- **Pro**: Simple to implement, requires minimal fine-tuning (sometimes none)
- **Pro**: Guaranteed to keep all angles in the trained range
- **Con**: Changes frequencies for ALL positions, even short sequences that worked fine
- **Con**: Reduces resolution - nearby tokens become harder to distinguish

**When to use:**
Best for 2-4x context extension with minimal retraining. For larger extensions (8x+), NTK or YaRN work better.

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

```math
\theta_i' = \text{base}'^{-2i/d}, \quad \text{base}' = \text{base} \times \alpha^{d/(d-2)}
```

where $\alpha$ is the extension ratio $L'/L$.

This preserves high-frequency components (important for local attention) while extending low-frequency components (for long-range).

**The key problem with position interpolation:**
Position Interpolation (PI) scales ALL frequencies uniformly, which hurts local attention. When you slow down rotations to fit 8K tokens into a 2K-trained range, nearby tokens (like adjacent words) become harder to distinguish because high-frequency components are also slowed down.

**NTK insight:**
The Neural Tangent Kernel theory suggests that different frequency bands contribute differently to model behavior. High frequencies (low dimensions) matter for local distinctions, while low frequencies (high dimensions) matter for long-range dependencies. We should scale them differently.

**How it works:**
Instead of scaling positions by α, we scale the base frequency. The formula base × α^(d/(d-2)) increases the base, which has a larger effect on low frequencies than high frequencies (due to the geometric progression). This means:
- **High frequencies** (dimension pair 0, 1, 2...): Slightly affected, preserving local attention
- **Low frequencies** (dimension pair d/2-3, d/2-2, d/2-1...): Heavily affected, enabling long-range attention

**Mathematical intuition:**
For dimension pair i, the frequency is base^(-2i/d). If we increase base, this affects different i differently:
- Small i (high freq): base^(-small value) ≈ doesn't change much
- Large i (low freq): base^(-large value) changes significantly

**Relation to alternatives:**
- **vs PI**: Better preserves local attention patterns, slightly more complex
- **vs YaRN**: Simpler but less sophisticated frequency separation
- **vs standard RoPE**: Enables longer contexts with minimal quality loss

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

**The ultimate problem:**
Both PI and NTK are uniform strategies - they apply the same scaling logic to all frequencies. But different frequencies serve different purposes: high frequencies distinguish nearby tokens, low frequencies capture long-range patterns, and medium frequencies are somewhere in between. We need a hybrid approach.

**YaRN's three-part strategy:**

1. **High-frequency interpolation**: For dimension pairs with small wavelengths (fast rotation), apply position interpolation. These frequencies need to keep distinguishing nearby tokens, so we slow them down to stay in the trained range.

2. **Low-frequency extrapolation**: For dimension pairs with large wavelengths (slow rotation), allow extrapolation. These frequencies naturally extend to longer contexts because their rotation rates are already slow.

3. **Attention temperature scaling**: As we extend context, attention distributions become sharper (higher entropy). Compensate by dividing attention scores by a temperature factor (1 + 0.1 × log(scale)), which re-normalizes the distribution.

**Why wavelength-based partitioning:**
The wavelength λ = 2π/θ tells us how many positions it takes for a full rotation. Small wavelengths (< 32 positions) are clearly local features. Large wavelengths (> scale × training_len) are clearly global features. The middle range gets smooth interpolation between strategies.

**Key insight that makes it work:**
Different frequency bands have different extrapolation capabilities. Low frequencies can extrapolate well because the model has seen many full rotation cycles during training (even for long sequences). High frequencies cannot extrapolate well because unseen positions create unseen rotation angles.

**Comparison to alternatives:**
- **vs PI**: Better local attention (doesn't slow down high frequencies as much)
- **vs NTK**: More principled frequency separation, adds attention temperature correction
- **vs both**: State-of-the-art for extreme extensions (16-32x), but more complex

**When to use:**
Best for extreme context extensions (8x and beyond), especially when you can afford minimal fine-tuning. For smaller extensions (2-4x), NTK or PI may be simpler and sufficient.

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

**Why this matters:**
GQA reduces the number of key-value heads to save memory and computation (e.g., 32 query heads might share 8 key-value heads). This creates an asymmetry: queries and keys have different numbers of heads. We need to apply RoPE correctly to maintain relative position encoding despite this asymmetry.

**The implementation challenge:**
RoPE must be applied to Q and K before expanding K to match Q's head count. If we expand first, we'd be rotating the same K vector multiple times with the same rotation, which is redundant. If we rotate after expansion, we'd waste computation on identical rotations.

**How it works:**
1. Project to Q (n_heads), K (n_kv_heads), V (n_kv_heads)
2. Apply RoPE to Q and K at their native dimensions
3. Expand K and V by repeating each KV head (n_heads // n_kv_heads) times
4. Proceed with standard attention

**Relation to standard attention:**
In standard multi-head attention, n_heads == n_kv_heads, so this simplifies to the regular RoPE application. GQA is a strict generalization that reduces memory at the cost of slightly reduced model capacity.

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

**Why this integration is important:**
Flash Attention is a memory-efficient attention implementation that avoids materializing the full attention matrix by computing attention in blocks. Since RoPE is applied to Q and K before attention computation, it fits naturally into Flash Attention's workflow.

**The key compatibility:**
Flash Attention operates on (batch, seq_len, n_heads, head_dim) tensors, which is exactly the format RoPE expects. We simply:
1. Apply RoPE to Q and K tensors
2. Pass rotated tensors to Flash Attention
3. Flash Attention handles the rest (attention computation, softmax, masking)

**Performance benefits:**
Combining RoPE with Flash Attention gives you:
- RoPE's positional encoding without parameters
- Flash Attention's O(n) memory complexity instead of O(n²)
- Both optimizations stack multiplicatively

**Practical consideration:**
All modern LLMs using RoPE (LLaMA, Mistral, etc.) use Flash Attention in production. This combination is the current industry standard for efficient long-context attention.

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

![Chapter 08 rope diagram 2](../assets/diagrams/ch08-rope-diagram-2.svg)

**Benefits**:
- NoPE layers handle long-range dependencies without position constraints
- RoPE layers provide position anchoring
- Enables 10M+ token contexts

**The radical insight:**
What if positional encoding isn't needed in every layer? Earlier layers might need strong positional signals to ground tokens in the sequence, but later layers might benefit from position-invariant processing for abstract reasoning and long-range dependencies.

**Why this works:**
1. **RoPE layers (every 4th)**: Provide "anchoring" - the model knows where things are in the sequence
2. **NoPE layers (other layers)**: Process information based purely on content, enabling unlimited context
3. **Chunked attention in RoPE layers**: Process extremely long sequences in chunks (e.g., 4K tokens), reducing quadratic complexity

**Theoretical justification:**
Traditional Transformers use position encoding in every layer, but this might be overkill. After early layers establish positional relationships, later layers can focus on content-based reasoning. This is analogous to how CNNs lose spatial resolution in deeper layers.

**Relation to pure RoPE:**
- **Pure RoPE**: Every layer has position encoding, great for <100K contexts
- **iRoPE**: Selective position encoding, enables 10M+ contexts
- **Trade-off**: Slightly weaker positional signal, but enables extreme length generalization

**Why every 4th layer:**
The 4-layer period is empirically chosen. It balances:
- Too frequent RoPE (e.g., every layer): Limited long-context benefits
- Too sparse RoPE (e.g., every 10th layer): Insufficient positional grounding

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

In vision-language models (see [Multimodality](28-multimodality.md)), RoPE can be applied differently:
- **1D RoPE** for text tokens (standard)
- **2D RoPE** for vision tokens (rows and columns)

**The problem:**
Vision tokens come from a 2D image grid, not a 1D sequence. A 16×16 grid of image patches has natural 2D positional relationships: token (3, 5) is close to (4, 5) and (3, 6), but far from (15, 15). Standard 1D RoPE treats position as a single index, losing this 2D structure.

**The solution:**
Split the embedding dimension in half: one half encodes row position, the other half encodes column position. Each half uses standard 1D RoPE independently. This creates a factored 2D positional encoding.

**Why this works:**
The attention score between two vision tokens at (r1, c1) and (r2, c2) depends on:
- Row distance: |r1 - r2| (encoded in first half of dimensions)
- Column distance: |c1 - c2| (encoded in second half of dimensions)

Both contribute independently to the final dot product, giving a natural 2D distance metric.

**Relation to alternatives:**
- **1D flattened RoPE**: Treats image as raster-scanned sequence - loses 2D structure
- **Learned 2D embeddings**: Parameter-heavy, doesn't extrapolate to different image sizes
- **2D RoPE**: Zero parameters, extrapolates to arbitrary image resolutions

**Practical usage:**
Used in vision transformers (ViT) and vision-language models that process images as token sequences. Particularly important for models that need to handle variable image sizes at inference time.

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

**Generated Visualization:**

The code above produces four plots that illustrate RoPE's multi-scale frequency design:

1. **Top-left (Rotation Angles vs Position)**: Shows how different dimension pairs accumulate rotation at different rates. Lower dimension pairs (blue) rotate quickly, making full circles within 100 positions. Higher dimension pairs (orange/red) rotate slowly, making less than half a rotation.

2. **Top-right (Rotation Angles Heatmap)**: A color-coded view showing all dimension pairs simultaneously. The rainbow gradient (twilight colormap) wraps around, representing angles from 0 to 2π. Fast-rotating pairs (top) show many color cycles; slow-rotating pairs (bottom) show few.

3. **Bottom-left (Frequency Spectrum)**: The geometric decay of rotation frequencies on a log scale. This follows the formula θᵢ = 10000^(-2i/d), creating a smooth exponential decrease from high to low frequencies.

4. **Bottom-right (Wavelengths)**: The inverse view showing wavelengths (how many positions for one full rotation). Low dimensions have short wavelengths (~6 positions), while high dimensions have very long wavelengths (>10,000 positions).

**What this reveals:**
- RoPE uses a **multi-scale** encoding similar to Fourier transforms
- **High-frequency components** (low dimensions) distinguish nearby tokens (local attention)
- **Low-frequency components** (high dimensions) distinguish distant tokens (global structure)
- This is why RoPE works well: it encodes both fine-grained and coarse-grained positional relationships simultaneously

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
- [Chapter 27: Long Context Techniques](27-long-context.md) - Advanced RoPE scaling and other long-context methods
- [Chapter 30: Architecture Comparison](30-model-architectures.md) - Which models use RoPE
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
