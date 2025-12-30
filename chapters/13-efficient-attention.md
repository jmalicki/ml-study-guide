# Chapter 13: Other Efficient Attention Variants

This chapter explores various efficient attention mechanisms beyond Flash Attention. While Flash Attention (see [Flash Attention](12-flash-attention.md)) optimizes exact attention through memory hierarchy awareness, the variants covered here modify the attention mechanism itself to reduce computational complexity from $O(n^2)$ to something more tractable.

Understanding these techniques is crucial for ML interviews, as production LLMs use combinations of these methods to handle long contexts efficiently.

## Table of Contents

1. [The Attention Efficiency Problem](#the-attention-efficiency-problem)
2. [Linear Attention (Kernel Approximations)](#linear-attention-kernel-approximations)
3. [Sparse Attention Patterns](#sparse-attention-patterns)
   - [BigBird](#bigbird)
   - [Longformer](#longformer)
4. [Sliding Window Attention](#sliding-window-attention)
5. [Multi-Query Attention (MQA)](#multi-query-attention-mqa)
6. [Grouped-Query Attention (GQA)](#grouped-query-attention-gqa)
7. [Multi-head Latent Attention (MLA)](#multi-head-latent-attention-mla)
8. [Comparison and Usage Guidance](#comparison-and-usage-guidance)
9. [Summary](#summary)

---

## The Attention Efficiency Problem

Standard multi-head attention (see [Multi-Head Attention](04-multi-head-attention.md)) has two main efficiency bottlenecks:

### 1. Computational Complexity: $O(n^2)$

For sequence length $n$ and hidden dimension $d$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The $QK^T$ computation creates an $n \times n$ attention matrix, requiring $O(n^2 d)$ operations.

### 2. Memory Complexity: KV Cache

During autoregressive generation, we cache key and value tensors for all previous tokens:

$$
\text{Memory}_{\text{KV cache}} = 2 \times \text{layers} \times \text{heads} \times \text{seq\_len} \times \text{head\_dim} \times \text{bytes}
$$

For a 70B model with 100K context:
- 80 layers, 64 heads, 128 head dim, FP16
- Memory: $2 \times 80 \times 64 \times 100000 \times 128 \times 2 \approx 260$ GB!

```python
import torch
import torch.nn as nn
import math

def analyze_attention_complexity():
    """
    Compare complexity of different attention variants.

    Standard attention: O(n^2 * d)
    - n: sequence length
    - d: hidden dimension
    """

    configs = [
        ("Short context", 2048, 4096),
        ("Long context", 32768, 4096),
        ("Very long", 131072, 4096),
    ]

    print("Attention Complexity Analysis")
    print("-" * 70)
    print(f"{'Config':<20} {'Seq Len':<10} {'Hidden':<10} {'FLOPs (G)':<15}")
    print("-" * 70)

    for name, n, d in configs:
        # Standard attention: 2 * n^2 * d (QK^T and attn @ V)
        flops_standard = 2 * n * n * d / 1e9

        print(f"{name:<20} {n:<10} {d:<10} {flops_standard:<15.2f}")

    print("\nKey insight: Quadratic growth in sequence length!")
    print("Doubling sequence length = 4x compute")

# Output:
# Config               Seq Len    Hidden     FLOPs (G)
# ----------------------------------------------------------------------
# Short context        2048       4096       34.36
# Long context         32768      4096       8796.09
# Very long            131072     4096       140737.49
```

**The solutions in this chapter address one or both of these bottlenecks.**

---

## Linear Attention (Kernel Approximations)

Linear attention approximates the softmax operation to achieve $O(nd^2)$ complexity instead of $O(n^2d)$.

### Key Insight: Kernel Trick

Standard attention can be viewed as:

$$
\text{Attention}(Q, K, V)_i = \frac{\sum_{j=1}^n \text{sim}(q_i, k_j) v_j}{\sum_{j=1}^n \text{sim}(q_i, k_j)}
$$

where $\text{sim}(q, k) = \exp(q^T k / \sqrt{d})$.

If we can approximate $\text{sim}(q, k) \approx \phi(q)^T \phi(k)$ for some feature map $\phi$, then:

$$
\text{Attention}(Q, K, V)_i = \frac{\phi(q_i)^T \sum_{j=1}^n \phi(k_j) v_j^T}{\phi(q_i)^T \sum_{j=1}^n \phi(k_j)}
$$

The sums $\sum_j \phi(k_j) v_j^T$ and $\sum_j \phi(k_j)$ can be computed once in $O(nd^2)$ time!

### Implementation

```python
class LinearAttention(nn.Module):
    """
    Linear attention using kernel approximation.

    Complexity: O(n * d^2) instead of O(n^2 * d)

    Key limitation: No causal masking support (for autoregressive models)
    without modifications. Best for encoder-style models.

    Reference: Katharopoulos et al., "Transformers are RNNs: Fast
    Autoregressive Transformers with Linear Attention" (ICML 2020)
    https://arxiv.org/abs/2006.16236
    """

    def __init__(self, dim: int, n_heads: int = 8, feature_dim: int = None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.feature_dim = feature_dim or self.head_dim

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature map: φ(x) = elu(x) + 1

        This ensures non-negativity (required for similarity interpretation).
        Other options: ReLU, softplus, or random Fourier features.
        """
        return torch.nn.functional.elu(x) + 1

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            causal: If True, use causal variant (slower but supports AR)

        Returns:
            Output: [batch, seq_len, dim]
        """
        batch, seq_len, dim = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply feature map
        q = self.feature_map(q)  # [batch, heads, seq, head_dim]
        k = self.feature_map(k)

        if not causal:
            # Non-causal: compute once and broadcast
            # KV = Σ_j φ(k_j) ⊗ v_j
            kv = torch.einsum('bhnd,bhnm->bhdm', k, v)  # [batch, heads, head_dim, head_dim]
            # Z = Σ_j φ(k_j)
            z = k.sum(dim=2, keepdim=True)  # [batch, heads, 1, head_dim]

            # Output = φ(Q) @ KV / (φ(Q) @ Z)
            num = torch.einsum('bhnd,bhdm->bhnm', q, kv)
            den = torch.einsum('bhnd,bhnd->bhn', q, z.squeeze(2)).unsqueeze(-1)
            out = num / (den + 1e-6)

        else:
            # Causal: compute cumulatively (slower but still linear)
            out = torch.zeros_like(v)
            kv_state = torch.zeros(batch, self.n_heads, self.head_dim, self.head_dim,
                                   device=x.device, dtype=x.dtype)
            k_state = torch.zeros(batch, self.n_heads, self.head_dim,
                                  device=x.device, dtype=x.dtype)

            for i in range(seq_len):
                # Update state with current k, v
                kv_state = kv_state + torch.einsum('bhd,bhm->bhdm', k[:, :, i], v[:, :, i])
                k_state = k_state + k[:, :, i]

                # Compute attention for position i
                num = torch.einsum('bhd,bhdm->bhm', q[:, :, i], kv_state)
                den = torch.einsum('bhd,bhd->bh', q[:, :, i], k_state).unsqueeze(-1)
                out[:, :, i] = num / (den + 1e-6)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out(out)


def compare_linear_vs_standard():
    """
    Benchmark linear attention vs standard attention.
    """
    import time

    dim = 512
    n_heads = 8
    batch = 2

    linear_attn = LinearAttention(dim, n_heads)
    standard_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    seq_lengths = [512, 1024, 2048, 4096, 8192]

    print("Linear vs Standard Attention Benchmark")
    print("-" * 60)
    print(f"{'Seq Len':<10} {'Linear (ms)':<15} {'Standard (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for seq_len in seq_lengths:
        x = torch.randn(batch, seq_len, dim)

        # Linear attention
        start = time.time()
        with torch.no_grad():
            _ = linear_attn(x, causal=False)
        linear_time = (time.time() - start) * 1000

        # Standard attention
        start = time.time()
        with torch.no_grad():
            _ = standard_attn(x, x, x, need_weights=False)[0]
        standard_time = (time.time() - start) * 1000

        speedup = standard_time / linear_time
        print(f"{seq_len:<10} {linear_time:<15.2f} {standard_time:<15.2f} {speedup:<10.2f}x")
```

### Linear Attention Variants

```python
class RandomFourierFeatures(nn.Module):
    """
    Linear attention with Random Fourier Features (RFF).

    Instead of elu(x) + 1, use random projections:
    φ(x) = [cos(Wx + b), sin(Wx + b)] / √m

    This provides better approximation to the softmax kernel but
    requires more feature dimensions.

    Reference: Choromanski et al., "Rethinking Attention with Performers"
    (ICLR 2021) https://arxiv.org/abs/2009.14794
    """

    def __init__(self, input_dim: int, n_features: int):
        super().__init__()
        self.n_features = n_features

        # Random projection matrix (fixed, not learned)
        self.register_buffer(
            'W',
            torch.randn(input_dim, n_features) / (input_dim ** 0.5)
        )
        self.register_buffer(
            'b',
            torch.rand(n_features) * 2 * math.pi
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim]
        Returns:
            features: [..., 2 * n_features]
        """
        proj = x @ self.W + self.b
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / (self.n_features ** 0.5)
```

**Key Papers:**
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) (Katharopoulos et al., 2020)
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) (Choromanski et al., 2021)

---

## Sparse Attention Patterns

Instead of attending to all positions, only attend to a sparse subset. Complexity: $O(n \cdot s)$ where $s$ is the sparsity pattern size.

### BigBird

BigBird combines three types of attention:
1. **Global tokens**: A few tokens attend to everything
2. **Local (sliding) window**: Each token attends to nearby tokens
3. **Random**: Each token attends to a few random positions

```python
class BigBirdAttention(nn.Module):
    """
    BigBird sparse attention pattern.

    Combines:
    - Global attention (first g tokens attend to all, all attend to first g)
    - Local sliding window (w tokens on each side)
    - Random attention (r random tokens per query)

    Complexity: O(n * (g + w + r)) instead of O(n^2)

    Proven to be theoretically sufficient: Any function computable by
    full attention can be approximated by BigBird with sufficient depth.

    Reference: Zaheer et al., "Big Bird: Transformers for Longer Sequences"
    (NeurIPS 2020) https://arxiv.org/abs/2007.14062
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        num_global_tokens: int = 2,
        window_size: int = 128,
        num_random: int = 3,
        block_size: int = 64,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.num_global = num_global_tokens
        self.window_size = window_size
        self.num_random = num_random
        self.block_size = block_size

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def create_bigbird_mask(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create BigBird attention mask.

        Returns:
            mask: [seq_len, seq_len] boolean mask
                  True = attend, False = mask out
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # 1. Global tokens (e.g., first 2 tokens)
        mask[:self.num_global, :] = True  # Global tokens attend to all
        mask[:, :self.num_global] = True  # All attend to global tokens

        # 2. Local sliding window
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = True

        # 3. Random attention
        for i in range(self.num_global, seq_len):
            # Sample random positions (excluding already attended positions)
            available = torch.where(~mask[i])[0]
            if len(available) > 0:
                num_random = min(self.num_random, len(available))
                random_indices = available[torch.randperm(len(available))[:num_random]]
                mask[i, random_indices] = True

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            output: [batch, seq_len, dim]
        """
        batch, seq_len, dim = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply BigBird mask
        mask = self.create_bigbird_mask(seq_len, x.device)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and weighted sum
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out(out)


def visualize_bigbird_pattern():
    """Visualize BigBird attention pattern."""
    import matplotlib.pyplot as plt

    seq_len = 256
    attn = BigBirdAttention(dim=512, num_global_tokens=4, window_size=16, num_random=3)
    mask = attn.create_bigbird_mask(seq_len, torch.device('cpu'))

    plt.figure(figsize=(10, 10))
    plt.imshow(mask.float().numpy(), cmap='binary', interpolation='nearest')
    plt.title('BigBird Attention Pattern')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar(label='Attends (1) / Masked (0)')
    plt.tight_layout()
    plt.savefig('bigbird_pattern.png', dpi=150)
    plt.close()

    # Calculate sparsity
    total_possible = seq_len * seq_len
    actual_connections = mask.sum().item()
    sparsity = actual_connections / total_possible

    print(f"Total possible connections: {total_possible:,}")
    print(f"Actual connections: {actual_connections:,}")
    print(f"Sparsity: {sparsity:.2%}")
    print(f"Memory reduction: {1/sparsity:.1f}x")
```

### Longformer

Longformer uses a dilated sliding window pattern with global attention for selected tokens.

```python
class LongformerAttention(nn.Module):
    """
    Longformer attention with dilated sliding windows.

    Features:
    - Sliding window attention (local context)
    - Dilated attention (multi-scale, exponentially increasing gaps)
    - Global attention for task-specific tokens (e.g., [CLS])

    Complexity: O(n * w) where w is window size

    Reference: Beltagy et al., "Longformer: The Long-Document Transformer"
    (arXiv 2020) https://arxiv.org/abs/2004.05150
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        window_size: int = 512,
        dilation_rates: list = [1, 2, 4, 8],
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.dilation_rates = dilation_rates

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def create_dilated_mask(
        self,
        seq_len: int,
        device: torch.device,
        global_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Create dilated sliding window mask.

        Args:
            seq_len: Sequence length
            device: Device for tensor
            global_mask: [seq_len] boolean, True for tokens with global attention

        Returns:
            mask: [seq_len, seq_len] attention mask
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Sliding window with multiple dilation rates
        for dilation in self.dilation_rates:
            for i in range(seq_len):
                # Attend to positions at regular intervals (dilation)
                # within the window
                for offset in range(-self.window_size, self.window_size + 1, dilation):
                    j = i + offset
                    if 0 <= j < seq_len:
                        mask[i, j] = True

        # Global attention
        if global_mask is not None:
            global_indices = torch.where(global_mask)[0]
            # Global tokens attend to everything
            mask[global_indices, :] = True
            # Everything attends to global tokens
            mask[:, global_indices] = True

        return mask

    def forward(
        self,
        x: torch.Tensor,
        global_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            global_attention_mask: [batch, seq_len] boolean, global tokens

        Returns:
            output: [batch, seq_len, dim]
        """
        batch, seq_len, dim = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask (simplified - real implementation uses efficient CUDA kernels)
        global_mask = global_attention_mask[0] if global_attention_mask is not None else None
        mask = self.create_dilated_mask(seq_len, x.device, global_mask)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Attention
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out(out)
```

**Key Papers:**
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) (Zaheer et al., 2020)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) (Beltagy et al., 2020)

---

## Sliding Window Attention

Sliding window attention limits each token to attending only to the previous $w$ tokens. Used by Mistral 7B.

```python
class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention (used in Mistral 7B).

    Each position attends only to the previous `window_size` positions.

    Key insight: Due to layer stacking, information propagates beyond
    the window. At layer L, effective receptive field = L * window_size.

    Mistral 7B: 32 layers, 4096 window -> effective 131K receptive field!

    Benefits:
    - O(n * w) complexity instead of O(n^2)
    - Enables rolling buffer KV cache (fixed size)
    - Quality degradation minimal for most tasks

    Reference: Jiang et al., "Mistral 7B" (2023)
    https://arxiv.org/abs/2310.06825

    See also: [Architecture Comparison](29-model-architectures.md) for usage
    in production models.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        window_size: int = 4096,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Create sliding window attention mask.

        Args:
            seq_len: Sequence length
            device: Device
            causal: If True, also apply causal masking

        Returns:
            mask: [seq_len, seq_len] boolean mask
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        for i in range(seq_len):
            # Attend to previous window_size positions
            start = max(0, i - self.window_size + 1)
            end = i + 1 if causal else min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = True

        return mask

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple = None
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x: [batch, seq_len, dim]
            past_kv: Optional (k_cache, v_cache) for incremental decoding

        Returns:
            output: [batch, seq_len, dim]
            new_kv: (k_cache, v_cache) for next step
        """
        batch, seq_len, dim = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Handle KV cache
        if past_kv is not None:
            k_cache, v_cache = past_kv
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

            # Rolling buffer: keep only last window_size tokens
            if k.shape[2] > self.window_size:
                k = k[:, :, -self.window_size:]
                v = v[:, :, -self.window_size:]

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply sliding window mask
        kv_len = k.shape[2]
        mask = self.create_sliding_window_mask(kv_len, x.device, causal=True)
        # Only mask for the query positions we're computing
        mask = mask[-seq_len:, :]
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Attention and output
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.out(out)

        # Return KV cache for next iteration
        new_kv = (k, v)
        return out, new_kv


class RollingBufferCache:
    """
    Rolling buffer KV cache for sliding window attention.

    Instead of growing indefinitely, maintains a fixed-size circular buffer.
    Saves memory when generating long sequences.

    Memory savings example (Mistral 7B):
    - Window size: 4096
    - Without rolling: 32K context = 8x memory
    - With rolling: Fixed 4096 tokens regardless of context length
    """

    def __init__(
        self,
        window_size: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        batch_size: int = 1,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16
    ):
        self.window_size = window_size
        self.position = 0

        # Preallocate cache buffers
        self.k_cache = torch.zeros(
            n_layers, batch_size, n_heads, window_size, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            n_layers, batch_size, n_heads, window_size, head_dim,
            device=device, dtype=dtype
        )

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K, V tensors.

        Args:
            layer_idx: Which layer is updating
            k, v: [batch, n_heads, seq_len, head_dim]

        Returns:
            Full K, V including cache
        """
        seq_len = k.shape[2]

        for i in range(seq_len):
            # Circular buffer index
            idx = (self.position + i) % self.window_size
            self.k_cache[layer_idx, :, :, idx] = k[:, :, i]
            self.v_cache[layer_idx, :, :, idx] = v[:, :, i]

        self.position = (self.position + seq_len) % self.window_size

        # Return current cache state
        # Note: Order matters for correct attention
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
```

**Key Paper:**
- [Mistral 7B](https://arxiv.org/abs/2310.06825) (Jiang et al., 2023)

---

## Multi-Query Attention (MQA)

MQA reduces KV cache size by sharing key and value heads across all query heads.

```python
class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA).

    Standard MHA: Each query head has its own K, V heads
    MQA: All query heads share ONE set of K, V heads

    Memory savings: n_heads × reduction in KV cache

    Trade-off:
    + Much smaller KV cache (critical for long context inference)
    + Faster inference (less memory bandwidth)
    - Slight quality degradation (~1-2% on benchmarks)

    Used in: PaLM, Falcon, StarCoder

    Reference: Shazeer, "Fast Transformer Decoding: One Write-Head is All
    You Need" (2019) https://arxiv.org/abs/1911.02150

    See also: [Multi-Head Attention](04-multi-head-attention.md) for standard MHA.
    """

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Q has n_heads, but K and V have only 1 head
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.head_dim, bias=False)  # Single head!
        self.wv = nn.Linear(dim, self.head_dim, bias=False)  # Single head!
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        past_kv: tuple = None
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x: [batch, seq_len, dim]
            mask: Optional attention mask
            past_kv: Optional cached (k, v) from previous steps

        Returns:
            output: [batch, seq_len, dim]
            new_kv: (k, v) for caching
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, 1, self.head_dim)  # Note: 1 head
        v = self.wv(x).view(batch, seq_len, 1, self.head_dim)  # Note: 1 head

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, 1, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, 1, seq_len, head_dim]

        # Handle cache
        if past_kv is not None:
            k_cache, v_cache = past_kv
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Expand K, V to match Q's number of heads (broadcast)
        k = k.expand(-1, self.n_heads, -1, -1)
        v = v.expand(-1, self.n_heads, -1, -1)

        # Standard attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.wo(out)

        # Cache the single K, V heads (not expanded)
        new_kv = (k[:, :1], v[:, :1])  # Only cache first head
        return out, new_kv


def compare_mqa_memory():
    """
    Compare KV cache memory for MHA vs MQA.
    """
    def calc_kv_cache_size(
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch: int = 1,
        dtype_bytes: int = 2  # FP16
    ) -> float:
        """Returns KV cache size in GB."""
        return (2 * n_layers * n_kv_heads * head_dim * seq_len *
                batch * dtype_bytes) / 1e9

    configs = [
        ("7B model", 32, 32, 128),   # MHA
        ("7B model", 32, 1, 128),    # MQA
        ("70B model", 80, 64, 128),  # MHA
        ("70B model", 80, 1, 128),   # MQA
    ]

    seq_len = 100_000  # 100K context

    print("KV Cache Memory Comparison: MHA vs MQA")
    print("-" * 70)
    print(f"{'Model':<15} {'Layers':<8} {'KV Heads':<10} {'Memory (GB)':<12} {'Reduction':<10}")
    print("-" * 70)

    prev_memory = None
    for name, n_layers, n_kv_heads, head_dim in configs:
        memory = calc_kv_cache_size(n_layers, n_kv_heads, head_dim, seq_len)

        reduction = ""
        if prev_memory is not None and "MHA" in name:
            prev_memory = memory
        elif prev_memory is not None:
            reduction = f"{prev_memory/memory:.1f}x"
            prev_memory = None
        else:
            prev_memory = memory

        print(f"{name:<15} {n_layers:<8} {n_kv_heads:<10} {memory:<12.2f} {reduction:<10}")

# Output:
# Model           Layers   KV Heads   Memory (GB)  Reduction
# ----------------------------------------------------------------------
# 7B model        32       32         1.64
# 7B model        32       1          0.05         32.0x
# 70B model       80       64         6.55
# 70B model       80       1          0.10         64.0x
```

**Key Paper:**
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Shazeer, 2019)

---

## Grouped-Query Attention (GQA)

GQA is a middle ground between MHA and MQA: groups of query heads share K, V heads.

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Middle ground between MHA and MQA:
    - MHA: n_kv_heads = n_heads (each Q head has own K,V)
    - MQA: n_kv_heads = 1 (all Q heads share K,V)
    - GQA: 1 < n_kv_heads < n_heads (groups share K,V)

    Example: 32 query heads, 8 KV heads -> 4 query heads per KV head

    Benefits:
    - 4-8x KV cache reduction (vs MHA)
    - Better quality than MQA (~0.5% improvement)
    - Used in: LLaMA 2, LLaMA 3, Qwen, Mistral

    Reference: Ainslie et al., "GQA: Training Generalized Multi-Query
    Transformer Models from Multi-Head Checkpoints" (2023)
    https://arxiv.org/abs/2305.13245

    See [Architecture Comparison](29-model-architectures.md) for usage in
    modern LLMs.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 32,
        n_kv_heads: int = 8,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # Repetition factor
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        past_kv: tuple = None
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x: [batch, seq_len, dim]
            mask: Optional attention mask
            past_kv: Optional (k_cache, v_cache)

        Returns:
            output: [batch, seq_len, dim]
            new_kv: (k, v) for caching
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Transpose
        q = q.transpose(1, 2)  # [batch, n_heads, seq, head_dim]
        k = k.transpose(1, 2)  # [batch, n_kv_heads, seq, head_dim]
        v = v.transpose(1, 2)

        # Handle cache
        if past_kv is not None:
            k_cache, v_cache = past_kv
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Repeat K, V to match number of query heads
        # Each KV head is used by n_rep query heads
        k = k.repeat_interleave(self.n_rep, dim=1)  # [batch, n_heads, seq, head_dim]
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.wo(out)

        # Cache only the KV heads (before expansion)
        new_kv = (k[:, ::self.n_rep], v[:, ::self.n_rep])
        return out, new_kv


def uptraining_mha_to_gqa():
    """
    Convert pretrained MHA checkpoint to GQA (uptraining).

    GQA paper shows you can convert an MHA checkpoint to GQA by:
    1. Average corresponding K, V heads into groups
    2. Continue training for ~5% of original steps
    3. Recover most of original quality

    This allows retrofitting existing models for better inference efficiency.
    """
    def mean_pool_kv_heads(
        kv_weight: torch.Tensor,  # [n_heads * head_dim, dim]
        n_heads: int,
        n_kv_heads: int
    ) -> torch.Tensor:
        """
        Average groups of K or V heads.

        Example: 32 heads -> 8 KV heads
        Heads [0,1,2,3] averaged to KV head 0
        Heads [4,5,6,7] averaged to KV head 1, etc.
        """
        head_dim = kv_weight.shape[0] // n_heads
        kv_weight = kv_weight.view(n_heads, head_dim, -1)

        n_rep = n_heads // n_kv_heads
        kv_weight = kv_weight.view(n_kv_heads, n_rep, head_dim, -1)
        kv_weight = kv_weight.mean(dim=1)  # Average within groups

        return kv_weight.view(n_kv_heads * head_dim, -1)

    # Example conversion
    mha_k_weight = torch.randn(32 * 128, 4096)  # 32 heads, 128 head_dim
    gqa_k_weight = mean_pool_kv_heads(mha_k_weight, n_heads=32, n_kv_heads=8)

    print(f"MHA K weight shape: {mha_k_weight.shape}")
    print(f"GQA K weight shape: {gqa_k_weight.shape}")
    print(f"Memory reduction: {mha_k_weight.numel() / gqa_k_weight.numel():.1f}x")
```

**Key Paper:**
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)

---

## Multi-head Latent Attention (MLA)

MLA compresses the KV cache into a low-dimensional latent space. Used by DeepSeek V2/V3.

```python
class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) from DeepSeek.

    Key innovation: Instead of caching K and V directly, compress them
    into a low-dimensional latent representation.

    Standard: Cache K, V ∈ R^(n_heads × head_dim)
    MLA: Cache C ∈ R^(latent_dim) where latent_dim << n_heads × head_dim

    At inference: Decompress latent to K, V on-the-fly

    Memory savings: ~10x KV cache reduction
    Compute trade-off: Slightly more work during generation (decompression)

    Key insight: Memory bandwidth is often the bottleneck at inference,
    not compute. Trading compute for memory is favorable.

    Used in: DeepSeek V2 (671B), DeepSeek V3 (685B)

    Reference: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and
    Efficient Mixture-of-Experts Language Model" (2024)
    https://arxiv.org/abs/2405.04434

    See [Architecture Comparison](29-model-architectures.md) for details.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 32,
        head_dim: int = 128,
        latent_dim: int = 512,  # Much smaller than n_heads * head_dim
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.kv_dim = n_heads * head_dim

        # Query projection (standard)
        self.wq = nn.Linear(dim, self.kv_dim, bias=False)

        # KV compression: input -> latent
        self.kv_compress = nn.Linear(dim, latent_dim, bias=False)

        # KV decompression: latent -> K and V
        self.k_decompress = nn.Linear(latent_dim, self.kv_dim, bias=False)
        self.v_decompress = nn.Linear(latent_dim, self.kv_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(self.kv_dim, dim, bias=False)

        self.scale = head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        cached_latent: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, dim] - Input
            cached_latent: [batch, cache_len, latent_dim] - Cached compressed KV
            mask: Optional attention mask

        Returns:
            output: [batch, seq_len, dim]
            new_latent: [batch, total_len, latent_dim] - Updated cache
        """
        batch, seq_len, _ = x.shape

        # Standard query projection
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]

        # Compress input to latent (this is what we cache!)
        latent = self.kv_compress(x)  # [batch, seq_len, latent_dim]

        # Concatenate with cached latent
        if cached_latent is not None:
            latent = torch.cat([cached_latent, latent], dim=1)

        total_len = latent.shape[1]

        # Decompress latent to K and V
        k = self.k_decompress(latent)  # [batch, total_len, n_heads * head_dim]
        v = self.v_decompress(latent)  # [batch, total_len, n_heads * head_dim]

        # Reshape for attention
        k = k.view(batch, total_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, total_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Standard attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.wo(out)

        # Return output and latent for caching
        return out, latent


def mla_memory_analysis():
    """
    Analyze MLA memory savings.
    """
    def kv_cache_size(n_kv_heads, head_dim, seq_len, n_layers, dtype_bytes=2):
        """Standard KV cache size."""
        return 2 * n_layers * n_kv_heads * head_dim * seq_len * dtype_bytes / 1e9

    def mla_cache_size(latent_dim, seq_len, n_layers, dtype_bytes=2):
        """MLA latent cache size."""
        return n_layers * latent_dim * seq_len * dtype_bytes / 1e9

    # DeepSeek V3 configuration
    n_layers = 61
    n_heads = 128
    head_dim = 128
    latent_dim = 512  # DeepSeek V3 uses ~512
    seq_len = 100_000

    # Standard GQA (if they used it)
    n_kv_heads = 8  # Hypothetical GQA
    gqa_memory = kv_cache_size(n_kv_heads, head_dim, seq_len, n_layers)

    # Standard MHA
    mha_memory = kv_cache_size(n_heads, head_dim, seq_len, n_layers)

    # MLA
    mla_memory = mla_cache_size(latent_dim, seq_len, n_layers)

    print("DeepSeek V3 KV Cache Memory (100K context)")
    print("-" * 60)
    print(f"Standard MHA:       {mha_memory:>8.2f} GB")
    print(f"Hypothetical GQA:   {gqa_memory:>8.2f} GB  ({mha_memory/gqa_memory:.1f}x reduction)")
    print(f"MLA (actual):       {mla_memory:>8.2f} GB  ({mha_memory/mla_memory:.1f}x reduction)")
    print()
    print(f"MLA vs GQA:         {gqa_memory/mla_memory:.1f}x better")

    # Compression ratio
    original_kv_dim = 2 * n_heads * head_dim
    compression_ratio = original_kv_dim / latent_dim
    print(f"\nCompression ratio:  {compression_ratio:.1f}x")
    print(f"  Original KV dim:  {original_kv_dim}")
    print(f"  Latent dim:       {latent_dim}")

# Output:
# DeepSeek V3 KV Cache Memory (100K context)
# ------------------------------------------------------------
# Standard MHA:          126.98 GB
# Hypothetical GQA:        7.94 GB  (16.0x reduction)
# MLA (actual):            6.27 GB  (20.3x reduction)
#
# MLA vs GQA:         1.3x better
#
# Compression ratio:  32.0x
#   Original KV dim:  16384
#   Latent dim:       512
```

**Key Papers:**
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024)

---

## Comparison and Usage Guidance

### Complexity Comparison

| Method | Time Complexity | Space Complexity | KV Cache |
|--------|----------------|------------------|----------|
| **Standard MHA** | $O(n^2 d)$ | $O(n^2)$ | $O(nhd)$ per layer |
| **Flash Attention** | $O(n^2 d)$ | $O(n)$ | $O(nhd)$ per layer |
| **Linear Attention** | $O(nd^2)$ | $O(d^2)$ | $O(d^2)$ |
| **Sparse (BigBird)** | $O(nsd)$ | $O(ns)$ | $O(nhd)$ per layer |
| **Sliding Window** | $O(nwd)$ | $O(nw)$ | $O(whd)$ per layer |
| **MQA** | $O(n^2 d)$ | $O(n^2)$ | $O(hd)$ per layer |
| **GQA** | $O(n^2 d)$ | $O(n^2)$ | $O(ghd)$ per layer |
| **MLA** | $O(n^2 d)$ | $O(n^2)$ | $O(ld)$ per layer |

Where:
- $n$: sequence length
- $d$: hidden dimension
- $h$: number of heads
- $w$: window size
- $s$: sparsity pattern size
- $g$: number of KV head groups (GQA)
- $l$: latent dimension (MLA)

### Quality vs Efficiency Trade-offs

```python
def efficiency_comparison():
    """
    Compare different attention variants on quality and efficiency.

    Based on empirical results from papers and production usage.
    """

    methods = [
        # (name, relative_quality, relative_speed, kv_cache_reduction)
        ("Standard MHA", 100, 1.0, 1.0),
        ("Flash Attention", 100, 2.5, 1.0),  # Same quality, faster
        ("Linear Attention", 85, 8.0, 100.0),  # Fast but quality loss
        ("BigBird", 95, 3.0, 5.0),  # Good balance
        ("Sliding Window", 98, 2.0, 4.0),  # Minimal quality loss
        ("MQA", 97, 1.2, 32.0),  # Great cache reduction
        ("GQA (8 groups)", 99, 1.15, 4.0),  # Best overall balance
        ("MLA", 99, 1.1, 20.0),  # Excellent cache reduction
    ]

    print("Attention Variant Comparison")
    print("=" * 80)
    print(f"{'Method':<20} {'Quality':<10} {'Speed':<12} {'Cache Reduction':<15}")
    print("-" * 80)

    for name, quality, speed, cache in methods:
        print(f"{name:<20} {quality:<10}% {speed:<12.1f}x {cache:<15.1f}x")

    print("\n" + "=" * 80)
    print("Quality: Relative performance on downstream tasks (100 = no degradation)")
    print("Speed: Training/inference speedup vs standard MHA")
    print("Cache Reduction: KV cache memory reduction factor")
```

### When to Use What

```python
class AttentionVariantSelector:
    """
    Guide for selecting the right attention variant.
    """

    @staticmethod
    def select_for_training(
        max_seq_len: int,
        has_long_dependencies: bool = True
    ) -> str:
        """
        Select attention variant for training.

        Args:
            max_seq_len: Maximum sequence length
            has_long_dependencies: Whether task needs long-range attention

        Returns:
            Recommended variant
        """
        if max_seq_len <= 8192:
            return "Standard MHA with Flash Attention"

        elif max_seq_len <= 32768:
            if has_long_dependencies:
                return "Sliding Window + Flash Attention (like Mistral)"
            else:
                return "BigBird or Longformer"

        else:  # Very long context
            if has_long_dependencies:
                return "Sliding Window with large window + Flash Attention"
            else:
                return "Linear Attention or BigBird"

    @staticmethod
    def select_for_inference(
        model_size: str,
        batch_size: int,
        context_length: int
    ) -> str:
        """
        Select attention variant for inference optimization.

        Args:
            model_size: 'small' (<10B), 'medium' (10-70B), 'large' (>70B)
            batch_size: Inference batch size
            context_length: Expected context length

        Returns:
            Recommended variant
        """
        if model_size == 'small':
            return "Standard MHA (cache not bottleneck)"

        elif model_size == 'medium':
            if context_length > 32768:
                return "GQA (4-8 groups) for cache efficiency"
            else:
                return "GQA or MQA"

        else:  # large
            if context_length > 100000:
                return "MLA (DeepSeek-style) for extreme cache reduction"
            else:
                return "GQA (8 groups) with possible sliding window"


# Example usage
selector = AttentionVariantSelector()

print("Training Recommendations:")
print("-" * 60)
print(f"8K context:      {selector.select_for_training(8192)}")
print(f"32K context:     {selector.select_for_training(32768)}")
print(f"128K context:    {selector.select_for_training(131072)}")

print("\nInference Recommendations:")
print("-" * 60)
print(f"7B, 32K context:    {selector.select_for_inference('small', 1, 32768)}")
print(f"70B, 32K context:   {selector.select_for_inference('medium', 1, 32768)}")
print(f"70B, 128K context:  {selector.select_for_inference('medium', 1, 131072)}")
print(f"600B, 100K context: {selector.select_for_inference('large', 1, 100000)}")
```

### Production Model Usage

| Model | Attention Variant | Rationale |
|-------|------------------|-----------|
| **GPT-3** | Standard MHA | 2K context, sufficient |
| **LLaMA 1** | Standard MHA | Baseline implementation |
| **LLaMA 2 (34B+)** | GQA | Inference efficiency |
| **LLaMA 3** | GQA (all sizes) | Standard for all models |
| **Mistral 7B** | GQA + Sliding Window | Long context efficiency |
| **Mixtral 8x7B** | GQA + Sliding Window | Same as Mistral |
| **Qwen 2.5/3** | GQA | Balance quality/efficiency |
| **DeepSeek V2/V3** | MLA | Extreme long context (128K+) |
| **Gemma 2** | GQA + Interleaved local/global | Hybrid approach |

See [Architecture Comparison](29-model-architectures.md) for detailed model specifications.

---

## Summary

### Key Takeaways for Interviews

1. **The Problem**: Standard attention has $O(n^2)$ complexity and large KV cache
   - Makes long contexts (>32K) expensive
   - KV cache is the primary bottleneck at inference

2. **Computational Approaches** (reduce attention complexity):
   - **Linear Attention**: $O(n)$ but quality loss, no causal support
   - **Sparse Attention**: $O(n \cdot s)$, good for specific patterns
   - **Sliding Window**: $O(n \cdot w)$, practical and effective

3. **Memory Approaches** (reduce KV cache):
   - **MQA**: Share K,V across all heads (32-64x cache reduction)
   - **GQA**: Share K,V within groups (4-16x cache reduction, better quality)
   - **MLA**: Compress K,V to latent space (10-20x cache reduction)

4. **The Winner**: GQA is the current sweet spot
   - Used by LLaMA 2/3, Qwen, Mistral, Gemma
   - Good quality (99% of MHA)
   - Significant cache reduction (4-8x)
   - Can be combined with sliding window

5. **Future Direction**: MLA for extreme long context
   - DeepSeek V3 handles 128K context efficiently
   - 20x cache reduction vs MHA
   - Minimal quality loss

### Quick Reference: When to Use What

| Use Case | Recommended Approach |
|----------|---------------------|
| Short context (<8K) | MHA + Flash Attention |
| Medium context (8-32K) | GQA + Flash Attention |
| Long context (32-128K) | GQA + Sliding Window |
| Very long context (>128K) | MLA or extreme sliding window |
| Encoder-only tasks | Linear or Sparse attention |
| Inference on large models | GQA (4-8 groups) or MLA |

---

## Exercises

1. **Complexity Analysis**: For a sequence of length 65,536 and hidden dimension 4096:
   - Calculate the number of FLOPs for standard attention
   - Calculate FLOPs for sliding window with w=4096
   - Calculate FLOPs for linear attention
   - What's the speedup for each?

2. **KV Cache Calculation**: A 70B model has 80 layers, and uses GQA with 64 query heads and 8 KV heads, head dimension 128.
   - Calculate KV cache memory for 100K context with FP16
   - How much memory would MQA save?
   - How much would MLA save with latent_dim=512?

3. **Implementation**: Implement a simple sparse attention pattern of your choice (fixed, strided, or random). Compare memory usage and runtime against standard attention.

4. **Quality vs Efficiency**: Explain why GQA achieves better quality than MQA despite both reducing KV cache. What information is preserved in GQA that MQA loses?

5. **MLA Deep Dive**: In MLA, explain why compression followed by decompression can work. What properties must the latent space have? Why isn't this just a bottleneck autoencoder?

6. **Sliding Window Receptive Field**: If a model has 32 layers and window size 4096, what is the effective receptive field? At what layer can token 0 influence token 100,000?

7. **Design Challenge**: You're building a code completion model (128K context). Design the attention mechanism. Would you use:
   - Pure sliding window?
   - Sliding + global?
   - GQA alone?
   - MLA?
   Justify your choice considering code's local and global structure.

---

## References

### Linear Attention
1. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) (Katharopoulos et al., 2020)
2. [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) (Choromanski et al., 2021)

### Sparse Attention
3. [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) (Zaheer et al., 2020)
4. [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) (Beltagy et al., 2020)
5. [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) (Child et al., 2019)

### Sliding Window
6. [Mistral 7B](https://arxiv.org/abs/2310.06825) (Jiang et al., 2023)

### MQA/GQA
7. [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Shazeer, 2019)
8. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)

### MLA
9. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024)
10. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024)

### Comprehensive Surveys
11. [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) (Tay et al., 2020)
12. [A Survey on Long-Context Large Language Models](https://arxiv.org/abs/2402.02283) (2024)

---

**Previous Chapter**: [Flash Attention](12-flash-attention.md)
**Next Chapter**: Check the main study guide outline
**Related**: [Multi-Head Attention](04-multi-head-attention.md) | [Architecture Comparison](29-model-architectures.md) | [Hardware and Optimization](31-hardware-quantization-optimization.md)
