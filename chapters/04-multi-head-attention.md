# Chapter 4: Multi-Head Attention

Multi-Head Attention (MHA) is a core component of the Transformer architecture, enabling models to learn different representation subspaces and attend to information from different positions simultaneously. This chapter builds on the foundation laid in [Basic Attention](03-basic-attention.md) and introduces the multi-head mechanism that powers modern LLMs.

## Table of Contents

1. [Why Multiple Heads?](#why-multiple-heads)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Implementation from Scratch](#implementation-from-scratch)
4. [Concatenation and Projection](#concatenation-and-projection)
5. [Multi-Query Attention (MQA)](#multi-query-attention-mqa)
6. [Grouped-Query Attention (GQA)](#grouped-query-attention-gqa)
7. [Comparison: MHA vs MQA vs GQA](#comparison-mha-vs-mqa-vs-gqa)
8. [Practical Considerations](#practical-considerations)
9. [Exercises](#exercises)
10. [References](#references)

---

## Why Multiple Heads?

The intuition behind multi-head attention is that different attention heads can learn to focus on different aspects of the input:

- **Head 1**: Might learn syntactic relationships (e.g., subject-verb agreement)
- **Head 2**: Might learn semantic relationships (e.g., co-reference resolution)
- **Head 3**: Might learn positional patterns (e.g., attending to adjacent words)
- **Head 4**: Might learn long-range dependencies (e.g., matching opening and closing brackets)

By having multiple heads operating in parallel, the model can:

1. **Attend to different positions simultaneously**: One head can look at nearby words while another examines distant context
2. **Learn different representation subspaces**: Each head operates in its own subspace, allowing richer feature extraction
3. **Increase model capacity**: More parameters to capture complex patterns
4. **Improve robustness**: Multiple perspectives reduce the risk of missing important information

### Empirical Evidence

Research has shown that attention heads do specialize:

- Heads in lower layers tend to focus on positional and syntactic patterns
- Heads in middle layers capture semantic relationships
- Heads in upper layers perform task-specific reasoning
- Some heads are more important than others (can be pruned without significant loss)

---

## Mathematical Formulation

### Single-Head Attention Recap

From [Basic Attention](03-basic-attention.md), recall that scaled dot-product attention is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:
- $Q \in \mathbb{R}^{n \times d_k}$ are queries
- $K \in \mathbb{R}^{m \times d_k}$ are keys
- $V \in \mathbb{R}^{m \times d_v}$ are values
- $d_k$ is the key/query dimension
- $n$ is the target sequence length, $m$ is the source sequence length

### Multi-Head Attention

Multi-head attention runs $h$ attention operations in parallel, each with its own learned projection matrices:

$$
\begin{align}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{align}
$$

The projection matrices are:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$ projects queries for head $i$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ projects keys for head $i$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ projects values for head $i$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ projects the concatenated output

### Dimension Constraints

Typically:
- $d_k = d_v = d_{\text{model}} / h$ (the head dimension)
- With $h$ heads, the total dimension is $h \times (d_{\text{model}} / h) = d_{\text{model}}$
- This keeps the computational cost similar to single-head attention with full dimension

**Example**: For GPT-2 with $d_{\text{model}} = 768$ and $h = 12$:
- Each head has dimension $d_k = d_v = 768 / 12 = 64$
- Total parameters: roughly the same as single-head attention

---

## Implementation from Scratch

Let's implement multi-head attention in PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to put heads next to batch dimension.

        Args:
            x: tensor of shape (batch_size, seq_len, d_model)

        Returns:
            tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            query: tensor of shape (batch_size, seq_len_q, d_model)
            key: tensor of shape (batch_size, seq_len_k, d_model)
            value: tensor of shape (batch_size, seq_len_v, d_model)
            mask: optional tensor for masking attention weights
                  shape (batch_size, 1, 1, seq_len_k) or (batch_size, 1, seq_len_q, seq_len_k)

        Returns:
            output: tensor of shape (batch_size, seq_len_q, d_model)
            attention_weights: tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.W_v(value)  # (batch_size, seq_len_v, d_model)

        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len_v, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)

        # Apply mask if provided (for causal attention or padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # context shape: (batch_size, num_heads, seq_len_q, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        # shape: (batch_size, seq_len_q, num_heads, d_k)

        context = context.view(batch_size, -1, self.d_model)
        # shape: (batch_size, seq_len_q, d_model)

        # Final linear projection
        output = self.W_o(context)

        return output, attention_weights


# Example usage
def test_multi_head_attention():
    """Test the multi-head attention implementation."""
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)

    # Forward pass (self-attention)
    output, attention_weights = mha(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Verify shapes
    assert output.shape == x.shape
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    print("\nMulti-head attention test passed!")

    # Create a causal mask for autoregressive models
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
    output_masked, attn_masked = mha(x, x, x, mask=causal_mask)

    print(f"\nWith causal mask:")
    print(f"Output shape: {output_masked.shape}")
    print(f"Attention weights shape: {attn_masked.shape}")

    # Visualize attention pattern for the first head
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(attention_weights[0, 0].detach().numpy(), cmap='viridis')
    plt.title('Attention Pattern (Head 0, No Mask)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(attn_masked[0, 0].detach().numpy(), cmap='viridis')
    plt.title('Attention Pattern (Head 0, Causal Mask)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('multihead_attention_patterns.png', dpi=150, bbox_inches='tight')
    print("\nAttention patterns saved to 'multihead_attention_patterns.png'")

if __name__ == "__main__":
    test_multi_head_attention()
```

### Key Implementation Details

1. **Projection**: We use a single linear layer for each of Q, K, V that projects to `d_model`, then split into heads
2. **Split Heads**: Reshape from `(batch, seq_len, d_model)` to `(batch, num_heads, seq_len, d_k)`
3. **Parallel Computation**: All heads compute attention simultaneously using batch matrix multiplication
4. **Concatenation**: After attention, transpose and reshape to concatenate head outputs
5. **Output Projection**: A final linear layer mixes information across heads

---

## Concatenation and Projection

### Why Concatenate?

After computing attention for each head independently, we concatenate the outputs:

```python
# Each head output has shape (batch_size, seq_len_q, d_k)
# After concatenating h heads:
concatenated = torch.cat([head_1, head_2, ..., head_h], dim=-1)
# Shape: (batch_size, seq_len_q, h * d_k) = (batch_size, seq_len_q, d_model)
```

### Why Final Projection?

The output projection $W^O$ serves several purposes:

1. **Mixing head information**: Allows heads to interact and share information
2. **Dimensionality control**: Ensures output dimension matches input
3. **Additional capacity**: Adds learnable parameters for the final transformation
4. **Residual connection**: Output can be added directly to the input (see [The Transformer Block](09-transformer-block.md))

### Alternative: Separate Projection per Head

Some implementations use separate projections for each head:

```python
# Alternative approach (less common)
class MultiHeadAttentionAlt(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(d_model, d_model // num_heads)
            for _ in range(num_heads)
        ])
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Compute each head separately
        head_outputs = [head(query, key, value, mask) for head in self.heads]

        # Concatenate
        concatenated = torch.cat(head_outputs, dim=-1)

        # Project
        output = self.W_o(concatenated)
        return output
```

This is less efficient due to the loop, but conceptually equivalent.

---

## Multi-Query Attention (MQA)

Multi-Query Attention (MQA) is a variant that reduces memory usage and inference latency by sharing keys and values across all attention heads while keeping separate queries for each head.

### Motivation

In autoregressive inference (like in GPT models), the KV cache can become a memory bottleneck:

- **Standard MHA**: Each head stores its own K and V for all previous tokens
- **Memory per layer**: $2 \times h \times \text{seq\_len} \times d_k \times \text{batch\_size}$
- For long sequences and many heads, this becomes prohibitive

### Mathematical Formulation

$$
\begin{align}
\text{MQA}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW^K, VW^V)
\end{align}
$$

Key difference: All heads share the same $W^K$ and $W^V$ (no subscript $i$).

### Implementation

```python
class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) - shares keys and values across all heads.

    Used in models like PaLM, Falcon, and GPT-J for faster inference.

    Args:
        d_model: Dimension of the model
        num_heads: Number of query heads
        dropout: Dropout probability
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Separate query projection for each head
        self.W_q = nn.Linear(d_model, d_model)

        # Shared key and value projections (only d_k dimensions, not d_model)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project queries (multi-head)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len_q, d_k)

        # Project keys and values (shared across heads)
        K = self.W_k(key).unsqueeze(1)  # (batch_size, 1, seq_len_k, d_k)
        V = self.W_v(value).unsqueeze(1)  # (batch_size, 1, seq_len_v, d_k)

        # K and V will broadcast across the num_heads dimension

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        context = torch.matmul(attention_weights, V)
        # (batch_size, num_heads, seq_len_q, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        # Output projection
        output = self.W_o(context)

        return output, attention_weights


def compare_mha_mqa_memory():
    """Compare memory usage of MHA vs MQA."""
    d_model = 4096
    num_heads = 32
    seq_len = 2048
    batch_size = 1

    # MHA parameters for K and V projections
    mha_kv_params = 2 * (d_model * d_model)

    # MQA parameters for K and V projections
    d_k = d_model // num_heads
    mqa_kv_params = 2 * (d_model * d_k)

    print("Parameter Comparison:")
    print(f"MHA K,V params: {mha_kv_params:,}")
    print(f"MQA K,V params: {mqa_kv_params:,}")
    print(f"Reduction: {mha_kv_params / mqa_kv_params:.1f}x")

    # KV cache size
    mha_cache = 2 * batch_size * num_heads * seq_len * d_k * 2  # 2 bytes for fp16
    mqa_cache = 2 * batch_size * 1 * seq_len * d_k * 2

    print(f"\nKV Cache Size (FP16):")
    print(f"MHA cache: {mha_cache / 1e6:.1f} MB")
    print(f"MQA cache: {mqa_cache / 1e6:.1f} MB")
    print(f"Reduction: {mha_cache / mqa_cache:.1f}x")

if __name__ == "__main__":
    compare_mha_mqa_memory()
```

### MQA Trade-offs

**Advantages:**
- Reduced KV cache size by factor of $h$ (number of heads)
- Faster inference, especially for long sequences
- Lower memory bandwidth requirements

**Disadvantages:**
- Slightly lower quality (empirically ~1% worse on some benchmarks)
- Less expressive: heads must share the same key/value representations
- Training time benefits are minimal (KV cache is an inference concern)

**Used in**: PaLM, Falcon, GPT-J, StarCoder

---

## Grouped-Query Attention (GQA)

Grouped-Query Attention (GQA) is a middle ground between MHA and MQA, grouping queries into sets that share keys and values.

### Motivation

GQA aims to balance the quality of MHA with the efficiency of MQA:

- **MHA**: $h$ separate K, V projections (highest quality, most memory)
- **MQA**: 1 shared K, V projection (lowest memory, slight quality loss)
- **GQA**: $g$ groups of K, V projections where $1 < g < h$ (balanced trade-off)

### Mathematical Formulation

With $h$ total heads divided into $g$ groups of size $h/g$:

$$
\begin{align}
\text{GQA}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW_{j(i)}^K, VW_{j(i)}^V)
\end{align}
$$

Here $j(i) = \lfloor i / (h/g) \rfloor$ maps head $i$ to its group.

### Implementation

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) - middle ground between MHA and MQA.

    Used in LLaMA 2, LLaMA 3, Mistral, and other modern LLMs.

    Args:
        d_model: Dimension of the model
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (must divide num_heads evenly)
        dropout: Dropout probability
    """

    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        # Query projections for all heads
        self.W_q = nn.Linear(d_model, d_model)

        # Key and value projections for KV heads only
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Project and split queries (all heads)
        Q = self.W_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len_q, d_k)

        # Project and split keys and values (KV heads only)
        K = self.W_k(key).view(batch_size, seq_len_k, self.num_kv_heads, self.d_k)
        K = K.transpose(1, 2)  # (batch_size, num_kv_heads, seq_len_k, d_k)

        V = self.W_v(value).view(batch_size, seq_len_k, self.num_kv_heads, self.d_k)
        V = V.transpose(1, 2)  # (batch_size, num_kv_heads, seq_len_v, d_k)

        # Repeat K and V for each query group
        # Shape: (batch_size, num_heads, seq_len_k, d_k)
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Now K, V have the same number of heads as Q
        # Proceed with standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        # (batch_size, num_heads, seq_len_q, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len_q, self.d_model)

        # Output projection
        output = self.W_o(context)

        return output, attention_weights


def test_gqa():
    """Test GQA with different group configurations."""
    batch_size = 2
    seq_len = 16
    d_model = 512

    configurations = [
        (8, 8, "MHA (8 heads, 8 KV heads)"),
        (8, 4, "GQA (8 heads, 4 KV heads)"),
        (8, 2, "GQA (8 heads, 2 KV heads)"),
        (8, 1, "MQA (8 heads, 1 KV head)"),
    ]

    x = torch.randn(batch_size, seq_len, d_model)

    print("Comparing different GQA configurations:\n")

    for num_heads, num_kv_heads, description in configurations:
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)

        # Count parameters
        num_params = sum(p.numel() for p in gqa.parameters())

        # Forward pass
        output, _ = gqa(x, x, x)

        print(f"{description}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Queries per KV: {num_heads // num_kv_heads}")
        print(f"  Output shape: {output.shape}")
        print()

if __name__ == "__main__":
    test_gqa()
```

### GQA Trade-offs

**Advantages:**
- Flexible trade-off between quality and efficiency
- Significantly reduces KV cache compared to MHA
- Minimal quality loss compared to MHA (much better than MQA)
- Can be chosen per model requirements

**Disadvantages:**
- More complex implementation than MHA or MQA
- Requires careful tuning of the number of KV heads

**Typical Configurations:**
- **LLaMA 2 70B**: 64 query heads, 8 KV heads (8:1 ratio)
- **LLaMA 3**: Similar ratios
- **Mistral 7B**: 32 query heads, 8 KV heads (4:1 ratio)

---

## Comparison: MHA vs MQA vs GQA

Let's create a comprehensive comparison:

```python
import torch
import torch.nn as nn
import time

def benchmark_attention_variants():
    """
    Benchmark MHA, MQA, and GQA in terms of:
    - Parameter count
    - Memory usage
    - Inference speed
    """
    d_model = 4096
    num_heads = 32
    seq_len = 2048
    batch_size = 1

    print("=" * 80)
    print(f"Benchmark Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  seq_len: {seq_len}")
    print(f"  batch_size: {batch_size}")
    print("=" * 80)

    # Create models
    mha = MultiHeadAttention(d_model, num_heads)
    mqa = MultiQueryAttention(d_model, num_heads)
    gqa_4 = GroupedQueryAttention(d_model, num_heads, num_kv_heads=8)
    gqa_2 = GroupedQueryAttention(d_model, num_heads, num_kv_heads=4)

    models = [
        ("MHA (32 heads)", mha),
        ("GQA-8 (32 heads, 8 KV)", gqa_4),
        ("GQA-4 (32 heads, 4 KV)", gqa_2),
        ("MQA (32 heads, 1 KV)", mqa),
    ]

    # Generate input
    x = torch.randn(batch_size, seq_len, d_model)

    print("\nParameter Count Comparison:")
    print("-" * 80)
    for name, model in models:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name:30s}: {num_params:,} parameters")

    # KV Cache analysis
    print("\nKV Cache Size (per token, FP16):")
    print("-" * 80)
    d_k = d_model // num_heads

    cache_sizes = [
        ("MHA", 2 * num_heads * d_k * 2),  # 2 for K,V; 2 for FP16 bytes
        ("GQA-8", 2 * 8 * d_k * 2),
        ("GQA-4", 2 * 4 * d_k * 2),
        ("MQA", 2 * 1 * d_k * 2),
    ]

    base_size = cache_sizes[0][1]
    for name, size in cache_sizes:
        reduction = base_size / size
        print(f"{name:30s}: {size:6d} bytes/token ({reduction:4.1f}x reduction)")

    # Total cache for full sequence
    print(f"\nTotal KV Cache for {seq_len} tokens:")
    print("-" * 80)
    for name, size_per_token in cache_sizes:
        total_mb = (size_per_token * seq_len) / (1024 * 1024)
        print(f"{name:30s}: {total_mb:6.2f} MB")

    # Speed benchmark (CPU)
    print("\nInference Speed (CPU, 10 runs):")
    print("-" * 80)

    for name, model in models:
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(x, x, x)

            # Benchmark
            start = time.time()
            for _ in range(10):
                _ = model(x, x, x)
            elapsed = time.time() - start

            print(f"{name:30s}: {elapsed:.4f}s total, {elapsed/10:.4f}s per run")

    print("=" * 80)


def visualize_attention_heads():
    """Visualize what different heads might learn."""
    import matplotlib.pyplot as plt

    seq_len = 20
    d_model = 128
    num_heads = 4

    # Create input sequence
    x = torch.randn(1, seq_len, d_model)

    # Initialize MHA
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
    mha.eval()

    with torch.no_grad():
        _, attention_weights = mha(x, x, x)

    # attention_weights: (1, num_heads, seq_len, seq_len)
    attention_weights = attention_weights[0].numpy()  # (num_heads, seq_len, seq_len)

    fig, axes = plt.subplots(1, num_heads, figsize=(16, 4))

    for i in range(num_heads):
        im = axes[i].imshow(attention_weights[i], cmap='viridis', vmin=0, vmax=0.5)
        axes[i].set_title(f'Head {i+1}')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig('attention_heads_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved attention heads visualization to 'attention_heads_visualization.png'")


if __name__ == "__main__":
    benchmark_attention_variants()
    print("\n")
    visualize_attention_heads()
```

### Summary Table

| Variant | Query Heads | KV Heads | KV Cache Size | Quality | Speed | Used In |
|---------|-------------|----------|---------------|---------|-------|---------|
| **MHA** | $h$ | $h$ | $2 \times h \times d_k$ | Best | Baseline | GPT-2, GPT-3, BERT |
| **GQA** | $h$ | $g$ | $2 \times g \times d_k$ | Very Good | Faster | LLaMA 2/3, Mistral |
| **MQA** | $h$ | $1$ | $2 \times 1 \times d_k$ | Good | Fastest | PaLM, Falcon |

---

## Practical Considerations

### 1. Choosing the Number of Heads

**Common configurations:**
- Small models (100M params): 8-12 heads
- Medium models (1B params): 16-32 heads
- Large models (10B+ params): 32-64 heads

**Guidelines:**
- More heads = more capacity but more memory
- Ensure $d_k = d_{\text{model}} / h$ is not too small (typically 64-128)
- Powers of 2 are preferred for hardware efficiency

### 2. Head Pruning

Research shows many heads can be removed without significant performance loss:

```python
def analyze_head_importance(model, dataloader):
    """
    Analyze which attention heads are most important.
    Based on "Are Sixteen Heads Really Better than One?" (Michel et al., 2019)
    """
    head_importance = torch.zeros(model.num_layers, model.num_heads)

    model.eval()
    for batch in dataloader:
        # Forward pass with hooks to capture head outputs
        outputs = model(batch)
        loss = outputs.loss

        # Compute gradients
        loss.backward()

        # Accumulate head importance (gradient-based)
        for layer_idx, layer in enumerate(model.layers):
            for head_idx in range(model.num_heads):
                # Importance = gradient magnitude
                head_importance[layer_idx, head_idx] += layer.attention.heads[head_idx].grad.abs().sum()

    return head_importance
```

### 3. Initialization

Proper initialization is crucial for multi-head attention:

```python
def init_multihead_attention(module):
    """Initialize multi-head attention weights."""
    if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    # Some models use scaled initialization for output projection
    # to account for residual connection paths
    if hasattr(module, 'W_o'):
        nn.init.xavier_uniform_(module.W_o.weight, gain=1/math.sqrt(2))
```

### 4. Computational Complexity

For sequence length $n$ and model dimension $d$:

| Operation | Complexity |
|-----------|------------|
| Q, K, V projections | $O(n \cdot d^2)$ |
| Attention scores | $O(n^2 \cdot d)$ |
| Attention output | $O(n^2 \cdot d)$ |
| Output projection | $O(n \cdot d^2)$ |
| **Total** | $O(n^2 \cdot d + n \cdot d^2)$ |

For long sequences ($n > d$), the $O(n^2 \cdot d)$ term dominates, which motivates efficient attention variants (see [Flash Attention](12-flash-attention.md) and [Other Efficient Attention Variants](13-efficient-attention.md)).

### 5. Memory Optimization

```python
class MemoryEfficientMultiHeadAttention(nn.Module):
    """
    Memory-efficient MHA using chunking for very long sequences.
    Computes attention in chunks to reduce peak memory.
    """

    def __init__(self, d_model, num_heads, chunk_size=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.chunk_size = chunk_size

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        # Project
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention in chunks
        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            j = min(i + self.chunk_size, seq_len)
            Q_chunk = Q[:, :, i:j, :]

            # Compute attention for this chunk
            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask[:, :, i:j, :] == 0, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            output_chunk = torch.matmul(attn, V)
            outputs.append(output_chunk)

        # Concatenate chunks
        context = torch.cat(outputs, dim=2)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context), None
```

---

## Exercises

### Exercise 1: Implement Head-Specific Analysis

Write code to analyze what different attention heads learn:

```python
def analyze_attention_patterns(model, text_samples):
    """
    Analyze attention patterns across different heads.

    Tasks:
    1. Compute average attention distance for each head
    2. Identify heads that focus on local vs global context
    3. Visualize attention patterns for sample inputs
    """
    # Your implementation here
    pass
```

### Exercise 2: Convert MHA to GQA

Given a pre-trained model with MHA, write a function to convert it to GQA:

```python
def convert_mha_to_gqa(mha_model, num_kv_heads):
    """
    Convert a Multi-Head Attention model to Grouped-Query Attention.

    Hint: You'll need to merge K and V projections from multiple heads.
    Consider averaging or more sophisticated merging strategies.
    """
    # Your implementation here
    pass
```

### Exercise 3: Benchmark on Real Data

Implement a benchmark comparing MHA, MQA, and GQA on a language modeling task:

```python
def benchmark_on_language_modeling():
    """
    1. Train small transformers with MHA, MQA, and GQA
    2. Compare training time, memory usage, and final perplexity
    3. Analyze the trade-offs
    """
    # Your implementation here
    pass
```

### Exercise 4: Implement Flash Attention Interface

Modify the MultiHeadAttention class to support flash attention:

```python
class MultiHeadAttentionWithFlash(MultiHeadAttention):
    """
    Multi-head attention with optional flash attention backend.

    When use_flash=True, use torch.nn.functional.scaled_dot_product_attention
    which implements flash attention when available.
    """

    def __init__(self, d_model, num_heads, dropout=0.1, use_flash=True):
        super().__init__(d_model, num_heads, dropout)
        self.use_flash = use_flash

    def forward(self, query, key, value, mask=None):
        # Your implementation here
        # Hint: Use F.scaled_dot_product_attention when use_flash=True
        pass
```

### Exercise 5: Head Pruning

Implement head pruning based on importance scores:

```python
def prune_attention_heads(model, keep_ratio=0.5):
    """
    Prune less important attention heads.

    1. Compute head importance (e.g., gradient-based)
    2. Identify least important heads
    3. Remove them and adjust model architecture
    """
    # Your implementation here
    pass
```

---

## References

### Foundational Papers

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer paper introducing multi-head attention
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. **Fast Transformer Decoding: One Write-Head is All You Need** (Shazeer, 2019)
   - Introduces Multi-Query Attention (MQA)
   - [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)

3. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Ainslie et al., 2023)
   - Introduces Grouped-Query Attention
   - [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

### Analysis and Understanding

4. **Are Sixteen Heads Really Better than One?** (Michel et al., 2019)
   - Shows many attention heads can be pruned
   - [arXiv:1905.10650](https://arxiv.org/abs/1905.10650)

5. **Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned** (Voita et al., 2019)
   - Analyzes head specialization and importance
   - [arXiv:1905.09418](https://arxiv.org/abs/1905.09418)

### Applications in Modern LLMs

6. **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023)
   - Uses multi-head attention with RMSNorm
   - [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

7. **LLaMA 2: Open Foundation and Fine-Tuned Chat Models** (Touvron et al., 2023)
   - Introduces GQA for improved efficiency
   - [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)

8. **Mistral 7B** (Jiang et al., 2023)
   - Uses GQA with sliding window attention
   - [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)

### Efficient Attention Variants

9. **Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., 2022)
   - Algorithm for efficient attention computation
   - See [Flash Attention](12-flash-attention.md)
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

10. **Self-Attention Does Not Need O(n²) Memory** (Rabe & Staats, 2021)
    - Memory-efficient attention computation
    - [arXiv:2112.05682](https://arxiv.org/abs/2112.05682)

---

## Summary

In this chapter, we covered:

1. **Multi-Head Attention**: The standard mechanism that enables parallel attention patterns
2. **Implementation Details**: How to split heads, compute attention, and recombine outputs
3. **Multi-Query Attention (MQA)**: Shares K,V across heads for memory efficiency
4. **Grouped-Query Attention (GQA)**: Balances quality and efficiency with grouped K,V
5. **Trade-offs**: Parameter count, memory usage, inference speed, and model quality
6. **Practical Considerations**: Initialization, head pruning, and optimization strategies

Multi-head attention is a cornerstone of modern LLMs. Understanding the trade-offs between MHA, MQA, and GQA is essential for both interviews and practical model development.

**Next Chapter**: [Bidirectional vs Causal Attention](05-bidirectional-causal-attention.md) - Learn about different attention masking patterns for different tasks.

**Related Chapters**:
- [Basic Attention](03-basic-attention.md) - Foundation of attention mechanisms
- [Flash Attention](12-flash-attention.md) - Efficient attention computation
- [The Transformer Block](09-transformer-block.md) - How MHA fits into the full architecture
- [Architecture Comparison: Modern LLMs](29-model-architectures.md) - See which models use which attention variants
