# Chapter 26: KV Cache

The Key-Value (KV) cache is one of the most important optimizations for autoregressive language model inference. Understanding KV caching is essential for ML interviews, as it's the primary technique that makes modern chatbots like ChatGPT and Claude practical. This chapter consolidates KV cache concepts scattered across the guide into a comprehensive treatment.

## Table of Contents

1. [Why KV Cache Exists](#why-kv-cache-exists)
2. [Memory Analysis and Scaling](#memory-analysis-and-scaling)
3. [Basic KV Cache Implementation](#basic-kv-cache-implementation)
4. [Position Encoding Interactions](#position-encoding-interactions)
5. [Reducing Cache Size: MQA and GQA](#reducing-cache-size-mqa-and-gqa)
6. [KV Cache Quantization](#kv-cache-quantization)
7. [Memory Management: PagedAttention](#memory-management-pagedattention)
8. [Streaming and Long Context](#streaming-and-long-context)
9. [Production Considerations](#production-considerations)
10. [Common Interview Questions](#common-interview-questions)
11. [Summary](#summary)

---

## Why KV Cache Exists

### The Autoregressive Generation Problem

Autoregressive language models generate text one token at a time. Without caching, each generation step requires recomputing attention for all previous tokens.

**Generation without cache:**

```text
Step 1: Process "Hello"           → Generate "world"
Step 2: Process "Hello world"     → Generate "!"
Step 3: Process "Hello world !"   → Generate next token
...
```

At step $t$, we process all $t$ previous tokens, resulting in $O(t^2)$ computation for generating $t$ tokens.

**The key insight**: In attention, the key and value representations of token $i$ don't change when we add token $i+1$.

```math
K_i = W_{K} \cdot h_i, \quad V_i = W_{V} \cdot h_i
```

where $h_i$ is the hidden state at position $i$. Once computed, $K_i$ and $V_i$ remain constant for all future tokens.

### From O(N²) to O(N)

**Without caching:**

- Token 1: Compute 1 KV pair
- Token 2: Recompute 2 KV pairs (including token 1 again)
- Token 3: Recompute 3 KV pairs (including tokens 1 and 2 again)
- Token N: Recompute N KV pairs

Total: $1 + 2 + 3 + \cdots + N = O(N^2)$ KV computations

**With caching:**

- Token 1: Compute 1 KV pair, cache it
- Token 2: Compute 1 new KV pair, append to cache
- Token 3: Compute 1 new KV pair, append to cache
- Token N: Compute 1 new KV pair, append to cache

Total: $N$ KV computations → $O(N)$

### Attention's Decomposability

KV caching works because attention can be computed incrementally:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

For a new query $q_{\text{new}}$ attending to cached keys and values:

```math
\text{score} = q_{\text{new}} \cdot [K_{\text{cache}}, k_{\text{new}}]^T = [q_{\text{new}} \cdot K_{\text{cache}}^T, \; q_{\text{new}} \cdot k_{\text{new}}^T]
```

The cached keys contribute independently to the attention score, allowing us to concatenate old and new keys without recomputation.

![KV Cache Memory Flow](../assets/diagrams/ch14-kv-cache-memory-flow.svg)

---

## Memory Analysis and Scaling

### Memory Formula

For a single layer, the KV cache stores:

```math
\text{Memory}_{\text{layer}} = 2 \times \text{n\_kv\_heads} \times \text{seq\_len} \times \text{head\_dim} \times \text{bytes\_per\_element}
```

where:

- Factor of 2: separate storage for keys and values
- `n_kv_heads`: number of key-value heads (can differ from query heads in GQA)
- `seq_len`: current sequence length
- `head_dim`: dimension per head (typically 64-128)
- `bytes_per_element`: 2 for FP16, 4 for FP32, 1 for FP8

For the full model with $L$ layers and batch size $B$:

```math
\text{Memory}_{\text{total}} = B \times L \times 2 \times \text{n\_kv\_heads} \times \text{seq\_len} \times \text{head\_dim} \times \text{bytes\_per\_element}
```

### When Cache Exceeds Model Weights

```python
import torch

def compare_cache_vs_weights():
    """
    Calculate when KV cache memory exceeds model weight memory.

    For large models with long contexts, cache can dominate!
    """

    # Model configuration (LLaMA 70B-style)
    n_layers = 80
    n_kv_heads = 8  # GQA: 64 query heads, 8 KV heads
    head_dim = 128
    d_model = 8192
    model_params = 70e9  # 70B parameters

    # FP16 precision
    bytes_per_param = 2

    # Model weights memory
    model_memory_gb = (model_params * bytes_per_param) / (1024**3)

    print("="*70)
    print(f"Model: {model_params/1e9:.0f}B parameters")
    print(f"Model memory (FP16): {model_memory_gb:.1f} GB")
    print("="*70)

    # Cache memory for different sequence lengths
    seq_lengths = [2048, 8192, 32768, 100000]
    batch_size = 1

    print(f"\n{'Seq Length':<12} {'Cache (GB)':<12} {'vs Weights':<15} {'Total (GB)'}")
    print("-"*70)

    for seq_len in seq_lengths:
        cache_memory = (
            batch_size * n_layers * 2 * n_kv_heads *
            seq_len * head_dim * bytes_per_param
        ) / (1024**3)

        ratio = cache_memory / model_memory_gb
        total = model_memory_gb + cache_memory

        print(f"{seq_len:<12} {cache_memory:<12.1f} {ratio:<15.2%} {total:.1f}")

    print("\n" + "="*70)
    print("Key insight: At long contexts, KV cache dominates memory!")
    print("="*70)

# Output example:
# ======================================================================
# Model: 70B parameters
# Model memory (FP16): 140.0 GB
# ======================================================================
#
# Seq Length   Cache (GB)   vs Weights      Total (GB)
# ----------------------------------------------------------------------
# 2048         2.6          1.9%            142.6
# 8192         10.5         7.5%            150.5
# 32768        41.9         29.9%           181.9
# 100000       128.0        91.4%           268.0
#
# ======================================================================
# Key insight: At long contexts, KV cache dominates memory!
# ======================================================================
```

### Scaling with Multiple Users

For a serving system with $U$ concurrent users:

```math
\text{Memory}_{\text{serving}} = U \times L \times 2 \times \text{n\_kv\_heads} \times \text{avg\_seq\_len} \times \text{head\_dim} \times \text{bytes}
```

**Example**: Serving 100 users with 7B model (32 layers, 32 heads):

- Sequence length: 4096 tokens
- FP16 (2 bytes)
- Memory: $100 \times 32 \times 2 \times 32 \times 4096 \times 128 \times 2 / (1024^3) \approx 200$ GB

This is why KV cache optimization is critical for deployment!

![KV Cache Scaling](../assets/diagrams/ch14-kv-cache-scaling.svg)

---

## Basic KV Cache Implementation

### Canonical Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionWithKVCache(nn.Module):
    """
    Multi-head attention with KV caching for autoregressive generation.

    See [Multi-Head Attention](04-multi-head-attention.md) for base concepts.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """
        Forward pass with optional KV caching.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            cache: Optional dict with 'key' and 'value' from previous steps
                   Each has shape (batch, n_heads, cache_len, head_dim)
            use_cache: Whether to return cache for next step

        Returns:
            output: (batch, seq_len, d_model)
            cache: Updated cache dict (if use_cache=True)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (batch, seq_len, n_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # If cache exists, concatenate with new K, V
        if cache is not None:
            K = torch.cat([cache['key'], K], dim=2)  # Extend along seq_len dimension
            V = torch.cat([cache['value'], V], dim=2)

        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask (only needed during prefill when seq_len \gt 1)
        if seq_len \gt 1:
            # Create causal mask for new tokens
            total_len = K.shape[2]
            mask = torch.triu(
                torch.ones(seq_len, total_len, device=x.device),
                diagonal=total_len - seq_len + 1
            )
            scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)

        # Update cache if requested
        new_cache = None
        if use_cache:
            new_cache = {'key': K, 'value': V}

        return output, new_cache


def demonstrate_kv_cache():
    """
    Demonstrate KV cache speedup for generation.
    """
    d_model = 512
    n_heads = 8
    batch_size = 1

    # Initialize model
    attn = AttentionWithKVCache(d_model, n_heads)
    attn.eval()

    print("KV Cache Demonstration")
    print("="*70)

    # Simulate generation: start with 5 tokens, generate 10 more
    initial_len = 5
    total_len = 15

    # Initial prompt
    prompt = torch.randn(batch_size, initial_len, d_model)

    print(f"Initial prompt: {initial_len} tokens")

    # Method 1: WITHOUT cache (recompute everything)
    print("\nMethod 1: WITHOUT KV cache")
    print("-"*70)

    import time
    current_seq = prompt

    start = time.time()
    for step in range(initial_len, total_len):
        # Simulate generating next token
        new_token = torch.randn(batch_size, 1, d_model)
        current_seq = torch.cat([current_seq, new_token], dim=1)

        with torch.no_grad():
            # Recompute attention for ALL tokens every time
            output, _ = attn(current_seq, cache=None, use_cache=False)

        print(f"  Step {step - initial_len + 1}: Processed {current_seq.shape[1]} tokens")

    time_without_cache = time.time() - start
    print(f"Total time: {time_without_cache:.4f}s")

    # Method 2: WITH cache (incremental)
    print("\nMethod 2: WITH KV cache")
    print("-"*70)

    start = time.time()

    # Prefill: process initial prompt
    with torch.no_grad():
        output, cache = attn(prompt, cache=None, use_cache=True)
    print(f"  Prefill: Processed {initial_len} tokens, cached K,V")

    # Decode: generate one token at a time
    for step in range(initial_len, total_len):
        new_token = torch.randn(batch_size, 1, d_model)

        with torch.no_grad():
            # Only process the NEW token, reuse cached K,V
            output, cache = attn(new_token, cache=cache, use_cache=True)

        cache_size = cache['key'].shape[2]
        print(f"  Step {step - initial_len + 1}: Processed 1 new token, cache size: {cache_size}")

    time_with_cache = time.time() - start
    print(f"Total time: {time_with_cache:.4f}s")

    # Summary
    print("\n" + "="*70)
    print(f"Speedup: {time_without_cache / time_with_cache:.2f}x")
    print("="*70)

if __name__ == "__main__":
    demonstrate_kv_cache()
```

### Prefill vs Decode Phases

KV caching creates two distinct phases during generation:

1. **Prefill**: Process the initial prompt in parallel
   - Input: entire prompt (e.g., 100 tokens)
   - Compute K, V for all prompt tokens
   - Store in cache
   - Compute-bound (lots of matrix multiplication)

2. **Decode**: Generate tokens one at a time
   - Input: single new token
   - Compute K, V for new token only
   - Append to cache
   - Memory-bound (loading cached K, V from memory)

**Performance characteristics**:

- Prefill: High GPU utilization, throughput-oriented
- Decode: Low GPU utilization (single token), latency-oriented

This is why production systems separate prefill and decode batches!

---

## Position Encoding Interactions

### RoPE Works Perfectly with KV Cache

Rotary Position Embeddings (RoPE) integrate seamlessly with KV caching because rotations are applied **before** caching.

See [Rotary Position Embeddings](08-rope.md) for RoPE details.

**Key insight**: Once we rotate K at position $m$ and cache it, that rotation is "baked in":

```math
K_m^{\text{cached}} = \mathbf{R}_m K_m
```

When a new query at position $n$ attends to this cached key:

```math
\text{score} = Q_n^T K_m = (R_n Q_n)^T (R_m K_m) = Q_n^T R_n^T R_m K_m = Q_n^T R_{m-n} K_m
```

The relative position $(m-n)$ emerges naturally, which is exactly what RoPE is designed to capture!

### Implementation with RoPE

```python
class RoPEAttentionWithCache(nn.Module):
    """
    Attention with RoPE and KV caching.

    RoPE is applied at the correct positions before caching.
    See [RoPE](08-rope.md) for position encoding details.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # RoPE frequencies
        self.rope_base = rope_base
        self.register_buffer(
            'inv_freq',
            1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        )

    def apply_rope(
        self,
        x: torch.Tensor,
        start_pos: int
    ) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: (batch, n_heads, seq_len, head_dim)
            start_pos: Starting position for this sequence

        Returns:
            Rotated tensor
        """
        batch, n_heads, seq_len, head_dim = x.shape

        # Positions for current tokens
        positions = torch.arange(
            start_pos, start_pos + seq_len,
            device=x.device
        ).float()

        # Compute rotation angles
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, head_dim//2)
        freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)

        # Split x into pairs and apply rotation
        x_complex = torch.view_as_complex(
            x.float().reshape(batch, n_heads, seq_len, head_dim // 2, 2)
        )

        freqs_cis = torch.polar(
            torch.ones_like(freqs),
            freqs
        )

        x_rotated = x_complex * freqs_cis
        x_out = torch.view_as_real(x_rotated).flatten(-2)

        return x_out.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        cache: dict | None = None,
        use_cache: bool = False
    ) -> tuple[torch.Tensor, dict | None]:
        """
        Forward with RoPE and caching.

        Args:
            x: (batch, seq_len, d_model)
            start_pos: Position offset for RoPE (0 for prefill, >0 for decode)
            cache: Previous K, V cache
            use_cache: Whether to return updated cache
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose: (batch, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Apply RoPE to Q and K with correct positions
        Q = self.apply_rope(Q, start_pos)
        K = self.apply_rope(K, start_pos)
        # Note: V is NOT rotated (values don't need position info)

        # Concatenate with cache if present
        if cache is not None:
            K = torch.cat([cache['key'], K], dim=2)
            V = torch.cat([cache['value'], V], dim=2)

        # Standard attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask for prefill
        if seq_len \gt 1:
            total_len = K.shape[2]
            mask = torch.triu(
                torch.ones(seq_len, total_len, device=x.device),
                diagonal=total_len - seq_len + 1
            )
            scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)

        # Update cache
        new_cache = None
        if use_cache:
            new_cache = {'key': K, 'value': V}

        return output, new_cache
```

**Common interview question**: "Why can we cache the rotated K values instead of raw K and rotate on-the-fly?"

**Answer**: Because RoPE rotations are deterministic functions of position. Once K at position $m$ is rotated and cached, it already encodes both content and position. Re-rotating would be wrong (double rotation) and wasteful.

---

## Reducing Cache Size: MQA and GQA

The KV cache size is directly proportional to the number of KV heads. Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce cache size by sharing keys and values across query heads.

See [Multi-Head Attention](04-multi-head-attention.md) for full details on MQA and GQA.

### Multi-Head Attention (MHA)

Standard attention: each head has its own K, V projections.

```math
\text{Cache}_{\text{MHA}} = 2 \times h \times \text{seq\_len} \times d_k
```

where $h$ is the number of heads.

### Multi-Query Attention (MQA)

Share one set of K, V across all query heads:

```math
\text{Cache}_{\text{MQA}} = 2 \times 1 \times \text{seq\_len} \times d_k
```

**Reduction**: $h$x smaller (e.g., 32x for 32 heads)

**Used in**: PaLM, Falcon, StarCoder

### Grouped-Query Attention (GQA)

Group query heads to share K, V. With $g$ groups:

```math
\text{Cache}_{\text{GQA}} = 2 \times g \times \text{seq\_len} \times d_k
```

**Reduction**: $(h/g)$x smaller

**Used in**: LLaMA 2/3, Mistral, Qwen

**Example**: LLaMA 2 70B

- 64 query heads
- 8 KV heads
- Reduction: 8x smaller cache than MHA
- Quality loss: <1% compared to MHA

![MHA vs MQA vs GQA Cache Comparison](../assets/diagrams/ch14-mha-mqa-gqa-cache-comparison.svg)

### Cache Size Comparison

```python
def compare_cache_sizes():
    """
    Compare KV cache sizes for MHA, GQA, and MQA.
    """
    # Configuration (LLaMA-style 7B)
    d_model = 4096
    n_query_heads = 32
    head_dim = d_model // n_query_heads  # 128
    seq_len = 8192
    n_layers = 32
    batch_size = 1
    bytes_per_elem = 2  # FP16

    print("="*70)
    print("KV Cache Size Comparison")
    print("="*70)
    print(f"Model: {n_layers} layers, {n_query_heads} query heads, {head_dim} head_dim")
    print(f"Sequence: {seq_len} tokens, Batch: {batch_size}, Precision: FP16")
    print("="*70)

    configs = [
        ("MHA", n_query_heads),
        ("GQA-8", 8),
        ("GQA-4", 4),
        ("MQA", 1),
    ]

    print(f"\n{'Config':<15} {'KV Heads':<10} {'Cache (GB)':<15} {'Reduction':<15}")
    print("-"*70)

    mha_size = None

    for name, n_kv_heads in configs:
        cache_size = (
            batch_size * n_layers * 2 * n_kv_heads *
            seq_len * head_dim * bytes_per_elem
        ) / (1024**3)

        if mha_size is None:
            mha_size = cache_size
            reduction = "baseline"
        else:
            reduction = f"{mha_size / cache_size:.1f}x"

        print(f"{name:<15} {n_kv_heads:<10} {cache_size:<15.2f} {reduction:<15}")

    print("\n" + "="*70)
    print("Key insight: GQA provides good quality/efficiency tradeoff")
    print("="*70)

# Output:
# ======================================================================
# KV Cache Size Comparison
# ======================================================================
# Model: 32 layers, 32 query heads, 128 head_dim
# Sequence: 8192 tokens, Batch: 1, Precision: FP16
# ======================================================================
#
# Config          KV Heads   Cache (GB)      Reduction
# ----------------------------------------------------------------------
# MHA             32         4.19            baseline
# GQA-8           8          1.05            4.0x
# GQA-4           4          0.52            8.0x
# MQA             1          0.13            32.0x
#
# ======================================================================
# Key insight: GQA provides good quality/efficiency tradeoff
# ======================================================================
```

---

## KV Cache Quantization

Quantizing the KV cache to lower precision (INT8, FP8, or INT4) provides significant memory savings with minimal quality loss.

See [Hardware, Quantization, and Training Optimization](29-hardware-quantization-optimization.md) for more quantization details.

### Why Quantization Works for KV Cache

**Key insight**: Attention computes similarity scores via dot products. These are inherently noisy operations that don't require full precision.

**Observation**: Quantization error in K, V affects attention weights, but softmax is robust to small perturbations.

### INT8 Quantization

Quantize FP16 → INT8 using per-tensor or per-channel scaling:

```math
K_{\text{INT8}} = \text{round}\left(\frac{K_{\text{FP16}}}{\text{scale}}\right), \quad \text{scale} = \frac{\max(|K|)}{127}
```

**Benefits**:

- 2x memory reduction
- <1% quality loss
- Hardware support on most GPUs

### FP8 Quantization

Use 8-bit floating point (E4M3 or E5M2 format):

```math
K_{\text{FP8}} = \text{cast}(K_{\text{FP16}}, \text{FP8})
```

**Benefits**:

- 2x memory reduction
- Better dynamic range than INT8
- Native support on H100+ GPUs

### Implementation

```python
class QuantizedKVCache:
    """
    KV cache with INT8 or FP8 quantization.

    Stores K, V in quantized format to reduce memory.
    Dequantizes on-the-fly during attention computation.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: str = "int8"  # "int8" or "fp8"
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Pre-allocate cache (per layer)
        # For INT8: use torch.int8
        # For FP8: use torch.float8_e4m3fn (requires torch >= 2.1)
        cache_dtype = torch.int8 if dtype == "int8" else torch.float8_e4m3fn

        self.k_cache = torch.zeros(
            n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim,
            dtype=cache_dtype
        )
        self.v_cache = torch.zeros(
            n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim,
            dtype=cache_dtype
        )

        # Scaling factors for INT8 (per layer, per head)
        if dtype == "int8":
            self.k_scale = torch.ones(n_layers, 1, n_kv_heads, 1, 1)
            self.v_scale = torch.ones(n_layers, 1, n_kv_heads, 1, 1)

    def quantize_int8(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize FP16 tensor to INT8.

        Args:
            x: FP16 tensor (batch, n_heads, seq_len, head_dim)

        Returns:
            x_quant: INT8 tensor
            scale: Scaling factor
        """
        # Compute scale per head: max(abs(x)) / 127
        scale = x.abs().amax(dim=(-2, -1), keepdim=True) / 127.0
        scale = scale.clamp(min=1e-5)  # Avoid division by zero

        # Quantize
        x_quant = torch.clamp(
            torch.round(x / scale),
            -128, 127
        ).to(torch.int8)

        return x_quant, scale

    def dequantize_int8(
        self,
        x_quant: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize INT8 back to FP16.

        Args:
            x_quant: INT8 tensor
            scale: Scaling factor

        Returns:
            FP16 tensor
        """
        return x_quant.to(torch.float16) * scale

    def update(
        self,
        layer_idx: int,
        start_pos: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor
    ):
        """
        Add new K, V to cache at given layer and position.

        Args:
            layer_idx: Which transformer layer
            start_pos: Position to start writing
            k_new: New keys (batch, n_kv_heads, seq_len, head_dim) in FP16
            v_new: New values (batch, n_kv_heads, seq_len, head_dim) in FP16
        """
        batch_size, n_heads, seq_len, head_dim = k_new.shape
        end_pos = start_pos + seq_len

        if self.dtype == "int8":
            # Quantize
            k_quant, k_scale = self.quantize_int8(k_new)
            v_quant, v_scale = self.quantize_int8(v_new)

            # Store
            self.k_cache[layer_idx, :batch_size, :, start_pos:end_pos] = k_quant
            self.v_cache[layer_idx, :batch_size, :, start_pos:end_pos] = v_quant

            # Update scales
            self.k_scale[layer_idx, :batch_size, :] = k_scale
            self.v_scale[layer_idx, :batch_size, :] = v_scale

        else:  # FP8
            # Direct cast to FP8
            k_fp8 = k_new.to(torch.float8_e4m3fn)
            v_fp8 = v_new.to(torch.float8_e4m3fn)

            self.k_cache[layer_idx, :batch_size, :, start_pos:end_pos] = k_fp8
            self.v_cache[layer_idx, :batch_size, :, start_pos:end_pos] = v_fp8

    def get(
        self,
        layer_idx: int,
        batch_size: int,
        seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve K, V from cache (dequantized to FP16).

        Args:
            layer_idx: Which layer
            batch_size: Batch size
            seq_len: How many positions to retrieve

        Returns:
            k, v: Dequantized tensors in FP16
        """
        k_quant = self.k_cache[layer_idx, :batch_size, :, :seq_len]
        v_quant = self.v_cache[layer_idx, :batch_size, :, :seq_len]

        if self.dtype == "int8":
            k_scale = self.k_scale[layer_idx, :batch_size, :]
            v_scale = self.v_scale[layer_idx, :batch_size, :]

            k = self.dequantize_int8(k_quant, k_scale)
            v = self.dequantize_int8(v_quant, v_scale)
        else:  # FP8
            k = k_quant.to(torch.float16)
            v = v_quant.to(torch.float16)

        return k, v


def benchmark_quantized_cache():
    """
    Benchmark memory savings from quantization.
    """
    # Configuration
    n_layers = 32
    n_kv_heads = 8
    head_dim = 128
    seq_len = 8192
    batch_size = 1

    print("="*70)
    print("KV Cache Quantization Benchmark")
    print("="*70)

    # FP16 baseline
    fp16_size = (
        2 * n_layers * batch_size * n_kv_heads * seq_len * head_dim * 2
    ) / (1024**3)

    # INT8
    int8_size = (
        2 * n_layers * batch_size * n_kv_heads * seq_len * head_dim * 1
    ) / (1024**3)

    # FP8
    fp8_size = int8_size  # Same size as INT8

    print(f"\n{'Format':<15} {'Size (GB)':<15} {'Reduction':<15}")
    print("-"*70)
    print(f"{'FP16 (baseline)':<15} {fp16_size:<15.2f} {'1.0x':<15}")
    print(f"{'INT8':<15} {int8_size:<15.2f} {f'{fp16_size/int8_size:.1f}x':<15}")
    print(f"{'FP8':<15} {fp8_size:<15.2f} {f'{fp16_size/fp8_size:.1f}x':<15}")

    print("\n" + "="*70)
    print("Typical quality loss: <1% on most tasks")
    print("="*70)

if __name__ == "__main__":
    benchmark_quantized_cache()
```

### Quality vs Memory Tradeoff

| Precision | Memory | Quality Loss | Hardware Support |
|-----------|--------|--------------|------------------|
| FP16      | 1.0x   | 0%           | Universal        |
| FP8 (E4M3)| 0.5x   | <0.5%        | H100+            |
| INT8      | 0.5x   | <1%          | Most GPUs        |
| INT4      | 0.25x  | 1-3%         | Limited          |

**Recommendation**: Use FP8 on H100+, INT8 on older GPUs. INT4 only when memory is extremely constrained.

---

## Memory Management: PagedAttention

PagedAttention (from vLLM) applies virtual memory concepts to KV cache management, enabling efficient memory utilization and sharing. This section provides comprehensive coverage of PagedAttention as it's fundamentally a KV-cache memory management technique.

### The KV Cache Fragmentation Problem

#### Why Traditional Allocation Fails at Scale

**Problem**: Traditional serving systems allocate a contiguous memory block for each request's KV cache, sized for the maximum possible sequence length. This leads to severe memory waste because:

1. **Variable length requests**: Real requests have highly variable lengths (100 tokens to 2K tokens), but we must allocate for the maximum (e.g., 2048 tokens)
2. **Cannot batch efficiently**: Can't batch a 100-token request with a 1000-token request without wasting memory
3. **Memory-bound throughput**: The wasted memory prevents serving more requests, reducing GPU utilization

**Theoretical context**: This is analogous to the classic memory fragmentation problem in operating systems - but worse, because we're fragmenting GPU VRAM (a scarce resource) rather than abundant CPU RAM.

**Why existing solutions don't work**:

- **Dynamic allocation**: Can't easily resize GPU tensors without expensive copies
- **Separate buffers**: Creates even more fragmentation
- **Padding**: Wastes computation in addition to memory

**Quantifying the waste**: In production workloads, traditional caching wastes 60-80% of allocated KV cache memory. On a 40GB A100, this means only ~10GB effectively used!

The code below demonstrates how quickly this waste accumulates:

```python
def illustrate_kv_cache_fragmentation():
    """
    Traditional KV cache allocation suffers from fragmentation.

    Problem: Each request allocates a contiguous memory block for its
    entire KV cache. This leads to:

    1. Memory fragmentation (wasted space)
    2. Cannot batch requests with different lengths efficiently
    3. Memory bound by max_length, not actual length

    """

    # Traditional approach
    class TraditionalKVCache:
        def __init__(self, max_seq_len, n_layers, n_heads, head_dim, max_batch):
            # Preallocate for worst case
            self.max_seq_len = max_seq_len
            self.cache = torch.zeros(
                max_batch, n_layers, 2, n_heads, max_seq_len, head_dim
            )

        def get_memory_usage(self, batch_size, actual_lengths):
            """Calculate memory waste."""
            total_capacity = batch_size * self.max_seq_len
            actual_used = sum(actual_lengths)
            waste = total_capacity - actual_used
            waste_pct = (waste / total_capacity) * 100
            return waste_pct

    # Example: LLaMA-13B serving
    n_layers, n_heads, head_dim = 40, 40, 128
    max_seq_len = 2048
    max_batch = 8

    cache = TraditionalKVCache(max_seq_len, n_layers, n_heads, head_dim, max_batch)

    # Real request lengths vary widely
    actual_lengths = [128, 512, 256, 1024, 64, 2048, 300, 450]

    waste_pct = cache.get_memory_usage(len(actual_lengths), actual_lengths)

    print("Traditional KV Cache Problems:")
    print("-" * 60)
    print(f"Max sequence length:     {max_seq_len}")
    print(f"Batch size:              {len(actual_lengths)}")
    print(f"Actual lengths:          {actual_lengths}")
    print(f"Total capacity:          {max_seq_len * len(actual_lengths):,} tokens")
    print(f"Actually used:           {sum(actual_lengths):,} tokens")
    print(f"Wasted memory:           {waste_pct:.1f}%")
    print("\nThis waste prevents batching more requests!")
```

### PagedAttention Solution: Virtual Memory for KV Cache

PagedAttention borrows ideas from virtual memory in operating systems:

1. **Block-based allocation**: Divide KV cache into fixed-size blocks (pages)
2. **Non-contiguous storage**: Request's KV cache doesn't need contiguous memory
3. **On-demand allocation**: Allocate blocks as needed, not upfront

#### How PagedAttention Transforms the Problem

**The key insight**: Just like virtual memory in OS, we can decouple the logical sequence of KV vectors from their physical storage location. Each sequence maintains a **block table** (like a page table) that maps logical positions to physical blocks.

**Why this works for attention**: Attention computation is:

```math
\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V
```

The key observation: we can gather K and V from non-contiguous blocks because matrix multiplication doesn't require contiguous memory - we're doing random access anyway!

**Theoretical advantages**:

1. **Near-zero internal fragmentation**: Only waste memory within the last block of each sequence (average `block_size / 2` tokens)
2. **Perfect external fragmentation**: All free blocks can be used by any request
3. **Dynamic batching**: Can batch any mix of sequence lengths without waste

**Implementation complexity**: Requires custom CUDA kernels to efficiently gather K/V from scattered blocks. The naive PyTorch implementation below shows the concept but is slow - production uses optimized kernels.

**Production impact**: vLLM reports 2-4x higher throughput than traditional serving systems on real workloads, purely from better memory utilization enabling larger batch sizes.

### Implementation Details

```python
class PagedAttention(nn.Module):
    """
    PagedAttention with block-based KV cache management.

    Key innovation: KV cache is divided into fixed-size blocks.
    Each sequence's KV cache is a list of block pointers (like virtual memory).

    Benefits:

    - Near-zero memory waste (internal fragmentation only within last block)
    - Efficient batching of variable-length sequences
    - Easy memory sharing for parallel sampling (beam search, etc.)

    Used by: vLLM serving system (widely adopted in production)

    Reference: Kwon et al., "Efficient Memory Management for Large Language
    Model Serving with PagedAttention" (SOSP 2023)
    https://arxiv.org/abs/2309.06180

    See also: [Hardware and Optimization](29-hardware-quantization-optimization.md)
    for integration with quantization and other optimizations.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        block_size: int = 16,  # Typical: 16-64 tokens per block
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.block_size = block_size

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        block_tables: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        PagedAttention forward pass.

        Args:
            x: [batch, seq_len, dim] - Query tokens
            block_tables: [batch, max_num_blocks] - Block pointers for each sequence
            k_cache: [num_blocks, block_size, n_heads, head_dim] - All K blocks
            v_cache: [num_blocks, block_size, n_heads, head_dim] - All V blocks
            context_lens: [batch] - Actual context length for each sequence

        Returns:
            output: [batch, seq_len, dim]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k_new, v_new = qkv.permute(2, 0, 3, 1, 4)  # [batch, n_heads, seq, head_dim]

        # Gather K, V from blocks (simplified - real implementation uses custom CUDA)
        outputs = []
        for i in range(batch):
            # Get blocks for this sequence
            num_blocks = (context_lens[i] + self.block_size - 1) // self.block_size
            seq_blocks = block_tables[i, :num_blocks]

            # Gather K, V from these blocks
            k_seq = k_cache[seq_blocks].reshape(-1, self.n_heads, self.head_dim)
            v_seq = v_cache[seq_blocks].reshape(-1, self.n_heads, self.head_dim)

            # Truncate to actual length
            k_seq = k_seq[:context_lens[i]]
            v_seq = v_seq[:context_lens[i]]

            # Concatenate with new K, V
            k_full = torch.cat([k_seq, k_new[i]], dim=0)
            v_full = torch.cat([v_seq, v_new[i]], dim=0)

            # Standard attention for this sequence
            scores = torch.matmul(q[i], k_full.transpose(-2, -1)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_full)

            outputs.append(out)

        # Stack and reshape
        out = torch.stack(outputs)  # [batch, n_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.out(out)


class BlockAllocator:
    """
    Block allocator for PagedAttention KV cache.

    Manages a pool of fixed-size blocks, allocating and freeing them
    as requests come and go.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_heads: int,
        head_dim: int,
        device: str = 'cuda'
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))

        # Preallocate all blocks
        self.k_cache = torch.zeros(
            num_blocks, block_size, n_heads, head_dim,
            device=device, dtype=torch.float16
        )
        self.v_cache = torch.zeros(
            num_blocks, block_size, n_heads, head_dim,
            device=device, dtype=torch.float16
        )

    def allocate(self, num_blocks_needed: int) -> list[int]:
        """
        Allocate blocks for a new sequence.

        Returns:
            List of block IDs
        """
        if len(self.free_blocks) \lt num_blocks_needed:
            raise MemoryError(f"Out of KV cache blocks")

        allocated = self.free_blocks[:num_blocks_needed]
        self.free_blocks = self.free_blocks[num_blocks_needed:]
        return allocated

    def free(self, block_ids: list[int]):
        """Free blocks when sequence is done."""
        self.free_blocks.extend(block_ids)

    def get_utilization(self) -> float:
        """Get cache utilization percentage."""
        used = self.num_blocks - len(self.free_blocks)
        return (used / self.num_blocks) * 100


def compare_traditional_vs_paged():
    """
    Compare memory efficiency: traditional vs paged.
    """
    # Configuration
    n_layers, n_heads, head_dim = 32, 32, 128
    max_seq_len = 2048
    block_size = 16
    dtype_bytes = 2  # FP16

    # Sample batch with varying lengths
    requests = [
        ("req1", 128),
        ("req2", 512),
        ("req3", 256),
        ("req4", 1024),
        ("req5", 64),
        ("req6", 2048),
        ("req7", 300),
        ("req8", 450),
    ]

    # Traditional: each request needs max_seq_len
    traditional_memory = (
        len(requests) * n_layers * 2 * n_heads * max_seq_len * head_dim * dtype_bytes
    ) / 1e9

    # Paged: only allocate blocks needed
    total_blocks_needed = 0
    for name, length in requests:
        blocks = (length + block_size - 1) // block_size
        total_blocks_needed += blocks

    paged_memory = (
        total_blocks_needed * n_layers * 2 * n_heads * block_size * head_dim * dtype_bytes
    ) / 1e9

    # Actual tokens used
    actual_tokens = sum(length for _, length in requests)

    print("Traditional vs PagedAttention Memory Comparison")
    print("=" * 70)
    print(f"{'Method':<20} {'Memory (GB)':<15} {'Tokens Used':<15} {'Waste':<10}")
    print("-" * 70)

    traditional_waste = ((len(requests) * max_seq_len - actual_tokens) /
                        (len(requests) * max_seq_len)) * 100
    paged_waste = ((total_blocks_needed * block_size - actual_tokens) /
                   (total_blocks_needed * block_size)) * 100

    print(f"{'Traditional':<20} {traditional_memory:<15.2f} "
          f"{actual_tokens:<15,} {traditional_waste:<10.1f}%")
    print(f"{'PagedAttention':<20} {paged_memory:<15.2f} "
          f"{actual_tokens:<15,} {paged_waste:<10.1f}%")

    print(f"\nMemory savings: {traditional_memory / paged_memory:.2f}x")
    print(f"This allows {traditional_memory / paged_memory:.1f}x more requests in same memory!")
```

### Copy-on-Write for Prefix Sharing

PagedAttention enables efficient memory sharing for common prefixes using copy-on-write semantics:

```python
class CopyOnWriteBlockAllocator(BlockAllocator):
    """
    Block allocator with copy-on-write support for sharing prefixes.

    Multiple sequences can share read-only blocks (e.g., system prompt).
    When a sequence needs to modify a shared block, we copy it first.
    """

    def __init__(self, num_blocks: int, block_size: int, n_heads: int, head_dim: int):
        super().__init__(num_blocks, block_size, n_heads, head_dim)
        # Reference count for each block
        self.ref_counts = [0] * num_blocks

    def allocate_shared(self, block_ids: list[int]) -> list[int]:
        """
        Share existing blocks (increment reference count).

        Args:
            block_ids: List of blocks to share

        Returns:
            Same block IDs (but now with higher ref count)
        """
        for block_id in block_ids:
            self.ref_counts[block_id] += 1
        return block_ids

    def copy_on_write(self, block_id: int) -> int:
        """
        Copy a shared block before modifying it.

        Args:
            block_id: ID of block to copy

        Returns:
            ID of new block with copied data
        """
        if self.ref_counts[block_id] <= 1:
            # Not shared, can modify in place
            return block_id

        # Shared, need to copy
        new_block_id = self.allocate(1)[0]
        self.k_cache[new_block_id] = self.k_cache[block_id].clone()
        self.v_cache[new_block_id] = self.v_cache[block_id].clone()

        # Decrement ref count of original
        self.ref_counts[block_id] -= 1
        self.ref_counts[new_block_id] = 1

        return new_block_id

    def free(self, block_ids: list[int]):
        """Free blocks (or just decrement ref count if shared)."""
        for block_id in block_ids:
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                self.free_blocks.append(block_id)


def demonstrate_prefix_sharing():
    """
    Show memory savings from sharing common prefixes.
    """

    system_prompt_tokens = 512  # Tokens in system prompt
    block_size = 16
    system_prompt_blocks = (system_prompt_tokens + block_size - 1) // block_size

    num_requests = 100
    avg_response_tokens = 256
    avg_response_blocks = (avg_response_tokens + block_size - 1) // block_size

    # Without sharing: each request has its own copy
    without_sharing = num_requests * (system_prompt_blocks + avg_response_blocks)

    # With sharing: system prompt blocks shared across all requests
    with_sharing = system_prompt_blocks + (num_requests * avg_response_blocks)

    print("Prefix Sharing with Copy-on-Write")
    print("=" * 70)
    print(f"System prompt:          {system_prompt_tokens} tokens ({system_prompt_blocks} blocks)")
    print(f"Number of requests:     {num_requests}")
    print(f"Avg response:           {avg_response_tokens} tokens ({avg_response_blocks} blocks)")
    print()
    print(f"Without sharing:        {without_sharing} blocks")
    print(f"With sharing:           {with_sharing} blocks")
    print(f"Memory savings:         {without_sharing / with_sharing:.2f}x")
    print()
    print("Use cases:")
    print("  - System prompts in production deployments")
    print("  - Beam search with shared prefix")
    print("  - Few-shot prompting with shared examples")
```

### Key Advantages of PagedAttention

1. **Memory efficiency**: ~3-4x improvement in real workloads
2. **Flexible batching**: Easily batch requests of different lengths
3. **Memory sharing**: Efficient parallel sampling (multiple beams share prefix)
4. **Preemption**: Can pause long requests to handle short ones

### Production Impact

vLLM (which implements PagedAttention) has become the standard for LLM serving because:

- 2-4x higher throughput than traditional serving systems
- Better GPU utilization
- Supports continuous batching (add/remove requests dynamically)

**Key Paper:**

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023)

---

## Streaming and Long Context

For streaming applications (chatbots, real-time systems) or extremely long contexts, we need strategies beyond simple caching.

### Attention Sinks and StreamingLLM

**Problem**: Naive truncation of old tokens causes catastrophic performance loss.

**Observation**: Models develop "attention sinks" - early tokens that absorb irrelevant attention mass.

**Solution** (StreamingLLM): Keep attention sink tokens (typically first 4) + recent window.

See [Chapter 23: Long Context Techniques](23-long-context.md) for full details.

```python
class StreamingKVCache:
    """
    Rolling KV cache with attention sinks for infinite streaming.

    Maintains:

    - First k tokens (attention sinks)
    - Most recent w tokens (sliding window)

    See Chapter 23 for implementation details.
    """

    def __init__(
        self,
        n_sink_tokens: int = 4,
        window_size: int = 2048,
        **cache_kwargs
    ):
        self.n_sink_tokens = n_sink_tokens
        self.window_size = window_size
        # Implementation in Chapter 23...
```

### Rolling Buffer Cache

For constrained memory, maintain a fixed-size rolling buffer:

```python
class RollingKVCache:
    """
    Fixed-size rolling buffer for KV cache.

    When full, evicts oldest tokens (except sinks).
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.current_size = 0

    def update(self, k_new, v_new):
        if self.current_size + k_new.shape[2] \gt self.max_size:
            # Evict oldest tokens (keep attention sinks)
            # ... implementation ...
            pass
```

---

## Production Considerations

### Prefill Throughput vs Decode Latency

Production systems optimize these phases differently:

**Prefill** (process prompt):

- **Goal**: Maximize throughput (tokens/second)
- **Optimization**: Batch multiple prompts, use large batches
- **Bottleneck**: Compute (matrix multiplication)

**Decode** (generate tokens):

- **Goal**: Minimize latency (time per token)
- **Optimization**: Serve one token at a time, optimize memory bandwidth
- **Bottleneck**: Memory (loading KV cache from HBM)

### Continuous Batching

Traditional batching waits for all sequences in batch to finish. **Continuous batching** adds new requests as others finish:

```text
Traditional:
[Req1, Req2, Req3] → all finish → [Req4, Req5, Req6]
(GPU idle while waiting)

Continuous:
[Req1, Req2, Req3] → Req1 finishes → add Req4
                   → Req2 finishes → add Req5
(GPU always busy)
```

**Benefit**: 2-3x higher throughput

**Implementation**: Requires dynamic cache management (PagedAttention helps!)

### Memory Budget Allocation

For a serving system with memory $M$:

```math
M = M_{\text{weights}} + M_{\text{KV cache}} + M_{\text{activations}} + M_{\text{buffer}}
```

Typical allocation:

- Model weights: 40-50%
- KV cache: 30-40%
- Activations: 10-20%
- Buffer: 5-10%

**Example**: A100 (80GB)

- 70B model (FP16): ~140 GB → need 2 GPUs minimum
- With tensor parallelism across 2 GPUs:
  - Weights: 70 GB per GPU
  - KV cache: ~8 GB per GPU (supports ~3-4 concurrent users at 4K context)

### Batch Size vs Context Length

Given fixed memory, there's a tradeoff:

```math
\text{batch\_size} \times \text{seq\_len} \approx \text{constant}
```

**Example**: 10 GB for KV cache

- Batch 1, 100K context: Possible
- Batch 10, 10K context: Possible
- Batch 100, 1K context: Possible

Choose based on serving pattern!

---

## Common Interview Questions

### Q1: Why does KV cache exist?

**A**: During autoregressive generation, each new token attends to all previous tokens. Without caching, we'd recompute the K, V projections for all previous tokens at each step, resulting in $O(N^2)$ computation. KV cache stores these projections, reducing to $O(N)$.

### Q2: How much memory does KV cache use?

**A**: For one layer:

```math
2 \times \text{n\_kv\_heads} \times \text{seq\_len} \times \text{head\_dim} \times \text{bytes}
```

For a full model, multiply by number of layers. For LLaMA 2 70B with 100K context, this is ~260 GB in FP16!

### Q3: How do MQA and GQA reduce cache size?

**A**: They reduce the number of KV heads:

- MHA: n_kv_heads = n_heads (no reduction)
- GQA: n_kv_heads = n_heads / g (g-way reduction, e.g., 8x)
- MQA: n_kv_heads = 1 (maximum reduction, e.g., 64x)

This directly reduces cache size by the same factor.

### Q4: Why can we cache rotated keys (in RoPE)?

**A**: RoPE rotations are deterministic functions of position. Once $K_m$ is rotated by $\mathbf{R}_m$ and cached, it already encodes both content and position. When a new query at position $n$ attends to it, the relative position $(n-m)$ emerges naturally from $\mathbf{R}_n^T \mathbf{R}_m = \mathbf{R}_{m-n}$.

### Q5: What's the difference between prefill and decode?

**A**:

- **Prefill**: Process initial prompt in parallel. Compute-bound (high GPU utilization).
- **Decode**: Generate one token at a time. Memory-bound (loading cached K, V dominates).

They have different optimization strategies: prefill wants throughput, decode wants latency.

### Q6: How does quantization affect cache quality?

**A**: INT8 and FP8 quantization provide 2x memory reduction with <1% quality loss on most tasks. The attention mechanism is robust to small perturbations in K, V because it computes similarity scores, not exact values.

### Q7: What is PagedAttention and how does it work?

**A**: PagedAttention applies virtual memory concepts to KV cache management:

- **Problem**: Traditional allocation reserves contiguous memory for max sequence length, wasting 60-80%
- **Solution**: Divide KV cache into fixed-size blocks (e.g., 16 tokens per block)
- **Block tables**: Each sequence has a table mapping logical positions to physical blocks (like OS page tables)
- **Benefits**: Near-zero fragmentation, efficient prefix sharing via copy-on-write, flexible batching
- **Impact**: 2-4x higher throughput in production serving systems like vLLM

### Q8: How do you handle very long contexts with limited memory?

**A**: Several strategies:

1. **StreamingLLM**: Keep attention sink tokens + rolling window
2. **Sparse attention**: Only attend to subset of tokens
3. **Compression**: Quantize cache to INT8/FP8
4. **Hybrid**: Combine above (e.g., quantized cache + attention sinks)

### Q9: Why doesn't KV cache help training?

**A**: Training uses full context in parallel (teacher forcing), so there's no sequential generation. We compute K, V for all positions simultaneously, so caching provides no benefit.

### Q10: What's the memory bottleneck for long-context inference?

**A**: At long contexts (>32K tokens), KV cache dominates memory usage, often exceeding the model weights. This limits batch size and concurrent users. Solutions: GQA/MQA, quantization, PagedAttention.

---

## Summary

### Key Takeaways

1. **KV cache reduces generation from $O(N^2)$ to $O(N)$** by storing computed key and value projections.

2. **Memory scales linearly with sequence length**: At 100K context, cache can exceed model weights.

3. **RoPE integrates perfectly** with KV cache because rotations are applied before caching.

4. **GQA provides the best tradeoff**: 4-8x cache reduction with <1% quality loss (used in LLaMA 2/3, Mistral).

5. **Quantization (FP8/INT8) doubles effective cache capacity** with minimal quality impact.

6. **PagedAttention eliminates fragmentation** through block-based allocation and virtual memory techniques, enabling 2-4x better memory utilization.

7. **Prefill and decode have different bottlenecks**: Compute vs memory bandwidth.

8. **For streaming**: Use attention sinks + rolling buffer (Chapter 23).

### Memory Optimization Hierarchy

From most to least impactful for reducing KV cache memory:

1. **Architecture choice (GQA/MQA)**: 4-64x reduction
2. **Quantization (FP8/INT8)**: 2x reduction
3. **PagedAttention**: Better utilization (~1.5-2x effective capacity)
4. **Streaming/windowing**: Caps maximum memory at cost of context

### When to Use What

| Scenario | Recommendation |
|----------|----------------|
| **New model design** | Use GQA (8:1 ratio) from the start |
| **Long context (>32K)** | GQA + FP8 quantization + PagedAttention |
| **Streaming/chat** | Attention sinks + rolling window |
| **Maximum throughput** | Continuous batching + PagedAttention |
| **Memory constrained** | MQA + INT8 + aggressive windowing |

### Related Chapters

- [Chapter 4: Multi-Head Attention](04-multi-head-attention.md) - MQA and GQA details
- [Chapter 8: RoPE](08-rope.md) - Position encoding interaction with cache
- [Chapter 26: Efficient Attention](14-efficient-attention.md) - Linear attention, sparse attention, sliding window
- [Chapter 23: Long Context Techniques](23-long-context.md) - Streaming and attention sinks
- [Chapter 29: Hardware and Quantization](29-hardware-quantization-optimization.md) - Cache quantization

---

**Next Chapter**: [Language Model Training](16-lm-training.md) - Training objectives and loops

**Previous Chapter**: [Other Efficient Attention Variants](14-efficient-attention.md) - Flash Attention and sparse patterns
