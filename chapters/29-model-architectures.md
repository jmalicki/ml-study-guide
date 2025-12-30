# Chapter 29: Architecture Comparison - Modern LLMs

This chapter provides a comprehensive comparison of architectural choices across major Large Language Models. Understanding these differences is crucial for ML interviews, as it demonstrates knowledge of the practical trade-offs that shape production systems.

For each technique mentioned, we reference the relevant chapter where it is explained in detail.

## Table of Contents

1. [Overview](#overview)
2. [GPT Series (OpenAI)](#gpt-series-openai)
3. [Claude (Anthropic)](#claude-anthropic)
4. [Gemini (Google DeepMind)](#gemini-google-deepmind)
5. [LLaMA Series (Meta)](#llama-series-meta)
6. [Qwen Series (Alibaba)](#qwen-series-alibaba)
7. [Mistral and Mixtral](#mistral-and-mixtral)
8. [DeepSeek](#deepseek)
9. [Gemma (Google)](#gemma-google)
10. [WeDLM (Tencent)](#wedlm-tencent)
11. [Comprehensive Comparison Table](#comprehensive-comparison-table)
12. [Key Architectural Innovations Timeline](#key-architectural-innovations-timeline)

---

## Overview

Modern LLMs share a common foundation—the Transformer architecture (see [The Transformer Block](09-transformer-block.md))—but differ significantly in their specific implementations. The key architectural dimensions include:

| Dimension | Options | Trade-offs |
|-----------|---------|------------|
| **Attention Type** | MHA, MQA, GQA, MLA | Memory vs. quality |
| **Positional Encoding** | Learned, Sinusoidal, RoPE, ALiBi | Extrapolation vs. complexity |
| **Normalization** | LayerNorm, RMSNorm, Pre/Post-norm | Stability vs. compute |
| **Activation** | ReLU, GELU, SwiGLU | Expressiveness vs. parameters |
| **Architecture** | Dense, MoE | Efficiency vs. complexity |
| **Generation** | Autoregressive, Diffusion | Quality vs. speed |

---

## GPT Series (OpenAI)

### GPT-2 and GPT-3

The GPT series established the decoder-only autoregressive paradigm that dominates modern LLMs.

**Architecture:**
- **Attention**: Multi-Head Attention (MHA) (see [Multi-Head Attention](04-multi-head-attention.md))
- **Positional Encoding**: Learned absolute positional embeddings (see [Positional Encodings](07-positional-encodings.md))
- **Normalization**: Pre-LayerNorm (GPT-2 onward)
- **Activation**: GELU
- **Context Length**: 1024 (GPT-2), 2048 (GPT-3)

**Key Papers:**
- GPT-2: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019)
- GPT-3: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)

### GPT-4

OpenAI has not released official architectural details for GPT-4. The technical report explicitly states: *"Given both the competitive landscape and the safety implications of large-scale models like GPT-4, this report contains no further details about the architecture."*

**Rumored/Leaked Details** (unconfirmed):
- **Architecture**: Mixture of Experts (MoE) with ~8 experts
- **Parameters**: ~1.8 trillion total, ~220B per expert
- **Positional Encoding**: Likely RoPE (referenced in technical report bibliography)
- **Training Data**: ~13 trillion tokens

**Key Papers:**
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) (OpenAI, 2023)

```python
# GPT-2 style architecture (simplified)
import torch
import torch.nn as nn

class GPT2Block(nn.Module):
    """GPT-2 transformer block with pre-norm."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm with residual connections
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x
```

---

## Claude (Anthropic)

Anthropic has not published detailed architectural specifications for Claude models. The company focuses its publications on safety research rather than architecture papers.

**Known Details:**
- **Architecture**: Transformer-based, decoder-only
- **Context Length**: 200K tokens (standard), 1M tokens (preview for Claude 4/4.5)
- **Training**: RLHF + Constitutional AI (RLAIF)

**Model Family (as of 2025):**
| Model | Release | Notes |
|-------|---------|-------|
| Claude 3 Haiku | March 2024 | Fastest, smallest |
| Claude 3 Sonnet | March 2024 | Balanced |
| Claude 3 Opus | March 2024 | Most capable |
| Claude 3.5 Sonnet | June 2024 | Improved Sonnet |
| Claude Sonnet 4 | May 2025 | Current generation |
| Claude Opus 4.5 | January 2025 | Current flagship |

**Key Papers:**
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (Bai et al., 2022)
- [Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) (Anthropic, 2022)

**Training Approach:**
Claude uses a combination of:
1. **RLHF** (see [RLHF](20-rlhf.md)): Reinforcement Learning from Human Feedback
2. **RLAIF**: Reinforcement Learning from AI Feedback, where a "trainer" model evaluates responses against constitutional principles

---

## Gemini (Google DeepMind)

Gemini models use a Mixture-of-Experts architecture with native multimodality.

**Architecture:**
- **Architecture**: Sparse Mixture of Experts (MoE)
- **Multimodal**: Native vision, audio, and text processing
- **Context Length**: 1M+ tokens (Gemini 1.5 Pro and later)

**Model Evolution:**
| Version | Release | Key Features |
|---------|---------|--------------|
| Gemini 1.0 | Dec 2023 | Initial multimodal model |
| Gemini 1.5 | Feb 2024 | MoE, 1M context window |
| Gemini 2.0 | Dec 2024 | Real-time multimodal, tool use |
| Gemini 2.5 | Mar 2025 | Deep Think reasoning, 1M context |
| Gemini 3.0 | Nov 2025 | Enhanced reasoning, TPU-trained |

**Key Technical Features:**
- **MoE Architecture**: Selective expert activation for efficiency (see [Other Efficient Attention Variants](13-efficient-attention.md) for MoE concepts)
- **Long Context**: Uses techniques similar to RoPE scaling for extended context

**Key Papers:**
- [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) (Gemini Team, 2023)
- [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](https://arxiv.org/abs/2403.05530) (Gemini Team, 2024)

**Training Infrastructure:**
Google trains Gemini entirely on TPUs (Tensor Processing Units):
- TPU v5: Used for Gemini 1.0/1.5
- TPU v6e "Trillium": Used for Gemini 2.0+
- TPU v7 "Ironwood": Previewed for future models

---

## LLaMA Series (Meta)

The LLaMA series has become the foundation for much of the open-source LLM ecosystem.

### LLaMA 1 (February 2023)

Introduced key architectural improvements over GPT-3:

- **Normalization**: RMSNorm instead of LayerNorm (see [The Transformer Block](09-transformer-block.md))
- **Activation**: SwiGLU instead of GELU
- **Positional Encoding**: RoPE (see [Rotary Position Embeddings](08-rope.md))
- **Attention**: Standard MHA
- **No Bias**: Removed bias terms from linear layers

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Compared to LayerNorm, RMSNorm:
    - Removes mean centering (only normalizes by RMS)
    - Saves 5-15% compute per normalization layer
    - Maintains training stability

    See [The Transformer Block](09-transformer-block.md) for detailed explanation.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

```python
class SwiGLU(nn.Module):
    """SwiGLU activation function.

    SwiGLU(x) = Swish(xW) * (xV)

    Benefits over GELU:
    - Gating mechanism learns to selectively process information
    - No dead neuron problem (unlike ReLU)
    - Improved training performance

    Note: Requires 3 linear layers instead of 2 in FFN.

    See [The Transformer Block](09-transformer-block.md) for detailed explanation.
    """
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)  # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)  # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(x) = x * sigmoid(x)
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
```

**Key Paper:**
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023)

### LLaMA 2 (July 2023)

Key changes:
- **Attention**: Grouped Query Attention (GQA) for 34B and 70B models (see [Multi-Head Attention](04-multi-head-attention.md))
- **Context Length**: Extended to 4096 tokens
- **Training**: More data (2T tokens) + RLHF

```python
class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA).

    GQA is a middle ground between MHA and MQA:
    - MHA: Each query head has its own K,V heads (n_kv_heads = n_heads)
    - MQA: All query heads share one K,V head (n_kv_heads = 1)
    - GQA: Groups of query heads share K,V heads (1 < n_kv_heads < n_heads)

    Benefits:
    - Reduces KV cache memory by factor of (n_heads / n_kv_heads)
    - Faster inference with minimal quality loss

    See [Multi-Head Attention](04-multi-head-attention.md) for detailed explanation.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int = None
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = head_dim or dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Repeat K, V for each group
        # This expands n_kv_heads to n_heads by repeating
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)

        # Standard attention computation
        q = q.transpose(1, 2)  # (batch, n_heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out)
```

**Key Paper:**
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) (Touvron et al., 2023)

### LLaMA 3 (April 2024)

Key changes:
- **Attention**: GQA for ALL model sizes (including 8B)
- **Tokenizer**: New tokenizer with 128K vocabulary (vs 32K in LLaMA 2)
- **Context Length**: 8K tokens (extended to 128K in LLaMA 3.1)
- **Training**: 15T+ tokens

**Key Paper:**
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Llama Team, 2024)

### LLaMA 4 (April 2025)

Major architectural shift to Mixture of Experts:

| Model | Total Params | Active Params | Experts | Context |
|-------|-------------|---------------|---------|---------|
| Scout | 109B | 17B | 16 | 10M tokens |
| Maverick | 400B | 17B | 128 | 1M tokens |
| Behemoth | 2T | 288B | 16 | - |

**Key Innovations:**

1. **iRoPE Architecture**: Interleaved use of RoPE and NoPE (No Position Encoding) layers
   - NoPE layers every 4th layer for long-context handling
   - RoPE layers use chunked attention

2. **MoE Design**:
   - Alternating dense and MoE layers (Maverick)
   - Each token routed to 1 of N experts plus a shared expert

3. **Native Multimodality**: Built-in vision encoder

4. **Co-distillation**: Maverick distilled from Behemoth using dynamic loss weighting

**Key Paper:**
- [The Llama 4 Herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) (Meta AI, 2025)

---

## Qwen Series (Alibaba)

### Qwen 2.5

**Architecture:**
- **Attention**: GQA (28 Q heads, 4 KV heads for 7B model)
- **Positional Encoding**: RoPE with ABF (base frequency scaled to 1M)
- **Normalization**: RMSNorm with pre-norm
- **Activation**: SwiGLU
- **Context Length**: 128K tokens (extended to 1M with YARN + DCA)

**Model Specifications (7B):**
| Component | Value |
|-----------|-------|
| Layers | 28 |
| Hidden dim | 3584 |
| Q heads | 28 |
| KV heads | 4 |
| Vocabulary | 152K |

**Key Paper:**
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) (Qwen Team, 2024)

### Qwen 3

Key changes from Qwen 2.5:
- **Attention**: Removed QKV-bias, added QK-Norm for training stability
- **Context Length**: 32K (small models) to 128K (large models)
- **MoE Variants**: 128 total experts, 8 activated per token
- **Languages**: Expanded from 29 to 119 languages

**Key Paper:**
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (Qwen Team, 2025)

```python
class QKNorm(nn.Module):
    """QK-Norm: Normalize Q and K before attention.

    Added in Qwen3 to stabilize training at scale.
    Applied after projection, before attention computation.
    """
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)
```

---

## Mistral and Mixtral

### Mistral 7B

Introduced sliding window attention for efficient long-context handling:

**Architecture:**
- **Attention**: GQA + Sliding Window Attention (SWA) (see [Other Efficient Attention Variants](13-efficient-attention.md))
- **Window Size**: 4096 tokens
- **Positional Encoding**: RoPE
- **Normalization**: RMSNorm
- **Activation**: SiLU (similar to Swish)
- **Context Length**: 8K (effective attention span ~128K via stacking)

```python
def sliding_window_attention_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """Create sliding window attention mask.

    Each position can only attend to the previous `window_size` positions.
    This reduces memory from O(n^2) to O(n * window_size).

    Key insight: Due to layer stacking, information can propagate
    beyond the window size. At layer k, a token can effectively
    access information from k * window_size positions back.

    See [Other Efficient Attention Variants](13-efficient-attention.md) for detailed explanation.
    """
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, :start] = False
        mask[i, i+1:] = False  # Causal: can't see future
    return mask
```

**Rolling Buffer Cache:**
Instead of storing the full KV cache, Mistral uses a fixed-size rotating buffer:

```python
class RollingKVCache:
    """Rolling buffer for KV cache with sliding window.

    Saves 50% memory for sequences of length 2 * window_size.
    """
    def __init__(self, window_size: int, n_heads: int, head_dim: int):
        self.window_size = window_size
        self.cache_k = torch.zeros(1, window_size, n_heads, head_dim)
        self.cache_v = torch.zeros(1, window_size, n_heads, head_dim)
        self.position = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple:
        seq_len = k.shape[1]
        for i in range(seq_len):
            idx = (self.position + i) % self.window_size
            self.cache_k[:, idx] = k[:, i]
            self.cache_v[:, idx] = v[:, i]
        self.position = (self.position + seq_len) % self.window_size
        return self.cache_k, self.cache_v
```

**Key Paper:**
- [Mistral 7B](https://arxiv.org/abs/2310.06825) (Jiang et al., 2023)

### Mixtral 8x7B

Sparse Mixture of Experts model:

**Architecture:**
- **MoE Configuration**: 8 experts, 2 active per token
- **Total Parameters**: 47B
- **Active Parameters**: 13B per forward pass
- **Attention**: GQA + Sliding Window
- **Context Length**: 32K tokens

```python
class MixtralMoELayer(nn.Module):
    """Sparse Mixture of Experts layer (Mixtral style).

    Each token is routed to top-k experts (k=2 in Mixtral).
    This gives 8x capacity with only 2x compute.

    Architecture:
    1. Router computes expert scores
    2. Top-k experts selected per token
    3. Experts process token in parallel
    4. Outputs weighted by router scores
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_experts: int = 8,
        top_k: int = 2
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router: learns which experts to use
        self.router = nn.Linear(dim, n_experts, bias=False)

        # Each expert is a SwiGLU FFN
        self.experts = nn.ModuleList([
            SwiGLU(dim, hidden_dim) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (batch * seq, dim)

        # Compute router logits and select top-k experts
        router_logits = self.router(x_flat)  # (batch * seq, n_experts)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = torch.softmax(top_k_logits, dim=-1)

        # Compute expert outputs (simplified - real impl uses sparse ops)
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x_flat[mask])
                # Weight by router probability
                weight_idx = (top_k_indices[mask] == i).float()
                weights = (top_k_weights[mask] * weight_idx).sum(dim=-1, keepdim=True)
                output[mask] += expert_out * weights

        return output.view(batch, seq_len, dim)
```

**Key Paper:**
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Jiang et al., 2024)

---

## DeepSeek

### DeepSeek V2/V3

DeepSeek introduced Multi-head Latent Attention (MLA), a novel attention mechanism that compresses KV cache.

**Architecture:**
- **Attention**: Multi-head Latent Attention (MLA)
- **MoE**: Fine-grained experts (256 in V3, 8 active)
- **Parameters**: 671B total, 37B active per token
- **Context Length**: 128K tokens
- **Training**: FP8 mixed precision

**Multi-head Latent Attention (MLA):**
MLA compresses K and V into a lower-dimensional latent space before caching:

```python
class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention (MLA) from DeepSeek.

    Key insight: Instead of caching full K,V tensors, compress them
    into a smaller latent space. At inference, decompress on-the-fly.

    Memory savings: KV cache reduced by factor of (dim / latent_dim)

    Trade-off: Slightly more compute during inference for decompression,
    but significantly reduced memory bandwidth (often the bottleneck).
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        latent_dim: int,  # Compressed dimension
        head_dim: int = None
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim or dim // n_heads
        self.latent_dim = latent_dim

        # Query projection (standard)
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)

        # KV compression: project to latent space
        self.kv_compress = nn.Linear(dim, latent_dim, bias=False)

        # KV decompression: project from latent to K and V
        self.k_decompress = nn.Linear(latent_dim, n_heads * self.head_dim, bias=False)
        self.v_decompress = nn.Linear(latent_dim, n_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cached_latent: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape

        # Standard query projection
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)

        # Compress KV to latent space (this is what we cache!)
        latent = self.kv_compress(x)  # (batch, seq, latent_dim)

        # Concatenate with cached latent if provided
        if cached_latent is not None:
            latent = torch.cat([cached_latent, latent], dim=1)

        # Decompress K and V from latent
        k = self.k_decompress(latent).view(batch, -1, self.n_heads, self.head_dim)
        v = self.v_decompress(latent).view(batch, -1, self.n_heads, self.head_dim)

        # Standard attention (simplified)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(out), latent  # Return latent for caching
```

**Auxiliary-Loss-Free Load Balancing:**
DeepSeek V3 eliminates auxiliary losses for MoE load balancing, using bias terms instead:

```python
class DeepSeekRouter(nn.Module):
    """DeepSeek V3 router with auxiliary-loss-free load balancing.

    Previous approaches (V2) used auxiliary losses to prevent
    routing collapse, but these hurt model quality.

    V3 solution: Add learnable bias terms to routing scores.
    Bias is used for routing decisions but not in final loss.
    """
    def __init__(self, dim: int, n_experts: int, top_k: int):
        super().__init__()
        self.router = nn.Linear(dim, n_experts, bias=False)
        # Learnable bias for load balancing (not included in loss)
        self.expert_bias = nn.Parameter(torch.zeros(n_experts))
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        # Add bias for routing decision only
        routing_logits = logits + self.expert_bias
        top_k_logits, top_k_indices = torch.topk(routing_logits, self.top_k, dim=-1)
        # Use original logits (without bias) for weights
        original_top_k = torch.gather(logits, -1, top_k_indices)
        weights = torch.softmax(original_top_k, dim=-1)
        return weights, top_k_indices
```

**Key Papers:**
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024)

---

## Gemma (Google)

### Gemma 2

Gemma 2 uses interleaved local (sliding window) and global attention:

**Architecture:**
- **Attention**: GQA with interleaved local/global attention
- **Local Window**: 4096 tokens (every other layer)
- **Global Span**: 8192 tokens (alternating layers)
- **Positional Encoding**: RoPE
- **Normalization**: RMSNorm (both pre and post sub-layer)
- **Context Length**: 8192 tokens

**Key Innovation - Interleaved Attention:**
```python
class Gemma2Attention(nn.Module):
    """Gemma 2 attention with interleaved local/global attention.

    Odd layers: Local sliding window attention (4K window)
    Even layers: Global full attention (8K span)

    Benefits:
    - Global layers maintain long-range dependencies
    - Local layers are more efficient
    - Combined: quality of global + efficiency of local
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        layer_idx: int,
        local_window: int = 4096,
        global_span: int = 8192
    ):
        super().__init__()
        self.is_local = (layer_idx % 2 == 1)
        self.window = local_window if self.is_local else global_span
        # ... rest of attention implementation

    def get_attention_mask(self, seq_len: int) -> torch.Tensor:
        if self.is_local:
            return sliding_window_attention_mask(seq_len, self.window)
        else:
            # Global causal mask
            return torch.tril(torch.ones(seq_len, seq_len))
```

**Logit Soft-Capping:**
Gemma 2 caps attention logits to prevent numerical instability:

```python
def soft_cap(logits: torch.Tensor, cap: float = 50.0) -> torch.Tensor:
    """Soft-cap logits to prevent extreme values.

    Used in Gemma 2 for both attention logits and final logits.
    Prevents logits from growing excessively during training.
    """
    return cap * torch.tanh(logits / cap)
```

**Key Paper:**
- [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) (Gemma Team, 2024)

---

## WeDLM (Tencent)

WeDLM represents a paradigm shift: a **diffusion language model** that uses causal attention.

### Key Innovation

Most diffusion language models use bidirectional attention, which breaks KV cache compatibility. WeDLM uses standard causal attention while still performing parallel token generation.

**Architecture:**
- **Generation**: Diffusion-based (not autoregressive)
- **Attention**: Causal attention (compatible with KV cache)
- **Base Model**: Initialized from Qwen2.5-7B
- **Inference**: Parallel mask recovery

**How It Works:**

1. Start with masked/noisy token sequence
2. Predict multiple tokens in parallel (unlike AR which predicts one)
3. Use causal attention (unlike other diffusion LMs which use bidirectional)
4. Iterate to refine predictions

```python
class DiffusionLanguageModel(nn.Module):
    """Simplified diffusion language model concept.

    Unlike autoregressive models that generate one token at a time,
    diffusion models start with noise/masks and iteratively denoise.

    WeDLM's key insight: Use causal attention so that KV cache,
    FlashAttention (see [Hardware, Quantization, and Training Optimization](31-hardware-quantization-optimization.md)),
    and other optimizations still work.

    See [Diffusion Model Fundamentals](23-diffusion-fundamentals.md) for diffusion model details.
    """
    def __init__(self, base_model: nn.Module, n_diffusion_steps: int = 10):
        super().__init__()
        self.model = base_model
        self.n_steps = n_diffusion_steps

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Predict denoised tokens at diffusion step t.

        Args:
            x: Current noisy/masked sequence
            t: Current diffusion timestep (n_steps -> 0)

        Returns:
            Predicted clean tokens
        """
        # Embed timestep and add to input
        # Use causal attention (standard transformer)
        # Predict token logits for all positions
        pass

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, length: int) -> torch.Tensor:
        """Generate tokens using iterative denoising.

        Unlike AR generation (one token at a time),
        we predict all tokens simultaneously and refine.
        """
        # Start with masked tokens
        x = torch.full((1, length), MASK_TOKEN)
        x[:, :prompt.shape[1]] = prompt

        # Iteratively denoise
        for t in range(self.n_steps, 0, -1):
            logits = self.forward(x, t)
            # Sample or take argmax for non-prompt positions
            predictions = logits[:, prompt.shape[1]:].argmax(dim=-1)
            x[:, prompt.shape[1]:] = predictions

        return x
```

**Performance:**
- 3-6x faster than vLLM-optimized Qwen3-8B on math/code tasks
- Largest gains on structured, low-entropy tasks

**Key Resources:**
- [WeDLM GitHub](https://github.com/Tencent/WeDLM)
- [WeDLM on Hugging Face](https://huggingface.co/tencent/WeDLM-8B-Instruct)

---

## Comprehensive Comparison Table

### Attention Mechanisms

| Model | Attention Type | KV Heads (7-8B) | Notes |
|-------|---------------|-----------------|-------|
| GPT-2/3 | MHA | n_heads | Standard multi-head |
| GPT-4 | Unknown | Unknown | Likely MoE |
| Claude | Unknown | Unknown | Not published |
| Gemini | Unknown | Unknown | MoE architecture |
| LLaMA 1 | MHA | n_heads | - |
| LLaMA 2 | GQA (34B+) | 8 (70B) | MHA for smaller |
| LLaMA 3/4 | GQA | 8 | All sizes |
| Qwen 2.5 | GQA | 4 (7B) | With QKV bias |
| Qwen 3 | GQA | 4 (7B) | QK-Norm, no bias |
| Mistral | GQA + SWA | 8 | 4K window |
| Mixtral | GQA + SWA | 8 | MoE + sliding |
| DeepSeek V3 | MLA | Latent | Compressed KV |
| Gemma 2 | GQA | 4 (9B) | Interleaved local/global |

### Positional Encoding

| Model | Position Encoding | Max Pretrain Context | Notes |
|-------|-------------------|---------------------|-------|
| GPT-2 | Learned Absolute | 1024 | Fixed positions |
| GPT-3 | Learned Absolute | 2048 | Fixed positions |
| LLaMA 1-4 | RoPE | 8K-256K | ABF scaling |
| Qwen 2.5/3 | RoPE | 32K-128K | ABF + YARN |
| Mistral/Mixtral | RoPE | 8K-32K | - |
| DeepSeek V3 | RoPE | 128K | - |
| Gemma 2 | RoPE | 8K | - |
| LLaMA 4 | iRoPE | 256K | RoPE + NoPE interleaved |

### Normalization and Activation

| Model | Normalization | Activation | FFN Ratio |
|-------|--------------|------------|-----------|
| GPT-2/3 | Pre-LayerNorm | GELU | 4x |
| LLaMA 1-4 | RMSNorm | SwiGLU | 8/3x |
| Qwen 2.5/3 | RMSNorm | SwiGLU | 8/3x |
| Mistral/Mixtral | RMSNorm | SiLU | 8/3x |
| DeepSeek V3 | RMSNorm | SwiGLU | varies |
| Gemma 2 | RMSNorm (pre+post) | GeGLU | 8/3x |

### Architecture Type

| Model | Architecture | Total Params | Active Params | Experts |
|-------|-------------|--------------|---------------|---------|
| GPT-3 | Dense | 175B | 175B | - |
| GPT-4 | MoE (rumored) | ~1.8T | ~220B | 8 |
| Gemini 1.5+ | MoE | Unknown | Unknown | Unknown |
| LLaMA 1-3 | Dense | 7B-405B | All | - |
| LLaMA 4 Scout | MoE | 109B | 17B | 16 |
| LLaMA 4 Maverick | MoE | 400B | 17B | 128 |
| Qwen 2.5 | Dense | 0.5B-72B | All | - |
| Qwen 3 MoE | MoE | 235B | ~30B | 128/8 |
| Mixtral 8x7B | MoE | 47B | 13B | 8/2 |
| DeepSeek V3 | MoE | 671B | 37B | 256/8 |

### Generation Paradigm

| Model | Generation | Decoding |
|-------|-----------|----------|
| GPT, LLaMA, Qwen, etc. | Autoregressive | One token at a time |
| WeDLM | Diffusion | Multiple tokens in parallel |

---

## Key Architectural Innovations Timeline

```
2017: Transformer (Vaswani et al.)
      └── Attention Is All You Need

2018: GPT-1 (OpenAI)
      └── Decoder-only, autoregressive

2019: GPT-2
      └── Pre-norm, larger scale

2020: GPT-3
      └── Few-shot learning, 175B parameters

2021: RoPE (Su et al.)
      └── Rotary Position Embeddings

2022: Chinchilla (DeepMind)
      └── Optimal compute allocation, training tokens matter

2023: LLaMA 1 (Meta)
      ├── RMSNorm
      ├── SwiGLU
      └── RoPE

2023: Mistral 7B
      ├── Sliding Window Attention
      └── Rolling Buffer Cache

2023: GPT-4 (OpenAI)
      └── Rumored MoE, multimodal

2024: Mixtral 8x7B
      └── Sparse MoE for open models

2024: LLaMA 2/3
      └── GQA for inference efficiency

2024: DeepSeek V2/V3
      ├── Multi-head Latent Attention (MLA)
      └── Fine-grained MoE

2024: Gemma 2
      └── Interleaved local/global attention

2024: Gemini 1.5
      └── 1M+ token context

2025: LLaMA 4
      ├── iRoPE (interleaved RoPE/NoPE)
      ├── Native multimodal
      └── MoE for open models

2025: Qwen 3
      └── QK-Norm for training stability

2025: WeDLM
      └── Diffusion LM with causal attention
```

---

## Summary

### Key Takeaways for Interviews

1. **Attention Evolution**: MHA → MQA → GQA → MLA
   - Trade-off: Memory efficiency vs. model quality
   - GQA is now standard for most models

2. **Position Encoding**: Learned → Sinusoidal → RoPE → iRoPE
   - RoPE dominates due to extrapolation ability
   - iRoPE (LLaMA 4) extends to 10M+ tokens

3. **Normalization**: Post-norm → Pre-norm → RMSNorm
   - Pre-RMSNorm is now standard
   - Simpler, faster, equally effective

4. **Activation**: ReLU → GELU → SwiGLU
   - SwiGLU adds parameters but improves quality
   - Most modern models use SwiGLU or variants

5. **Architecture**: Dense → Sparse MoE
   - MoE enables scaling with constant inference cost
   - Key frontier models (GPT-4, Gemini, LLaMA 4) use MoE

6. **New Paradigms**: Autoregressive → Diffusion (WeDLM)
   - Potential for faster generation
   - Active research area

### What to Know for Each Model

- **GPT**: Established the paradigm, details not published
- **Claude**: Focus on safety techniques (Constitutional AI)
- **Gemini**: MoE, multimodal, very long context
- **LLaMA**: Open weights, clean architecture, good baselines
- **Qwen**: Competitive open models, good documentation
- **Mistral**: Efficient attention (SWA), open MoE
- **DeepSeek**: MLA innovation, cost-efficient training
- **Gemma**: Google's open alternative, novel techniques

---

## References

### Key Papers

1. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. Radford et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
3. Brown et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)
4. Su et al. (2021). [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
5. Shazeer (2020). [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
6. Zhang & Sennrich (2019). [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
7. Ainslie et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
8. Touvron et al. (2023). [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
9. Touvron et al. (2023). [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
10. Llama Team (2024). [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
11. Meta AI (2025). [The Llama 4 Herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
12. Jiang et al. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825)
13. Jiang et al. (2024). [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
14. DeepSeek-AI (2024). [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
15. DeepSeek-AI (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
16. Qwen Team (2024). [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
17. Qwen Team (2025). [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
18. Gemini Team (2023). [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805)
19. Gemini Team (2024). [Gemini 1.5](https://arxiv.org/abs/2403.05530)
20. Gemma Team (2024). [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118)
21. Bai et al. (2022). [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
22. OpenAI (2023). [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

### Additional Resources

- [WeDLM GitHub](https://github.com/Tencent/WeDLM)
- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) - Sebastian Raschka
- [A Technical Tour of the DeepSeek Models](https://magazine.sebastianraschka.com/p/technical-deepseek) - Sebastian Raschka

---

## Exercises

1. **Implement GQA**: Modify the MHA implementation to support arbitrary numbers of KV heads.

2. **Compare Memory Usage**: Calculate the KV cache memory requirements for MHA, GQA (with 8 KV heads), MQA, and MLA for a 70B parameter model with 80 layers, 64 heads, and 128 head dimension at 100K context length.

3. **Sliding Window Trade-offs**: Explain why sliding window attention can still capture long-range dependencies despite the limited window. What is the effective receptive field after N layers?

4. **MoE Load Balancing**: Why do MoE models need load balancing? What happens if all tokens are routed to the same expert? Compare auxiliary loss approaches vs. DeepSeek's bias approach.

5. **Architecture Design**: If you were designing a new LLM today, which architectural choices would you make and why? Consider:
   - Attention type (MHA/GQA/MLA)
   - Positional encoding
   - Normalization
   - Dense vs. MoE
   - Target context length
