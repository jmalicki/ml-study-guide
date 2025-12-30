# Chapter 26: Long Context Techniques

Extending the context window of Large Language Models is one of the most active areas of research. While early models like GPT-2 were limited to 1024 tokens, modern LLMs can handle 100K+ tokens, with some models like Gemini 1.5 Pro supporting over 1 million tokens. This chapter explores the techniques that make long-context modeling possible.

## Table of Contents

1. [Introduction: The Challenge of Long Context](#introduction-the-challenge-of-long-context)
2. [RoPE Scaling Methods](#rope-scaling-methods)
3. [Attention Sinks and StreamingLLM](#attention-sinks-and-streamingllm)
4. [Memory-Augmented Architectures](#memory-augmented-architectures)
5. [Landmark Attention](#landmark-attention)
6. [Ring Attention for Distributed Long Context](#ring-attention-for-distributed-long-context)
7. [Evaluation on Long-Range Tasks](#evaluation-on-long-range-tasks)
8. [Complete Implementation: Long Context Transformer](#complete-implementation-long-context-transformer)
9. [Summary and Best Practices](#summary-and-best-practices)
10. [Exercises](#exercises)

---

## Introduction: The Challenge of Long Context

### Why Long Context Matters

Long context capabilities enable models to:
- Process entire books or codebases in a single forward pass
- Maintain coherent conversations over many turns
- Retrieve information from large documents
- Understand complex dependencies across long sequences

### Computational Challenges

For a sequence of length $n$, standard attention has:
- **Time complexity**: $O(n^2 d)$ where $d$ is the hidden dimension
- **Memory complexity**: $O(n^2)$ for attention scores + $O(n d)$ for KV cache
- **Position encoding**: May not extrapolate beyond training length

For $n = 100,000$ tokens, the attention matrix alone requires ~40GB of memory in FP32!

### Three Approaches to Long Context

1. **Position Encoding Extension**: Modify positional embeddings to extrapolate
2. **Attention Efficiency**: Reduce computational/memory complexity
3. **Architecture Modifications**: Fundamental changes to how information flows

This chapter covers techniques across all three categories.

---

## RoPE Scaling Methods

Rotary Position Embeddings (RoPE) (see [Rotary Position Embeddings](08-rope.md)) are the dominant positional encoding method in modern LLMs. However, they struggle to extrapolate beyond their training length due to frequency-based encoding.

### The RoPE Extrapolation Problem

Recall that RoPE applies rotation to query and key vectors:

$$
\mathbf{q}_m = \mathbf{R}_m \mathbf{q}, \quad \mathbf{k}_n = \mathbf{R}_n \mathbf{k}
$$

where $\mathbf{R}_m$ is a rotation matrix dependent on position $m$ and base frequencies $\theta_i = 10000^{-2i/d}$.

**Problem**: When inference positions exceed training positions, the model sees rotation angles it was never trained on, leading to degraded performance.

### Linear Scaling (Naive Approach)

The simplest approach: scale positions linearly.

$$
\mathbf{R}_m' = \mathbf{R}_{m/s}
$$

where $s$ is the scaling factor. If trained on 2K context and want 8K, use $s = 4$.

**Issues**:
- Compresses all positions into trained range
- Changes relative distances between tokens
- Often requires fine-tuning to recover performance

```python
import torch
import torch.nn as nn
import math

class LinearScalingRoPE(nn.Module):
    """RoPE with linear position interpolation.

    Instead of extrapolating to new positions, we interpolate
    by scaling positions down to the training range.

    Example: Trained on 2048 positions, want 8192.
    Position 8000 -> 8000/4 = 2000 (within training range)
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (not used, just for shape)
            seq_len: Sequence length

        Returns:
            cos, sin: Cosine and sine of rotation angles [seq_len, dim]
        """
        # Scale positions to fit in training range
        positions = torch.arange(seq_len, device=x.device).float()
        positions = positions / self.scaling_factor

        # Compute angles: outer product of positions and frequencies
        freqs = torch.outer(positions, self.inv_freq)  # [seq_len, dim//2]

        # Create full frequency tensor [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos(), emb.sin()


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        q: Query tensor [batch, seq_len, n_heads, head_dim]
        k: Key tensor [batch, seq_len, n_heads, head_dim]
        cos, sin: Precomputed cos/sin [seq_len, head_dim]

    Returns:
        Rotated q, k tensors
    """
    # Reshape for rotation
    def rotate_half(x):
        """Split and swap for rotation."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    # Apply rotation
    q_embed = q * cos.unsqueeze(0).unsqueeze(2) + rotate_half(q) * sin.unsqueeze(0).unsqueeze(2)
    k_embed = k * cos.unsqueeze(0).unsqueeze(2) + rotate_half(k) * sin.unsqueeze(0).unsqueeze(2)

    return q_embed, k_embed
```

### NTK-Aware Scaling

**Neural Tangent Kernel (NTK) scaling** modifies the base frequency instead of positions.

**Key insight**: Instead of compressing positions, expand the wavelengths of the sinusoidal functions.

$$
\theta_i' = \theta_i \cdot s^{d/(d-2)} = 10000^{-2i/d} \cdot s^{d/(d-2)}
$$

where $s$ is the target scaling factor.

**Why it works**:
- Preserves relative position information better
- Smoothly extends to longer contexts
- Often works **without fine-tuning**

```python
class NTKScalingRoPE(nn.Module):
    """RoPE with NTK-aware scaling.

    Instead of scaling positions, we scale the base frequency.
    This changes the wavelengths of the rotation functions.

    Paper: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor

        # NTK scaling: scale the base frequency
        # Formula: base' = base * scaling_factor^(dim / (dim - 2))
        ntk_base = base * (scaling_factor ** (dim / (dim - 2)))

        # Compute inverse frequencies with scaled base
        inv_freq = 1.0 / (ntk_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
```

### Dynamic NTK Scaling

**Problem with static NTK**: Scaling is fixed at model load time.

**Dynamic NTK** adjusts the scaling based on actual sequence length:

$$
\alpha(L) = \begin{cases}
1 & \text{if } L \leq L_{\text{train}} \\
\left(\frac{L}{L_{\text{train}}}\right)^{d/(d-2)} & \text{otherwise}
\end{cases}
$$

Then use base frequency: $\theta_i' = \theta_i \cdot \alpha(L)$

```python
class DynamicNTKScalingRoPE(nn.Module):
    """Dynamic NTK scaling that adapts to sequence length.

    Key advantage: Scales only when needed, preserving original
    behavior for sequences within training length.
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Store original inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq_base", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute dynamic scaling factor
        if seq_len > self.max_position_embeddings:
            scale = (seq_len / self.max_position_embeddings) ** (self.dim / (self.dim - 2))
            inv_freq = self.inv_freq_base / scale
        else:
            inv_freq = self.inv_freq_base

        positions = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
```

### YaRN: Yet another RoPE extensioN

YaRN combines multiple techniques for optimal long-context performance:

1. **NTK-by-parts**: Different scaling for high/low frequency components
2. **Attention temperature**: Scale attention scores during fine-tuning
3. **Targeted fine-tuning**: Only fine-tune on long sequences

**Key Paper**: [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) (Peng et al., 2023)

**Frequency-dependent scaling**:

$$
\theta_i' = \begin{cases}
\theta_i & \text{if } i < i_{\text{low}} \\
\theta_i \cdot s^{(i - i_{\text{low}})/(i_{\text{high}} - i_{\text{low}})} & \text{if } i_{\text{low}} \leq i < i_{\text{high}} \\
\theta_i \cdot s & \text{if } i \geq i_{\text{high}}
\end{cases}
$$

where:
- $i_{\text{low}}$, $i_{\text{high}}$ are frequency band boundaries
- $s$ is the scaling factor
- Low frequencies (encoding long-range info) are scaled more
- High frequencies (encoding local info) are scaled less

```python
class YaRNScalingRoPE(nn.Module):
    """YaRN (Yet another RoPE extensioN) scaling.

    Applies different scaling to different frequency bands:
    - Low frequencies (long-range): More scaling
    - High frequencies (local): Less scaling

    Paper: https://arxiv.org/abs/2309.00071
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 2048,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings

        # Compute frequency bands
        # Low freq indices: scale more (capture long-range)
        # High freq indices: scale less (capture local patterns)
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale

        # Get inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # Compute wavelengths
        wavelengths = 2 * math.pi / inv_freq

        # Determine scaling per frequency
        freq_scales = torch.ones_like(inv_freq)
        for i, wavelength in enumerate(wavelengths):
            if wavelength < beta_fast:
                # High frequency (short wavelength): no scaling
                freq_scales[i] = 1.0
            elif wavelength > beta_slow * scaling_factor:
                # Low frequency (long wavelength): full scaling
                freq_scales[i] = scaling_factor
            else:
                # Interpolate between no scaling and full scaling
                # based on wavelength
                ratio = (wavelength - beta_fast) / (beta_slow * scaling_factor - beta_fast)
                freq_scales[i] = 1.0 + (scaling_factor - 1.0) * ratio

        # Apply frequency-dependent scaling
        inv_freq_scaled = inv_freq / freq_scales
        self.register_buffer("inv_freq", inv_freq_scaled)

        # mscale: attention entropy preservation
        self.mscale_factor = self._compute_mscale()

    def _compute_mscale(self) -> float:
        """Compute mscale to preserve attention entropy.

        YaRN uses this to prevent attention from becoming too peaked
        or too uniform when extending context.
        """
        if self.scaling_factor <= 1.0:
            return 1.0

        # Formula from YaRN paper
        return 0.1 * math.log(self.scaling_factor) + 1.0

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Apply mscale to attention (done in attention computation)
        return emb.cos(), emb.sin()
```

### ABF (Adjusted Base Frequency)

Used in models like Qwen, ABF adjusts the base frequency (typically from 10000 to 1000000) to support longer contexts.

**Simple formula**:

$$
\text{base}_{\text{new}} = \text{base}_{\text{old}} \times \left(\frac{L_{\text{target}}}{L_{\text{original}}}\right)^{d/(d-2)}
$$

This is essentially NTK scaling with a larger base adjustment.

```python
class ABFScalingRoPE(nn.Module):
    """Adjusted Base Frequency (ABF) RoPE scaling.

    Used in Qwen models to extend from 32K to 128K+ context.
    Essentially NTK scaling with aggressive base frequency adjustment.
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        target_max_position: int = 131072,  # 128K
    ):
        super().__init__()
        self.dim = dim

        # Compute scaling factor
        scaling_factor = target_max_position / max_position_embeddings

        # Apply aggressive base frequency adjustment
        # For Qwen: 10000 -> 1000000 (100x increase)
        adjusted_base = base * (scaling_factor ** (dim / (dim - 2)))

        inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
```

### Comparison of RoPE Scaling Methods

| Method | Fine-tuning Required? | Strengths | Weaknesses |
|--------|----------------------|-----------|------------|
| Linear | Yes | Simple | Distorts relative positions |
| NTK | No (often) | Preserves relative positions | May degrade at very long contexts |
| Dynamic NTK | No | Adaptive to length | Slight overhead |
| YaRN | Yes (minimal) | Best performance | More complex |
| ABF | Yes | Used in production (Qwen) | Requires retraining |

**Best practices**:
- For quick extension without training: **Dynamic NTK**
- For production deployment: **YaRN** with short fine-tuning
- For new model training: **ABF** with long context from start

---

## Attention Sinks and StreamingLLM

### The Attention Sink Phenomenon

**Surprising discovery**: In causal language models, the **first token** receives disproportionately high attention scores, even when semantically irrelevant.

**Why?** Softmax must sum to 1. When no token is particularly relevant, attention "leaks" to early tokens, especially the first.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

For position $i$, if all keys are equally (ir)relevant:

$$
\text{score}_{i,j} \approx 0 \text{ for all } j \Rightarrow \text{softmax needs a "sink"}
$$

The first token becomes this sink.

**Key Paper**: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (Xiao et al., 2023)

### StreamingLLM

StreamingLLM enables LLMs to handle **infinite-length** sequences by:

1. **Keep attention sink tokens**: Preserve first few tokens in KV cache
2. **Use rolling KV cache**: Keep only recent tokens
3. **Discard middle tokens**: Remove old, irrelevant history

**Algorithm**:
- Cache size: $N$ tokens
- Keep first $k$ tokens (attention sinks)
- Keep most recent $N - k$ tokens
- Discard everything in between

```python
class StreamingLLMCache:
    """Streaming KV cache with attention sinks.

    Maintains:
    - First k tokens (attention sinks)
    - Most recent (cache_size - k) tokens
    - Discards everything in between

    This enables infinite-length streaming while keeping memory constant.

    Paper: https://arxiv.org/abs/2309.17453
    """
    def __init__(
        self,
        cache_size: int = 2048,
        n_sink_tokens: int = 4,
        n_layers: int = 32,
        n_heads: int = 32,
        head_dim: int = 128,
        device: str = "cuda"
    ):
        self.cache_size = cache_size
        self.n_sink_tokens = n_sink_tokens
        self.recent_size = cache_size - n_sink_tokens
        self.n_layers = n_layers

        # Initialize cache for each layer
        # Separate storage for sink tokens and recent tokens
        self.sink_k = [
            torch.zeros(1, n_sink_tokens, n_heads, head_dim, device=device)
            for _ in range(n_layers)
        ]
        self.sink_v = [
            torch.zeros(1, n_sink_tokens, n_heads, head_dim, device=device)
            for _ in range(n_layers)
        ]

        # Rolling buffer for recent tokens
        self.recent_k = [
            torch.zeros(1, self.recent_size, n_heads, head_dim, device=device)
            for _ in range(n_layers)
        ]
        self.recent_v = [
            torch.zeros(1, self.recent_size, n_heads, head_dim, device=device)
            for _ in range(n_layers)
        ]

        # Track number of tokens seen
        self.n_seen = 0
        # Position in rolling buffer
        self.recent_position = 0

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new K, V tensors.

        Args:
            layer_idx: Which transformer layer
            k, v: New key/value tensors [batch, seq_len, n_heads, head_dim]

        Returns:
            Full K, V to use for attention (includes sink + recent)
        """
        seq_len = k.shape[1]

        for i in range(seq_len):
            pos = self.n_seen + i

            if pos < self.n_sink_tokens:
                # Store in sink cache
                self.sink_k[layer_idx][:, pos] = k[:, i]
                self.sink_v[layer_idx][:, pos] = v[:, i]
            else:
                # Store in rolling recent cache
                idx = self.recent_position % self.recent_size
                self.recent_k[layer_idx][:, idx] = k[:, i]
                self.recent_v[layer_idx][:, idx] = v[:, i]
                self.recent_position += 1

        self.n_seen += seq_len

        # Return combined cache for attention
        if self.n_seen <= self.cache_size:
            # Haven't filled cache yet, return everything
            k_combined = torch.cat([
                self.sink_k[layer_idx][:, :min(self.n_seen, self.n_sink_tokens)],
                self.recent_k[layer_idx][:, :max(0, self.n_seen - self.n_sink_tokens)]
            ], dim=1)
            v_combined = torch.cat([
                self.sink_v[layer_idx][:, :min(self.n_seen, self.n_sink_tokens)],
                self.recent_v[layer_idx][:, :max(0, self.n_seen - self.n_sink_tokens)]
            ], dim=1)
        else:
            # Cache is full, return sink + recent (in correct order)
            # Need to reorder rolling buffer to be chronological
            start_idx = self.recent_position % self.recent_size
            k_recent = torch.cat([
                self.recent_k[layer_idx][:, start_idx:],
                self.recent_k[layer_idx][:, :start_idx]
            ], dim=1)
            v_recent = torch.cat([
                self.recent_v[layer_idx][:, start_idx:],
                self.recent_v[layer_idx][:, :start_idx]
            ], dim=1)

            k_combined = torch.cat([self.sink_k[layer_idx], k_recent], dim=1)
            v_combined = torch.cat([self.sink_v[layer_idx], v_recent], dim=1)

        return k_combined, v_combined

    def reset(self):
        """Reset cache for new sequence."""
        self.n_seen = 0
        self.recent_position = 0
```

### Practical Considerations

**When to use StreamingLLM**:
- Chatbots with very long conversations
- Document processing where full context isn't needed
- Streaming applications (video captioning, live transcription)

**Limitations**:
- Loses information in the middle of context
- Best for tasks where recent + initial context matter most
- Not suitable for retrieval tasks requiring full document access

---

## Memory-Augmented Architectures

Instead of fitting everything in context, augment the model with external memory.

### Retrieval-Augmented Generation (RAG)

Split long context into chunks, retrieve relevant ones, feed to model.

**Architecture**:
1. Embed document chunks with embedding model
2. Store in vector database
3. At query time, retrieve top-k relevant chunks
4. Concatenate with query and feed to LLM

```python
class SimpleRAG:
    """Simple Retrieval-Augmented Generation system.

    Instead of fitting 100K tokens in context, we:
    1. Chunk and embed documents
    2. Retrieve most relevant chunks for query
    3. Use only those chunks (e.g., 4K tokens) as context

    This allows "accessing" much larger contexts than model supports.
    """
    def __init__(
        self,
        embedding_model,
        llm_model,
        chunk_size: int = 512,
        top_k: int = 5
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.top_k = top_k

        # Storage
        self.chunks = []
        self.embeddings = []

    def index_document(self, document: str):
        """Chunk and embed a document for later retrieval."""
        # Simple chunking (in practice, use smarter chunking)
        words = document.split()
        chunks = [
            ' '.join(words[i:i+self.chunk_size])
            for i in range(0, len(words), self.chunk_size)
        ]

        # Embed chunks
        with torch.no_grad():
            embeddings = self.embedding_model.encode(chunks)

        self.chunks.extend(chunks)
        self.embeddings.append(embeddings)

    def retrieve(self, query: str, k: int = None) -> list[str]:
        """Retrieve top-k most relevant chunks for a query."""
        k = k or self.top_k

        # Embed query
        query_emb = self.embedding_model.encode([query])[0]

        # Compute similarities
        all_embeddings = torch.cat(self.embeddings, dim=0)
        similarities = torch.cosine_similarity(
            query_emb.unsqueeze(0),
            all_embeddings,
            dim=-1
        )

        # Get top-k
        top_indices = similarities.topk(k).indices
        return [self.chunks[i] for i in top_indices]

    def generate(self, query: str) -> str:
        """Generate answer using retrieved context."""
        # Retrieve relevant chunks
        context_chunks = self.retrieve(query)
        context = '\n\n'.join(context_chunks)

        # Build prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Generate with LLM
        return self.llm_model.generate(prompt)
```

### Memorizing Transformers

**Key idea**: Add a kNN lookup to retrieve from past hidden states.

**Paper**: [Memorizing Transformers](https://arxiv.org/abs/2203.08913) (Wu et al., 2022)

At each layer:
1. Standard attention over recent context (e.g., 2K tokens)
2. kNN retrieval from long-term memory of past (key, value) pairs
3. Combine both sources of information

$$
\text{Output} = \text{Attention}(Q, K_{\text{local}}, V_{\text{local}}) + \lambda \cdot \text{kNN}(Q, \mathcal{M})
$$

where $\mathcal{M}$ is the external memory of past activations.

```python
class MemorizingAttention(nn.Module):
    """Attention augmented with kNN memory lookup.

    Combines:
    - Local attention over recent context
    - kNN retrieval from long-term memory

    Paper: https://arxiv.org/abs/2203.08913
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        local_window: int = 2048,
        memory_size: int = 65536,  # 64K memory slots
        k_neighbors: int = 32
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.local_window = local_window
        self.k_neighbors = k_neighbors

        # Standard attention components
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Memory: stored keys and values
        # In practice, stored on disk or distributed
        self.memory_keys = torch.zeros(memory_size, dim)
        self.memory_values = torch.zeros(memory_size, dim)
        self.memory_position = 0

        # Gating: how much to trust memory vs local attention
        self.memory_gate = nn.Linear(dim, 1)

    def add_to_memory(self, k: torch.Tensor, v: torch.Tensor):
        """Add keys and values to long-term memory."""
        batch, seq_len, _ = k.shape

        for i in range(seq_len):
            idx = self.memory_position % len(self.memory_keys)
            self.memory_keys[idx] = k[0, i].detach()  # Assuming batch=1
            self.memory_values[idx] = v[0, i].detach()
            self.memory_position += 1

    def knn_lookup(self, q: torch.Tensor) -> torch.Tensor:
        """Retrieve k-nearest neighbors from memory for each query.

        Args:
            q: Query tensor [batch, seq_len, dim]

        Returns:
            Retrieved values [batch, seq_len, dim]
        """
        batch, seq_len, _ = q.shape

        # Compute similarities with all memory keys
        # q: [batch, seq_len, dim]
        # memory_keys: [memory_size, dim]
        similarities = torch.matmul(q, self.memory_keys.T)  # [batch, seq_len, memory_size]

        # Get top-k
        top_k_scores, top_k_indices = similarities.topk(self.k_neighbors, dim=-1)
        top_k_scores = torch.softmax(top_k_scores / math.sqrt(self.dim), dim=-1)

        # Retrieve values
        retrieved = torch.zeros(batch, seq_len, self.dim, device=q.device)
        for b in range(batch):
            for s in range(seq_len):
                indices = top_k_indices[b, s]
                scores = top_k_scores[b, s]
                retrieved[b, s] = (self.memory_values[indices] * scores.unsqueeze(-1)).sum(dim=0)

        return retrieved

    def forward(
        self,
        x: torch.Tensor,
        use_memory: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: Input [batch, seq_len, dim]
            use_memory: Whether to use kNN memory lookup
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Local attention (standard)
        # For simplicity, using full attention here
        # In practice, would use sliding window
        q_heads = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k_heads = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v_heads = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        local_out = torch.matmul(attn, v_heads)
        local_out = local_out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)

        # Memory retrieval (kNN)
        if use_memory and self.memory_position > 0:
            memory_out = self.knn_lookup(q)

            # Gate: balance local vs memory
            gate = torch.sigmoid(self.memory_gate(x))
            output = (1 - gate) * local_out + gate * memory_out
        else:
            output = local_out

        # Add current K, V to memory for future lookups
        if use_memory:
            self.add_to_memory(k, v)

        return self.o_proj(output)
```

---

## Landmark Attention

**Key idea**: Compress long sequences into "landmark" tokens that summarize blocks.

**Paper**: [Landmark Attention: Random-Access Infinite Context Length for Transformers](https://arxiv.org/abs/2305.16300) (Mohtashami & Jaggi, 2023)

**Architecture**:
- Divide sequence into blocks (e.g., 50 tokens each)
- Create a landmark token for each block (via pooling or learned compression)
- Full attention over landmarks (cheap since few landmarks)
- Local attention within each block
- Cross-attention from tokens to landmarks

```python
class LandmarkAttention(nn.Module):
    """Landmark Attention for long sequences.

    Divides sequence into blocks, creates landmark tokens,
    and uses hierarchical attention.

    Paper: https://arxiv.org/abs/2305.16300
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        block_size: int = 50,
        landmark_pooling: str = "max"  # or "mean", "learned"
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.block_size = block_size
        self.landmark_pooling = landmark_pooling

        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # For learned landmark creation
        if landmark_pooling == "learned":
            self.landmark_compression = nn.Linear(dim * block_size, dim)

    def create_landmarks(self, x: torch.Tensor) -> torch.Tensor:
        """Create landmark tokens from blocks.

        Args:
            x: Input [batch, seq_len, dim]

        Returns:
            Landmarks [batch, n_blocks, dim]
        """
        batch, seq_len, dim = x.shape
        n_blocks = seq_len // self.block_size

        # Reshape into blocks
        # Truncate to multiple of block_size
        truncated_len = n_blocks * self.block_size
        x_blocks = x[:, :truncated_len].view(batch, n_blocks, self.block_size, dim)

        if self.landmark_pooling == "max":
            landmarks = x_blocks.max(dim=2)[0]
        elif self.landmark_pooling == "mean":
            landmarks = x_blocks.mean(dim=2)
        elif self.landmark_pooling == "learned":
            # Flatten blocks and project
            x_flat = x_blocks.view(batch, n_blocks, -1)
            landmarks = self.landmark_compression(x_flat)
        else:
            raise ValueError(f"Unknown pooling: {self.landmark_pooling}")

        return landmarks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies:
        1. Local attention within blocks
        2. Cross-attention to landmarks
        3. Attention among landmarks
        """
        batch, seq_len, dim = x.shape

        # Create landmarks
        landmarks = self.create_landmarks(x)  # [batch, n_blocks, dim]
        n_blocks = landmarks.shape[1]

        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Landmark K, V
        k_landmarks = self.k_proj(landmarks)
        v_landmarks = self.v_proj(landmarks)

        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        k_landmarks = k_landmarks.view(batch, n_blocks, self.n_heads, self.head_dim).transpose(1, 2)
        v_landmarks = v_landmarks.view(batch, n_blocks, self.n_heads, self.head_dim).transpose(1, 2)

        # 1. Local attention within blocks
        local_out = torch.zeros_like(q.transpose(1, 2)).transpose(1, 2)
        for block_idx in range(n_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size

            q_block = q[:, :, start:end, :]
            k_block = k[:, :, start:end, :]
            v_block = v[:, :, start:end, :]

            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)
            local_out[:, :, start:end, :] = torch.matmul(attn, v_block)

        # 2. Cross-attention to landmarks
        scores_to_landmarks = torch.matmul(q, k_landmarks.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_to_landmarks = torch.softmax(scores_to_landmarks, dim=-1)
        landmark_out = torch.matmul(attn_to_landmarks, v_landmarks)

        # Combine local and landmark attention
        # Simple averaging here; could use learned gating
        output = (local_out + landmark_out) / 2

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.o_proj(output)
```

**Benefits**:
- Reduces complexity from $O(n^2)$ to $O(n \cdot \frac{n}{b} + (\frac{n}{b})^2) = O(\frac{n^2}{b})$
- Can handle very long sequences
- Preserves global information via landmarks

**Tradeoffs**:
- Information loss in compression
- Requires tuning block size

---

## Ring Attention for Distributed Long Context

When context is too large for a single GPU, **Ring Attention** distributes it across devices.

**Paper**: [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (Liu et al., 2023)

### The Core Idea

Instead of splitting the attention computation across sequence length (which requires all-to-all communication), Ring Attention:

1. Splits sequence into chunks across devices
2. Computes attention blockwise
3. Passes KV blocks in a ring between devices
4. Each device sees all KV pairs eventually, but one block at a time

**Complexity**:
- Communication: $O(n \cdot d)$ instead of $O(n^2)$
- Enables context lengths in the millions

### Algorithm

For $N$ devices, each holding sequence chunk of length $\frac{L}{N}$:

1. Each device computes attention using its local KV
2. Pass KV to next device in ring
3. Compute attention with new KV block and accumulate
4. Repeat $N$ times until each device has seen all KV

```python
class RingAttention(nn.Module):
    """Ring Attention for distributed long-context computation.

    Enables sequence lengths in millions by distributing across GPUs.
    Each GPU processes its chunk and passes KV to neighbors in a ring.

    Paper: https://arxiv.org/abs/2310.01889

    Note: This is a simplified illustration. Real implementation uses
    NCCL or CUDA-aware MPI for efficient GPU communication.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        world_size: int,  # Number of GPUs
        rank: int,  # This GPU's rank
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.world_size = world_size
        self.rank = rank

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Local chunk [batch, local_seq_len, dim]

        Each GPU processes 1/world_size of the sequence.
        """
        batch, local_seq_len, _ = x.shape

        # Project local chunk
        q = self.q_proj(x)  # Local queries
        k = self.k_proj(x)  # Local keys
        v = self.v_proj(x)  # Local values

        # Reshape for multi-head
        q = q.view(batch, local_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, local_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, local_seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Initialize output and normalization terms
        output = torch.zeros_like(q)
        max_scores = torch.full(
            (batch, self.n_heads, local_seq_len, 1),
            float('-inf'),
            device=x.device
        )
        sum_exp_scores = torch.zeros(
            batch, self.n_heads, local_seq_len, 1,
            device=x.device
        )

        # Ring: receive KV from all devices
        k_block, v_block = k, v  # Start with our own

        for step in range(self.world_size):
            # Compute attention scores for this KV block
            scores = torch.matmul(q, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # For causal attention, mask future positions
            # Position offset: which absolute positions is this block?
            block_start_pos = ((self.rank + step) % self.world_size) * local_seq_len
            if block_start_pos > self.rank * local_seq_len:
                # This block is in the future, mask entirely
                scores.fill_(float('-inf'))

            # Numerically stable softmax using online normalization
            # (See Flash Attention chapter for details)
            new_max_scores = torch.maximum(max_scores, scores.max(dim=-1, keepdim=True)[0])

            # Rescale previous values
            exp_scores = torch.exp(scores - new_max_scores)
            rescale_factor = torch.exp(max_scores - new_max_scores)

            output = output * rescale_factor + torch.matmul(exp_scores, v_block)
            sum_exp_scores = sum_exp_scores * rescale_factor + exp_scores.sum(dim=-1, keepdim=True)
            max_scores = new_max_scores

            # Ring communication: send KV to next GPU, receive from previous
            # In real implementation, this would be:
            # k_block = ring_send_recv(k_block, dest=(rank+1)%world_size, src=(rank-1)%world_size)
            # v_block = ring_send_recv(v_block, dest=(rank+1)%world_size, src=(rank-1)%world_size)
            # For this illustration, we'll skip actual communication

        # Final normalization
        output = output / sum_exp_scores

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch, local_seq_len, self.dim)
        return self.o_proj(output)
```

### Comparison with Other Parallelism Strategies

| Strategy | Splits | Communication | Max Context |
|----------|--------|---------------|-------------|
| Data Parallel | Batch | Gradients | Single GPU limit |
| Tensor Parallel | Model | Activations (all-to-all) | Single GPU limit |
| Sequence Parallel | Sequence (naively) | Attention matrix ($O(n^2)$) | Limited |
| **Ring Attention** | Sequence (ring) | KV blocks ($O(nd)$) | **Millions** |

**Use cases**:
- Training on extremely long documents
- Genomic sequences (millions of base pairs)
- Video understanding (thousands of frames)

---

## Evaluation on Long-Range Tasks

How do we know if long-context techniques actually work?

### Needle-in-a-Haystack

**Setup**: Hide a "needle" (specific fact) in a "haystack" (long irrelevant text), ask model to retrieve it.

**Example**:
```
[10,000 words of Paul Graham essays]
The secret password is "strawberry".
[10,000 more words of Paul Graham essays]

Question: What is the secret password?
```

**Metrics**:
- Accuracy at different needle positions (beginning, middle, end)
- Accuracy vs. context length
- Degradation beyond training length

```python
def needle_in_haystack_eval(
    model,
    tokenizer,
    context_lengths: list[int] = [1024, 2048, 4096, 8192, 16384],
    n_trials: int = 10
) -> dict:
    """Evaluate model on needle-in-haystack task.

    Returns:
        Dictionary with accuracy for each context length and position.
    """
    results = {}

    # Template
    haystack_text = load_haystack_corpus()  # Long, irrelevant text
    needle = "The magic number is 73."

    for ctx_len in context_lengths:
        accuracies = []

        for trial in range(n_trials):
            # Create haystack of appropriate length
            haystack = generate_text_of_length(haystack_text, ctx_len - 100)

            # Insert needle at random position
            needle_pos = random.randint(100, len(haystack) - 100)
            text = haystack[:needle_pos] + needle + haystack[needle_pos:]

            # Query
            query = "\n\nQuestion: What is the magic number? Answer:"
            full_prompt = text + query

            # Generate
            tokens = tokenizer.encode(full_prompt)
            output = model.generate(tokens, max_new_tokens=10)
            response = tokenizer.decode(output)

            # Check if correct
            correct = "73" in response
            accuracies.append(correct)

        results[ctx_len] = sum(accuracies) / len(accuracies)

    return results
```

### RULER: A Comprehensive Long-Context Benchmark

**Paper**: [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) (Hsieh et al., 2024)

RULER tests 4 task categories across different lengths:

1. **Retrieval**: Find specific information
   - Needle-in-a-haystack (single and multi-needle)
   - Variable tracking

2. **Multi-hop Reasoning**: Connect information across distance
   - Common words extraction
   - Frequent words extraction

3. **Aggregation**: Combine information from entire context
   - QA with aggregation

4. **Length Extrapolation**: Handle lengths beyond training

**Implementation sketch**:

```python
class RULERBenchmark:
    """RULER: Comprehensive long-context evaluation.

    Paper: https://arxiv.org/abs/2404.06654
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def needle_single(self, context_length: int, needle_depth: float) -> bool:
        """Single needle in haystack.

        Args:
            context_length: Total context length
            needle_depth: Where to place needle (0.0 = start, 1.0 = end)
        """
        # Similar to above
        pass

    def needle_multi(self, context_length: int, n_needles: int) -> float:
        """Multiple needles in haystack.

        Returns:
            Fraction of needles successfully retrieved.
        """
        pass

    def variable_tracking(self, context_length: int, n_variables: int) -> float:
        """Track multiple variable assignments across context.

        Example:
        x = 5
        [many lines]
        y = 10
        [many lines]
        z = x + y
        Question: What is z?
        """
        pass

    def common_words(self, context_length: int) -> bool:
        """Multi-hop: find words appearing in all documents.

        Requires aggregating information across entire context.
        """
        pass

    def run_full_benchmark(
        self,
        lengths: list[int] = [4096, 8192, 16384, 32768, 65536, 131072]
    ) -> dict:
        """Run all RULER tasks across context lengths."""
        results = {}

        for length in lengths:
            results[length] = {
                'needle_single_depth_0.0': self.needle_single(length, 0.0),
                'needle_single_depth_0.5': self.needle_single(length, 0.5),
                'needle_single_depth_1.0': self.needle_single(length, 1.0),
                'needle_multi_5': self.needle_multi(length, 5),
                'variable_tracking': self.variable_tracking(length, 10),
                'common_words': self.common_words(length),
            }

        return results
```

### Perplexity on Long Documents

Standard language modeling evaluation on long documents (books, papers).

**Key insight**: A model that truly uses long context should achieve lower perplexity as context increases.

```python
def evaluate_perplexity_vs_context(
    model,
    tokenizer,
    documents: list[str],
    context_lengths: list[int] = [512, 1024, 2048, 4096, 8192]
) -> dict:
    """Measure perplexity with different amounts of context.

    A model that uses long context well should show:
    - Decreasing perplexity as context increases
    - Continued improvement beyond training length (if extended properly)
    """
    results = {}

    for ctx_len in context_lengths:
        total_loss = 0
        total_tokens = 0

        for doc in documents:
            tokens = tokenizer.encode(doc)

            # Skip documents shorter than context length
            if len(tokens) < ctx_len + 256:
                continue

            # Use ctx_len tokens as context, predict next 256
            context = tokens[:ctx_len]
            target = tokens[ctx_len:ctx_len+256]

            # Compute loss
            with torch.no_grad():
                logits = model(context + target)
                loss = nn.functional.cross_entropy(
                    logits[ctx_len-1:-1].view(-1, logits.shape[-1]),
                    torch.tensor(target),
                    reduction='sum'
                )

            total_loss += loss.item()
            total_tokens += len(target)

        perplexity = math.exp(total_loss / total_tokens)
        results[ctx_len] = perplexity

    return results
```

### Passkey Retrieval

Similar to needle-in-haystack but tests specific formatting:

```
There is an important info hidden inside a lot of irrelevant text.
Find it and memorize it. I will quiz you about the important
information there.

The pass key is 12345. Remember it. 12345 is the pass key.

[Many lines of irrelevant text]

What is the pass key?
```

**Metrics**: Exact match accuracy

---

## Complete Implementation: Long Context Transformer

Let's build a complete transformer with multiple long-context techniques:

```python
class LongContextTransformer(nn.Module):
    """Complete transformer with long-context techniques.

    Combines:
    - YaRN RoPE scaling for position encoding
    - Flash Attention for memory efficiency (see [Flash Attention](12-flash-attention.md))
    - Sliding window + global attention (Gemma-style)
    - Optional StreamingLLM for inference
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 2048,
        n_layers: int = 24,
        n_heads: int = 16,
        n_kv_heads: int = 4,  # GQA
        d_ff: int = 5632,
        max_seq_len: int = 4096,
        rope_scaling_factor: float = 4.0,
        sliding_window: int = 2048,
        use_flash: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # RoPE with YaRN scaling
        self.rope = YaRNScalingRoPE(
            dim=d_model // n_heads,
            max_position_embeddings=max_seq_len,
            scaling_factor=rope_scaling_factor
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            LongContextTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                layer_idx=i,
                sliding_window=sliding_window,
                use_flash=use_flash,
                dropout=dropout
            )
            for i in range(n_layers)
        ])

        # Output
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Optional: tie embeddings
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
        cache: list = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            use_cache: Whether to use/update KV cache
            cache: Existing KV cache (for generation)
        """
        batch, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]

        # Get RoPE embeddings
        cos, sin = self.rope(x, seq_len)

        # Initialize cache if needed
        if use_cache and cache is None:
            cache = [None] * self.n_layers

        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x, new_cache = layer(x, cos, sin, layer_cache)
            if use_cache:
                cache[i] = new_cache

        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, cache
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        use_streaming: bool = False,
        streaming_cache_size: int = 2048
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            use_streaming: Use StreamingLLM cache
            streaming_cache_size: Size of streaming cache
        """
        if use_streaming:
            # Use StreamingLLM cache
            cache = StreamingLLMCache(
                cache_size=streaming_cache_size,
                n_layers=self.n_layers,
                device=input_ids.device
            )
        else:
            cache = None

        for _ in range(max_new_tokens):
            # Get logits for last token
            logits, cache = self.forward(input_ids, use_cache=True, cache=cache)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_k_indices, -1, next_token_idx)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class LongContextTransformerLayer(nn.Module):
    """Single transformer layer with long-context optimizations."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        layer_idx: int,
        sliding_window: int,
        use_flash: bool,
        dropout: float
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Gemma-style: alternate between sliding window and global
        self.is_sliding = (layer_idx % 2 == 1)
        self.window_size = sliding_window if self.is_sliding else None

        # Attention
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_flash=use_flash,
            window_size=self.window_size
        )

        # FFN
        self.ffn = SwiGLU(d_model, d_ff)

        # Norms
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: tuple = None
    ) -> tuple[torch.Tensor, tuple]:
        # Attention with residual
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_cache


class GroupedQueryAttention(nn.Module):
    """GQA with optional sliding window and flash attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        use_flash: bool = True,
        window_size: int = None
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.use_flash = use_flash
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: tuple = None
    ) -> tuple[torch.Tensor, tuple]:
        batch, seq_len, _ = x.shape

        # Project
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Update cache if provided
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        # Expand KV for GQA
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)

        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        if self.use_flash and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use PyTorch's built-in flash attention
            # See [Flash Attention](12-flash-attention.md) for details
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                # Note: window_size would need custom kernel
            )
        else:
            # Manual attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Causal mask
            mask = torch.triu(torch.ones(seq_len, k.shape[2], dtype=torch.bool, device=x.device), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))

            # Sliding window mask
            if self.window_size is not None:
                window_mask = torch.ones_like(mask)
                for i in range(seq_len):
                    start = max(0, i - self.window_size)
                    window_mask[i, start:i+1] = False
                scores.masked_fill_(window_mask, float('-inf'))

            attn = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn, v)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(attn_out)

        # Return output and new cache
        new_cache = (k[:, :, :seq_len], v[:, :, :seq_len]) if cache is not None else None
        return output, new_cache


class SwiGLU(nn.Module):
    """SwiGLU activation for FFN.

    See [Architecture Comparison](29-model-architectures.md) for details.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    See [The Transformer Block](09-transformer-block.md) for details.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

---

## Summary and Best Practices

### Summary of Techniques

| Technique | Type | Complexity Reduction | Fine-tuning Required? | Best For |
|-----------|------|---------------------|----------------------|----------|
| **Linear RoPE Scaling** | Position | None | Yes | Quick extension |
| **NTK Scaling** | Position | None | Often No | Zero-shot extension |
| **YaRN** | Position | None | Minimal | Production deployment |
| **StreamingLLM** | Attention | $O(n^2) \to O(w \cdot n)$ | No | Infinite streaming |
| **Sliding Window** | Attention | $O(n^2) \to O(w \cdot n)$ | Architecture change | Efficient long context |
| **Landmark** | Attention | $O(n^2) \to O(n^2/b)$ | Architecture change | Hierarchical info |
| **Ring Attention** | Distributed | Same, but distributed | No | Multi-GPU, very long |
| **RAG** | Architecture | Depends on retrieval | No | External knowledge |

### When to Use Each Technique

**Extending existing model**:
1. Start with **Dynamic NTK** (zero-shot)
2. If insufficient, try **YaRN** with short fine-tuning
3. For streaming: add **StreamingLLM**

**Training new model**:
1. Use **ABF** or **YaRN** RoPE from start
2. Consider **sliding window** attention for efficiency
3. Alternate global and local layers (Gemma-style)
4. Use **Flash Attention** for memory efficiency (see [Flash Attention](12-flash-attention.md))

**Extreme length** (1M+ tokens):
1. **Ring Attention** for distributed training
2. **RAG** for inference (don't fit everything in context)
3. **Landmark attention** for hierarchical processing

### Best Practices

1. **Always benchmark**: Use RULER, needle-in-haystack, and perplexity
2. **Position encoding matters most**: Often bigger bottleneck than attention
3. **Don't assume linear scaling**: Test at target lengths
4. **Mind the KV cache**: At 100K context, cache can be 100GB+
5. **Consider task requirements**: Do you really need full attention over all tokens?

---

## Exercises

1. **RoPE Scaling Comparison**
   - Implement all RoPE scaling methods
   - Train a small model (100M params) on 2K context
   - Evaluate each method at 4K, 8K, 16K without fine-tuning
   - Which performs best? Why?

2. **Attention Sink Analysis**
   - Visualize attention weights in a pretrained model
   - Verify the attention sink phenomenon
   - Implement StreamingLLM and compare perplexity with/without sink tokens
   - How many sink tokens are needed?

3. **Landmark Attention Implementation**
   - Implement landmark attention with different pooling strategies
   - Compare max pooling, mean pooling, and learned compression
   - Measure speed and accuracy vs. full attention
   - What block size works best?

4. **Long Context Evaluation**
   - Implement the needle-in-haystack benchmark
   - Test a model at different context lengths
   - Plot accuracy vs. needle depth and context length
   - Where does the model fail?

5. **Hybrid Architecture**
   - Build a model combining:
     - YaRN RoPE scaling
     - Sliding window attention
     - StreamingLLM for inference
   - Compare with baseline full attention
   - Measure: speed, memory, accuracy on RULER

6. **Memory-Augmented RAG**
   - Implement a simple RAG system
   - Compare RAG with 4K chunks vs. 16K full context
   - When does RAG win? When does it fail?
   - Add re-ranking and measure improvement

7. **Ring Attention Simulation**
   - Simulate Ring Attention on a single GPU (with manual KV passing)
   - Measure communication overhead
   - How does it scale with number of "devices"?
   - Calculate theoretical max context for 8 A100 GPUs

8. **Context Length Ablation**
   - Take a long-context model (e.g., Llama 3.1 with 128K context)
   - Evaluate at 4K, 8K, 16K, 32K, 64K, 128K
   - Plot perplexity vs. context length
   - Does it show "context collapse" at any length?

---

## References

### Key Papers

1. **RoPE and Scaling**
   - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
   - [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) (Peng et al., 2023)
   - [Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/abs/2306.15595) (Chen et al., 2023)

2. **Attention Efficiency**
   - [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (Xiao et al., 2023)
   - [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (Liu et al., 2023)
   - [Landmark Attention: Random-Access Infinite Context Length for Transformers](https://arxiv.org/abs/2305.16300) (Mohtashami & Jaggi, 2023)

3. **Memory-Augmented Models**
   - [Memorizing Transformers](https://arxiv.org/abs/2203.08913) (Wu et al., 2022)
   - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)

4. **Evaluation**
   - [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) (Hsieh et al., 2024)
   - [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (Liu et al., 2023)

5. **Production Systems**
   - [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](https://arxiv.org/abs/2403.05530) (Gemini Team, 2024)
   - [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Llama Team, 2024)

### Related Chapters

- [Rotary Position Embeddings (RoPE)](08-rope.md) - Base RoPE mechanism
- [Flash Attention](12-flash-attention.md) - Memory-efficient attention
- [Other Efficient Attention Variants](13-efficient-attention.md) - Sparse and linear attention
- [Architecture Comparison: Modern LLMs](29-model-architectures.md) - Context lengths in production models
- [Distributed Training and Parallelism](16-distributed-training.md) - Parallelism strategies

### Additional Resources

- [NTK-Aware RoPE Scaling (Reddit Discussion)](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
- [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://github.com/THUDM/LongBench)
- [Long Context Models Catalog](https://github.com/Mooler0410/LLMsPracticalGuide#long-context)

---

**Next Chapter**: [Multimodality](27-multimodality.md) - Extending LLMs to vision, audio, and beyond.

**Previous Chapter**: [Advanced Diffusion Topics](25-diffusion-advanced.md) - Classifier-free guidance, latent diffusion, and more.
