# Chapter 6: Cross-Attention

Cross-attention is a fundamental mechanism that enables models to relate information from two different sequences. While self-attention allows a sequence to attend to itself, cross-attention enables one sequence to attend to another. This is the core mechanism behind encoder-decoder architectures and is increasingly important in multimodal models that combine vision, text, and other modalities.

## Table of Contents

1. [Introduction](#introduction)
2. [Self-Attention vs Cross-Attention](#self-attention-vs-cross-attention)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Implementation](#implementation)
5. [Cross-Attention in Encoder-Decoder Models](#cross-attention-in-encoder-decoder-models)
6. [Multimodal Cross-Attention](#multimodal-cross-attention)
7. [Practical Considerations](#practical-considerations)
8. [Advanced Topics](#advanced-topics)
9. [Summary](#summary)
10. [Exercises](#exercises)
11. [References](#references)

---

## Introduction

Cross-attention was introduced as part of the original Transformer architecture in the seminal "Attention Is All You Need" paper (Vaswani et al., 2017). It addresses a fundamental question: **How can we align and integrate information from two different sources?**

**Key Applications:**
- **Machine Translation**: Aligning source and target language sequences
- **Image Captioning**: Relating visual features to textual descriptions
- **Question Answering**: Connecting questions to context passages
- **Vision-Language Models**: Fusing visual and textual representations (e.g., CLIP, LLaVA, GPT-4V)
- **Audio-Visual Learning**: Synchronizing audio and video streams

**Key Insight**: In cross-attention, the **queries** come from one sequence (e.g., decoder), while **keys** and **values** come from another sequence (e.g., encoder). This allows the model to selectively retrieve relevant information from the source sequence based on the current state of the target sequence.

---

## Self-Attention vs Cross-Attention

Understanding the difference between self-attention and cross-attention is crucial for ML interviews.

### Self-Attention

In self-attention (see [Multi-Head Attention](04-multi-head-attention.md)), all three components (Q, K, V) come from the same sequence:

$$
\text{SelfAttention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:
- $Q = XW_Q$
- $K = XW_K$
- $V = XW_V$
- $X$ is the input sequence

**Purpose**: Model relationships within a single sequence (e.g., how words relate to each other in a sentence).

### Cross-Attention

In cross-attention, queries come from one sequence while keys and values come from another:

$$
\text{CrossAttention}(X, Y) = \text{softmax}\left(\frac{Q(K^T)}{\sqrt{d_k}}\right)V
$$

where:
- $Q = XW_Q$ (queries from target sequence)
- $K = YW_K$ (keys from source sequence)
- $V = YW_V$ (values from source sequence)
- $X$ is the target/decoder sequence
- $Y$ is the source/encoder sequence

**Purpose**: Model relationships between two different sequences (e.g., how decoder states relate to encoder outputs).

### Visual Comparison

```
Self-Attention:
Input: [The, cat, sat, on, mat]
       ↓   ↓   ↓   ↓   ↓
       Q   K   V (all from same sequence)

Cross-Attention (e.g., Translation):
Source (Encoder): [Le, chat, dort]  → K, V
                                     ↗
Target (Decoder): [The, cat, ___]  → Q
```

---

## Mathematical Formulation

### Standard Cross-Attention

Given:
- Target sequence: $X \in \mathbb{R}^{n \times d_{\text{model}}}$ (e.g., decoder states)
- Source sequence: $Y \in \mathbb{R}^{m \times d_{\text{model}}}$ (e.g., encoder outputs)

The cross-attention operation is:

$$
\text{CrossAttn}(X, Y) = \text{Attention}(XW_Q, YW_K, YW_V)
$$

Breaking this down:

1. **Project to Q, K, V**:
   $$
   Q = XW_Q \in \mathbb{R}^{n \times d_k}
   $$
   $$
   K = YW_K \in \mathbb{R}^{m \times d_k}
   $$
   $$
   V = YW_V \in \mathbb{R}^{m \times d_v}
   $$

2. **Compute attention scores**:
   $$
   S = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times m}
   $$

   Note: The score matrix is $n \times m$ (target length × source length), not $n \times n$ as in self-attention.

3. **Apply softmax** (over source dimension):
   $$
   A = \text{softmax}(S) \in \mathbb{R}^{n \times m}
   $$

   For each target position $i$: $A_{i,:} = \text{softmax}(S_{i,:})$

   This gives us a probability distribution over source positions for each target position.

4. **Weighted sum of values**:
   $$
   \text{Output} = AV \in \mathbb{R}^{n \times d_v}
   $$

5. **Final projection**:
   $$
   \text{CrossAttn}(X, Y) = (AV)W_O \in \mathbb{R}^{n \times d_{\text{model}}}
   $$

### Multi-Head Cross-Attention

Just like self-attention, cross-attention benefits from multiple heads (see [Multi-Head Attention](04-multi-head-attention.md)):

$$
\text{MultiHead}(X, Y) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O
$$

where each head is:

$$
\text{head}_i = \text{Attention}(XW_Q^i, YW_K^i, YW_V^i)
$$

**Key difference**: While self-attention has all projections from the same sequence, cross-attention projects queries from $X$ and keys/values from $Y$.

### Attention Pattern Interpretation

The attention matrix $A \in \mathbb{R}^{n \times m}$ shows alignment between sequences:

- **Row $i$**: Shows which source positions target position $i$ attends to
- **High value at $A_{i,j}$**: Target position $i$ strongly attends to source position $j$

Example in machine translation:
```
Source (French):  Le  chat  dort  sur  le  tapis
Target (English): The  cat  is  sleeping  on  the  mat

Attention might show:
"The" attends to "Le"
"cat" attends to "chat"
"sleeping" attends to "dort"
```

---

## Implementation

### Basic Cross-Attention

Let's implement cross-attention from scratch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """
    Single-head cross-attention mechanism.

    Args:
        d_model: Dimension of model (both input sequences should have this dimension)
        d_k: Dimension of queries and keys
        d_v: Dimension of values
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        d_k: int = None,
        d_v: int = None,
        dropout: float = 0.1
    ):
        super().__init__()

        # Default to d_model if not specified
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        self.d_model = d_model

        # Query projection (from target sequence)
        self.w_q = nn.Linear(d_model, self.d_k, bias=False)

        # Key and Value projections (from source sequence)
        self.w_k = nn.Linear(d_model, self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.d_v, bias=False)

        # Output projection
        self.w_o = nn.Linear(self.d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,      # (batch, n, d_model)
        source: torch.Tensor,      # (batch, m, d_model)
        mask: torch.Tensor = None  # (batch, n, m) or (n, m)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target: Target sequence (provides queries)
            source: Source sequence (provides keys and values)
            mask: Optional attention mask (True = attend, False = mask out)

        Returns:
            output: Cross-attended features (batch, n, d_model)
            attention_weights: Attention weights (batch, n, m)
        """
        batch_size = target.size(0)
        n = target.size(1)  # target sequence length
        m = source.size(1)  # source sequence length

        # Project to Q, K, V
        Q = self.w_q(target)  # (batch, n, d_k)
        K = self.w_k(source)  # (batch, m, d_k)
        V = self.w_v(source)  # (batch, m, d_v)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # (batch, n, d_k) @ (batch, d_k, m) -> (batch, n, m)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            # Convert boolean mask to additive mask
            # True (attend) -> 0, False (mask) -> -inf
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                scores = scores + mask

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch, n, m)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # (batch, n, m) @ (batch, m, d_v) -> (batch, n, d_v)
        output = torch.matmul(attention_weights, V)

        # Final output projection
        output = self.w_o(output)  # (batch, n, d_model)

        return output, attention_weights


# Example usage
if __name__ == "__main__":
    batch_size = 2
    target_len = 10  # decoder sequence length
    source_len = 15  # encoder sequence length
    d_model = 512

    # Create random target and source sequences
    target = torch.randn(batch_size, target_len, d_model)
    source = torch.randn(batch_size, source_len, d_model)

    # Initialize cross-attention
    cross_attn = CrossAttention(d_model=d_model)

    # Forward pass
    output, attn_weights = cross_attn(target, source)

    print(f"Target shape: {target.shape}")        # (2, 10, 512)
    print(f"Source shape: {source.shape}")        # (2, 15, 512)
    print(f"Output shape: {output.shape}")        # (2, 10, 512)
    print(f"Attention weights shape: {attn_weights.shape}")  # (2, 10, 15)

    # Verify attention weights sum to 1 for each target position
    assert torch.allclose(
        attn_weights.sum(dim=-1),
        torch.ones(batch_size, target_len),
        atol=1e-6
    ), "Attention weights should sum to 1 across source dimension"
    print("✓ Attention weights correctly normalized")
```

### Multi-Head Cross-Attention

Now let's implement multi-head cross-attention:

```python
class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism.

    This is the version used in the original Transformer (Vaswani et al., 2017).

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # Combined Q, K, V projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_heads, d_k).

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back to original dimension.

        Args:
            x: (batch, n_heads, seq_len, d_k)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, n_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2)  # (batch, seq_len, n_heads, d_k)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target: Target sequence (batch, n, d_model)
            source: Source sequence (batch, m, d_model)
            mask: Optional mask (batch, 1, n, m) or (1, 1, n, m)

        Returns:
            output: (batch, n, d_model)
            attention_weights: (batch, n_heads, n, m)
        """
        batch_size = target.size(0)
        n = target.size(1)
        m = source.size(1)

        # Project and split into heads
        Q = self.split_heads(self.w_q(target))  # (batch, n_heads, n, d_k)
        K = self.split_heads(self.w_k(source))  # (batch, n_heads, m, d_k)
        V = self.split_heads(self.w_v(source))  # (batch, n_heads, m, d_k)

        # Compute scaled dot-product attention
        # (batch, n_heads, n, d_k) @ (batch, n_heads, d_k, m)
        # -> (batch, n_heads, n, m)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                scores = scores + mask

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # (batch, n_heads, n, m) @ (batch, n_heads, m, d_k)
        # -> (batch, n_heads, n, d_k)
        output = torch.matmul(attention_weights, V)

        # Combine heads
        output = self.combine_heads(output)  # (batch, n, d_model)

        # Final projection
        output = self.w_o(output)

        return output, attention_weights


# Example usage
if __name__ == "__main__":
    batch_size = 2
    target_len = 10
    source_len = 15
    d_model = 512
    n_heads = 8

    target = torch.randn(batch_size, target_len, d_model)
    source = torch.randn(batch_size, source_len, d_model)

    mh_cross_attn = MultiHeadCrossAttention(d_model=d_model, n_heads=n_heads)

    output, attn_weights = mh_cross_attn(target, source)

    print(f"Output shape: {output.shape}")  # (2, 10, 512)
    print(f"Attention weights shape: {attn_weights.shape}")  # (2, 8, 10, 15)
    print("✓ Multi-head cross-attention working correctly")
```

### Visualization of Cross-Attention

Let's create a function to visualize cross-attention patterns:

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_cross_attention(
    attention_weights: torch.Tensor,
    source_tokens: list[str],
    target_tokens: list[str],
    head_idx: int = 0
):
    """
    Visualize cross-attention weights as a heatmap.

    Args:
        attention_weights: (batch, n_heads, n_target, n_source) or (n_heads, n_target, n_source)
        source_tokens: List of source sequence tokens
        target_tokens: List of target sequence tokens
        head_idx: Which attention head to visualize
    """
    # Handle both batched and unbatched inputs
    if attention_weights.dim() == 4:
        attn = attention_weights[0, head_idx].detach().cpu().numpy()
    else:
        attn = attention_weights[head_idx].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(attn, cmap='viridis', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(source_tokens)))
    ax.set_yticks(np.arange(len(target_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha='right')
    ax.set_yticklabels(target_tokens)

    # Labels
    ax.set_xlabel('Source Sequence', fontsize=12)
    ax.set_ylabel('Target Sequence', fontsize=12)
    ax.set_title(f'Cross-Attention Weights (Head {head_idx})', fontsize=14)

    # Colorbar
    plt.colorbar(im, ax=ax, label='Attention Weight')

    # Add text annotations for weights
    for i in range(len(target_tokens)):
        for j in range(len(source_tokens)):
            text = ax.text(j, i, f'{attn[i, j]:.2f}',
                          ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    return fig

# Example: Translation scenario
if __name__ == "__main__":
    # Simulate a translation task: French -> English
    source_tokens = ["Le", "chat", "dort", "sur", "le", "tapis"]
    target_tokens = ["The", "cat", "sleeps", "on", "the", "mat"]

    # Create dummy attention weights with realistic patterns
    # In real translation, we'd expect:
    # - "The" to attend to "Le"
    # - "cat" to attend to "chat"
    # - etc.
    n_heads = 4
    n_target = len(target_tokens)
    n_source = len(source_tokens)

    # Create somewhat realistic attention pattern
    attention_weights = torch.zeros(1, n_heads, n_target, n_source)

    # Head 0: Strong diagonal alignment (word-to-word)
    for i in range(min(n_target, n_source)):
        attention_weights[0, 0, i, i] = 0.8
        if i > 0:
            attention_weights[0, 0, i, i-1] = 0.1
        if i < n_source - 1:
            attention_weights[0, 0, i, i+1] = 0.1

    # Normalize
    attention_weights[0, 0] = F.softmax(attention_weights[0, 0] * 10, dim=-1)

    # Visualize
    fig = visualize_cross_attention(
        attention_weights,
        source_tokens,
        target_tokens,
        head_idx=0
    )
    plt.savefig('/tmp/cross_attention_example.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to /tmp/cross_attention_example.png")
```

---

## Cross-Attention in Encoder-Decoder Models

Cross-attention is a critical component of encoder-decoder architectures, particularly in the original Transformer model (see [Building a Complete Transformer](11-complete-transformer.md)).

### Architecture Overview

In a standard encoder-decoder Transformer:

1. **Encoder**: Processes the source sequence using self-attention
   - Input: Source sequence (e.g., French sentence)
   - Output: Contextual representations of source tokens

2. **Decoder**: Generates the target sequence using:
   - **Masked self-attention**: Attends to previously generated target tokens (causal)
   - **Cross-attention**: Attends to encoder outputs
   - **Feed-forward network**: Additional transformation

```
Encoder                  Decoder
  ↓                        ↓
[Self-Attention]    [Masked Self-Attention]
  ↓                        ↓
[Feed-Forward]            (Q)
  ↓                        ↓
  ├────────────────→ [Cross-Attention] ← (K, V)
                           ↓
                    [Feed-Forward]
                           ↓
                    [Output Projection]
```

### Decoder Layer with Cross-Attention

Here's a complete decoder layer implementation:

```python
class TransformerDecoderLayer(nn.Module):
    """
    Single layer of a Transformer decoder with cross-attention.

    Components:
    1. Masked self-attention (causal)
    2. Cross-attention to encoder outputs
    3. Position-wise feed-forward network

    All with residual connections and layer normalization.
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        # Masked self-attention (see Chapter 5: Bidirectional vs Causal Attention)
        self.self_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention to encoder
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (batch, target_len, d_model)
            encoder_output: Encoder output (batch, source_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Optional mask for cross-attention (e.g., padding)

        Returns:
            Decoder output (batch, target_len, d_model)
        """
        # 1. Masked self-attention with residual connection
        # Both target and source are x (self-attention)
        self_attn_out, _ = self.self_attn(x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # 2. Cross-attention with residual connection
        # Query from decoder, Key/Value from encoder
        cross_attn_out, _ = self.cross_attn(x, encoder_output, cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # 3. Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


# Example: Full encoder-decoder forward pass
if __name__ == "__main__":
    batch_size = 2
    source_len = 15
    target_len = 10
    d_model = 512

    # Simulated encoder output (in practice, from encoder stack)
    encoder_output = torch.randn(batch_size, source_len, d_model)

    # Decoder input (e.g., shifted target sequence with <START> token)
    decoder_input = torch.randn(batch_size, target_len, d_model)

    # Create causal mask for decoder self-attention
    # See Chapter 5: Bidirectional vs Causal Attention
    causal_mask = torch.triu(
        torch.ones(target_len, target_len, dtype=torch.bool),
        diagonal=1
    )
    causal_mask = ~causal_mask  # Invert: True = attend, False = mask
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, target_len, target_len)

    # Initialize decoder layer
    decoder_layer = TransformerDecoderLayer(d_model=d_model)

    # Forward pass
    output = decoder_layer(
        decoder_input,
        encoder_output,
        self_attn_mask=causal_mask,
        cross_attn_mask=None
    )

    print(f"Decoder output shape: {output.shape}")  # (2, 10, 512)
    print("✓ Decoder layer with cross-attention working correctly")
```

### Sequence-to-Sequence Example: Machine Translation

Let's build a minimal translation model:

```python
class Seq2SeqTransformer(nn.Module):
    """
    Simplified sequence-to-sequence Transformer for demonstration.

    This is a minimal version for educational purposes.
    For a complete implementation, see Chapter 11: Building a Complete Transformer.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)

        # Encoder (simplified: just self-attention layers)
        self.encoder_layers = nn.ModuleList([
            MultiHeadCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_encoder_layers)
        ])

        # Decoder layers with cross-attention
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding (see Chapter 7)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: Source token IDs (batch, src_len)

        Returns:
            Encoder output (batch, src_len, d_model)
        """
        seq_len = src.size(1)

        # Embed and add positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)

        # Apply encoder layers (self-attention)
        for layer in self.encoder_layers:
            x, _ = layer(x, x)  # Self-attention: both target and source are x

        return x

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        causal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode target sequence with cross-attention to encoder output.

        Args:
            tgt: Target token IDs (batch, tgt_len)
            encoder_output: Encoded source (batch, src_len, d_model)
            causal_mask: Causal mask for decoder self-attention

        Returns:
            Logits over vocabulary (batch, tgt_len, vocab_size)
        """
        seq_len = tgt.size(1)

        # Embed and add positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)

        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, causal_mask)

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            src: Source token IDs (batch, src_len)
            tgt: Target token IDs (batch, tgt_len)

        Returns:
            Logits (batch, tgt_len, vocab_size)
        """
        # Create causal mask for decoder
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
            diagonal=1
        )
        causal_mask = ~causal_mask
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Encode source
        encoder_output = self.encode(src)

        # Decode with cross-attention
        logits = self.decode(tgt, encoder_output, causal_mask)

        return logits


# Example usage
if __name__ == "__main__":
    # Small vocabulary sizes for demonstration
    src_vocab_size = 10000  # French
    tgt_vocab_size = 8000   # English

    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=256,  # Smaller for demo
        n_heads=8,
        n_encoder_layers=3,
        n_decoder_layers=3,
        d_ff=1024,
        dropout=0.1
    )

    # Random input (in practice, these would be tokenized sentences)
    batch_size = 2
    src_len = 12
    tgt_len = 10

    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

    # Forward pass
    logits = model(src, tgt)

    print(f"Source shape: {src.shape}")      # (2, 12)
    print(f"Target shape: {tgt.shape}")      # (2, 10)
    print(f"Logits shape: {logits.shape}")   # (2, 10, 8000)

    # Compute loss (teacher forcing)
    # In practice, tgt would be shifted right by 1 position
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(
        logits.view(-1, tgt_vocab_size),
        tgt.view(-1)
    )
    print(f"Loss: {loss.item():.4f}")
    print("✓ Seq2Seq model with cross-attention working correctly")
```

---

## Multimodal Cross-Attention

Cross-attention is essential for multimodal models that combine different modalities (vision + language, audio + text, etc.). For more details, see [Multimodality](27-multimodality.md).

### Vision-Language Cross-Attention

In vision-language models like LLaVA, BLIP-2, and Flamingo, cross-attention allows the language model to attend to visual features:

```python
class VisionLanguageCrossAttention(nn.Module):
    """
    Cross-attention for vision-language models.

    Query: Language features (text tokens)
    Key/Value: Vision features (image patches)

    This allows the language model to "look at" the image when generating text.
    """
    def __init__(
        self,
        d_text: int,      # Text model dimension
        d_vision: int,    # Vision model dimension
        d_model: int,     # Internal dimension
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Project text and vision to common dimension
        self.text_proj = nn.Linear(d_text, d_model, bias=False)
        self.vision_proj_k = nn.Linear(d_vision, d_model, bias=False)
        self.vision_proj_v = nn.Linear(d_vision, d_model, bias=False)

        # Multi-head attention components
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_text, bias=False)  # Project back to text dim

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: torch.Tensor,    # (batch, n_text_tokens, d_text)
        vision_features: torch.Tensor   # (batch, n_patches, d_vision)
    ) -> torch.Tensor:
        """
        Args:
            text_features: Language model hidden states
            vision_features: Vision encoder outputs (image patches)

        Returns:
            Cross-attended text features (batch, n_text_tokens, d_text)
        """
        batch_size = text_features.size(0)
        n_text = text_features.size(1)
        n_patches = vision_features.size(1)

        # Project to common dimension
        text_proj = self.text_proj(text_features)      # (batch, n_text, d_model)
        vision_k = self.vision_proj_k(vision_features)  # (batch, n_patches, d_model)
        vision_v = self.vision_proj_v(vision_features)  # (batch, n_patches, d_model)

        # Multi-head projections
        Q = self.w_q(text_proj)    # (batch, n_text, d_model)
        K = self.w_k(vision_k)     # (batch, n_patches, d_model)
        V = self.w_v(vision_v)     # (batch, n_patches, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, n_text, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, n_patches, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, n_patches, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (batch, n_heads, n_text, d_k)

        # Combine heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, n_text, self.d_model)

        # Project back to text dimension
        output = self.w_o(output)  # (batch, n_text, d_text)

        return output


# Example: Image captioning scenario
if __name__ == "__main__":
    batch_size = 4
    n_text_tokens = 20    # Caption length
    n_patches = 196       # 14x14 patches from image (e.g., ViT with 224x224 image, 16x16 patches)
    d_text = 768          # BERT/GPT-2 dimension
    d_vision = 1024       # Vision Transformer dimension

    # Simulate text and vision features
    text_features = torch.randn(batch_size, n_text_tokens, d_text)
    vision_features = torch.randn(batch_size, n_patches, d_vision)

    # Initialize cross-attention
    vl_cross_attn = VisionLanguageCrossAttention(
        d_text=d_text,
        d_vision=d_vision,
        d_model=512,  # Internal dimension
        n_heads=8
    )

    # Forward pass
    output = vl_cross_attn(text_features, vision_features)

    print(f"Text features shape: {text_features.shape}")    # (4, 20, 768)
    print(f"Vision features shape: {vision_features.shape}") # (4, 196, 1024)
    print(f"Output shape: {output.shape}")                  # (4, 20, 768)
    print("✓ Vision-language cross-attention working correctly")
```

### Perceiver Architecture

The Perceiver (Jaegle et al., 2021) uses cross-attention in a unique way: a small set of learned "latent queries" attend to a large input:

```python
class PerceiverCrossAttention(nn.Module):
    """
    Perceiver-style cross-attention.

    Key idea: Use a small set of learned latent queries to attend
    to a large (possibly multimodal) input. This reduces computation
    from O(n^2) to O(m*n) where m << n.

    Example:
    - Input: 50,000 pixels (n=50,000)
    - Latents: 512 queries (m=512)
    - Attention cost: O(512 * 50,000) instead of O(50,000^2)
    """
    def __init__(
        self,
        num_latents: int,
        d_latent: int,
        d_input: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_latents = num_latents
        self.d_latent = d_latent

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, d_latent))

        # Cross-attention from latents to input
        self.cross_attn = MultiHeadCrossAttention(
            d_model=d_latent,
            n_heads=n_heads,
            dropout=dropout
        )

        # Input projection (if dimensions don't match)
        self.input_proj = nn.Linear(d_input, d_latent) if d_input != d_latent else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, n_inputs, d_input)

        Returns:
            Latent representations (batch, num_latents, d_latent)
        """
        batch_size = x.size(0)

        # Project input if needed
        x = self.input_proj(x)  # (batch, n_inputs, d_latent)

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: latents attend to input
        # Query: latents, Key/Value: input
        output, _ = self.cross_attn(latents, x)

        return output


# Example usage
if __name__ == "__main__":
    batch_size = 2
    n_inputs = 50000  # Large input (e.g., pixels, audio samples)
    d_input = 3       # RGB channels
    num_latents = 512 # Small bottleneck
    d_latent = 256

    # Simulate large input
    x = torch.randn(batch_size, n_inputs, d_input)

    # Perceiver cross-attention
    perceiver = PerceiverCrossAttention(
        num_latents=num_latents,
        d_latent=d_latent,
        d_input=d_input,
        n_heads=8
    )

    # Forward pass
    latents = perceiver(x)

    print(f"Input shape: {x.shape}")        # (2, 50000, 3)
    print(f"Latent shape: {latents.shape}") # (2, 512, 256)
    print(f"Compression ratio: {n_inputs / num_latents:.1f}x")
    print("✓ Perceiver cross-attention working correctly")
```

---

## Practical Considerations

### 1. Memory and Computation

**Cross-attention complexity**: $O(n \cdot m)$ where $n$ is target length and $m$ is source length.

- **Self-attention**: $O(n^2)$ for sequence length $n$
- **Cross-attention**: $O(n \cdot m)$ for target length $n$ and source length $m$

**Memory for attention scores**:
- Self-attention: $\text{batch} \times \text{heads} \times n \times n$
- Cross-attention: $\text{batch} \times \text{heads} \times n \times m$

For encoder-decoder models with similar source and target lengths, cross-attention is comparable to self-attention in cost.

### 2. KV Caching in Cross-Attention

During autoregressive decoding, encoder outputs don't change, so we can cache cross-attention keys and values:

```python
class CrossAttentionWithKVCache(nn.Module):
    """
    Cross-attention with KV caching for efficient inference.

    During autoregressive generation, the encoder output (source) doesn't change,
    so we can precompute and cache K and V projections.
    """
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Cache for encoder K, V
        self.cached_k = None
        self.cached_v = None

    def cache_encoder_kv(self, encoder_output: torch.Tensor):
        """
        Precompute and cache K, V from encoder output.
        Call this once before autoregressive decoding.

        Args:
            encoder_output: (batch, source_len, d_model)
        """
        batch_size = encoder_output.size(0)
        source_len = encoder_output.size(1)

        # Project to K, V
        K = self.w_k(encoder_output)
        V = self.w_v(encoder_output)

        # Reshape for multi-head
        K = K.view(batch_size, source_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, source_len, self.n_heads, self.d_k).transpose(1, 2)

        # Cache
        self.cached_k = K
        self.cached_v = V

    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_output: torch.Tensor = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Args:
            decoder_state: Current decoder state (batch, 1, d_model) for generation
            encoder_output: Only needed if not using cache
            use_cache: Whether to use cached K, V

        Returns:
            Output (batch, 1, d_model)
        """
        batch_size = decoder_state.size(0)

        # Project query from decoder state
        Q = self.w_q(decoder_state)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Get K, V (from cache or compute)
        if use_cache and self.cached_k is not None:
            K = self.cached_k
            V = self.cached_v
        else:
            assert encoder_output is not None, "Need encoder_output if not using cache"
            self.cache_encoder_kv(encoder_output)
            K = self.cached_k
            V = self.cached_v

        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Combine heads and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.w_o(output)

        return output

    def clear_cache(self):
        """Clear cached K, V."""
        self.cached_k = None
        self.cached_v = None


# Example: Autoregressive generation with KV caching
if __name__ == "__main__":
    batch_size = 1
    source_len = 20
    d_model = 512
    max_gen_len = 10

    # Encoder output (computed once)
    encoder_output = torch.randn(batch_size, source_len, d_model)

    # Cross-attention with caching
    cross_attn_cached = CrossAttentionWithKVCache(d_model=d_model, n_heads=8)

    # Cache encoder K, V (do this once before generation)
    cross_attn_cached.cache_encoder_kv(encoder_output)

    # Simulate autoregressive generation
    print("Generating tokens autoregressively with KV cache:")
    decoder_state = torch.randn(batch_size, 1, d_model)  # Start token

    for step in range(max_gen_len):
        # Cross-attend to encoder (using cached K, V)
        output = cross_attn_cached(decoder_state, use_cache=True)

        # In practice, would use this to predict next token
        # decoder_state = update_with_next_token(output)
        print(f"  Step {step}: output shape {output.shape}")  # (1, 1, 512)

    print("✓ KV caching working correctly")
```

**Speedup**: By caching encoder K and V, we avoid recomputing them at each decoding step, saving significant computation.

### 3. Masking in Cross-Attention

Cross-attention masks are typically used for:

1. **Padding masks**: Prevent attention to padding tokens in source sequence
2. **Conditional masking**: Selectively attend to parts of source (e.g., attend only to relevant image regions)

```python
def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create mask for padding tokens.

    Args:
        seq: Token IDs (batch, seq_len)
        pad_idx: Padding token ID

    Returns:
        Mask (batch, 1, 1, seq_len) - True for real tokens, False for padding
    """
    # True for non-padding tokens
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
    return mask


# Example usage
if __name__ == "__main__":
    # Source sequence with padding (0 = PAD)
    # Sequence: [5, 10, 15, 20, 0, 0, 0]
    src = torch.tensor([[5, 10, 15, 20, 0, 0, 0],
                        [3, 7, 0, 0, 0, 0, 0]])

    mask = create_padding_mask(src, pad_idx=0)
    print("Source sequences:")
    print(src)
    print("\nPadding mask (True = attend, False = ignore):")
    print(mask.squeeze())

    # In cross-attention, this mask prevents decoder from attending to padding
```

### 4. Grouped-Query Cross-Attention

Modern models use Grouped-Query Attention (GQA) for efficiency (see [Multi-Head Attention](04-multi-head-attention.md)). This can also be applied to cross-attention:

```python
class GroupedQueryCrossAttention(nn.Module):
    """
    Grouped-Query Cross-Attention for reduced KV cache.

    Instead of n_heads separate K,V heads, use n_kv_heads < n_heads.
    Each KV head is shared by n_heads // n_kv_heads query heads.

    This reduces KV cache size, which is especially important for
    cross-attention when source sequences are long.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,  # Number of KV heads (< n_heads)
        dropout: float = 0.1
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(n_heads * self.d_k, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = target.size(0)
        n = target.size(1)
        m = source.size(1)

        # Project to Q, K, V
        Q = self.w_q(target).view(batch_size, n, self.n_heads, self.d_k)
        K = self.w_k(source).view(batch_size, m, self.n_kv_heads, self.d_k)
        V = self.w_v(source).view(batch_size, m, self.n_kv_heads, self.d_k)

        # Transpose for attention
        Q = Q.transpose(1, 2)  # (batch, n_heads, n, d_k)
        K = K.transpose(1, 2)  # (batch, n_kv_heads, m, d_k)
        V = V.transpose(1, 2)  # (batch, n_kv_heads, m, d_k)

        # Repeat K, V for each group
        K = K.repeat_interleave(self.n_groups, dim=1)  # (batch, n_heads, m, d_k)
        V = V.repeat_interleave(self.n_groups, dim=1)  # (batch, n_heads, m, d_k)

        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, n, -1)
        output = self.w_o(output)

        return output


# Example: Memory savings
if __name__ == "__main__":
    batch_size = 1
    n = 100      # Target length
    m = 10000    # Source length (e.g., long document or many image patches)
    d_model = 512
    n_heads = 8
    n_kv_heads = 2  # Use only 2 KV heads instead of 8

    target = torch.randn(batch_size, n, d_model)
    source = torch.randn(batch_size, m, d_model)

    # Standard cross-attention
    standard = MultiHeadCrossAttention(d_model, n_heads)

    # GQA cross-attention
    gqa = GroupedQueryCrossAttention(d_model, n_heads, n_kv_heads)

    # Compare KV cache size
    d_k = d_model // n_heads

    standard_kv_size = batch_size * m * n_heads * d_k * 2  # K and V
    gqa_kv_size = batch_size * m * n_kv_heads * d_k * 2

    print(f"Standard KV cache size: {standard_kv_size:,} elements")
    print(f"GQA KV cache size: {gqa_kv_size:,} elements")
    print(f"Memory reduction: {standard_kv_size / gqa_kv_size:.1f}x")
    print("✓ GQA cross-attention reduces KV cache significantly")
```

---

## Advanced Topics

### 1. Bidirectional Cross-Attention (Prefix LM)

Some models use bidirectional cross-attention for prefix positions while remaining causal for generation:

```python
def create_prefix_causal_mask(
    prefix_len: int,
    total_len: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Create mask for prefix language modeling.

    - Prefix tokens (0 to prefix_len-1): Bidirectional attention
    - Generation tokens (prefix_len to total_len-1): Causal attention

    This is used in models like T5 and UL2.

    Args:
        prefix_len: Length of prefix (can attend bidirectionally)
        total_len: Total sequence length

    Returns:
        Mask (total_len, total_len)
    """
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)

    # Prefix: bidirectional (all True within prefix)
    mask[:prefix_len, :prefix_len] = True

    # Generation: can see prefix + causal within generation
    for i in range(prefix_len, total_len):
        mask[i, :i+1] = True  # Can see all previous (including prefix)

    return mask


# Visualization
if __name__ == "__main__":
    prefix_len = 5
    total_len = 12

    mask = create_prefix_causal_mask(prefix_len, total_len)

    print("Prefix-Causal Mask (1 = can attend):")
    print(mask.int().numpy())
    print(f"\nPrefix tokens (0-{prefix_len-1}): Bidirectional attention")
    print(f"Generation tokens ({prefix_len}-{total_len-1}): Causal attention to all previous")
```

### 2. Cross-Attention with Relative Position Bias

For very long sequences, adding relative position information to cross-attention can help:

```python
class CrossAttentionWithRelativeBias(nn.Module):
    """
    Cross-attention with relative position bias (T5-style).

    Adds learned bias based on relative position between query and key.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Relative position bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Map relative positions to buckets.

        Args:
            relative_position: (n, m) matrix of relative positions

        Returns:
            Bucket indices (n, m)
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        # Half buckets for exact positions, half for logarithmically larger bins
        num_buckets //= 2
        buckets = (relative_position > 0).long() * num_buckets

        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Logarithmic bucketing for larger distances
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return buckets

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor
    ) -> torch.Tensor:
        batch_size = target.size(0)
        n = target.size(1)
        m = source.size(1)

        # Project to Q, K, V
        Q = self.w_q(target).view(batch_size, n, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(source).view(batch_size, m, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(source).view(batch_size, m, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add relative position bias
        # Create relative position matrix
        target_pos = torch.arange(n, device=target.device).unsqueeze(1)
        source_pos = torch.arange(m, device=source.device).unsqueeze(0)
        relative_position = source_pos - target_pos  # (n, m)

        # Get buckets and bias
        buckets = self._relative_position_bucket(relative_position)
        bias = self.relative_attention_bias(buckets)  # (n, m, n_heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, n, m)

        # Add bias to scores
        scores = scores + bias

        # Attention
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        # Combine and project
        output = output.transpose(1, 2).contiguous().view(batch_size, n, self.d_model)
        output = self.w_o(output)

        return output
```

### 3. Cross-Attention Variants in Modern Models

Different models use cross-attention differently:

| Model | Cross-Attention Usage |
|-------|----------------------|
| **BERT** | No cross-attention (encoder-only) |
| **GPT** | No cross-attention (decoder-only) |
| **T5** | Standard encoder-decoder cross-attention |
| **BART** | Standard encoder-decoder cross-attention |
| **CLIP** | Cross-modal contrastive (not traditional cross-attention) |
| **LLaVA** | Cross-attention from language to vision features |
| **Flamingo** | Interleaved cross-attention to vision features |
| **Perceiver** | Latent queries attend to large inputs |
| **Prefix LM** | Bidirectional on prefix, causal on generation |

---

## Summary

### Key Takeaways

1. **Self-Attention vs Cross-Attention**:
   - Self-attention: Q, K, V from same sequence
   - Cross-attention: Q from target, K/V from source

2. **Primary Use Cases**:
   - Encoder-decoder models (translation, summarization)
   - Multimodal models (vision + language)
   - Conditional generation (attending to context)

3. **Computational Complexity**:
   - $O(n \cdot m)$ for target length $n$ and source length $m$
   - KV caching critical for efficient inference

4. **Modern Optimizations**:
   - Grouped-Query Attention for reduced KV cache
   - KV caching for autoregressive generation
   - Relative position bias for long sequences

5. **Architecture Patterns**:
   - Traditional: Encoder-decoder with cross-attention
   - Modern: Decoder-only with occasional cross-attention for multimodality
   - Perceiver: Latent bottleneck with cross-attention

### When to Use Cross-Attention

**Use cross-attention when**:
- You have two different sequences to align (e.g., translation)
- Building multimodal models (vision + language, audio + text)
- Need conditional generation (attend to context/memory)
- Want to compress large inputs (Perceiver-style)

**Don't use cross-attention when**:
- Single sequence modeling (use self-attention)
- Decoder-only LLMs (concatenate contexts instead)
- Real-time streaming (cross-attention requires full source sequence)

---

## Exercises

1. **Implement from Scratch**: Implement cross-attention without using the code above. Verify it gives the same results.

2. **Attention Visualization**: Create a function that generates realistic cross-attention patterns for a translation task. Visualize alignment between source and target.

3. **Memory Analysis**: Calculate the memory requirements for:
   - Encoder-decoder Transformer with L=6 layers, d=512, h=8, source_len=1000, target_len=500
   - Compare memory for self-attention vs cross-attention components
   - How much memory is saved by GQA with n_kv_heads=2?

4. **Masked Cross-Attention**: Implement a function that creates a cross-attention mask where:
   - Target tokens can only attend to source tokens at or before their position
   - This is useful for monotonic alignment tasks (e.g., speech recognition)

5. **Bidirectional vs Unidirectional**: In what scenarios would you want bidirectional cross-attention vs causal cross-attention? Provide examples.

6. **Multimodal Fusion**: Design a vision-language model architecture that uses cross-attention. Where would you place cross-attention layers? Would you use it in both directions (text→image and image→text)?

7. **KV Cache Efficiency**: Implement a full autoregressive generation loop with KV caching for both self-attention and cross-attention. Measure speedup vs. naive approach.

8. **Perceiver Variants**: Extend the Perceiver implementation to:
   - Use iterative cross-attention (multiple rounds of latent refinement)
   - Add self-attention among latents between cross-attention layers

9. **Positional Encoding**: How do positional encodings interact with cross-attention? Should source and target use the same or different positional encodings? Why?

10. **Attention Pattern Analysis**: Given a trained translation model, extract and analyze cross-attention weights. Do they form meaningful alignments? Are there heads that specialize in different patterns (e.g., monotonic alignment, long-range dependencies)?

---

## References

### Key Papers

1. **Vaswani et al. (2017)**. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - Original Transformer paper introducing cross-attention for encoder-decoder models

2. **Bahdanau et al. (2015)**. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
   - Introduced attention mechanism for seq2seq (precursor to Transformer cross-attention)

3. **Luong et al. (2015)**. [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
   - Alternative attention formulations for translation

4. **Raffel et al. (2020)**. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
   - T5 model with encoder-decoder architecture and relative position bias

5. **Lewis et al. (2020)**. [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
   - BART uses standard encoder-decoder with cross-attention

6. **Jaegle et al. (2021)**. [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)
   - Perceiver architecture using cross-attention for multimodal inputs

7. **Jaegle et al. (2022)**. [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)
   - Extension of Perceiver with flexible outputs

8. **Liu et al. (2023)**. [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
   - Vision-language model using cross-attention

9. **Alayrac et al. (2022)**. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
   - Interleaved cross-attention for vision and language

10. **Radford et al. (2021)**. [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
    - Contrastive vision-language learning (different approach from cross-attention)

11. **Li et al. (2022)**. [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
    - Vision-language model with cross-attention

### Additional Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Lilian Weng
- [Cross-Attention in Multimodal Models](https://huggingface.co/blog/vision-language-pretraining) - Hugging Face Blog

### Related Chapters

- [Basic Attention](03-basic-attention.md) - Foundation of attention mechanisms
- [Multi-Head Attention](04-multi-head-attention.md) - Multiple attention heads for richer representations
- [Bidirectional vs Causal Attention](05-bidirectional-causal-attention.md) - Attention masking patterns
- [Building a Complete Transformer](11-complete-transformer.md) - Full encoder-decoder implementation
- [Multimodality](27-multimodality.md) - Cross-modal attention in modern multimodal models

---

**Next Chapter**: [Positional Encodings](07-positional-encodings.md) - How Transformers encode sequence order
