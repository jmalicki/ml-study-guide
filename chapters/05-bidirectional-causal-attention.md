# Chapter 5: Bidirectional vs Causal Attention

Understanding when and how to mask attention is crucial for building effective transformer models. This chapter explores two fundamental attention patterns: bidirectional (full) attention used in BERT-style encoders, and causal (masked) attention used in GPT-style autoregressive models.

## Table of Contents

1. [Overview](#overview)
2. [Bidirectional (Full) Attention](#bidirectional-full-attention)
   - [Concept and Use Cases](#concept-and-use-cases)
   - [Mathematical Formulation](#mathematical-formulation)
   - [Implementation](#implementation)
3. [Causal (Masked) Attention](#causal-masked-attention)
   - [Autoregressive Modeling](#autoregressive-modeling)
   - [Causal Mask Construction](#causal-mask-construction)
   - [Implementation](#causal-implementation)
4. [Attention Masks in Detail](#attention-masks-in-detail)
   - [Mask Types](#mask-types)
   - [Combining Masks](#combining-masks)
   - [Padding Masks](#padding-masks)
5. [Encoder vs Decoder Architectures](#encoder-vs-decoder-architectures)
6. [Practical Considerations](#practical-considerations)
7. [Performance Implications](#performance-implications)
8. [Complete Examples](#complete-examples)
9. [Summary](#summary)
10. [Exercises](#exercises)

---

## Overview

Attention mechanisms allow tokens to attend to other tokens in a sequence, but not all attention patterns are appropriate for all tasks:

| Attention Type | Description | Used In | Key Property |
|----------------|-------------|---------|--------------|
| **Bidirectional** | Each token attends to all tokens | BERT, encoders | Context from both directions |
| **Causal** | Each token attends only to previous tokens | GPT, decoders | Cannot see future tokens |
| **Cross-Attention** | Attend to different sequence | Encoder-decoder | See [Chapter 6](06-cross-attention.md) |

The choice between bidirectional and causal attention fundamentally determines what tasks your model can perform:

- **Bidirectional attention** is ideal for understanding tasks: classification, named entity recognition, question answering
- **Causal attention** is essential for generation tasks: text generation, code completion, language modeling

---

## Bidirectional (Full) Attention

### Concept and Use Cases

Bidirectional attention allows each token to attend to **all** tokens in the sequence, including both previous and future tokens. This provides rich contextual information from both directions.

**Example:** Consider the sentence "The bank by the river was steep."

With bidirectional attention:
- "bank" can attend to both "river" (right) and "the" (left)
- This helps disambiguate "bank" (riverbank vs financial institution)

**Use Cases:**
1. **Text Classification**: Sentiment analysis, topic classification
2. **Named Entity Recognition (NER)**: Identifying entities in text
3. **Question Answering**: Understanding context to extract answers
4. **Masked Language Modeling**: BERT-style pre-training
5. **Encoders in Seq2Seq**: Processing source sequences

### Mathematical Formulation

Recall from [Chapter 3: Basic Attention](03-basic-attention.md) that scaled dot-product attention is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

For bidirectional attention, we compute attention scores for **all** pairs of tokens:

$$
\text{Attention}_{\text{bi}}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where the attention score matrix $A = QK^T / \sqrt{d_k}$ is a full $n \times n$ matrix (where $n$ is sequence length), with no masking applied.

**Attention Score Matrix (Bidirectional):**

For a sequence of length 4, all positions can attend to all positions:

```
        Token 0  Token 1  Token 2  Token 3
Token 0    ✓        ✓        ✓        ✓
Token 1    ✓        ✓        ✓        ✓
Token 2    ✓        ✓        ✓        ✓
Token 3    ✓        ✓        ✓        ✓
```

### Implementation

```python
import torch
import torch.nn as nn
import math

class BidirectionalAttention(nn.Module):
    """Bidirectional (full) attention mechanism.

    Each token can attend to all tokens in the sequence,
    including both past and future tokens.

    Args:
        d_model: Dimension of the model
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # (batch, seq_len, d_model)
        key: torch.Tensor,    # (batch, seq_len, d_model)
        value: torch.Tensor,  # (batch, seq_len, d_model)
        mask: torch.Tensor = None  # Optional padding mask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional padding mask (1 for valid, 0 for padding)

        Returns:
            output: Attention output (batch, seq_len, d_model)
            attention_weights: Attention weights (batch, seq_len, seq_len)
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))  # (batch, seq_len, seq_len)
        scores = scores / math.sqrt(self.d_model)

        # Apply padding mask if provided
        if mask is not None:
            # mask shape: (batch, seq_len) or (batch, 1, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len)
            # Expand mask to (batch, seq_len, seq_len)
            mask = mask.unsqueeze(1).expand(-1, scores.size(1), -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


# Example usage
def test_bidirectional_attention():
    """Test bidirectional attention on a simple sequence."""
    batch_size = 2
    seq_len = 5
    d_model = 8

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize attention
    attention = BidirectionalAttention(d_model)

    # Forward pass (Q, K, V all from same input for self-attention)
    output, attn_weights = attention(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"\nAttention weights for first item in batch:")
    print(attn_weights[0].detach().numpy())
    print(f"\nEach row sums to 1: {attn_weights[0].sum(dim=-1)}")

if __name__ == "__main__":
    test_bidirectional_attention()
```

**Key Points:**
- No masking means all positions can attend to all other positions
- Each row of the attention matrix sums to 1 (after softmax)
- Padding masks can still be applied to ignore padding tokens

---

## Causal (Masked) Attention

### Autoregressive Modeling

Causal attention is essential for autoregressive models, where we generate one token at a time and cannot "peek" at future tokens during training or generation.

**Why Causal?**

When training a language model to predict the next token, we must ensure the model cannot see future tokens:

```
Input:     "The cat sat on the"
Target:    "cat sat on the mat"
           ↑   ↑   ↑   ↑   ↑
Predict:   cat sat on the mat (without seeing future!)
```

If we allowed the model to see "mat" when predicting "cat", it would cheat and learn nothing useful.

**Autoregressive Generation:**

During generation, we produce tokens sequentially:

```
Step 1: "The" → predict "cat"
Step 2: "The cat" → predict "sat"
Step 3: "The cat sat" → predict "on"
...
```

At each step, we can only use previously generated tokens.

### Causal Mask Construction

A causal mask is a lower triangular matrix that prevents attention to future positions:

```python
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal (lower triangular) mask.

    Args:
        seq_len: Sequence length

    Returns:
        mask: Boolean mask of shape (seq_len, seq_len)
              True for positions that can be attended to
              False for future positions that should be masked
    """
    # Create lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.bool()


# Example: Causal mask for sequence length 4
mask = create_causal_mask(4)
print("Causal mask (1 = can attend, 0 = masked):")
print(mask.int())
```

Output:
```
Causal mask (1 = can attend, 0 = masked):
tensor([[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]])
```

**Visualization:**

```
        Token 0  Token 1  Token 2  Token 3
Token 0    ✓        ✗        ✗        ✗     (can only see itself)
Token 1    ✓        ✓        ✗        ✗     (can see 0 and 1)
Token 2    ✓        ✓        ✓        ✗     (can see 0, 1, 2)
Token 3    ✓        ✓        ✓        ✓     (can see all)
```

### Causal Implementation

```python
class CausalAttention(nn.Module):
    """Causal (masked) attention for autoregressive models.

    Each token can only attend to previous tokens and itself,
    preventing information flow from future positions.

    Args:
        d_model: Dimension of the model
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # We'll create the mask on-the-fly to handle variable sequence lengths

    def forward(
        self,
        query: torch.Tensor,  # (batch, seq_len, d_model)
        key: torch.Tensor,    # (batch, seq_len, d_model)
        value: torch.Tensor,  # (batch, seq_len, d_model)
        mask: torch.Tensor = None  # Optional additional mask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional additional mask (e.g., padding mask)

        Returns:
            output: Attention output (batch, seq_len, d_model)
            attention_weights: Attention weights (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))  # (batch, seq_len, seq_len)
        scores = scores / math.sqrt(self.d_model)

        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device))
        causal_mask = causal_mask.bool()

        # Apply causal mask
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Apply additional mask if provided (e.g., padding mask)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len)
            mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


# Example usage
def test_causal_attention():
    """Test causal attention and visualize masking."""
    batch_size = 1
    seq_len = 5
    d_model = 8

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize attention
    attention = CausalAttention(d_model, dropout=0.0)  # No dropout for visualization

    # Forward pass
    output, attn_weights = attention(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"\nCausal attention weights (each row sums to 1):")
    print(attn_weights[0].detach().numpy())
    print(f"\nNote: Upper triangle is zero (masked out)")

if __name__ == "__main__":
    test_causal_attention()
```

---

## Attention Masks in Detail

### Mask Types

There are several types of masks commonly used in transformers:

#### 1. Causal Mask (Look-Ahead Mask)

Prevents attending to future positions:

```python
def causal_mask(seq_len: int) -> torch.Tensor:
    """Lower triangular mask for autoregressive models."""
    return torch.tril(torch.ones(seq_len, seq_len)).bool()
```

#### 2. Padding Mask

Prevents attending to padding tokens:

```python
def padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Create mask for padding tokens.

    Args:
        seq: Input sequence (batch, seq_len) with token indices
        pad_idx: Index used for padding (default: 0)

    Returns:
        mask: Boolean mask (batch, seq_len), True for real tokens
    """
    return seq != pad_idx
```

#### 3. Custom Masks

You can create custom masks for specific patterns:

```python
def block_diagonal_mask(seq_len: int, block_size: int) -> torch.Tensor:
    """Create a block diagonal mask.

    Useful for processing multiple independent sequences in parallel.
    """
    mask = torch.zeros(seq_len, seq_len)
    num_blocks = seq_len // block_size
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        mask[start:end, start:end] = 1
    return mask.bool()
```

### Combining Masks

Often, you need to combine multiple masks (e.g., causal + padding):

```python
def combine_masks(
    causal_mask: torch.Tensor,  # (seq_len, seq_len)
    padding_mask: torch.Tensor,  # (batch, seq_len)
) -> torch.Tensor:
    """Combine causal and padding masks.

    Args:
        causal_mask: Causal mask (seq_len, seq_len)
        padding_mask: Padding mask (batch, seq_len)

    Returns:
        combined_mask: Combined mask (batch, seq_len, seq_len)
    """
    batch_size, seq_len = padding_mask.shape

    # Expand causal mask to batch dimension
    causal = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

    # Expand padding mask to attention dimension
    # Padding mask: (batch, seq_len) -> (batch, 1, seq_len) -> (batch, seq_len, seq_len)
    padding = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)

    # Combine with logical AND
    combined = causal & padding

    return combined


# Example: Combining masks
def test_combined_masks():
    """Test combining causal and padding masks."""
    seq_len = 6
    batch_size = 2

    # Create sequences with padding (0 = padding)
    seq = torch.tensor([
        [1, 2, 3, 4, 5, 0],  # Last token is padding
        [1, 2, 3, 0, 0, 0],  # Last three tokens are padding
    ])

    # Create masks
    causal = causal_mask(seq_len)
    padding = padding_mask(seq)
    combined = combine_masks(causal, padding)

    print("Causal mask:")
    print(causal.int())
    print("\nPadding mask (batch 0):")
    print(padding[0].int())
    print("\nPadding mask (batch 1):")
    print(padding[1].int())
    print("\nCombined mask (batch 0):")
    print(combined[0].int())
    print("\nCombined mask (batch 1):")
    print(combined[1].int())

if __name__ == "__main__":
    test_combined_masks()
```

### Padding Masks

Padding masks ensure that padding tokens don't contribute to attention:

```python
class AttentionWithPadding(nn.Module):
    """Attention with proper padding mask handling.

    Demonstrates how to handle variable-length sequences with padding.
    """
    def __init__(self, d_model: int, causal: bool = False, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # (batch, seq_len, d_model)
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None  # (batch, seq_len), True for valid tokens
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: QKV tensors
            key_padding_mask: Mask for padding tokens (True for valid, False for padding)
        """
        batch_size, seq_len, _ = query.shape

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)

        # Create mask (starts as all ones)
        mask = torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool)

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device))
            mask = mask & causal_mask.bool()

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq_len)
            # We need to broadcast this to (batch, seq_len, seq_len)
            # A position can attend to a key only if that key is not padding
            padding_mask = key_padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
            mask = mask.unsqueeze(0) & padding_mask
        else:
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply mask to scores
        scores = scores.masked_fill(~mask, float('-inf'))

        # Compute attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Replace NaN values (can occur if entire row is masked) with 0
        attention_weights = attention_weights.masked_fill(torch.isnan(attention_weights), 0.0)

        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
```

---

## Encoder vs Decoder Architectures

Understanding the relationship between attention types and architecture is crucial:

### BERT-Style Encoder (Bidirectional)

```python
class EncoderLayer(nn.Module):
    """Transformer encoder layer with bidirectional attention.

    Used in BERT-style models for understanding tasks.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Optional padding mask
        """
        # Self-attention with residual connection
        # Note: No causal mask, so this is bidirectional
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
```

### GPT-Style Decoder (Causal)

```python
class DecoderLayer(nn.Module):
    """Transformer decoder layer with causal attention.

    Used in GPT-style models for generation tasks.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Optional padding mask (causal mask is automatic)
        """
        batch_size, seq_len, _ = x.shape

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Self-attention with causal mask
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=mask
        )
        x = self.norm1(x + attn_out)

        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
```

**Key Differences:**

| Aspect | Encoder (BERT) | Decoder (GPT) |
|--------|----------------|---------------|
| Attention Type | Bidirectional | Causal |
| Use Case | Understanding | Generation |
| Training Objective | Masked LM (MLM) | Next token prediction |
| Can see future? | Yes | No |
| Examples | BERT, RoBERTa, ALBERT | GPT-2/3/4, LLaMA, Claude |

See [Chapter 11: Building a Complete Transformer](11-complete-transformer.md) for full encoder-decoder architectures.

---

## Practical Considerations

### When to Use Bidirectional Attention

Use bidirectional attention when:

1. **Task requires full context**: Classification, NER, QA
2. **No generation needed**: The model doesn't produce sequences
3. **Training with MLM**: Using masked language modeling
4. **Fixed-length outputs**: Like classification labels

**Example Tasks:**
- Sentiment classification: "This movie is great!" → Positive
- Named Entity Recognition: "Apple is in Cupertino" → [ORG] [LOC]
- Question Answering: Extract answer span from context

### When to Use Causal Attention

Use causal attention when:

1. **Autoregressive generation**: Producing sequences token-by-token
2. **Language modeling**: Predicting next tokens
3. **Preventing information leakage**: Training must match inference
4. **Streaming applications**: Processing text incrementally

**Example Tasks:**
- Text generation: "Once upon a time" → continue the story
- Code completion: "def fibonacci(" → complete the function
- Language modeling: Evaluate likelihood of sequences

### Hybrid Approaches

Some models use both:

**T5 (Encoder-Decoder):**
- Encoder: Bidirectional attention on input
- Decoder: Causal self-attention + cross-attention to encoder

**Prefix LM:**
- Bidirectional attention on prefix (prompt)
- Causal attention on generated tokens

```python
def create_prefix_lm_mask(seq_len: int, prefix_len: int) -> torch.Tensor:
    """Create mask for prefix language modeling.

    Tokens in the prefix can attend bidirectionally to the prefix.
    Tokens after the prefix use causal attention to everything.

    Args:
        seq_len: Total sequence length
        prefix_len: Length of the bidirectional prefix

    Returns:
        mask: Attention mask (seq_len, seq_len)
    """
    mask = torch.zeros(seq_len, seq_len)

    # Prefix tokens can attend to all prefix tokens (bidirectional)
    mask[:prefix_len, :prefix_len] = 1

    # Generated tokens use causal attention to everything
    for i in range(prefix_len, seq_len):
        mask[i, :i+1] = 1  # Can attend to prefix + previous generated tokens

    return mask.bool()


# Example: Prefix LM mask
prefix_mask = create_prefix_lm_mask(seq_len=8, prefix_len=3)
print("Prefix LM mask (prefix_len=3):")
print(prefix_mask.int())
```

---

## Performance Implications

### Computational Complexity

Both bidirectional and causal attention have the same asymptotic complexity:

$$
\text{Time Complexity: } O(n^2 d)
$$
$$
\text{Space Complexity: } O(n^2)
$$

where $n$ is sequence length and $d$ is model dimension.

**However**, causal attention can be optimized during **autoregressive generation**:

### KV Caching for Causal Attention

During generation, causal attention allows for KV caching:

```python
class CausalAttentionWithCache(nn.Module):
    """Causal attention with KV caching for efficient generation.

    During autoregressive generation, we can cache previous K,V values
    and only compute new ones for the current token.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        query: torch.Tensor,  # (batch, 1, d_model) for generation
        key: torch.Tensor,    # (batch, current_len, d_model)
        value: torch.Tensor,  # (batch, current_len, d_model)
        cached_k: torch.Tensor = None,  # Previous keys
        cached_v: torch.Tensor = None,  # Previous values
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query for current token
            key: Key for current token
            value: Value for current token
            cached_k: Cached keys from previous tokens
            cached_v: Cached values from previous tokens

        Returns:
            output: Attention output
            new_cached_k: Updated key cache
            new_cached_v: Updated value cache
        """
        # Concatenate with cache
        if cached_k is not None:
            key = torch.cat([cached_k, key], dim=1)
            value = torch.cat([cached_v, value], dim=1)

        # Compute attention (query is only for current token)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)
        # No need for causal mask since query is only current token

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)

        return output, key, value  # Return updated cache


def demonstrate_kv_caching():
    """Demonstrate KV caching for efficient generation."""
    d_model = 64
    attention = CausalAttentionWithCache(d_model)

    # Initial cache
    cached_k, cached_v = None, None

    # Generate 5 tokens
    for i in range(5):
        # Current token query, key, value
        q = torch.randn(1, 1, d_model)  # Single token
        k = torch.randn(1, 1, d_model)
        v = torch.randn(1, 1, d_model)

        # Forward with cache
        output, cached_k, cached_v = attention(q, k, v, cached_k, cached_v)

        print(f"Step {i+1}: Cache size = {cached_k.shape[1]}")

    print(f"\nFinal cache contains keys/values for all {cached_k.shape[1]} tokens")

if __name__ == "__main__":
    demonstrate_kv_caching()
```

**Performance Benefit:**

- Without caching: $O(n^2)$ operations per new token
- With caching: $O(n)$ operations per new token

This is why GPT-style models (causal) can generate efficiently, while BERT-style models (bidirectional) cannot do incremental generation.

### Memory Considerations

**Training:**
- Both require $O(n^2)$ memory for attention matrices
- Causal attention can use Flash Attention optimizations (see [Chapter 12: Flash Attention](12-flash-attention.md))

**Inference:**
- Bidirectional: Always computes full $O(n^2)$ attention
- Causal: Can use KV caching to reduce compute (not memory of cache itself)

---

## Complete Examples

### Example 1: BERT-Style Masked Language Modeling

```python
class BERTSelfAttention(nn.Module):
    """Self-attention for BERT (bidirectional).

    This is used in masked language modeling where tokens can
    attend to all other tokens in both directions.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, d_model)
        padding_mask: torch.Tensor = None  # (batch, seq_len)
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply padding mask if provided (no causal mask!)
        if padding_mask is not None:
            # padding_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights and output
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


# Example: Masked Language Modeling
def test_bert_mlm():
    """Demonstrate BERT-style masked language modeling."""
    batch_size = 2
    seq_len = 10
    d_model = 64
    n_heads = 8

    # Create input with some masked tokens
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize BERT attention
    bert_attn = BERTSelfAttention(d_model, n_heads)

    # Forward pass (bidirectional)
    output = bert_attn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Each token attended to ALL other tokens (bidirectional)")

if __name__ == "__main__":
    test_bert_mlm()
```

### Example 2: GPT-Style Autoregressive Generation

```python
class GPTSelfAttention(nn.Module):
    """Self-attention for GPT (causal).

    This is used in autoregressive language modeling where tokens
    can only attend to previous tokens.
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Register causal mask as buffer (not a parameter)
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, d_model)
        padding_mask: torch.Tensor = None  # (batch, seq_len)
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (always for GPT)
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Apply padding mask if provided
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights and output
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


# Example: Autoregressive generation
def test_gpt_generation():
    """Demonstrate GPT-style autoregressive generation."""
    batch_size = 1
    seq_len = 10
    d_model = 64
    n_heads = 8

    # Create input
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize GPT attention
    gpt_attn = GPTSelfAttention(d_model, n_heads)

    # Forward pass (causal)
    output = gpt_attn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Each token attended to ONLY previous tokens (causal)")
    print("This ensures the model can't cheat during training")

if __name__ == "__main__":
    test_gpt_generation()
```

### Example 3: Comparing Both

```python
def compare_attention_patterns():
    """Visualize the difference between bidirectional and causal attention."""
    import matplotlib.pyplot as plt

    seq_len = 8
    d_model = 64

    # Create sample input
    x = torch.randn(1, seq_len, d_model)

    # Bidirectional attention
    bi_attn = BidirectionalAttention(d_model, dropout=0.0)
    _, bi_weights = bi_attn(x, x, x)

    # Causal attention
    causal_attn = CausalAttention(d_model, dropout=0.0)
    _, causal_weights = causal_attn(x, x, x)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bidirectional
    im1 = ax1.imshow(bi_weights[0].detach().numpy(), cmap='viridis')
    ax1.set_title('Bidirectional Attention\n(All positions visible)')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1)

    # Causal
    im2 = ax2.imshow(causal_weights[0].detach().numpy(), cmap='viridis')
    ax2.set_title('Causal Attention\n(Only past positions visible)')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig('attention_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to attention_comparison.png")

if __name__ == "__main__":
    compare_attention_patterns()
```

---

## Summary

### Key Takeaways

1. **Bidirectional Attention**:
   - Each token can attend to all tokens in the sequence
   - Used for understanding tasks (classification, NER, QA)
   - Examples: BERT, RoBERTa, encoders in seq2seq models
   - Cannot be used for autoregressive generation

2. **Causal Attention**:
   - Each token can only attend to previous tokens and itself
   - Essential for autoregressive generation
   - Used in language modeling and text generation
   - Examples: GPT-2/3/4, LLaMA, Claude
   - Enables KV caching for efficient generation

3. **Attention Masks**:
   - **Causal mask**: Lower triangular matrix, prevents future attention
   - **Padding mask**: Prevents attention to padding tokens
   - **Combined masks**: Use logical AND to combine multiple masks
   - Masks are applied before softmax by setting masked positions to $-\infty$

4. **Architecture Patterns**:
   - **Encoder-only** (BERT): Bidirectional attention throughout
   - **Decoder-only** (GPT): Causal attention throughout
   - **Encoder-decoder** (T5): Bidirectional in encoder, causal + cross-attention in decoder

5. **Performance**:
   - Same computational complexity: $O(n^2 d)$
   - Causal attention enables KV caching: $O(n)$ per token during generation
   - Both can use Flash Attention optimizations

### Choosing the Right Attention

| Criterion | Bidirectional | Causal |
|-----------|---------------|--------|
| Task type | Understanding | Generation |
| Training objective | MLM, classification | Next token prediction |
| Can generate text? | No | Yes |
| Inference efficiency | Standard | KV caching |
| Context | Full sequence | Left-to-right |

### References

**Key Papers:**
1. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - Original transformer with both encoder (bidirectional) and decoder (causal)

2. Devlin et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
   - Bidirectional attention for language understanding

3. Radford et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
   - Causal attention for language modeling

4. Brown et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)
   - Scaling causal language models

5. Liu et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
   - Improved bidirectional pre-training

**Related Chapters:**
- [Chapter 3: Basic Attention](03-basic-attention.md) - Foundation
- [Chapter 4: Multi-Head Attention](04-multi-head-attention.md) - Multi-head with masking
- [Chapter 6: Cross-Attention](06-cross-attention.md) - Attention between sequences
- [Chapter 11: Building a Complete Transformer](11-complete-transformer.md) - Full architectures
- [Chapter 12: Flash Attention](12-flash-attention.md) - Efficient attention computation

---

## Exercises

### Exercise 1: Implement Custom Masks

Create a function that generates a "future window" mask where each position can see the next K positions (in addition to all previous positions):

```python
def future_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """Create a mask allowing limited future attention.

    Each position can attend to:
    - All previous positions
    - Itself
    - The next window_size positions

    Args:
        seq_len: Sequence length
        window_size: Number of future positions to see

    Returns:
        mask: Boolean mask (seq_len, seq_len)
    """
    # TODO: Implement this
    pass

# Test your implementation
mask = future_window_mask(8, window_size=2)
print(mask.int())
```

### Exercise 2: Analyze Attention Patterns

Given a sequence "The quick brown fox jumps over the lazy dog", implement code to:
1. Compute attention weights using bidirectional attention
2. Compute attention weights using causal attention
3. Identify which words "fox" attends to most in each case
4. Explain the differences

### Exercise 3: KV Cache Memory Calculation

Calculate the memory required for KV caching for a GPT-style model:
- Parameters: 7B total
- Layers: 32
- Heads: 32
- Head dimension: 128
- Context length: 4096 tokens
- Batch size: 8
- Precision: FP16

Show your work and compare to the model weight memory.

### Exercise 4: Implement Sliding Window Attention

Implement a sliding window attention mask where each position can only attend to the previous W positions:

```python
def sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """Create sliding window attention mask.

    Each position can attend to:
    - Itself
    - Previous window_size positions

    This is used in models like Mistral for efficient long-context attention.

    Args:
        seq_len: Sequence length
        window_size: Size of the sliding window

    Returns:
        mask: Boolean mask (seq_len, seq_len)
    """
    # TODO: Implement this
    pass

# Test
mask = sliding_window_mask(10, window_size=3)
print(mask.int())
```

### Exercise 5: Prefix Language Modeling

Implement a complete attention module that supports prefix language modeling:
- Given a prefix length, apply bidirectional attention within the prefix
- Apply causal attention to the rest of the sequence (attending to prefix + previous tokens)
- Test with a concrete example

### Exercise 6: Attention Mask Debugging

Given the following code that isn't working correctly, identify and fix the bug:

```python
def broken_causal_attention(query, key, value):
    """This code has a bug - find and fix it!"""
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))

    # Create causal mask
    seq_len = query.size(1)
    mask = torch.triu(torch.ones(seq_len, seq_len))

    # Apply mask
    scores = scores.masked_fill(mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, value)
    return output
```

### Exercise 7: Multi-Head Causal Attention

Extend the `CausalAttention` class to support multi-head attention as shown in [Chapter 4: Multi-Head Attention](04-multi-head-attention.md). Ensure the causal mask is correctly applied across all heads.

### Exercise 8: Compare Training Efficiency

Implement a simple training loop for both:
1. Bidirectional attention (BERT-style with masked tokens)
2. Causal attention (GPT-style with next token prediction)

Measure and compare:
- Time per batch
- Memory usage
- Convergence speed (loss decrease)

Use a small toy dataset and model (d_model=128, 2 layers).

---

**Next Chapter:** [Chapter 6: Cross-Attention](06-cross-attention.md) - Learn about attention between different sequences, used in encoder-decoder models and multimodal architectures.
