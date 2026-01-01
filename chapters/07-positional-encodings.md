# Chapter 7: Positional Encodings

Transformers fundamentally rely on attention mechanisms that treat input sequences as sets—they are **permutation invariant**. Without additional information, a transformer cannot distinguish between "the cat sat on the mat" and "mat the on sat cat the." Positional encodings solve this critical problem by injecting sequence order information into the model.

This chapter covers the foundational approaches to positional encoding, from the original sinusoidal encodings to modern learned embeddings. For advanced methods like RoPE (used in most modern LLMs), see [Chapter 8: Rotary Position Embeddings](08-rope.md).

## Table of Contents

1. [Why Position Matters](#why-position-matters)
2. [Positional Encoding Requirements](#positional-encoding-requirements)
3. [Sinusoidal Positional Encoding](#sinusoidal-positional-encoding)
4. [Learned Positional Embeddings](#learned-positional-embeddings)
5. [Comparison: Sinusoidal vs Learned](#comparison-sinusoidal-vs-learned)
6. [Relative Positional Encodings](#relative-positional-encodings)
7. [Complete Implementation](#complete-implementation)
8. [Exercises](#exercises)
9. [References](#references)

---

## Why Position Matters

### The Permutation Invariance Problem

The attention mechanism (see [Basic Attention](03-basic-attention.md)) computes relationships between tokens using queries, keys, and values:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Notice that this operation is **permutation invariant**: if we shuffle the input sequence, the attention scores remain the same (just reordered). The model has no inherent notion of token order.

```python
import torch
import torch.nn as nn

def demonstrate_permutation_invariance():
    """Show that attention is permutation invariant without positional encoding."""
    d_model = 64
    seq_len = 5

    # Simple attention (no position encoding)
    q = k = v = torch.randn(1, seq_len, d_model)

    # Compute attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    output1 = torch.matmul(attn, v)

    # Permute the input sequence
    perm = torch.tensor([2, 0, 4, 1, 3])
    q_perm = q[:, perm, :]
    k_perm = k[:, perm, :]
    v_perm = v[:, perm, :]

    # Compute attention on permuted input
    scores_perm = torch.matmul(q_perm, k_perm.transpose(-2, -1)) / (d_model ** 0.5)
    attn_perm = torch.softmax(scores_perm, dim=-1)
    output2 = torch.matmul(attn_perm, v_perm)

    # Outputs are the same (just reordered)
    print(f"Output 1 shape: {output1.shape}")
    print(f"Output 2 shape: {output2.shape}")
    print(f"Are outputs permutations of each other? {torch.allclose(output1[:, perm, :], output2)}")
    # This will print True!

if __name__ == "__main__":
    demonstrate_permutation_invariance()
```

### Why Order Matters in Language

Language is fundamentally sequential:

- **Syntax**: "The dog bit the man" ≠ "The man bit the dog"
- **Temporal relationships**: "After lunch, I went home" vs "I went home after lunch"
- **Long-range dependencies**: "The keys that I left on the table ... are missing"

Without positional information, transformers would treat all these sequences identically.

---

## Positional Encoding Requirements

A good positional encoding scheme should satisfy several properties:

### 1. Uniqueness
Each position should have a unique encoding:

```math
\text{PE}(pos_i) \neq \text{PE}(pos_j) \quad \text{for } i \neq j
```

### 2. Bounded Values
Encodings should have bounded magnitudes to prevent numerical instability:

```math
\|\text{PE}(pos)\| \leq C \quad \text{for some constant } C
```

### 3. Deterministic (for some methods)
For sinusoidal encodings, the same position should always get the same encoding, regardless of sequence length.

### 4. Relative Position Awareness (desirable)
The model should be able to learn that the relationship between positions $i$ and $j$ depends on $|i - j|$, not absolute positions.

### 5. Extrapolation (desirable)
The model should generalize to sequence lengths longer than those seen during training.

---

## Sinusoidal Positional Encoding

The original Transformer paper ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) introduced sinusoidal positional encodings.

### Mathematical Formulation

For a position $pos$ and dimension $i$, the positional encoding is:

```math
\begin{aligned}
\text{PE}_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
\text{PE}_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\end{aligned}
```

Where:

- $pos$ is the position in the sequence (0, 1, 2, ...)
- $i$ is the dimension index (0 to $d_{model}/2 - 1$)
- $d_{model}$ is the embedding dimension
- $10000$ is an empirically chosen base that provides wavelengths ranging from $2\pi$ to $10000 \cdot 2\pi$ across dimensions, creating a geometric progression of frequencies

![Position-Dimension Heatmap](../assets/diagrams/ch07-dimension-heatmap.svg)

The heatmap above illustrates how sinusoidal encoding varies across both positions (horizontal axis) and dimensions (vertical axis). Notice how:

- **Top rows** (low dimensions) show rapid alternation between values - these high-frequency components capture fine-grained position differences
- **Bottom rows** (high dimensions) change slowly across positions - these low-frequency components provide coarse positional context
- Each **column** (position) has a unique pattern across all dimensions, creating a distinctive "fingerprint"

### Intuition

The sinusoidal encoding uses different frequencies for different dimensions:

- **Low dimensions**: High frequency (changes rapidly with position)
- **High dimensions**: Low frequency (changes slowly with position)

This creates a unique "fingerprint" for each position, similar to binary encoding but with smooth, continuous values.

![Sinusoidal Positional Encoding Pattern](../assets/diagrams/ch07-sinusoidal-pattern.svg)

The visualization above shows how different dimensions oscillate at different frequencies. High-frequency dimensions (like dimension 0) change rapidly with each position, providing fine-grained positional information. Low-frequency dimensions (like dimension 120) change slowly, capturing coarse-grained positional information. Together, these create a unique pattern for each position.

### Why This Approach Works

Before implementing sinusoidal encodings, it's important to understand the theoretical foundation:

**The Problem Being Solved**: We need a function that maps position integers to continuous vectors such that:

- Each position gets a unique representation
- The model can learn to identify relative positions (e.g., "5 positions apart")
- The encoding generalizes to unseen sequence lengths

**Theoretical Justification**: The sinusoidal approach leverages trigonometric properties to achieve these goals:

1. **Uniqueness through frequency decomposition**: By using different frequencies across dimensions (geometric progression from high to low), each position gets a unique "signature." This is analogous to how Fourier analysis decomposes signals into frequency components.

2. **Linear transformation property**: The angle addition formulas for sine and cosine mean that $\text{PE}(pos + k)$ can be expressed as a linear transformation of $\text{PE}(pos)$. This makes it trivial for the model to learn relative position relationships through simple linear operations (what attention naturally does).

3. **Bounded and smooth**: Unlike incrementing integers (0, 1, 2, ...), which grow unbounded, sine and cosine values stay in [-1, 1], preventing numerical instability and ensuring the positional information doesn't dominate the semantic token embeddings.

**Relation to Alternatives**:

- **vs. Learned embeddings**: Sinusoidal is deterministic and extrapolates, but learned embeddings can be more task-adaptive
- **vs. Binary encoding**: Binary (0/1 patterns) would give unique positions but lacks smoothness for the model to interpolate
- **vs. Simple incrementing**: Position values 0, 1, 2, ... would grow unbounded and lack the linear transformation property

**Key Insight**: The genius of sinusoidal encoding is that it provides a **dense, continuous representation** where positions that are close together have similar encodings, while still maintaining uniqueness. The multiple frequencies ensure both local patterns (high frequency) and global patterns (low frequency) are captured.

### Implementation

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from 'Attention Is All You Need'.

    Each position gets a unique vector of sin/cos values at different frequencies.
    This allows the model to easily learn to attend by relative positions.

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length to precompute
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Create position indices [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create dimension indices and compute division term
        # div_term has shape (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input embeddings of shape (batch, seq_len, d_model)

        Returns:
            Embeddings with positional encoding added
        """
        # Add positional encoding to input
        # self.pe[:, :x.size(1)] extracts the first seq_len positions
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def visualize_sinusoidal_encoding(d_model: int = 128, max_len: int = 100):
    """Visualize the sinusoidal positional encoding pattern.

    This helps understand how different dimensions encode position differently.
    """
    pe_layer = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)

    # Get the positional encodings
    pe = pe_layer.pe[0].numpy()  # Shape: (max_len, d_model)

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Heatmap of all positional encodings
    im = axes[0].imshow(pe.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Dimension')
    axes[0].set_title('Sinusoidal Positional Encoding Heatmap')
    plt.colorbar(im, ax=axes[0])

    # Plot 2: A few dimensions over positions
    axes[1].plot(pe[:, 0], label='Dimension 0 (sin, high freq)')
    axes[1].plot(pe[:, 1], label='Dimension 1 (cos, high freq)')
    axes[1].plot(pe[:, 64], label='Dimension 64 (sin, mid freq)')
    axes[1].plot(pe[:, 65], label='Dimension 65 (cos, mid freq)')
    axes[1].plot(pe[:, 126], label='Dimension 126 (sin, low freq)')
    axes[1].plot(pe[:, 127], label='Dimension 127 (cos, low freq)')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Encoding Value')
    axes[1].set_title('Selected Dimensions Over Positions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/sinusoidal_encoding.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to /tmp/sinusoidal_encoding.png")


def demonstrate_relative_position_property():
    """Demonstrate that sinusoidal encoding encodes relative positions.

    Key property: PE(pos + k) can be expressed as a linear function of PE(pos).
    This means the model can learn to attend to relative positions easily.
    """
    d_model = 64
    max_len = 100

    pe_layer = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    pe = pe_layer.pe[0]  # Shape: (max_len, d_model)

    # Pick a position and an offset
    pos = 10
    offset = 5

    # Get PE(pos) and PE(pos + offset)
    pe_pos = pe[pos]
    pe_pos_offset = pe[pos + offset]

    # The key property is that we can compute dot product between any two positions
    # and it only depends on their relative distance
    dot_products = []
    for offset in range(1, 20):
        dot_prod = torch.dot(pe[pos], pe[pos + offset])
        dot_products.append(dot_prod.item())

    # Now try from a different absolute position with same offsets
    pos2 = 30
    dot_products2 = []
    for offset in range(1, 20):
        dot_prod = torch.dot(pe[pos2], pe[pos2 + offset])
        dot_products2.append(dot_prod.item())

    # These should be very similar (not exactly equal due to boundary effects)
    print("\nDot products for different relative distances:")
    print("From pos=10:", [f"{x:.4f}" for x in dot_products[:5]])
    print("From pos=30:", [f"{x:.4f}" for x in dot_products2[:5]])
    print("\nNotice the values follow similar patterns!")


if __name__ == "__main__":
    visualize_sinusoidal_encoding()
    demonstrate_relative_position_property()
```

### Key Properties

**1. Bounded Values**: All values are in $[-1, 1]$ due to sine and cosine.

**2. Deterministic**: Same position always gets same encoding.

**3. Relative Position Information**: The encoding at position $pos + k$ can be represented as a linear transformation of the encoding at position $pos$. Using the trigonometric angle addition formulas:

```math
\begin{aligned}
\sin(\alpha + \beta) &= \sin(\alpha)\cos(\beta) + \cos(\alpha)\sin(\beta) \\
\cos(\alpha + \beta) &= \cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta)
\end{aligned}
```

We can express this relationship as a matrix transformation. For each frequency $\omega_i = 1/10000^{2i/d_{model}}$, the encoding at position $pos + k$ can be written as:

```math
\begin{bmatrix}
\text{PE}_{(pos+k, 2i)} \\
\text{PE}_{(pos+k, 2i+1)}
\end{bmatrix}
=
\begin{bmatrix}
\cos(k \omega_i) & -\sin(k \omega_i) \\
\sin(k \omega_i) & \cos(k \omega_i)
\end{bmatrix}
\begin{bmatrix}
\text{PE}_{(pos, 2i)} \\
\text{PE}_{(pos, 2i+1)}
\end{bmatrix}
```

This means $\text{PE}(pos + k) = M_k \cdot \text{PE}(pos)$, where the transformation matrix $M_k$ depends only on the offset $k$, not the absolute position $pos$. This property allows the model to easily learn to attend based on relative positions, as the relationship between positions is encoded in a linear transformation that is consistent regardless of absolute position.

**4. Infinite Length**: Can generate encodings for any position, even beyond training sequence lengths.

---

## Learned Positional Embeddings

An alternative approach is to treat positional encodings as learnable parameters, similar to token embeddings (see [Embeddings](02-embeddings.md)).

### Mathematical Formulation

For each position $pos \in \{0, 1, ..., max\_len - 1\}$, learn a vector $\mathbf{p}_{pos} \in \mathbb{R}^{d_{model}}$:

```math
\text{PE}(pos) = \mathbf{p}_{pos}
```

The positional embeddings are stored in an embedding matrix $P \in \mathbb{R}^{max\_len \times d_{model}}$ and updated during training via backpropagation.

### Why Learn Position Embeddings?

**The Problem Being Solved**: While sinusoidal encodings provide a mathematically elegant solution, they impose a fixed structure on how positions are represented. But what if the optimal positional representation is task-specific? For example, in sentiment analysis, nearby words might be more critical than in machine translation.

**Theoretical Justification**: Learned positional embeddings offer flexibility through optimization:

1. **Task adaptation**: The model learns the positional structure that best serves the specific task during training. For instance, in BERT's masked language modeling, the model might learn that positions near the mask matter more than distant ones.

2. **Implicit pattern discovery**: Rather than hard-coding the sinusoidal pattern, the model discovers patterns in the data. Empirical studies show that learned embeddings often develop local smoothness (nearby positions become similar) naturally through gradient descent.

3. **Simplicity**: Conceptually simpler than sinusoidal—just another embedding table, using the same machinery as token embeddings.

**Relation to Alternatives**:

- **vs. Sinusoidal**: Trades determinism and extrapolation for task-specific flexibility
- **vs. No position encoding**: Absolutely necessary—without any positional information, transformers are order-agnostic
- **vs. Relative methods**: Simpler to implement but encodes absolute position rather than relative distances

**Key Insight**: Learned embeddings demonstrate that **position is a learnable feature** just like word semantics. The model doesn't need to be told how to encode position—it figures out the optimal representation from the training data. This works remarkably well when sequence lengths are fixed or bounded.

**Practical Consideration**: The main limitation is extrapolation—if trained on sequences up to length 512 (like BERT), the model literally has no embedding parameters for position 513. This is a hard constraint, unlike sinusoidal encodings which can generate encodings for any position.

### Implementation

```python
class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings (used in GPT-2, BERT, etc.).

    Each position has a learnable embedding vector that is trained
    alongside the model. This allows the model to learn the optimal
    positional representation for the task.

    Drawback: Cannot extrapolate beyond max_len seen during training.

    Args:
        max_len: Maximum sequence length
        d_model: Embedding dimension
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create learnable positional embedding matrix
        # This is updated during training via backpropagation
        self.position_embeddings = nn.Embedding(max_len, d_model)

        # Initialize with small random values
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings to input.

        Args:
            x: Input embeddings of shape (batch, seq_len, d_model)

        Returns:
            Embeddings with positional encoding added

        Raises:
            RuntimeError: If seq_len \gt max_len
        """
        batch_size, seq_len, d_model = x.shape

        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Expand to batch size: (1, seq_len) -> (batch, seq_len)
        positions = positions.expand(batch_size, seq_len)

        # Get positional embeddings
        pos_embeddings = self.position_embeddings(positions)

        # Add to input embeddings
        x = x + pos_embeddings
        return self.dropout(x)


def analyze_learned_positions():
    """Analyze what learned positional embeddings learn.

    This demonstrates that learned embeddings often capture:

    1. Nearby positions have similar embeddings
    2. Some periodic structure (though not as regular as sinusoidal)

    """
    # Create and "train" a simple model with learned positions
    max_len = 100
    d_model = 64

    pos_emb = LearnedPositionalEmbedding(max_len, d_model, dropout=0.0)

    # Get the learned embeddings (before training, they're random)
    with torch.no_grad():
        embeddings = pos_emb.position_embeddings.weight.numpy()

    # Compute similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(embeddings)

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Similarity heatmap
    im1 = axes[0].imshow(similarity, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Position')
    axes[0].set_title('Position Similarity Matrix (Before Training)')
    plt.colorbar(im1, ax=axes[0])

    # Plot 2: Similarity to position 0
    axes[1].plot(similarity[0], label='Similarity to position 0')
    axes[1].plot(similarity[25], label='Similarity to position 25')
    axes[1].plot(similarity[50], label='Similarity to position 50')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Similarity Patterns')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/learned_positions.png', dpi=150, bbox_inches='tight')
    print("Learned position visualization saved to /tmp/learned_positions.png")


if __name__ == "__main__":
    analyze_learned_positions()
```

### Advantages

1. **Flexibility**: The model can learn task-specific positional information
2. **Simplicity**: Just an embedding lookup, very simple implementation
3. **Performance**: Often works as well as or better than sinusoidal on tasks with fixed-length sequences

### Disadvantages

1. **No Extrapolation**: Cannot handle sequences longer than `max_len`
2. **More Parameters**: Requires $max\_len \times d_{model}$ additional parameters
3. **Less Interpretable**: No clear structure like sinusoidal encoding

---

## Comparison: Sinusoidal vs Learned

Now that we understand both approaches theoretically, let's empirically compare them to validate the trade-offs.

![Positional Encoding Methods Comparison](../assets/diagrams/ch07-encoding-comparison.svg)

The comparison above illustrates the key differences between four major positional encoding approaches:

- **Sinusoidal**: Regular wave patterns with different frequencies across dimensions
- **Learned**: More irregular patterns optimized for the specific task
- **ALiBi**: Distance-based bias matrix applied directly to attention scores
- **RoPE**: Position encoded as rotations in complex space (detailed in Chapter 8)

### Why Compare Empirically?

**The Problem**: Theory tells us sinusoidal should extrapolate better and learned should be more task-adaptive, but how significant are these differences in practice?

**What We're Testing**:

1. **Learning capability**: Can both methods learn to use positional information effectively?
2. **Extrapolation**: What happens when we test on sequences longer than training?
3. **Convergence speed**: Does one approach learn faster than the other?

**Methodology**: We'll use a toy task (predicting next position) to isolate the effect of positional encoding from other factors. While artificial, this clearly demonstrates how the model uses position information.

Let's empirically compare the two approaches:

```python
class PositionalEncodingComparison:
    """Compare sinusoidal and learned positional encodings."""

    @staticmethod
    def compare_on_toy_task(
        encoding_type: str = "sinusoidal",
        seq_len: int = 50,
        d_model: int = 64,
        num_epochs: int = 100
    ):
        """Train a simple model with different positional encodings.

        Task: Predict the next position given embeddings with position encoding.
        This is a toy task to demonstrate learning capability.
        """
        # Create positional encoding
        if encoding_type == "sinusoidal":
            pos_enc = SinusoidalPositionalEncoding(d_model, max_len=seq_len, dropout=0.0)
        else:
            pos_enc = LearnedPositionalEmbedding(max_len=seq_len, d_model=d_model, dropout=0.0)

        # Simple prediction model
        model = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len)
        )

        optimizer = torch.optim.Adam(
            list(pos_enc.parameters()) + list(model.parameters()),
            lr=0.001
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        losses = []
        for epoch in range(num_epochs):
            # Create random embeddings
            x = torch.randn(32, seq_len, d_model)

            # Add positional encoding
            x_with_pos = pos_enc(x)

            # Predict next position for each position
            # (Toy task: given position i, predict i+1)
            predictions = model(x_with_pos[:, :-1, :])  # (batch, seq_len-1, seq_len)

            # Target: next position
            targets = torch.arange(1, seq_len, device=x.device).unsqueeze(0).expand(32, -1)

            loss = criterion(predictions.reshape(-1, seq_len), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return losses

    @staticmethod
    def extrapolation_test():
        """Test extrapolation capability on longer sequences.

        Sinusoidal: Should work on any length
        Learned: Will fail on sequences longer than max_len
        """
        d_model = 64
        train_max_len = 50
        test_seq_len = 100  # Longer than training!

        # Create encodings
        sin_enc = SinusoidalPositionalEncoding(d_model, max_len=test_seq_len, dropout=0.0)
        learned_enc = LearnedPositionalEmbedding(max_len=train_max_len, d_model=d_model, dropout=0.0)

        # Test input
        x = torch.randn(1, test_seq_len, d_model)

        # Sinusoidal works fine
        try:
            sin_output = sin_enc(x)
            print(f"✓ Sinusoidal encoding: Successfully processed {test_seq_len} tokens")
        except Exception as e:
            print(f"✗ Sinusoidal encoding failed: {e}")

        # Learned will fail
        try:
            learned_output = learned_enc(x)
            print(f"✓ Learned encoding: Successfully processed {test_seq_len} tokens")
        except Exception as e:
            print(f"✗ Learned encoding failed: {e}")

        # Learned can only handle up to max_len
        x_short = torch.randn(1, train_max_len, d_model)
        learned_output_short = learned_enc(x_short)
        print(f"✓ Learned encoding: Successfully processed {train_max_len} tokens (max_len)")


def compare_methods():
    """Run all comparisons."""
    print("\n=== Training Comparison ===")
    comp = PositionalEncodingComparison()

    sin_losses = comp.compare_on_toy_task("sinusoidal")
    learned_losses = comp.compare_on_toy_task("learned")

    print(f"Sinusoidal final loss: {sin_losses[-1]:.4f}")
    print(f"Learned final loss: {learned_losses[-1]:.4f}")

    print("\n=== Extrapolation Test ===")
    comp.extrapolation_test()


if __name__ == "__main__":
    compare_methods()
```

### Summary Table

| Property | Sinusoidal | Learned |
|----------|-----------|---------|
| **Extrapolation** | ✓ Works on any length | ✗ Limited to max_len |
| **Parameters** | 0 (deterministic) | max_len × d_model |
| **Memory** | O(d_model) buffer | O(max_len × d_model) params |
| **Computation** | One-time O(max_len × d_model) precomputation | O(batch × seq_len) embedding lookups per forward pass |
| **Interpretability** | ✓ Clear frequency structure | ✗ Opaque learned patterns |
| **Training Time** | Faster (no gradient computation for PE) | Slightly slower (gradients for all positions) |
| **Performance** | Good | Often slightly better on fixed lengths |
| **Used In** | Original Transformer, many seq2seq models | GPT-2, GPT-3, BERT |

---

## Relative Positional Encodings

Both sinusoidal and learned encodings are **absolute**: they encode the absolute position in the sequence. An alternative is **relative positional encoding**, which directly encodes the distance between positions.

### Motivation

Consider the sentence: "The cat sat on the mat."

For absolute encoding:

- "cat" gets position 1
- "mat" gets position 5

For relative encoding when processing "sat":

- "cat" is -1 positions away
- "mat" is +3 positions away

Relative encodings can be more robust to position shifts and better capture linguistic relationships.

### Brief Overview

Relative positional encodings modify the attention mechanism to include position information:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + R}{\sqrt{d_k}}\right)V
```

Where $R$ is a matrix of relative position biases.

**Note**: This is just an introduction. Modern relative positional encoding methods include:

- **RoPE (Rotary Position Embeddings)**: See [Chapter 8](08-rope.md) for detailed coverage
- **ALiBi (Attention with Linear Biases)**: Simple yet effective approach covered below
- **T5's relative position biases**: Learned biases for relative distances

### ALiBi: Attention with Linear Biases

ALiBi ([Press et al., 2021](https://arxiv.org/abs/2108.12409)) provides a remarkably simple alternative to positional encodings. Instead of adding position information to the input embeddings, ALiBi adds a static, non-learned bias directly to the attention scores based on the distance between query and key positions.

#### Mathematical Formulation

The attention mechanism is modified as follows:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{bias}_{ij}\right)V
```

where the bias for position pair $(i, j)$ is:

```math
\text{bias}_{ij} = -m \cdot |i - j|
```

Here, $m$ is a head-specific slope (a constant, not learned), and $|i - j|$ is the distance between the query position $i$ and key position $j$. Different attention heads use different values of $m$, typically following a geometric sequence: $m \in \{2^{-1}, 2^{-2}, 2^{-3}, ..., 2^{-n}\}$ for $n$ heads.

#### Key Properties

**Advantages**:

1. **No positional embeddings**: Saves parameters and computation—no need to add anything to the input
2. **Excellent extrapolation**: Can train on sequences of length 512 and generalize to 2048+ tokens with minimal degradation
3. **Extremely simple**: Just a single bias addition to attention scores
4. **Head-specific distances**: Different heads can focus on different distance ranges

**Disadvantages**:

1. **Linear penalty only**: May not capture all positional relationships as well as more complex methods
2. **Less common in practice**: RoPE has become more standard for modern LLMs

**Used in**: BLOOM (176B parameter model), MPT models

#### Simple Implementation

```python
class ALiBiAttention(nn.Module):
    """Attention with Linear Biases (ALiBi).

    Instead of adding positional encodings to inputs, we add position-dependent
    biases directly to attention scores.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Compute head-specific slopes (not learned, fixed)
        # For 8 heads: [2^-1, 2^-2, 2^-3, ..., 2^-8]
        slopes = torch.tensor([2 ** (-i) for i in range(1, num_heads + 1)])
        self.register_buffer('slopes', slopes)

    def get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute ALiBi bias matrix.

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            Bias tensor of shape (num_heads, seq_len, seq_len)
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)

        # Compute distances: |i - j|
        distances = torch.abs(positions[None, :] - positions[:, None])  # (seq_len, seq_len)

        # Apply head-specific slopes: -m * |i - j|
        # slopes: (num_heads,), distances: (seq_len, seq_len)
        # Result: (num_heads, seq_len, seq_len)
        bias = -self.slopes[:, None, None] * distances[None, :, :]

        return bias

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with ALiBi.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head attention
        # Shape: (batch, num_heads, seq_len, d_k)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Add ALiBi bias
        alibi_bias = self.get_alibi_bias(seq_len, x.device)
        scores = scores + alibi_bias.unsqueeze(0)  # Broadcast over batch

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and apply to values
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)

        return output


def demonstrate_alibi():
    """Demonstrate ALiBi bias patterns."""
    num_heads = 4
    seq_len = 20

    # Create ALiBi attention module
    alibi_attn = ALiBiAttention(d_model=64, num_heads=num_heads)

    # Get bias matrix
    bias = alibi_attn.get_alibi_bias(seq_len, torch.device('cpu'))

    print(f"ALiBi bias shape: {bias.shape}")
    print(f"Head slopes: {alibi_attn.slopes.tolist()}")

    # Visualize bias patterns
    fig, axes = plt.subplots(1, num_heads, figsize=(16, 4))

    for head in range(num_heads):
        im = axes[head].imshow(bias[head].numpy(), cmap='RdYlGn', aspect='auto')
        axes[head].set_title(f'Head {head} (slope={alibi_attn.slopes[head]:.3f})')
        axes[head].set_xlabel('Key Position')
        axes[head].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[head])

    plt.suptitle('ALiBi Bias Patterns: Darker = More Negative = Less Attention', y=1.02)
    plt.tight_layout()
    plt.savefig('/tmp/alibi_patterns.png', dpi=150, bbox_inches='tight')
    print("\nALiBi visualization saved to /tmp/alibi_patterns.png")
    print("\nNotice: Diagonal is 0 (no penalty for same position)")
    print("        Bias becomes more negative farther from diagonal")
    print("        Different heads have different slopes (different 'attention ranges')")


if __name__ == "__main__":
    demonstrate_alibi()
```

The beauty of ALiBi lies in its simplicity: by penalizing attention to distant tokens linearly, the model learns to focus on nearby context while still maintaining the ability to attend to far positions when necessary. The lack of learned parameters for position encoding means the method naturally extrapolates to longer sequences than seen during training.

**When to use ALiBi vs RoPE**: For modern production LLMs, RoPE ([Chapter 8](08-rope.md)) has become the dominant choice and is used in models like LLaMA, GPT-NeoX, and PaLM. However, ALiBi remains an elegant and effective alternative, especially valued for its simplicity and strong extrapolation properties.

### T5-Style Relative Position Biases

Beyond ALiBi, another influential approach is the relative position bias used in T5 (Text-to-Text Transfer Transformer).

**The Problem Being Solved**: Both absolute encodings (sinusoidal and learned) encode "position 5" and "position 10" as fixed vectors. But linguistically, what often matters is that two words are "5 positions apart," regardless of their absolute positions. Relative position bias directly models this distance relationship.

**Theoretical Justification**:

1. **Translation invariance**: The relationship between words should depend on their relative distance, not absolute positions. "The cat sat" has the same structure whether it appears at the start or middle of a document.

2. **Learnable distance importance**: Different distances (adjacent, nearby, far) may have different importance for different heads. The model learns these importance weights through the bias parameters.

3. **Attention-level encoding**: Rather than adding position to embeddings (which then propagates through all layers possibly getting diluted), relative biases directly influence attention scores where position matters most.

**Relation to Alternatives**:

- **vs. Absolute encodings**: More robust to position shifts; models relationships rather than locations
- **vs. ALiBi**: More flexible (learned) but requires parameters; ALiBi is simpler and parameter-free
- **vs. RoPE**: T5-style is conceptually simpler but RoPE has better theoretical properties for long-range extrapolation

**Key Insight**: By making the bias depend only on $|i - j|$ (the distance), the model learns a **distance function** rather than position embeddings. This means "5 positions apart" always has the same bias, whether that's positions (0,5), (10,15), or (100,105).

**Practical Note**: T5 uses bucketing to limit the number of parameters—distances beyond a threshold are grouped into buckets (e.g., all distances \gt 128 use the same bias). This simplifies the model while retaining the relative position benefits.

```python
class SimpleRelativePositionalBias(nn.Module):
    """Simplified relative positional bias (similar to T5).

    Instead of adding position to embeddings, we add learned biases
    directly to attention scores based on relative distance.

    This is a simplified version for demonstration. For the full
    modern approach used in LLaMA, GPT-NeoX, etc., see RoPE in Chapter 8.

    Args:
        num_heads: Number of attention heads
        max_distance: Maximum relative distance to consider
    """
    def __init__(self, num_heads: int, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance

        # Learnable bias for each relative position and head
        # We use 2*max_distance + 1 to cover [-max_distance, +max_distance]
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_distance + 1)
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        """Compute relative position bias matrix.

        Args:
            seq_len: Sequence length

        Returns:
            Bias matrix of shape (num_heads, seq_len, seq_len)
        """
        # Compute relative positions
        positions = torch.arange(seq_len, device=self.relative_bias.device)
        relative_positions = positions[None, :] - positions[:, None]  # (seq_len, seq_len)

        # Clip to max_distance
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_distance,
            self.max_distance
        )

        # Shift to [0, 2*max_distance]
        relative_positions = relative_positions + self.max_distance

        # Get biases: (num_heads, seq_len, seq_len)
        bias = self.relative_bias[:, relative_positions]

        return bias


def demonstrate_relative_bias():
    """Show how relative position bias works."""
    num_heads = 4
    max_distance = 8
    seq_len = 10

    rel_bias = SimpleRelativePositionalBias(num_heads, max_distance)

    # Get bias matrix
    bias = rel_bias(seq_len)  # (num_heads, seq_len, seq_len)

    print(f"Bias shape: {bias.shape}")
    print(f"\nBias for head 0:")
    print(bias[0].detach().numpy())

    # Visualize
    fig, axes = plt.subplots(1, num_heads, figsize=(16, 4))
    for head in range(num_heads):
        im = axes[head].imshow(bias[head].detach().numpy(), cmap='RdBu', aspect='auto')
        axes[head].set_title(f'Head {head}')
        axes[head].set_xlabel('Key Position')
        axes[head].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[head])

    plt.tight_layout()
    plt.savefig('/tmp/relative_bias.png', dpi=150, bbox_inches='tight')
    print("\nRelative bias visualization saved to /tmp/relative_bias.png")


if __name__ == "__main__":
    demonstrate_relative_bias()
```

---

## Complete Implementation

Now we bring together all the concepts to show how positional encodings integrate into a full transformer architecture.

### Why a Complete Implementation?

**The Problem**: Understanding individual components (embeddings, positional encoding, attention) is crucial, but seeing how they **compose** into a working system is equally important. The complete picture shows:

- How position information flows through the model
- The interaction between token semantics and position
- Practical considerations for building production systems

**What This Demonstrates**:

1. **Component integration**: Token embeddings + positional encoding + transformer layers + output projection
2. **Flexibility**: Easy to swap between different positional encoding schemes
3. **End-to-end training**: How gradients flow back through all components

**Architecture Flow**:

```text
Input tokens (batch, seq_len)
    ↓ [Token Embedding]
Token embeddings (batch, seq_len, d_model)
    ↓ [+ Positional Encoding]
Position-aware embeddings (batch, seq_len, d_model)
    ↓ [Transformer Layers]
Contextualized representations (batch, seq_len, d_model)
    ↓ [Output Projection]
Logits (batch, seq_len, vocab_size)
```

**Key Insight**: Positional encoding happens **exactly once** at the input. After that, the position information is implicitly carried through the residual connections and attention patterns. The transformer layers don't need to re-encode position—it's already baked into the representations.

Let's put it all together with a complete example showing how positional encodings integrate with the full transformer pipeline:

```python
class TransformerWithPositionalEncoding(nn.Module):
    """Complete example: Embeddings + Positional Encoding + Transformer.

    This demonstrates how positional encodings fit into the full architecture.
    See also:

    - [Embeddings](02-embeddings.md) for token embedding details
    - [Basic Attention](03-basic-attention.md) for attention mechanism
    - [The Transformer Block](09-transformer-block.md) for full transformer details

    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        max_len: int = 5000,
        pos_encoding: str = "sinusoidal",
        dropout: float = 0.1
    ):
        super().__init__()

        # Token embeddings (see Chapter 2)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        if pos_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == "learned":
            self.pos_encoding = LearnedPositionalEmbedding(max_len, d_model, dropout)
        else:
            raise ValueError(f"Unknown positional encoding: {pos_encoding}")

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input token indices of shape (batch, seq_len)
            mask: Optional attention mask

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # 1. Token embeddings
        x = self.token_embedding(x)  # (batch, seq_len, d_model)

        # 2. Add positional encoding
        x = self.pos_encoding(x)  # (batch, seq_len, d_model)

        # 3. Apply transformer
        x = self.transformer(x, mask=mask)  # (batch, seq_len, d_model)

        # 4. Project to vocabulary
        logits = self.output_projection(x)  # (batch, seq_len, vocab_size)

        return logits


def train_toy_language_model():
    """Train a tiny language model to demonstrate positional encoding in action.

    Task: Learn to predict the next token in a sequence.
    This is a minimal example to show how positional encoding matters.
    """
    # Hyperparameters
    vocab_size = 100
    d_model = 128
    nhead = 4
    num_layers = 2
    batch_size = 16
    seq_len = 20
    num_epochs = 50

    # Create models with different positional encodings
    models = {
        "sinusoidal": TransformerWithPositionalEncoding(
            vocab_size, d_model, nhead, num_layers, pos_encoding="sinusoidal"
        ),
        "learned": TransformerWithPositionalEncoding(
            vocab_size, d_model, nhead, num_layers, pos_encoding="learned"
        )
    }

    # Training
    results = {}
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(num_epochs):
            # Generate random data (in practice, this would be real text)
            x = torch.randint(0, vocab_size, (batch_size, seq_len))

            # Predict next token
            logits = model(x[:, :-1])  # Input: all but last
            targets = x[:, 1:]  # Target: all but first

            # Compute loss
            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        results[name] = losses
        print(f"{name}: Final loss = {losses[-1]:.4f}")

    # Plot results
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss: Sinusoidal vs Learned Positional Encoding')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/tmp/training_comparison.png', dpi=150, bbox_inches='tight')
    print("\nTraining comparison saved to /tmp/training_comparison.png")


def visualize_positional_impact():
    """Visualize how positional encoding affects the embedding space.

    This shows that positions become distinguishable after adding positional encoding.
    """
    vocab_size = 100
    d_model = 64
    seq_len = 50

    # Create embeddings
    token_emb = nn.Embedding(vocab_size, d_model)
    sin_pos = SinusoidalPositionalEncoding(d_model, max_len=seq_len, dropout=0.0)

    # Random token sequence
    tokens = torch.randint(0, vocab_size, (1, seq_len))

    # Get embeddings
    emb_only = token_emb(tokens)  # (1, seq_len, d_model)
    emb_with_pos = sin_pos(emb_only)  # (1, seq_len, d_model)

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform

    dist_only = squareform(pdist(emb_only[0].detach().numpy(), metric='cosine'))
    dist_with_pos = squareform(pdist(emb_with_pos[0].detach().numpy(), metric='cosine'))

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im1 = axes[0].imshow(dist_only, cmap='viridis', aspect='auto')
    axes[0].set_title('Token Embeddings Only\n(Position-invariant)')
    axes[0].set_xlabel('Token Position')
    axes[0].set_ylabel('Token Position')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(dist_with_pos, cmap='viridis', aspect='auto')
    axes[1].set_title('Token Embeddings + Positional Encoding\n(Position-aware)')
    axes[1].set_xlabel('Token Position')
    axes[1].set_ylabel('Token Position')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig('/tmp/positional_impact.png', dpi=150, bbox_inches='tight')
    print("Positional impact visualization saved to /tmp/positional_impact.png")


if __name__ == "__main__":
    print("Training toy language model...")
    train_toy_language_model()

    print("\nVisualizing positional impact...")
    visualize_positional_impact()
```

---

## Exercises

### Exercise 1: Understanding Sinusoidal Frequencies

**Question**: Why does the sinusoidal encoding use different frequencies for different dimensions?

**Task**:

1. Modify the `SinusoidalPositionalEncoding` class to use the same frequency for all dimensions
2. Train a small model and compare performance
3. Explain why varied frequencies are beneficial

<details>
<summary>Hint</summary>
Different frequencies allow the model to attend to both fine-grained (adjacent positions) and coarse-grained (distant positions) relationships.
</details>

### Exercise 2: Extrapolation Analysis

**Question**: How well do learned embeddings extrapolate to longer sequences?

**Task**:

1. Train a model with learned positional embeddings on sequences of length 50
2. Test it on sequences of length 25, 50, 75, and 100
3. For sequences longer than 50, try:
   - Interpolating the learned embeddings
   - Using only the first 50 position embeddings repeatedly
   - Using a sinusoidal encoding for positions \gt 50
4. Compare the approaches

### Exercise 3: Relative Position Bias

**Question**: Implement a simple attention mechanism with relative position bias.

**Task**:

1. Modify the basic attention from [Chapter 3](03-basic-attention.md) to include `SimpleRelativePositionalBias`
2. Train two models: one with absolute sinusoidal encoding, one with relative bias
3. Test on sequences with different lengths
4. Which generalizes better to longer sequences?

### Exercise 4: Hybrid Approach

**Question**: Can we combine sinusoidal and learned encodings?

**Task**:

1. Implement a hybrid encoding: `PE = α * Sinusoidal + (1-α) * Learned`
2. Make α a learnable parameter
3. After training, what value does α converge to?
4. Does this outperform either method alone?

### Exercise 5: Position Embedding Analysis

**Question**: What patterns do learned position embeddings capture?

**Task**:

1. Train a model with learned positional embeddings on a real task (e.g., language modeling on a small corpus)
2. Extract the learned position embeddings
3. Analyze:
   - Nearest neighbors for each position (using cosine similarity)
   - PCA/t-SNE visualization
   - Correlation with sinusoidal encodings
4. What patterns emerge?

### Exercise 6: Implementation from Scratch

**Question**: Implement the complete positional encoding formula from scratch without using PyTorch's built-in functions (except basic operations).

**Task**:

```python
def positional_encoding_from_scratch(max_len: int, d_model: int) -> np.ndarray:
    """
    Implement sinusoidal positional encoding using only numpy.

    Args:
        max_len: Maximum sequence length
        d_model: Embedding dimension (must be even)

    Returns:
        Positional encoding matrix of shape (max_len, d_model)
    """
    # Your implementation here
    pass
```

Verify your implementation matches PyTorch's by comparing outputs.

### Exercise 7: Modern Alternatives

**Question**: Research and summarize modern positional encoding methods.

**Task**:

1. Read about RoPE ([Chapter 8](08-rope.md) or the [paper](https://arxiv.org/abs/2104.09864))
2. Read about ALiBi ([Press et al., 2021](https://arxiv.org/abs/2108.12409))
3. Create a comparison table:
   - Extrapolation capability
   - Computational cost
   - Parameter count
   - Use cases
4. When would you choose each method?

---

## References

### Key Papers

1. **Vaswani et al. (2017)**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - Original Transformer paper introducing sinusoidal positional encoding
   - Foundation for modern NLP

2. **Devlin et al. (2018)**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
   - Uses learned positional embeddings
   - Demonstrates effectiveness on multiple NLP tasks

3. **Radford et al. (2019)**: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
   - Uses learned absolute positional embeddings
   - Max length: 1024 tokens

4. **Shaw et al. (2018)**: [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
   - Introduces relative positional encoding for transformers
   - Shows improvements on translation tasks

5. **Raffel et al. (2020)**: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) (T5)
   - Uses simplified relative position biases
   - Extensive ablation studies on positional encoding

6. **Press et al. (2021)**: [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) (ALiBi)
   - Alternative to positional encodings
   - Better extrapolation to longer sequences

7. **Su et al. (2021)**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (RoPE)
   - Modern approach used in LLaMA, GPT-NeoX, and many others
   - See [Chapter 8: Rotary Position Embeddings](08-rope.md) for details

### Additional Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
  - Excellent visual explanation of positional encodings

- [Transformer Architecture: Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) by Amirhossein Kazemnejad
  - Mathematical deep dive into sinusoidal encoding properties

- [Positional Encoding - Pytorch Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
  - Official PyTorch implementation and tutorial

### Related Chapters

- **Previous**: [Cross-Attention](06-cross-attention.md)
- **Next**: [Rotary Position Embeddings (RoPE)](08-rope.md)
- **Related**: [Embeddings](02-embeddings.md), [Basic Attention](03-basic-attention.md)

---

## Summary

Positional encodings are essential for transformers to understand sequence order:

1. **Problem**: Attention is permutation-invariant
2. **Solution**: Add position information to embeddings

**Main Approaches**:

| Method | Pros | Cons | Used In |
|--------|------|------|---------|
| **Sinusoidal** | Deterministic, extrapolates, 0 params | Less flexible | Original Transformer, many seq2seq |
| **Learned** | Task-adaptive, simple | No extrapolation, more params | GPT-2/3, BERT |
| **Relative** | Position-agnostic, robust | More complex | T5, modern variants |
| **RoPE** | Excellent extrapolation, efficient | More complex math | LLaMA, modern LLMs |

**Key Insights**:

- Sinusoidal uses multiple frequencies to encode both local and global position information
- Learned embeddings are flexible but don't extrapolate
- Relative methods encode distance between positions rather than absolute positions
- Modern LLMs (LLaMA, GPT-NeoX) use RoPE for better extrapolation

**Next Steps**:

- Study [RoPE (Chapter 8)](08-rope.md) for the modern approach used in most current LLMs
- See [The Transformer Block (Chapter 9)](09-transformer-block.md) to understand how positional encodings fit in the complete architecture

---

**Navigation**:

- Previous: [Cross-Attention](06-cross-attention.md)
- Next: [Rotary Position Embeddings (RoPE)](08-rope.md)
- [Back to Table of Contents](../README.md)
