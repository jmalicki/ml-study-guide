# Chapter 2: Embeddings

Embeddings are the foundation of modern NLP and LLMs. They transform discrete tokens (words, subwords, or characters) into continuous vector representations that neural networks can process. This chapter covers the evolution from classical word embeddings to modern learned embeddings used in transformers.

## Table of Contents

1. [Introduction](#introduction)
2. [Why Embeddings?](#why-embeddings)
3. [Classical Word Embeddings](#classical-word-embeddings)
   - [Word2Vec](#word2vec)
   - [GloVe](#glove)
   - [Key Insights](#key-insights-from-classical-embeddings)
4. [Learned Embeddings in Neural Networks](#learned-embeddings-in-neural-networks)
5. [Embedding Layers in PyTorch](#embedding-layers-in-pytorch)
6. [Embeddings in Modern LLMs](#embeddings-in-modern-llms)
7. [Advanced Topics](#advanced-topics)
8. [Summary](#summary)
9. [Exercises](#exercises)

---

## Introduction

After tokenization (see [Tokenization](01-tokenization.md)), we have a sequence of discrete token IDs. But neural networks operate on continuous values. **Embeddings** bridge this gap by mapping each discrete token to a dense, continuous vector.

For example, with a vocabulary of 50,000 tokens and an embedding dimension of 768:
- Token ID `42` → Vector of shape `(768,)` with learnable values
- Token ID `1337` → Different vector of shape `(768,)` with learnable values

These vectors are learned during training to capture semantic and syntactic relationships between tokens.

---

## Why Embeddings?

### The Problem with One-Hot Encoding

A naive approach would be to use one-hot encoding:

```python
import torch

vocab_size = 50000
token_id = 42

# One-hot encoding
one_hot = torch.zeros(vocab_size)
one_hot[token_id] = 1.0

print(f"Shape: {one_hot.shape}")  # (50000,)
print(f"Non-zero elements: {one_hot.sum()}")  # 1.0
```

**Problems with one-hot encoding:**

1. **High dimensionality**: For a vocabulary of 50K tokens, each token is represented by a 50K-dimensional vector
2. **Sparsity**: 99.998% of values are zeros (wasteful storage and computation)
3. **No semantic relationships**: The distance between any two tokens is always the same
   - `distance("cat", "dog")` = `distance("cat", "car")` = $\sqrt{2}$
   - Can't capture that "cat" and "dog" are more similar than "cat" and "car"

### The Solution: Dense Embeddings

Instead, we use **dense embeddings**:

```python
vocab_size = 50000
embedding_dim = 768

# Embedding matrix: each row is a token's embedding
embedding_matrix = torch.randn(vocab_size, embedding_dim)

# Look up token 42
token_embedding = embedding_matrix[42]

print(f"Shape: {token_embedding.shape}")  # (768,)
print(f"All values non-zero: {(token_embedding != 0).all()}")  # True
```

**Benefits:**

1. **Compact**: 768 dimensions instead of 50,000
2. **Dense**: All values are meaningful
3. **Learnable**: Embeddings are optimized to capture relationships
4. **Semantic**: Similar tokens have similar embeddings (after training)

---

## Classical Word Embeddings

Before transformers, researchers developed methods to learn word embeddings from large corpora. While modern LLMs learn embeddings end-to-end, understanding these classical methods provides important intuition.

### Word2Vec

Word2Vec (Mikolov et al., 2013) introduced two efficient methods for learning word embeddings:

#### Skip-Gram Model

**Intuition**: Predict context words from a center word.

**Objective**: Given a center word $w_t$, maximize the probability of observing context words $w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k}$:

$$
\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-k \leq j \leq k, j \neq 0} \log p(w_{t+j} | w_t)
$$

where the probability is modeled as:

$$
p(w_O | w_I) = \frac{\exp(\mathbf{v}_{w_O}^\top \mathbf{v}_{w_I})}{\sum_{w=1}^{V} \exp(\mathbf{v}_w^\top \mathbf{v}_{w_I})}
$$

**Key Insight**: Words that appear in similar contexts have similar embeddings.

#### CBOW (Continuous Bag of Words)

**Intuition**: Predict center word from context words (reverse of Skip-Gram).

**Objective**: Given context words, predict the center word:

$$
\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \log p(w_t | w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k})
$$

**Simplified Implementation (Skip-Gram concept):**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    """Simplified Skip-Gram Word2Vec model.

    Predicts context words from center word.
    Uses negative sampling for efficient training.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        # Each word has two embeddings: as center word and as context word
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with small random values
        self.center_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)

    def forward(
        self,
        center_word: torch.Tensor,  # (batch_size,)
        context_word: torch.Tensor,  # (batch_size,)
        negative_words: torch.Tensor = None  # (batch_size, n_negatives)
    ) -> torch.Tensor:
        """
        Compute loss for center-context pairs with negative sampling.

        Args:
            center_word: Center word IDs
            context_word: Positive context word IDs
            negative_words: Negative sample word IDs

        Returns:
            Loss value
        """
        # Get embeddings
        center_emb = self.center_embeddings(center_word)  # (batch, emb_dim)
        context_emb = self.context_embeddings(context_word)  # (batch, emb_dim)

        # Positive score: dot product between center and context
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # (batch,)
        pos_loss = F.logsigmoid(pos_score)

        # Negative sampling loss
        if negative_words is not None:
            neg_emb = self.context_embeddings(negative_words)  # (batch, n_neg, emb_dim)
            # Compute scores with all negatives
            neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)  # (batch, n_neg)
            neg_loss = F.logsigmoid(-neg_score).sum(dim=1)  # (batch,)
        else:
            neg_loss = 0

        # Maximize positive score, minimize negative scores
        return -(pos_loss + neg_loss).mean()

# Example usage
vocab_size = 10000
embedding_dim = 300

model = SkipGramModel(vocab_size, embedding_dim)

# Batch of training examples
center_words = torch.tensor([10, 20, 30])  # "cat", "dog", "tree"
context_words = torch.tensor([11, 21, 31])  # "meow", "bark", "leaf"
negative_words = torch.randint(0, vocab_size, (3, 5))  # 5 negative samples each

loss = model(center_words, context_words, negative_words)
print(f"Loss: {loss.item():.4f}")

# After training, we can use the embeddings
word_id = 10
embedding = model.center_embeddings.weight[word_id]
print(f"Embedding for word {word_id}: {embedding[:5]}")  # First 5 dimensions
```

**Negative Sampling**: Instead of computing the full softmax over all vocabulary (expensive), sample a few negative examples:

$$
\log \sigma(\mathbf{v}_{w_O}^\top \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[\log \sigma(-\mathbf{v}_{w_i}^\top \mathbf{v}_{w_I})\right]
$$

### GloVe

GloVe (Global Vectors, Pennington et al., 2014) takes a different approach: directly model word co-occurrence statistics.

**Key Idea**: The ratio of co-occurrence probabilities encodes semantic relationships.

**Example**:
- $P(\text{solid} | \text{ice})$ is high, $P(\text{solid} | \text{steam})$ is low
- $P(\text{gas} | \text{ice})$ is low, $P(\text{gas} | \text{steam})$ is high
- The ratio $\frac{P(\text{solid} | \text{ice})}{P(\text{solid} | \text{steam})}$ is large
- The ratio $\frac{P(\text{gas} | \text{ice})}{P(\text{gas} | \text{steam})}$ is small

**Objective**: Learn embeddings such that their dot product approximates log co-occurrence:

$$
\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j = \log X_{ij}
$$

where $X_{ij}$ is the number of times word $j$ appears in the context of word $i$.

**Full Objective with Weighting**:

$$
J = \sum_{i,j=1}^{V} f(X_{ij}) \left(\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2
$$

where $f(x)$ is a weighting function that prevents rare and very frequent co-occurrences from dominating:

$$
f(x) = \begin{cases}
(x/x_{\max})^{0.75} & \text{if } x < x_{\max} \\
1 & \text{otherwise}
\end{cases}
$$

```python
class GloVeModel(nn.Module):
    """Simplified GloVe model.

    Minimizes the difference between dot product and log co-occurrence.
    In practice, co-occurrence matrix is computed separately from corpus.
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        # Two sets of embeddings (word and context)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Bias terms
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)

        # Initialize
        for param in self.parameters():
            param.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)

    def forward(
        self,
        word_ids: torch.Tensor,  # (batch_size,)
        context_ids: torch.Tensor,  # (batch_size,)
        cooccurrence: torch.Tensor,  # (batch_size,) log co-occurrence counts
        weights: torch.Tensor  # (batch_size,) f(X_ij)
    ) -> torch.Tensor:
        """
        Compute weighted least squares loss.

        Args:
            word_ids: Word indices
            context_ids: Context word indices
            cooccurrence: Log of co-occurrence counts
            weights: Weight function f(X_ij)

        Returns:
            Weighted loss
        """
        # Get embeddings and biases
        word_emb = self.word_embeddings(word_ids)  # (batch, emb_dim)
        context_emb = self.context_embeddings(context_ids)  # (batch, emb_dim)
        word_bias = self.word_biases(word_ids).squeeze()  # (batch,)
        context_bias = self.context_biases(context_ids).squeeze()  # (batch,)

        # Dot product + biases
        prediction = torch.sum(word_emb * context_emb, dim=1) + word_bias + context_bias

        # Weighted squared error
        loss = weights * (prediction - cooccurrence) ** 2

        return loss.mean()

# Example: Simulated co-occurrence data
vocab_size = 10000
embedding_dim = 300

model = GloVeModel(vocab_size, embedding_dim)

# Simulated batch
word_ids = torch.tensor([10, 20, 30])
context_ids = torch.tensor([11, 21, 31])
log_cooccurrence = torch.tensor([5.2, 4.8, 6.1])  # log(X_ij)
weights = torch.tensor([0.8, 0.9, 0.7])  # f(X_ij)

loss = model(word_ids, context_ids, log_cooccurrence, weights)
print(f"GloVe loss: {loss.item():.4f}")
```

### Key Insights from Classical Embeddings

1. **Distributional Hypothesis**: "You shall know a word by the company it keeps" (Firth, 1957)
   - Words in similar contexts have similar meanings
   - Captured by both Word2Vec and GloVe

2. **Semantic Arithmetic**: Famous examples:
   - `king - man + woman ≈ queen`
   - `Paris - France + Italy ≈ Rome`

```python
def demonstrate_word_arithmetic():
    """Demonstrate semantic arithmetic with embeddings."""
    # Simulate pre-trained embeddings (in practice, load from file)
    vocab = {
        'king': 0, 'queen': 1, 'man': 2, 'woman': 3,
        'paris': 4, 'france': 5, 'rome': 6, 'italy': 7
    }

    # Random embeddings for demonstration
    # In practice, these would be trained
    embeddings = torch.randn(len(vocab), 300)

    # Normalize embeddings (common practice)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    def find_closest(target_embedding, exclude_ids=[]):
        """Find closest word to target embedding."""
        similarities = torch.matmul(embeddings, target_embedding)
        # Exclude the input words
        for idx in exclude_ids:
            similarities[idx] = -float('inf')
        closest_idx = similarities.argmax().item()
        closest_word = [w for w, i in vocab.items() if i == closest_idx][0]
        return closest_word, similarities[closest_idx].item()

    # king - man + woman = ?
    result_emb = (embeddings[vocab['king']]
                  - embeddings[vocab['man']]
                  + embeddings[vocab['woman']])
    result_emb = F.normalize(result_emb, p=2, dim=0)

    closest, similarity = find_closest(
        result_emb,
        exclude_ids=[vocab['king'], vocab['man'], vocab['woman']]
    )
    print(f"king - man + woman = {closest} (similarity: {similarity:.3f})")

demonstrate_word_arithmetic()
```

3. **Dimensionality**: Typical dimensions: 50-300 for Word2Vec/GloVe
   - Smaller: faster, less expressive
   - Larger: more expressive, risk of overfitting

4. **Static vs. Contextual**: Word2Vec and GloVe produce **static** embeddings
   - Each word has one embedding regardless of context
   - "bank" (river) and "bank" (financial) have the same embedding
   - Modern transformers solve this with **contextual** embeddings

---

## Learned Embeddings in Neural Networks

In modern deep learning, embeddings are learned end-to-end as part of the model.

### The Embedding Layer

An embedding layer is simply a **lookup table**:

$$
\text{Embedding}: \mathbb{N} \rightarrow \mathbb{R}^d
$$

where:
- Input: Token ID $t \in \{0, 1, \ldots, V-1\}$
- Output: Dense vector $\mathbf{e}_t \in \mathbb{R}^d$

**Mathematically**, it's a matrix multiplication with one-hot vectors:

$$
\mathbf{e}_t = \mathbf{E} \cdot \text{one\_hot}(t)
$$

where $\mathbf{E} \in \mathbb{R}^{V \times d}$ is the embedding matrix.

**In practice**, we just index into the matrix (much more efficient):

$$
\mathbf{e}_t = \mathbf{E}[t, :]
$$

### Training Embeddings

Embeddings are learned via backpropagation along with the rest of the network.

**Forward pass**: Look up embeddings
**Backward pass**: Gradient updates only affect the embeddings that were actually used

```python
# Gradient flow illustration
import torch
import torch.nn as nn

vocab_size = 100
embedding_dim = 16
batch_size = 4
seq_len = 10

# Create embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim)

# Input token IDs
token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

# Forward pass
embedded = embedding(token_ids)  # (batch, seq_len, embedding_dim)

# Dummy loss (just for illustration)
loss = embedded.sum()

# Backward pass
loss.backward()

# Check which embeddings got gradients
print(f"Gradient shape: {embedding.weight.grad.shape}")  # (vocab_size, embedding_dim)
print(f"Non-zero gradient rows: {(embedding.weight.grad.abs().sum(dim=1) > 0).sum()}")
# Only embeddings for tokens in token_ids have non-zero gradients
unique_tokens = token_ids.unique()
print(f"Unique tokens in batch: {len(unique_tokens)}")
```

**Key Properties**:

1. **Sparse updates**: Only embeddings of tokens in the batch get updated
2. **Shared parameters**: Same token ID always maps to same embedding
3. **Position-independent**: Token gets same embedding regardless of position (position added separately, see [Positional Encodings](07-positional-encodings.md))

---

## Embedding Layers in PyTorch

PyTorch provides `nn.Embedding` for efficient embedding lookup.

### Basic Usage

```python
import torch
import torch.nn as nn

# Create embedding layer
vocab_size = 50000
embedding_dim = 768

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Check parameters
print(f"Number of parameters: {vocab_size * embedding_dim:,}")
print(f"Weight shape: {embedding.weight.shape}")  # (50000, 768)

# Forward pass with single token
token_id = torch.tensor([42])
embedded = embedding(token_id)
print(f"Output shape: {embedded.shape}")  # (1, 768)

# Forward pass with sequence
sequence = torch.tensor([10, 20, 30, 40, 50])
embedded_seq = embedding(sequence)
print(f"Sequence output shape: {embedded_seq.shape}")  # (5, 768)

# Forward pass with batch of sequences
batch = torch.tensor([
    [10, 20, 30, 40],
    [15, 25, 35, 45],
    [11, 21, 31, 41]
])
embedded_batch = embedding(batch)
print(f"Batch output shape: {embedded_batch.shape}")  # (3, 4, 768)
```

### Initialization Strategies

Different initialization strategies affect training dynamics:

```python
def compare_initializations():
    """Compare different embedding initialization strategies."""

    vocab_size = 1000
    embedding_dim = 128

    # 1. Default initialization (uniform)
    emb_default = nn.Embedding(vocab_size, embedding_dim)
    print(f"Default - Mean: {emb_default.weight.mean():.4f}, "
          f"Std: {emb_default.weight.std():.4f}")

    # 2. Normal initialization
    emb_normal = nn.Embedding(vocab_size, embedding_dim)
    nn.init.normal_(emb_normal.weight, mean=0.0, std=0.02)
    print(f"Normal(0, 0.02) - Mean: {emb_normal.weight.mean():.4f}, "
          f"Std: {emb_normal.weight.std():.4f}")

    # 3. Xavier/Glorot initialization
    emb_xavier = nn.Embedding(vocab_size, embedding_dim)
    nn.init.xavier_uniform_(emb_xavier.weight)
    print(f"Xavier - Mean: {emb_xavier.weight.mean():.4f}, "
          f"Std: {emb_xavier.weight.std():.4f}")

    # 4. Scaled normal (common for transformers)
    emb_scaled = nn.Embedding(vocab_size, embedding_dim)
    nn.init.normal_(emb_scaled.weight, mean=0.0, std=embedding_dim**-0.5)
    print(f"Normal(0, d^-0.5) - Mean: {emb_scaled.weight.mean():.4f}, "
          f"Std: {emb_scaled.weight.std():.4f}")

compare_initializations()
```

**Common Initialization in Transformers**:
- GPT-2, GPT-3: $\mathcal{N}(0, 0.02)$
- BERT: $\mathcal{N}(0, 0.02)$
- Many others: $\mathcal{N}(0, d^{-0.5})$ where $d$ is embedding dimension

### Padding and Masking

When working with variable-length sequences, we need padding:

```python
class EmbeddingWithPadding(nn.Module):
    """Embedding layer with padding support."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0
    ):
        super().__init__()
        # padding_idx ensures padding token always has zero embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        self.padding_idx = padding_idx

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: (batch, seq_len) token IDs

        Returns:
            embeddings: (batch, seq_len, embedding_dim)
            mask: (batch, seq_len) boolean mask (True = real token, False = padding)
        """
        embeddings = self.embedding(token_ids)

        # Create attention mask (True for real tokens, False for padding)
        mask = (token_ids != self.padding_idx)

        return embeddings, mask

# Example with variable length sequences
vocab_size = 1000
embedding_dim = 128
padding_idx = 0

model = EmbeddingWithPadding(vocab_size, embedding_dim, padding_idx)

# Batch with different lengths (padded to same length)
batch = torch.tensor([
    [5, 10, 15, 20, 25, 0, 0],  # Length 5 + 2 padding
    [7, 12, 17, 22, 0, 0, 0],   # Length 4 + 3 padding
    [3, 8, 13, 18, 23, 28, 33]  # Length 7 (no padding)
])

embeddings, mask = model(batch)

print(f"Embeddings shape: {embeddings.shape}")  # (3, 7, 128)
print(f"Mask shape: {mask.shape}")  # (3, 7)
print(f"Mask:\n{mask}")
print(f"\nPadding embedding (should be zeros):\n{embeddings[0, 5][:5]}")  # First 5 dims
print(f"Non-padding embedding:\n{embeddings[0, 0][:5]}")
```

### Pre-trained Embeddings

Sometimes we want to use pre-trained embeddings (e.g., from Word2Vec or GloVe):

```python
def load_pretrained_embeddings():
    """Example of loading and using pre-trained embeddings."""

    # Simulate pre-trained embeddings (in practice, load from file)
    vocab_size = 10000
    embedding_dim = 300
    pretrained_weights = torch.randn(vocab_size, embedding_dim)

    # Option 1: Load and freeze
    embedding_frozen = nn.Embedding(vocab_size, embedding_dim)
    embedding_frozen.weight = nn.Parameter(pretrained_weights)
    embedding_frozen.weight.requires_grad = False  # Freeze

    print(f"Frozen embeddings trainable: {embedding_frozen.weight.requires_grad}")

    # Option 2: Load and fine-tune
    embedding_finetune = nn.Embedding(vocab_size, embedding_dim)
    embedding_finetune.weight = nn.Parameter(pretrained_weights)
    # Leave requires_grad=True (default)

    print(f"Fine-tunable embeddings trainable: {embedding_finetune.weight.requires_grad}")

    # Option 3: Initialize from pretrained, but with special tokens
    special_tokens = 100  # e.g., [PAD], [UNK], [CLS], [SEP], etc.
    new_vocab_size = vocab_size + special_tokens

    embedding_extended = nn.Embedding(new_vocab_size, embedding_dim)
    # Copy pretrained weights
    embedding_extended.weight.data[:vocab_size] = pretrained_weights
    # Randomly initialize special tokens
    embedding_extended.weight.data[vocab_size:].normal_(mean=0.0, std=0.02)

    print(f"Extended vocab size: {new_vocab_size}")

load_pretrained_embeddings()
```

---

## Embeddings in Modern LLMs

Modern LLMs use learned embeddings with several key characteristics:

### 1. Large Embedding Dimensions

| Model | Vocabulary Size | Embedding Dim | Parameters |
|-------|----------------|---------------|------------|
| GPT-2 (small) | 50,257 | 768 | 38.6M |
| GPT-2 (large) | 50,257 | 1,280 | 64.3M |
| BERT-base | 30,522 | 768 | 23.4M |
| LLaMA 2 (7B) | 32,000 | 4,096 | 131M |
| LLaMA 3 (8B) | 128,256 | 4,096 | 525M |
| GPT-3 | 50,257 | 12,288 | 617M |

**Note**: Embedding parameters can be a significant fraction of total parameters, especially in smaller models!

### 2. Tied Embeddings

Many models **tie** input embeddings and output projection weights to reduce parameters:

```python
class LanguageModel(nn.Module):
    """Language model with tied embeddings.

    The same weight matrix is used for:
    1. Input: token ID → embedding
    2. Output: hidden state → logits over vocabulary

    This reduces parameters and can improve performance.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        tie_weights: bool = True
    ):
        super().__init__()

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Model body (simplified - would be transformer layers)
        self.body = nn.Linear(embedding_dim, hidden_dim)

        # Output projection
        if tie_weights:
            # Ensure dimensions match
            assert embedding_dim == hidden_dim, \
                "Hidden dim must equal embedding dim for weight tying"
            # Share weights with embedding
            self.output_projection = nn.Linear(hidden_dim, vocab_size, bias=False)
            self.output_projection.weight = self.embedding.weight
        else:
            # Separate output projection
            self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Input embedding
        x = self.embedding(token_ids)  # (batch, seq_len, embedding_dim)

        # Model body
        x = self.body(x)  # (batch, seq_len, hidden_dim)

        # Output projection
        logits = self.output_projection(x)  # (batch, seq_len, vocab_size)

        return logits

# Example
vocab_size = 50000
d_model = 768

# With weight tying: vocab_size * d_model parameters (shared)
model_tied = LanguageModel(vocab_size, d_model, d_model, tie_weights=True)
tied_params = sum(p.numel() for p in model_tied.parameters())

# Without weight tying: 2 * vocab_size * d_model parameters (separate)
model_untied = LanguageModel(vocab_size, d_model, d_model, tie_weights=False)
untied_params = sum(p.numel() for p in model_untied.parameters())

print(f"Parameters with tied weights: {tied_params:,}")
print(f"Parameters without tied weights: {untied_params:,}")
print(f"Savings: {untied_params - tied_params:,} ({(1 - tied_params/untied_params)*100:.1f}%)")
```

**Models that tie weights**: GPT-2, BERT, T5, many others
**Models that don't**: Some larger models keep them separate

### 3. Embedding + Position

Embeddings alone don't encode position. Modern transformers add positional information:

```python
class TransformerEmbedding(nn.Module):
    """Combined token + position embedding for transformers.

    This is the first layer of transformer models.
    See [Positional Encodings](07-positional-encodings.md) for details on position.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Position embeddings (learned)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Register position IDs buffer (not a parameter, but part of state)
        self.register_buffer(
            'position_ids',
            torch.arange(max_seq_len).expand((1, -1))
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) token IDs

        Returns:
            embeddings: (batch, seq_len, embedding_dim)
        """
        batch_size, seq_len = token_ids.shape

        # Token embeddings
        token_emb = self.token_embedding(token_ids)  # (batch, seq_len, emb_dim)

        # Position embeddings
        positions = self.position_ids[:, :seq_len]  # (1, seq_len)
        position_emb = self.position_embedding(positions)  # (1, seq_len, emb_dim)

        # Combine (broadcast addition)
        embeddings = token_emb + position_emb  # (batch, seq_len, emb_dim)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings

# Example usage
vocab_size = 50000
max_seq_len = 2048
embedding_dim = 768

model = TransformerEmbedding(vocab_size, max_seq_len, embedding_dim)

# Input sequence
batch = torch.tensor([
    [10, 20, 30, 40, 50],
    [15, 25, 35, 45, 55]
])

output = model(batch)
print(f"Output shape: {output.shape}")  # (2, 5, 768)
```

**Modern variants**:
- **Learned positions** (GPT, BERT): As shown above
- **Sinusoidal positions** (original Transformer): Fixed, deterministic patterns
- **Rotary embeddings (RoPE)** (LLaMA, GPT-Neo): Applied in attention, not added to embeddings
- See [Positional Encodings](07-positional-encodings.md) for full details

### 4. Embedding Scaling

Some models scale embeddings by $\sqrt{d_{\text{model}}}$:

```python
class ScaledEmbedding(nn.Module):
    """Embedding layer with scaling (used in original Transformer).

    Scaling helps balance the magnitude of embeddings and positional encodings.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.scale = embedding_dim ** 0.5

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids) * self.scale

# Example
embedding_dim = 512
scaled_emb = ScaledEmbedding(10000, embedding_dim)

token = torch.tensor([42])
output = scaled_emb(token)

print(f"Embedding dimension: {embedding_dim}")
print(f"Scale factor: {embedding_dim ** 0.5}")
print(f"Output magnitude: {output.norm().item():.2f}")
```

**Who uses scaling**:
- Original Transformer (Vaswani et al., 2017): Yes
- BERT: No
- GPT-2/3: No
- T5: Yes

---

## Advanced Topics

### Subword Embeddings

After subword tokenization (see [Tokenization](01-tokenization.md)), we embed subword tokens:

```python
# Example: BPE tokens
# "unhappiness" might be tokenized as: ["un", "happiness"]
# Each subword gets its own embedding

vocab = {
    "un": 0,
    "happiness": 1,
    "pre": 2,
    "process": 3,
    # ... etc
}

embedding = nn.Embedding(len(vocab), 128)

# "unhappiness"
tokens = torch.tensor([0, 1])  # ["un", "happiness"]
embedded = embedding(tokens)  # (2, 128)

print(f"Token embeddings shape: {embedded.shape}")
```

**Advantage**: Vocabulary stays manageable while handling arbitrary words
**Disadvantage**: Multi-token words have no single embedding (must process sequence)

### Character-Level Embeddings

Some models use character-level embeddings, then aggregate to word level:

```python
class CharCNN(nn.Module):
    """Character-level CNN for word embeddings.

    Used in models like ELMo. Processes character sequence with CNN
    to produce word-level representation.
    """

    def __init__(
        self,
        char_vocab_size: int,
        char_embedding_dim: int,
        output_dim: int,
        kernel_sizes: list[int] = [3, 4, 5]
    ):
        super().__init__()

        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)

        # Multiple filter sizes (like in Kim CNN)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embedding_dim, output_dim // len(kernel_sizes), k)
            for k in kernel_sizes
        ])

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: (batch, max_word_len) character IDs for one word

        Returns:
            word_embedding: (batch, output_dim)
        """
        # Embed characters
        x = self.char_embedding(char_ids)  # (batch, max_word_len, char_emb_dim)
        x = x.permute(0, 2, 1)  # (batch, char_emb_dim, max_word_len)

        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch, filters, seq_len - k + 1)
            pooled = torch.max(conv_out, dim=2)[0]  # (batch, filters)
            conv_outputs.append(pooled)

        # Concatenate all filters
        word_embedding = torch.cat(conv_outputs, dim=1)  # (batch, output_dim)

        return word_embedding

# Example
char_vocab = 256  # ASCII characters
word = "hello"
char_ids = torch.tensor([[ord(c) for c in word]])  # (1, 5)

model = CharCNN(char_vocab, char_embedding_dim=16, output_dim=128)
word_emb = model(char_ids)
print(f"Word embedding shape: {word_emb.shape}")  # (1, 128)
```

### Contextualized Embeddings

Classical embeddings (Word2Vec, GloVe) are **static**: same embedding regardless of context.

Modern transformers produce **contextualized** embeddings: the embedding depends on the entire sequence.

```python
def demonstrate_contextualized_embeddings():
    """Show how transformer embeddings are contextualized."""

    # Simplified transformer (just 1 self-attention layer)
    embedding_dim = 128
    vocab_size = 1000

    # Initial (static) embeddings
    embedding = nn.Embedding(vocab_size, embedding_dim)

    # Self-attention layer (see [Basic Attention](03-basic-attention.md))
    attention = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)

    # Two sentences with word "bank"
    # Sentence 1: "river bank" (token IDs: [100, 42])
    # Sentence 2: "money bank" (token IDs: [200, 42])

    sent1 = torch.tensor([[100, 42]])  # river bank
    sent2 = torch.tensor([[200, 42]])  # money bank

    # Static embeddings (same for "bank" in both contexts)
    emb1_static = embedding(sent1)
    emb2_static = embedding(sent2)

    bank_emb1_static = emb1_static[0, 1]  # bank in "river bank"
    bank_emb2_static = emb2_static[0, 1]  # bank in "money bank"

    print("Static embeddings:")
    print(f"Are 'bank' embeddings identical? {torch.allclose(bank_emb1_static, bank_emb2_static)}")

    # Contextualized embeddings (different after attention)
    emb1_context, _ = attention(emb1_static, emb1_static, emb1_static)
    emb2_context, _ = attention(emb2_static, emb2_static, emb2_static)

    bank_emb1_context = emb1_context[0, 1]  # bank in "river bank"
    bank_emb2_context = emb2_context[0, 1]  # bank in "money bank"

    print("\nContextualized embeddings:")
    print(f"Are 'bank' embeddings identical? {torch.allclose(bank_emb1_context, bank_emb2_context)}")
    print(f"Cosine similarity: {F.cosine_similarity(bank_emb1_context.unsqueeze(0), bank_emb2_context.unsqueeze(0)).item():.3f}")

demonstrate_contextualized_embeddings()
```

**Key insight**: The **input** to a transformer is static embeddings, but the **output** of each layer is contextualized. By the final layer, each token's representation incorporates information from the entire sequence.

---

## Summary

### Key Takeaways

1. **Embeddings map discrete tokens to continuous vectors**
   - Enables neural network processing
   - Learned end-to-end during training

2. **Evolution of embeddings**:
   - **Static** (Word2Vec, GloVe): One embedding per word
   - **Contextualized** (Transformers): Embedding depends on context

3. **Modern LLM embeddings**:
   - Large dimensions (768-12,288)
   - Often tied with output projection
   - Combined with positional encodings
   - Can represent 20-50% of model parameters in small models

4. **PyTorch implementation**:
   - `nn.Embedding` is a simple lookup table
   - Efficient: only used embeddings get gradients
   - Supports padding, pre-trained weights, weight tying

5. **Connection to next chapters**:
   - Embeddings are the **input** to attention mechanisms
   - Position is added via positional encodings ([Positional Encodings](07-positional-encodings.md))
   - Attention creates **contextualized** representations ([Basic Attention](03-basic-attention.md))

### Comparison Table

| Aspect | Word2Vec/GloVe | Transformer Embeddings |
|--------|----------------|------------------------|
| **Learning** | Pre-training objective | End-to-end with model |
| **Context** | Static (one per word) | Contextualized (via attention) |
| **Typical Dim** | 50-300 | 768-12,288 |
| **Position** | None | Added separately |
| **Subwords** | No | Yes (BPE, etc.) |
| **Parameters** | vocab × dim | vocab × dim (input) |

---

## References

### Key Papers

1. Mikolov et al. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Word2Vec)
2. Mikolov et al. (2013). [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (Word2Vec improvements, negative sampling)
3. Pennington et al. (2014). [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/)
4. Bojanowski et al. (2017). [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) (FastText)
5. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Original Transformer)
6. Devlin et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
7. Radford et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)

### Additional Resources

- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - Jay Alammar
- [Word embeddings tutorial](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html) - PyTorch
- [Understanding word embeddings](https://www.tensorflow.org/text/guide/word_embeddings) - TensorFlow

---

## Exercises

### Conceptual Questions

1. **Why embeddings?** Explain why we use dense embeddings instead of one-hot encodings. What are the three main advantages?

2. **Static vs. Contextualized**: What is the difference between static embeddings (Word2Vec) and contextualized embeddings (from transformers)? Give an example where this matters.

3. **Word2Vec variants**: What is the difference between Skip-Gram and CBOW? When would you prefer one over the other?

4. **Weight tying**: Explain weight tying between input embeddings and output projection. What are the benefits? What is the requirement?

5. **Embedding dimension**: Why don't we just use very large embedding dimensions (e.g., 10,000)? What are the trade-offs?

### Coding Exercises

#### Exercise 1: Implement Word2Vec Skip-Gram with Negative Sampling

Complete the training loop for the Skip-Gram model:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SkipGramDataset(Dataset):
    """Dataset for Skip-Gram training."""

    def __init__(self, text, vocab, window_size=2, n_negatives=5):
        """
        Args:
            text: List of token IDs
            vocab: Vocabulary (dict mapping token to ID)
            window_size: Context window size
            n_negatives: Number of negative samples
        """
        self.text = text
        self.vocab_size = len(vocab)
        self.window_size = window_size
        self.n_negatives = n_negatives

        # Create positive pairs (center, context)
        self.pairs = []
        for i in range(len(text)):
            center = text[i]
            for j in range(max(0, i - window_size),
                          min(len(text), i + window_size + 1)):
                if i != j:
                    context = text[j]
                    self.pairs.append((center, context))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        # Sample negative examples (random tokens)
        negatives = torch.randint(0, self.vocab_size, (self.n_negatives,))
        return torch.tensor(center), torch.tensor(context), negatives

# TODO: Implement training loop
def train_skipgram(text, vocab, embedding_dim=128, epochs=5, lr=0.01):
    """
    Train Skip-Gram model.

    Your task:
    1. Create dataset and dataloader
    2. Initialize model and optimizer
    3. Implement training loop
    4. Return trained embeddings
    """
    # Your code here
    pass

# Test with small corpus
corpus = ["the", "cat", "sat", "on", "the", "mat", "the", "dog", "sat", "on", "the", "log"]
vocab = {word: i for i, word in enumerate(set(corpus))}
text = [vocab[word] for word in corpus]

# Train model
# embeddings = train_skipgram(text, vocab)
# print(f"Learned embeddings shape: {embeddings.shape}")
```

#### Exercise 2: Implement Embedding with Dynamic Padding

Implement a collate function that handles variable-length sequences:

```python
def collate_with_padding(batch, padding_idx=0):
    """
    Collate function for DataLoader that pads sequences to same length.

    Args:
        batch: List of sequences (each is a list of token IDs)
        padding_idx: ID to use for padding

    Returns:
        padded_batch: (batch_size, max_len) tensor
        lengths: (batch_size,) tensor of original lengths

    TODO: Implement this function
    """
    # Your code here
    pass

# Test
batch = [
    [1, 2, 3, 4, 5],
    [1, 2, 3],
    [1, 2, 3, 4, 5, 6, 7]
]

# padded, lengths = collate_with_padding(batch)
# print(f"Padded shape: {padded.shape}")
# print(f"Lengths: {lengths}")
# print(f"Padded batch:\n{padded}")
```

#### Exercise 3: Compare Initialization Strategies

Empirically test different initialization strategies:

```python
def test_initialization_impact():
    """
    Test how initialization affects training.

    TODO:
    1. Create a simple language model
    2. Train with different embedding initializations:
       - Random uniform
       - Normal(0, 0.02)
       - Normal(0, d^-0.5)
       - Xavier/Glorot
    3. Compare convergence speed and final loss
    4. Plot results
    """
    # Your code here
    pass
```

#### Exercise 4: Visualize Embeddings

Use t-SNE or PCA to visualize learned embeddings:

```python
def visualize_embeddings(embeddings, vocab, method='tsne'):
    """
    Visualize embeddings in 2D.

    Args:
        embeddings: (vocab_size, embedding_dim) tensor
        vocab: Dictionary mapping word to ID
        method: 'tsne' or 'pca'

    TODO:
    1. Reduce dimensionality to 2D
    2. Plot points
    3. Label with words
    4. Observe clustering (e.g., similar words nearby)
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # Your code here
    pass
```

#### Exercise 5: Implement Tied Embeddings

Create a simple language model with optional weight tying:

```python
class SimpleLM(nn.Module):
    """
    Simple language model to test tied vs untied embeddings.

    TODO:
    1. Implement forward pass
    2. Support both tied and untied weights
    3. Compare:
       - Number of parameters
       - Training speed
       - Final perplexity
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        tie_weights: bool = True
    ):
        super().__init__()
        # Your code here
        pass

    def forward(self, x):
        # Your code here
        pass

# Compare tied vs untied
# model_tied = SimpleLM(10000, 512, 512, 2, tie_weights=True)
# model_untied = SimpleLM(10000, 512, 512, 2, tie_weights=False)
# Compare parameter counts and training
```

### Challenge Exercise

**Build a complete embedding + positional encoding module** that:
1. Supports both learned and sinusoidal positional encodings
2. Handles padding correctly
3. Includes dropout
4. Supports weight tying with output layer
5. Can load pre-trained embeddings

This will be useful for implementing transformers in later chapters!

---

**Next**: [Basic Attention](03-basic-attention.md) - Learn how attention mechanisms use embeddings to create contextualized representations.
