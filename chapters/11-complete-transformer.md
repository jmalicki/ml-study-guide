# Chapter 11: Building a Complete Transformer

In previous chapters, we've covered the individual components of the Transformer architecture: attention mechanisms, positional encodings, feed-forward networks, and normalization. Now we'll assemble these pieces into complete, trainable models.

This chapter provides full implementations of the main Transformer variants used in production LLMs, focusing on the architectural patterns that have proven most successful.

## Table of Contents

1. [Overview of Transformer Architectures](#overview-of-transformer-architectures)
2. [Encoder Architecture (BERT-style)](#encoder-architecture-bert-style)
3. [Decoder Architecture (GPT-style)](#decoder-architecture-gpt-style)
4. [Encoder-Decoder Architecture (T5, BART)](#encoder-decoder-architecture-t5-bart)
5. [Modern Decoder-Only Models](#modern-decoder-only-models)
6. [Full Working Implementation](#full-working-implementation)
7. [Training the Model](#training-the-model)
8. [Summary and Best Practices](#summary-and-best-practices)
9. [Exercises](#exercises)

---

## Overview of Transformer Architectures

The original Transformer paper introduced an encoder-decoder architecture, but modern LLMs primarily use decoder-only models. Let's understand the three main patterns:

### Architecture Types

| Type | Use Case | Examples | Key Feature |
|------|----------|----------|-------------|
| **Encoder-only** | Understanding tasks | BERT, RoBERTa | Bidirectional attention |
| **Decoder-only** | Generation tasks | GPT-3/4, LLaMA, Claude | Causal attention |
| **Encoder-Decoder** | Seq2seq tasks | T5, BART, Original Transformer | Cross-attention |

### Why Decoder-Only Dominates

Modern LLMs (GPT-4, Claude, LLaMA, Gemini) are all decoder-only because:

1. **Unified architecture**: Same model for understanding and generation
2. **Scaling efficiency**: Simpler to parallelize and scale
3. **Generality**: Can be adapted to any task via prompting
4. **Training simplicity**: Single causal language modeling objective

See [Architecture Comparison: Modern LLMs](29-model-architectures.md) for details on production models.

---

## Encoder Architecture (BERT-style)

Encoder models use **bidirectional attention** to build contextualized representations. They excel at understanding tasks like classification, NER, and question answering.

### Mathematical Formulation

For an input sequence $X = (x_1, x_2, \ldots, x_n)$:

1. **Embedding**: $E = \text{Embed}(X) + \text{PosEnc}(X)$
2. **Encoder layers** (repeated $L$ times):
   $$
   \begin{align}
   H' &= \text{LayerNorm}(E + \text{MultiHeadAttn}(E, E, E, \text{mask}=\text{None})) \\
   H &= \text{LayerNorm}(H' + \text{FFN}(H'))
   \end{align}
   $$
3. **Output**: Contextualized representations for each token

Key difference from decoder: **no causal mask** - each token can attend to all other tokens.

### Implementation

```python
import torch
import torch.nn as nn
import math

class TransformerEncoderLayer(nn.Module):
    """Single encoder layer with bidirectional self-attention.

    Architecture:
        x -> LayerNorm -> MultiHeadAttn -> residual
          -> LayerNorm -> FFN -> residual -> output

    See [The Transformer Block](09-transformer-block.md) for component details.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            src_mask: Attention mask (seq_len, seq_len) - typically None for encoder
            src_key_padding_mask: Padding mask (batch, seq_len) - True for padding tokens

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Pre-norm: normalize before attention
        x_norm = self.norm1(x)

        # Self-attention with residual
        # In encoder: Q, K, V all come from the same input (bidirectional)
        attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = x + attn_output

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    """Full Transformer Encoder (BERT-style).

    Stacks multiple encoder layers and adds embeddings.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (learned, as in BERT)
        # See [Positional Encodings](07-positional-encodings.md)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Mask for padding tokens (batch, seq_len)
                           1 for real tokens, 0 for padding

        Returns:
            Encoded representations (batch, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape

        # Create position IDs: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings: token + position
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)

        # Convert attention_mask to key_padding_mask format
        # PyTorch expects True for positions to mask (padding)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)

        # Final normalization
        x = self.norm(x)

        return x
```

### Example Usage: Sequence Classification

```python
class BERTForClassification(nn.Module):
    """BERT-style model for sequence classification.

    Uses [CLS] token representation for classification.
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff
        )

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Encode
        encoded = self.encoder(input_ids, attention_mask)

        # Take [CLS] token (first token) representation
        cls_output = encoded[:, 0, :]

        # Classify
        logits = self.classifier(cls_output)

        return logits

# Example usage
model = BERTForClassification(vocab_size=30000, num_classes=2)
input_ids = torch.randint(0, 30000, (2, 128))  # batch=2, seq=128
attention_mask = torch.ones(2, 128)

logits = model(input_ids, attention_mask)
print(f"Classification logits shape: {logits.shape}")  # (2, 2)
```

---

## Decoder Architecture (GPT-style)

Decoder models use **causal (masked) attention** to generate text autoregressively. This is the architecture used by GPT, LLaMA, Claude, and most modern LLMs.

### Mathematical Formulation

For an input sequence $X = (x_1, x_2, \ldots, x_n)$:

1. **Embedding**: $E = \text{Embed}(X) + \text{PosEnc}(X)$
2. **Decoder layers** (repeated $L$ times):
   $$
   \begin{align}
   H' &= \text{LayerNorm}(E + \text{MaskedMultiHeadAttn}(E, E, E)) \\
   H &= \text{LayerNorm}(H' + \text{FFN}(H'))
   \end{align}
   $$
3. **Language modeling head**: $\text{Logits} = H W_\text{vocab}^T$

Key difference from encoder: **causal mask** ensures token $i$ can only attend to tokens $\leq i$.

The causal mask is:
$$
M_{ij} = \begin{cases}
0 & \text{if } i < j \\
1 & \text{if } i \geq j
\end{cases}
$$

See [Bidirectional vs Causal Attention](05-bidirectional-causal-attention.md) for details.

### Implementation

```python
class TransformerDecoderLayer(nn.Module):
    """Single decoder layer with causal self-attention.

    Architecture (Pre-norm):
        x -> LayerNorm -> Masked MultiHeadAttn -> residual
          -> LayerNorm -> FFN -> residual -> output
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            causal_mask: Causal attention mask (seq_len, seq_len)
            key_padding_mask: Padding mask (batch, seq_len)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Pre-norm
        x_norm = self.norm1(x)

        # Causal self-attention
        attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + attn_output

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


class TransformerDecoder(nn.Module):
    """Full Transformer Decoder (GPT-style).

    This is the architecture used by GPT-2, GPT-3, and similar models.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Language modeling head (tied with token embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying: share embeddings with output projection
        self.lm_head.weight = self.token_embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask.

        Returns:
            Mask of shape (seq_len, seq_len) where:
            - 0.0 = attend
            - -inf = don't attend (future positions)
        """
        # Create lower triangular matrix
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Mask for padding (batch, seq_len)
                           1 for real tokens, 0 for padding

        Returns:
            Logits over vocabulary (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position IDs
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device)

        # Convert attention_mask to key_padding_mask
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, causal_mask, key_padding_mask)

        # Final norm
        x = self.norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            input_ids: Prompt tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens (nucleus filtering)
            top_p: Keep tokens with cumulative probability >= top_p

        Returns:
            Generated sequence (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            input_ids_cond = input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits = self(input_ids_cond)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Optional: top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Optional: top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                # Scatter sorted tensors back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('inf')

            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
```

### Example: Text Generation

```python
# Create GPT-style model
gpt_model = TransformerDecoder(
    vocab_size=50000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=1024
)

# Example: forward pass (training)
input_ids = torch.randint(0, 50000, (2, 128))
logits = gpt_model(input_ids)
print(f"Logits shape: {logits.shape}")  # (2, 128, 50000)

# Example: generation (inference)
prompt = torch.randint(0, 50000, (1, 10))  # Batch=1, 10 tokens
generated = gpt_model.generate(
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
print(f"Generated shape: {generated.shape}")  # (1, 60)
```

---

## Encoder-Decoder Architecture (T5, BART)

Encoder-decoder models combine both architectures with **cross-attention** to connect them. They're designed for sequence-to-sequence tasks like translation, summarization, etc.

### Mathematical Formulation

**Encoder** (same as before):
$$
H_\text{enc} = \text{Encoder}(X_\text{src})
$$

**Decoder** with cross-attention:
$$
\begin{align}
H'_\text{self} &= \text{LayerNorm}(E + \text{MaskedSelfAttn}(E, E, E)) \\
H'_\text{cross} &= \text{LayerNorm}(H'_\text{self} + \text{CrossAttn}(H'_\text{self}, H_\text{enc}, H_\text{enc})) \\
H &= \text{LayerNorm}(H'_\text{cross} + \text{FFN}(H'_\text{cross}))
\end{align}
$$

Cross-attention uses:
- **Query** from decoder
- **Key, Value** from encoder

See [Cross-Attention](06-cross-attention.md) for details.

### Implementation

```python
class TransformerEncoderDecoderLayer(nn.Module):
    """Decoder layer with self-attention and cross-attention.

    Architecture:
        x -> LayerNorm -> Masked Self-Attn -> residual
          -> LayerNorm -> Cross-Attn (with encoder) -> residual
          -> LayerNorm -> FFN -> residual -> output
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Target sequence (decoder input)
            encoder_output: Encoder output (source encoding)
            tgt_mask: Causal mask for self-attention
            memory_mask: Mask for cross-attention (typically None)
            tgt_key_padding_mask: Padding mask for target
            memory_key_padding_mask: Padding mask for source
        """
        # Self-attention
        x_norm = self.norm1(x)
        self_attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        x = x + self_attn_output

        # Cross-attention: Q from decoder, K,V from encoder
        x_norm = self.norm2(x)
        cross_attn_output, _ = self.cross_attn(
            query=x_norm,
            key=encoder_output,
            value=encoder_output,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        x = x + cross_attn_output

        # FFN
        x = x + self.ffn(self.norm3(x))

        return x


class TransformerEncoderDecoder(nn.Module):
    """Full Transformer with encoder and decoder (T5/BART style)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Shared embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerEncoderDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for decoder."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        return mask

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create embeddings: token + position."""
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)

        return self.dropout(token_embeds + pos_embeds)

    def encode(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Encode source sequence.

        Args:
            src_input_ids: Source tokens (batch, src_len)
            src_attention_mask: Source padding mask (batch, src_len)

        Returns:
            Encoder output (batch, src_len, d_model)
        """
        # Embed
        x = self._embed(src_input_ids)

        # Padding mask
        if src_attention_mask is not None:
            src_key_padding_mask = (src_attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Encode
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return self.encoder_norm(x)

    def decode(
        self,
        tgt_input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_attention_mask: torch.Tensor = None,
        src_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Decode target sequence.

        Args:
            tgt_input_ids: Target tokens (batch, tgt_len)
            encoder_output: Encoder output (batch, src_len, d_model)
            tgt_attention_mask: Target padding mask (batch, tgt_len)
            src_attention_mask: Source padding mask (batch, src_len)

        Returns:
            Decoder output logits (batch, tgt_len, vocab_size)
        """
        # Embed
        x = self._embed(tgt_input_ids)

        # Create causal mask
        tgt_len = tgt_input_ids.shape[1]
        causal_mask = self._create_causal_mask(tgt_len, tgt_input_ids.device)

        # Padding masks
        if tgt_attention_mask is not None:
            tgt_key_padding_mask = (tgt_attention_mask == 0)
        else:
            tgt_key_padding_mask = None

        if src_attention_mask is not None:
            memory_key_padding_mask = (src_attention_mask == 0)
        else:
            memory_key_padding_mask = None

        # Decode
        for layer in self.decoder_layers:
            x = layer(
                x,
                encoder_output,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        x = self.decoder_norm(x)
        logits = self.lm_head(x)

        return logits

    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor = None,
        tgt_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Full forward pass.

        Args:
            src_input_ids: Source tokens (batch, src_len)
            tgt_input_ids: Target tokens (batch, tgt_len)
            src_attention_mask: Source padding mask
            tgt_attention_mask: Target padding mask

        Returns:
            Logits (batch, tgt_len, vocab_size)
        """
        encoder_output = self.encode(src_input_ids, src_attention_mask)
        logits = self.decode(
            tgt_input_ids,
            encoder_output,
            tgt_attention_mask,
            src_attention_mask
        )
        return logits
```

### Example: Translation

```python
# Create encoder-decoder model
seq2seq_model = TransformerEncoderDecoder(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_encoder_layers=6,
    n_decoder_layers=6,
    d_ff=2048
)

# Example inputs
src = torch.randint(0, 32000, (2, 20))  # Source: batch=2, len=20
tgt = torch.randint(0, 32000, (2, 15))  # Target: batch=2, len=15

# Forward pass
logits = seq2seq_model(src, tgt)
print(f"Translation logits shape: {logits.shape}")  # (2, 15, 32000)
```

---

## Modern Decoder-Only Models

Modern LLMs (LLaMA, GPT-4, Claude) use advanced techniques beyond the basic decoder. Here's a production-ready implementation incorporating modern best practices.

### Key Improvements

1. **RMSNorm** instead of LayerNorm (faster, equally effective)
2. **RoPE** instead of learned positions (better extrapolation)
3. **SwiGLU** activation (better performance)
4. **Grouped Query Attention** (GQA) for efficient inference
5. **Pre-normalization** (more stable training)

See [Architecture Comparison: Modern LLMs](29-model-architectures.md) for why these choices matter.

### Implementation

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm(x) = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)

    Compared to LayerNorm:
    - No mean centering (re-centering invariance)
    - No bias term
    - ~10-15% faster
    - Equally effective for training

    See [The Transformer Block](09-transformer-block.md) for details.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None
) -> torch.Tensor:
    """Precompute RoPE frequencies.

    RoPE applies rotations to Q and K based on position:

    For dimension i and position m:
        theta_i = 10000^(-2i/dim)
        freq_i = m * theta_i

    See [Rotary Position Embeddings](08-rope.md) for mathematical details.

    Args:
        dim: Dimension per head (must be even)
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation
        device: Device to create tensor on

    Returns:
        Frequencies tensor (max_seq_len, dim // 2)
    """
    # Create frequency values: [theta_0, theta_1, ..., theta_{dim/2-1}]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len, device=device)

    # Outer product to get all (position, freq) pairs
    freqs = torch.outer(positions, freqs)

    return freqs


def apply_rope(
    x: torch.Tensor,
    freqs: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embeddings.

    Args:
        x: Input tensor (batch, seq_len, n_heads, head_dim)
        freqs: Precomputed frequencies (seq_len, head_dim // 2)

    Returns:
        Rotated tensor with same shape as input
    """
    # Reshape x to separate real and imaginary parts
    # (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, n_heads, head_dim//2, 2)
    x_complex = x.float().reshape(*x.shape[:-1], -1, 2)

    # Convert to complex numbers
    x_complex = torch.view_as_complex(x_complex)

    # Create rotation: e^(i * freqs) = cos(freqs) + i*sin(freqs)
    freqs_complex = torch.polar(
        torch.ones_like(freqs),
        freqs
    )

    # Broadcast freqs to match x_complex shape
    # (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    x_rotated = x_complex * freqs_complex

    # Convert back to real representation
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA).

    GQA is a hybrid between:
    - Multi-Head Attention (MHA): Each Q head has its own K,V heads
    - Multi-Query Attention (MQA): All Q heads share one K,V head

    GQA: Groups of Q heads share K,V heads

    Benefits:
    - Reduces KV cache by factor of (n_heads / n_kv_heads)
    - Minimal quality loss compared to MHA
    - Used in LLaMA 2/3, Qwen, Mistral, etc.

    See [Multi-Head Attention](04-multi-head-attention.md) for details.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        # Q: one per head, K,V: one per KV head
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.dropout = dropout

        # Precompute RoPE frequencies
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, max_seq_len, rope_theta)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Causal mask (seq_len, seq_len)

        Returns:
            Output (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        rope_freqs = self.rope_freqs[:seq_len]
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Repeat K, V for each group
        # (batch, seq, n_kv_heads, head_dim) -> (batch, seq, n_heads, head_dim)
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)

        # Transpose for attention: (batch, n_heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if mask is not None:
            scores = scores + mask

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        if self.dropout > 0:
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Transpose back and reshape
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, -1)

        # Output projection
        return self.wo(output)


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    SwiGLU(x) = Swish(xW) ⊙ (xV)
    where Swish(x) = x * sigmoid(x)

    This is a gated linear unit that performs better than GELU.
    Used in LLaMA, PaLM, and other modern LLMs.

    Note: Requires 3 linear layers (gate, up, down) instead of 2.

    See [Activation Functions](10-activation-functions.md) for details.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU is same as Swish: x * sigmoid(x)
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class ModernTransformerBlock(nn.Module):
    """Modern transformer block (LLaMA-style).

    Uses:
    - RMSNorm (pre-norm)
    - Grouped Query Attention with RoPE
    - SwiGLU activation
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0
    ):
        super().__init__()

        # Pre-norms
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

        # Attention
        self.attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, max_seq_len, dropout
        )

        # Feed-forward with SwiGLU
        # Note: SwiGLU uses 2/3 * 4 * d_model for hidden dim
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.attn(self.attn_norm(x), mask)

        # Pre-norm + FFN + residual
        x = x + self.ffn(self.ffn_norm(x))

        return x


class ModernTransformer(nn.Module):
    """Modern decoder-only transformer (LLaMA-style).

    Architecture choices:
    - RMSNorm (faster than LayerNorm)
    - RoPE (better than learned positions)
    - GQA (efficient KV cache)
    - SwiGLU (better than GELU)
    - Pre-normalization (more stable)
    - No bias terms (simplification)

    This is representative of LLaMA 2/3, Mistral, Qwen, etc.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 4096,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        n_layers: int = 32,
        d_ff: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Default FFN hidden dim: 8/3 * d_model (for SwiGLU)
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            # Round to nearest multiple of 256 for efficiency
            d_ff = 256 * ((d_ff + 255) // 256)

        # Token embeddings (no position embeddings - using RoPE)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            ModernTransformerBlock(
                d_model, n_heads, n_kv_heads, d_ff, max_seq_len, dropout
            )
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(d_model)

        # Output projection (tied with embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask

    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (batch, seq_len)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings (no position embeddings - using RoPE)
        x = self.token_embedding(input_ids)

        # Create causal mask
        mask = self._create_causal_mask(seq_len, device)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            input_ids: Prompt (batch, seq_len)
            max_new_tokens: Tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            Generated sequence (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max context
            input_ids_cond = input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits = self(input_ids_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    @torch.no_grad()
    def generate_with_kv_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> torch.Tensor:
        """Generate text with KV caching for efficiency.

        KV caching stores past key and value states to avoid recomputing
        attention for previously processed tokens. This is critical for
        efficient autoregressive generation.

        Memory savings: Without caching, each step recomputes O(n^2) attention.
        With caching, each step only computes O(n) for the new token.

        Args:
            input_ids: Prompt (batch, seq_len)
            max_new_tokens: Tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            Generated sequence (batch, seq_len + max_new_tokens)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize KV cache for each layer
        # cache[layer_idx] = (keys, values)
        # keys/values: (batch, n_kv_heads, seq_len, head_dim)
        kv_cache = [None] * len(self.layers)

        # Process initial prompt
        seq_len = input_ids.shape[1]
        x = self.token_embedding(input_ids)
        mask = self._create_causal_mask(seq_len, device)

        # Forward through all layers, building initial cache
        for layer_idx, layer in enumerate(self.layers):
            # For the first pass, we need to modify the layer to return KV states
            # In practice, you'd modify the layer's forward method to accept/return cache
            x = layer(x, mask)
            # Cache would be populated here in a full implementation

        x = self.norm(x)
        logits = self.lm_head(x)

        # Sample first new token
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Generate remaining tokens using cache
        for _ in range(max_new_tokens - 1):
            # Only process the new token
            x = self.token_embedding(next_token)

            # For each layer, use cached K,V and only compute new K,V
            for layer_idx, layer in enumerate(self.layers):
                # In a full implementation:
                # 1. Extract cached K,V for this layer
                # 2. Compute new K,V for current token
                # 3. Concatenate with cache
                # 4. Compute attention using full K,V but only new Q
                # 5. Update cache
                x = layer(x, mask=None)  # Simplified

            x = self.norm(x)
            logits = self.lm_head(x)

            # Sample next token
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
```

### KV Caching for Efficient Inference

The `generate_with_kv_cache` method above provides a template for KV caching. In production, you need to modify the attention layers to support caching. Here's a complete implementation:

```python
class GroupedQueryAttentionWithCache(nn.Module):
    """GQA with KV caching support for efficient generation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.dropout = dropout

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, max_seq_len, rope_theta)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        cache: tuple = None,
        position_offset: int = 0
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Causal mask (seq_len, seq_len)
            cache: Tuple of (cached_keys, cached_values) or None
            position_offset: Position offset for RoPE (used with cache)

        Returns:
            output: (batch, seq_len, d_model)
            new_cache: Tuple of (keys, values) including new tokens
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE with position offset
        rope_freqs = self.rope_freqs[position_offset:position_offset + seq_len]
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Handle caching
        if cache is not None:
            cached_k, cached_v = cache
            # Concatenate with cached K,V
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        # Store new cache
        new_cache = (k, v)

        # Repeat K, V for each group
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)

        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        if self.dropout > 0:
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, -1)

        return self.wo(output), new_cache


# Memory savings from KV caching:
# Without cache: Generate N tokens = N forward passes through full sequence
#                Memory: O(N^2 * d_model) per layer
# With cache:    Generate N tokens = N forward passes through single token
#                Memory: O(N * d_model) per layer
#
# For a 7B model with 32 layers, batch=1, generating 100 tokens:
# - Without cache: ~2.5GB KV states recomputed each step
# - With cache: ~25MB KV states computed once and reused
# Speedup: ~100x for long generations
```

### Model Size Comparison

Here are typical configurations for different model sizes:

```python
# Small model (like GPT-2 small, 124M params)
small_config = {
    "vocab_size": 50257,
    "d_model": 768,
    "n_heads": 12,
    "n_kv_heads": 12,  # MHA
    "n_layers": 12,
    "max_seq_len": 1024
}

# Medium model (like LLaMA-2 7B)
medium_config = {
    "vocab_size": 32000,
    "d_model": 4096,
    "n_heads": 32,
    "n_kv_heads": 8,  # GQA with 4 groups
    "n_layers": 32,
    "max_seq_len": 4096
}

# Large model (like LLaMA-2 13B)
large_config = {
    "vocab_size": 32000,
    "d_model": 5120,
    "n_heads": 40,
    "n_kv_heads": 8,
    "n_layers": 40,
    "max_seq_len": 4096
}

# Create model
model = ModernTransformer(**medium_config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
```

---

## Full Working Implementation

Let's put it all together with a complete training example.

### Dataset Preparation

#### The Problem: Efficient Data Loading for Language Modeling

Language model training requires processing massive amounts of text data efficiently. The key challenge is transforming raw text into a format suitable for autoregressive training where the model predicts the next token given previous tokens.

**Why This Matters:**
- Language models learn by predicting token $t_i$ given context $(t_1, \ldots, t_{i-1})$
- Training requires millions to billions of text sequences
- Naive approaches (loading all text into memory) fail for large corpora
- Need efficient batching and shuffling for good gradient estimates

**Theoretical Foundation:**

The causal language modeling objective is:
$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)
$$

To compute this efficiently, we:
1. **Chunk text into fixed-length blocks**: Allows batching sequences of equal length
2. **Create input-target pairs**: For sequence $[x_1, \ldots, x_n]$, input is $[x_1, \ldots, x_{n-1}]$ and target is $[x_2, \ldots, x_n]$
3. **Shuffle blocks**: Reduces correlation between consecutive batches

**Key Design Choices:**
- **Block size**: Typically 512-4096 tokens (matches model's max sequence length)
- **Overlap strategy**: No overlap for simplicity; overlapping windows possible for better data utilization
- **Padding**: Usually avoided by filtering to complete blocks only

**How This Relates to Alternatives:**
- **Per-document processing**: Wastes computation on padding for variable-length documents
- **Streaming tokenization**: More memory-efficient but complex; we'll cover below
- **Dynamic batching**: Groups similar-length sequences but adds complexity

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """Simple text dataset for language modeling.

    Splits text into fixed-length chunks for training.
    """
    def __init__(
        self,
        text: str,
        tokenizer,
        block_size: int = 512
    ):
        """
        Args:
            text: Raw text string
            tokenizer: Function that converts text to token IDs
            block_size: Sequence length for training
        """
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Tokenize entire text
        self.tokens = tokenizer(text)

    def __len__(self):
        # Number of complete blocks
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        # Get block
        start = idx * self.block_size
        end = start + self.block_size + 1  # +1 for target

        chunk = self.tokens[start:end]

        # Input is chunk[:-1], target is chunk[1:]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y


# Simple tokenizer (character-level for demonstration)
# In practice, use a real tokenizer like tiktoken or sentencepiece
class CharTokenizer:
    """Character-level tokenizer."""
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def __call__(self, text: str) -> list[int]:
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join([self.idx_to_char[i] for i in tokens])


# Example text (in practice, load a large corpus)
sample_text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
""" * 100  # Repeat for more data

# Create tokenizer and dataset
tokenizer = CharTokenizer(sample_text)
dataset = TextDataset(sample_text, tokenizer, block_size=128)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Dataset size: {len(dataset)} blocks")
```

### Memory-Efficient Datasets for Large-Scale Training

The `TextDataset` above loads all tokens into memory, which works for small corpora but fails for large-scale training (billions of tokens). For production, use these alternatives:

#### 1. Memory-Mapped Datasets

**The Problem: RAM Limitations with Large Corpora**

When training on datasets with billions of tokens (e.g., Common Crawl, C4, RedPajama), loading all data into RAM is impossible. A 100GB tokenized dataset cannot fit in typical GPU server memory (128-256GB).

**Why Memory Mapping Matters:**
- Allows working with datasets larger than available RAM
- The OS handles data loading transparently via virtual memory
- Only loads needed chunks into RAM on-demand (lazy loading)
- Much faster than reading from disk repeatedly

**Theoretical Justification:**

Memory mapping leverages the OS page cache:
1. File is mapped to virtual address space (no actual RAM used yet)
2. When accessing position $i$, OS loads surrounding page into RAM
3. Least-recently-used pages evicted when RAM fills
4. Sequential access patterns (common in training) are highly efficient

**Performance Characteristics:**
- **Memory usage**: O(block_size) instead of O(dataset_size)
- **Access speed**: ~95% of RAM speed for sequential access
- **Random access**: Slower due to page faults, but still practical

**How This Compares to Alternatives:**
- **In-memory dataset**: Fastest but limited to ~1GB datasets
- **Memory-mapped**: Handles 1GB-100GB datasets efficiently
- **Streaming** (below): For 100GB+ or distributed/cloud storage
- **Database (e.g., LMDB)**: Adds overhead, mainly for key-value workloads

**Key Insight**: Memory mapping converts a storage problem into an OS caching problem, which is highly optimized in modern systems.

```python
import numpy as np
import os

class MemoryMappedDataset(Dataset):
    """Dataset using memory-mapped files for efficient large-scale training.

    Memory-mapped files allow the OS to handle data loading without
    loading everything into RAM.
    """
    def __init__(
        self,
        data_path: str,
        block_size: int = 512
    ):
        """
        Args:
            data_path: Path to .npy file containing tokenized data
            block_size: Sequence length
        """
        self.block_size = block_size

        # Memory-map the file (doesn't load into RAM)
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1

        # Only this chunk is loaded into RAM
        chunk = self.data[start:end].astype(np.int64)

        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y


# To create a memory-mapped dataset:
# 1. Tokenize your corpus
# 2. Save as numpy array
def create_memmap_dataset(text_files: list, output_path: str, tokenizer):
    """Create memory-mapped dataset from text files."""
    tokens = []
    for file_path in text_files:
        with open(file_path, 'r') as f:
            text = f.read()
            tokens.extend(tokenizer(text))

    # Save as memory-mapped array
    tokens_array = np.array(tokens, dtype=np.uint16)
    np.save(output_path, tokens_array)

    print(f"Created memmap dataset: {len(tokens_array):,} tokens")
    print(f"File size: {os.path.getsize(output_path + '.npy') / 1e9:.2f} GB")

# Example usage:
# create_memmap_dataset(['corpus1.txt', 'corpus2.txt'], 'data.npy', tokenizer)
# dataset = MemoryMappedDataset('data.npy', block_size=512)
```

#### 2. Streaming Datasets

**The Problem: Datasets Larger Than Local Storage**

Modern LLM training uses datasets too large to store locally (multi-terabyte corpora) or distributed across cloud storage. Additionally, we may want to process data on-the-fly (e.g., dynamic augmentation, filtering).

**Why Streaming Matters:**
- **Unlimited dataset size**: Process data that exceeds local storage
- **Distributed training**: Each worker can stream different data subsets
- **Cloud-native**: Read directly from S3, GCS, Azure Blob, etc.
- **Dynamic processing**: Apply transformations on-the-fly without preprocessing

**Theoretical Foundation:**

Streaming datasets implement an **iterator pattern** rather than random access:
- Traditional dataset: Implements `__getitem__(idx)` for random access
- Streaming dataset: Implements `__iter__()` for sequential access

This enables:
$$
\text{Sample} \sim \text{Stream}(\text{DataSource}) \rightarrow \text{Process} \rightarrow \text{Batch}
$$

**Key Advantages:**
1. **Constant memory**: O(buffer_size) regardless of dataset size
2. **Deterministic shuffling**: Via shuffle buffers (reservoir sampling)
3. **Fault tolerance**: Can resume from any point in stream
4. **Multi-worker friendly**: Each worker gets different shard

**How This Compares to Alternatives:**
- **Memory-mapped**: Requires all data on local disk
- **Streaming**: Works with cloud storage, infinite data streams
- **Trade-off**: Cannot randomly sample; must process sequentially

**Key Insight**: Streaming converts large-scale training from a "storage problem" to a "network bandwidth problem" - much easier to solve in cloud environments.

**Implementation Details:**
The shuffle buffer uses **reservoir sampling** to maintain randomness:
1. Fill buffer with first $k$ samples
2. For each new sample $x_i$ (where $i > k$):
   - With probability $k/i$, replace random buffer element with $x_i$
3. Ensures each sample has equal probability of being in buffer

```python
from torch.utils.data import IterableDataset
import random

class StreamingDataset(IterableDataset):
    """Streaming dataset that loads data on-the-fly.

    Useful for:
    - Very large datasets that don't fit on disk
    - Distributed training across many nodes
    - Datasets stored in cloud storage
    """
    def __init__(
        self,
        file_paths: list,
        tokenizer,
        block_size: int = 512,
        shuffle_buffer_size: int = 10000
    ):
        """
        Args:
            file_paths: List of paths to text files
            tokenizer: Tokenizer function
            block_size: Sequence length
            shuffle_buffer_size: Size of shuffle buffer
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.shuffle_buffer_size = shuffle_buffer_size

    def _process_file(self, file_path: str):
        """Process a single file and yield chunks."""
        with open(file_path, 'r') as f:
            # Read in chunks to avoid loading entire file
            buffer = []
            for line in f:
                tokens = self.tokenizer(line.strip())
                buffer.extend(tokens)

                # Yield complete blocks
                while len(buffer) >= self.block_size + 1:
                    chunk = buffer[:self.block_size + 1]
                    buffer = buffer[self.block_size:]

                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y

    def __iter__(self):
        # Shuffle file order
        file_paths = self.file_paths.copy()
        random.shuffle(file_paths)

        # Process files
        for file_path in file_paths:
            yield from self._process_file(file_path)


# Example usage:
# dataset = StreamingDataset(['file1.txt', 'file2.txt'], tokenizer, block_size=512)
# dataloader = DataLoader(dataset, batch_size=16)
```

#### 3. HuggingFace Datasets (Recommended for Production)

```python
from datasets import load_dataset

def create_hf_dataset(dataset_name: str, tokenizer, block_size: int = 512):
    """Use HuggingFace datasets library for efficient data loading.

    Benefits:
    - Automatic caching and memory mapping
    - Supports streaming from cloud storage
    - Built-in data processing pipelines
    - Works seamlessly with distributed training
    """
    # Load dataset (automatically cached and memory-mapped)
    dataset = load_dataset(dataset_name, split='train', streaming=True)

    def tokenize_function(examples):
        # Tokenize text
        tokens = tokenizer(examples['text'])
        return {'input_ids': tokens}

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    def group_texts(examples):
        # Concatenate all texts and split into chunks
        concatenated = sum(examples['input_ids'], [])

        # Split into blocks
        total_length = len(concatenated)
        total_length = (total_length // block_size) * block_size

        result = {
            'input_ids': [
                concatenated[i:i + block_size]
                for i in range(0, total_length, block_size)
            ]
        }
        return result

    # Group into blocks
    blocked_dataset = tokenized_dataset.map(
        group_texts,
        batched=True
    )

    return blocked_dataset


# Example usage:
# from datasets import load_dataset
# dataset = create_hf_dataset('openwebtext', tokenizer, block_size=512)
# dataloader = DataLoader(dataset, batch_size=16)
```

#### Dataset Comparison

| Method | Memory Usage | Speed | Complexity | Use Case |
|--------|--------------|-------|------------|----------|
| **In-memory** | High (all data in RAM) | Fastest | Simple | Small datasets (<1GB) |
| **Memory-mapped** | Low (OS manages) | Fast | Medium | Large local datasets |
| **Streaming** | Very low | Medium | Medium | Very large or remote data |
| **HuggingFace** | Low (automatic) | Fast | Low | Production (recommended) |

#### Best Practices

1. **For datasets < 1GB**: Use in-memory dataset
2. **For datasets 1GB-100GB**: Use memory-mapped dataset
3. **For datasets > 100GB**: Use streaming or HuggingFace datasets
4. **For distributed training**: Always use HuggingFace datasets or custom streaming

---

## Training the Model

### The Problem: Stable and Efficient Optimization

Training large language models is notoriously difficult due to:
1. **Numerical instability**: Gradients can explode or vanish in deep networks
2. **Optimization challenges**: High-dimensional non-convex loss landscape
3. **Computational cost**: Billions of parameters and tokens require efficient training

**Why These Training Techniques Matter:**

Modern LLMs use a carefully designed training recipe that has been refined over years:
- **AdamW optimizer**: Handles sparse gradients better than SGD, decouples weight decay
- **Learning rate warmup**: Prevents early training instability
- **Cosine decay**: Gradually reduces LR for better final convergence
- **Gradient clipping**: Prevents explosive gradients in deep networks

**Theoretical Justification:**

**1. AdamW Optimizer:**
$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(momentum)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(variance)} \\
\hat{m}_t &= m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t) \quad \text{(bias correction)} \\
\theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \theta_{t-1} \quad \text{(update with weight decay)}
\end{align}
$$

Key insight: AdamW applies weight decay **directly to weights**, not to gradients (unlike Adam). This improves generalization.

**2. Learning Rate Schedule:**

Warmup + cosine decay:
$$
\alpha(t) = \begin{cases}
\alpha_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\
\alpha_{\text{max}} \cdot \frac{1}{2}\left(1 + \cos\left(\pi \frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}}\right)\right) & \text{otherwise}
\end{cases}
$$

- **Warmup** (1-2% of steps): Prevents large updates when Adam statistics are poorly estimated
- **Cosine decay**: Smooth reduction allows model to settle into better minima

**3. Gradient Clipping:**

Scale gradients if their norm exceeds threshold:
$$
\tilde{g} = \begin{cases}
g & \text{if } \|g\| \leq \tau \\
\tau \frac{g}{\|g\|} & \text{otherwise}
\end{cases}
$$

This prevents rare catastrophic updates that can destabilize training.

**How This Relates to Alternatives:**
- **SGD with momentum**: Simpler but requires careful LR tuning, slower convergence
- **Adam**: Original version couples weight decay with gradients (less effective)
- **Lion, Sophia**: Recent optimizers with potential benefits, less battle-tested

**Key Insight**: The training recipe is as important as the architecture. Using wrong hyperparameters (e.g., no warmup, wrong LR) can completely prevent training.

Now let's train our modern transformer on the text data.

See [Language Model Training](15-lm-training.md) for more advanced training techniques.

```python
import torch.nn.functional as F
from tqdm import tqdm

def train_language_model(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    grad_clip: float = 1.0
):
    """Train a language model with causal language modeling objective.

    Args:
        model: Transformer model
        dataloader: Training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        grad_clip: Gradient clipping threshold
    """
    model = model.to(device)
    model.train()

    # AdamW optimizer (standard for transformers)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Learning rate schedule: warmup + cosine decay
    def get_lr(step, warmup_steps=100, max_steps=num_epochs * len(dataloader)):
        if step < warmup_steps:
            return learning_rate * (step / warmup_steps)
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            logits = model(x)

            # Compute loss: cross-entropy over vocabulary
            # Reshape: (batch * seq_len, vocab_size) and (batch * seq_len)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.6f}'
            })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    return model


### The Problem: GPU Memory Constraints and Training Speed

**Why Mixed Precision Training is Essential:**

Training large models is limited by:
1. **GPU memory**: A 7B model in FP32 requires ~28GB just for parameters
2. **Memory bandwidth**: Moving FP32 tensors is slow
3. **Compute throughput**: Modern GPUs (Ampere, Hopper) have 2-8x more FP16/BF16 compute than FP32

**Theoretical Foundation:**

Mixed precision uses different numerical formats for different operations:

**Floating Point Formats:**
- **FP32** (32 bits): 1 sign + 8 exponent + 23 mantissa → range $\approx 10^{-38}$ to $10^{38}$, precision $\approx 7$ decimal digits
- **FP16** (16 bits): 1 sign + 5 exponent + 10 mantissa → range $\approx 10^{-5}$ to $10^{5}$, precision $\approx 3$ decimal digits
- **BF16** (16 bits): 1 sign + 8 exponent + 7 mantissa → same range as FP32, precision $\approx 2$ decimal digits

**The Mixed Precision Strategy:**
$$
\begin{align}
\text{Forward/Backward:} &\quad \text{FP16/BF16} \quad \text{(2x memory, 2-8x faster compute)} \\
\text{Weights (master copy):} &\quad \text{FP32} \quad \text{(numerical stability)} \\
\text{Optimizer states:} &\quad \text{FP32} \quad \text{(accumulation precision)} \\
\text{Loss scaling:} &\quad \text{FP32} \quad \text{(prevent underflow)}
\end{align}
$$

**Key Technique: Loss Scaling**

Problem: FP16 gradients can underflow (become 0) for small values.
Solution: Scale loss by $S$ before backward pass:
$$
\begin{align}
\mathcal{L}_{\text{scaled}} &= S \cdot \mathcal{L} \\
g_{\text{scaled}} &= \nabla_\theta \mathcal{L}_{\text{scaled}} = S \cdot g \\
g &= g_{\text{scaled}} / S \quad \text{(unscale before optimizer)}
\end{align}
$$

The GradScaler dynamically adjusts $S$:
- Increase $S$ if no overflow occurs (capture smaller gradients)
- Decrease $S$ if overflow detected (prevent NaN)

**Memory Savings Breakdown:**

For a 7B parameter model (batch=4, seq=2048):
- **Model weights**: 28GB (FP32) → 14GB (FP16) = **14GB saved**
- **Activations**: ~16GB (FP32) → ~8GB (FP16) = **8GB saved**
- **Gradients**: 28GB (FP32) → 14GB (FP16) = **14GB saved**
- **Total**: ~72GB → ~36GB = **50% reduction**

**BF16 vs FP16:**
- **BF16 advantages**: Same range as FP32, no loss scaling needed, more stable
- **FP16 advantages**: Higher precision for values in range
- **Modern practice**: Use BF16 on Ampere+ GPUs (A100, H100), FP16 on older (V100)

**How This Relates to Alternatives:**
- **Pure FP32**: Most stable but 2x memory and slower
- **Pure FP16**: Risk of underflow/overflow, needs careful tuning
- **Mixed precision**: Best of both worlds - speed + stability
- **INT8 quantization**: Even faster but mainly for inference

**Key Insight**: Mixed precision is a form of "computational regularization" - the slight noise from reduced precision can actually improve generalization.

def train_with_mixed_precision(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    grad_clip: float = 1.0,
    gradient_accumulation_steps: int = 1
):
    """Train with mixed precision (FP16/BF16) for memory efficiency and speed.

    Mixed precision training:
    - Uses FP16/BF16 for forward/backward passes (2x memory savings)
    - Uses FP32 for optimizer states (maintains numerical stability)
    - Can train 2x larger models or use 2x larger batch sizes
    - Often provides 1.5-3x speedup on modern GPUs

    Args:
        model: Transformer model
        dataloader: Training data
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        grad_clip: Gradient clipping threshold
        gradient_accumulation_steps: Accumulate gradients over N steps
                                       (simulates larger batch size)
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # GradScaler for mixed precision training
    # Handles loss scaling to prevent underflow in FP16
    scaler = torch.cuda.amp.GradScaler()

    # Learning rate schedule
    def get_lr(step, warmup_steps=100, max_steps=num_epochs * len(dataloader)):
        if step < warmup_steps:
            return learning_rate * (step / warmup_steps)
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Mixed precision forward pass
            # autocast automatically uses FP16/BF16 for compatible operations
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            # Scaled backward pass
            scaler.scale(loss).backward()

            # Only update weights every N steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients and clip
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                # Optimizer step with scaling
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Track metrics
            epoch_loss += loss.item() * gradient_accumulation_steps
            global_step += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'lr': f'{lr:.6f}'
            })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    return model


# Example: Training with different precision modes
"""
# FP32 (baseline)
model_fp32 = train_language_model(model, dataloader)

# FP16 (2x memory savings, ~2x speedup)
model_fp16 = train_with_mixed_precision(model, dataloader)

# BF16 (better numerical stability, requires Ampere+ GPUs)
# Change autocast dtype to torch.bfloat16

# With gradient accumulation (simulate larger batch)
model_grad_accum = train_with_mixed_precision(
    model,
    dataloader,
    gradient_accumulation_steps=4  # Effective batch = 4x actual batch
)
"""


# Create model
model = ModernTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=256,
    n_heads=8,
    n_kv_heads=4,
    n_layers=6,
    max_seq_len=128,
    dropout=0.1
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Train
trained_model = train_language_model(
    model,
    dataloader,
    num_epochs=10,
    learning_rate=3e-4
)
```

### Generation Example

```python
@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> str:
    """Generate text from a prompt.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Text prompt
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature
        device: Device

    Returns:
        Generated text
    """
    model.eval()
    model = model.to(device)

    # Encode prompt
    input_ids = torch.tensor(
        tokenizer(prompt), dtype=torch.long
    ).unsqueeze(0).to(device)

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=50
    )

    # Decode
    output_text = tokenizer.decode(output_ids[0].tolist())

    return output_text


# Generate some text
prompt = "To be, or not to be"
generated = generate_text(
    trained_model,
    tokenizer,
    prompt,
    max_tokens=100,
    temperature=0.8
)
print(f"\nPrompt: {prompt}")
print(f"Generated: {generated}")
```

### Model Saving and Loading

Production models need to be saved and loaded for checkpointing, deployment, and sharing. Here's how to properly save and load transformer models:

```python
import os
from pathlib import Path

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str,
    model_config: dict = None
):
    """Save model checkpoint with all training state.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        checkpoint_path: Path to save checkpoint
        model_config: Model configuration dict (for reconstruction)
    """
    # Create checkpoint directory
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Include model config for easy reconstruction
    if model_config is not None:
        checkpoint['model_config'] = model_config

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into (if None, only returns checkpoint)
        optimizer: Optimizer to load state into (optional)
        device: Device to load model on

    Returns:
        Dictionary with checkpoint info (epoch, loss, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"with loss {checkpoint['loss']:.4f}")

    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'model_config': checkpoint.get('model_config', None)
    }


def save_model_for_inference(
    model: nn.Module,
    save_path: str,
    model_config: dict
):
    """Save model for inference (weights only, no optimizer state).

    Args:
        model: Trained model
        save_path: Path to save model
        model_config: Model configuration
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config
    }

    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")


def load_model_for_inference(
    load_path: str,
    model_class: type,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
    """Load model for inference.

    Args:
        load_path: Path to saved model
        model_class: Model class (e.g., ModernTransformer)
        device: Device to load on

    Returns:
        Loaded model in eval mode
    """
    save_dict = torch.load(load_path, map_location=device)

    # Reconstruct model from config
    model = model_class(**save_dict['model_config'])
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded from {load_path}")
    return model


# Example: Training with checkpointing
def train_with_checkpointing(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    checkpoint_dir: str = './checkpoints',
    save_every: int = 1,
    model_config: dict = None
):
    """Train model with periodic checkpointing.

    Args:
        model: Model to train
        dataloader: Training data
        num_epochs: Number of epochs
        learning_rate: Learning rate
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        model_config: Model configuration dict
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Resume from checkpoint if exists
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest.pt')
    start_epoch = 0
    if os.path.exists(latest_checkpoint):
        info = load_checkpoint(latest_checkpoint, model, optimizer, device)
        start_epoch = info['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            # Save numbered checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'
            )
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                checkpoint_path, model_config
            )

            # Save as latest (for resuming)
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                latest_checkpoint, model_config
            )

    # Save final model for inference
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    save_model_for_inference(model, final_model_path, model_config)

    return model


# Example usage:
model_config = {
    'vocab_size': tokenizer.vocab_size,
    'd_model': 256,
    'n_heads': 8,
    'n_kv_heads': 4,
    'n_layers': 6,
    'max_seq_len': 128
}

model = ModernTransformer(**model_config)

# Train with checkpointing
trained_model = train_with_checkpointing(
    model,
    dataloader,
    num_epochs=10,
    checkpoint_dir='./checkpoints',
    save_every=2,  # Save every 2 epochs
    model_config=model_config
)

# Later: Load for inference
loaded_model = load_model_for_inference(
    './checkpoints/final_model.pt',
    ModernTransformer
)
```

### Best Practices for Model Persistence

1. **Checkpointing Strategy**:
   - Save checkpoints every N epochs or steps
   - Keep the last K checkpoints (delete old ones to save space)
   - Save best checkpoint based on validation loss
   - Save optimizer state for resuming training

2. **What to Save**:
   - **Training checkpoint**: Model weights + optimizer state + epoch + loss
   - **Inference model**: Model weights + config only
   - **Full snapshot**: Add learning rate scheduler, RNG state for exact reproducibility

3. **File Organization**:

   <svg viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; background-color: white;">
     <!-- Root directory -->
     <text x="20" y="30" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#333" font-weight="600">checkpoints/</text>

     <!-- Vertical line from root -->
     <line x1="20" y1="40" x2="20" y2="170" stroke="#4A90A4" stroke-width="2"/>

     <!-- File 1: latest.pt -->
     <line x1="20" y1="50" x2="40" y2="50" stroke="#4A90A4" stroke-width="2"/>
     <text x="45" y="55" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#333">latest.pt</text>
     <text x="140" y="55" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#666" font-style="italic"># Most recent checkpoint (for resuming)</text>

     <!-- File 2: best.pt -->
     <line x1="20" y1="75" x2="40" y2="75" stroke="#4A90A4" stroke-width="2"/>
     <text x="45" y="80" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#333">best.pt</text>
     <text x="140" y="80" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#666" font-style="italic"># Best validation loss</text>

     <!-- File 3: checkpoint_epoch_10.pt -->
     <line x1="20" y1="100" x2="40" y2="100" stroke="#4A90A4" stroke-width="2"/>
     <text x="45" y="105" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#333">checkpoint_epoch_10.pt</text>
     <text x="240" y="105" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#666" font-style="italic"># Periodic checkpoints</text>

     <!-- File 4: checkpoint_epoch_20.pt -->
     <line x1="20" y1="125" x2="40" y2="125" stroke="#4A90A4" stroke-width="2"/>
     <text x="45" y="130" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#333">checkpoint_epoch_20.pt</text>

     <!-- File 5: final_model.pt (last item, use corner) -->
     <line x1="20" y1="150" x2="20" y2="150" stroke="#4A90A4" stroke-width="2"/>
     <line x1="20" y1="150" x2="40" y2="150" stroke="#4A90A4" stroke-width="2"/>
     <path d="M 20 150 Q 20 150, 25 150" fill="none" stroke="#4A90A4" stroke-width="2"/>
     <text x="45" y="155" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#333">final_model.pt</text>
     <text x="165" y="155" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#666" font-style="italic"># Final trained model (inference only)</text>
   </svg>

4. **Memory Considerations**:
   - Save models on CPU to free GPU memory: `model.cpu()`
   - Use `torch.save(..., _use_new_zipfile_serialization=True)` for large models
   - Consider saving in FP16 for smaller file sizes (inference only)

### Evaluation: Perplexity

**The Problem: Measuring Language Model Quality**

How do we quantify how well a language model has learned? We need a metric that:
1. Measures predictive accuracy across all vocabulary
2. Is interpretable and comparable across models
3. Correlates with downstream task performance

**Why Perplexity Matters:**

Perplexity is the standard metric for evaluating generative language models. It directly measures how "surprised" the model is by the test data.

**Theoretical Foundation:**

**Definition:**
Perplexity is the exponential of the average negative log-likelihood:
$$
\text{PPL}(X) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)\right)
$$

where $T$ is the total number of tokens and $P(x_t \mid x_{<t}; \theta)$ is the model's predicted probability of token $x_t$.

**Intuitive Interpretation:**

Perplexity can be interpreted as the **effective vocabulary size** the model is uncertain about at each step.

- **PPL = 1**: Perfect prediction (model assigns probability 1 to correct token)
- **PPL = V**: Random guessing (uniform over vocabulary of size $V$)
- **PPL = 20**: Model is as uncertain as choosing uniformly among 20 tokens

**Mathematical Derivation:**

For a perfect uniform distribution over $k$ equally likely outcomes:
$$
H = -\sum_{i=1}^{k} \frac{1}{k} \log \frac{1}{k} = \log k
$$
$$
\text{PPL} = \exp(H) = \exp(\log k) = k
$$

**Relationship to Cross-Entropy:**
$$
\text{PPL} = \exp(\text{CrossEntropy}) = \exp\left(\mathcal{L}_{\text{CE}}\right)
$$

This means:
- **Lower perplexity = better model**
- Perplexity of 10 means average cross-entropy loss of $\ln(10) \approx 2.3$
- Perplexity of 100 means average cross-entropy loss of $\ln(100) \approx 4.6$

**Typical Perplexity Values:**

| Model Type | Dataset | Typical PPL |
|------------|---------|-------------|
| Small model (100M) | WikiText-103 | 30-50 |
| Medium model (1B) | WikiText-103 | 15-25 |
| Large model (7B+) | WikiText-103 | 10-15 |
| Character-level | Text | 1.5-3.0 |

**How This Relates to Alternatives:**
- **Accuracy**: Too coarse (ignores confidence in predictions)
- **Cross-entropy loss**: Less interpretable (perplexity has intuitive meaning)
- **BLEU, ROUGE**: For specific generation tasks (translation, summarization)
- **Human evaluation**: Gold standard but expensive and not reproducible

**Key Insights:**
1. Perplexity is **dataset-dependent**: Only comparable on same test set
2. Lower perplexity ≠ better generation quality (correlation not perfect)
3. Perplexity measures **calibration**: How well probabilities match true distribution
4. For downstream tasks, task-specific metrics often more relevant

**Limitations:**
- Doesn't measure generation quality (fluency, factuality)
- Sensitive to tokenization (BPE vs character vs word level)
- Can be "gamed" by overfitting to test distribution

```python
@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> float:
    """Compute perplexity on a dataset.

    Perplexity = exp(average cross-entropy loss)

    Lower perplexity indicates better model.

    Args:
        model: Trained model
        dataloader: Data to evaluate on
        device: Device

    Returns:
        Perplexity value
    """
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    total_tokens = 0

    for x, y in tqdm(dataloader, desc="Computing perplexity"):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


# Compute perplexity
perplexity = compute_perplexity(trained_model, dataloader)
print(f"Perplexity: {perplexity:.2f}")
```

---

## Summary and Best Practices

### Architecture Choices

**For Production LLMs (2024+):**

| Component | Recommendation | Reason |
|-----------|----------------|--------|
| **Architecture** | Decoder-only | Simplicity, scalability, versatility |
| **Normalization** | RMSNorm (pre-norm) | Faster, simpler, equally effective |
| **Position** | RoPE | Better extrapolation to longer sequences |
| **Attention** | GQA | Balanced memory/quality trade-off |
| **Activation** | SwiGLU | Better performance (worth extra params) |
| **Bias Terms** | No bias | Simplification with minimal impact |

**Model Scaling:**

- **Small (100M-1B)**: Use for experimentation, distillation targets
- **Medium (7B-13B)**: Good balance for most applications
- **Large (30B-70B)**: State-of-the-art quality, expensive inference
- **Very Large (100B+)**: Typically use MoE for efficiency

### Training Best Practices

1. **Initialization**:
   - Use scaled initialization (std=0.02 for embeddings)
   - Scale down residual paths by 1/sqrt(n_layers)

2. **Optimization**:
   - AdamW optimizer with weight decay
   - Warmup learning rate (1-2% of total steps)
   - Cosine decay schedule
   - Gradient clipping (norm=1.0)

3. **Regularization**:
   - Dropout (0.1) during training, 0.0 for large models
   - Weight decay (0.1)
   - Label smoothing for classification

4. **Data**:
   - Large, diverse corpus (trillions of tokens for large models)
   - Careful deduplication and filtering
   - See [Data Curation and Preprocessing](14-data-curation.md)

5. **Efficiency**:
   - Mixed precision training (BF16/FP16) - see training section above
   - Gradient accumulation for large batches - see `train_with_mixed_precision`
   - Gradient checkpointing for memory savings
   - Flash Attention for long sequences
   - See [Flash Attention](12-flash-attention.md)

### Gradient Checkpointing

Gradient checkpointing (also called activation checkpointing) trades compute for memory by recomputing activations during the backward pass instead of storing them. This allows training much larger models on limited GPU memory.

**Trade-off**:
- Memory savings: ~50-75% reduction in activation memory
- Compute cost: ~30-40% increase in training time
- Use when: Memory-bound (can't fit larger batch or model)

```python
import torch.utils.checkpoint as checkpoint

class ModernTransformerBlockWithCheckpointing(nn.Module):
    """Transformer block with optional gradient checkpointing."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        use_checkpointing: bool = False
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, max_seq_len, dropout
        )
        self.ffn = SwiGLU(d_model, d_ff)

    def _forward_impl(self, x: torch.Tensor, mask: torch.Tensor):
        """The actual forward computation."""
        # Attention
        x = x + self.attn(self.attn_norm(x), mask)
        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing
            # Don't store intermediate activations, recompute during backward
            return checkpoint.checkpoint(
                self._forward_impl,
                x,
                mask,
                use_reentrant=False
            )
        else:
            # Normal forward pass
            return self._forward_impl(x, mask)


class ModernTransformerWithCheckpointing(nn.Module):
    """Modern transformer with gradient checkpointing support."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 4096,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        n_layers: int = 32,
        d_ff: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        use_checkpointing: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = 256 * ((d_ff + 255) // 256)

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            ModernTransformerBlockWithCheckpointing(
                d_model, n_heads, n_kv_heads, d_ff, max_seq_len,
                dropout, use_checkpointing
            )
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        device = input_ids.device

        x = self.token_embedding(input_ids)
        mask = self._create_causal_mask(seq_len, device)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


# Memory comparison for a 7B model (d_model=4096, 32 layers):
# Without checkpointing: ~24GB activation memory (batch=1, seq=2048)
# With checkpointing:    ~6GB activation memory (batch=1, seq=2048)
# Benefit: Can use 4x larger batch size or train on smaller GPUs

# Example usage:
model_without_cp = ModernTransformerWithCheckpointing(
    vocab_size=32000,
    d_model=4096,
    n_layers=32,
    use_checkpointing=False  # Store all activations
)

model_with_cp = ModernTransformerWithCheckpointing(
    vocab_size=32000,
    d_model=4096,
    n_layers=32,
    use_checkpointing=True  # Recompute activations during backward
)

# When to use gradient checkpointing:
# 1. Training very large models (>1B parameters)
# 2. Limited GPU memory
# 3. Want to use larger batch sizes
# 4. Memory-bound rather than compute-bound
#
# When NOT to use:
# 1. Small models that fit comfortably in memory
# 2. Already compute-bound (will make training even slower)
# 3. Inference (no backward pass, so no benefit)
```

### Memory-Efficient Training Techniques Comparison

| Technique | Memory Savings | Speed Impact | When to Use |
|-----------|----------------|--------------|-------------|
| **Mixed Precision (FP16)** | 50% | +50% faster | Always (modern GPUs) |
| **Gradient Checkpointing** | 50-75% activations | -30% slower | Memory-bound training |
| **Gradient Accumulation** | None | Neutral | Simulate larger batch |
| **Flash Attention** | O(N²) → O(N) | +2-4x faster | Long sequences (>512) |
| **GQA vs MHA** | 50-75% KV cache | Minimal | Inference with long context |

**Recommended Stack for Large Model Training**:
1. Mixed precision (BF16 on A100/H100, FP16 on V100)
2. Flash Attention for sequence length >512
3. Gradient checkpointing if memory-bound
4. Gradient accumulation to reach target batch size
5. GQA for efficient inference

### Common Pitfalls

1. **Not using pre-normalization**: Post-norm is harder to train at scale
2. **Wrong attention mask**: Ensure causal mask for decoders
3. **Forgetting weight tying**: Tie embeddings with output projection
4. **Poor initialization**: Can lead to training instability
5. **Too aggressive learning rate**: Use warmup and decay

---

## References

### Papers

1. **Original Transformer**:
   - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

2. **Decoder-Only (GPT)**:
   - [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019)
   - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)

3. **Encoder-Only (BERT)**:
   - [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)

4. **Encoder-Decoder**:
   - [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683) (Raffel et al., 2019)
   - [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) (Lewis et al., 2019)

5. **Modern Improvements**:
   - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
   - [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019)
   - [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020)
   - [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)

6. **LLaMA Series**:
   - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023)
   - [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) (Touvron et al., 2023)
   - [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Llama Team, 2024)

### Code References

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal GPT implementation
- [minGPT](https://github.com/karpathy/minGPT) - Educational GPT
- [LLaMA](https://github.com/facebookresearch/llama) - Meta's LLaMA
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face library

---

## Exercises

### Conceptual Questions

1. **Attention Patterns**: Why do encoders use bidirectional attention while decoders use causal attention? What happens if you swap them?

2. **Cross-Attention**: In an encoder-decoder model, why does the decoder's cross-attention use queries from the decoder but keys and values from the encoder?

3. **Weight Tying**: Why is it beneficial to tie the token embedding matrix with the output projection matrix? What are the trade-offs?

4. **Pre-norm vs Post-norm**: Explain why pre-normalization (normalize before sub-layer) is more stable than post-normalization (normalize after) for deep models.

5. **Architecture Choice**: When would you choose an encoder-decoder model over a decoder-only model? Give specific use cases.

### Implementation Exercises

1. **Add KV Caching**: Implement key-value caching for the decoder to speed up autoregressive generation. The cache should store past key and value states to avoid recomputing them.

2. **Implement MQA**: Modify the `GroupedQueryAttention` class to support Multi-Query Attention (MQA) where `n_kv_heads=1`. Compare KV cache memory usage with MHA and GQA.

3. **Add ALiBi**: Replace RoPE with ALiBi (Attention with Linear Biases) positional encoding. ALiBi adds a constant bias to attention scores based on distance.

4. **Beam Search**: Implement beam search for generation instead of sampling. Compare the quality of outputs.

5. **Flash Attention Integration**: Replace the standard attention computation with Flash Attention (if you have access to the library). Measure the speedup for long sequences.

### Analysis Exercises

1. **Memory Analysis**: Calculate the memory required for:
   - Model parameters (7B model with d_model=4096, n_layers=32)
   - KV cache (batch=8, seq_len=2048, n_layers=32, n_kv_heads=8, head_dim=128)
   - Activations during training (batch=4, seq_len=2048)

2. **Attention Visualization**: Create visualizations of attention patterns for:
   - First layer vs last layer
   - Different attention heads
   - Short vs long-range dependencies

3. **Scaling Laws**: Train the same model at 3 different sizes (e.g., 6 layers, 12 layers, 24 layers). Plot loss vs model size and analyze the relationship.

4. **Positional Encoding Comparison**: Train three models with:
   - Learned absolute positions
   - Sinusoidal positions
   - RoPE

   Test their ability to extrapolate to longer sequences than seen during training.

5. **Activation Function Ablation**: Compare SwiGLU, GELU, and ReLU activations. Measure training speed, model quality, and parameter count.

### Project: Build a Mini-LLM

**Objective**: Train a small language model (10M-100M parameters) from scratch.

**Steps**:
1. Collect a small text corpus (e.g., all of Shakespeare's works)
2. Train a tokenizer (BPE with 2000-5000 tokens)
3. Implement and train the modern transformer architecture
4. Implement temperature, top-k, and top-p sampling
5. Create a simple chat interface
6. Fine-tune on a specific task (e.g., question answering)
7. Measure perplexity and generate sample outputs

**Extensions**:
- Add instruction tuning (see [Supervised Fine-tuning](18-sft.md))
- Implement LoRA for efficient fine-tuning (see [LoRA and PEFT](19-peft.md))
- Deploy with inference optimizations

---

**Next Chapter**: [Flash Attention](12-flash-attention.md) - Learn how to make attention computations faster and more memory-efficient.

**Previous Chapters**:
- [The Transformer Block](09-transformer-block.md) - Building blocks
- [Multi-Head Attention](04-multi-head-attention.md) - Attention mechanism
- [Rotary Position Embeddings (RoPE)](08-rope.md) - Position encoding
- [Activation Functions](10-activation-functions.md) - SwiGLU and others
