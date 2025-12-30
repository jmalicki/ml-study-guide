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

---

## Training the Model

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

### Evaluation: Perplexity

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
   - Mixed precision training (BF16)
   - Gradient accumulation for large batches
   - Flash Attention for long sequences
   - See [Flash Attention](12-flash-attention.md)

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
