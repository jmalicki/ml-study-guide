# Chapter 27: Multimodality

Multimodal models extend language models to understand and generate content across multiple modalities—vision, audio, and text. This chapter covers the architectures and techniques that enable models like GPT-4V, LLaVA, and Gemini to process images, audio, and text together.

Understanding multimodality is crucial for modern ML interviews, as the frontier of AI research increasingly focuses on models that can reason across different input types.

## Table of Contents

1. [Introduction to Multimodality](#introduction-to-multimodality)
2. [Vision Encoders](#vision-encoders)
   - [Vision Transformer (ViT)](#vision-transformer-vit)
   - [CLIP](#clip)
   - [SigLIP](#siglip)
3. [Cross-Modal Attention](#cross-modal-attention)
4. [Vision-Language Models](#vision-language-models)
   - [LLaVA Architecture](#llava-architecture)
   - [GPT-4V and Gemini](#gpt-4v-and-gemini)
   - [Flamingo and GIT](#flamingo-and-git)
5. [Audio and Speech Integration](#audio-and-speech-integration)
   - [Whisper](#whisper)
   - [Speech-Language Models](#speech-language-models)
6. [Multimodal Tokenization Strategies](#multimodal-tokenization-strategies)
7. [Training Multimodal Models](#training-multimodal-models)
8. [Putting It All Together](#putting-it-all-together)

---

## Introduction to Multimodality

Language models traditionally operate on text, but the world contains information in many forms. Multimodal models bridge this gap by:

1. **Encoding** non-text data (images, audio) into representations compatible with language models
2. **Fusing** information from different modalities using attention mechanisms
3. **Generating** outputs that can be text, images, or other modalities

### Key Challenges

| Challenge | Solution Approaches |
|-----------|-------------------|
| **Representation Gap** | Learn shared embedding spaces (CLIP) |
| **Data Alignment** | Contrastive learning, caption generation |
| **Computational Cost** | Efficient encoders, adapter layers |
| **Training Data** | Web-scraped image-text pairs, synthetic data |

---

## Vision Encoders

Vision encoders transform images into token sequences that can be processed by language models.

### Vision Transformer (ViT)

ViT applies the transformer architecture directly to image patches, treating them as tokens.

**Architecture:**
1. Split image into fixed-size patches (e.g., 16x16 pixels)
2. Linearly embed each patch
3. Add positional embeddings
4. Process with standard transformer encoder
5. Use [CLS] token for image-level representation

```python
import torch
import torch.nn as nn
from typing import Tuple

class PatchEmbedding(nn.Module):
    """Convert image into sequence of patch embeddings.

    For a 224x224 image with 16x16 patches:
    - Number of patches: (224/16)^2 = 196
    - Each patch: 16*16*3 = 768 values (for RGB)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Convolution to extract patches and project to embed_dim
        # This is equivalent to splitting into patches + linear projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            (batch, n_patches, embed_dim)
        """
        # Conv2d output: (batch, embed_dim, n_patches_h, n_patches_w)
        x = self.proj(x)

        # Flatten spatial dimensions
        x = x.flatten(2)  # (batch, embed_dim, n_patches)

        # Transpose to (batch, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) implementation.

    Key insight: Images can be treated as sequences of patches,
    just like text is a sequence of tokens.

    Architecture:
    1. Patch embedding
    2. Position embedding
    3. Transformer encoder
    4. Classification head (or use as encoder)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 1000
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        n_patches = self.patch_embed.n_patches

        # Learnable [CLS] token for image representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings for patches + CLS token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        # In practice, use nn.TransformerEncoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head (optional)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            Image embeddings (batch, n_patches+1, embed_dim)
            or class logits if using as classifier
        """
        batch_size = x.shape[0]

        # Patch embeddings
        x = self.patch_embed(x)  # (batch, n_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return all tokens (for multimodal) or just CLS (for classification)
        return x

    def get_image_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image features for multimodal models."""
        return self.forward(x)  # Returns all patch embeddings


class TransformerBlock(nn.Module):
    """Standard transformer block for ViT."""
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x
```

**Key Papers:**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020)

### CLIP

CLIP (Contrastive Language-Image Pre-training) learns aligned representations of images and text through contrastive learning.

**Architecture:**
- **Image Encoder**: ViT or ResNet
- **Text Encoder**: Transformer
- **Training**: Contrastive loss on image-text pairs

$$
\mathcal{L}_{\text{CLIP}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)} \right]
$$

where $\text{sim}(I, T) = \frac{I \cdot T}{\|I\| \|T\|}$ is cosine similarity and $\tau$ is temperature.

```python
class CLIP(nn.Module):
    """Simplified CLIP implementation.

    CLIP learns to align image and text representations by
    maximizing similarity between matching pairs and minimizing
    similarity between non-matching pairs.

    Training data: 400M image-text pairs from the web
    """
    def __init__(
        self,
        embed_dim: int = 512,
        vision_width: int = 768,
        text_width: int = 512,
        vocab_size: int = 49408,
        max_text_len: int = 77
    ):
        super().__init__()

        # Vision encoder (ViT)
        self.visual = VisionTransformer(
            embed_dim=vision_width,
            depth=12,
            n_heads=12
        )

        # Text encoder (Transformer)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_width,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=12
        )

        self.token_embedding = nn.Embedding(vocab_size, text_width)
        self.positional_embedding = nn.Parameter(
            torch.zeros(max_text_len, text_width)
        )

        # Projection heads to shared embedding space
        self.visual_projection = nn.Linear(vision_width, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_width, embed_dim, bias=False)

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to shared embedding space."""
        x = self.visual(image)
        # Use CLS token (first token)
        x = x[:, 0]
        x = self.visual_projection(x)
        # L2 normalize
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to shared embedding space.

        Args:
            text: (batch, seq_len) token indices
        """
        # Embed tokens
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:text.shape[1]]

        # Encode with transformer
        x = self.text_encoder(x)

        # Use last token (EOS token) as text representation
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = self.text_projection(x)

        # L2 normalize
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarity matrix between images and texts.

        Returns:
            logits_per_image: (batch, batch) - image-to-text similarities
            logits_per_text: (batch, batch) - text-to-image similarities
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text


def clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Symmetric cross-entropy loss for CLIP.

    The diagonal represents correct image-text pairs.
    Off-diagonal elements are negatives.
    """
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)

    loss_i2t = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t2i = nn.functional.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2


# Example usage
def train_clip_step(model: CLIP, images: torch.Tensor, texts: torch.Tensor):
    """Single training step for CLIP."""
    logits_per_image, logits_per_text = model(images, texts)
    loss = clip_loss(logits_per_image, logits_per_text)
    return loss
```

**Key Capabilities:**
- Zero-shot image classification by comparing image embeddings with text embeddings of class names
- Image retrieval from text queries
- Foundation for vision-language models

**Key Papers:**
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021)

### SigLIP

SigLIP (Sigmoid Loss for Language Image Pre-training) improves upon CLIP by using a sigmoid loss instead of softmax.

**Key Difference:**
- **CLIP**: Uses softmax over all pairs in batch (requires large batches)
- **SigLIP**: Uses sigmoid on individual pairs (more stable, smaller batches)

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i,j} \log \sigma(y_{ij} \cdot z_{ij})
$$

where $y_{ij} = 1$ if image $i$ matches text $j$ else $-1$, and $z_{ij}$ is the similarity score.

```python
def siglip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """SigLIP loss using sigmoid instead of softmax.

    Benefits:
    - More stable training
    - Works with smaller batch sizes
    - Better performance on some tasks
    """
    batch_size = image_features.shape[0]

    # Compute similarity matrix
    similarities = temperature * image_features @ text_features.T

    # Create labels: 1 for diagonal (matching pairs), -1 for off-diagonal
    labels = 2 * torch.eye(batch_size, device=similarities.device) - 1

    # Sigmoid loss
    loss = -torch.log(torch.sigmoid(labels * similarities)).mean()

    return loss
```

**Key Papers:**
- [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) (Zhai et al., 2023)

---

## Cross-Modal Attention

Cross-modal attention allows the language model to attend to visual features. See [Cross-Attention](06-cross-attention.md) for the general mechanism.

```python
class CrossModalAttention(nn.Module):
    """Cross-attention from text (query) to visual features (key, value).

    This allows language tokens to attend to relevant parts of an image.
    Used in models like Flamingo, BLIP-2, and others.
    """
    def __init__(
        self,
        text_dim: int,
        visual_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = text_dim // n_heads

        # Query from text
        self.q_proj = nn.Linear(text_dim, text_dim)

        # Key and Value from visual features
        self.k_proj = nn.Linear(visual_dim, text_dim)
        self.v_proj = nn.Linear(visual_dim, text_dim)

        self.out_proj = nn.Linear(text_dim, text_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text: torch.Tensor,
        visual: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            text: (batch, text_len, text_dim) - language model hidden states
            visual: (batch, n_patches, visual_dim) - visual encoder outputs
            mask: Optional attention mask

        Returns:
            (batch, text_len, text_dim) - attended text features
        """
        batch_size, text_len, _ = text.shape

        # Project to Q, K, V
        Q = self.q_proj(text).view(batch_size, text_len, self.n_heads, self.head_dim)
        K = self.k_proj(visual).view(batch_size, -1, self.n_heads, self.head_dim)
        V = self.v_proj(visual).view(batch_size, -1, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (batch, n_heads, text_len, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, text_len, -1)

        return self.out_proj(out)


class PerceiverResampler(nn.Module):
    """Perceiver Resampler from Flamingo.

    Compresses variable-length visual features into fixed number of tokens.
    This is more efficient than attending to all image patches.

    Key idea: Use learnable query tokens that attend to all visual features.
    """
    def __init__(
        self,
        visual_dim: int,
        output_dim: int,
        n_queries: int = 64,
        depth: int = 6,
        n_heads: int = 8
    ):
        super().__init__()

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, n_queries, output_dim))

        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': CrossModalAttention(output_dim, visual_dim, n_heads),
                'self_attn': nn.MultiheadAttention(output_dim, n_heads, batch_first=True),
                'mlp': nn.Sequential(
                    nn.Linear(output_dim, output_dim * 4),
                    nn.GELU(),
                    nn.Linear(output_dim * 4, output_dim)
                ),
                'norm1': nn.LayerNorm(output_dim),
                'norm2': nn.LayerNorm(output_dim),
                'norm3': nn.LayerNorm(output_dim),
            })
            for _ in range(depth)
        ])

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, n_patches, visual_dim)

        Returns:
            (batch, n_queries, output_dim) - compressed visual features
        """
        batch_size = visual_features.shape[0]

        # Expand queries for batch
        x = self.queries.expand(batch_size, -1, -1)

        for layer in self.layers:
            # Cross-attention to visual features
            x = x + layer['cross_attn'](layer['norm1'](x), visual_features)

            # Self-attention among query tokens
            attn_out, _ = layer['self_attn'](
                layer['norm2'](x), layer['norm2'](x), layer['norm2'](x)
            )
            x = x + attn_out

            # MLP
            x = x + layer['mlp'](layer['norm3'](x))

        return x
```

---

## Vision-Language Models

### LLaVA Architecture

LLaVA (Large Language and Vision Assistant) connects a vision encoder to an LLM using a simple projection layer.

**Architecture Components:**
1. **Vision Encoder**: Pre-trained CLIP ViT
2. **Projection**: Linear layer to map visual features to LLM embedding space
3. **Language Model**: Pre-trained LLM (Vicuna, LLaMA)

```python
class LLaVA(nn.Module):
    """LLaVA: Large Language and Vision Assistant.

    Simple but effective architecture:
    1. Encode image with CLIP
    2. Project visual features to LLM embedding space
    3. Concatenate with text embeddings
    4. Process with LLM

    Training stages:
    1. Pre-training: Align vision and language (projection layer only)
    2. Instruction tuning: Fine-tune on visual instruction data
    """
    def __init__(
        self,
        vision_encoder: nn.Module,  # Pre-trained CLIP
        language_model: nn.Module,   # Pre-trained LLM
        vision_dim: int = 1024,
        language_dim: int = 4096
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.language_model = language_model

        # Simple linear projection (can also use MLP)
        self.vision_projection = nn.Linear(vision_dim, language_dim)

        # Freeze encoders initially
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images and project to language space."""
        # Get visual features from CLIP
        with torch.no_grad():
            visual_features = self.vision_encoder.get_image_features(images)

        # visual_features: (batch, n_patches, vision_dim)
        # Project to language dimension
        visual_embeds = self.vision_projection(visual_features)

        return visual_embeds

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            images: (batch, channels, height, width)
            input_ids: (batch, text_len) - text tokens
            attention_mask: Optional mask

        Returns:
            Language model outputs
        """
        # Encode image
        visual_embeds = self.encode_image(images)
        batch_size, n_visual_tokens, _ = visual_embeds.shape

        # Get text embeddings from language model
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Concatenate visual and text embeddings
        # Format: [visual tokens] [text tokens]
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # Create attention mask for combined sequence
        if attention_mask is not None:
            visual_mask = torch.ones(
                batch_size, n_visual_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask
        )

        return outputs

    def generate(
        self,
        images: torch.Tensor,
        prompts: torch.Tensor,
        max_length: int = 512,
        **generation_kwargs
    ) -> torch.Tensor:
        """Generate text from image and prompt."""
        visual_embeds = self.encode_image(images)
        prompt_embeds = self.language_model.get_input_embeddings()(prompts)

        # Combine embeddings
        combined_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)

        # Generate (simplified - actual implementation more complex)
        return self.language_model.generate(
            inputs_embeds=combined_embeds,
            max_length=max_length,
            **generation_kwargs
        )


# Training LLaVA
def train_llava_stage1(
    model: LLaVA,
    image_text_pairs: list,
    num_epochs: int = 1
):
    """Stage 1: Pre-training for vision-language alignment.

    Only train the projection layer.
    Use image captioning data (e.g., CC3M, LAION).
    """
    # Freeze everything except projection
    for param in model.parameters():
        param.requires_grad = False
    for param in model.vision_projection.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.vision_projection.parameters(),
        lr=1e-3
    )

    # Training loop (simplified)
    for epoch in range(num_epochs):
        for images, captions in image_text_pairs:
            # Forward pass
            outputs = model(images, captions)

            # Language modeling loss
            loss = outputs.loss  # Cross-entropy on next token prediction

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_llava_stage2(
    model: LLaVA,
    instruction_data: list,
    num_epochs: int = 3
):
    """Stage 2: Instruction tuning.

    Fine-tune projection + LLM on visual instruction data.
    Keep vision encoder frozen.
    """
    # Unfreeze LLM and projection
    for param in model.language_model.parameters():
        param.requires_grad = True
    for param in model.vision_projection.parameters():
        param.requires_grad = True

    # Use LoRA for efficient fine-tuning (see Chapter 19)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5
    )

    # Training loop (simplified)
    for epoch in range(num_epochs):
        for images, instructions, responses in instruction_data:
            # Forward pass
            outputs = model(images, instructions)

            # Only compute loss on response tokens
            loss = compute_instruction_loss(outputs, responses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_instruction_loss(outputs, responses):
    """Compute loss only on response tokens, not instruction."""
    # Implementation depends on how data is formatted
    # Typically mask out instruction tokens in loss computation
    pass
```

**LLaVA Training Data:**
- **Stage 1**: 595K image-caption pairs from COCO (filtered for quality)
- **Stage 2**: 158K visual instruction-following samples (GPT-4 generated)

**Key Papers:**
- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (Liu et al., 2023)
- [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744) (Liu et al., 2023)

### GPT-4V and Gemini

GPT-4V (GPT-4 with Vision) and Gemini are proprietary multimodal models with unpublished architectures.

**Known Characteristics:**

| Model | Vision Integration | Context | Key Features |
|-------|-------------------|---------|--------------|
| **GPT-4V** | Late fusion | Text + images | OCR, charts, complex reasoning |
| **Gemini** | Native multimodal | 1M+ tokens | Video, audio, text, images |

**Gemini Architecture (from papers):**
- Trained multimodally from scratch (not vision encoder + LLM)
- Sparse Mixture of Experts
- Native support for images, video, and audio
- Uses interleaved image-text sequences during training

```python
# Conceptual Gemini-style multimodal tokenization
class MultimodalTokenizer:
    """Conceptual approach for native multimodal models.

    Instead of separate encoders + fusion, learn a unified
    tokenizer that handles all modalities.
    """
    def __init__(self, vocab_size: int = 256000):
        # Text tokens: 0-128K
        # Image tokens: 128K-200K (learned via VQ-VAE)
        # Audio tokens: 200K-256K
        self.vocab_size = vocab_size
        self.text_vocab_size = 128000
        self.image_vocab_size = 72000
        self.audio_vocab_size = 56000

    def tokenize(self, data: dict) -> torch.Tensor:
        """
        Tokenize multimodal data into unified token sequence.

        Args:
            data: Dict with 'text', 'images', 'audio' keys

        Returns:
            Unified token sequence
        """
        tokens = []

        if 'text' in data:
            tokens.extend(self.tokenize_text(data['text']))

        if 'images' in data:
            # Use VQ-VAE or similar to discretize images
            image_tokens = self.tokenize_images(data['images'])
            tokens.extend(image_tokens + self.text_vocab_size)

        if 'audio' in data:
            audio_tokens = self.tokenize_audio(data['audio'])
            tokens.extend(
                audio_tokens + self.text_vocab_size + self.image_vocab_size
            )

        return torch.tensor(tokens)

    def tokenize_text(self, text: str) -> list:
        """Standard text tokenization (BPE, etc.)."""
        pass

    def tokenize_images(self, images: torch.Tensor) -> list:
        """Discretize images into tokens using VQ-VAE."""
        pass

    def tokenize_audio(self, audio: torch.Tensor) -> list:
        """Discretize audio into tokens."""
        pass
```

### Flamingo and GIT

**Flamingo** (DeepMind) uses Perceiver Resampler and gated cross-attention.

**GIT** (Generative Image-to-Text) uses a simpler architecture similar to early LLaVA.

**Key Papers:**
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (Alayrac et al., 2022)
- [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) (Wang et al., 2022)

---

## Audio and Speech Integration

### Whisper

Whisper is an encoder-decoder model for speech recognition and translation.

**Architecture:**
1. **Audio Encoder**: Convolutional layers + Transformer encoder
2. **Decoder**: Transformer decoder for text generation
3. **Multi-task Training**: Transcription, translation, language ID

```python
class WhisperEncoder(nn.Module):
    """Whisper audio encoder.

    Processes raw audio waveforms into embeddings suitable for
    transformer processing.
    """
    def __init__(
        self,
        n_mels: int = 80,
        n_ctx: int = 1500,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6
    ):
        super().__init__()

        # Mel spectrogram will be computed externally
        # Input: (batch, n_mels, time_steps)

        # Convolutional layers
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.zeros(n_ctx, d_model))

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        self.ln_post = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, n_mels, time) - mel spectrogram

        Returns:
            (batch, time, d_model) - audio embeddings
        """
        # Convolutional layers
        x = torch.relu(self.conv1(mel))
        x = torch.relu(self.conv2(x))

        # Transpose to (batch, time, d_model)
        x = x.transpose(1, 2)

        # Add positional embeddings
        x = x + self.positional_embedding[:x.shape[1]]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)


class WhisperDecoder(nn.Module):
    """Whisper text decoder."""
    def __init__(
        self,
        vocab_size: int = 51865,
        n_ctx: int = 448,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(n_ctx, d_model))

        # Transformer decoder blocks with cross-attention
        # See [Cross-Attention](06-cross-attention.md)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'cross_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'mlp': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
                'ln3': nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len) - text token IDs
            audio_features: (batch, audio_len, d_model) - from encoder

        Returns:
            (batch, seq_len, vocab_size) - logits
        """
        # Embed tokens
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding[:tokens.shape[1]]

        # Create causal mask for self-attention
        causal_mask = torch.triu(
            torch.ones(tokens.shape[1], tokens.shape[1]),
            diagonal=1
        ).bool()

        # Decoder blocks
        for block in self.blocks:
            # Self-attention (causal)
            attn_out, _ = block['self_attn'](
                block['ln1'](x),
                block['ln1'](x),
                block['ln1'](x),
                attn_mask=causal_mask
            )
            x = x + attn_out

            # Cross-attention to audio
            attn_out, _ = block['cross_attn'](
                block['ln2'](x),
                audio_features,
                audio_features
            )
            x = x + attn_out

            # MLP
            x = x + block['mlp'](block['ln3'](x))

        x = self.ln(x)

        # Project to vocabulary (weight tying with embeddings)
        logits = x @ self.token_embedding.weight.T

        return logits


class Whisper(nn.Module):
    """Complete Whisper model."""
    def __init__(self):
        super().__init__()
        self.encoder = WhisperEncoder()
        self.decoder = WhisperDecoder()

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        audio_features = self.encoder(mel)
        logits = self.decoder(tokens, audio_features)
        return logits

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor) -> str:
        """Transcribe audio to text."""
        # Convert audio to mel spectrogram (using librosa or similar)
        mel = audio_to_mel(audio)

        # Encode audio
        audio_features = self.encoder(mel)

        # Decode autoregressively
        tokens = [self.START_TOKEN]
        for _ in range(self.max_length):
            logits = self.decoder(torch.tensor([tokens]), audio_features)
            next_token = logits[0, -1].argmax().item()
            if next_token == self.END_TOKEN:
                break
            tokens.append(next_token)

        return self.decode_tokens(tokens)


def audio_to_mel(audio: torch.Tensor, n_mels: int = 80) -> torch.Tensor:
    """Convert audio waveform to mel spectrogram."""
    # Using torchaudio or librosa
    import torchaudio

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        n_mels=n_mels
    )

    mel = mel_transform(audio)
    # Convert to log scale
    mel = torch.log(mel + 1e-8)

    return mel
```

**Whisper Training:**
- 680,000 hours of multilingual and multitask data
- Trained on: transcription, translation, language identification, voice activity detection

**Key Papers:**
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Radford et al., 2022)

### Speech-Language Models

Modern speech-language models combine speech encoding with LLMs.

**Approaches:**
1. **Cascaded**: Whisper → LLM (simple but loses paralinguistic info)
2. **End-to-End**: Direct audio encoder → LLM (preserves tone, emotion)

```python
class SpeechLLM(nn.Module):
    """End-to-end speech language model.

    Similar to LLaVA but for audio instead of images.
    """
    def __init__(
        self,
        audio_encoder: nn.Module,  # e.g., Whisper encoder
        language_model: nn.Module,
        audio_dim: int = 512,
        language_dim: int = 4096
    ):
        super().__init__()

        self.audio_encoder = audio_encoder
        self.language_model = language_model

        # Project audio features to LLM space
        self.audio_projection = nn.Linear(audio_dim, language_dim)

    def forward(
        self,
        audio: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Process audio and text together."""
        # Encode audio
        audio_features = self.audio_encoder(audio)
        audio_embeds = self.audio_projection(audio_features)

        # Get text embeddings
        text_embeds = self.language_model.get_input_embeddings()(text_tokens)

        # Concatenate
        combined_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

        # Forward through LLM
        return self.language_model(inputs_embeds=combined_embeds)
```

---

## Multimodal Tokenization Strategies

Different approaches to handling multiple modalities:

### 1. Late Fusion (LLaVA, BLIP-2)

Encode each modality separately, then concatenate/fuse.

```python
def late_fusion(image_tokens, text_tokens):
    """
    Pros: Simple, can use pre-trained encoders
    Cons: Limited early interaction between modalities
    """
    return torch.cat([image_tokens, text_tokens], dim=1)
```

### 2. Early Fusion (Gemini approach)

Unified tokenizer for all modalities.

```python
def early_fusion(image, text):
    """
    Pros: Rich cross-modal interaction from start
    Cons: Requires training from scratch, more data
    """
    image_tokens = vq_vae.encode(image)  # Discrete image tokens
    text_tokens = tokenizer.encode(text)
    return interleave(image_tokens, text_tokens)
```

### 3. Hybrid (Flamingo)

Use Perceiver Resampler to compress visual tokens, then cross-attend.

```python
def hybrid_fusion(image_features, text_tokens, perceiver):
    """
    Pros: Efficient, flexible number of visual tokens
    Cons: More complex architecture
    """
    compressed_visual = perceiver(image_features)  # Fixed size
    # Use gated cross-attention in LM layers
    return process_with_cross_attention(compressed_visual, text_tokens)
```

---

## Training Multimodal Models

### Pre-training Strategies

```python
class MultimodalPretraining:
    """Common pre-training objectives for multimodal models."""

    @staticmethod
    def image_text_matching(image_emb, text_emb):
        """Binary classification: do image and text match?"""
        similarity = torch.cosine_similarity(image_emb, text_emb)
        # Labels: 1 for matching pairs, 0 for random pairs
        return similarity

    @staticmethod
    def masked_language_modeling(text_tokens, image_features, model):
        """MLM conditioned on image."""
        # Mask some text tokens
        masked_tokens = mask_tokens(text_tokens)

        # Predict masked tokens using image context
        predictions = model(image_features, masked_tokens)

        return predictions

    @staticmethod
    def image_captioning(image, caption, model):
        """Generate caption from image (autoregressive)."""
        # Standard language modeling loss
        logits = model(image, caption[:-1])
        loss = cross_entropy(logits, caption[1:])
        return loss

    @staticmethod
    def visual_question_answering(image, question, answer, model):
        """Answer questions about images."""
        outputs = model(image, question)
        # Generate answer or classify
        return outputs
```

### Data Efficiency

```python
def create_instruction_data_from_gpt4(images, captions):
    """
    LLaVA approach: Use GPT-4 to generate diverse instructions.

    Given image captions, GPT-4 generates:
    - Detailed descriptions
    - Questions and answers
    - Reasoning tasks
    """
    instruction_data = []

    for image, caption in zip(images, captions):
        # Prompt GPT-4 (text-only)
        prompt = f"""
        Given an image with caption: "{caption}"

        Generate:
        1. A detailed description instruction
        2. Three question-answer pairs
        3. A reasoning task
        """

        gpt4_response = call_gpt4(prompt)

        # Parse response and create training examples
        instruction_data.append({
            'image': image,
            'conversations': parse_gpt4_response(gpt4_response)
        })

    return instruction_data
```

---

## Putting It All Together

### Complete Multimodal Model Example

```python
class CompleteMultimodalModel(nn.Module):
    """
    Production-ready multimodal model combining best practices.

    Components:
    1. Vision encoder (CLIP or SigLIP)
    2. Audio encoder (Whisper)
    3. Perceiver resampler for each modality
    4. LLM with cross-modal attention
    """
    def __init__(
        self,
        vision_encoder: nn.Module,
        audio_encoder: nn.Module,
        language_model: nn.Module,
        vision_dim: int = 1024,
        audio_dim: int = 512,
        lm_dim: int = 4096,
        n_visual_queries: int = 64,
        n_audio_queries: int = 32
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.language_model = language_model

        # Perceiver resamplers to compress modalities
        self.vision_resampler = PerceiverResampler(
            vision_dim, lm_dim, n_visual_queries
        )
        self.audio_resampler = PerceiverResampler(
            audio_dim, lm_dim, n_audio_queries
        )

    def forward(
        self,
        images: torch.Tensor = None,
        audio: torch.Tensor = None,
        text_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """Process any combination of image, audio, and text."""

        modal_embeds = []

        # Process image if provided
        if images is not None:
            visual_features = self.vision_encoder(images)
            visual_embeds = self.vision_resampler(visual_features)
            modal_embeds.append(visual_embeds)

        # Process audio if provided
        if audio is not None:
            audio_features = self.audio_encoder(audio)
            audio_embeds = self.audio_resampler(audio_features)
            modal_embeds.append(audio_embeds)

        # Process text
        if text_tokens is not None:
            text_embeds = self.language_model.get_input_embeddings()(text_tokens)
            modal_embeds.append(text_embeds)

        # Concatenate all modalities
        combined_embeds = torch.cat(modal_embeds, dim=1)

        # Forward through language model
        outputs = self.language_model(inputs_embeds=combined_embeds)

        return outputs

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor = None,
        audio: torch.Tensor = None,
        prompt: str = "",
        max_length: int = 512
    ) -> str:
        """Generate text response from multimodal inputs."""

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)

        # Get outputs
        outputs = self.forward(images, audio, prompt_tokens)

        # Decode (simplified)
        return self.tokenizer.decode(outputs)


# Example usage
def example_multimodal_usage():
    """Demonstrate multimodal model usage."""

    # Initialize model
    model = CompleteMultimodalModel(
        vision_encoder=VisionTransformer(),
        audio_encoder=WhisperEncoder(),
        language_model=load_pretrained_llm()
    )

    # Load inputs
    image = load_image("cat.jpg")
    audio = load_audio("question.wav")
    prompt = "What is in this image and what does the audio say?"

    # Generate response
    response = model.generate(
        images=image,
        audio=audio,
        prompt=prompt
    )

    print(response)
    # "The image shows a cat sitting on a couch.
    #  The audio asks 'What color is the cat?'"
```

---

## Summary

### Key Takeaways

1. **Vision Encoders**:
   - ViT treats images as sequences of patches
   - CLIP aligns vision and language through contrastive learning
   - SigLIP improves CLIP with sigmoid loss

2. **Cross-Modal Fusion**:
   - Late fusion: Simple, uses pre-trained models
   - Early fusion: Rich interaction, requires more training
   - Perceiver Resampler: Efficient compression

3. **Architecture Patterns**:
   - **LLaVA**: Vision encoder → Projection → LLM (simple, effective)
   - **Flamingo**: Perceiver + gated cross-attention (efficient)
   - **Gemini**: Native multimodal from scratch (most capable)

4. **Training Stages**:
   - Pre-training: Align modalities (freeze most parameters)
   - Instruction tuning: Teach instruction following (fine-tune)

5. **Audio Integration**:
   - Whisper: Encoder-decoder for speech
   - Can integrate with LLMs similarly to vision

### Best Practices

1. **Use pre-trained encoders** when possible (CLIP, Whisper)
2. **Stage training**: Align first, then instruct
3. **LoRA for efficiency** when fine-tuning large LLMs (see [LoRA and PEFT](19-peft.md))
4. **Perceiver Resampler** for variable-length inputs
5. **Data quality over quantity** for instruction tuning

---

## References

### Key Papers

1. Dosovitskiy et al. (2020). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (ViT)
2. Radford et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (CLIP)
3. Zhai et al. (2023). [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) (SigLIP)
4. Liu et al. (2023). [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (LLaVA)
5. Liu et al. (2023). [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744) (LLaVA-1.5)
6. Alayrac et al. (2022). [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
7. Wang et al. (2022). [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100)
8. Radford et al. (2022). [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) (Whisper)
9. Gemini Team (2023). [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805)
10. Li et al. (2023). [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)

### Additional Resources

- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [OpenAI CLIP GitHub](https://github.com/openai/CLIP)
- [Whisper GitHub](https://github.com/openai/whisper)
- [Hugging Face Transformers Vision](https://huggingface.co/docs/transformers/modality_vision)

---

## Exercises

1. **Implement Patch Embedding**: Write a patch embedding layer that converts a 224x224 image into patches. Compare Conv2d approach vs. manual patch extraction.

2. **CLIP Zero-Shot Classification**: Implement zero-shot image classification using CLIP. Given an image and a list of class names, predict the most likely class.

3. **Vision-Language Alignment**: Train a simple projection layer to align CLIP image features with a small LM's embedding space. Measure alignment quality.

4. **Perceiver Resampler**: Implement a Perceiver Resampler and compare its efficiency to using all image patches as input to an LLM. Calculate memory savings.

5. **Multimodal Data Pipeline**: Create a data pipeline that processes image-text pairs for LLaVA-style training. Include data augmentation and proper formatting.

6. **Audio Mel Spectrogram**: Convert audio waveforms to mel spectrograms suitable for Whisper. Visualize the spectrograms and understand the parameters (n_fft, hop_length, n_mels).

7. **Cross-Modal Retrieval**: Implement image-text retrieval using CLIP. Given a text query, retrieve the most similar images from a dataset.

8. **Architecture Comparison**: Calculate the computational cost (FLOPs and memory) of:
   - LLaVA-style late fusion
   - Flamingo-style perceiver resampler
   - Processing all 196 ViT patches directly

   For a 7B parameter LLM with 2048 context length.

9. **Instruction Data Generation**: Write prompts to generate diverse visual instruction data from image captions. Compare quality with different prompt strategies.

10. **Multimodal Tokenization**: Design a tokenization scheme for a native multimodal model. How would you allocate vocabulary space for text, image, and audio tokens? What are the trade-offs?
