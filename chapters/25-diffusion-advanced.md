# Chapter 25: Advanced Diffusion Topics

This chapter explores advanced techniques in diffusion models, building on the fundamentals covered in previous chapters. We focus on key innovations that have made diffusion models practical for production use, including classifier-free guidance, latent diffusion, conditioning mechanisms, and recent advances like flow matching. We also cover discrete diffusion for language modeling.

For foundational diffusion concepts, see [Diffusion Model Fundamentals](23-diffusion-fundamentals.md) and [Implementing Diffusion Models](24-diffusion-implementation.md).

## Table of Contents

1. [Classifier-Free Guidance (CFG)](#classifier-free-guidance-cfg)
2. [Latent Diffusion Models](#latent-diffusion-models)
   - [Stable Diffusion Architecture](#stable-diffusion-architecture)
   - [VAE Encoder/Decoder](#vae-encoderdecoder)
3. [Conditioning Mechanisms](#conditioning-mechanisms)
   - [Text Conditioning](#text-conditioning)
   - [Cross-Attention for Conditioning](#cross-attention-for-conditioning)
   - [Image Conditioning](#image-conditioning)
4. [Recent Advances](#recent-advances)
   - [Flow Matching](#flow-matching)
   - [Rectified Flows](#rectified-flows)
   - [Consistency Models](#consistency-models)
5. [Diffusion for Language Models](#diffusion-for-language-models)
   - [Discrete Diffusion](#discrete-diffusion)
   - [Continuous Relaxations](#continuous-relaxations)
6. [Putting It All Together](#putting-it-all-together)

---

## Classifier-Free Guidance (CFG)

Classifier-Free Guidance (CFG) is a technique for steering diffusion models toward conditional generation without requiring a separate classifier. It has become the standard approach for conditional diffusion models.

### The Problem with Classifier Guidance

Original classifier guidance (Dhariwal & Nichol, 2021) required training a separate noise-robust classifier $p_\phi(y|x_t)$ to guide the diffusion process:

$$\nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + s \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$

where $s$ is the guidance scale.

**Problems:**
- Requires training a separate classifier on noisy images
- Classifier must be noise-robust across all timesteps
- Additional computational cost and complexity

### Classifier-Free Guidance Solution

Classifier-Free Guidance (Ho & Salimans, 2021) eliminates the need for a separate classifier by training a single conditional model that can perform both conditional and unconditional generation.

**Key Insight:** Train one model that learns both:
- Conditional distribution: $\epsilon_\theta(x_t, t, c)$ (with condition $c$)
- Unconditional distribution: $\epsilon_\theta(x_t, t, \emptyset)$ (without condition)

During sampling, interpolate between conditional and unconditional predictions:

$$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))$$

where:
- $s$ is the guidance scale (typically 7.5 for Stable Diffusion)
- $s = 0$: unconditional generation
- $s = 1$: standard conditional generation
- $s > 1$: stronger adherence to condition (at cost of diversity)

**Mathematical Intuition:**

The CFG formulation approximates:
$$\nabla_{x_t} \log p(x_t|c) \approx \nabla_{x_t} \log p(x_t) + s \cdot (\nabla_{x_t} \log p(x_t|c) - \nabla_{x_t} \log p(x_t))$$

This pushes the sample toward the conditional distribution while moving away from the unconditional distribution.

### Implementation

```python
import torch
import torch.nn as nn
from typing import Optional

class ClassifierFreeGuidanceMixin:
    """
    Mixin for classifier-free guidance in diffusion models.

    During training:
    - Randomly drop condition with probability p_uncond
    - Model learns both conditional and unconditional distributions

    During sampling:
    - Compute both conditional and unconditional predictions
    - Interpolate with guidance scale
    """

    def __init__(self, p_uncond: float = 0.1):
        """
        Args:
            p_uncond: Probability of unconditional training (typically 0.1)
        """
        self.p_uncond = p_uncond

    def apply_cfg_training(
        self,
        condition: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Apply dropout to condition during training.

        Args:
            condition: Conditioning tensor [batch, ...]
            training: Whether in training mode

        Returns:
            Condition with random samples set to null condition
        """
        if not training:
            return condition

        # Create mask for unconditional samples
        batch_size = condition.shape[0]
        mask = torch.rand(batch_size, device=condition.device) < self.p_uncond

        # Replace masked conditions with zeros (null condition)
        # In practice, you might use a learned null embedding
        masked_condition = condition.clone()
        masked_condition[mask] = 0

        return masked_condition

    def apply_cfg_sampling(
        self,
        noise_pred_fn,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        guidance_scale: float = 7.5,
        null_condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply classifier-free guidance during sampling.

        Args:
            noise_pred_fn: Function that predicts noise given (x_t, t, c)
            x_t: Noisy input at timestep t
            t: Current timestep
            condition: Conditioning tensor
            guidance_scale: Guidance strength (s in formula)
            null_condition: Explicit null condition (default: zeros)

        Returns:
            Guided noise prediction
        """
        if guidance_scale == 1.0:
            # Standard conditional generation, no guidance
            return noise_pred_fn(x_t, t, condition)

        # Prepare null condition
        if null_condition is None:
            null_condition = torch.zeros_like(condition)

        # Compute unconditional prediction
        noise_uncond = noise_pred_fn(x_t, t, null_condition)

        # Compute conditional prediction
        noise_cond = noise_pred_fn(x_t, t, condition)

        # Apply classifier-free guidance
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        return noise_pred


class ConditionalUNet(nn.Module):
    """
    Example U-Net with classifier-free guidance support.

    This is a simplified version - production models like Stable Diffusion
    use more sophisticated architectures with attention and cross-attention.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        cond_dim: int = 768,  # e.g., CLIP embedding dimension
        base_channels: int = 128,
        p_uncond: float = 0.1
    ):
        super().__init__()
        self.p_uncond = p_uncond

        # Simplified U-Net encoder
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)

        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, base_channels * 2)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, base_channels * 2)
        )

        # Simplified decoder
        self.dec1 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.dec2 = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with conditioning.

        Args:
            x: Input tensor [batch, channels, height, width]
            t: Timestep [batch]
            condition: Conditioning tensor [batch, cond_dim]
            return_dict: Whether to return dictionary (for compatibility)

        Returns:
            Noise prediction
        """
        # Apply CFG training (randomly drop condition)
        if self.training:
            condition = self._apply_cfg_training(condition)

        # Time embedding
        t_emb = self._get_timestep_embedding(t, self.enc1.out_channels)
        t_emb = self.time_mlp(t_emb)

        # Condition embedding
        c_emb = self.cond_proj(condition)

        # Combine time and condition
        emb = t_emb + c_emb

        # Simplified U-Net forward
        h = self.enc1(x)
        h = h + emb.unsqueeze(-1).unsqueeze(-1)
        h = self.enc2(h)
        h = self.dec1(h)
        out = self.dec2(h)

        return out

    def _apply_cfg_training(self, condition: torch.Tensor) -> torch.Tensor:
        """Apply unconditional training dropout."""
        batch_size = condition.shape[0]
        mask = torch.rand(batch_size, device=condition.device) < self.p_uncond
        masked_condition = condition.clone()
        masked_condition[mask] = 0
        return masked_condition

    def _get_timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


def cfg_sampling_example():
    """Example of classifier-free guidance sampling."""
    import torch.nn.functional as F

    # Setup
    model = ConditionalUNet()
    model.eval()

    # Initialize from noise
    x = torch.randn(1, 3, 64, 64)
    condition = torch.randn(1, 768)  # e.g., CLIP text embedding

    # Sampling parameters
    num_steps = 50
    guidance_scale = 7.5

    # DDPM sampling with CFG
    for i in reversed(range(num_steps)):
        t = torch.tensor([i])

        # Predict noise with CFG
        with torch.no_grad():
            # Unconditional prediction
            noise_uncond = model(x, t, torch.zeros_like(condition))

            # Conditional prediction
            noise_cond = model(x, t, condition)

            # Apply guidance
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Denoise step (simplified)
        alpha_t = 1 - i / num_steps
        x = (x - (1 - alpha_t) * noise_pred) / torch.sqrt(torch.tensor(alpha_t))

    return x
```

**Key Papers:**
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (Ho & Salimans, 2022)
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (Dhariwal & Nichol, 2021) - Original classifier guidance

---

## Latent Diffusion Models

Latent Diffusion Models (LDMs) perform diffusion in a compressed latent space rather than directly in pixel space. This is the key innovation behind Stable Diffusion.

### Motivation

**Pixel-Space Diffusion Problems:**
1. **Computational Cost**: High-resolution images require massive compute
   - 1024×1024 RGB image = 3.1M dimensions
   - Each denoising step processes all pixels
2. **Memory**: Storing intermediate states is expensive
3. **Redundancy**: Natural images have high redundancy

**Latent Diffusion Solution:**
- Train VAE to compress images to latent space (4-8× smaller)
- Run diffusion in latent space
- Decode final latent to pixel space

**Benefits:**
- 4-8× faster than pixel diffusion
- Same quality with much less compute
- Can train on consumer GPUs

### Stable Diffusion Architecture

Stable Diffusion consists of three main components:

1. **VAE (Variational Autoencoder)**: Compress images to/from latent space
2. **U-Net**: Diffusion model operating in latent space
3. **Text Encoder**: CLIP for text conditioning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model architecture (Stable Diffusion style).

    Components:
    1. VAE encoder: Image → Latent (compression)
    2. U-Net: Diffusion in latent space
    3. VAE decoder: Latent → Image (decompression)
    4. Text encoder: Text → Embeddings (conditioning)

    Latent space is typically 8× smaller per dimension:
    - 512×512 image → 64×64×4 latent
    - Compression ratio: 8×8×3/4 = 48×
    """

    def __init__(
        self,
        vae_encoder,
        vae_decoder,
        unet,
        text_encoder,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215
    ):
        super().__init__()
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.unet = unet
        self.text_encoder = text_encoder
        self.scaling_factor = scaling_factor

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            image: [batch, 3, height, width] in range [-1, 1]

        Returns:
            Latent: [batch, 4, height//8, width//8]
        """
        # VAE encoder outputs mean and log_variance
        posterior = self.vae_encoder(image)
        latent = posterior.sample()

        # Scale latent (important for stable training)
        latent = latent * self.scaling_factor

        return latent

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image space.

        Args:
            latent: [batch, 4, height//8, width//8]

        Returns:
            Image: [batch, 3, height, width]
        """
        # Unscale latent
        latent = latent / self.scaling_factor

        # VAE decoder
        image = self.vae_decoder(latent)

        return image

    @torch.no_grad()
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode text to conditioning embeddings.

        Args:
            text_tokens: [batch, seq_len] token IDs

        Returns:
            Text embeddings: [batch, seq_len, embed_dim]
        """
        return self.text_encoder(text_tokens)

    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in latent space.

        Args:
            latent: Noisy latent [batch, 4, h, w]
            t: Timestep [batch]
            text_embeddings: Text conditioning [batch, seq_len, dim]

        Returns:
            Predicted noise [batch, 4, h, w]
        """
        return self.unet(latent, t, text_embeddings)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate image from text prompt.

        This is a simplified version of Stable Diffusion's pipeline.
        """
        # 1. Encode text
        text_tokens = self._tokenize(prompt)
        text_embeddings = self.encode_text(text_tokens)

        # 2. Also get unconditional embeddings for CFG
        uncond_tokens = self._tokenize("")
        uncond_embeddings = self.encode_text(uncond_tokens)

        # 3. Initialize latent from noise
        latent_h, latent_w = height // 8, width // 8
        latent = torch.randn(
            (1, 4, latent_h, latent_w),
            generator=generator,
            device=text_embeddings.device
        )

        # 4. Denoising loop
        for t in self.scheduler.timesteps:
            # Expand latent for CFG (unconditional + conditional)
            latent_model_input = torch.cat([latent, latent] * 2)

            # Predict noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                torch.cat([uncond_embeddings, text_embeddings])
            )

            # Perform CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Denoise step
            latent = self.scheduler.step(noise_pred, t, latent)

        # 5. Decode latent to image
        image = self.decode_latent(latent)

        return image
```

### VAE Encoder/Decoder

The VAE compresses images to a lower-dimensional latent space while preserving perceptual quality.

```python
class VAEEncoder(nn.Module):
    """
    VAE Encoder for Latent Diffusion.

    Architecture (simplified):
    - Series of downsampling blocks (conv + downsample)
    - ResNet blocks at each resolution
    - Final conv to latent distribution parameters

    Stable Diffusion uses 4 downsampling stages:
    512×512 → 256×256 → 128×128 → 64×64 → 64×64
    (Last stage doesn't downsample spatially)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_multipliers: tuple = (1, 2, 4, 4)
    ):
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        channels = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult

            # ResNet block
            self.down_blocks.append(
                ResnetBlock(channels, out_channels)
            )

            # Downsample (except last layer)
            if i < len(channel_multipliers) - 1:
                self.down_blocks.append(Downsample(out_channels))

            channels = out_channels

        # Middle blocks
        self.mid_block1 = ResnetBlock(channels, channels)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResnetBlock(channels, channels)

        # Output projection to latent distribution
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, 2 * latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> 'DiagonalGaussianDistribution':
        """
        Encode image to latent distribution.

        Args:
            x: Image tensor [batch, 3, height, width]

        Returns:
            Posterior distribution over latents
        """
        # Initial conv
        h = self.conv_in(x)

        # Downsample
        for block in self.down_blocks:
            h = block(h)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # Split into mean and log_variance
        return DiagonalGaussianDistribution(h)


class VAEDecoder(nn.Module):
    """
    VAE Decoder for Latent Diffusion.

    Mirror of encoder with upsampling instead of downsampling.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple = (1, 2, 4, 4)
    ):
        super().__init__()

        # Input projection
        channels = base_channels * channel_multipliers[-1]
        self.conv_in = nn.Conv2d(latent_channels, channels, 3, padding=1)

        # Middle blocks
        self.mid_block1 = ResnetBlock(channels, channels)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResnetBlock(channels, channels)

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_multipliers)):
            out_channels = base_channels * mult

            # ResNet block
            self.up_blocks.append(
                ResnetBlock(channels, out_channels)
            )

            # Upsample (except last layer)
            if i < len(channel_multipliers) - 1:
                self.up_blocks.append(Upsample(out_channels))

            channels = out_channels

        # Output
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.

        Args:
            z: Latent tensor [batch, 4, height, width]

        Returns:
            Image [batch, 3, height*8, width*8]
        """
        # Input
        h = self.conv_in(z)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        # Upsample
        for block in self.up_blocks:
            h = block(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian distribution for VAE latent.

    The encoder outputs parameters of a Gaussian distribution.
    During training, we sample from this distribution (reparameterization trick).
    During inference, we can use the mean or sample.
    """

    def __init__(self, parameters: torch.Tensor):
        """
        Args:
            parameters: [batch, 2*latent_dim, height, width]
                       First half is mean, second half is log_variance
        """
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from the distribution using reparameterization trick."""
        # z = μ + σ * ε, where ε ~ N(0, 1)
        epsilon = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype
        )
        return self.mean + self.std * epsilon

    def mode(self) -> torch.Tensor:
        """Return the mode (mean) of the distribution."""
        return self.mean

    def kl(self, other: 'DiagonalGaussianDistribution' = None) -> torch.Tensor:
        """
        Compute KL divergence.

        If other is None, compute KL(self || N(0,1))
        """
        if other is None:
            # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (μ² + σ² - 1 - log(σ²))
            return 0.5 * torch.sum(
                self.mean ** 2 + self.std ** 2 - 1.0 - self.logvar,
                dim=[1, 2, 3]
            )
        else:
            # KL between two Gaussians
            return 0.5 * torch.sum(
                (self.mean - other.mean) ** 2 / other.std ** 2
                + self.std ** 2 / other.std ** 2
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3]
            )


# Helper modules
class ResnetBlock(nn.Module):
    """ResNet block with GroupNorm and skip connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    """Downsample by factor of 2 using strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by factor of 2 using nearest neighbor + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for spatial features."""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        # Normalize
        h = self.norm(x)

        # QKV
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.view(batch, self.num_heads, channels // self.num_heads, height * width)
        k = k.view(batch, self.num_heads, channels // self.num_heads, height * width)
        v = v.view(batch, self.num_heads, channels // self.num_heads, height * width)

        # Attention
        scale = (channels // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bhci,bhcj->bhij', q, k) * scale, dim=-1)
        h = torch.einsum('bhij,bhcj->bhci', attn, v)

        # Reshape back
        h = h.reshape(batch, channels, height, width)
        h = self.proj(h)

        return x + h
```

**Key Papers:**
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Rombach et al., 2022) - Stable Diffusion
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma & Welling, 2013) - VAE

---

## Conditioning Mechanisms

### Text Conditioning

Text conditioning allows generating images from text descriptions. The key is to encode text into a representation the diffusion model can use.

**CLIP Text Encoder:**

Stable Diffusion uses OpenAI's CLIP text encoder to convert text to embeddings:

```python
class CLIPTextEncoder(nn.Module):
    """
    CLIP text encoder for conditioning.

    CLIP (Contrastive Language-Image Pre-training) learns aligned
    text and image representations. We use only the text encoder.

    Architecture:
    - Token embedding
    - Positional embedding
    - Transformer encoder (12 layers for CLIP-ViT-L)
    - Final layer norm and projection

    Output: [batch, seq_len, 768] for ViT-L
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        max_position_embeddings: int = 77,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12
    ):
        super().__init__()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_position_embeddings, embed_dim)

        # Transformer encoder
        self.layers = nn.ModuleList([
            CLIPEncoderLayer(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text tokens.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Mask for padding [batch, seq_len]

        Returns:
            Text embeddings [batch, seq_len, embed_dim]
        """
        seq_len = input_ids.shape[1]

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.position_embedding(positions)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final norm
        x = self.final_layer_norm(x)

        return x


class CLIPEncoderLayer(nn.Module):
    """Single transformer layer for CLIP encoder."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x
```

### Cross-Attention for Conditioning

The U-Net uses cross-attention to incorporate text conditioning at each layer.

**Mathematical Formulation:**

Self-attention (within image features):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

where $Q, K, V$ all come from image features.

Cross-attention (image conditioned on text):
$$\text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

where:
- $Q$ comes from image features (queries)
- $K, V$ come from text embeddings (keys, values)

This allows each image feature to attend to relevant parts of the text.

```python
class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for conditioning U-Net on text.

    Architecture:
    1. Self-attention on image features
    2. Cross-attention from image to text
    3. Feed-forward network

    Each with residual connections and layer normalization.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64
    ):
        super().__init__()

        inner_dim = num_heads * head_dim

        # Self-attention (image features)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )

        # Cross-attention (image → text)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, num_heads, head_dim)

        # Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-attention conditioning.

        Args:
            x: Image features [batch, height*width, dim]
            context: Text embeddings [batch, seq_len, context_dim]

        Returns:
            Conditioned features [batch, height*width, dim]
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = residual + x

        # Cross-attention
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, context)
        x = residual + x

        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + x

        return x


class CrossAttention(nn.Module):
    """
    Cross-attention layer.

    Implements: Attention(Q, K, V) where Q comes from one sequence
    and K, V come from another sequence.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64
    ):
        super().__init__()

        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor [batch, seq_len_q, query_dim]
            context: Context tensor [batch, seq_len_c, context_dim]

        Returns:
            Output [batch, seq_len_q, query_dim]
        """
        batch_size = x.shape[0]

        # Project to Q, K, V
        q = self.to_q(x)  # [batch, seq_len_q, inner_dim]
        k = self.to_k(context)  # [batch, seq_len_c, inner_dim]
        v = self.to_v(context)  # [batch, seq_len_c, inner_dim]

        # Reshape for multi-head attention
        # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: [batch, num_heads, seq_len_q, seq_len_c]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch, num_heads, seq_len_q, head_dim]

        # Reshape back
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)

        # Output projection
        out = self.to_out(out)

        return out
```

### Image Conditioning

For tasks like inpainting, super-resolution, or image-to-image translation, we condition on input images.

**Common Approaches:**

1. **Concatenation**: Concatenate input image with noisy latent
2. **ControlNet**: Add parallel network for strong spatial conditioning
3. **Adapter**: Lightweight conditioning module

```python
class ImageConditionedUNet(nn.Module):
    """
    U-Net with image conditioning via concatenation.

    Used for:
    - Inpainting: Concatenate masked image + mask
    - Super-resolution: Concatenate low-resolution image
    - Image-to-image: Concatenate source image
    """

    def __init__(
        self,
        in_channels: int = 4,  # Latent channels
        cond_channels: int = 4,  # Conditioning image channels
        out_channels: int = 4,
        base_channels: int = 320
    ):
        super().__init__()

        # Input accepts both noisy latent and conditioning
        total_in_channels = in_channels + cond_channels

        self.conv_in = nn.Conv2d(total_in_channels, base_channels, 3, padding=1)

        # Rest of U-Net architecture...
        # (Similar to standard U-Net but with doubled input channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_image: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy latent [batch, 4, h, w]
            t: Timestep [batch]
            cond_image: Conditioning image latent [batch, 4, h, w]
            text_embeddings: Optional text conditioning

        Returns:
            Predicted noise [batch, 4, h, w]
        """
        # Concatenate noisy input with conditioning
        x = torch.cat([x, cond_image], dim=1)

        # Process through U-Net
        return self.unet_forward(x, t, text_embeddings)


class ControlNet(nn.Module):
    """
    ControlNet for strong spatial conditioning.

    Key idea: Create a trainable copy of the U-Net encoder that
    processes the conditioning signal. Add its outputs to the
    main U-Net at corresponding layers.

    Benefits:
    - Precise spatial control (e.g., pose, edges, depth)
    - Preserves pretrained model quality
    - Fast to train (only trains ControlNet, freezes main model)

    Reference: Zhang et al., "Adding Conditional Control to Text-to-Image
    Diffusion Models" (2023)
    """

    def __init__(
        self,
        base_unet: nn.Module,
        conditioning_channels: int = 3
    ):
        super().__init__()

        # Frozen pretrained model
        self.base_unet = base_unet
        for param in self.base_unet.parameters():
            param.requires_grad = False

        # Trainable copy of encoder (ControlNet)
        self.control_encoder = self._clone_encoder(base_unet)

        # Zero-initialized projections to add to U-Net
        # (Zero initialization ensures no impact at start of training)
        self.zero_convs = nn.ModuleList([
            self._make_zero_conv(channels)
            for channels in self._get_encoder_channels(base_unet)
        ])

        # Input processing for conditioning
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(conditioning_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy latent
            t: Timestep
            conditioning: Conditioning signal (e.g., Canny edges, pose)

        Returns:
            Predicted noise
        """
        # Process conditioning through ControlNet
        cond_features = self.input_hint_block(conditioning)
        control_outputs = self.control_encoder(x + cond_features, t)

        # Apply zero convolutions
        control_outputs = [
            zero_conv(feat) for zero_conv, feat in zip(self.zero_convs, control_outputs)
        ]

        # Run base U-Net with control additions
        return self.base_unet(x, t, control_features=control_outputs, **kwargs)

    @staticmethod
    def _make_zero_conv(channels: int) -> nn.Module:
        """Create zero-initialized 1x1 convolution."""
        conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv
```

**Key Papers:**
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021) - CLIP
- [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) (Zhang et al., 2023) - ControlNet

---

## Recent Advances

### Flow Matching

Flow Matching is an alternative to diffusion that learns to transform noise to data via continuous normalizing flows (CNFs).

**Key Differences from Diffusion:**

| Aspect | Diffusion | Flow Matching |
|--------|-----------|---------------|
| Forward process | Fixed (add noise) | Learned interpolation |
| Training objective | Denoising score matching | Regression to vector field |
| Sampling | DDPM/DDIM | ODE integration |
| Flexibility | Fixed noise schedule | Arbitrary paths |

**Mathematical Framework:**

Instead of a fixed diffusion process, Flow Matching learns a time-dependent vector field $v_t(x)$ such that:

$$\frac{dx_t}{dt} = v_t(x_t)$$

with $x_0 \sim p_\text{data}$ and $x_1 \sim p_\text{noise}$.

**Training Objective:**

$$\mathcal{L} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - u_t(x_t|x_0, x_1)\|^2\right]$$

where $u_t$ is the conditional vector field from $x_1$ to $x_0$.

**Simple Flow Matching:**

For straight-line paths: $x_t = t x_1 + (1-t) x_0$

The target vector field is simply: $u_t = x_1 - x_0$

```python
class FlowMatching(nn.Module):
    """
    Flow Matching for generative modeling.

    Key advantages over diffusion:
    1. Simpler training (direct regression, no noise schedules)
    2. Faster sampling (fewer ODE steps needed)
    3. More flexible (can use any interpolation path)

    Reference: Lipman et al., "Flow Matching for Generative Modeling" (2023)
    """

    def __init__(
        self,
        vector_field_model: nn.Module,
        sigma: float = 0.0  # Optional noise for conditioning
    ):
        super().__init__()
        self.model = vector_field_model
        self.sigma = sigma

    def get_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional flow and target vector field.

        Uses straight-line interpolation:
        x_t = t * x1 + (1 - t) * x0 + σ * noise

        Args:
            x0: Data samples [batch, ...]
            x1: Noise samples [batch, ...]
            t: Time in [0, 1]

        Returns:
            (x_t, target_velocity)
        """
        # Interpolate
        x_t = t * x1 + (1 - t) * x0

        # Add optional Gaussian conditioning
        if self.sigma > 0:
            noise = torch.randn_like(x0)
            x_t = x_t + self.sigma * noise

        # Target vector field (derivative of path)
        # For straight line: dx_t/dt = x1 - x0
        u_t = x1 - x0

        return x_t, u_t

    def training_step(
        self,
        x0: torch.Tensor,
        x1: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            x0: Real data samples
            x1: Noise samples (if None, sample from N(0,1))

        Returns:
            Loss value
        """
        if x1 is None:
            x1 = torch.randn_like(x0)

        # Sample random time
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=x0.device)

        # Get interpolated point and target
        x_t, u_t = self.get_conditional_flow(x0, x1, t.view(-1, 1, 1, 1))

        # Predict vector field
        v_pred = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_pred, u_t)

        return loss

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: int = 100,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Sample by integrating the learned vector field.

        Args:
            shape: Output shape
            num_steps: Number of integration steps
            method: 'euler' or 'rk4' (Runge-Kutta 4)

        Returns:
            Generated samples
        """
        # Start from noise
        x = torch.randn(shape, device=next(self.model.parameters()).device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = i / num_steps
            t_tensor = torch.full((shape[0],), t, device=x.device)

            if method == 'euler':
                # Euler integration: x_{t+dt} = x_t + dt * v_t
                v = self.model(x, t_tensor)
                x = x + dt * v

            elif method == 'rk4':
                # Runge-Kutta 4th order (more accurate, slower)
                k1 = self.model(x, t_tensor)
                k2 = self.model(x + 0.5 * dt * k1, t_tensor + 0.5 * dt)
                k3 = self.model(x + 0.5 * dt * k2, t_tensor + 0.5 * dt)
                k4 = self.model(x + dt * k3, t_tensor + dt)
                x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return x


class OptimalTransportConditionalFlowMatching(FlowMatching):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM).

    Uses optimal transport to find better interpolation paths
    between noise and data. Leads to straighter paths and
    faster sampling.

    Reference: Tong et al., "Improving and Generalizing Flow-Based
    Generative Models with Minibatch Optimal Transport" (2023)
    """

    def __init__(self, vector_field_model: nn.Module, sigma: float = 0.0):
        super().__init__(vector_field_model, sigma)

    def compute_ot_plan(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute optimal transport plan (simplified).

        In practice, use POT library or mini-batch OT.
        For mini-batches, can use simple bipartite matching.
        """
        # Simplified: Just match based on distance
        # Real implementation would use Sinkhorn or exact OT
        batch_size = x0.shape[0]

        # Compute pairwise distances
        x0_flat = x0.view(batch_size, -1)
        x1_flat = x1.view(batch_size, -1)

        cost = torch.cdist(x0_flat, x1_flat, p=2)

        # Find matching (greedy approximation)
        # Real OT would solve assignment problem properly
        indices = cost.argmin(dim=1)

        return indices

    def training_step(self, x0: torch.Tensor) -> torch.Tensor:
        """Training with OT matching."""
        # Sample noise
        x1 = torch.randn_like(x0)

        # Compute OT plan (matching)
        indices = self.compute_ot_plan(x0, x1)
        x1_matched = x1[indices]

        # Rest is same as standard flow matching
        return super().training_step(x0, x1_matched)
```

### Rectified Flows

Rectified Flows learn to "straighten" the probability flow, making sampling faster and more efficient.

**Key Idea:**

Standard diffusion/flow paths are curved. Rectified flows iteratively straighten these paths through "reflow" operations.

**Reflow Algorithm:**

1. Train initial flow matching model
2. Generate pairs $(x_0, x_1)$ by running the flow
3. Retrain on these pairs (which have straighter paths)
4. Repeat 2-3 until paths are nearly straight

**Benefits:**
- 1-step generation possible after rectification
- Better FID scores with fewer steps
- Simpler sampling (straight lines easier to integrate)

```python
class RectifiedFlow:
    """
    Rectified Flow - Straightening probability flows.

    Algorithm:
    1. Train flow matching model M_0
    2. For k = 1, 2, ...
       a. Sample x_0 ~ data, generate x_1 by running M_{k-1}
       b. Train M_k on straight paths from x_0 to x_1
    3. M_k has increasingly straight paths

    After enough reflows, can do 1-step generation!

    Reference: Liu et al., "Flow Straight and Fast: Learning to Generate
    and Transfer Data with Rectified Flow" (2023)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.reflow_iterations = 0

    @torch.no_grad()
    def generate_reflow_data(
        self,
        data_samples: torch.Tensor,
        num_steps: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate (x_0, x_1) pairs by running current flow.

        Args:
            data_samples: Real data x_0
            num_steps: Steps for ODE integration

        Returns:
            (x_0, x_1) pairs where x_1 is result of flowing x_0
        """
        x_0 = data_samples

        # Start from data, flow to noise
        x = x_0
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = i / num_steps
            t_tensor = torch.full((x.shape[0],), t, device=x.device)

            # Integrate backwards (data -> noise)
            v = self.model(x, t_tensor)
            x = x + dt * v

        x_1 = x

        return x_0, x_1

    def reflow_step(
        self,
        dataloader,
        num_epochs: int = 1
    ):
        """
        Perform one reflow iteration.

        1. Generate (x_0, x_1) pairs from current model
        2. Retrain on straight paths between them
        """
        print(f"Reflow iteration {self.reflow_iterations + 1}")

        # Generate reflow data
        all_x0 = []
        all_x1 = []

        for batch in dataloader:
            x_0, x_1 = self.generate_reflow_data(batch)
            all_x0.append(x_0)
            all_x1.append(x_1)

        # Retrain on straight paths
        flow_matcher = FlowMatching(self.model)

        for epoch in range(num_epochs):
            for x_0, x_1 in zip(all_x0, all_x1):
                loss = flow_matcher.training_step(x_0, x_1)
                # Optimization step...

        self.reflow_iterations += 1

    @torch.no_grad()
    def sample_one_step(self, noise: torch.Tensor) -> torch.Tensor:
        """
        One-step generation (only works well after rectification).

        After enough reflows, the path is nearly straight, so:
        x_0 ≈ x_1 + v(x_1, t=1)
        """
        t = torch.ones(noise.shape[0], device=noise.device)
        v = self.model(noise, t)
        return noise + v


def demonstrate_rectified_flow():
    """
    Demonstrate how path straightness improves with reflows.
    """
    import matplotlib.pyplot as plt

    # After 0 reflows: Curved paths, need 100 steps
    # After 1 reflow: Less curved, need 10 steps
    # After 2 reflows: Nearly straight, need 1-2 steps

    steps_needed = {
        'Initial': 100,
        'Reflow 1': 10,
        'Reflow 2': 2,
        'Reflow 3': 1
    }

    print("Sampling steps needed:")
    for name, steps in steps_needed.items():
        print(f"  {name}: {steps} steps")
```

### Consistency Models

Consistency Models learn to map any point on a diffusion trajectory directly to the origin (data).

**Key Insight:**

Standard diffusion requires many steps: $x_T \to x_{T-1} \to \cdots \to x_0$

Consistency models learn: $x_t \to x_0$ in one step (for any $t$).

**Consistency Property:**

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t') = x_0$$

for any $t, t'$ on the same trajectory.

```python
class ConsistencyModel(nn.Module):
    """
    Consistency Models for fast sampling.

    Key property: Maps any point on trajectory to origin
    f(x_t, t) = x_0 for all t

    Training:
    - Consistency Distillation: Distill from pretrained diffusion
    - Consistency Training: Train from scratch

    Inference:
    - 1-step: f(x_T, T) = x_0
    - Multi-step: Can still use multiple steps for quality

    Reference: Song et al., "Consistency Models" (2023)
    """

    def __init__(self, denoiser: nn.Module):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Map x_t to x_0.

        Architecture ensures f(x, 0) = x (boundary condition)
        """
        # Skip connection at t=0
        skip_weight = self._skip_scaling(t)
        out_weight = self._output_scaling(t)

        # F(x, t) = c_skip(t) * x + c_out(t) * D(x, t)
        return skip_weight * x_t + out_weight * self.denoiser(x_t, t)

    def _skip_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """Ensure f(x, 0) = x."""
        # Example: c_skip(0) = 1, c_skip(T) = 0
        return torch.exp(-t)

    def _output_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """Scaling for denoiser output."""
        # Example: c_out(0) = 0, c_out(T) = 1
        return 1 - torch.exp(-t)

    @torch.no_grad()
    def sample_one_step(self, noise: torch.Tensor) -> torch.Tensor:
        """One-step generation from noise."""
        T = torch.ones(noise.shape[0], device=noise.device)
        return self.forward(noise, T)

    @torch.no_grad()
    def sample_multi_step(
        self,
        noise: torch.Tensor,
        num_steps: int = 4
    ) -> torch.Tensor:
        """
        Multi-step sampling for higher quality.

        Use consistency model as iterative denoiser.
        """
        x = noise

        for i in range(num_steps):
            t = (num_steps - i) / num_steps
            t_tensor = torch.full((x.shape[0],), t, device=x.device)
            x = self.forward(x, t_tensor)

        return x
```

**Key Papers:**
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., 2023)
- [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482) (Tong et al., 2023)
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) (Liu et al., 2023)
- [Consistency Models](https://arxiv.org/abs/2303.01469) (Song et al., 2023)

---

## Diffusion for Language Models

### Discrete Diffusion

Applying diffusion to text is challenging because text is discrete (tokens), not continuous (pixels).

**Challenges:**
1. **Discrete space**: Can't add Gaussian noise to tokens
2. **Autoregressive tradition**: LLMs work well autoregressively
3. **Evaluation**: Harder to define quality metrics

**Approaches:**

1. **Discrete state space diffusion**: Corruption process in token space
2. **Continuous embedding diffusion**: Add noise to embeddings
3. **Score-based discrete diffusion**: Define scores over discrete distributions

```python
class DiscreteDiscreteDiffusion:
    """
    Discrete diffusion in token space.

    Forward process: Gradually corrupt tokens
    - Replace tokens with [MASK]
    - Replace with random tokens
    - Delete tokens

    Backward process: Predict original tokens

    Reference: Austin et al., "Structured Denoising Diffusion Models
    in Discrete State-Spaces" (2021)
    """

    def __init__(
        self,
        vocab_size: int,
        num_steps: int = 1000,
        mask_token_id: int = None
    ):
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.mask_token_id = mask_token_id or vocab_size  # Use special mask token

        # Define transition matrices Q_t
        # Q_t[i,j] = probability of token i transitioning to j at step t
        self.transition_matrices = self._build_transition_matrices()

    def _build_transition_matrices(self) -> list[torch.Tensor]:
        """
        Build transition matrices for discrete diffusion.

        Common choices:
        1. Uniform: All tokens equally likely
        2. Absorbing: Gradual transition to [MASK]
        3. Discretized Gaussian: Nearby tokens more likely
        """
        matrices = []

        for t in range(self.num_steps):
            # Absorbing state diffusion (→ [MASK])
            alpha_t = 1 - (t / self.num_steps)

            # Q_t[i,j] = alpha_t if i==j else (1-alpha_t)/(vocab_size)
            Q_t = torch.ones(self.vocab_size + 1, self.vocab_size + 1)
            Q_t *= (1 - alpha_t) / (self.vocab_size + 1)
            Q_t[range(self.vocab_size + 1), range(self.vocab_size + 1)] = alpha_t

            matrices.append(Q_t)

        return matrices

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        Apply forward diffusion: x_0 -> x_t

        Args:
            x0: Original tokens [batch, seq_len]
            t: Timestep

        Returns:
            Corrupted tokens x_t
        """
        Q_t = self.transition_matrices[t]

        # Sample x_t ~ Q_t(x_t | x_0)
        # For each token in x0, sample from Q_t[x0[i], :]
        x_t = []
        for i in range(x0.shape[0]):
            for j in range(x0.shape[1]):
                token = x0[i, j].item()
                probs = Q_t[token]
                sampled_token = torch.multinomial(probs, 1)
                x_t.append(sampled_token)

        x_t = torch.tensor(x_t).reshape(x0.shape)
        return x_t

    def reverse_diffusion_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """
        Reverse diffusion step: x_t -> x_{t-1}

        Model predicts p(x_0 | x_t), then we sample x_{t-1}
        """
        # Model predicts logits for x_0
        logits_x0 = model(x_t, t)  # [batch, seq_len, vocab_size]

        # Sample x_0
        probs_x0 = torch.softmax(logits_x0, dim=-1)

        # Compute p(x_{t-1} | x_t, x_0) using Bayes rule
        # This involves Q_t and Q_{t-1}
        # (Simplified - real implementation more complex)

        x_t_minus_1 = torch.multinomial(probs_x0.view(-1, self.vocab_size), 1)
        x_t_minus_1 = x_t_minus_1.reshape(x_t.shape)

        return x_t_minus_1


class ContinuousEmbeddingDiffusion(nn.Module):
    """
    Diffusion in continuous embedding space (Diffusion-LM approach).

    Key idea:
    1. Embed discrete tokens to continuous space
    2. Add Gaussian noise (standard diffusion)
    3. Denoise in embedding space
    4. Round to nearest token embeddings

    Advantage: Can use standard diffusion machinery
    Disadvantage: Rounding can cause issues

    Reference: Li et al., "Diffusion-LM Improves Controllable Text
    Generation" (2022)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        denoiser: nn.Module
    ):
        super().__init__()

        # Token embeddings
        self.embeddings = nn.Embedding(vocab_size, embed_dim)

        # Denoising model (operates on embeddings)
        self.denoiser = denoiser

        # Standard diffusion parameters
        self.num_steps = 1000
        self.betas = self._make_beta_schedule()
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _make_beta_schedule(self) -> torch.Tensor:
        """Linear beta schedule."""
        return torch.linspace(0.0001, 0.02, self.num_steps)

    def forward_diffusion(
        self,
        x0_embeddings: torch.Tensor,
        t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to embeddings.

        Args:
            x0_embeddings: Token embeddings [batch, seq_len, embed_dim]
            t: Timesteps [batch]

        Returns:
            (noisy_embeddings, noise)
        """
        noise = torch.randn_like(x0_embeddings)

        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = torch.sqrt(alpha_bar_t) * x0_embeddings + \
              torch.sqrt(1 - alpha_bar_t) * noise

        return x_t, noise

    def training_step(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Training step for embedding diffusion.

        Args:
            tokens: Token IDs [batch, seq_len]

        Returns:
            Loss value
        """
        # Embed tokens
        x0_embeddings = self.embeddings(tokens)

        # Sample timesteps
        t = torch.randint(0, self.num_steps, (tokens.shape[0],))

        # Add noise
        x_t, noise = self.forward_diffusion(x0_embeddings, t)

        # Predict noise
        noise_pred = self.denoiser(x_t, t)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int
    ) -> torch.Tensor:
        """
        Generate text by denoising embeddings.

        Returns:
            Token IDs [batch_size, seq_len]
        """
        # Start from noise
        x = torch.randn(batch_size, seq_len, self.embeddings.embedding_dim)

        # Denoise
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((batch_size,), t)

            # Predict noise
            noise_pred = self.denoiser(x, t_tensor)

            # DDPM update
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            x = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) / \
                torch.sqrt(alpha_t)

            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(self.betas[t])
                x = x + sigma_t * noise

        # Round to nearest embedding
        # Compute distances to all embeddings
        # (Simplified - real implementation uses more efficient search)
        distances = torch.cdist(
            x.view(-1, x.shape[-1]),
            self.embeddings.weight
        )
        tokens = distances.argmin(dim=-1).view(batch_size, seq_len)

        return tokens
```

### Continuous Relaxations

Some approaches use continuous relaxations of discrete distributions.

```python
class GumbelSoftmaxDiffusion:
    """
    Diffusion with Gumbel-Softmax relaxation.

    Key idea:
    - Represent discrete distribution as Gumbel-Softmax
    - Diffusion operates on logits
    - Can use reparameterization trick for gradients

    Gumbel-Softmax: Continuous relaxation of categorical distribution
    y_i = exp((log π_i + g_i) / τ) / Σ_j exp((log π_j + g_j) / τ)

    where g_i ~ Gumbel(0, 1) and τ is temperature.
    """

    def __init__(self, vocab_size: int, temperature: float = 1.0):
        self.vocab_size = vocab_size
        self.temperature = temperature

    def sample_gumbel(self, shape: tuple, device: str = 'cpu') -> torch.Tensor:
        """Sample from Gumbel(0, 1) distribution."""
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + 1e-20) + 1e-20)

    def gumbel_softmax_sample(
        self,
        logits: torch.Tensor,
        temperature: float = None
    ) -> torch.Tensor:
        """
        Sample from Gumbel-Softmax distribution.

        Args:
            logits: [batch, seq_len, vocab_size]
            temperature: Softmax temperature

        Returns:
            Soft samples [batch, seq_len, vocab_size]
        """
        if temperature is None:
            temperature = self.temperature

        gumbel_noise = self.sample_gumbel(logits.shape, logits.device)
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)

    def forward_diffusion(
        self,
        logits_0: torch.Tensor,
        t: int,
        num_steps: int = 1000
    ) -> torch.Tensor:
        """
        Diffuse logits by adding noise.

        Gradually corrupt toward uniform distribution.
        """
        # Noise schedule
        alpha_t = 1 - t / num_steps

        # Add noise to logits
        noise = torch.randn_like(logits_0)
        logits_t = torch.sqrt(torch.tensor(alpha_t)) * logits_0 + \
                   torch.sqrt(1 - alpha_t) * noise

        return logits_t
```

**Practical Note:**

As of 2024/2025, autoregressive models (GPT-style) still dominate for text generation. Diffusion for language is an active research area but not yet production-ready for most applications.

However, some recent work like **WeDLM** (see [Architecture Comparison: Modern LLMs](29-model-architectures.md)) shows promise by using causal attention in diffusion models, making them compatible with standard LLM infrastructure.

**Key Papers:**
- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) (Austin et al., 2021)
- [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) (Li et al., 2022)
- [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) (Jang et al., 2017)

---

## Putting It All Together

### Complete Stable Diffusion Training Pipeline

```python
class StableDiffusionTrainer:
    """
    Complete training pipeline for Latent Diffusion Model.

    Components:
    1. VAE (pretrained or train separately)
    2. U-Net with cross-attention
    3. Text encoder (CLIP, pretrained)
    4. Noise scheduler
    """

    def __init__(
        self,
        vae: nn.Module,
        unet: nn.Module,
        text_encoder: nn.Module,
        noise_scheduler,
        device: str = 'cuda'
    ):
        self.vae = vae.to(device).eval()
        self.unet = unet.to(device)
        self.text_encoder = text_encoder.to(device).eval()
        self.scheduler = noise_scheduler
        self.device = device

        # Freeze VAE and text encoder
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def training_step(
        self,
        images: torch.Tensor,
        captions: list[str],
        cfg_prob: float = 0.1
    ) -> torch.Tensor:
        """
        Single training step.

        Args:
            images: Batch of images [batch, 3, height, width]
            captions: List of text captions
            cfg_prob: Probability of unconditional training

        Returns:
            Loss value
        """
        batch_size = images.shape[0]

        # 1. Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode_image(images)

        # 2. Encode text
        text_tokens = self._tokenize_captions(captions)
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_tokens)

        # 3. Apply CFG dropout
        if cfg_prob > 0:
            mask = torch.rand(batch_size) < cfg_prob
            text_embeddings[mask] = 0  # Replace with null embedding

        # 4. Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps,
            (batch_size,), device=self.device
        )

        # 5. Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # 6. Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings)

        # 7. Compute loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> torch.Tensor:
        """
        Generate image from text prompt.

        Full Stable Diffusion pipeline with CFG.
        """
        # 1. Encode prompts
        text_tokens = self._tokenize_captions([prompt])
        text_embeddings = self.text_encoder(text_tokens)

        neg_tokens = self._tokenize_captions([negative_prompt])
        uncond_embeddings = self.text_encoder(neg_tokens)

        # 2. Initialize latent
        latent_h, latent_w = height // 8, width // 8
        latents = torch.randn(
            (1, 4, latent_h, latent_w),
            device=self.device
        )

        # 3. Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # 4. Denoising loop
        for t in self.scheduler.timesteps:
            # Expand for CFG
            latent_model_input = torch.cat([latents] * 2)

            # Predict noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                torch.cat([uncond_embeddings, text_embeddings])
            )

            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents)

        # 5. Decode latents
        image = self.vae.decode_latent(latents)

        # 6. Post-process
        image = (image + 1) / 2  # [-1, 1] -> [0, 1]
        image = torch.clamp(image, 0, 1)

        return image


def train_stable_diffusion_example():
    """
    Example training loop for Stable Diffusion.
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # Initialize components
    vae = VAEEncoder(...)  # Pretrained
    unet = LatentDiffusionUNet(...)
    text_encoder = CLIPTextEncoder(...)  # Pretrained
    scheduler = DDPMScheduler(...)

    trainer = StableDiffusionTrainer(vae, unet, text_encoder, scheduler)

    # Optimizer
    optimizer = optim.AdamW(unet.parameters(), lr=1e-4)

    # Training loop
    dataloader = DataLoader(...)  # Dataset of (image, caption) pairs

    for epoch in range(num_epochs):
        for images, captions in dataloader:
            # Training step
            loss = trainer.training_step(images, captions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

        # Generate samples
        if epoch % 10 == 0:
            sample = trainer.generate("A beautiful sunset over mountains")
            save_image(sample, f"sample_epoch_{epoch}.png")
```

---

## Summary

### Key Takeaways for Interviews

1. **Classifier-Free Guidance (CFG)**
   - Eliminates need for separate classifier
   - Train single model with dropout on conditions
   - Interpolate between conditional and unconditional predictions
   - Guidance scale controls adherence (typical: 7.5)

2. **Latent Diffusion / Stable Diffusion**
   - Run diffusion in compressed latent space (8× smaller)
   - VAE compresses images: 512×512 → 64×64×4
   - 4-8× faster than pixel-space diffusion
   - Three components: VAE, U-Net, Text Encoder

3. **Conditioning Mechanisms**
   - Text: CLIP encoder + cross-attention
   - Image: Concatenation or ControlNet
   - Cross-attention allows spatial conditioning

4. **Recent Advances**
   - Flow Matching: Simpler than diffusion, learns vector fields
   - Rectified Flows: Straighten paths for faster sampling
   - Consistency Models: One-step generation

5. **Diffusion for Language**
   - Still experimental vs. autoregressive
   - Discrete diffusion or continuous embeddings
   - WeDLM shows promise with causal attention

### Comparison Table

| Method | Training | Sampling Speed | Quality | Flexibility |
|--------|----------|----------------|---------|-------------|
| Standard Diffusion | Moderate | Slow (50-100 steps) | Excellent | High |
| Latent Diffusion | Moderate | Faster (20-50 steps) | Excellent | High |
| Flow Matching | Simpler | Fast (10-20 steps) | Excellent | Very High |
| Rectified Flow | Multiple stages | Very Fast (1-5 steps) | Excellent | High |
| Consistency Models | Complex | Fastest (1 step) | Good | Moderate |

### When to Use What

- **Stable Diffusion**: Production image generation
- **ControlNet**: Precise spatial control (pose, edges)
- **Flow Matching**: Research, new modalities
- **Rectified Flow**: When speed is critical
- **Consistency Models**: Real-time applications

---

## References

### Classifier-Free Guidance
1. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (Ho & Salimans, 2022)
2. [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (Dhariwal & Nichol, 2021)

### Latent Diffusion
3. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Rombach et al., 2022)
4. [Stable Diffusion GitHub](https://github.com/Stability-AI/stablediffusion)

### Conditioning
5. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (Radford et al., 2021) - CLIP
6. [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) (Zhang et al., 2023) - ControlNet

### Recent Advances
7. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., 2023)
8. [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482) (Tong et al., 2023)
9. [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) (Liu et al., 2023)
10. [Consistency Models](https://arxiv.org/abs/2303.01469) (Song et al., 2023)

### Diffusion for Language
11. [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) (Austin et al., 2021)
12. [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) (Li et al., 2022)
13. [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) (Jang et al., 2017)

---

## Exercises

1. **Implement CFG**: Modify a simple diffusion model to support classifier-free guidance. Train it on MNIST with digit labels as conditioning.

2. **VAE Compression**: Implement a simple VAE and measure the compression ratio and reconstruction quality. How does latent dimension affect quality vs. speed?

3. **Cross-Attention Analysis**: Visualize cross-attention maps from a text-to-image model. Which words attend to which image regions?

4. **Flow Matching vs. Diffusion**: Implement both flow matching and DDPM on a 2D toy dataset. Compare:
   - Training stability
   - Sampling speed
   - Path straightness

5. **Discrete Diffusion**: Implement discrete diffusion for a small vocabulary (e.g., DNA sequences with 4 tokens). Compare absorbing state vs. uniform transition.

6. **Rectified Flow**: Implement one reflow iteration. Visualize how the probability flow straightens. How many reflows are needed for 1-step generation?

7. **Conditioning Ablation**: Train a conditional diffusion model. Compare quality with:
   - No conditioning
   - Conditioning without CFG
   - Different guidance scales (1.0, 5.0, 10.0, 20.0)
