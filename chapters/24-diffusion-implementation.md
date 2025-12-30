# Chapter 24: Implementing Diffusion Models

This chapter provides a comprehensive, hands-on guide to implementing diffusion models with runnable PyTorch code. We build on the theoretical foundations from [Diffusion Model Fundamentals](23-diffusion-fundamentals.md) and implement all the core components needed for a working diffusion model.

## Table of Contents

1. [Overview](#overview)
2. [U-Net Architecture for Denoising](#u-net-architecture-for-denoising)
3. [Time Embedding](#time-embedding)
4. [Noise Scheduling](#noise-scheduling)
5. [Training Loop Implementation](#training-loop-implementation)
6. [Sampling Algorithms](#sampling-algorithms)
7. [Complete Working Example](#complete-working-example)
8. [Practical Considerations](#practical-considerations)
9. [Exercises](#exercises)
10. [References](#references)

---

## Overview

Diffusion models generate data by learning to reverse a gradual noising process. The key components we'll implement are:

1. **U-Net**: The neural network that predicts noise given a noisy image and timestep
2. **Time Embedding**: How we encode the diffusion timestep for the network
3. **Noise Schedule**: How we control noise levels across diffusion steps
4. **Training**: How we train the denoising network
5. **Sampling**: How we generate new samples using the trained model

The core training objective from [Diffusion Model Fundamentals](23-diffusion-fundamentals.md) is:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t) \|^2 \right]
$$

where:
- $\mathbf{x}_0$ is the original data
- $t$ is the timestep sampled uniformly from $\{1, \ldots, T\}$
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is random noise
- $\mathbf{x}_t$ is the noisy version at timestep $t$
- $\epsilon_\theta$ is our neural network

---

## U-Net Architecture for Denoising

The U-Net architecture is the most popular choice for diffusion models. It features:

- **Encoder-decoder structure** with skip connections
- **Downsampling** to capture coarse features
- **Upsampling** to reconstruct fine details
- **Skip connections** to preserve spatial information
- **Time conditioning** injected at multiple resolutions

### Architecture Diagram

```
Input (noisy image) + Time Embedding
    ↓
[Conv Block] ──────────────────────┐
    ↓ downsample                    │
[Conv Block] ──────────────┐       │
    ↓ downsample            │       │
[Conv Block] ──────┐       │       │
    ↓ downsample    │       │       │
[Bottleneck]        │       │       │
    ↓ upsample      │       │       │
[Conv Block] ←──────┘       │       │
    ↓ upsample              │       │
[Conv Block] ←──────────────┘       │
    ↓ upsample                      │
[Conv Block] ←──────────────────────┘
    ↓
Output (predicted noise)
```

### Basic Building Blocks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """Residual block with time embedding.

    Architecture:
        x -> [GroupNorm -> SiLU -> Conv3x3] -> [GroupNorm -> SiLU -> Conv3x3] -> out
        t -> [SiLU -> Linear] ───────────────────────────────────────────────────┘
        x ───────────────────────────────────────────────────────────────────> + out
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        # First convolution path
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Second convolution path
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W)
            t_emb: (batch, time_emb_dim)

        Returns:
            (batch, out_channels, H, W)
        """
        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding (broadcast over spatial dimensions)
        time_emb = self.time_mlp(t_emb)[:, :, None, None]  # (batch, out_channels, 1, 1)
        h = h + time_emb

        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for spatial features.

    Used in the bottleneck and optionally in decoder blocks
    to capture long-range dependencies.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W)

        Returns:
            (batch, channels, H, W)
        """
        B, C, H, W = x.shape

        # Normalize and compute Q, K, V
        h = self.norm(x)
        qkv = self.qkv(h)  # (B, 3*C, H, W)

        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention (see [Basic Attention](03-basic-attention.md))
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        h = attn @ v  # (B, num_heads, H*W, head_dim)

        # Reshape back
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        h = self.proj(h)

        return x + h  # Residual connection


class DownBlock(nn.Module):
    """Downsampling block with optional attention."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        downsample: bool = True,
        use_attention: bool = False
    ):
        super().__init__()

        self.resblocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            )
            for i in range(num_layers)
        ])

        self.attention = AttentionBlock(out_channels) if use_attention else None

        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1) \
            if downsample else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: Downsampled features
            skip: Features before downsampling (for skip connections)
        """
        for resblock in self.resblocks:
            x = resblock(x, t_emb)

        if self.attention is not None:
            x = self.attention(x)

        skip = x

        if self.downsample is not None:
            x = self.downsample(x)

        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        upsample: bool = True,
        use_attention: bool = False
    ):
        super().__init__()

        self.resblocks = nn.ModuleList([
            ResidualBlock(
                in_channels + out_channels if i == 0 else out_channels,  # +out_channels for skip
                out_channels,
                time_emb_dim
            )
            for i in range(num_layers)
        ])

        self.attention = AttentionBlock(out_channels) if use_attention else None

        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1) \
            if upsample else None

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Upsampled features from previous layer
            skip: Skip connection from corresponding down block
            t_emb: Time embedding
        """
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)

        for resblock in self.resblocks:
            x = resblock(x, t_emb)

        if self.attention is not None:
            x = self.attention(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x
```

### Complete U-Net

```python
class UNet(nn.Module):
    """U-Net architecture for diffusion models.

    References:
        - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
        - [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
    """
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        channel_mult: tuple = (1, 2, 2, 4),
        attention_resolutions: tuple = (2,),  # Which levels to use attention
        dropout: float = 0.0,
        time_emb_dim: int = 512
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        # Time embedding (see next section for details)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        channels = [model_channels]
        now_channels = model_channels

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(DownBlock(
                    now_channels,
                    out_ch,
                    time_emb_dim,
                    downsample=False,
                    use_attention=(level in attention_resolutions)
                ))
                now_channels = out_ch
                channels.append(now_channels)

            # Downsample at end of level (except last)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(DownBlock(
                    now_channels,
                    now_channels,
                    time_emb_dim,
                    downsample=True,
                    use_attention=False
                ))
                channels.append(now_channels)

        # Bottleneck
        self.mid_block1 = ResidualBlock(now_channels, now_channels, time_emb_dim)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResidualBlock(now_channels, now_channels, time_emb_dim)

        # Upsample blocks
        self.up_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(UpBlock(
                    now_channels,
                    out_ch,
                    time_emb_dim,
                    upsample=(i == num_res_blocks and level != 0),
                    use_attention=(level in attention_resolutions)
                ))
                now_channels = out_ch

        # Final output
        self.norm_out = nn.GroupNorm(32, now_channels)
        self.conv_out = nn.Conv2d(now_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, H, W) - noisy images
            t: (batch,) - timesteps

        Returns:
            (batch, out_channels, H, W) - predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial convolution
        x = self.conv_in(x)

        # Downsample
        skips = []
        for block in self.down_blocks:
            x, skip = block(x, t_emb)
            skips.append(skip)

        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Upsample
        for block in self.up_blocks:
            skip = skips.pop()
            x = block(x, skip, t_emb)

        # Final output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x
```

---

## Time Embedding

The diffusion timestep $t$ must be encoded so the network knows how much noise is present. We use sinusoidal position embeddings (similar to [Positional Encodings](07-positional-encodings.md)) because they:

1. **Generalize** to unseen timesteps
2. **Capture** periodic patterns in the denoising process
3. **Are continuous** which helps with interpolation

### Sinusoidal Time Embedding

```python
class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for time conditioning.

    Similar to positional encodings in Transformers (see [Positional Encodings](07-positional-encodings.md)),
    but for scalar timesteps instead of sequence positions.

    For timestep t:
        emb[2i] = sin(t / 10000^(2i/dim))
        emb[2i+1] = cos(t / 10000^(2i/dim))

    Reference:
        [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) - integer timesteps

        Returns:
            (batch, dim) - sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2

        # Compute frequencies: 1, 1/10000^(2/dim), 1/10000^(4/dim), ...
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Compute sinusoidal embeddings
        emb = t[:, None] * emb[None, :]  # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch, dim)

        return emb


# Alternative: Learned time embeddings
class LearnedTimeEmbedding(nn.Module):
    """Learned lookup table for time embeddings.

    Simpler but doesn't generalize to unseen timesteps.
    Useful for fixed, small number of timesteps.
    """
    def __init__(self, num_timesteps: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_timesteps, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.embedding(t)
```

### Why Sinusoidal Works

The sinusoidal encoding creates smooth, continuous representations where:
- Nearby timesteps have similar embeddings
- The network can interpolate between timesteps
- No learned parameters are needed

Mathematically, for timestep $t$ and embedding dimension $i$:

$$
\text{emb}_{t, 2i} = \sin\left(\frac{t}{10000^{2i/d}}\right)
$$

$$
\text{emb}_{t, 2i+1} = \cos\left(\frac{t}{10000^{2i/d}}\right)
$$

This creates a unique encoding for each timestep that the network learns to interpret.

---

## Noise Scheduling

The noise schedule $\{\beta_t\}_{t=1}^T$ controls how quickly noise is added during the forward process. This is one of the most important design choices in diffusion models.

### Key Concepts

From [Diffusion Model Fundamentals](23-diffusion-fundamentals.md), recall:
- $\alpha_t = 1 - \beta_t$ (how much signal to keep)
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ (cumulative signal retention)
- We can sample $\mathbf{x}_t$ directly: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

### Linear Schedule

The original DDPM paper used a linear schedule:

$$
\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})
$$

```python
def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """Linear noise schedule from DDPM.

    Reference:
        [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)

    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting noise level
        beta_end: Ending noise level

    Returns:
        betas: (timesteps,) noise levels
    """
    return torch.linspace(beta_start, beta_end, timesteps)
```

### Cosine Schedule

The cosine schedule from [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021) provides better results by:
- Adding noise more gradually at the start
- Preventing too much noise at the end
- Maintaining more signal throughout

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
$$

Then $\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$

```python
def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """Cosine noise schedule from Improved DDPM.

    Benefits over linear:
    - Smoother noise addition
    - Better preservation of structure early
    - Improved sample quality

    Reference:
        [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
        (Nichol & Dhariwal, 2021)

    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent beta_t = 0 at t=0

    Returns:
        betas: (timesteps,) noise levels
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)

    # Compute alpha_bar using cosine schedule
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Compute betas from alpha_bar
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    # Clip to prevent instabilities
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """Quadratic schedule - middle ground between linear and cosine."""
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """Sigmoid schedule - smooth transition."""
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
```

### Noise Schedule Helper Class

```python
class NoiseSchedule:
    """Manages noise scheduling and precomputes constants.

    Precomputes all schedule-dependent values for efficiency.
    """
    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        self.timesteps = timesteps

        # Get beta schedule
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Precompute constants
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For q(x_{t-1} | x_t, x_0) - used in DDPM sampling
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Sample from q(x_t | x_0) - forward diffusion.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_0: (batch, C, H, W) - original images
            t: (batch,) - timesteps
            noise: Optional pre-sampled noise

        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract schedule values for timestep t
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def to(self, device):
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self
```

### Visualizing Schedules

```python
import matplotlib.pyplot as plt

def plot_noise_schedules():
    """Visualize different noise schedules."""
    timesteps = 1000

    schedules = {
        'Linear': linear_beta_schedule(timesteps),
        'Cosine': cosine_beta_schedule(timesteps),
        'Quadratic': quadratic_beta_schedule(timesteps),
        'Sigmoid': sigmoid_beta_schedule(timesteps)
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, betas) in zip(axes.flat, schedules.items()):
        alphas_cumprod = torch.cumprod(1 - betas, dim=0)

        ax.plot(betas.numpy(), label=r'$\beta_t$', alpha=0.7)
        ax.plot(alphas_cumprod.numpy(), label=r'$\bar{\alpha}_t$', alpha=0.7)
        ax.set_title(f'{name} Schedule')
        ax.set_xlabel('Timestep')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('noise_schedules.png', dpi=150)
    plt.close()
```

---

## Training Loop Implementation

The training procedure is remarkably simple:

1. Sample a batch of real images $\mathbf{x}_0$
2. Sample random timesteps $t$
3. Sample noise $\epsilon$
4. Create noisy images $\mathbf{x}_t$
5. Predict the noise with our model
6. Compute MSE loss

```python
class DiffusionModel(nn.Module):
    """Complete diffusion model with training and sampling."""
    def __init__(
        self,
        unet: UNet,
        timesteps: int = 1000,
        schedule: str = "cosine"
    ):
        super().__init__()
        self.unet = unet
        self.noise_schedule = NoiseSchedule(timesteps, schedule)
        self.timesteps = timesteps

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """Training forward pass.

        Args:
            x_0: (batch, C, H, W) - real images

        Returns:
            loss: Scalar MSE loss
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps for each image in batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()

        # Sample noise
        noise = torch.randn_like(x_0)

        # Create noisy images
        x_t = self.noise_schedule.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.unet(x_t, t)

        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss


def train_diffusion_model(
    model: DiffusionModel,
    dataloader,
    num_epochs: int = 100,
    learning_rate: float = 2e-4,
    device: str = "cuda"
):
    """Training loop for diffusion model.

    Args:
        model: DiffusionModel instance
        dataloader: PyTorch DataLoader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for AdamW
        device: Device to train on
    """
    model = model.to(device)
    model.noise_schedule.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Cosine learning rate schedule with warmup
    # See [Scaling Laws and Optimization](17-scaling-optimization.md)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Forward pass
            loss = model(images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}] Average Loss: {avg_loss:.4f}")

        # Generate samples periodically
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                samples = sample_ddpm(model, num_samples=16, device=device)
                save_image_grid(samples, f"samples_epoch_{epoch}.png")
```

### Training Tips

1. **Batch Size**: Use the largest batch size that fits in memory (typically 128-256)
2. **Learning Rate**: Start with 2e-4, reduce if training is unstable
3. **Gradient Clipping**: Essential for stability, clip to norm of 1.0
4. **EMA**: Use exponential moving average of weights for better samples
5. **Mixed Precision**: Use torch.cuda.amp for faster training

```python
class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights that are more stable.
    Use EMA weights for sampling, not training weights.
    """
    def __init__(self, model, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)

    def apply_shadow(self):
        """Replace model parameters with EMA values."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
```

---

## Sampling Algorithms

Once trained, we generate samples by reversing the diffusion process. Two main algorithms:

1. **DDPM**: Stochastic sampling (high quality, slow)
2. **DDIM**: Deterministic sampling (faster, similar quality)

### DDPM Sampling

DDPM uses the full reverse process with learned variance:

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}
$$

where $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ and $\sigma_t = \sqrt{\beta_t}$ or $\sqrt{\tilde{\beta}_t}$

```python
@torch.no_grad()
def sample_ddpm(
    model: DiffusionModel,
    num_samples: int = 16,
    image_size: int = 32,
    channels: int = 3,
    device: str = "cuda"
) -> torch.Tensor:
    """Generate samples using DDPM sampling.

    Reference:
        [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
        (Ho et al., 2020)

    Args:
        model: Trained diffusion model
        num_samples: Number of images to generate
        image_size: Size of generated images
        channels: Number of channels
        device: Device to sample on

    Returns:
        (num_samples, channels, image_size, image_size) generated images
    """
    model.eval()

    # Start from pure noise
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)

    # Reverse diffusion process
    for t in reversed(range(model.timesteps)):
        # Create batch of timesteps
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model.unet(x, t_batch)

        # Extract schedule values
        alpha = model.noise_schedule.alphas[t]
        alpha_bar = model.noise_schedule.alphas_cumprod[t]
        beta = model.noise_schedule.betas[t]

        # Compute mean of q(x_{t-1} | x_t, x_0)
        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * epsilon)
        x = (1 / torch.sqrt(alpha)) * (
            x - ((beta) / torch.sqrt(1 - alpha_bar)) * predicted_noise
        )

        # Add noise (except at final step t=0)
        if t > 0:
            noise = torch.randn_like(x)
            # sigma_t = sqrt(beta_t)
            sigma = torch.sqrt(beta)
            x += sigma * noise

    return x


@torch.no_grad()
def sample_ddpm_with_variance(
    model: DiffusionModel,
    num_samples: int = 16,
    image_size: int = 32,
    channels: int = 3,
    device: str = "cuda"
) -> torch.Tensor:
    """DDPM sampling with learned/fixed posterior variance.

    Uses the posterior variance from the forward process:
    tilde_beta_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
    """
    model.eval()
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)

    for t in reversed(range(model.timesteps)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model.unet(x, t_batch)

        # Get precomputed coefficients
        posterior_mean_coef1 = model.noise_schedule.posterior_mean_coef1[t]
        posterior_mean_coef2 = model.noise_schedule.posterior_mean_coef2[t]
        posterior_variance = model.noise_schedule.posterior_variance[t]

        # Predict x_0 from x_t and predicted noise
        sqrt_recip_alpha_bar = 1 / model.noise_schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = model.noise_schedule.sqrt_one_minus_alphas_cumprod[t]
        pred_x_0 = sqrt_recip_alpha_bar * x - sqrt_one_minus_alpha_bar * predicted_noise

        # Clamp x_0 to valid range
        pred_x_0 = torch.clamp(pred_x_0, -1, 1)

        # Compute posterior mean
        mean = posterior_mean_coef1 * pred_x_0 + posterior_mean_coef2 * x

        # Add noise
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_variance) * noise
        else:
            x = mean

    return x
```

### DDIM Sampling

DDIM (Denoising Diffusion Implicit Models) enables:
- **Deterministic** sampling (reproducible results)
- **Faster** sampling (skip timesteps)
- **Interpolation** in latent space

The key insight is replacing the stochastic reverse process with a deterministic one:

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } \mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(\mathbf{x}_t, t) + \sigma_t \epsilon_t
$$

When $\sigma_t = 0$, this is fully deterministic.

```python
@torch.no_grad()
def sample_ddim(
    model: DiffusionModel,
    num_samples: int = 16,
    image_size: int = 32,
    channels: int = 3,
    ddim_steps: int = 50,
    eta: float = 0.0,  # 0 = deterministic, 1 = stochastic like DDPM
    device: str = "cuda"
) -> torch.Tensor:
    """Generate samples using DDIM sampling.

    DDIM allows:
    - Deterministic sampling (eta=0)
    - Faster sampling by skipping timesteps
    - Meaningful latent space interpolation

    Reference:
        [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
        (Song et al., 2020)

    Args:
        model: Trained diffusion model
        num_samples: Number of images to generate
        image_size: Size of generated images
        channels: Number of channels
        ddim_steps: Number of sampling steps (< model.timesteps for speedup)
        eta: Stochasticity parameter (0=deterministic, 1=stochastic)
        device: Device to sample on

    Returns:
        Generated images
    """
    model.eval()

    # Create subsequence of timesteps
    # E.g., if timesteps=1000 and ddim_steps=50, use [0, 20, 40, ..., 980]
    skip = model.timesteps // ddim_steps
    timesteps = list(range(0, model.timesteps, skip))
    timesteps = list(reversed(timesteps))

    # Start from noise
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # Predict noise
        predicted_noise = model.unet(x, t_batch)

        # Get alpha values
        alpha_bar_t = model.noise_schedule.alphas_cumprod[t]

        # Get alpha for previous timestep
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_bar_t_prev = model.noise_schedule.alphas_cumprod[t_prev]
        else:
            alpha_bar_t_prev = torch.tensor(1.0)

        # Predict x_0
        pred_x_0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        pred_x_0 = torch.clamp(pred_x_0, -1, 1)

        # Compute variance
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
        )

        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * predicted_noise

        # Compute x_{t-1}
        x = torch.sqrt(alpha_bar_t_prev) * pred_x_0 + dir_xt

        # Add noise
        if sigma_t > 0 and i < len(timesteps) - 1:
            noise = torch.randn_like(x)
            x += sigma_t * noise

    return x
```

### Comparison: DDPM vs DDIM

| Property | DDPM | DDIM |
|----------|------|------|
| **Sampling** | Stochastic | Deterministic (eta=0) |
| **Speed** | Slow (needs all T steps) | Fast (can skip steps) |
| **Quality** | High | Similar |
| **Interpolation** | Not meaningful | Meaningful |
| **Use Case** | Best quality | Production, fast sampling |

---

## Complete Working Example

Let's put it all together with a minimal example on MNIST:

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Smaller U-Net for MNIST (28x28 grayscale)
def create_mnist_unet():
    """Create a smaller U-Net suitable for MNIST."""
    return UNet(
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        attention_resolutions=(1,),
        time_emb_dim=256
    )

# Training on MNIST
def train_mnist_diffusion():
    """Complete training example on MNIST."""

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    num_epochs = 50

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # Model
    unet = create_mnist_unet()
    model = DiffusionModel(unet, timesteps=1000, schedule="cosine")
    model = model.to(device)
    model.noise_schedule.to(device)

    # EMA
    ema = EMA(model, decay=0.9999)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Forward
            loss = model(images)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update EMA
            ema.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        # Generate samples with EMA model
        if (epoch + 1) % 5 == 0:
            ema.apply_shadow()
            model.eval()

            with torch.no_grad():
                # DDPM sampling
                samples_ddpm = sample_ddpm(
                    model,
                    num_samples=64,
                    image_size=28,
                    channels=1,
                    device=device
                )

                # DDIM sampling (faster)
                samples_ddim = sample_ddim(
                    model,
                    num_samples=64,
                    image_size=28,
                    channels=1,
                    ddim_steps=50,
                    device=device
                )

            # Save samples
            save_image_grid(samples_ddpm, f"ddpm_epoch_{epoch}.png", nrow=8)
            save_image_grid(samples_ddim, f"ddim_epoch_{epoch}.png", nrow=8)

            ema.restore()

    return model


def save_image_grid(images, filename, nrow=8):
    """Save a grid of images."""
    from torchvision.utils import save_image

    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    save_image(images, filename, nrow=nrow)


# Run training
if __name__ == "__main__":
    model = train_mnist_diffusion()
```

---

## Practical Considerations

### Memory Optimization

For high-resolution images, memory becomes a bottleneck:

```python
# 1. Gradient checkpointing
class UNetCheckpointed(UNet):
    """U-Net with gradient checkpointing for memory efficiency."""
    def forward(self, x, t):
        from torch.utils.checkpoint import checkpoint

        # Checkpoint expensive blocks
        def custom_forward(module):
            def forward(*inputs):
                return module(*inputs)
            return forward

        # Use checkpointing for down/up blocks
        # Trades compute for memory
        # See [Distributed Training and Parallelism](16-distributed-training.md)
        pass


# 2. Mixed precision training
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, dataloader, optimizer):
    """Training with automatic mixed precision."""
    scaler = GradScaler()

    for images, _ in dataloader:
        optimizer.zero_grad()

        # Forward in mixed precision
        with autocast():
            loss = model(images)

        # Backward with scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
```

### Inference Optimization

```python
# 1. Compile model (PyTorch 2.0+)
model = torch.compile(model)

# 2. Use DDIM with fewer steps
samples = sample_ddim(model, ddim_steps=20)  # 50x faster than DDPM with 1000 steps

# 3. Batch generation
def generate_batch(model, batch_size=64):
    """Generate multiple samples efficiently."""
    return sample_ddim(model, num_samples=batch_size)
```

### Common Issues and Solutions

**Issue 1: Mode Collapse**
- Symptoms: Model generates similar images
- Solutions:
  - Use cosine schedule instead of linear
  - Increase model capacity
  - Train longer

**Issue 2: Blurry Samples**
- Symptoms: Samples lack fine details
- Solutions:
  - Use more timesteps during training
  - Improve noise schedule
  - Add attention at multiple resolutions

**Issue 3: Training Instability**
- Symptoms: Loss spikes, NaN values
- Solutions:
  - Reduce learning rate
  - Use gradient clipping
  - Check data normalization
  - Use EMA

**Issue 4: Slow Sampling**
- Symptoms: Takes too long to generate
- Solutions:
  - Use DDIM instead of DDPM
  - Reduce ddim_steps (25-50 is usually enough)
  - Consider distillation (see [Model Merging and Distillation](30-merging-distillation.md))

---

## Exercises

### Exercise 1: Implement v-prediction

Instead of predicting noise $\epsilon$, predict the "velocity" $v$:

$$
v_t = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1 - \bar{\alpha}_t} \mathbf{x}_0
$$

Modify the training loop and sampling to use v-prediction. This is used in Stable Diffusion 2.0+.

### Exercise 2: Progressive Distillation

Implement progressive distillation to reduce sampling steps:
1. Train a student model to predict 2 teacher steps in 1 step
2. Repeat to get 4x, 8x, 16x speedup

Reference: [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)

### Exercise 3: Classifier-Free Guidance

Implement classifier-free guidance for conditional generation:

$$
\tilde{\epsilon}_\theta(\mathbf{x}_t, c) = \epsilon_\theta(\mathbf{x}_t, \emptyset) + s \cdot (\epsilon_\theta(\mathbf{x}_t, c) - \epsilon_\theta(\mathbf{x}_t, \emptyset))
$$

where $c$ is a condition (e.g., class label) and $s$ is guidance scale.

See [Advanced Diffusion Topics](25-diffusion-advanced.md) for details.

### Exercise 4: FID Evaluation

Implement FID (Fréchet Inception Distance) to quantitatively evaluate sample quality:

1. Generate 10k samples from your model
2. Extract Inception-v3 features
3. Compute FID between generated and real distributions

### Exercise 5: Latent Diffusion

Implement a simple latent diffusion model:
1. Train an autoencoder (VAE or VQ-VAE)
2. Train diffusion model in latent space
3. Decode samples to pixel space

This is the foundation of Stable Diffusion.

### Exercise 6: Noise Schedule Ablation

Train models with different noise schedules and compare:
- Sample quality (visual inspection)
- FID scores
- Training convergence speed

Which schedule works best for your dataset?

---

## References

### Core Papers

1. **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
   - Original diffusion model formulation
   - Simplified training objective
   - U-Net architecture

2. **Improved DDPM**: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
   - Cosine noise schedule
   - Learned variance
   - Improved sample quality

3. **DDIM**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2020)
   - Deterministic sampling
   - Accelerated generation
   - Latent space interpolation

4. **U-Net**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015)
   - Encoder-decoder with skip connections
   - Foundation for diffusion architectures

### Advanced Topics

5. **Score-Based Models**: [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) (Song et al., 2021)
   - Continuous-time diffusion
   - SDE formulation
   - Connection to score matching

6. **Classifier-Free Guidance**: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (Ho & Salimans, 2022)
   - Conditional generation without classifier
   - Guidance scale tuning

7. **Progressive Distillation**: [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) (Salimans & Ho, 2022)
   - Iterative distillation
   - 2-4 step sampling

8. **Latent Diffusion**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Rombach et al., 2022)
   - Stable Diffusion
   - Diffusion in latent space
   - Cross-attention conditioning

### Implementation Resources

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - Production-ready implementations
- [Phil Wang's Denoising Diffusion PyTorch](https://github.com/lucidrains/denoising-diffusion-pytorch) - Clean, educational implementations
- [OpenAI's Improved DDPM](https://github.com/openai/improved-diffusion) - Official implementation

### Related Chapters

- [Diffusion Model Fundamentals](23-diffusion-fundamentals.md) - Theoretical foundations
- [Advanced Diffusion Topics](25-diffusion-advanced.md) - Classifier-free guidance, latent diffusion
- [Distributed Training and Parallelism](16-distributed-training.md) - Scaling to larger models
- [Hardware, Quantization, and Training Optimization](31-hardware-quantization-optimization.md) - Optimization techniques

---

## Summary

In this chapter, we implemented a complete diffusion model from scratch:

1. **U-Net Architecture**: Encoder-decoder with skip connections and time conditioning
2. **Time Embedding**: Sinusoidal embeddings for continuous timestep representation
3. **Noise Scheduling**: Linear, cosine, and other schedules for controlling diffusion
4. **Training**: Simple MSE loss between predicted and actual noise
5. **Sampling**: DDPM (slow, high quality) and DDIM (fast, deterministic)

Key takeaways:
- Training is simple: predict the noise added to an image
- Sampling is iterative: gradually denoise from pure noise
- U-Net with skip connections preserves spatial information
- DDIM enables fast sampling by skipping timesteps
- EMA and proper schedules are crucial for quality

Next, explore [Advanced Diffusion Topics](25-diffusion-advanced.md) for classifier-free guidance, latent diffusion, and recent advances.
