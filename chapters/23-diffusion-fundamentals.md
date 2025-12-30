# Chapter 23: Diffusion Model Fundamentals

## Introduction

Diffusion models have emerged as one of the most powerful classes of generative models, achieving state-of-the-art results in image generation, audio synthesis, and even text generation. Unlike autoregressive models (like GPT) or variational autoencoders (VAEs), diffusion models learn to generate data by reversing a gradual noising process.

This chapter covers the mathematical foundations and core concepts behind diffusion models, preparing you for implementation in [Chapter 24: Implementing Diffusion Models](24-diffusion-implementation.md) and advanced techniques in [Chapter 25: Advanced Diffusion Topics](25-diffusion-advanced.md).

**Key Papers:**
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) - Song et al., 2021
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) - Nichol & Dhariwal, 2021
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) - Song & Ermon, 2019

## Overview of Generative Models

Before diving into diffusion models, let's briefly compare them to other generative model families:

| Model Type | Training Objective | Sampling Process | Likelihood |
|------------|-------------------|------------------|------------|
| **VAEs** | ELBO maximization | Single forward pass | Tractable (approximate) |
| **GANs** | Adversarial min-max | Single forward pass | Intractable |
| **Autoregressive** | Log-likelihood | Sequential generation | Tractable |
| **Normalizing Flows** | Exact likelihood | Single invertible pass | Tractable |
| **Diffusion Models** | Denoising objective | Iterative refinement | Tractable |

Diffusion models offer several advantages:
- **Stable training**: Unlike GANs, no adversarial dynamics
- **High-quality samples**: State-of-the-art image generation
- **Flexible architectures**: Can use various neural network backbones
- **Principled framework**: Strong theoretical foundations

## The Diffusion Process: Intuition

The core idea behind diffusion models is beautifully simple:

1. **Forward Process**: Gradually add Gaussian noise to data until it becomes pure noise
2. **Reverse Process**: Learn to reverse this process, starting from noise and recovering data

This is analogous to:
- Watching a drop of ink diffuse in water (forward)
- Learning to reverse time and reconstitute the original drop (reverse)

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Simple visualization of the diffusion process
def visualize_diffusion_process(x0, num_steps=10):
    """
    Visualize how an image gets progressively noisier

    Args:
        x0: Original image tensor [C, H, W]
        num_steps: Number of diffusion steps to show
    """
    # Define variance schedule (we'll explain this later)
    betas = torch.linspace(0.0001, 0.02, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    images = [x0]
    x_t = x0

    for t in range(num_steps):
        # Add noise according to the schedule
        noise = torch.randn_like(x_t)
        alpha_t = alphas_cumprod[t]
        x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        images.append(x_t)

    return images

# Example usage (requires actual image data)
# x0 = load_image()  # Shape: [C, H, W]
# noisy_images = visualize_diffusion_process(x0)
```

## Forward Diffusion Process

### Mathematical Formulation

The forward diffusion process is a **fixed** Markov chain that gradually adds Gaussian noise to data $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ over $T$ timesteps:

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

where:
- $\beta_t \in (0, 1)$ is the **variance schedule** (controls how much noise to add at step $t$)
- $t \in \{1, ..., T\}$ is the timestep
- $\mathbf{x}_0$ is the original data
- $\mathbf{x}_T$ is approximately pure Gaussian noise

### Key Property: Closed-Form Sampling

A crucial property of this process is that we can sample $\mathbf{x}_t$ at any timestep $t$ directly from $\mathbf{x}_0$ without iterating through all previous steps. Define:

$$
\alpha_t := 1 - \beta_t, \quad \bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s
$$

Then:

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

This can be rewritten using the **reparameterization trick**:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

**Proof sketch:**
Starting from $q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} \mathbf{x}_{t-1}, (1-\alpha_t)\mathbf{I})$, we can use the property that the sum of Gaussians is Gaussian to recursively derive this formula.

### Implementation

```python
class ForwardDiffusion:
    """
    Implements the forward diffusion process q(x_t | x_0)
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Args:
            num_timesteps: Total number of diffusion steps T
            beta_start: Initial variance (β_1)
            beta_end: Final variance (β_T)
        """
        self.num_timesteps = num_timesteps

        # Define beta schedule (linear schedule)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) using the closed-form formula

        Args:
            x_0: Original data [batch_size, ...]
            t: Timestep tensor [batch_size] with values in [0, T-1]
            noise: Optional pre-sampled noise (for reproducibility)

        Returns:
            x_t: Noised data at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract the appropriate values for timestep t
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # Apply the forward diffusion formula
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

    def _extract(self, a, t, x_shape):
        """
        Extract values from a at indices t and reshape for broadcasting

        Args:
            a: Tensor to extract from [T]
            t: Timestep indices [batch_size]
            x_shape: Shape of data for broadcasting

        Returns:
            Extracted values reshaped to [batch_size, 1, 1, ...]
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# Example usage
forward_diffusion = ForwardDiffusion(num_timesteps=1000)

# Create sample data
x_0 = torch.randn(4, 3, 32, 32)  # Batch of 4 images

# Sample at different timesteps
t = torch.tensor([0, 250, 500, 999])  # Different noise levels
x_t, noise = forward_diffusion.q_sample(x_0, t)

print(f"Original data shape: {x_0.shape}")
print(f"Noised data shape: {x_t.shape}")
print(f"Noise shape: {noise.shape}")
```

### Variance Schedules

The choice of $\beta_t$ (variance schedule) significantly impacts training and sampling. Common schedules include:

1. **Linear Schedule** (DDPM):
   $$\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$$

2. **Cosine Schedule** (Improved DDPM):
   $$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

3. **Quadratic Schedule**:
   $$\beta_t = \beta_1 + \left(\frac{t-1}{T-1}\right)^2 (\beta_T - \beta_1)$$

```python
def get_beta_schedule(schedule_name, num_timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Generate various beta schedules

    Args:
        schedule_name: One of 'linear', 'cosine', 'quadratic'
        num_timesteps: Number of diffusion steps
        beta_start: Starting beta value (for linear/quadratic)
        beta_end: Ending beta value (for linear/quadratic)

    Returns:
        betas: Tensor of shape [num_timesteps]
    """
    if schedule_name == 'linear':
        return torch.linspace(beta_start, beta_end, num_timesteps)

    elif schedule_name == 'cosine':
        # Cosine schedule from Improved DDPM
        steps = num_timesteps + 1
        s = 0.008  # Offset to prevent beta_t from being too small near t=0
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    elif schedule_name == 'quadratic':
        t = torch.linspace(0, 1, num_timesteps)
        return beta_start + (beta_end - beta_start) * t ** 2

    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

# Visualize different schedules
import matplotlib.pyplot as plt

schedules = ['linear', 'cosine', 'quadratic']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for schedule, ax in zip(schedules, axes):
    betas = get_beta_schedule(schedule, 1000)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)

    ax.plot(betas.numpy(), label='beta_t', alpha=0.7)
    ax.plot(alphas_cumprod.numpy(), label='alpha_bar_t', alpha=0.7)
    ax.set_title(f'{schedule.capitalize()} Schedule')
    ax.set_xlabel('Timestep t')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('variance_schedules.png')
```

## Reverse Diffusion Process

### Learning to Denoise

The reverse process aims to **undo** the forward diffusion, transforming noise back into data. We want to learn:

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
$$

Starting from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, we iteratively sample:

$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
$$

### Reverse Process is Gaussian (with small $\beta_t$)

A key insight: if $\beta_t$ is small, the reverse process $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$ is also Gaussian (when conditioned on $\mathbf{x}_0$):

$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
$$

where:

$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t
$$

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

### Parameterization Choices

We can parameterize $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ in several ways:

1. **Predict $\mathbf{x}_0$ directly**: $\mathbf{x}_\theta(\mathbf{x}_t, t)$
2. **Predict the noise $\boldsymbol{\epsilon}$**: $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ (DDPM approach)
3. **Predict the mean directly**: $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$
4. **Predict the score**: $\mathbf{s}_\theta(\mathbf{x}_t, t) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$ (Score-based models)

The DDPM paper showed that **predicting the noise $\boldsymbol{\epsilon}$** works best. Given $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$, we can compute:

$$
\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)
$$

Then plug this into $\tilde{\boldsymbol{\mu}}_t$ to get the mean of the reverse distribution.

## Score Matching and Score Functions

### What is a Score Function?

The **score function** is the gradient of the log-probability density:

$$
\mathbf{s}(\mathbf{x}) = \nabla_\mathbf{x} \log p(\mathbf{x})
$$

The score function points in the direction of increasing probability density. If we know the score at any point, we can:
1. Move toward higher-density regions (sampling)
2. Estimate the underlying distribution

### Connection to Denoising

There's a deep connection between denoising and score matching. The optimal denoiser for Gaussian noise is:

$$
\mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t] = \mathbf{x}_t + (1 - \bar{\alpha}_t) \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)
$$

This means learning to denoise is equivalent to learning the score function!

Since $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$, we have:

$$
\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}
$$

Therefore, predicting the noise $\boldsymbol{\epsilon}$ is equivalent to predicting the score (up to scaling).

### Score-Based Generative Models

Score-based models (Song et al.) directly learn the score function $\mathbf{s}_\theta(\mathbf{x}, \sigma)$ at different noise levels $\sigma$. They use **score matching** objectives like:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{p(\mathbf{x})} \mathbb{E}_{p(\tilde{\mathbf{x}}|\mathbf{x})} \left[\left\| \mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma) - \nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}} | \mathbf{x}) \right\|^2 \right]
$$

where $p(\tilde{\mathbf{x}} | \mathbf{x}) = \mathcal{N}(\mathbf{x}, \sigma^2 \mathbf{I})$.

This is closely related to DDPM's formulation, and both frameworks can be unified under the **Stochastic Differential Equation (SDE)** perspective.

```python
class ScoreNetwork(nn.Module):
    """
    Simple score network that predicts the score function
    (gradient of log probability)
    """
    def __init__(self, data_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),  # +1 for noise level
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, sigma):
        """
        Args:
            x: Data [batch_size, data_dim]
            sigma: Noise level [batch_size, 1]

        Returns:
            score: Predicted score [batch_size, data_dim]
        """
        # Concatenate data with noise level
        inp = torch.cat([x, sigma], dim=-1)
        return self.net(inp)

# Denoising score matching loss
def dsm_loss(score_net, x, sigma_min=0.01, sigma_max=1.0):
    """
    Denoising score matching loss

    Args:
        score_net: Score network
        x: Clean data [batch_size, data_dim]
        sigma_min, sigma_max: Range of noise levels

    Returns:
        loss: DSM loss
    """
    # Sample noise level
    sigma = torch.exp(torch.rand(x.shape[0], 1) * (np.log(sigma_max) - np.log(sigma_min)) + np.log(sigma_min))
    sigma = sigma.to(x.device)

    # Add noise
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise

    # Predict score
    score = score_net(x_noisy, sigma)

    # True score for Gaussian perturbation: -noise / sigma
    target_score = -noise / sigma

    # MSE between predicted and true score
    loss = torch.mean((score - target_score) ** 2)

    return loss
```

## DDPM Formulation

### Training Objective

The DDPM training objective is derived from the **variational lower bound (VLB)** on the negative log-likelihood:

$$
\mathcal{L}_{\text{VLB}} = \mathbb{E}_q \left[ \underbrace{D_{KL}(q(\mathbf{x}_T | \mathbf{x}_0) \| p(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{KL}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 | \mathbf{x}_1)}_{L_0} \right]
$$

However, Ho et al. showed that a **simplified objective** works better in practice:

$$
\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right\|^2 \right]
$$

where:
- $t \sim \text{Uniform}(\{1, ..., T\})$
- $\mathbf{x}_0 \sim q(\mathbf{x}_0)$
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$

This is simply **mean squared error** between the true noise and predicted noise!

### Why Does the Simplified Objective Work?

The simplified objective can be seen as a weighted version of the VLB where certain terms are reweighted. Empirically, it leads to:
- Better sample quality
- Faster training
- Simpler implementation

The connection to score matching also provides theoretical justification: we're learning the score function at different noise levels.

### DDPM Algorithm

**Training:**
1. Sample $\mathbf{x}_0 \sim q(\mathbf{x}_0)$
2. Sample $t \sim \text{Uniform}(\{1, ..., T\})$
3. Sample $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
4. Compute $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$
5. Compute loss $\mathcal{L} = \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2$
6. Update $\theta$ using gradient descent

**Sampling:**
1. Sample $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. For $t = T, ..., 1$:
   - Predict $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
   - Compute predicted $\mathbf{x}_0$: $\hat{\mathbf{x}}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t))$
   - Compute mean: $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ using $\tilde{\boldsymbol{\mu}}_t$ formula
   - Sample $\mathbf{x}_{t-1} \sim \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \tilde{\beta}_t \mathbf{I})$
3. Return $\mathbf{x}_0$

### Implementation

```python
import torch.nn.functional as F

class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM)
    """
    def __init__(self, model, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Args:
            model: Neural network that predicts noise epsilon_theta(x_t, t)
            num_timesteps: Number of diffusion steps T
            beta_start, beta_end: Variance schedule parameters
        """
        self.model = model
        self.num_timesteps = num_timesteps

        # Use the ForwardDiffusion class we defined earlier
        self.diffusion = ForwardDiffusion(num_timesteps, beta_start, beta_end)

    def train_step(self, x_0):
        """
        Single training step for DDPM

        Args:
            x_0: Clean data [batch_size, ...]

        Returns:
            loss: Simplified DDPM loss (MSE between predicted and true noise)
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Get noisy data at timestep t
        x_t, _ = self.diffusion.q_sample(x_0, t, noise=noise)

        # Predict noise
        noise_pred = self.model(x_t, t)

        # Compute loss (simple MSE)
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def sample(self, shape, device='cuda'):
        """
        Generate samples using the reverse diffusion process

        Args:
            shape: Shape of samples to generate [batch_size, ...]
            device: Device to generate samples on

        Returns:
            x_0: Generated samples
        """
        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Iteratively denoise
        for t in reversed(range(self.num_timesteps)):
            # Create timestep tensor
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.model(x_t, t_batch)

            # Compute x_0 prediction
            alpha_t = self.diffusion.alphas_cumprod[t]
            alpha_t_prev = self.diffusion.alphas_cumprod_prev[t]
            beta_t = self.diffusion.betas[t]

            # Predict x_0
            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # Compute mean of p_theta(x_{t-1} | x_t)
            coef1 = torch.sqrt(alpha_t_prev) * beta_t / (1 - alpha_t)
            coef2 = torch.sqrt(self.diffusion.alphas[t]) * (1 - alpha_t_prev) / (1 - alpha_t)
            mean = coef1 * x_0_pred + coef2 * x_t

            if t > 0:
                # Add noise (except at t=0)
                variance = self.diffusion.posterior_variance[t]
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(variance) * noise
            else:
                x_t = mean

        return x_t

    @torch.no_grad()
    def sample_progressive(self, shape, device='cuda', save_every=100):
        """
        Generate samples and save intermediate steps

        Args:
            shape: Shape of samples
            device: Device
            save_every: Save every N steps

        Returns:
            intermediates: List of intermediate samples
        """
        x_t = torch.randn(shape, device=device)
        intermediates = [x_t.cpu()]

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            noise_pred = self.model(x_t, t_batch)

            alpha_t = self.diffusion.alphas_cumprod[t]
            alpha_t_prev = self.diffusion.alphas_cumprod_prev[t]
            beta_t = self.diffusion.betas[t]

            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            coef1 = torch.sqrt(alpha_t_prev) * beta_t / (1 - alpha_t)
            coef2 = torch.sqrt(self.diffusion.alphas[t]) * (1 - alpha_t_prev) / (1 - alpha_t)
            mean = coef1 * x_0_pred + coef2 * x_t

            if t > 0:
                variance = self.diffusion.posterior_variance[t]
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(variance) * noise
            else:
                x_t = mean

            if t % save_every == 0:
                intermediates.append(x_t.cpu())

        return intermediates
```

### Simple UNet for Noise Prediction

For images, the DDPM paper uses a UNet architecture with time embeddings. Here's a simplified version:

```python
class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding (similar to positional encoding in Transformers)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: Timestep tensor [batch_size]

        Returns:
            Time embeddings [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class SimpleDiffusionModel(nn.Module):
    """
    Simplified diffusion model for demonstration
    (For real applications, use UNet architecture)
    """
    def __init__(self, channels=3, time_dim=128):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_dim)

        # Simple convolutional network
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, channels, 3, padding=1)

        # Time projection layers
        self.time_proj1 = nn.Linear(time_dim, 64)
        self.time_proj2 = nn.Linear(time_dim, 64)

        self.act = nn.SiLU()  # Swish activation

    def forward(self, x, t):
        """
        Args:
            x: Noisy input [batch_size, channels, height, width]
            t: Timestep [batch_size]

        Returns:
            Predicted noise [batch_size, channels, height, width]
        """
        # Get time embeddings
        t_emb = self.time_embedding(t)  # [batch_size, time_dim]

        # First conv block
        h = self.act(self.conv1(x))
        h = h + self.time_proj1(t_emb)[:, :, None, None]  # Add time info

        # Second conv block
        h = self.act(self.conv2(h))
        h = h + self.time_proj2(t_emb)[:, :, None, None]

        # Output
        h = self.conv3(h)

        return h


# Example: Training loop
def train_ddpm(model, dataloader, num_epochs=100, device='cuda'):
    """
    Simple training loop for DDPM

    Args:
        model: SimpleDiffusionModel instance
        dataloader: DataLoader providing training images
        num_epochs: Number of training epochs
        device: Device to train on
    """
    model = model.to(device)
    ddpm = DDPM(model, num_timesteps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x_0, _) in enumerate(dataloader):
            x_0 = x_0.to(device)

            # Training step
            loss = ddpm.train_step(x_0)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model

# Example: Generate samples
# model = SimpleDiffusionModel(channels=3)
# trained_model = train_ddpm(model, dataloader)
# samples = ddpm.sample((16, 3, 32, 32), device='cuda')
```

## Connection to VAEs and Other Generative Models

### Variational Lower Bound

Both VAEs and diffusion models optimize a variational lower bound (ELBO/VLB) on the log-likelihood:

**VAE:**
$$
\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

**Diffusion:**
$$
\log p(x_0) \geq \mathbb{E}_q[-D_{KL}(q(x_T|x_0) \| p(x_T))] - \sum_{t=2}^T \mathbb{E}_q[D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))] + \mathbb{E}_q[\log p_\theta(x_0|x_1)]
$$

### Key Differences

| Aspect | VAE | Diffusion Model |
|--------|-----|-----------------|
| **Latent structure** | Single latent $z$ | Hierarchical $x_1, ..., x_T$ |
| **Encoder** | Learned $q_\phi(z\|x)$ | Fixed $q(x_t\|x_0)$ |
| **Decoder** | Learned $p_\theta(x\|z)$ | Learned $p_\theta(x_{t-1}\|x_t)$ |
| **Latent prior** | Simple $\mathcal{N}(0, I)$ | Simple $\mathcal{N}(0, I)$ |
| **Training** | Single-step reconstruction | Multi-step denoising |
| **Sampling** | Single forward pass | Iterative refinement |

### Hierarchical VAE Connection

Diffusion models can be viewed as a **hierarchical VAE** where:
- The encoder $q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$ is fixed (not learned)
- All latents $x_1, ..., x_T$ have the same dimensionality as the data
- The prior $p(x_T)$ is a simple Gaussian

This connection was explored in detail in the paper ["Variational Diffusion Models"](https://arxiv.org/abs/2107.00630).

### Comparison with Other Models

```python
# Pseudocode comparison

# VAE
class VAE:
    def encode(self, x):
        return mu, log_var  # q(z|x)

    def decode(self, z):
        return x_recon  # p(x|z)

    def sample(self):
        z = torch.randn(latent_dim)
        return self.decode(z)

# GAN
class GAN:
    def generator(self, z):
        return x_fake

    def discriminator(self, x):
        return real_or_fake

    def sample(self):
        z = torch.randn(latent_dim)
        return self.generator(z)

# Diffusion Model
class DiffusionModel:
    def add_noise(self, x, t):
        return x_noisy  # q(x_t|x_0) - fixed

    def denoise(self, x_t, t):
        return x_t_minus_1  # p(x_{t-1}|x_t) - learned

    def sample(self):
        x = torch.randn(data_shape)
        for t in reversed(range(T)):
            x = self.denoise(x, t)
        return x
```

## Practical Considerations

### Computational Cost

- **Training**: Comparable to VAEs (single forward/backward pass per sample)
- **Sampling**: Much slower than VAEs/GANs (requires T forward passes, typically T=1000)
- **Solution**: Fast sampling methods (DDIM, DPM-Solver) covered in [Chapter 25: Advanced Diffusion Topics](25-diffusion-advanced.md)

### Memory Requirements

```python
# Memory-efficient training: don't need to store all x_t
def memory_efficient_training_step(x_0, model, diffusion):
    """
    Only compute x_t on-the-fly, don't store intermediate steps
    """
    t = torch.randint(0, diffusion.num_timesteps, (x_0.shape[0],))
    noise = torch.randn_like(x_0)
    x_t, _ = diffusion.q_sample(x_0, t, noise)

    noise_pred = model(x_t, t)
    loss = F.mse_loss(noise_pred, noise)

    return loss  # x_t is automatically freed after this
```

### Hyperparameter Sensitivity

Key hyperparameters to tune:
1. **Variance schedule** ($\beta_1$, $\beta_T$, schedule type)
2. **Number of timesteps** $T$ (typically 1000, but can be reduced)
3. **Model architecture** (UNet depth, channels, attention)
4. **Learning rate** (typically 1e-4 to 2e-4)

### Common Pitfalls

1. **Improper variance schedule**: Too aggressive → training instability
2. **Not enough timesteps**: Poor sample quality
3. **Insufficient model capacity**: Can't model complex distributions
4. **Wrong noise prediction scaling**: Check that noise and x_t are properly normalized

## Interview Questions and Answers

### Q1: Explain the forward diffusion process mathematically.

**Answer:**
The forward process adds Gaussian noise over $T$ steps:
$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

The key insight is we can sample any $\mathbf{x}_t$ directly from $\mathbf{x}_0$ using:
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$$

where $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$.

This closed-form expression makes training efficient because we don't need to actually perform $t$ steps of noise addition.

### Q2: Why do diffusion models predict noise instead of the data directly?

**Answer:**
Several reasons:

1. **Empirical performance**: Ho et al. found noise prediction works better than predicting $\mathbf{x}_0$ directly
2. **Connection to score matching**: Predicting $\boldsymbol{\epsilon}$ is equivalent to predicting the score function $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$ (up to scaling)
3. **Training stability**: Noise has consistent scale across timesteps, whereas $\mathbf{x}_0$ predictions can have varying scales
4. **Theoretical justification**: Minimizing noise prediction error corresponds to maximizing a weighted VLB

### Q3: How do diffusion models compare to GANs and VAEs?

**Answer:**

**vs GANs:**
- **Pros**: More stable training (no adversarial dynamics), better mode coverage, explicit likelihood
- **Cons**: Much slower sampling (1000 steps vs 1 step)

**vs VAEs:**
- **Pros**: Better sample quality, no posterior collapse issues
- **Cons**: Slower sampling, higher computational cost

**Unique advantages:**
- Iterative refinement allows trading off quality vs speed
- Strong theoretical foundations
- Flexible architectures

### Q4: What is the relationship between diffusion models and score-based models?

**Answer:**
They are deeply connected:

1. **Score function**: $\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}$ for diffusion models
2. **Noise prediction = Score prediction**: $\boldsymbol{\epsilon}_\theta = -\sqrt{1-\bar{\alpha}_t} \cdot \mathbf{s}_\theta$
3. **Unified view**: Both can be described using SDEs (Stochastic Differential Equations)
4. **Same objective**: Denoising score matching ≈ DDPM simplified objective

The main difference is the parameterization and how they're presented, but mathematically they learn the same quantity.

### Q5: Derive the simplified DDPM objective from the variational lower bound.

**Answer:**
The VLB contains multiple KL divergence terms. For $L_{t-1}$, we compare:
- $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ (known, Gaussian)
- $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ (learned, Gaussian)

Both are Gaussians, so the KL divergence reduces to comparing means:
$$L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_\theta\|^2\right]$$

Substituting the reparameterization $\boldsymbol{\mu}_\theta = f(\boldsymbol{\epsilon}_\theta)$ and simplifying (ignoring constant factors):
$$L_{t-1} \propto \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

This is the simplified objective. The full derivation involves careful algebraic manipulation shown in the DDPM paper appendix.

## Exercises

### Exercise 1: Implement Variance Schedules
Implement and compare three variance schedules (linear, cosine, quadratic) on a simple 2D dataset. Visualize how $\bar{\alpha}_t$ affects the noise level at different timesteps.

<details>
<summary>Solution Sketch</summary>

```python
import matplotlib.pyplot as plt

# Generate 2D Gaussian data
data = torch.randn(1000, 2) * 0.5 + torch.tensor([2.0, 2.0])

# Test schedules
for schedule in ['linear', 'cosine', 'quadratic']:
    betas = get_beta_schedule(schedule, 1000)
    diffusion = ForwardDiffusion(1000, beta_start=0.0001, beta_end=0.02)

    # Sample at different timesteps
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for idx, t in enumerate([0, 250, 500, 999]):
        x_t, _ = diffusion.q_sample(data, torch.full((1000,), t))
        axes[idx].scatter(x_t[:, 0], x_t[:, 1], alpha=0.5)
        axes[idx].set_title(f't={t}')
    plt.suptitle(f'{schedule} schedule')
    plt.show()
```
</details>

### Exercise 2: Train on 2D Data
Train a small diffusion model on a 2D mixture of Gaussians. Visualize the reverse diffusion process.

<details>
<summary>Solution Sketch</summary>

```python
class Simple2DModel(nn.Module):
    def __init__(self, time_dim=16):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)

# Create mixture of Gaussians dataset
# Train model
# Visualize sampling process
```
</details>

### Exercise 3: Implement DDIM Sampling
Implement the DDIM (Denoising Diffusion Implicit Models) deterministic sampling method and compare with DDPM.

<details>
<summary>Hint</summary>
DDIM uses the same model but different sampling:
$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0^{pred} + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_\theta$$

This is deterministic (no added noise) and allows skipping timesteps.
</details>

### Exercise 4: Analyze the Score Function
Given a trained diffusion model, visualize the learned score function $\mathbf{s}_\theta(\mathbf{x}_t, t)$ at different noise levels for a 2D dataset.

### Exercise 5: Compare Parameterizations
Implement both noise prediction ($\boldsymbol{\epsilon}_\theta$) and data prediction ($\mathbf{x}_\theta$) versions. Train both on the same dataset and compare convergence speed and sample quality.

## Summary

In this chapter, we covered:

1. **Forward Diffusion**: Fixed noise-adding process with closed-form sampling
2. **Reverse Process**: Learned denoising using neural networks
3. **Score Functions**: Connection between denoising and score matching
4. **DDPM**: Simplified training objective (noise prediction MSE)
5. **Connections**: Relationship to VAEs and other generative models

**Key Takeaways:**
- Diffusion models learn to reverse a noise-adding process
- The simplified objective (predict noise) is just MSE, making training stable
- Score matching provides theoretical foundations
- Iterative sampling trades speed for quality

**Next Steps:**
- [Chapter 24: Implementing Diffusion Models](24-diffusion-implementation.md): Build complete image generation models with UNet architecture
- [Chapter 25: Advanced Diffusion Topics](25-diffusion-advanced.md): Fast sampling, conditional generation, guidance, and latent diffusion
- [Chapter 29: Architecture Comparison: Modern LLMs](29-model-architectures.md): See how diffusion applies to language models (WeDLM)

## References

1. **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., NeurIPS 2020
2. **Score-Based Models**: [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) - Song et al., ICLR 2021
3. **Improved DDPM**: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) - Nichol & Dhariwal, ICML 2021
4. **Score Matching**: [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) - Song & Ermon, NeurIPS 2019
5. **Variational Diffusion**: [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) - Kingma et al., NeurIPS 2021
6. **DDIM**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) - Song et al., ICLR 2021
7. **Tutorial**: [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - Lilian Weng's blog
