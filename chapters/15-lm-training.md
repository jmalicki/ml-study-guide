# Chapter 15: Language Model Training

This chapter covers the fundamentals of training large language models (LLMs) through causal language modeling. We'll explore the training objective, loss functions, implementation details, and practical techniques like gradient accumulation and mixed precision training. Understanding these fundamentals is essential for ML interviews, as they form the foundation of how modern LLMs are trained.

## Table of Contents

1. [Causal Language Modeling](#causal-language-modeling)
   - [Autoregressive Modeling](#autoregressive-modeling)
   - [Next Token Prediction](#next-token-prediction)
   - [Mathematical Formulation](#mathematical-formulation)
2. [Cross-Entropy Loss](#cross-entropy-loss)
   - [Derivation and Intuition](#derivation-and-intuition)
   - [Implementation](#implementation)
   - [Label Smoothing](#label-smoothing)
3. [Training Loop Implementation](#training-loop-implementation)
   - [Basic Training Loop](#basic-training-loop)
   - [Logging and Monitoring](#logging-and-monitoring)
   - [Checkpointing](#checkpointing)
4. [Gradient Accumulation](#gradient-accumulation)
   - [Why Gradient Accumulation](#why-gradient-accumulation)
   - [Implementation](#gradient-accumulation-implementation)
   - [Effective Batch Size](#effective-batch-size)
5. [Mixed Precision Training](#mixed-precision-training)
   - [Automatic Mixed Precision (AMP)](#automatic-mixed-precision-amp)
   - [BF16 vs FP16](#bf16-vs-fp16)
   - [Loss Scaling](#loss-scaling)
6. [Advanced Training Techniques](#advanced-training-techniques)
   - [Gradient Clipping](#gradient-clipping)
   - [Learning Rate Warmup](#learning-rate-warmup)
   - [Weight Decay](#weight-decay)
7. [Putting It All Together](#putting-it-all-together)

---

## Causal Language Modeling

### Autoregressive Modeling

Causal language modeling, also known as autoregressive language modeling, is the core training objective for decoder-only models like GPT, LLaMA, and Claude.

**Key Concept**: The model predicts each token based only on previous tokens, never future tokens. This creates a natural left-to-right generation process.

```python
import torch
import torch.nn as nn
from typing import Optional

class CausalLanguageModel(nn.Module):
    """
    Conceptual causal language model.

    At each position t, the model predicts token t+1 based on tokens 0..t.
    This is enforced by the causal attention mask.

    See [Building a Complete Transformer](11-complete-transformer.md) for
    full implementation details.
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        # Token embeddings + transformer + output projection
        # (Details omitted - see Chapter 11)

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len]
        labels: Optional[torch.Tensor] = None  # [batch, seq_len]
    ) -> dict:
        """
        Forward pass with optional loss computation.

        Args:
            input_ids: Input token IDs
            labels: Target token IDs (typically input_ids shifted by 1)

        Returns:
            Dictionary with logits and optional loss
        """
        # Get model outputs
        logits = self.get_logits(input_ids)  # [batch, seq_len, vocab_size]

        output = {"logits": logits}

        if labels is not None:
            # Compute cross-entropy loss
            loss = self.compute_loss(logits, labels)
            output["loss"] = loss

        return output

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits for each position (implementation details omitted)."""
        pass

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss (detailed below)."""
        pass
```

### Next Token Prediction

The fundamental task in causal language modeling is **next token prediction**: given tokens $x_1, x_2, \ldots, x_t$, predict token $x_{t+1}$.

**Example Sequence:**

```
Input:  "The cat sat on the"
Target: "cat sat on the mat"
```

For training, we shift the input by one position to create labels:

```python
def prepare_training_data(text_tokens):
    """
    Prepare input and target sequences for language modeling.

    Example:
        tokens = [1, 2, 3, 4, 5]  # "The cat sat on the"
        input_ids = [1, 2, 3, 4]  # "The cat sat on"
        labels    = [2, 3, 4, 5]  # "cat sat on the"
    """
    input_ids = text_tokens[:-1]
    labels = text_tokens[1:]
    return input_ids, labels


# Example with actual tokens
example_text = "The cat sat on the mat"
# After tokenization: [464, 3857, 3332, 319, 262, 2603]

input_ids = torch.tensor([[464, 3857, 3332, 319, 262]])  # "The cat sat on the"
labels = torch.tensor([[3857, 3332, 319, 262, 2603]])     # "cat sat on the mat"

# Model predicts:
# Position 0 (input "The") -> predicts "cat"
# Position 1 (input "The cat") -> predicts "sat"
# Position 2 (input "The cat sat") -> predicts "on"
# etc.
```

### Mathematical Formulation

Given a sequence of tokens $\mathbf{x} = (x_1, x_2, \ldots, x_T)$, the probability of the sequence is factorized as:

$$
P(\mathbf{x}) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})
$$

The training objective is to maximize the log-likelihood:

$$
\mathcal{L} = \sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})
$$

In practice, we minimize the negative log-likelihood (NLL):

$$
\text{Loss} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})
$$

The model outputs logits $\mathbf{z}_t \in \mathbb{R}^{V}$ at each position, which are converted to probabilities via softmax:

$$
P(x_t = i \mid x_{<t}) = \frac{\exp(z_{t,i})}{\sum_{j=1}^{V} \exp(z_{t,j})}
$$

where $V$ is the vocabulary size.

---

## Cross-Entropy Loss

### Derivation and Intuition

Cross-entropy loss measures the difference between the predicted probability distribution and the true distribution (one-hot for the correct token).

**Mathematical Definition:**

For a single position $t$ with true token $y_t$ and predicted logits $\mathbf{z}_t$:

$$
\mathcal{L}_{\text{CE}}(y_t, \mathbf{z}_t) = -\log \frac{\exp(z_{t, y_t})}{\sum_{j=1}^{V} \exp(z_{t,j})}
$$

This simplifies to:

$$
\mathcal{L}_{\text{CE}}(y_t, \mathbf{z}_t) = -z_{t, y_t} + \log \sum_{j=1}^{V} \exp(z_{t,j})
$$

The second term is the **log-sum-exp** (LSE), which acts as a normalizing constant.

**Intuition:**

- **First term** $-z_{t, y_t}$: Encourages high logit for correct token
- **Second term** $\log \sum \exp(z_{t,j})$: Discourages high logits for incorrect tokens

The loss is minimized when the model assigns all probability mass to the correct token.

### Implementation

```python
import torch.nn.functional as F

class CrossEntropyLoss:
    """
    Cross-entropy loss for language modeling.

    PyTorch's F.cross_entropy combines LogSoftmax and NLLLoss efficiently.
    It's numerically stable and optimized for GPU computation.
    """

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,  # [batch, seq_len, vocab_size]
        labels: torch.Tensor,  # [batch, seq_len]
        ignore_index: int = -100
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for language modeling.

        Args:
            logits: Model predictions
            labels: Ground truth token IDs
            ignore_index: Token IDs to ignore (e.g., padding tokens)

        Returns:
            Scalar loss value
        """
        # Reshape for cross_entropy: [batch * seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape

        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Compute cross-entropy loss
        # ignore_index masks out padding tokens
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=ignore_index,
            reduction='mean'
        )

        return loss

    @staticmethod
    def compute_loss_manual(
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Manual implementation for educational purposes.

        This shows what happens under the hood in F.cross_entropy.
        In practice, always use F.cross_entropy for efficiency.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Compute log probabilities (log-softmax)
        log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

        # Gather log probabilities of true tokens
        # This selects log_prob[i, j, labels[i, j]] for each i, j
        true_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq_len]

        # Negative log-likelihood
        loss = -true_log_probs.mean()

        return loss


# Example usage
def example_loss_computation():
    """Example of computing language modeling loss."""
    batch_size = 2
    seq_len = 5
    vocab_size = 10000

    # Model outputs
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # True tokens (labels)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1)
    )

    print(f"Loss: {loss.item():.4f}")

    # Expected loss for random predictions: log(vocab_size) ≈ 9.21
    # As training progresses, loss should decrease significantly
```

### Label Smoothing

Label smoothing softens the one-hot target distribution, preventing overconfidence and improving generalization.

**Standard target:** $[0, 0, 1, 0, 0]$ (one-hot)
**Smoothed target:** $[0.02, 0.02, 0.92, 0.02, 0.02]$ (smoothing = 0.1)

$$
y'_i = \begin{cases}
1 - \epsilon + \frac{\epsilon}{V} & \text{if } i = y \\
\frac{\epsilon}{V} & \text{otherwise}
\end{cases}
$$

```python
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Label smoothing is rarely used in LLM pretraining but common in
    fine-tuning and some vision tasks.

    Typical values: ε = 0.1 for vision, ε = 0.01-0.05 for NLP
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: [batch * seq_len, vocab_size]
            labels: [batch * seq_len]
        """
        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (vocab_size - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), self.confidence)

        # KL divergence between smoothed target and prediction
        loss = -torch.sum(true_dist * log_probs, dim=-1).mean()

        return loss
```

---

## Training Loop Implementation

### Basic Training Loop

```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class LanguageModelTrainer:
    """
    Basic trainer for causal language models.

    This implements a standard training loop with:
    - Forward pass
    - Loss computation
    - Backward pass
    - Optimizer step
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss


# Example training script
def train_language_model():
    """Example of training a language model."""
    from torch.optim import AdamW

    # Initialize model and optimizer
    model = CausalLanguageModel(
        vocab_size=50000,
        d_model=768,
        n_layers=12
    )

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Create trainer
    trainer = LanguageModelTrainer(model, optimizer, device='cuda')

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = trainer.train_epoch(train_dataloader)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        val_loss = trainer.evaluate(val_dataloader)
        print(f"Val Loss: {val_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'checkpoint_epoch_{epoch}.pt')
```

### Logging and Monitoring

```python
from torch.utils.tensorboard import SummaryWriter
import wandb

class TrainerWithLogging(LanguageModelTrainer):
    """
    Enhanced trainer with logging capabilities.

    Supports TensorBoard and Weights & Biases (wandb).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        log_dir: str = 'runs',
        use_wandb: bool = False,
        wandb_project: str = 'lm-training'
    ):
        super().__init__(model, optimizer, device)

        # TensorBoard
        self.writer = SummaryWriter(log_dir)

        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project)
            wandb.watch(model)

        self.global_step = 0

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch with logging."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log every N steps
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)

                if self.use_wandb:
                    wandb.log({
                        'train_loss_step': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })

        avg_loss = total_loss / num_batches

        # Log epoch metrics
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        if self.use_wandb:
            wandb.log({'train_loss_epoch': avg_loss, 'epoch': epoch})

        return avg_loss

    def close(self):
        """Clean up logging resources."""
        self.writer.close()
        if self.use_wandb:
            wandb.finish()


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity measures how "surprised" the model is by the data.
    Lower perplexity = better model.

    PPL = exp(loss)

    Typical values:
    - Random baseline: exp(log(vocab_size)) = vocab_size
    - Good LLM: 10-30 on validation data
    """
    import math
    return math.exp(loss)


# Example with perplexity logging
def log_metrics(loss: float, step: int, writer: SummaryWriter):
    """Log loss and perplexity."""
    perplexity = compute_perplexity(loss)

    writer.add_scalar('Loss/cross_entropy', loss, step)
    writer.add_scalar('Metrics/perplexity', perplexity, step)

    print(f"Step {step}: Loss = {loss:.4f}, Perplexity = {perplexity:.2f}")
```

### Checkpointing

```python
import os
from pathlib import Path

class CheckpointManager:
    """
    Manage model checkpoints during training.

    Features:
    - Save checkpoints at regular intervals
    - Keep only the best N checkpoints
    - Resume training from checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        keep_best_n: int = 3
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n
        self.best_checkpoints = []  # List of (loss, path) tuples

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        loss: float,
        is_best: bool = False
    ) -> Path:
        """
        Save a training checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            step: Current global step
            loss: Current loss
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        # Save checkpoint
        filename = f'checkpoint_epoch{epoch}_step{step}.pt'
        if is_best:
            filename = f'best_{filename}'

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        # Track best checkpoints
        if is_best:
            self.best_checkpoints.append((loss, path))
            self.best_checkpoints.sort(key=lambda x: x[0])  # Sort by loss

            # Remove excess checkpoints
            if len(self.best_checkpoints) > self.keep_best_n:
                _, old_path = self.best_checkpoints.pop()
                if old_path.exists():
                    old_path.unlink()

        return path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> dict:
        """
        Load a checkpoint and resume training.

        Returns:
            Dictionary with epoch, step, and loss information
        """
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'epoch': checkpoint['epoch'],
            'global_step': checkpoint['global_step'],
            'loss': checkpoint['loss']
        }

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
        return checkpoints[-1] if checkpoints else None


# Example usage
def train_with_checkpointing():
    """Example of training with checkpointing."""
    model = CausalLanguageModel(vocab_size=50000, d_model=768, n_layers=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    checkpoint_manager = CheckpointManager(checkpoint_dir='checkpoints', keep_best_n=3)

    # Resume from checkpoint if available
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    start_epoch = 0

    if latest_checkpoint:
        info = checkpoint_manager.load_checkpoint(latest_checkpoint, model, optimizer)
        start_epoch = info['epoch'] + 1
        print(f"Resumed from epoch {info['epoch']}, step {info['global_step']}")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(model, optimizer, train_dataloader)
        val_loss = evaluate(model, val_dataloader)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, global_step, val_loss, is_best=is_best
        )
```

---

## Gradient Accumulation

### Why Gradient Accumulation

**Problem**: Large models require large batch sizes for stable training, but GPU memory is limited.

**Solution**: Accumulate gradients over multiple forward-backward passes before updating weights.

**Example:**
- Target batch size: 64
- GPU can fit: 16 samples
- Accumulation steps: 4
- Effective batch size: 16 × 4 = 64

**Key Insight**: Gradient accumulation simulates large batch training without additional memory cost.

```python
def gradient_accumulation_explained():
    """
    Understanding gradient accumulation.

    Standard training:
    1. Forward pass (batch_size=64)
    2. Backward pass
    3. Optimizer step

    With gradient accumulation (accumulation_steps=4):
    1. Forward pass (batch_size=16)
    2. Backward pass (gradients accumulate)
    3. Forward pass (batch_size=16)
    4. Backward pass (gradients accumulate)
    5. Forward pass (batch_size=16)
    6. Backward pass (gradients accumulate)
    7. Forward pass (batch_size=16)
    8. Backward pass (gradients accumulate)
    9. Optimizer step (use accumulated gradients)
    10. Zero gradients

    Effective batch size: 16 × 4 = 64
    Memory usage: Only 16 samples at a time
    """
    pass
```

### Gradient Accumulation Implementation

```python
class TrainerWithGradientAccumulation(LanguageModelTrainer):
    """
    Trainer with gradient accumulation support.

    This enables training with larger effective batch sizes
    than can fit in GPU memory.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        gradient_accumulation_steps: int = 1
    ):
        super().__init__(model, optimizer, device)
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch with gradient accumulation.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Zero gradients at start
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']

            # Scale loss by accumulation steps
            # This ensures gradients are averaged correctly
            scaled_loss = loss / self.gradient_accumulation_steps

            # Backward pass (accumulates gradients)
            scaled_loss.backward()

            # Accumulate loss for logging (use unscaled)
            total_loss += loss.item()
            num_batches += 1

            # Update weights every N steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Handle any remaining accumulated gradients
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / num_batches
        return avg_loss


# Example usage
def train_with_gradient_accumulation():
    """Example of training with gradient accumulation."""
    model = CausalLanguageModel(vocab_size=50000, d_model=768, n_layers=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Create trainer with gradient accumulation
    trainer = TrainerWithGradientAccumulation(
        model=model,
        optimizer=optimizer,
        device='cuda',
        gradient_accumulation_steps=4  # Effective batch size = micro_batch_size × 4
    )

    # Training proceeds normally
    train_loss = trainer.train_epoch(train_dataloader)
```

### Effective Batch Size

```python
def compute_effective_batch_size(
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int = 1
) -> int:
    """
    Compute effective batch size.

    Effective batch size = micro_batch_size × accumulation_steps × num_gpus

    Args:
        micro_batch_size: Batch size per GPU per forward pass
        gradient_accumulation_steps: Number of accumulation steps
        num_gpus: Number of GPUs (for distributed training)

    Returns:
        Effective batch size
    """
    return micro_batch_size * gradient_accumulation_steps * num_gpus


# Example configurations
def example_batch_size_configs():
    """Common batch size configurations for LLM training."""

    configs = [
        {
            'name': 'Small model on single GPU',
            'micro_batch_size': 32,
            'accumulation_steps': 1,
            'num_gpus': 1,
        },
        {
            'name': 'Large model on single GPU',
            'micro_batch_size': 4,
            'accumulation_steps': 32,
            'num_gpus': 1,
        },
        {
            'name': 'Large model on 8 GPUs',
            'micro_batch_size': 4,
            'accumulation_steps': 8,
            'num_gpus': 8,
        },
    ]

    for config in configs:
        effective_bs = compute_effective_batch_size(
            config['micro_batch_size'],
            config['accumulation_steps'],
            config['num_gpus']
        )

        print(f"{config['name']}:")
        print(f"  Micro batch: {config['micro_batch_size']}")
        print(f"  Accumulation: {config['accumulation_steps']}")
        print(f"  GPUs: {config['num_gpus']}")
        print(f"  Effective batch: {effective_bs}")
        print()


# Token-based batch sizing (more common for LLMs)
def compute_tokens_per_batch(
    batch_size: int,
    sequence_length: int
) -> int:
    """
    Compute total tokens per batch.

    Modern LLMs often report batch sizes in tokens rather than sequences.

    Example:
        - batch_size = 256 sequences
        - sequence_length = 2048 tokens
        - tokens_per_batch = 256 × 2048 = 524,288 tokens (~0.5M)

    Large models like GPT-3 train with 1M-4M tokens per batch.
    """
    return batch_size * sequence_length


# Example: GPT-3 style batch configuration
def gpt3_batch_config():
    """
    GPT-3 training configuration (from the paper).

    - Batch size: 3.2M tokens
    - Sequence length: 2048
    - Number of sequences: 3.2M / 2048 = 1562.5 ≈ 1563

    To achieve this on modern hardware:
    - micro_batch_size = 4 (per GPU)
    - sequence_length = 2048
    - gradient_accumulation_steps = 32
    - num_gpus = 8 × 16 = 128 (using 16 nodes with 8 GPUs each)

    Effective batch: 4 × 32 × 128 = 16,384 sequences = 33.5M tokens
    (Actual GPT-3 likely used even more aggressive accumulation)
    """
    pass
```

---

## Mixed Precision Training

### Automatic Mixed Precision (AMP)

Mixed precision training uses FP16 or BF16 for most operations while keeping FP32 for critical computations. This speeds up training and reduces memory usage.

**Benefits:**
- 2-3x faster training (on GPUs with Tensor Cores)
- ~40% less memory usage
- Minimal accuracy loss

See [Hardware, Quantization, and Training Optimization](31-hardware-quantization-optimization.md) for detailed coverage of mixed precision training, including FP8.

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer(LanguageModelTrainer):
    """
    Trainer with automatic mixed precision (AMP).

    Uses PyTorch's built-in AMP for efficient training.

    See [Hardware, Quantization, and Training Optimization](31-hardware-quantization-optimization.md)
    for more details on mixed precision training.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__(model, optimizer, device)
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

        # GradScaler only needed for FP16, not BF16
        self.scaler = GradScaler() if (use_amp and amp_dtype == torch.float16) else None

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch with mixed precision."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass with autocast
            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs['loss']

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                # FP16 with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # BF16 or FP32 (no scaling needed)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss
```

### BF16 vs FP16

```python
def compare_precision_formats():
    """
    Compare FP16 and BF16 for LLM training.

    FP16 (Float16):
    - 1 sign bit, 5 exponent bits, 10 mantissa bits
    - Range: ±65,504
    - Precision: ~3-4 decimal digits
    - Pros: Good precision for small values
    - Cons: Limited range, gradient underflow issues

    BF16 (BFloat16):
    - 1 sign bit, 8 exponent bits, 7 mantissa bits
    - Range: ±3.4×10^38 (same as FP32!)
    - Precision: ~2-3 decimal digits
    - Pros: Same range as FP32, no gradient underflow
    - Cons: Lower precision than FP16

    For LLM training, BF16 is preferred:
    - No loss scaling needed
    - Handles gradient magnitudes better
    - Simpler training pipeline
    - Supported on modern hardware (Ampere+, TPU, etc.)

    See [Hardware, Quantization, and Training Optimization](31-hardware-quantization-optimization.md)
    for more details.
    """

    # Example: Gradient underflow with FP16
    import torch

    # Small gradient value
    grad_fp32 = torch.tensor([1e-6], dtype=torch.float32)
    grad_fp16 = grad_fp32.to(torch.float16)
    grad_bf16 = grad_fp32.to(torch.bfloat16)

    print(f"FP32: {grad_fp32.item():.10f}")
    print(f"FP16: {grad_fp16.item():.10f}")  # May underflow to 0
    print(f"BF16: {grad_bf16.item():.10f}")  # Preserves value
```

### Loss Scaling

Loss scaling is used with FP16 to prevent gradient underflow. It's not needed with BF16.

```python
class LossScalingExplained:
    """
    Loss scaling for FP16 training.

    Problem: Gradients often have very small values (e.g., 1e-7).
    FP16 can't represent values smaller than ~6e-5, so they underflow to 0.

    Solution: Scale loss by large factor (e.g., 1024) before backward pass.

    Process:
    1. loss_scaled = loss × scale
    2. loss_scaled.backward()  # Gradients are now scaled up
    3. optimizer.step()  # Automatically unscales gradients
    4. If overflow detected, skip update and reduce scale

    PyTorch's GradScaler handles this automatically.

    With BF16, loss scaling is not needed because BF16 has the
    same dynamic range as FP32.
    """

    @staticmethod
    def manual_loss_scaling_example():
        """Manual implementation of loss scaling (educational)."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()

        # Dummy input and target
        x = torch.randn(32, 10, dtype=torch.float16)
        y = torch.randn(32, 10, dtype=torch.float16)

        # Forward pass with autocast
        with autocast(dtype=torch.float16):
            output = model(x)
            loss = F.mse_loss(output, y)

        # Backward with scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # Scale loss before backward

        # Optimizer step with unscaling
        scaler.step(optimizer)  # Unscales gradients internally
        scaler.update()  # Adjust scale for next iteration
```

---

## Advanced Training Techniques

### Gradient Clipping

Gradient clipping prevents exploding gradients by limiting their magnitude.

```python
import torch.nn.utils as nn_utils

class TrainerWithGradientClipping(LanguageModelTrainer):
    """
    Trainer with gradient clipping.

    Gradient clipping is essential for stable LLM training.
    Typical value: max_norm = 1.0
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        max_grad_norm: float = 1.0
    ):
        super().__init__(model, optimizer, device)
        self.max_grad_norm = max_grad_norm

    def train_step(self, batch: dict) -> float:
        """Single training step with gradient clipping."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        outputs = self.model(input_ids, labels=labels)
        loss = outputs['loss']

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        nn_utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()

        return loss.item()


def gradient_clipping_methods():
    """
    Different gradient clipping methods.

    1. Clip by norm (most common for LLMs):
       - Compute global gradient norm: ||g|| = sqrt(sum(g_i^2))
       - If ||g|| > max_norm: g = g × (max_norm / ||g||)
       - Preserves gradient direction

    2. Clip by value:
       - Clamp each gradient: g_i = clip(g_i, -max_val, max_val)
       - Can distort gradient direction
       - Less common for LLMs
    """

    # Method 1: Clip by norm (recommended)
    max_norm = 1.0
    nn_utils.clip_grad_norm_(model.parameters(), max_norm)

    # Method 2: Clip by value
    max_value = 1.0
    nn_utils.clip_grad_value_(model.parameters(), max_value)


def why_gradient_clipping():
    """
    Why gradient clipping is important for LLMs.

    1. Prevents exploding gradients:
       - Deep networks can have gradient magnitudes that grow exponentially
       - Without clipping, optimizer steps can be too large

    2. Stabilizes training:
       - Reduces variance in gradient updates
       - Allows using larger learning rates

    3. Empirical evidence:
       - GPT-3: max_norm = 1.0
       - LLaMA: max_norm = 1.0
       - Most LLMs use gradient clipping

    Typical values:
    - max_norm = 1.0 (most common)
    - max_norm = 0.5 (more conservative)
    - max_norm = 5.0 (for very large models)
    """
    pass
```

### Learning Rate Warmup

Learning rate warmup gradually increases the learning rate from 0 to the target value at the start of training.

```python
class WarmupScheduler:
    """
    Learning rate warmup scheduler.

    Warmup prevents early instability by starting with a small learning rate
    and gradually increasing to the target value.

    See [Hardware, Quantization, and Training Optimization](31-hardware-quantization-optimization.md)
    for more advanced schedules (cosine, WSD, etc.).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        initial_lr: float = 0.0,
        max_lr: float = 3e-4
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * \
                 (self.current_step / self.warmup_steps)
        else:
            # After warmup, use max_lr (or apply a decay schedule)
            lr = self.max_lr

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


# Example usage
def train_with_warmup():
    """Example of training with learning rate warmup."""
    model = CausalLanguageModel(vocab_size=50000, d_model=768, n_layers=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Create scheduler with 2000 warmup steps
    scheduler = WarmupScheduler(
        optimizer=optimizer,
        warmup_steps=2000,
        initial_lr=0.0,
        max_lr=3e-4
    )

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # Training step
            loss = train_step(model, optimizer, batch)

            # Update learning rate
            scheduler.step()

            # Log current learning rate
            if step % 100 == 0:
                print(f"Step {step}, LR: {scheduler.get_lr():.2e}, Loss: {loss:.4f}")


def why_warmup():
    """
    Why learning rate warmup is important.

    1. Prevents early instability:
       - At initialization, weights are random
       - Large learning rate can cause divergence
       - Warmup allows model to stabilize

    2. Helps with Adam/AdamW:
       - Adam's second moment estimate is biased initially
       - Warmup gives time for estimates to stabilize

    3. Empirical evidence:
       - GPT-3: 375M step warmup (out of 300B tokens)
       - LLaMA: 2000 step warmup
       - BERT: 10,000 step warmup

    Typical warmup duration:
    - Small models: 1,000-2,000 steps
    - Large models: 2,000-10,000 steps
    - Usually <1% of total training steps
    """
    pass
```

### Weight Decay

Weight decay (L2 regularization) prevents overfitting by penalizing large weights.

```python
def weight_decay_explained():
    """
    Weight decay in AdamW.

    Standard L2 regularization adds penalty to loss:
        loss_total = loss + λ/2 × ||θ||²

    This leads to gradient:
        ∇loss_total = ∇loss + λθ

    In AdamW, weight decay is applied directly to weights:
        θ_{t+1} = θ_t - lr × (m_t / sqrt(v_t) + wd × θ_t)

    where wd is the weight decay coefficient.

    Key difference from Adam:
    - Adam: Weight decay is scaled by adaptive learning rate
    - AdamW: Weight decay is applied at a constant rate

    AdamW is superior for transformer training.

    Typical values for LLMs:
    - weight_decay = 0.1 (most common)
    - weight_decay = 0.01 (lighter regularization)
    - weight_decay = 0.0 (no regularization, rare)
    """

    # Example: AdamW with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1  # Weight decay coefficient
    )


def selective_weight_decay():
    """
    Apply weight decay selectively.

    Common practice: Don't apply weight decay to:
    - Bias terms
    - Layer normalization parameters
    - Embeddings (sometimes)

    This improves training stability.
    """
    # Separate parameters into groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if 'bias' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Create optimizer with different weight decay
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=3e-4)
```

---

## Putting It All Together

### Complete Training Pipeline

```python
class CompleteTrainer:
    """
    Complete training pipeline with all best practices.

    Includes:
    - Mixed precision training (BF16)
    - Gradient accumulation
    - Gradient clipping
    - Learning rate warmup
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: dict
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Configuration
        self.device = config.get('device', 'cuda')
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.amp_dtype = torch.bfloat16 if config.get('use_bf16', True) else torch.float32

        # Learning rate scheduling
        self.warmup_steps = config.get('warmup_steps', 2000)
        self.max_lr = config.get('max_lr', 3e-4)

        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)

        # Logging
        self.log_interval = config.get('log_interval', 100)
        self.writer = SummaryWriter(config.get('log_dir', 'runs'))

        # State
        self.global_step = 0
        self.current_epoch = 0

        # Move model to device
        self.model.to(self.device)

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Update learning rate with warmup
            self._update_learning_rate()

            # Training step
            loss = self._train_step(batch, batch_idx)

            total_loss += loss
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'lr': f'{self.get_lr():.2e}'
            })

            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_metrics(loss)

        avg_loss = total_loss / num_batches
        return avg_loss

    def _train_step(self, batch: dict, batch_idx: int) -> float:
        """Single training step with gradient accumulation."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass with mixed precision
        with autocast(dtype=self.amp_dtype):
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss'] / self.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every N steps
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.global_step += 1

        return loss.item() * self.gradient_accumulation_steps

    def _update_learning_rate(self):
        """Update learning rate with warmup."""
        if self.global_step < self.warmup_steps:
            lr = self.max_lr * self.global_step / self.warmup_steps
        else:
            lr = self.max_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def _log_metrics(self, loss: float):
        """Log training metrics."""
        perplexity = compute_perplexity(loss)

        self.writer.add_scalar('Loss/train', loss, self.global_step)
        self.writer.add_scalar('Metrics/perplexity', perplexity, self.global_step)
        self.writer.add_scalar('Learning_rate', self.get_lr(), self.global_step)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            with autocast(dtype=self.amp_dtype):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs['loss']

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Log validation metrics
        self.writer.add_scalar('Loss/val', avg_loss, self.global_step)

        return avg_loss

    def train(self, num_epochs: int):
        """Complete training loop."""
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")

            # Evaluate
            val_loss = self.evaluate()
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                step=self.global_step,
                loss=val_loss,
                is_best=is_best
            )

        self.writer.close()


# Example usage
def main():
    """Complete example of training a language model."""
    from torch.utils.data import Dataset, DataLoader

    # Model configuration
    model = CausalLanguageModel(
        vocab_size=50000,
        d_model=768,
        n_layers=12
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Data loaders (see [Data Curation and Preprocessing](14-data-curation.md))
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    # Training configuration
    config = {
        'device': 'cuda',
        'gradient_accumulation_steps': 4,
        'max_grad_norm': 1.0,
        'use_bf16': True,
        'warmup_steps': 2000,
        'max_lr': 3e-4,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'runs',
        'log_interval': 100,
    }

    # Create trainer
    trainer = CompleteTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )

    # Train
    trainer.train(num_epochs=10)


if __name__ == '__main__':
    main()
```

---

## Summary

### Key Takeaways for Interviews

1. **Causal Language Modeling**
   - Next token prediction is the core objective
   - Autoregressive factorization: $P(x_{1:T}) = \prod_t P(x_t | x_{<t})$
   - Training minimizes negative log-likelihood

2. **Cross-Entropy Loss**
   - Measures difference between predicted and true distributions
   - $\mathcal{L} = -\log P(\text{true token})$
   - Lower loss = better predictions

3. **Gradient Accumulation**
   - Simulates large batch sizes without additional memory
   - Effective batch = micro_batch × accumulation_steps × num_gpus
   - Essential for training large models

4. **Mixed Precision Training**
   - BF16 preferred over FP16 for LLMs (no loss scaling needed)
   - 2-3x speedup, ~40% memory reduction
   - Minimal accuracy loss

5. **Best Practices**
   - Gradient clipping (max_norm = 1.0)
   - Learning rate warmup (2000 steps typical)
   - Weight decay (0.1 for AdamW)
   - Regular checkpointing
   - Monitor perplexity

### Typical LLM Training Configuration

| Component | Value |
|-----------|-------|
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Learning rate | 3e-4 (peak) |
| Weight decay | 0.1 |
| Gradient clipping | max_norm = 1.0 |
| Warmup steps | 2000 |
| Precision | BF16 |
| Effective batch size | 1M-4M tokens |

---

## References

### Core Papers

1. Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. Radford et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
3. Brown et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)
4. Touvron et al. (2023). [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

### Optimization and Training

5. Loshchilov & Hutter (2017). [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (AdamW)
6. Micikevicius et al. (2017). [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
7. Zhang et al. (2019). [Which Algorithmic Choices Matter at Which Batch Sizes?](https://arxiv.org/abs/1907.04164)
8. Hoffmann et al. (2022). [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla)

### Additional Resources

9. [PyTorch Mixed Precision Documentation](https://pytorch.org/docs/stable/amp.html)
10. [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) - Production-grade training implementation
11. [DeepSpeed](https://www.deepspeed.ai/) - Optimization library for large-scale training
12. [Weights & Biases Guides](https://wandb.ai/site/articles/intro-to-language-model-training) - Training best practices

---

## Exercises

1. **Implement Cross-Entropy Loss**: Write a function that computes cross-entropy loss from scratch (without using `F.cross_entropy`). Compare results with PyTorch's implementation.

2. **Gradient Accumulation Math**: If you train with micro_batch_size=8, gradient_accumulation_steps=4, and 2 GPUs, what is your effective batch size? How much memory do you save compared to batch_size=64 on a single GPU?

3. **Loss Analysis**: Given a validation loss of 2.5 on a model with vocab_size=50,000, compute the perplexity. Is this a good result? What would random guessing achieve?

4. **Learning Rate Schedule**: Implement a cosine learning rate schedule with warmup. Plot the learning rate over 100,000 steps with 2,000 warmup steps.

5. **Training Configuration**: Design a training configuration for a 7B parameter model on 8×A100 GPUs (80GB each). Choose appropriate micro_batch_size, gradient_accumulation_steps, sequence_length, and precision format. Justify your choices.

6. **Perplexity Interpretation**: Explain what perplexity measures intuitively. If Model A has perplexity 20 and Model B has perplexity 15, which is better and by how much?

7. **Memory Calculation**: Calculate the total memory required for training (excluding data loading) given:
   - Model parameters: 7B (FP32)
   - Optimizer states: 2× parameters (AdamW)
   - Gradients: 1× parameters
   - Activations: batch_size=4, seq_len=2048, d_model=4096, n_layers=32
   - Precision: BF16 for activations, FP32 for optimizer

8. **Implementation Challenge**: Implement a complete training loop for a small transformer (d_model=256, n_layers=4) on a toy dataset. Include gradient accumulation, mixed precision, and checkpointing.
