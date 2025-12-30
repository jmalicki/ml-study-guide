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
   - [Learning Rate Warmup and Schedules](#learning-rate-warmup-and-schedules)
   - [Weight Decay](#weight-decay)
   - [Gradient Checkpointing](#gradient-checkpointing)
7. [Common Issues and Solutions](#common-issues-and-solutions)
   - [Loss Not Decreasing](#loss-not-decreasing)
   - [NaN or Inf Values in Loss](#nan-or-inf-values-in-loss)
   - [Out of Memory (OOM) Errors](#out-of-memory-oom-errors)
   - [Slow Training](#slow-training)
   - [Model Not Converging](#model-not-converging)
8. [Putting It All Together](#putting-it-all-together)

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

#### Problem: Model Overconfidence

Standard cross-entropy training encourages models to assign probability 1.0 to the correct token and 0.0 to all others. This can lead to:
- **Overconfidence**: Model becomes too certain about predictions, even when multiple tokens might be valid
- **Poor calibration**: Predicted probabilities don't reflect true uncertainty
- **Reduced generalization**: Model memorizes training distribution rather than learning robust patterns

#### Theoretical Justification

Label smoothing acts as a regularizer by preventing the model from becoming overly confident:

1. **Entropy regularization**: By distributing some probability mass to incorrect tokens, label smoothing increases the entropy of the output distribution, encouraging the model to remain uncertain when appropriate.

2. **Implicit knowledge distillation**: The smoothed distribution can be viewed as a soft teacher that provides additional information about which wrong answers are "less wrong" (all wrong answers are treated equally).

3. **Prevents extreme logits**: Without smoothing, the model is incentivized to push logits to $\pm\infty$ to achieve 100% probability. Smoothing bounds the optimal logits, leading to more stable training.

#### Relationship to Alternatives

- **Dropout**: Both prevent overfitting, but dropout adds noise during training while label smoothing modifies the training objective
- **Temperature scaling**: Used at inference time for calibration; label smoothing affects training
- **Mixup**: Another regularization technique that interpolates between training examples; label smoothing is simpler and cheaper

#### Key Insight

The optimal logit for the correct class with label smoothing $\epsilon$ is bounded: rather than pushing logits to infinity, the model learns to output finite values. This creates a margin between correct and incorrect predictions without extreme values, leading to:
- Better generalization to unseen data
- More robust probability estimates
- Reduced sensitivity to mislabeled training data

Note: Label smoothing is **rarely used in LLM pretraining** (where data is abundant and diverse) but can be beneficial for fine-tuning on smaller datasets or classification tasks.

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

See [Hardware, Quantization, and Training Optimization](32-hardware-quantization-optimization.md) for detailed coverage of mixed precision training, including FP8.

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer(LanguageModelTrainer):
    """
    Trainer with automatic mixed precision (AMP).

    Uses PyTorch's built-in AMP for efficient training.

    See [Hardware, Quantization, and Training Optimization](32-hardware-quantization-optimization.md)
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

    See [Hardware, Quantization, and Training Optimization](32-hardware-quantization-optimization.md)
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

### Learning Rate Warmup and Schedules

Learning rate scheduling is crucial for stable and efficient LLM training. Modern training typically combines warmup with various decay strategies.

#### Learning Rate Warmup

Warmup gradually increases the learning rate from 0 to the target value at the start of training.

##### The Problem: Training Instability at Initialization

When training starts, the model parameters are randomly initialized and far from optimal. Using a large learning rate immediately can cause:

1. **Divergence**: Large gradient updates can push parameters to regions with even worse loss
2. **Optimizer state instability**: Adaptive optimizers (Adam, AdamW) need time to build accurate estimates of gradient statistics
3. **Layer-wise gradient imbalance**: Different layers may have vastly different gradient magnitudes initially

**Example**: Starting GPT-3 training at lr=3e-4 from random initialization can cause the loss to spike to infinity within the first few steps.

##### Theoretical Justification

**1. Optimizer Moment Estimation**

Adam/AdamW maintain running estimates of gradient mean $m_t$ and variance $v_t$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

Early in training, these estimates are unreliable (high bias toward initialization at 0). Warmup gives the optimizer time to build accurate statistics before taking large steps.

**2. Gradient Noise and Batch Statistics**

At initialization, gradients can have high variance. The effective learning rate in Adam is $\frac{\alpha}{\sqrt{v_t}}$. Without warmup:
- Small initial $v_t$ → very large effective learning rate → instability
- Warmup allows $v_t$ to stabilize before using the full learning rate

**3. Sharp Loss Landscapes**

Random initialization may place parameters in regions with:
- Sharp minima (high curvature)
- Large gradients in certain directions

Warmup allows the model to escape these regions gently before accelerating optimization.

##### Relationship to Alternatives

- **Learning rate decay**: Applied late in training to refine convergence; warmup is applied at the start
- **Gradient clipping**: Prevents exploding gradients; warmup prevents them from occurring in the first place
- **Batch size warmup**: Alternative approach where batch size increases instead of learning rate (less common)
- **Layer-wise learning rates**: Different learning rates per layer; warmup applies to all layers

Warmup and decay are often combined: warmup → stable → decay (see WSD schedule below).

##### Key Insights

1. **Duration is critical but small**: Typical warmup is only 0.5-2% of total training steps, but it prevents catastrophic failures
2. **Linear warmup is standard**: More complex schedules (exponential, etc.) provide minimal benefit
3. **Larger models need more warmup**: GPT-3 used 375M warmup steps; smaller models use 1,000-2,000
4. **Works across architectures**: Transformers, CNNs, ResNets all benefit from warmup
5. **Essential for adaptive optimizers**: Less critical for SGD with momentum, crucial for Adam/AdamW

##### Empirical Evidence

- **BERT**: 10,000 step warmup (critical for training stability)
- **GPT-3**: 375M warmup steps out of 300B tokens
- **LLaMA**: 2,000 step warmup
- **Chinchilla**: ~1% of total steps

Without warmup, these models would diverge or converge to poor solutions.

```python
class WarmupScheduler:
    """
    Learning rate warmup scheduler.

    Warmup prevents early instability by starting with a small learning rate
    and gradually increasing to the target value.
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

#### Cosine Annealing Schedule

Cosine annealing smoothly decays the learning rate following a cosine curve. This is one of the most popular schedules for LLM training.

##### The Problem: Balancing Exploration and Convergence

During training, we face competing objectives:
- **Early/mid training**: Need high learning rate to explore loss landscape and make rapid progress
- **Late training**: Need low learning rate to converge to a good minimum without overshooting

A constant learning rate can't satisfy both. Abrupt changes (step decay) can cause:
- Training instability when learning rate drops
- Wasted computation if decay happens too early or too late
- Difficulty in hyperparameter tuning (when to decay? by how much?)

##### Theoretical Justification

**1. Smooth Convergence**

The cosine schedule provides a smooth, continuous decay that:
- Gradually reduces learning rate as training progresses
- Avoids sudden jumps that can destabilize training
- Ensures the model smoothly transitions from exploration to exploitation

**2. Stochastic Gradient Descent Theory**

Classical SGD theory suggests that learning rate should decay as $O(1/\sqrt{t})$ or $O(1/t)$ for convergence guarantees. However, for non-convex deep learning:
- Cosine decay is empirically superior to polynomial decay
- The specific functional form matters less than smooth, monotonic decrease
- Cosine provides a good balance: fast initial decay, slower later decay

**3. Connection to Simulated Annealing**

Cosine annealing draws inspiration from simulated annealing in optimization:
- High "temperature" (learning rate) early: explore widely
- Gradually cool (decay lr): focus on promising regions
- Low temperature late: fine-tune solution

The cosine curve naturally implements this cooling schedule.

##### Relationship to Alternatives

- **Step decay**: Drops learning rate by fixed factor at intervals; cosine is smoother and requires less tuning
- **Exponential decay**: $\text{lr}(t) = \text{lr}_0 e^{-\lambda t}$; decays too quickly early, too slowly late
- **Linear decay**: $\text{lr}(t) = \text{lr}_0 (1 - t/T)$; simpler but cosine provides better empirical results
- **Polynomial decay**: $\text{lr}(t) = \text{lr}_0 (1 - t/T)^p$; cosine is special case with smoother transition
- **Inverse sqrt**: $\text{lr}(t) = \text{lr}_0 / \sqrt{t}$; never reaches zero, used when training time is unknown

**Why cosine wins in practice:**
- Single hyperparameter (min_lr) vs multiple for step decay
- Smooth transitions prevent instability
- Works well across diverse tasks and model sizes
- Strong empirical track record (GPT-3, LLaMA, etc.)

##### Key Insights

1. **Front-loaded decay**: Cosine schedule decays more aggressively early, slower late - matches typical training dynamics where most progress happens early

2. **Non-zero minimum**: Setting $\text{lr}_{\min} = 0.1 \times \text{lr}_{\max}$ (rather than 0) prevents premature stagnation and allows continued refinement

3. **Restarts possible**: Cosine schedule can be restarted (SGDR - Stochastic Gradient Descent with Warm Restarts) to escape local minima, though rarely used in LLM pretraining

4. **Schedule length must match training**: Unlike inverse-sqrt, cosine requires knowing total training steps upfront - a minor constraint in practice

5. **Empirical sweet spot**: The cosine curve happens to match the empirical behavior of loss improvement in transformers, though the theoretical reason is not fully understood

**Mathematical Formulation:**

After warmup, the learning rate follows:

$$
\text{lr}(t) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min}) \left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}} \pi\right)\right)
$$

where:
- $t$ is the current step
- $t_{\text{warmup}}$ is the warmup duration
- $T$ is the total training steps
- $\text{lr}_{\max}$ is the peak learning rate
- $\text{lr}_{\min}$ is the minimum learning rate (often 0.1 × lr_max)

**Properties:**
- At $t = t_{\text{warmup}}$: $\text{lr} = \text{lr}_{\max}$ (starts at peak)
- At $t = T$: $\text{lr} = \text{lr}_{\min}$ (ends at minimum)
- Derivative is continuous everywhere (smooth)
- Decay rate is fastest around $t = (T + t_{\text{warmup}})/2$ (midpoint)

```python
import math

class CosineAnnealingWithWarmup:
    """
    Cosine annealing learning rate schedule with warmup.

    Used in GPT-3, LLaMA, and many modern LLMs.

    Learning rate schedule:
    - Steps 0 to warmup_steps: Linear increase from 0 to max_lr
    - Steps warmup_steps to total_steps: Cosine decay from max_lr to min_lr
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float = 3e-4,
        min_lr: float = 3e-5
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / \
                      (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


# PyTorch built-in version
def get_cosine_schedule_with_warmup_pytorch(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int
):
    """
    Create cosine schedule using PyTorch's built-in scheduler.

    This is equivalent to the manual implementation above.
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
```

#### Linear Decay Schedule

Linear decay decreases the learning rate linearly after warmup. Simpler than cosine but still effective.

$$
\text{lr}(t) = \text{lr}_{\max} \times \left(1 - \frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\right)
$$

```python
class LinearDecayWithWarmup:
    """
    Linear learning rate decay with warmup.

    Simpler alternative to cosine annealing.
    Used in some BERT variants and older transformers.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float = 3e-4,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            # Linear decay
            progress = (self.current_step - self.warmup_steps) / \
                      (self.total_steps - self.warmup_steps)
            lr = self.max_lr - (self.max_lr - self.min_lr) * progress

        # Ensure lr doesn't go below min_lr
        lr = max(lr, self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
```

#### Inverse Square Root Schedule

The inverse square root schedule decays learning rate as $1/\sqrt{t}$. Popular in machine translation and early transformer models.

$$
\text{lr}(t) = \text{lr}_{\max} \times \min\left(1, \frac{1}{\sqrt{t}}, \frac{t}{t_{\text{warmup}}}\right)
$$

```python
class InverseSqrtSchedule:
    """
    Inverse square root learning rate schedule.

    Used in the original Transformer paper (Vaswani et al., 2017).
    Less common for modern LLM pretraining but still used in some settings.

    lr(t) = d_model^(-0.5) * min(t^(-0.5), t * warmup_steps^(-1.5))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        d_model: int = 512,  # Model dimension, used for scaling
        scale: float = 1.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.scale = scale
        self.current_step = 0

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1

        # Original Transformer formulation
        step = max(self.current_step, 1)  # Avoid division by zero
        lr = self.scale * (self.d_model ** -0.5) * \
             min(step ** -0.5, step * (self.warmup_steps ** -1.5))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
```

#### WSD (Warmup-Stable-Decay) Schedule

The WSD schedule is used in many recent large language models. It consists of three phases: warmup, stable training at peak LR, and final decay.

##### The Problem: Premature Decay vs Late Convergence

Traditional cosine schedules decay learning rate throughout training. However, for very large models and long training runs, this presents challenges:

1. **Uncertainty in training budget**: You may want to extend training if loss is still decreasing
2. **Most learning happens mid-training**: Early training is exploration; late training is refinement; mid-training does the heavy lifting
3. **Cosine decay starts immediately**: After warmup, learning rate begins decreasing - potentially too early for massive models

**Example**: PaLM (540B parameters) and Chinchilla (70B) trained on trillions of tokens. Starting decay immediately after warmup would waste the majority of training compute.

##### Theoretical Justification

**1. Staged Learning Dynamics**

Large-scale training exhibits distinct phases:
- **Phase 1 (Warmup)**: Stabilize optimizer and escape initialization
- **Phase 2 (Stable)**: Rapid loss reduction as model learns fundamental patterns
- **Phase 3 (Decay)**: Fine-tuning and convergence to final solution

WSD explicitly models these phases rather than smoothly transitioning between them.

**2. Computational Efficiency**

For models trained on 1-10 trillion tokens:
- Warmup: ~1% of steps (stabilization)
- Stable: ~80-90% of steps (main learning)
- Decay: ~10-15% of steps (convergence)

This allocation ensures maximum compute is spent with optimal learning rate, with decay reserved for final refinement.

**3. Flexibility and Continuation**

Unlike cosine (which requires knowing total steps upfront), WSD:
- Allows extending training in stable phase if beneficial
- Can transition to decay phase based on validation performance
- Separates exploration (stable) from exploitation (decay)

##### Relationship to Alternatives

- **Cosine annealing**: WSD is a generalization; cosine ≈ WSD with zero stable phase
- **Constant LR**: WSD's stable phase is constant LR, but adds warmup and decay
- **Step decay**: WSD uses smooth decay after stable, not abrupt steps
- **Inverse sqrt**: Never reaches minimum; WSD guarantees convergence via decay phase

**Why WSD for large-scale training:**
- Cosine: Good for fixed budgets, shorter training
- WSD: Better for long training runs, flexible stopping, very large models
- Most modern LLMs use WSD or variants (PaLM, Chinchilla, Gopher, etc.)

##### Key Insights

1. **Most learning happens at peak LR**: The stable phase (80-90% of training) should use the maximum learning rate, not a decaying one

2. **Decay for final convergence only**: The last 10-15% is enough to converge; decaying earlier wastes potential learning

3. **Empirical validation**: PaLM, Chinchilla, and Gopher all showed improved performance with WSD vs continuous decay

4. **Scaling law compatibility**: WSD aligns with Chinchilla scaling laws suggesting optimal compute allocation throughout training

5. **Monitoring is key**: Can observe when loss plateaus in stable phase to decide when to start decay phase

##### Typical Allocation

For a training run of $T$ total steps:
- **Warmup**: $0.01T$ to $0.02T$ (1-2%)
- **Stable**: $0.80T$ to $0.90T$ (80-90%)
- **Decay**: $0.10T$ to $0.15T$ (10-15%)

**Example (100K steps):**
- Warmup: 1,000 steps
- Stable: 85,000 steps
- Decay: 14,000 steps

```python
class WSDSchedule:
    """
    Warmup-Stable-Decay (WSD) learning rate schedule.

    Used in recent large language models like PaLM and Chinchilla.

    Three phases:
    1. Warmup: Linear increase to max_lr
    2. Stable: Constant max_lr for majority of training
    3. Decay: Cosine or linear decay to min_lr

    Typical allocation:
    - Warmup: 1-2% of total steps
    - Stable: 80-90% of total steps
    - Decay: 10-15% of total steps
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        max_lr: float = 3e-4,
        min_lr: float = 3e-5,
        decay_type: str = 'cosine'  # 'cosine' or 'linear'
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.decay_type = decay_type
        self.current_step = 0

        self.total_steps = warmup_steps + stable_steps + decay_steps

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Phase 1: Warmup
            lr = self.max_lr * self.current_step / self.warmup_steps

        elif self.current_step <= self.warmup_steps + self.stable_steps:
            # Phase 2: Stable
            lr = self.max_lr

        else:
            # Phase 3: Decay
            decay_progress = (self.current_step - self.warmup_steps - self.stable_steps) / \
                           self.decay_steps

            if self.decay_type == 'cosine':
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                     (1 + math.cos(math.pi * decay_progress))
            else:  # linear
                lr = self.max_lr - (self.max_lr - self.min_lr) * decay_progress

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# Example: Create WSD schedule for 100K steps
def example_wsd_schedule():
    """Example WSD schedule configuration."""
    total_steps = 100000

    # Typical allocation
    warmup_steps = int(0.01 * total_steps)      # 1% warmup
    stable_steps = int(0.85 * total_steps)      # 85% stable
    decay_steps = total_steps - warmup_steps - stable_steps  # 14% decay

    model = CausalLanguageModel(vocab_size=50000, d_model=768, n_layers=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    scheduler = WSDSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        max_lr=3e-4,
        min_lr=3e-5,
        decay_type='cosine'
    )

    return scheduler
```

#### Comparing Learning Rate Schedules

```python
def compare_lr_schedules():
    """
    Compare different learning rate schedules.

    Visualize how different schedules behave over training.
    """
    import matplotlib.pyplot as plt

    total_steps = 10000
    warmup_steps = 500
    max_lr = 3e-4
    min_lr = 3e-5

    # Create dummy optimizer
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

    # Create schedulers
    schedules = {
        'Cosine': CosineAnnealingWithWarmup(
            optimizer, warmup_steps, total_steps, max_lr, min_lr
        ),
        'Linear Decay': LinearDecayWithWarmup(
            optimizer, warmup_steps, total_steps, max_lr, min_lr
        ),
        'WSD': WSDSchedule(
            optimizer, warmup_steps, int(0.8 * total_steps),
            total_steps - warmup_steps - int(0.8 * total_steps),
            max_lr, min_lr, 'cosine'
        ),
    }

    # Track learning rates
    lr_histories = {name: [] for name in schedules}

    for step in range(total_steps):
        for name, scheduler in schedules.items():
            lr_histories[name].append(scheduler.get_lr())
            scheduler.step()

    # Plot
    plt.figure(figsize=(12, 6))
    for name, lrs in lr_histories.items():
        plt.plot(lrs, label=name, linewidth=2)

    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('lr_schedules.png')
    plt.close()

    print("Learning rate schedule comparison saved to lr_schedules.png")


def schedule_recommendations():
    """
    Recommendations for choosing learning rate schedules.

    1. Cosine Annealing (Most Common):
       - Best for: Pretraining large language models
       - Pros: Smooth decay, well-studied, good empirical results
       - Cons: Requires knowing total training steps in advance
       - Use cases: GPT-3, LLaMA, most modern LLMs

    2. WSD (Modern Choice):
       - Best for: Very large models, long training runs
       - Pros: Stable phase allows for more consistent training
       - Cons: More hyperparameters to tune
       - Use cases: PaLM, Chinchilla, Gopher

    3. Linear Decay:
       - Best for: Fine-tuning, smaller models
       - Pros: Simple, predictable
       - Cons: Less smooth than cosine, may not optimize as well
       - Use cases: BERT fine-tuning, classification tasks

    4. Inverse Square Root:
       - Best for: Machine translation, when training time is unknown
       - Pros: No need to specify total steps, mathematically principled
       - Cons: Less common for modern LLM pretraining
       - Use cases: Original Transformer, NMT models

    General Guidelines:
    - For pretraining: Use Cosine or WSD
    - For fine-tuning: Use Linear Decay or Cosine with short decay
    - Warmup steps: 0.5-2% of total training steps
    - min_lr: Typically 10% of max_lr (e.g., 3e-5 if max_lr is 3e-4)
    """
    pass
```

#### Example: Training with Cosine Schedule

```python
def train_with_cosine_schedule():
    """Example of training with cosine annealing schedule."""
    # Configuration
    total_steps = 100000
    warmup_steps = 2000
    max_lr = 3e-4
    min_lr = 3e-5

    # Model and optimizer
    model = CausalLanguageModel(vocab_size=50000, d_model=768, n_layers=12)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Create scheduler
    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr=max_lr,
        min_lr=min_lr
    )

    # Training loop
    for step in range(total_steps):
        # Training step
        loss = train_step(model, optimizer)

        # Update learning rate
        scheduler.step()

        # Log
        if step % 1000 == 0:
            current_lr = scheduler.get_lr()
            print(f"Step {step}/{total_steps}: "
                  f"Loss = {loss:.4f}, LR = {current_lr:.2e}")
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

### Gradient Checkpointing

Gradient checkpointing (also called activation checkpointing) is a memory-saving technique that trades computation for memory by recomputing activations during the backward pass instead of storing them.

#### The Problem: Activation Memory Bottleneck

During standard backpropagation, the forward pass computes and stores activations at every layer. These activations are needed during the backward pass to compute gradients via the chain rule. For deep networks:

- **Memory grows linearly with depth**: A 32-layer transformer stores 32 sets of activations
- **Activation memory dominates**: For large models, activations often consume 10-100x more memory than model parameters
- **Limits batch size and model scale**: With limited GPU memory, you're forced to use smaller batches or smaller models

**Example**: A 7B parameter LLaMA model with batch_size=8, seq_len=2048 requires ~100GB just for activations (far exceeding parameter storage of ~14GB in BF16).

#### Theoretical Justification: The Memory-Computation Trade-off

Gradient checkpointing is based on a fundamental insight: **activations can be recomputed from stored checkpoints**.

**Mathematical Formulation:**

For a sequential model $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$:
- Standard backprop: Store all intermediate activations $a_1, a_2, \ldots, a_L$
- Gradient checkpointing: Store only selected checkpoints (e.g., every $k$ layers)
- During backward: Recompute missing activations by re-running forward pass from nearest checkpoint

**Optimal checkpoint spacing** (Chen et al., 2016): For $L$ layers, placing $\sqrt{L}$ checkpoints reduces memory from $O(L)$ to $O(\sqrt{L})$ with only $O(\sqrt{L})$ recomputation overhead.

#### Relationship to Alternatives

- **Model parallelism**: Splits model across GPUs; gradient checkpointing reduces memory per GPU
- **Gradient accumulation**: Reduces batch size; gradient checkpointing allows larger batch sizes
- **Mixed precision**: Reduces memory per activation; gradient checkpointing reduces number of stored activations
- **Offloading**: Moves activations to CPU/disk; gradient checkpointing recomputes instead of transferring

These techniques are **complementary** and often used together in large-scale training.

#### Key Insights That Make It Work

1. **Computation is cheap, memory is expensive**: Modern GPUs have abundant compute (TFLOPS) but limited memory (tens of GB). Trading 20-30% more compute for 40-50% less memory is often worthwhile.

2. **Selective checkpointing**: Not all layers need checkpointing. Typically:
   - Checkpoint transformer blocks (large activations)
   - Don't checkpoint embeddings or output layers (small activations)
   - Checkpoint every $k$ layers to balance memory and compute

3. **Automatic differentiation compatibility**: PyTorch's autograd system seamlessly integrates checkpointing - no manual gradient calculations needed.

4. **Critical for scale**: Essential for training models >3B parameters on consumer hardware (e.g., single A100).

#### How Gradient Checkpointing Works

**Standard Training:**
1. Forward pass: Compute and store all activations
2. Backward pass: Use stored activations to compute gradients
3. Memory usage: O(n × layers) where n is batch size

**With Gradient Checkpointing:**
1. Forward pass: Only store activations at checkpoints (e.g., every few layers)
2. Backward pass: Recompute activations from checkpoints as needed
3. Memory usage: O(n × sqrt(layers)) - significant reduction!

**Trade-off:**
- Memory savings: 40-50% reduction in activation memory
- Computational cost: 20-30% slower (one extra forward pass per checkpoint)

```python
import torch.utils.checkpoint as checkpoint

class TransformerBlockWithCheckpointing(nn.Module):
    """
    Transformer block with gradient checkpointing support.

    Gradient checkpointing is essential for training very large models
    or using larger batch sizes with limited GPU memory.
    """

    def __init__(self, d_model: int, n_heads: int, use_checkpoint: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass with optional gradient checkpointing.
        """
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing during training
            return checkpoint.checkpoint(self._forward, x, mask)
        else:
            # Normal forward pass (inference or no checkpointing)
            return self._forward(x, mask)

    def _forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Actual forward computation."""
        # Self-attention block
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out

        # Feed-forward block
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out

        return x


class ModelWithCheckpointing(nn.Module):
    """
    Complete model with gradient checkpointing.

    Typically applied to transformer layers, not embeddings or output layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        use_checkpoint: bool = True
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(2048, d_model)

        # Transformer layers with checkpointing
        self.layers = nn.ModuleList([
            TransformerBlockWithCheckpointing(d_model, n_heads, use_checkpoint)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        """Forward pass with gradient checkpointing in layers."""
        batch_size, seq_len = input_ids.shape

        # Embeddings (not checkpointed - small memory footprint)
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(pos_ids)
        x = token_emb + pos_emb

        # Transformer layers (checkpointed)
        for layer in self.layers:
            x = layer(x)

        # Output (not checkpointed)
        x = self.output_norm(x)
        logits = self.output_projection(x)

        return logits


# Alternative: PyTorch's built-in checkpoint function
def example_checkpoint_function():
    """
    Example using PyTorch's checkpoint function directly.

    This is more flexible than the module-based approach above.
    """

    def custom_forward(module, x):
        """Custom forward function to checkpoint."""
        return module(x)

    class ModelWithFlexibleCheckpointing(nn.Module):
        def __init__(self, layers: nn.ModuleList, checkpoint_every: int = 1):
            super().__init__()
            self.layers = layers
            self.checkpoint_every = checkpoint_every

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                # Checkpoint every N layers
                if i % self.checkpoint_every == 0 and self.training:
                    x = checkpoint.checkpoint(custom_forward, layer, x)
                else:
                    x = layer(x)
            return x

    return ModelWithFlexibleCheckpointing


# Memory comparison
def memory_comparison():
    """
    Memory usage comparison: with and without gradient checkpointing.

    Example for a 7B parameter model:

    Without checkpointing:
    - Activations: ~100GB for batch_size=8, seq_len=2048
    - Cannot train on single 80GB A100

    With checkpointing:
    - Activations: ~50GB for same batch size
    - Can train on single 80GB A100

    Rule of thumb:
    - Checkpointing reduces activation memory by ~40-50%
    - Increases training time by ~20-30%
    - Essential for large models (>3B parameters) on consumer GPUs
    """
    pass


def when_to_use_checkpointing():
    """
    Guidelines for using gradient checkpointing.

    Use gradient checkpointing when:
    1. Training large models (>1B parameters)
    2. Using long sequences (>2048 tokens)
    3. Want to maximize batch size
    4. GPU memory is the bottleneck (not computation)

    Don't use gradient checkpointing when:
    1. Training small models (<500M parameters)
    2. Computational speed is critical
    3. Already have enough GPU memory
    4. Using very short sequences (<512 tokens)

    Best practices:
    - Checkpoint transformer layers, not embeddings/output layers
    - Consider checkpointing every N layers (e.g., every 2-3 layers)
    - Combine with other memory optimizations (mixed precision, etc.)
    - Profile to find optimal checkpoint frequency
    """
    pass


# Example usage in training
def train_with_gradient_checkpointing():
    """Example of training with gradient checkpointing."""
    # Create model with checkpointing enabled
    model = ModelWithCheckpointing(
        vocab_size=50000,
        d_model=4096,
        n_layers=32,
        n_heads=32,
        use_checkpoint=True  # Enable gradient checkpointing
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Training loop (same as before)
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to('cuda')
        labels = batch['labels'].to('cuda')

        # Forward pass (checkpointing happens automatically)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, model.output_projection.out_features),
            labels.view(-1)
        )

        # Backward pass (activations are recomputed as needed)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Training with gradient checkpointing complete!")


# Advanced: Selective checkpointing
class SelectiveCheckpointing:
    """
    Advanced gradient checkpointing strategies.

    Instead of checkpointing all layers, you can be selective:
    1. Checkpoint only expensive layers (e.g., large FFN layers)
    2. Checkpoint every N layers
    3. Checkpoint based on memory/computation trade-off
    """

    @staticmethod
    def checkpoint_every_n_layers(layers: nn.ModuleList, n: int = 3):
        """
        Checkpoint every N layers.

        Example: For 32 layers with n=3, checkpoint layers 0, 3, 6, 9, ...
        This balances memory savings with computation overhead.
        """
        class SelectiveCheckpointModel(nn.Module):
            def __init__(self, layers, checkpoint_interval):
                super().__init__()
                self.layers = layers
                self.checkpoint_interval = checkpoint_interval

            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    if i % self.checkpoint_interval == 0 and self.training:
                        x = checkpoint.checkpoint(layer, x)
                    else:
                        x = layer(x)
                return x

        return SelectiveCheckpointModel(layers, n)
```

---

## Common Issues and Solutions

This section covers common problems encountered during LLM training and how to debug them. These are essential for interviews as they demonstrate practical experience.

### Loss Not Decreasing

**Symptoms:**
- Training loss stays constant or decreases very slowly
- Validation loss doesn't improve
- Model appears to be "stuck"

**Common Causes and Solutions:**

```python
def debug_loss_not_decreasing():
    """
    Debugging checklist for loss not decreasing.

    1. Learning rate too low:
       - Check: Print current learning rate
       - Fix: Increase max_lr (try 1e-4, 3e-4, 6e-4)
       - Note: Use learning rate finder to find optimal LR

    2. Learning rate too high:
       - Check: Loss oscillates or increases
       - Fix: Decrease max_lr, increase warmup steps

    3. Gradient flow issues:
       - Check: Print gradient norms
       - Fix: Check for vanishing gradients, adjust initialization

    4. Data quality issues:
       - Check: Inspect training data
       - Fix: Verify labels match inputs, check for corruption

    5. Model too small/large:
       - Check: Monitor training vs validation loss
       - Fix: Adjust model size

    6. Incorrect loss computation:
       - Check: Verify loss calculation
       - Fix: Ensure labels are shifted correctly for LM
    """

    # Debug: Check learning rate
    def check_learning_rate(optimizer):
        lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {lr:.2e}")
        if lr < 1e-5:
            print("WARNING: Learning rate very low!")
        elif lr > 1e-3:
            print("WARNING: Learning rate very high!")

    # Debug: Check gradient norms
    def check_gradient_norms(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm:.4f}")

        if total_norm < 1e-6:
            print("WARNING: Vanishing gradients detected!")
        elif total_norm > 100:
            print("WARNING: Exploding gradients detected!")

    # Debug: Verify data
    def verify_data(batch):
        input_ids = batch['input_ids']
        labels = batch['labels']

        print(f"Input shape: {input_ids.shape}")
        print(f"Labels shape: {labels.shape}")

        # Check if labels are shifted correctly
        print(f"First input token: {input_ids[0, 0]}")
        print(f"First label token: {labels[0, 0]}")
        print(f"Should match: input[0,1] == labels[0,0]? "
              f"{input_ids[0, 1].item() == labels[0, 0].item()}")

    # Learning rate finder
    def learning_rate_finder(model, train_dataloader, device='cuda'):
        """
        Simple learning rate finder implementation.

        Gradually increases LR and tracks loss to find optimal range.
        """
        import matplotlib.pyplot as plt

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-8)

        lr_find_steps = 100
        start_lr = 1e-8
        end_lr = 1e-2

        lrs = []
        losses = []

        # Exponential LR increase
        gamma = (end_lr / start_lr) ** (1 / lr_find_steps)

        for step, batch in enumerate(train_dataloader):
            if step >= lr_find_steps:
                break

            # Update learning rate
            lr = start_lr * (gamma ** step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Training step
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lrs.append(lr)
            losses.append(loss.item())

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(alpha=0.3)
        plt.savefig('lr_finder.png')
        plt.close()

        print("Learning rate finder complete. Check lr_finder.png")
        print(f"Suggested LR: {lrs[losses.index(min(losses))]:.2e}")
```

### NaN or Inf Values in Loss

**Symptoms:**
- Loss becomes NaN (not a number) or inf (infinity)
- Training crashes or produces nonsensical outputs

**Common Causes and Solutions:**

```python
def debug_nan_loss():
    """
    Debugging NaN or inf loss values.

    1. Gradient explosion:
       - Symptom: Loss suddenly becomes NaN
       - Fix: Add/strengthen gradient clipping
       - Fix: Reduce learning rate
       - Fix: Increase warmup steps

    2. Numerical instability:
       - Symptom: NaN appears in specific operations
       - Fix: Use mixed precision (BF16, not FP16 without scaling)
       - Fix: Check for division by zero
       - Fix: Use numerically stable implementations

    3. Bad initialization:
       - Symptom: NaN on first few steps
       - Fix: Check weight initialization
       - Fix: Use proper layer norm placement

    4. Data issues:
       - Symptom: Random NaN appearances
       - Fix: Check for NaN/inf in input data
       - Fix: Verify data preprocessing

    5. FP16 underflow:
       - Symptom: NaN when using FP16
       - Fix: Use BF16 instead
       - Fix: Use loss scaling with FP16
    """

    # Add NaN detection
    def detect_nan_in_model(model):
        """Check for NaN in model parameters or gradients."""
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in parameter: {name}")
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradient: {name}")

    # Add NaN hooks
    def add_nan_hook(model):
        """Add hooks to detect where NaN first appears."""
        def nan_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    print(f"NaN detected in {module.__class__.__name__}")
                    raise RuntimeError(f"NaN in {module.__class__.__name__}")

        for module in model.modules():
            module.register_forward_hook(nan_hook)

    # Safer training loop
    def safe_training_step(model, batch, optimizer):
        """Training step with NaN detection and recovery."""
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Check input for NaN
        if torch.isnan(input_ids.float()).any():
            print("WARNING: NaN in input data, skipping batch")
            return None

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        # Check loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss: {loss.item()}, skipping batch")
            return None

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        detect_nan_in_model(model)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        return loss.item()
```

### Out of Memory (OOM) Errors

**Symptoms:**
- CUDA out of memory error
- Training crashes during forward or backward pass
- Inconsistent OOM (works sometimes, fails other times)

**Solutions:**

```python
def debug_oom_errors():
    """
    Strategies for handling out-of-memory errors.

    1. Reduce batch size:
       - Simplest solution
       - Use gradient accumulation to maintain effective batch size

    2. Use gradient checkpointing:
       - Reduces activation memory by 40-50%
       - See gradient checkpointing section above

    3. Use mixed precision:
       - BF16 reduces memory by ~40%
       - Essential for large models

    4. Optimize sequence length:
       - Memory scales quadratically with sequence length (attention)
       - Use shorter sequences or sparse attention

    5. Model parallelism:
       - Split model across multiple GPUs
       - See Chapter 16 on distributed training

    6. Offloading:
       - Move optimizer states to CPU
       - Use ZeRO optimizer (DeepSpeed)
    """

    # Memory monitoring
    def monitor_gpu_memory():
        """Monitor GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {allocated:.2f}GB allocated, "
                  f"{reserved:.2f}GB reserved")

    # Clear cache
    def clear_gpu_cache():
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # OOM-safe training
    def oom_safe_training():
        """Training loop that handles OOM gracefully."""
        model = ModelWithCheckpointing(
            vocab_size=50000,
            d_model=4096,
            n_layers=32,
            n_heads=32,
            use_checkpoint=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        for batch in train_dataloader:
            try:
                # Forward pass
                input_ids = batch['input_ids'].to('cuda')
                labels = batch['labels'].to('cuda')

                outputs = model(input_ids, labels=labels)
                loss = outputs['loss']

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Monitor memory
                if step % 100 == 0:
                    monitor_gpu_memory()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: OOM detected, clearing cache and skipping batch")
                    clear_gpu_cache()
                    continue
                else:
                    raise e

    # Memory estimation
    def estimate_memory_requirements(
        num_params: int,
        batch_size: int,
        seq_len: int,
        d_model: int,
        use_mixed_precision: bool = True,
        use_checkpointing: bool = True
    ):
        """
        Estimate memory requirements for training.

        Args:
            num_params: Number of model parameters
            batch_size: Batch size
            seq_len: Sequence length
            d_model: Model dimension
            use_mixed_precision: Whether using BF16/FP16
            use_checkpointing: Whether using gradient checkpointing

        Returns:
            Estimated memory in GB
        """
        bytes_per_param = 2 if use_mixed_precision else 4

        # Model parameters
        param_memory = num_params * bytes_per_param

        # Optimizer states (AdamW: 2x params for momentum and variance)
        optimizer_memory = num_params * 4 * 2  # Always FP32

        # Gradients
        gradient_memory = num_params * bytes_per_param

        # Activations (rough estimate)
        activation_memory = batch_size * seq_len * d_model * bytes_per_param * 20

        if use_checkpointing:
            activation_memory *= 0.5  # Roughly 50% reduction

        total_memory = (param_memory + optimizer_memory +
                       gradient_memory + activation_memory) / 1e9

        print(f"Estimated memory usage:")
        print(f"  Parameters: {param_memory / 1e9:.2f}GB")
        print(f"  Optimizer: {optimizer_memory / 1e9:.2f}GB")
        print(f"  Gradients: {gradient_memory / 1e9:.2f}GB")
        print(f"  Activations: {activation_memory / 1e9:.2f}GB")
        print(f"  Total: {total_memory:.2f}GB")

        return total_memory


# Example usage
def example_memory_estimation():
    """Example memory estimation for 7B model."""
    memory = estimate_memory_requirements(
        num_params=7_000_000_000,  # 7B parameters
        batch_size=4,
        seq_len=2048,
        d_model=4096,
        use_mixed_precision=True,
        use_checkpointing=True
    )
    print(f"\nCan fit on 80GB A100: {memory < 80}")
```

### Slow Training

**Symptoms:**
- Training is slower than expected
- Low GPU utilization
- High training time per batch

**Solutions:**

```python
def debug_slow_training():
    """
    Strategies for improving training speed.

    1. Data loading bottleneck:
       - Check: CPU usage, time spent waiting for data
       - Fix: Increase num_workers in DataLoader
       - Fix: Use faster data format (e.g., memory-mapped files)
       - Fix: Prefetch data to GPU

    2. Mixed precision not enabled:
       - Check: Verify BF16/FP16 is active
       - Fix: Enable autocast and use compatible hardware

    3. Gradient accumulation overhead:
       - Check: Profile gradient accumulation
       - Fix: Optimize accumulation loop

    4. Inefficient operations:
       - Check: Profile with PyTorch profiler
       - Fix: Replace slow operations, fuse operations

    5. Suboptimal hardware utilization:
       - Check: GPU utilization (nvidia-smi)
       - Fix: Increase batch size, use tensor cores
    """

    # Profile training
    def profile_training_step(model, batch, optimizer):
        """Profile a single training step."""
        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            input_ids = batch['input_ids'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Optimize DataLoader
    def optimize_dataloader():
        """Example of optimized DataLoader configuration."""
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster GPU transfer
            prefetch_factor=2,  # Prefetch batches
            persistent_workers=True  # Keep workers alive
        )
        return train_dataloader

    # Measure tokens per second
    def measure_throughput(model, train_dataloader, num_steps=100):
        """Measure training throughput in tokens/second."""
        import time

        model.train()
        torch.cuda.synchronize()
        start_time = time.time()

        total_tokens = 0

        for step, batch in enumerate(train_dataloader):
            if step >= num_steps:
                break

            batch_size, seq_len = batch['input_ids'].shape
            total_tokens += batch_size * seq_len

            # Training step (simplified)
            input_ids = batch['input_ids'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time
        tokens_per_sec = total_tokens / elapsed

        print(f"Throughput: {tokens_per_sec:.0f} tokens/second")
        print(f"Time per step: {elapsed / num_steps:.3f} seconds")

        return tokens_per_sec
```

### Model Not Converging

**Symptoms:**
- Validation loss stops improving while training loss decreases
- Model outputs don't make sense
- High variance in loss

**Solutions:**

```python
def debug_convergence_issues():
    """
    Debugging convergence problems.

    1. Overfitting:
       - Symptom: Train loss << val loss
       - Fix: Increase weight decay, add dropout
       - Fix: Use more training data
       - Fix: Reduce model size

    2. Underfitting:
       - Symptom: Train loss and val loss both high
       - Fix: Increase model size
       - Fix: Train longer
       - Fix: Increase learning rate

    3. Bad initialization:
       - Symptom: Loss doesn't decrease from start
       - Fix: Use proper initialization (Xavier, He, etc.)
       - Fix: Check for zero gradients

    4. Learning rate issues:
       - Symptom: Oscillating loss
       - Fix: Reduce learning rate
       - Fix: Use learning rate schedule

    5. Data distribution mismatch:
       - Symptom: Train and val loss diverge early
       - Fix: Check train/val split
       - Fix: Ensure data is shuffled properly
    """

    # Monitor overfitting
    def detect_overfitting(train_losses, val_losses, threshold=0.5):
        """Detect if model is overfitting."""
        recent_train = sum(train_losses[-10:]) / 10
        recent_val = sum(val_losses[-10:]) / 10

        gap = recent_val - recent_train

        if gap > threshold:
            print(f"WARNING: Possible overfitting detected!")
            print(f"Train loss: {recent_train:.4f}, Val loss: {recent_val:.4f}")
            print(f"Gap: {gap:.4f}")
            return True

        return False

    # Early stopping
    class EarlyStopping:
        """Early stopping to prevent overfitting."""

        def __init__(self, patience=5, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered after {self.counter} epochs")
                    return True
            else:
                self.best_loss = val_loss
                self.counter = 0

            return False
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
