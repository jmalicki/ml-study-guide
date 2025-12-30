# Chapter 21: Direct Preference Optimization (DPO)

## Introduction

Direct Preference Optimization (DPO) is a powerful technique for aligning language models with human preferences without the complexity of reinforcement learning. Introduced by Rafailov et al. in 2023, DPO simplifies the [RLHF](20-rlhf.md) pipeline by directly optimizing the language model using preference data, eliminating the need for a separate reward model and reinforcement learning optimization.

**Key Benefits:**
- No reward model training required
- No RL optimization (PPO, etc.)
- More stable training
- Simpler implementation
- Often better performance

## From RLHF to DPO: Motivation

### The RLHF Pipeline Complexity

As covered in [Chapter 20: RLHF](20-rlhf.md), the traditional RLHF pipeline involves:

1. **Supervised Fine-tuning (SFT)**: Train a model on demonstrations ([Chapter 18: SFT](18-sft.md))
2. **Reward Modeling**: Train a separate reward model on preference data
3. **RL Optimization**: Use PPO or similar to optimize the policy against the reward model

This pipeline has several challenges:
- **Instability**: RL training can be unstable
- **Complexity**: Multiple models and training stages
- **Hyperparameter sensitivity**: KL penalty coefficient, learning rates, etc.
- **Computational cost**: Requires running multiple models during training

### The DPO Insight

DPO makes a key observation: we can derive a closed-form expression for the optimal policy in terms of the reward function, then rearrange this to express the reward in terms of the policy. This allows us to:

1. Skip reward model training entirely
2. Optimize the policy directly using preference data
3. Implicitly maintain the KL constraint from RLHF

## Mathematical Foundation

### The Bradley-Terry Model

Human preferences are typically modeled using the Bradley-Terry model, which states that the probability of preferring response $y_w$ (chosen/winner) over $y_l$ (rejected/loser) given prompt $x$ is:

$$
P(y_w \succ y_l \mid x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))
$$

where $r(x, y)$ is the reward function and $\sigma$ is the sigmoid function.

### The RLHF Objective

In RLHF, we maximize:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)} [r(x, y)] - \beta \mathbb{D}_{\text{KL}}[\pi_\theta(y \mid x) \| \pi_{\text{ref}}(y \mid x)]
$$

where:
- $\pi_\theta$ is our policy (language model)
- $\pi_{\text{ref}}$ is the reference model (typically the SFT model)
- $\beta$ is the KL penalty coefficient
- $\mathbb{D}_{\text{KL}}$ is the KL divergence

### Closed-Form Optimal Policy

The optimal solution to this constrained optimization problem has a closed form:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ is the partition function.

### Reparameterizing the Reward

We can rearrange the above to express the reward in terms of the optimal policy:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

**Key insight**: The partition function $Z(x)$ depends only on $x$, not $y$. When we compute reward differences, it cancels out!

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
$$

### The DPO Objective

Substituting our reward reparameterization into the Bradley-Terry model:

$$
P(y_w \succ y_l \mid x) = \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)
$$

The DPO loss is the negative log-likelihood of the preference data:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

This can be rewritten more compactly as:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right)\right]
$$

## Implementation

### Basic DPO Training Loop

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import numpy as np

class DPOTrainer:
    """
    Direct Preference Optimization (DPO) trainer.

    Based on: "Direct Preference Optimization: Your Language Model is
    Secretly a Reward Model" (Rafailov et al., 2023)
    https://arxiv.org/abs/2305.18290
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
        max_length: int = 512,
    ):
        """
        Args:
            model: The policy model to train
            ref_model: The reference model (frozen, typically the SFT model)
            beta: Temperature parameter controlling KL divergence from reference
            learning_rate: Learning rate for optimization
            max_length: Maximum sequence length
        """
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.max_length = max_length

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Optimizer for the policy model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0
        )

    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of the labels under the model.

        Args:
            model: Language model
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target token ids [batch_size, seq_len] (same as input_ids for causal LM)

        Returns:
            log_probs: Sum of log probabilities for each sequence [batch_size]
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]

            # Shift logits and labels for next-token prediction
            # logits: [batch_size, seq_len-1, vocab_size]
            # labels: [batch_size, seq_len-1]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_attention_mask = attention_mask[:, 1:].contiguous()

            # Compute log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)

            # Gather log probs of the actual tokens
            # [batch_size, seq_len-1]
            per_token_log_probs = torch.gather(
                log_probs,
                dim=2,
                index=shift_labels.unsqueeze(2)
            ).squeeze(2)

            # Mask out padding tokens and sum
            # Only compute log prob where attention_mask and label != -100
            valid_mask = (shift_labels != -100) & (shift_attention_mask == 1)
            per_token_log_probs = per_token_log_probs * valid_mask

            # Sum log probs for each sequence
            sequence_log_probs = per_token_log_probs.sum(dim=1)

            return sequence_log_probs

    def compute_dpo_loss(
        self,
        policy_chosen_log_probs: torch.Tensor,
        policy_rejected_log_probs: torch.Tensor,
        reference_chosen_log_probs: torch.Tensor,
        reference_rejected_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the DPO loss.

        Args:
            policy_chosen_log_probs: Log probs of chosen responses under policy
            policy_rejected_log_probs: Log probs of rejected responses under policy
            reference_chosen_log_probs: Log probs of chosen responses under reference
            reference_rejected_log_probs: Log probs of rejected responses under reference

        Returns:
            loss: The DPO loss (scalar)
            metrics: Dictionary of metrics for logging
        """
        # Compute log ratios
        policy_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs
        reference_log_ratios = reference_chosen_log_probs - reference_rejected_log_probs

        # DPO loss: -log sigmoid(beta * (policy_log_ratio - ref_log_ratio))
        logits = self.beta * (policy_log_ratios - reference_log_ratios)
        loss = -F.logsigmoid(logits).mean()

        # Compute metrics
        with torch.no_grad():
            # Implicit reward (for logging/analysis)
            chosen_rewards = self.beta * (policy_chosen_log_probs - reference_chosen_log_probs)
            rejected_rewards = self.beta * (policy_rejected_log_probs - reference_rejected_log_probs)
            reward_margin = chosen_rewards - rejected_rewards

            # Accuracy: how often does the model prefer the chosen response?
            accuracy = (reward_margin > 0).float().mean()

            metrics = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'chosen_reward_mean': chosen_rewards.mean().item(),
                'rejected_reward_mean': rejected_rewards.mean().item(),
                'reward_margin_mean': reward_margin.mean().item(),
            }

        return loss, metrics

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Dictionary with keys:
                - chosen_input_ids: [batch_size, seq_len]
                - chosen_attention_mask: [batch_size, seq_len]
                - chosen_labels: [batch_size, seq_len]
                - rejected_input_ids: [batch_size, seq_len]
                - rejected_attention_mask: [batch_size, seq_len]
                - rejected_labels: [batch_size, seq_len]

        Returns:
            metrics: Dictionary of metrics from this step
        """
        self.model.train()

        # Get log probs from policy model
        policy_chosen_log_probs = self.get_log_probs(
            self.model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['chosen_labels'],
        )

        policy_rejected_log_probs = self.get_log_probs(
            self.model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['rejected_labels'],
        )

        # Get log probs from reference model (no gradients)
        with torch.no_grad():
            reference_chosen_log_probs = self.get_log_probs(
                self.ref_model,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch['chosen_labels'],
            )

            reference_rejected_log_probs = self.get_log_probs(
                self.ref_model,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch['rejected_labels'],
            )

        # Compute loss
        loss, metrics = self.compute_dpo_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            reference_chosen_log_probs,
            reference_rejected_log_probs,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        return metrics


class PreferenceDataset(Dataset):
    """
    Dataset for preference pairs (chosen vs. rejected responses).
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            data: List of dictionaries with keys 'prompt', 'chosen', 'rejected'
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']

        # Tokenize chosen response
        chosen_text = prompt + chosen
        chosen_encodings = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Tokenize rejected response
        rejected_text = prompt + rejected
        rejected_encodings = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Create labels (same as input_ids for causal LM, -100 for padding)
        chosen_labels = chosen_encodings['input_ids'].clone()
        chosen_labels[chosen_encodings['attention_mask'] == 0] = -100

        rejected_labels = rejected_encodings['input_ids'].clone()
        rejected_labels[rejected_encodings['attention_mask'] == 0] = -100

        return {
            'chosen_input_ids': chosen_encodings['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encodings['attention_mask'].squeeze(0),
            'chosen_labels': chosen_labels.squeeze(0),
            'rejected_input_ids': rejected_encodings['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encodings['attention_mask'].squeeze(0),
            'rejected_labels': rejected_labels.squeeze(0),
        }


# Example usage
def train_dpo_model():
    """
    Example training script for DPO.
    """
    # Initialize models
    model_name = 'gpt2'  # Or your SFT model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Policy model (will be trained)
    policy_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Reference model (frozen copy of SFT model)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create synthetic preference data
    preference_data = [
        {
            'prompt': 'Explain quantum computing in simple terms.',
            'chosen': ' Quantum computing uses quantum mechanics to perform computations that would be impossible for classical computers. It leverages superposition and entanglement to process information in fundamentally different ways.',
            'rejected': ' Quantum computing is really complicated and involves lots of physics stuff.',
        },
        {
            'prompt': 'What is the capital of France?',
            'chosen': ' The capital of France is Paris.',
            'rejected': ' I think it might be London or maybe Berlin.',
        },
        # Add more examples...
    ]

    # Create dataset and dataloader
    dataset = PreferenceDataset(preference_data, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize trainer
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        beta=0.1,
        learning_rate=1e-6,
    )

    # Training loop
    num_epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_model.to(device)
    ref_model.to(device)

    for epoch in range(num_epochs):
        epoch_metrics = []

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Training step
            metrics = trainer.train_step(batch)
            epoch_metrics.append(metrics)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Loss = {metrics['loss']:.4f}, "
                      f"Accuracy = {metrics['accuracy']:.4f}")

        # Compute epoch averages
        avg_metrics = {
            key: np.mean([m[key] for m in epoch_metrics])
            for key in epoch_metrics[0].keys()
        }
        print(f"\nEpoch {epoch} Summary: {avg_metrics}\n")

    return policy_model, trainer


if __name__ == "__main__":
    # Run example training
    trained_model, trainer = train_dpo_model()
```

### Label Masking for Prompt Tokens

In practice, we often want to only compute the loss on the completion tokens, not the prompt tokens. Here's an improved version:

```python
def create_labels_with_prompt_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """
    Create labels that mask out prompt tokens.

    Args:
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        prompt_length: Number of prompt tokens to mask

    Returns:
        labels: [batch_size, seq_len] with -100 for prompt and padding
    """
    labels = input_ids.clone()

    # Mask prompt tokens
    labels[:, :prompt_length] = -100

    # Mask padding tokens
    labels[attention_mask == 0] = -100

    return labels


class ImprovedPreferenceDataset(Dataset):
    """
    Improved dataset that properly masks prompt tokens.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']

        # Tokenize prompt separately to get its length
        prompt_encodings = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            return_tensors='pt',
        )
        prompt_length = prompt_encodings['input_ids'].shape[1]

        # Tokenize full sequences
        chosen_text = prompt + chosen
        chosen_encodings = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        rejected_text = prompt + rejected
        rejected_encodings = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # Create labels with prompt masking
        chosen_labels = create_labels_with_prompt_masking(
            chosen_encodings['input_ids'],
            chosen_encodings['attention_mask'],
            prompt_length,
        )

        rejected_labels = create_labels_with_prompt_masking(
            rejected_encodings['input_ids'],
            rejected_encodings['attention_mask'],
            prompt_length,
        )

        return {
            'chosen_input_ids': chosen_encodings['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encodings['attention_mask'].squeeze(0),
            'chosen_labels': chosen_labels.squeeze(0),
            'rejected_input_ids': rejected_encodings['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encodings['attention_mask'].squeeze(0),
            'rejected_labels': rejected_labels.squeeze(0),
        }
```

## DPO Variants

### Identity Preference Optimization (IPO)

IPO addresses a potential issue with DPO: the loss can be minimized by the model assigning very low probability to rejected responses, which can hurt generation quality. IPO replaces the sigmoid with a squared error:

$$
\mathcal{L}_{\text{IPO}}(\pi_\theta; \pi_{\text{ref}}) = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\left(\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \frac{1}{2\beta}\right)^2\right]
$$

**Paper**: [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) (Azar et al., 2023)

```python
def compute_ipo_loss(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    reference_chosen_log_probs: torch.Tensor,
    reference_rejected_log_probs: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the IPO (Identity Preference Optimization) loss.

    IPO uses a squared error instead of log-sigmoid, which can
    prevent the model from assigning very low probabilities to
    rejected responses.
    """
    # Compute log ratios
    policy_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs
    reference_log_ratios = reference_chosen_log_probs - reference_rejected_log_probs

    # IPO loss: (log_ratio_diff - 1/(2*beta))^2
    log_ratio_diff = policy_log_ratios - reference_log_ratios
    loss = ((log_ratio_diff - 1.0 / (2 * beta)) ** 2).mean()

    # Metrics
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_log_probs - reference_chosen_log_probs)
        rejected_rewards = beta * (policy_rejected_log_probs - reference_rejected_log_probs)
        reward_margin = chosen_rewards - rejected_rewards
        accuracy = (reward_margin > 0).float().mean()

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'rejected_reward_mean': rejected_rewards.mean().item(),
            'reward_margin_mean': reward_margin.mean().item(),
        }

    return loss, metrics
```

### Kahneman-Tversky Optimization (KTO)

KTO is designed for scenarios where you have unpaired preference data - i.e., examples labeled as "good" or "bad" but not explicitly compared. This is useful when you have thumbs up/down feedback but not pairwise comparisons.

$$
\mathcal{L}_{\text{KTO}}(\pi_\theta; \pi_{\text{ref}}) = \mathbb{E}_{x, y \sim \mathcal{D}} \left[w(y) \left(1 - \sigma\left(\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - z_{\text{ref}}\right)\right)\right]
$$

where $w(y) = \lambda_D$ if $y$ is desirable, $\lambda_U$ otherwise, and $z_{\text{ref}}$ is a reference point.

**Paper**: [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) (Ethayarajh et al., 2024)

```python
def compute_kto_loss(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    labels: torch.Tensor,  # 1 for desirable, 0 for undesirable
    beta: float = 0.1,
    lambda_d: float = 1.0,
    lambda_u: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the KTO (Kahneman-Tversky Optimization) loss.

    KTO works with unpaired preference data (thumbs up/down)
    rather than pairwise comparisons.

    Args:
        policy_log_probs: Log probs under policy [batch_size]
        reference_log_probs: Log probs under reference [batch_size]
        labels: 1 for desirable, 0 for undesirable [batch_size]
        beta: Temperature parameter
        lambda_d: Weight for desirable examples
        lambda_u: Weight for undesirable examples
    """
    # Compute KL term
    kl = policy_log_probs - reference_log_probs

    # Compute reference point (mean KL across batch)
    z_ref = kl.mean().detach()

    # Compute weights
    weights = torch.where(labels == 1, lambda_d, lambda_u)

    # KTO loss: weight * (1 - sigma(beta * (kl - z_ref)))
    # For desirable: want to maximize kl (move away from ref toward better)
    # For undesirable: want to minimize kl (stay close to ref or move away from worse)
    loss = weights * (1 - torch.sigmoid(beta * (kl - z_ref)))

    return loss.mean(), {
        'loss': loss.mean().item(),
        'kl_mean': kl.mean().item(),
        'z_ref': z_ref.item(),
    }
```

### Odds Ratio Preference Optimization (ORPO)

ORPO combines SFT and preference optimization in a single stage, eliminating the need for a separate reference model. It adds a penalty term that increases the odds ratio between chosen and rejected responses.

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}}
$$

where:
$$
\mathcal{L}_{\text{OR}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\log \frac{\text{odds}_\theta(y_w \mid x)}{\text{odds}_\theta(y_l \mid x)}\right)\right]
$$

and $\text{odds}_\theta(y \mid x) = \frac{\pi_\theta(y \mid x)}{1 - \pi_\theta(y \mid x)}$

**Paper**: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) (Hong et al., 2024)

```python
def compute_orpo_loss(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    sft_loss: torch.Tensor,
    lambda_or: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the ORPO (Odds Ratio Preference Optimization) loss.

    ORPO combines SFT loss with an odds ratio term, eliminating
    the need for a reference model.

    Args:
        policy_chosen_log_probs: Sum of log probs for chosen [batch_size]
        policy_rejected_log_probs: Sum of log probs for rejected [batch_size]
        chosen_lengths: Number of tokens in chosen responses [batch_size]
        rejected_lengths: Number of tokens in rejected responses [batch_size]
        sft_loss: Standard SFT loss (negative log likelihood)
        lambda_or: Weight for odds ratio term
    """
    # Compute average log probs (to normalize by length)
    chosen_avg_log_prob = policy_chosen_log_probs / chosen_lengths
    rejected_avg_log_prob = policy_rejected_log_probs / rejected_lengths

    # Compute log odds: log(p / (1 - p)) = log(p) - log(1 - p)
    # For small p, log(1 - p) ≈ log(exp(log(1-p))) = log(1 - exp(log_p))
    # We approximate: log_odds ≈ log_prob - log(1 - exp(log_prob))

    # Alternative formulation: use the ratio directly
    # log(odds_w / odds_l) = log(p_w/(1-p_w)) - log(p_l/(1-p_l))
    log_odds_ratio = chosen_avg_log_prob - rejected_avg_log_prob

    # Odds ratio loss
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    # Total loss
    total_loss = sft_loss + lambda_or * or_loss

    return total_loss, {
        'total_loss': total_loss.item(),
        'sft_loss': sft_loss.item(),
        'or_loss': or_loss.item(),
    }
```

### Simple Preference Optimization (SimPO)

SimPO simplifies DPO by:
1. Removing the reference model (like ORPO)
2. Using length-normalized rewards
3. Adding a target reward margin

$$
\mathcal{L}_{\text{SimPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \left(\frac{\log \pi_\theta(y_w \mid x)}{|y_w|} - \frac{\log \pi_\theta(y_l \mid x)}{|y_l|}\right) - \gamma\right)\right]
$$

where $|y|$ is the length of sequence $y$ and $\gamma$ is a target reward margin.

**Paper**: [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) (Meng et al., 2024)

```python
def compute_simpo_loss(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the SimPO (Simple Preference Optimization) loss.

    SimPO uses length-normalized rewards without a reference model
    and includes a target reward margin.

    Args:
        policy_chosen_log_probs: Sum of log probs for chosen [batch_size]
        policy_rejected_log_probs: Sum of log probs for rejected [batch_size]
        chosen_lengths: Number of tokens in chosen responses [batch_size]
        rejected_lengths: Number of tokens in rejected responses [batch_size]
        beta: Temperature parameter (typically higher than DPO, e.g., 2.0)
        gamma: Target reward margin
    """
    # Length-normalized log probs (average per token)
    chosen_avg_log_prob = policy_chosen_log_probs / chosen_lengths
    rejected_avg_log_prob = policy_rejected_log_probs / rejected_lengths

    # SimPO logits with target margin
    logits = beta * (chosen_avg_log_prob - rejected_avg_log_prob) - gamma

    # Loss
    loss = -F.logsigmoid(logits).mean()

    # Metrics
    with torch.no_grad():
        accuracy = (logits > 0).float().mean()
        reward_margin = chosen_avg_log_prob - rejected_avg_log_prob

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_avg_log_prob': chosen_avg_log_prob.mean().item(),
            'rejected_avg_log_prob': rejected_avg_log_prob.mean().item(),
            'reward_margin': reward_margin.mean().item(),
        }

    return loss, metrics
```

## Comparison: DPO vs RLHF

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Training Stages** | 3 (SFT, Reward Model, RL) | 1 (Direct optimization) |
| **Models Required** | 3-4 (SFT, Reward, Policy, Value) | 2 (Policy, Reference) |
| **Stability** | Can be unstable (RL training) | More stable (supervised learning) |
| **Hyperparameters** | Many (RL-specific) | Fewer (mainly $\beta$) |
| **Computational Cost** | High (multiple models in loop) | Lower (single training loop) |
| **Sample Efficiency** | Lower | Higher |
| **Flexibility** | High (can optimize complex rewards) | Limited to preference data |
| **Implementation** | Complex | Simpler |
| **Performance** | Strong | Often matches or exceeds RLHF |

### When to Use Each

**Use RLHF when:**
- You have complex, compositional reward functions
- You need to optimize for metrics beyond pairwise preferences
- You have extensive computational resources
- You need fine-grained control over the optimization process

**Use DPO when:**
- You have high-quality preference data
- You want simpler, more stable training
- You have limited computational resources
- You want faster iteration cycles
- Your goal is alignment with human preferences

## Practical Considerations

### Choosing Beta

The $\beta$ parameter controls the trade-off between:
- **High $\beta$**: Stay close to reference model (more conservative)
- **Low $\beta$**: Deviate more from reference (more aggressive optimization)

Typical values: $\beta \in [0.1, 0.5]$

```python
def analyze_beta_sensitivity(
    trainer: DPOTrainer,
    batch: Dict[str, torch.Tensor],
    beta_values: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
):
    """
    Analyze how different beta values affect the loss and metrics.
    """
    results = {}
    original_beta = trainer.beta

    for beta in beta_values:
        trainer.beta = beta
        with torch.no_grad():
            # Get log probs
            policy_chosen_log_probs = trainer.get_log_probs(
                trainer.model,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch['chosen_labels'],
            )
            policy_rejected_log_probs = trainer.get_log_probs(
                trainer.model,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch['rejected_labels'],
            )
            reference_chosen_log_probs = trainer.get_log_probs(
                trainer.ref_model,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch['chosen_labels'],
            )
            reference_rejected_log_probs = trainer.get_log_probs(
                trainer.ref_model,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch['rejected_labels'],
            )

            # Compute metrics
            _, metrics = trainer.compute_dpo_loss(
                policy_chosen_log_probs,
                policy_rejected_log_probs,
                reference_chosen_log_probs,
                reference_rejected_log_probs,
            )
            results[beta] = metrics

    # Restore original beta
    trainer.beta = original_beta

    # Print results
    print(f"{'Beta':<10} {'Loss':<10} {'Accuracy':<10} {'Reward Margin':<15}")
    print("-" * 50)
    for beta, metrics in results.items():
        print(f"{beta:<10.2f} {metrics['loss']:<10.4f} "
              f"{metrics['accuracy']:<10.4f} {metrics['reward_margin_mean']:<15.4f}")

    return results
```

### Data Quality

DPO's performance heavily depends on preference data quality:

1. **Clear preferences**: Chosen should be clearly better than rejected
2. **Diverse examples**: Cover wide range of prompts and response types
3. **Consistent labels**: Multiple annotators should agree
4. **Sufficient quantity**: Typically need thousands of examples

```python
def analyze_preference_data_quality(
    dataset: PreferenceDataset,
    model: nn.Module,
    tokenizer,
    num_samples: int = 100,
):
    """
    Analyze the quality of preference data using a trained model.
    """
    model.eval()
    device = next(model.parameters()).device

    stats = {
        'win_rates': [],
        'log_prob_differences': [],
        'length_ratios': [],
    }

    for idx in range(min(num_samples, len(dataset))):
        item = dataset[idx]

        # Move to device
        chosen_input_ids = item['chosen_input_ids'].unsqueeze(0).to(device)
        chosen_attention_mask = item['chosen_attention_mask'].unsqueeze(0).to(device)
        rejected_input_ids = item['rejected_input_ids'].unsqueeze(0).to(device)
        rejected_attention_mask = item['rejected_attention_mask'].unsqueeze(0).to(device)

        with torch.no_grad():
            # Get log probs
            chosen_log_prob = trainer.get_log_probs(
                model,
                chosen_input_ids,
                chosen_attention_mask,
                item['chosen_labels'].unsqueeze(0).to(device),
            )
            rejected_log_prob = trainer.get_log_probs(
                model,
                rejected_input_ids,
                rejected_attention_mask,
                item['rejected_labels'].unsqueeze(0).to(device),
            )

            # Compute statistics
            log_prob_diff = (chosen_log_prob - rejected_log_prob).item()
            stats['log_prob_differences'].append(log_prob_diff)
            stats['win_rates'].append(1.0 if log_prob_diff > 0 else 0.0)

            chosen_len = (item['chosen_attention_mask'] == 1).sum().item()
            rejected_len = (item['rejected_attention_mask'] == 1).sum().item()
            stats['length_ratios'].append(chosen_len / max(rejected_len, 1))

    # Print statistics
    print(f"Model Win Rate: {np.mean(stats['win_rates']):.2%}")
    print(f"Average Log Prob Difference: {np.mean(stats['log_prob_differences']):.4f}")
    print(f"Average Length Ratio (chosen/rejected): {np.mean(stats['length_ratios']):.2f}")

    return stats
```

### Preventing Reward Hacking

Even without an explicit reward model, DPO can exhibit reward hacking behaviors:

1. **Length exploitation**: Model generates very long responses
2. **Mode collapse**: Model generates only safe, generic responses
3. **Distribution shift**: Model deviates too far from reference

**Mitigation strategies:**

```python
def compute_dpo_loss_with_regularization(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    reference_chosen_log_probs: torch.Tensor,
    reference_rejected_log_probs: torch.Tensor,
    beta: float = 0.1,
    kl_penalty: float = 0.0,
    length_penalty: float = 0.0,
    chosen_lengths: torch.Tensor = None,
    rejected_lengths: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    DPO loss with additional regularization terms.
    """
    # Standard DPO loss
    policy_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs
    reference_log_ratios = reference_chosen_log_probs - reference_rejected_log_probs
    logits = beta * (policy_log_ratios - reference_log_ratios)
    dpo_loss = -F.logsigmoid(logits).mean()

    # KL regularization: penalize deviation from reference
    kl_loss = 0.0
    if kl_penalty > 0:
        kl_chosen = policy_chosen_log_probs - reference_chosen_log_probs
        kl_rejected = policy_rejected_log_probs - reference_rejected_log_probs
        kl_loss = kl_penalty * (kl_chosen.mean() + kl_rejected.mean())

    # Length regularization: penalize very long generations
    length_loss = 0.0
    if length_penalty > 0 and chosen_lengths is not None:
        length_loss = length_penalty * chosen_lengths.float().mean()

    # Total loss
    total_loss = dpo_loss + kl_loss + length_loss

    metrics = {
        'loss': total_loss.item(),
        'dpo_loss': dpo_loss.item(),
        'kl_loss': kl_loss if isinstance(kl_loss, float) else kl_loss.item(),
        'length_loss': length_loss if isinstance(length_loss, float) else length_loss.item(),
    }

    return total_loss, metrics
```

## Advanced Topics

### Online DPO

Standard DPO uses a fixed dataset. Online DPO generates new preference pairs during training:

1. Sample responses from current policy
2. Evaluate with reward model or human feedback
3. Add to preference dataset
4. Continue training

This can improve sample efficiency and help the model explore better responses.

### Multi-Objective DPO

Optimize for multiple objectives simultaneously (e.g., helpfulness AND harmlessness):

$$
\mathcal{L}_{\text{multi}} = \sum_{i=1}^{k} \alpha_i \mathcal{L}_{\text{DPO}}^{(i)}
$$

where each $\mathcal{L}_{\text{DPO}}^{(i)}$ uses preference data for objective $i$.

### Conditional DPO

Condition the optimization on different personas or styles:

$$
P(y_w \succ y_l \mid x, c) = \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x, c)}{\pi_{\text{ref}}(y_w \mid x, c)} - \beta \log \frac{\pi_\theta(y_l \mid x, c)}{\pi_{\text{ref}}(y_l \mid x, c)}\right)
$$

where $c$ is a conditioning variable (e.g., "be concise" vs "be detailed").

## Connection to Alignment

DPO is a key technique in the broader landscape of [AI safety and alignment](22-safety-alignment.md):

1. **Value Alignment**: Ensures model behavior aligns with human preferences
2. **Scalable Oversight**: Can incorporate feedback from various sources
3. **Robustness**: More stable than RL-based approaches
4. **Interpretability**: Preference data is more interpretable than reward scores

## Exercises

### Exercise 1: Implement Basic DPO

Implement the DPO training loop for a small GPT-2 model on a synthetic preference dataset.

**Tasks:**
1. Create 100 preference pairs (prompt, chosen, rejected)
2. Train for 5 epochs
3. Track loss, accuracy, and reward margins
4. Generate samples before and after training to see the difference

### Exercise 2: Compare DPO Variants

Implement and compare DPO, IPO, and SimPO on the same dataset.

**Questions to answer:**
1. Which achieves the best accuracy?
2. Which is most stable during training?
3. How do the final model outputs differ?
4. What are the computational trade-offs?

### Exercise 3: Beta Sensitivity Analysis

**Tasks:**
1. Train models with $\beta \in \{0.01, 0.1, 0.5, 1.0\}$
2. Measure KL divergence from reference model
3. Evaluate win rate on a held-out test set
4. Plot the trade-off between alignment and diversity

### Exercise 4: Data Quality Impact

**Tasks:**
1. Create three datasets with different quality levels:
   - High quality: clear preferences, diverse prompts
   - Medium quality: some ambiguous pairs
   - Low quality: random or reversed labels
2. Train DPO models on each
3. Compare final performance
4. Analyze what makes preference data "good"

### Exercise 5: Reward Hacking Detection

**Tasks:**
1. Train a DPO model that might exhibit length hacking
2. Implement metrics to detect:
   - Average response length over training
   - Diversity of responses (entropy)
   - KL divergence from reference
3. Add regularization to prevent hacking
4. Compare before/after behavior

### Exercise 6: From RLHF to DPO

**Tasks:**
1. Implement a simple reward model on preference data
2. Train using the reward model (RLHF-style)
3. Train using DPO on the same data
4. Compare:
   - Training time
   - Memory usage
   - Final performance
   - Stability

### Exercise 7: Conditional Preference Optimization

**Tasks:**
1. Extend DPO to support conditioning on style (e.g., "concise" vs "detailed")
2. Create preference data with style annotations
3. Train a conditional model
4. Test that the model follows style instructions

## Summary

Direct Preference Optimization (DPO) represents a major simplification of the alignment pipeline:

1. **Core Idea**: Directly optimize policy on preference data without reward modeling or RL
2. **Mathematical Foundation**: Derives from the closed-form solution to the RLHF objective
3. **Practical Benefits**: Simpler, more stable, and often more effective than RLHF
4. **Variants**: IPO, KTO, ORPO, SimPO offer different trade-offs
5. **Considerations**: Requires high-quality preference data, careful $\beta$ tuning

DPO has become the dominant approach for aligning modern LLMs, used in models like Claude, GPT-4, and Llama 3. Its simplicity and effectiveness make it an essential technique for ML practitioners working on language model alignment.

## References

1. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**
   Rafailov et al., 2023
   [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)

2. **A General Theoretical Paradigm to Understand Learning from Human Preferences**
   Azar et al., 2023
   [https://arxiv.org/abs/2310.12036](https://arxiv.org/abs/2310.12036)

3. **KTO: Model Alignment as Prospect Theoretic Optimization**
   Ethayarajh et al., 2024
   [https://arxiv.org/abs/2402.01306](https://arxiv.org/abs/2402.01306)

4. **ORPO: Monolithic Preference Optimization without Reference Model**
   Hong et al., 2024
   [https://arxiv.org/abs/2403.07691](https://arxiv.org/abs/2403.07691)

5. **SimPO: Simple Preference Optimization with a Reference-Free Reward**
   Meng et al., 2024
   [https://arxiv.org/abs/2405.14734](https://arxiv.org/abs/2405.14734)

6. **Training language models to follow instructions with human feedback**
   Ouyang et al., 2022
   [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

## Next Steps

- [Chapter 22: Safety and Alignment Techniques](22-safety-alignment.md): Explore broader alignment methods
- [Chapter 20: RLHF](20-rlhf.md): Review the method DPO simplifies
- [Chapter 18: Supervised Fine-tuning](18-sft.md): Understand the foundation for preference optimization
