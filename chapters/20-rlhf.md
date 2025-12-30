# Chapter 20: Reinforcement Learning from Human Feedback (RLHF)

Reinforcement Learning from Human Feedback (RLHF) is the technique that transformed base language models into helpful, harmless assistants like ChatGPT and Claude. This chapter covers the complete RLHF pipeline, from collecting human preferences to training models with PPO while maintaining alignment with the original supervised fine-tuned model.

## Table of Contents

1. [Overview](#overview)
2. [The RLHF Pipeline](#the-rlhf-pipeline)
3. [Reward Modeling](#reward-modeling)
4. [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
5. [KL Divergence Constraints](#kl-divergence-constraints)
6. [Complete Implementation](#complete-implementation)
7. [Practical Considerations](#practical-considerations)
8. [Exercises](#exercises)

---

## Overview

After supervised fine-tuning (see [Supervised Fine-tuning (SFT)](18-sft.md)), a language model can follow instructions, but it may not produce outputs that align with human preferences regarding helpfulness, harmlessness, and honesty. RLHF solves this by:

1. **Reward Modeling**: Training a model to predict human preferences
2. **RL Fine-tuning**: Using reinforcement learning (PPO) to maximize the reward while staying close to the SFT model

The technique was popularized by OpenAI's InstructGPT and Anthropic's Constitutional AI work, though simpler alternatives like DPO (see [Direct Preference Optimization (DPO)](21-dpo.md)) have since emerged.

**Key Papers:**
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, OpenAI, 2022)
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) (Stiennon et al., 2020)
- [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862) (Anthropic, 2022)

---

## The RLHF Pipeline

The complete RLHF pipeline consists of three stages:

```
┌─────────────────┐
│  Base LM        │
│  Pretraining    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Supervised     │ ← Instruction datasets
│  Fine-tuning    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Collect Human  │ ← Generate pairs, humans rank
│  Preferences    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Train Reward   │ ← Bradley-Terry model
│  Model          │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  RL Fine-tuning │ ← PPO with KL constraint
│  (PPO)          │
└─────────────────┘
```

**Stage 1: Supervised Fine-tuning (SFT)**
- Train on high-quality (prompt, response) pairs
- Creates the reference model $\pi_{\text{ref}}$
- See [Chapter 18](18-sft.md) for details

**Stage 2: Reward Model Training**
- Collect human preferences on model outputs
- Train a model to predict which response humans prefer
- Creates the reward model $r_\phi(x, y)$

**Stage 3: RL Fine-tuning**
- Use PPO to optimize the policy $\pi_\theta$ to maximize reward
- Add KL penalty to stay close to $\pi_{\text{ref}}$
- Prevents model from exploiting reward model weaknesses

---

## Reward Modeling

### The Preference Dataset

Instead of rating responses on an absolute scale, we collect **pairwise preferences**. Given a prompt $x$, the model generates two responses $y_w$ (winner) and $y_l$ (loser), and humans indicate which they prefer.

**Dataset structure:**
```python
{
    "prompt": "Explain quantum computing",
    "chosen": "Quantum computing uses quantum bits...",    # y_w
    "rejected": "Quantum is when computers are fast..."   # y_l
}
```

### Bradley-Terry Model

We model the probability that humans prefer response $y_w$ over $y_l$ using the **Bradley-Terry model**:

$$
P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

where:
- $r_\phi(x, y)$ is the scalar reward for prompt $x$ and response $y$
- $\sigma$ is the sigmoid function
- $\succ$ denotes preference

### Loss Function

We train the reward model to maximize the log-likelihood of the observed preferences:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

This is equivalent to binary cross-entropy loss.

### Reward Model Architecture

The reward model is typically initialized from the SFT model and modified to output a scalar reward:

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class RewardModel(nn.Module):
    """
    Reward model for RLHF.
    Takes a (prompt, response) pair and outputs a scalar reward.
    """
    def __init__(self, base_model_name: str, dropout: float = 0.1):
        super().__init__()
        # Load pretrained LM (typically the SFT model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Get hidden dimension
        config = self.model.config
        hidden_size = config.hidden_size

        # Replace LM head with reward head
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1, bias=False)
        )

        # Remove the original LM head to save memory
        self.model.lm_head = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            rewards: [batch_size] scalar rewards
        """
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get last hidden state: [batch_size, seq_len, hidden_size]
        hidden_states = outputs.hidden_states[-1]

        # Get the last token's hidden state (end of sequence)
        # We use attention_mask to find the last real token
        sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
        batch_size = hidden_states.shape[0]

        # Gather last token hidden states
        last_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ]  # [batch_size, hidden_size]

        # Get scalar reward
        rewards = self.reward_head(last_hidden_states).squeeze(-1)  # [batch_size]

        return rewards


def compute_reward_loss(
    reward_model: RewardModel,
    chosen_input_ids: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Bradley-Terry preference loss.

    Args:
        reward_model: The reward model
        chosen_input_ids: [batch_size, seq_len] for chosen responses
        chosen_attention_mask: [batch_size, seq_len]
        rejected_input_ids: [batch_size, seq_len] for rejected responses
        rejected_attention_mask: [batch_size, seq_len]

    Returns:
        loss: scalar loss
    """
    # Compute rewards for chosen and rejected
    r_chosen = reward_model(chosen_input_ids, chosen_attention_mask)
    r_rejected = reward_model(rejected_input_ids, rejected_attention_mask)

    # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
    # This is equivalent to binary cross-entropy
    loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()

    return loss


# Training loop for reward model
def train_reward_model(
    reward_model: RewardModel,
    train_dataloader,
    optimizer,
    num_epochs: int = 1,
    device: str = "cuda"
):
    """
    Train the reward model on preference data.
    """
    reward_model.to(device)
    reward_model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            # Move to device
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            # Compute loss
            loss = compute_reward_loss(
                reward_model,
                chosen_input_ids,
                chosen_attention_mask,
                rejected_input_ids,
                rejected_attention_mask
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

### Practical Tips for Reward Modeling

1. **Normalize rewards**: Subtract mean and divide by std to stabilize RL training
2. **Architecture**: Use the SFT model as initialization for better alignment
3. **Dataset size**: Typically 10k-100k preference pairs
4. **Accuracy**: A good reward model achieves 65-75% accuracy on held-out preferences

---

## Proximal Policy Optimization (PPO)

PPO is a policy gradient RL algorithm that has become the standard for RLHF. It's more stable than vanilla policy gradient methods and doesn't require complex hyperparameter tuning like TRPO.

### RL Formulation

We formulate language generation as an RL problem:
- **State** $s$: The prompt $x$ (or prompt + partial response)
- **Action** $a$: The next token to generate
- **Policy** $\pi_\theta(a|s)$: The language model's token probability distribution
- **Reward** $r$: From the reward model (typically given at the end of sequence)

### The RLHF Objective

The goal is to maximize expected reward while staying close to the reference model:

$$
\mathcal{J}(\theta) = \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta(\cdot|x) || \pi_{\text{ref}}(\cdot|x)) \right]
$$

where:
- $r_\phi(x, y)$ is the reward from the reward model
- $\beta$ is the KL penalty coefficient (typically 0.01-0.1)
- $D_{\text{KL}}$ is the KL divergence (see [KL Divergence Constraints](#kl-divergence-constraints))
- $\pi_{\text{ref}}$ is the frozen SFT model

### PPO Clipped Objective

PPO optimizes a clipped surrogate objective to prevent too large policy updates:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ is the probability ratio
- $\hat{A}_t$ is the advantage estimate
- $\epsilon$ is the clipping parameter (typically 0.2)

### Advantage Estimation

The advantage $A(s, a)$ measures how much better action $a$ is compared to the average:

$$
A(s, a) = Q(s, a) - V(s)
$$

We use Generalized Advantage Estimation (GAE) for lower variance:

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

### Value Network

We need a value function $V_\psi(s)$ to compute advantages. This is typically a copy of the policy model with a value head:

```python
class ValueHead(nn.Module):
    """Value head for estimating state values in PPO."""
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            values: [batch_size, seq_len]
        """
        return self.value_head(hidden_states).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Combined policy (actor) and value function (critic) model for PPO.
    """
    def __init__(self, base_model_name: str):
        super().__init__()
        self.policy = AutoModelForCausalLM.from_pretrained(base_model_name)
        config = self.policy.config
        self.value_head = ValueHead(config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_logits: bool = True
    ):
        """
        Forward pass returning both policy logits and value estimates.

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            values: [batch_size, seq_len]
        """
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        logits = outputs.logits if return_logits else None
        hidden_states = outputs.hidden_states[-1]
        values = self.value_head(hidden_states)

        return logits, values
```

### PPO Implementation

```python
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

def compute_advantages_and_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns.

    Args:
        rewards: [batch_size, seq_len] rewards at each timestep
        values: [batch_size, seq_len] value estimates
        gamma: discount factor
        lambda_: GAE lambda parameter

    Returns:
        advantages: [batch_size, seq_len]
        returns: [batch_size, seq_len]
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Compute advantages using GAE
    gae = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lambda_ * gae
        advantages[:, t] = gae

    # Returns are advantages + values
    returns = advantages + values

    return advantages, returns


def compute_ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2
) -> torch.Tensor:
    """
    Compute PPO clipped objective.

    Args:
        old_log_probs: [batch_size, seq_len] log probs from old policy
        new_log_probs: [batch_size, seq_len] log probs from current policy
        advantages: [batch_size, seq_len] advantage estimates
        clip_epsilon: clipping parameter

    Returns:
        loss: scalar PPO loss
    """
    # Compute probability ratio
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Compute clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    # Take minimum and negate (we want to maximize)
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    clip_epsilon: float = 0.2
) -> torch.Tensor:
    """
    Compute value function loss with clipping.

    Args:
        values: [batch_size, seq_len] current value estimates
        returns: [batch_size, seq_len] computed returns
        old_values: [batch_size, seq_len] old value estimates
        clip_epsilon: clipping parameter

    Returns:
        loss: scalar value loss
    """
    # Clip value updates
    values_clipped = old_values + torch.clamp(
        values - old_values,
        -clip_epsilon,
        clip_epsilon
    )

    # Compute losses
    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2

    # Take maximum
    value_loss = torch.max(vf_loss1, vf_loss2).mean()

    return value_loss
```

---

## KL Divergence Constraints

The KL divergence constraint is crucial for RLHF. Without it, the policy can exploit the reward model by generating adversarial outputs that score high rewards but are nonsensical.

### Per-Token KL Divergence

For discrete distributions (language models), the KL divergence is:

$$
D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) = \sum_{a} \pi_\theta(a|s) \log \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}
$$

For a generated sequence $y = (y_1, \ldots, y_T)$, we compute the average per-token KL:

$$
D_{\text{KL}}^{\text{avg}} = \frac{1}{T} \sum_{t=1}^{T} D_{\text{KL}}(\pi_\theta(\cdot|x, y_{<t}) || \pi_{\text{ref}}(\cdot|x, y_{<t}))
$$

### Implementation

```python
def compute_kl_divergence(
    logits_new: torch.Tensor,
    logits_ref: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-token KL divergence between new policy and reference policy.

    Args:
        logits_new: [batch_size, seq_len, vocab_size] from current policy
        logits_ref: [batch_size, seq_len, vocab_size] from reference policy
        attention_mask: [batch_size, seq_len]

    Returns:
        kl_div: [batch_size] average per-token KL divergence
    """
    # Convert logits to log probabilities
    log_probs_new = F.log_softmax(logits_new, dim=-1)
    log_probs_ref = F.log_softmax(logits_ref, dim=-1)

    # Compute KL divergence: sum over vocabulary
    # KL(new || ref) = sum_a p_new(a) * (log p_new(a) - log p_ref(a))
    probs_new = torch.exp(log_probs_new)
    kl = (probs_new * (log_probs_new - log_probs_ref)).sum(dim=-1)  # [batch_size, seq_len]

    # Apply attention mask and average
    kl = kl * attention_mask
    kl_avg = kl.sum(dim=1) / attention_mask.sum(dim=1)  # [batch_size]

    return kl_avg


def compute_rlhf_reward(
    reward_model_score: torch.Tensor,
    kl_divergence: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    Compute the final RLHF reward with KL penalty.

    Args:
        reward_model_score: [batch_size] rewards from reward model
        kl_divergence: [batch_size] KL divergence from reference
        beta: KL penalty coefficient

    Returns:
        reward: [batch_size] final reward
    """
    return reward_model_score - beta * kl_divergence
```

### Adaptive KL Penalty

Instead of a fixed $\beta$, some implementations use adaptive KL control:

$$
\beta_{t+1} = \begin{cases}
\beta_t / \alpha & \text{if } D_{\text{KL}} < D_{\text{target}} - \epsilon \\
\beta_t \times \alpha & \text{if } D_{\text{KL}} > D_{\text{target}} + \epsilon \\
\beta_t & \text{otherwise}
\end{cases}
$$

This keeps the KL divergence close to a target value (typically 5-10 nats).

```python
class AdaptiveKLController:
    """Adaptive KL penalty controller."""
    def __init__(
        self,
        init_beta: float = 0.1,
        target_kl: float = 6.0,
        horizon: int = 10000,
        alpha: float = 1.5
    ):
        self.beta = init_beta
        self.target_kl = target_kl
        self.horizon = horizon
        self.alpha = alpha

    def update(self, kl_divergence: float):
        """Update beta based on observed KL divergence."""
        proportional_error = kl_divergence / self.target_kl - 1

        # Multiplicative update
        multiplier = 1 + proportional_error * (self.alpha - 1)
        self.beta *= multiplier

        # Clip to reasonable range
        self.beta = max(0.001, min(self.beta, 1.0))

    def get_beta(self) -> float:
        return self.beta
```

---

## Complete Implementation

Let's put it all together into a complete RLHF training loop.

### PPO Trainer

```python
import torch
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

class PPOTrainer:
    """
    PPO trainer for RLHF.
    """
    def __init__(
        self,
        actor_critic: ActorCritic,
        ref_model: AutoModelForCausalLM,
        reward_model: RewardModel,
        optimizer: torch.optim.Optimizer,
        tokenizer,
        kl_coef: float = 0.1,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        ppo_epochs: int = 4,
        max_grad_norm: float = 1.0,
        device: str = "cuda"
    ):
        self.actor_critic = actor_critic.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.optimizer = optimizer
        self.tokenizer = tokenizer

        # Freeze reference and reward models
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Hyperparameters
        self.kl_coef = kl_coef
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.gamma = gamma
        self.lambda_ = lambda_
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm
        self.device = device

    def generate_responses(
        self,
        prompts: List[str],
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        Generate responses for given prompts.

        Returns dict with:
            - input_ids: full prompt + response
            - attention_mask
            - prompt_lengths: length of each prompt
        """
        self.actor_critic.eval()

        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        prompt_lengths = prompt_encodings.attention_mask.sum(dim=1)

        # Generate
        with torch.no_grad():
            outputs = self.actor_critic.policy.generate(
                **prompt_encodings,
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id
            )

        return {
            "input_ids": outputs,
            "attention_mask": (outputs != self.tokenizer.pad_token_id).long(),
            "prompt_lengths": prompt_lengths
        }

    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards from reward model and KL penalty.

        Returns:
            rewards: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape

        # Get reward model score (single scalar per sequence)
        with torch.no_grad():
            rm_scores = self.reward_model(input_ids, attention_mask)  # [batch_size]

        # Get KL divergence
        with torch.no_grad():
            # Current policy logits
            logits_policy, _ = self.actor_critic(input_ids, attention_mask)

            # Reference policy logits
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits_ref = ref_outputs.logits

        # Compute per-token KL
        kl_div = compute_kl_divergence(logits_policy, logits_ref, attention_mask)

        # Final reward (only at the end of sequence)
        rewards = torch.zeros(batch_size, seq_len, device=self.device)

        for i in range(batch_size):
            # Put reward at the last generated token
            last_idx = attention_mask[i].sum() - 1
            final_reward = rm_scores[i] - self.kl_coef * kl_div[i]
            rewards[i, last_idx] = final_reward

        return rewards

    def ppo_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single PPO update step.
        """
        self.actor_critic.train()

        # Forward pass
        logits, values = self.actor_critic(input_ids, attention_mask)

        # Compute log probabilities for actions taken
        # Get the log prob of each token in the sequence
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)  # [batch, seq-1, vocab]
        actions = input_ids[:, 1:]  # [batch, seq-1]

        # Gather log probs of selected actions
        new_log_probs = log_probs.gather(
            dim=-1,
            index=actions.unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq-1]

        # Match shapes with old log probs
        new_log_probs = new_log_probs * attention_mask[:, 1:]

        # Compute losses
        policy_loss = compute_ppo_loss(
            old_log_probs[:, :-1],
            new_log_probs,
            advantages[:, :-1],
            self.clip_epsilon
        )

        value_loss = compute_value_loss(
            values[:, :-1],
            returns[:, :-1],
            old_values[:, :-1],
            self.clip_epsilon
        )

        # Total loss
        loss = policy_loss + self.value_coef * value_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(),
            self.max_grad_norm
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        Complete PPO training step:
        1. Generate responses
        2. Compute rewards
        3. Compute advantages
        4. PPO updates
        """
        # Generate responses
        responses = self.generate_responses(prompts)
        input_ids = responses["input_ids"]
        attention_mask = responses["attention_mask"]

        # Compute rewards
        rewards = self.compute_rewards(
            input_ids,
            attention_mask,
            responses["prompt_lengths"]
        )

        # Get old log probs and values (for PPO clipping)
        with torch.no_grad():
            logits_old, values_old = self.actor_critic(input_ids, attention_mask)

            log_probs_old = F.log_softmax(logits_old[:, :-1], dim=-1)
            actions = input_ids[:, 1:]
            old_log_probs = log_probs_old.gather(
                dim=-1,
                index=actions.unsqueeze(-1)
            ).squeeze(-1)
            old_log_probs = old_log_probs * attention_mask[:, 1:]

        # Compute advantages and returns
        advantages, returns = compute_advantages_and_returns(
            rewards,
            values_old,
            gamma=self.gamma,
            lambda_=self.lambda_
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        stats = {"loss": 0, "policy_loss": 0, "value_loss": 0}
        for _ in range(self.ppo_epochs):
            step_stats = self.ppo_step(
                input_ids,
                attention_mask,
                old_log_probs,
                values_old,
                advantages,
                returns
            )
            for k, v in step_stats.items():
                stats[k] += v / self.ppo_epochs

        # Add reward stats
        stats["mean_reward"] = rewards.sum(dim=1).mean().item()

        return stats


# Example usage
def train_rlhf():
    """Example RLHF training pipeline."""
    # Initialize models
    model_name = "gpt2"  # Replace with your SFT model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load models
    actor_critic = ActorCritic(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    reward_model = RewardModel(model_name)

    # Optimizer
    optimizer = torch.optim.AdamW(actor_critic.parameters(), lr=1e-6)

    # Trainer
    trainer = PPOTrainer(
        actor_critic=actor_critic,
        ref_model=ref_model,
        reward_model=reward_model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        kl_coef=0.1,
        device="cuda"
    )

    # Training loop
    prompts = [
        "Explain quantum computing to a 5-year-old:",
        "Write a haiku about machine learning:",
        "What is the meaning of life?"
    ]

    num_steps = 1000
    for step in tqdm(range(num_steps)):
        stats = trainer.train_step(prompts)

        if step % 100 == 0:
            print(f"\nStep {step}")
            print(f"Loss: {stats['loss']:.4f}")
            print(f"Policy Loss: {stats['policy_loss']:.4f}")
            print(f"Value Loss: {stats['value_loss']:.4f}")
            print(f"Mean Reward: {stats['mean_reward']:.4f}")
```

---

## Practical Considerations

### 1. Computational Cost

RLHF is expensive because it requires:
- **4 models**: Actor, critic, reference, reward model
- **Generation**: Sampling responses on-the-fly
- **Multiple forward passes**: For computing rewards, KL, advantages

**Memory requirements:**
- Actor-critic: 2x model size (policy + value head)
- Reference: 1x model size
- Reward model: 1x model size
- Total: ~4x model size

**Solutions:**
- Use smaller models for reward/value
- Offload reference model to CPU
- Use parameter-efficient fine-tuning (see [LoRA](19-peft.md))

### 2. Hyperparameter Tuning

Key hyperparameters:
- **KL coefficient** $\beta$: 0.01-0.1 (higher = stay closer to SFT)
- **PPO clip** $\epsilon$: 0.1-0.3 (higher = larger updates)
- **Learning rate**: 1e-6 to 1e-5 (much lower than SFT)
- **Batch size**: 64-512 prompts per step
- **PPO epochs**: 2-4 per batch

### 3. Reward Hacking

The policy may find adversarial ways to maximize reward:
- **Repetition**: Repeating phrases that score high
- **Length gaming**: Very short or very long responses
- **Non-sequiturs**: Random high-reward tokens

**Mitigations:**
- Strong KL penalty
- Reward model diversity (train on diverse preferences)
- Length normalization
- Regular evaluation on held-out prompts

### 4. Training Stability

RLHF can be unstable:
- **Policy collapse**: Model generates nonsense
- **Value explosion**: Value estimates diverge
- **Reward spikes**: Extreme reward values

**Solutions:**
- Reward normalization/clipping
- Gradient clipping
- Smaller learning rates
- Monitor KL divergence closely

### 5. Alternatives to PPO

Other RL algorithms have been explored:
- **REINFORCE with baseline**: Simpler but higher variance
- **A2C/A3C**: More sample efficient
- **RLOO (REINFORCE Leave-One-Out)**: Used by some recent systems

However, **DPO** (see [Chapter 21](21-dpo.md)) has largely replaced PPO for many applications due to its simplicity and stability.

---

## Exercises

### Exercise 1: Implement a Simple Reward Model

Train a reward model on a small preference dataset.

```python
# Create a synthetic preference dataset
def create_preference_dataset(num_samples: int = 1000):
    """
    Create a synthetic dataset where longer, more diverse responses
    are preferred.
    """
    prompts = [
        "Explain machine learning:",
        "What is Python?",
        "Describe neural networks:",
    ]

    dataset = []
    for _ in range(num_samples):
        prompt = prompts[torch.randint(0, len(prompts), (1,)).item()]

        # Simulate chosen (longer, better) vs rejected (shorter, worse)
        chosen = prompt + " This is a detailed and helpful response that " \
                         "provides comprehensive information."
        rejected = prompt + " Short answer."

        dataset.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    return dataset

# TODO: Implement the training loop
# 1. Create the dataset
# 2. Tokenize chosen and rejected responses
# 3. Train the reward model
# 4. Evaluate accuracy on held-out data
```

### Exercise 2: Implement GAE from Scratch

Implement Generalized Advantage Estimation without using the provided function.

```python
def gae_from_scratch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> torch.Tensor:
    """
    Implement GAE.

    Hint: Work backwards from the last timestep.
    """
    # TODO: Your implementation here
    pass
```

### Exercise 3: Adaptive KL Controller

Implement and test the adaptive KL controller with different target KL values.

```python
# TODO: Test the AdaptiveKLController class
# 1. Initialize with different target_kl values
# 2. Simulate KL divergence observations
# 3. Plot how beta changes over time
# 4. Compare with fixed beta
```

### Exercise 4: Compare RLHF with SFT

Train a small model (e.g., GPT-2 small) with both SFT only and SFT + RLHF.

```python
# TODO:
# 1. Fine-tune GPT-2 on an instruction dataset (SFT)
# 2. Collect preferences on model outputs
# 3. Train reward model
# 4. Run RLHF with PPO
# 5. Compare outputs qualitatively
# 6. Measure reward model scores on held-out prompts
```

**Evaluation questions:**
- Does RLHF improve response quality?
- How much does KL divergence increase?
- What happens with different $\beta$ values?

### Exercise 5: Reward Model Analysis

Analyze what your reward model has learned.

```python
def analyze_reward_model(reward_model, tokenizer, prompts):
    """
    Analyze reward model behavior.

    Tasks:
    1. Score various responses to the same prompt
    2. Test sensitivity to response length
    3. Test preference for certain patterns
    4. Visualize reward distributions
    """
    # TODO: Your implementation
    pass
```

---

## Summary

RLHF is a powerful technique for aligning language models with human preferences:

**Key Components:**
1. **Reward Model**: Learns to predict human preferences using Bradley-Terry model
2. **PPO**: Optimizes policy to maximize reward with clipped objective
3. **KL Constraint**: Prevents reward hacking by staying close to reference model

**Advantages:**
- Can capture complex, hard-to-specify preferences
- Enables continuous improvement from feedback
- Has produced state-of-the-art aligned models (ChatGPT, Claude, etc.)

**Challenges:**
- Computationally expensive (4 models, generation)
- Training instability
- Reward hacking
- Requires high-quality preference data

**Alternatives:**
- **DPO** (see [Chapter 21](21-dpo.md)): Simpler, more stable
- **RLAIF**: Use AI feedback instead of human feedback
- **Constitutional AI** (see [Chapter 22](22-safety-alignment.md)): Self-improvement through principles

In practice, modern alignment pipelines often use a combination of SFT, RLHF/DPO, and other techniques to achieve safe and helpful behavior.

**Next Steps:**
- [Direct Preference Optimization (DPO)](21-dpo.md): A simpler alternative to RLHF
- [Safety and Alignment Techniques](22-safety-alignment.md): Additional alignment methods
- [Supervised Fine-tuning (SFT)](18-sft.md): Review the first step before RLHF
