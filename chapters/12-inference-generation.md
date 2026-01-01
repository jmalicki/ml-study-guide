# Chapter 12: Inference and Generation Strategies

Inference and generation strategies are critical for controlling how autoregressive language models produce text. Understanding these techniques is essential for ML interviews, as they determine the quality, diversity, and efficiency of model outputs. While models are trained to predict probability distributions, the method we use to sample from these distributions dramatically affects the generated text.

## Table of Contents

1. [Introduction](#introduction)
2. [Autoregressive Generation Basics](#autoregressive-generation-basics)
3. [Greedy Decoding](#greedy-decoding)
4. [Temperature Scaling](#temperature-scaling)
5. [Top-k Sampling](#top-k-sampling)
6. [Nucleus (Top-p) Sampling](#nucleus-top-p-sampling)
7. [Combining Strategies](#combining-strategies)
8. [Beam Search](#beam-search)
9. [Advanced Decoding Methods](#advanced-decoding-methods)
10. [Repetition Penalties](#repetition-penalties)
11. [Stop Conditions](#stop-conditions)
12. [Connection to Other Topics](#connection-to-other-topics)
13. [Production Considerations](#production-considerations)
14. [Common Interview Questions](#common-interview-questions)
15. [Summary](#summary)
16. [References](#references)

---

## Introduction

### Why Inference Strategies Matter

Training a language model produces a probability distribution over the vocabulary at each step:

```math
P(x_t | x_{< t}) = \text{softmax}(f_\theta(x_{< t}))
```

where $f_\theta(x_{< t})$ are the logits (raw model outputs) for the next token given previous context.

However, **how we choose $x_t$ from this distribution is not specified by training**. Different sampling strategies lead to:

- **Different text quality**: coherent vs nonsensical
- **Different diversity**: repetitive vs creative
- **Different computational costs**: fast vs slow
- **Different use cases**: translation vs creative writing

**Key insight**: The same model can behave completely differently depending on the decoding strategy used.

### The Generation Problem in Autoregressive LLMs

Autoregressive models generate text token-by-token, where each token depends on all previous tokens:

```math
P(x_1, x_2, \ldots, x_n) = \prod_{t=1}^n P(x_t | x_1, \ldots, x_{t-1})
```

At each step $t$, we have:

1. **Input**: Context $x_1, \ldots, x_{t-1}$
2. **Model output**: Logits $\mathbf{z}_t \in \mathbb{R}^{|V|}$ for each token in vocabulary $V$
3. **Probabilities**: $P(x_t = v | x_{<t}) = \text{softmax}(\mathbf{z}_t)_v$
4. **Decision**: Which token to select?

The choice at step $t$ affects all future steps, creating a **sequential decision problem**.

### Overview of the Decoding Process

The complete inference pipeline:

```text
Input Tokens → Model Forward Pass → Logits → [Decoding Strategy] → Next Token
                                              ↑
                                         [Parameters]
                                    - Temperature
                                    - Top-k
                                    - Top-p
                                    - Repetition penalty
                                    - etc.
```

**Interview tip**: Understand that decoding happens **after** the model forward pass. The model always outputs the same logits for a given input; the strategy determines how we interpret those logits.

---

## Autoregressive Generation Basics

### How LLMs Generate Text Token-by-Token

The basic generation loop:

```python
import torch
import torch.nn.functional as F

def basic_generation_loop(
    model,
    prompt_tokens: torch.Tensor,  # [1, seq_len]
    max_new_tokens: int = 50,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Simplest possible generation: greedy decoding.

    Args:
        model: A language model (e.g., GPT)
        prompt_tokens: Initial token IDs
        max_new_tokens: How many tokens to generate
        device: Device to run on

    Returns:
        Generated token sequence including prompt
    """
    # Start with prompt
    generated = prompt_tokens.to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass through model
            logits = model(generated)  # [1, current_len, vocab_size]

            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :]  # [1, vocab_size]

            # Choose next token (greedy: pick max probability)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [1, 1]

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for end-of-sequence token (optional)
            # if next_token.item() == eos_token_id:
            #     break

    return generated


# Example usage (pseudo-code)
# prompt = tokenizer.encode("The quick brown")
# generated = basic_generation_loop(model, prompt, max_new_tokens=10)
# text = tokenizer.decode(generated)
```

### The Logits → Probabilities → Sampling Pipeline

Every decoding strategy follows this pattern:

```python
def decoding_pipeline(logits: torch.Tensor) -> int:
    """
    Generic decoding pipeline.

    Args:
        logits: Raw model outputs [vocab_size]

    Returns:
        Selected token ID
    """
    # Step 1: Apply modifications to logits (optional)
    # - Temperature scaling
    # - Repetition penalty
    # - Top-k filtering
    # - Top-p filtering
    logits = modify_logits(logits)

    # Step 2: Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Step 3: Sample from distribution
    next_token = sample_from_distribution(probs)

    return next_token
```

**Key stages**:

1. **Logit manipulation**: Modify the raw outputs before converting to probabilities
2. **Normalization**: Apply softmax to get valid probability distribution
3. **Sampling**: Choose a token according to some strategy

**Interview insight**: Most decoding strategies operate by modifying logits before sampling, not by changing the sampling procedure itself.

### Simple PyTorch Code Showing the Basic Generation Loop

Complete working example:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGPT(nn.Module):
    """Minimal GPT-like model for demonstration."""

    def __init__(self, vocab_size: int, d_model: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=4
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] token IDs

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        embeddings = self.embedding(x)
        # Causal mask for autoregressive generation
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        hidden = self.transformer(embeddings, embeddings, tgt_mask=mask)
        logits = self.lm_head(hidden)
        return logits


def demonstrate_generation():
    """Show basic generation in action."""
    vocab_size = 1000
    model = SimpleGPT(vocab_size)
    model.eval()

    # Start with some tokens
    prompt = torch.tensor([[1, 2, 3]])  # Arbitrary token IDs

    print("Generating text token-by-token:")
    print(f"Initial prompt: {prompt.tolist()}")

    generated = prompt.clone()

    for step in range(5):
        # Forward pass
        with torch.no_grad():
            logits = model(generated)

        # Get logits for next position
        next_logits = logits[:, -1, :]  # [1, vocab_size]

        # Sample (greedy for now)
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        # Append
        generated = torch.cat([generated, next_token], dim=1)

        print(f"Step {step + 1}: Added token {next_token.item()}, "
              f"sequence now: {generated.tolist()}")

    print(f"\nFinal sequence: {generated.tolist()}")


if __name__ == "__main__":
    demonstrate_generation()
```

**Output** (example):

```text
Generating text token-by-token:
Initial prompt: [[1, 2, 3]]
Step 1: Added token 42, sequence now: [[1, 2, 3, 42]]
Step 2: Added token 137, sequence now: [[1, 2, 3, 42, 137]]
Step 3: Added token 501, sequence now: [[1, 2, 3, 42, 137, 501]]
Step 4: Added token 89, sequence now: [[1, 2, 3, 42, 137, 501, 89]]
Step 5: Added token 256, sequence now: [[1, 2, 3, 42, 137, 501, 89, 256]]

Final sequence: [[1, 2, 3, 42, 137, 501, 89, 256]]
```

---

## Greedy Decoding

### Always Picking the Highest Probability Token

Greedy decoding selects the token with maximum probability at each step:

```math
x_t = \arg\max_{v \in V} P(v | x_{<t})
```

**Mathematical formulation**:

```math
x_t = \arg\max_{v \in V} \text{softmax}(\mathbf{z}_t)_v = \arg\max_{v \in V} z_{t,v}
```

Since softmax is monotonic, we can skip the softmax computation and just take the argmax of logits directly.

### Pros and Cons

**Pros**:

1. **Fast**: No sampling required, just argmax
2. **Deterministic**: Same input always produces same output (useful for debugging)
3. **Simple**: Minimal code, no hyperparameters

**Cons**:

1. **Repetitive**: Often falls into repetitive loops
2. **Suboptimal sequences**: Locally optimal choice at each step doesn't guarantee globally optimal sequence
3. **No diversity**: Cannot generate multiple different completions
4. **Misses better paths**: A slightly lower probability token might lead to much better overall sequence

**Interview insight**: Greedy decoding finds the most likely **token** at each step, not the most likely **sequence** overall.

### Code Example

```python
import torch
import torch.nn.functional as F


def greedy_decode(
    logits: torch.Tensor,  # [batch, vocab_size]
    **kwargs  # Ignored, for API compatibility
) -> torch.Tensor:
    """
    Greedy decoding: select token with highest probability.

    Args:
        logits: Raw model outputs for next token

    Returns:
        Selected token IDs [batch]
    """
    # No need for softmax since argmax is monotonic
    next_token = torch.argmax(logits, dim=-1)
    return next_token


def greedy_generation(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: int = None
) -> torch.Tensor:
    """
    Generate text using greedy decoding.

    Args:
        model: Language model
        prompt: Initial tokens [1, seq_len]
        max_new_tokens: Maximum tokens to generate
        eos_token_id: Stop if this token is generated

    Returns:
        Generated sequence [1, total_len]
    """
    generated = prompt.clone()

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits = model(generated)
            next_token_logits = logits[:, -1, :]  # [1, vocab_size]

            # Greedy selection
            next_token = greedy_decode(next_token_logits)
            next_token = next_token.unsqueeze(0)  # [1, 1]

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return generated


def demonstrate_greedy_issues():
    """
    Demonstrate why greedy decoding can be suboptimal.

    Example: At each step, greedy picks highest probability token,
    but this can lead to worse overall sequence.
    """
    # Simulated probabilities for a vocabulary of 4 tokens
    # Position 0: Token A has 51% probability, Token B has 49%
    # Position 1: If we chose A, best next token has 10% probability
    #             If we chose B, best next token has 90% probability

    # Greedy would choose: A (51%) → low prob next token
    # Better would be:     B (49%) → high prob next token
    # Overall: P(A, next) = 0.51 * 0.10 = 0.051
    #          P(B, next) = 0.49 * 0.90 = 0.441  (much better!)

    print("Greedy Decoding Pitfall Example:")
    print("=" * 60)
    print("Position 0 probabilities:")
    print("  Token A: 51% (greedy chooses this)")
    print("  Token B: 49%")
    print()
    print("Position 1 probabilities:")
    print("  After A: best continuation has 10% probability")
    print("  After B: best continuation has 90% probability")
    print()
    print("Greedy sequence probability: 0.51 × 0.10 = 0.051")
    print("Better sequence probability: 0.49 × 0.90 = 0.441")
    print()
    print("Greedy misses the better sequence by being myopic!")


if __name__ == "__main__":
    demonstrate_greedy_issues()
```

**When to use greedy decoding**:

- Tasks requiring deterministic outputs (e.g., code generation for testing)
- Factual question answering where creativity is not desired
- Initial debugging of generation systems

**When to avoid**:

- Creative writing
- Chatbots (repetition is obvious to users)
- Any task requiring diversity in outputs

---

## Temperature Scaling

Temperature scaling is the most fundamental and widely-used technique for controlling generation randomness.

### Mathematical Formulation

Temperature modifies the logits before applying softmax:

```math
P(x_t = v | x_{<t}) = \frac{\exp(z_v / T)}{\sum_{v' \in V} \exp(z_{v'} / T)}
```

where:
- $z_v$ is the logit for token $v$
- $T > 0$ is the temperature parameter

**Effect on distribution**:

```math
\text{softmax}(\mathbf{z} / T) = \text{softmax}(T^{-1} \mathbf{z})
```

### Effect of Temperature on Probability Distribution

**Temperature $T$ controls the "sharpness" of the distribution**:

#### $T = 1$: Original distribution

```math
P(v) = \frac{\exp(z_v)}{\sum_{v'} \exp(z_{v'})}
```

No change from model's original outputs.

#### $T < 1$: Sharper (more confident)

```math
P(v) = \frac{\exp(z_v / 0.5)}{\sum_{v'} \exp(z_{v'} / 0.5)} = \frac{\exp(2z_v)}{\sum_{v'} \exp(2z_{v'})}
```

- Exaggerates differences between logits
- High-probability tokens become even more likely
- Low-probability tokens become less likely
- Distribution becomes "peaky"

**Example**:
- Original: [0.4, 0.3, 0.2, 0.1]
- T=0.5:   [0.52, 0.28, 0.14, 0.06] (more concentrated)

#### $T > 1$: Flatter (more random)

```math
P(v) = \frac{\exp(z_v / 2.0)}{\sum_{v'} \exp(z_{v'} / 2.0)} = \frac{\exp(0.5z_v)}{\sum_{v'} \exp(0.5z_{v'})}
```

- Smooths differences between logits
- Makes distribution more uniform
- Increases diversity

**Example**:
- Original: [0.4, 0.3, 0.2, 0.1]
- T=2.0:   [0.32, 0.28, 0.23, 0.17] (more uniform)

#### $T \to 0$: Approaches greedy

```math
\lim_{T \to 0} \text{softmax}(\mathbf{z} / T) = \text{one-hot}(\arg\max \mathbf{z})
```

Distribution collapses to deterministic choice.

#### $T \to \infty$: Approaches uniform

```math
\lim_{T \to \infty} \text{softmax}(\mathbf{z} / T) = \text{uniform}(V)
```

All tokens equally likely (random sampling).

### Code Example with Visualization

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.

    Args:
        logits: Raw model outputs [vocab_size] or [batch, vocab_size]
        temperature: Temperature parameter (default: 1.0 = no change)

    Returns:
        Temperature-scaled logits
    """
    return logits / temperature


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample token with temperature scaling.

    Args:
        logits: [batch, vocab_size]
        temperature: Scaling factor

    Returns:
        Sampled token IDs [batch]
    """
    # Scale logits by temperature
    scaled_logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sample from distribution
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return next_token


def visualize_temperature_effects():
    """
    Visualize how temperature affects probability distributions.
    """
    # Create example logits (e.g., 10 tokens)
    torch.manual_seed(42)
    logits = torch.randn(10)

    # Try different temperatures
    temperatures = [0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, T in enumerate(temperatures):
        # Apply temperature
        scaled_logits = logits / T
        probs = F.softmax(scaled_logits, dim=-1).numpy()

        # Plot
        ax = axes[idx]
        ax.bar(range(len(probs)), probs)
        ax.set_title(f'Temperature T = {T}')
        ax.set_xlabel('Token ID')
        ax.set_ylabel('Probability')
        ax.set_ylim([0, 1])

        # Add statistics
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_prob = probs.max()
        ax.text(0.7, 0.9, f'Entropy: {entropy:.2f}\nMax prob: {max_prob:.2f}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    plt.savefig('temperature_effects.png', dpi=150, bbox_inches='tight')
    print("Saved temperature_effects.png")

    # Print numerical comparison
    print("\nTemperature Effects on Probability Distribution:")
    print("=" * 70)
    print(f"{'Temperature':<15} {'Max Prob':<15} {'Min Prob':<15} {'Entropy':<15}")
    print("-" * 70)

    for T in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        scaled_logits = logits / T
        probs = F.softmax(scaled_logits, dim=-1)
        max_prob = probs.max().item()
        min_prob = probs.min().item()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        print(f"{T:<15.1f} {max_prob:<15.4f} {min_prob:<15.6f} {entropy:<15.4f}")

    print("\nKey observations:")
    print("  - Lower T: Higher max prob, lower entropy (more confident)")
    print("  - Higher T: Lower max prob, higher entropy (more random)")


def demonstrate_temperature_sampling():
    """
    Show how temperature affects generated sequences.
    """
    vocab_size = 10

    # Simulated logits favoring token 3
    logits = torch.tensor([1.0, 1.2, 1.5, 5.0, 1.3, 1.1, 0.9, 1.0, 1.2, 0.8])

    print("Temperature Sampling Demo")
    print("=" * 70)
    print(f"Logits: {logits.tolist()}")
    print(f"Token with highest logit: {torch.argmax(logits).item()}")
    print()

    temperatures = [0.1, 0.5, 1.0, 2.0]
    num_samples = 1000

    for T in temperatures:
        # Sample many times
        samples = []
        for _ in range(num_samples):
            token = sample_with_temperature(logits.unsqueeze(0), temperature=T)
            samples.append(token.item())

        # Count occurrences
        counts = torch.bincount(torch.tensor(samples), minlength=vocab_size)
        frequencies = counts.float() / num_samples

        print(f"T = {T}:")
        print(f"  Token 3 (highest logit) chosen: {frequencies[3]:.1%}")
        print(f"  Other tokens: {(1 - frequencies[3]):.1%}")
        print()


if __name__ == "__main__":
    visualize_temperature_effects()
    print()
    demonstrate_temperature_sampling()
```

### Interview Tips

**Q: "What temperature should I use?"**

**A**: Common values:
- **T = 0.7**: Good balance for chatbots (slight randomness, mostly coherent)
- **T = 1.0**: Default, use model's original probabilities
- **T = 0.1-0.5**: Factual tasks, code generation (more deterministic)
- **T = 1.5-2.0**: Creative writing, brainstorming (more diverse)

**Q: "Why not just use greedy decoding instead of very low temperature?"**

**A**: Low temperature (e.g., 0.3) still allows occasional diversity, preventing complete repetition loops. It's a "softer" version of greedy.

---

## Top-k Sampling

Top-k sampling restricts the sampling pool to the k most likely tokens.

### Only Sample from the Top k Tokens

**Algorithm**:

1. Sort tokens by probability (or logits)
2. Keep only the top k tokens
3. Set all other tokens' probabilities to 0
4. Renormalize and sample

**Mathematical formulation**:

Let $V_k(x_{<t})$ be the set of top-k tokens:

```math
V_k(x_{<t}) = \{v_1, v_2, \ldots, v_k\} \text{ where } P(v_i | x_{<t}) \geq P(v_{i+1} | x_{<t})
```

Then:

```math
P_{\text{top-k}}(v | x_{<t}) = \begin{cases}
\frac{P(v | x_{<t})}{\sum_{v' \in V_k} P(v' | x_{<t})} & \text{if } v \in V_k \\
0 & \text{otherwise}
\end{cases}
```

### Eliminates Low-Probability Tokens

**Motivation**: The tail of the distribution often contains nonsensical tokens. Top-k prevents sampling from this tail.

**Example**:

Original distribution:
```
Token 1: 0.35
Token 2: 0.25
Token 3: 0.15
Token 4: 0.10
Token 5: 0.05
... (500 more tokens with tiny probabilities)
```

With k=4, renormalized:
```
Token 1: 0.35 / 0.85 = 0.412
Token 2: 0.25 / 0.85 = 0.294
Token 3: 0.15 / 0.85 = 0.176
Token 4: 0.10 / 0.85 = 0.118
All others: 0
```

### Code Example

```python
import torch
import torch.nn.functional as F


def top_k_sampling(
    logits: torch.Tensor,  # [batch, vocab_size]
    k: int = 50,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample from top-k tokens.

    Args:
        logits: Raw model outputs
        k: Number of top tokens to consider
        temperature: Temperature scaling (applied before top-k)

    Returns:
        Sampled token IDs [batch]
    """
    # Apply temperature
    logits = logits / temperature

    # Get top-k logits and indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)  # [batch, k]

    # Convert top-k logits to probabilities
    top_k_probs = F.softmax(top_k_logits, dim=-1)  # [batch, k]

    # Sample from top-k distribution
    # Get indices in the top-k set (0 to k-1)
    sampled_indices = torch.multinomial(top_k_probs, num_samples=1)  # [batch, 1]

    # Map back to original vocabulary indices
    next_token = torch.gather(top_k_indices, -1, sampled_indices).squeeze(-1)  # [batch]

    return next_token


def top_k_logits_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Alternative implementation: filter logits then sample.

    Args:
        logits: [batch, vocab_size]
        k: Number of top tokens

    Returns:
        Filtered logits (non-top-k set to -inf)
    """
    # Get k-th largest value for each batch element
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    min_values = top_k_values[:, -1, None]  # [batch, 1]

    # Set all values below k-th largest to -inf
    filtered_logits = torch.where(
        logits < min_values,
        torch.full_like(logits, float('-inf')),
        logits
    )

    return filtered_logits


def demonstrate_top_k():
    """
    Show how top-k filtering affects sampling.
    """
    vocab_size = 100
    k = 10

    # Create example logits
    torch.manual_seed(42)
    logits = torch.randn(1, vocab_size)

    # Original probabilities
    original_probs = F.softmax(logits, dim=-1)

    # Top-k filtered
    filtered_logits = top_k_logits_filter(logits, k)
    filtered_probs = F.softmax(filtered_logits, dim=-1)

    print("Top-k Sampling Demonstration")
    print("=" * 70)
    print(f"Vocabulary size: {vocab_size}")
    print(f"k (top tokens to keep): {k}")
    print()

    # Show top tokens before and after
    print("Original top 10 probabilities:")
    top_10_probs, top_10_indices = torch.topk(original_probs[0], 10)
    for i, (idx, prob) in enumerate(zip(top_10_indices, top_10_probs)):
        print(f"  {i+1}. Token {idx.item()}: {prob.item():.4f}")

    print(f"\nAfter top-{k} filtering (renormalized):")
    nonzero_mask = filtered_probs[0] > 0
    nonzero_probs = filtered_probs[0][nonzero_mask]
    nonzero_indices = torch.arange(vocab_size)[nonzero_mask]
    sorted_indices = torch.argsort(nonzero_probs, descending=True)

    for i, idx in enumerate(sorted_indices):
        token_idx = nonzero_indices[idx]
        prob = nonzero_probs[idx]
        print(f"  {i+1}. Token {token_idx.item()}: {prob.item():.4f}")

    print(f"\nNumber of tokens with non-zero probability:")
    print(f"  Original: {(original_probs[0] > 1e-10).sum().item()}")
    print(f"  After top-{k}: {(filtered_probs[0] > 0).sum().item()}")

    # Sample multiple times to show distribution
    print(f"\nSampling 1000 times with k={k}:")
    samples = []
    for _ in range(1000):
        token = top_k_sampling(logits, k=k, temperature=1.0)
        samples.append(token.item())

    sample_counts = torch.bincount(torch.tensor(samples), minlength=vocab_size)
    num_unique = (sample_counts > 0).sum().item()
    print(f"  Unique tokens sampled: {num_unique} (out of {vocab_size})")
    print(f"  All sampled tokens are in top-{k}: {num_unique <= k}")


if __name__ == "__main__":
    demonstrate_top_k()
```

### Choosing k

**Common values**:
- k = 1: Equivalent to greedy decoding
- k = 10-50: Common for chatbots and creative tasks
- k = 100-500: Very diverse generation

**Tradeoff**:
- Small k: More focused, less diversity, fewer errors
- Large k: More diversity, more creative, more potential errors

**Interview insight**: Top-k has a **fixed-size** cutoff, which can be problematic when the distribution is very peaked (wastes budget on unlikely tokens) or very flat (cuts off too many reasonable tokens).

This limitation motivated **top-p sampling**.

---

## Nucleus (Top-p) Sampling

Nucleus sampling (top-p) is currently the **industry standard** for most production LLMs.

### Dynamic Vocabulary Size Based on Cumulative Probability

Instead of a fixed k, top-p chooses the **smallest set of tokens** whose cumulative probability exceeds a threshold p.

**Algorithm**:

1. Sort tokens by probability in descending order
2. Compute cumulative probability
3. Keep tokens until cumulative probability ≥ p
4. Renormalize and sample from this set

### Mathematical Formulation

Define the nucleus $V_p(x_{<t})$ as the smallest set satisfying:

```math
V_p(x_{<t}) = \min \left\{ V' \subseteq V : \sum_{v \in V'} P(v | x_{<t}) \geq p \right\}
```

The sampling distribution becomes:

```math
P_{\text{nucleus}}(v | x_{<t}) = \begin{cases}
\frac{P(v | x_{<t})}{\sum_{v' \in V_p} P(v' | x_{<t})} & \text{if } v \in V_p \\
0 & \text{otherwise}
\end{cases}
```

### Why It's Often Preferred Over Top-k

**Advantages of top-p over top-k**:

1. **Adaptive**: Adjusts vocabulary size based on distribution shape
   - Peaked distribution (confident): samples from fewer tokens
   - Flat distribution (uncertain): samples from more tokens

2. **Prevents issues**:
   - When distribution is very peaked, top-k might include many unlikely tokens
   - When distribution is flat, top-k might cut off many good options

**Example**:

Peaked distribution (model is confident):
```
Token 1: 0.80  ← Cumulative: 0.80
Token 2: 0.10  ← Cumulative: 0.90
Token 3: 0.05  ← Cumulative: 0.95 (p=0.9 stops here)
Token 4: 0.03
...
```
With p=0.9: nucleus size = 3 tokens

Flat distribution (model is uncertain):
```
Token 1: 0.15  ← Cumulative: 0.15
Token 2: 0.12  ← Cumulative: 0.27
Token 3: 0.11  ← Cumulative: 0.38
Token 4: 0.10  ← Cumulative: 0.48
Token 5: 0.09  ← Cumulative: 0.57
Token 6: 0.08  ← Cumulative: 0.65
Token 7: 0.08  ← Cumulative: 0.73
Token 8: 0.07  ← Cumulative: 0.80
Token 9: 0.06  ← Cumulative: 0.86
Token 10: 0.05 ← Cumulative: 0.91 (p=0.9 stops here)
...
```
With p=0.9: nucleus size = 10 tokens

**Fixed k=5 would**:
- In peaked case: include tokens 4-5 with tiny probability (wasted)
- In flat case: cut off tokens 6-10 which are reasonable (too restrictive)

### Code Example

```python
import torch
import torch.nn.functional as F


def nucleus_sampling(
    logits: torch.Tensor,  # [batch, vocab_size]
    p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling.

    Args:
        logits: Raw model outputs
        p: Cumulative probability threshold (typically 0.9 or 0.95)
        temperature: Temperature scaling

    Returns:
        Sampled token IDs [batch]
    """
    # Apply temperature
    logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)  # [batch, vocab_size]

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # [batch, vocab_size]

    # Find cutoff index: first position where cumulative prob > p
    # We want to include this position, so we use >= p-1e-10
    cutoff_mask = cumulative_probs > p

    # Keep at least one token (the most likely)
    cutoff_mask[:, 0] = False

    # Set probabilities outside nucleus to 0
    sorted_probs[cutoff_mask] = 0.0

    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample from the nucleus distribution
    sampled_sorted_indices = torch.multinomial(sorted_probs, num_samples=1)  # [batch, 1]

    # Map back to original vocabulary
    next_token = torch.gather(sorted_indices, -1, sampled_sorted_indices).squeeze(-1)

    return next_token


def nucleus_logits_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Alternative: filter logits then sample.

    Args:
        logits: [batch, vocab_size]
        p: Nucleus threshold

    Returns:
        Filtered logits (outside nucleus set to -inf)
    """
    # Sort logits (descending)
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)

    # Get probabilities and cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask for tokens to remove (keep tokens until cumulative_prob > p)
    sorted_mask = cumulative_probs > p
    sorted_mask[:, 0] = False  # Keep at least the top token

    # Scatter mask back to original indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)

    # Filter logits
    filtered_logits = logits.masked_fill(mask, float('-inf'))

    return filtered_logits


def demonstrate_nucleus_sampling():
    """
    Compare nucleus sampling behavior on different distributions.
    """
    print("Nucleus (Top-p) Sampling Demonstration")
    print("=" * 70)

    p = 0.9

    # Case 1: Peaked distribution (confident)
    peaked_logits = torch.tensor([[10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.1, 0.05]])
    peaked_probs = F.softmax(peaked_logits, dim=-1)

    # Case 2: Flat distribution (uncertain)
    flat_logits = torch.tensor([[2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1]])
    flat_probs = F.softmax(flat_logits, dim=-1)

    def analyze_nucleus(logits, probs, name):
        print(f"\n{name}:")
        print("-" * 70)

        # Sort and compute cumulative
        sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)

        # Find nucleus size
        nucleus_size = (cumulative <= p).sum().item() + 1  # +1 to exceed p

        print(f"Nucleus size (p={p}): {nucleus_size} tokens")
        print(f"\nTop tokens (cumulative probability):")
        for i in range(min(nucleus_size + 2, len(sorted_probs))):
            token_id = sorted_indices[i].item()
            prob = sorted_probs[i].item()
            cum_prob = cumulative[i].item()
            in_nucleus = "✓" if i < nucleus_size else "✗"
            print(f"  {in_nucleus} Token {token_id}: p={prob:.4f}, cumulative={cum_prob:.4f}")

    analyze_nucleus(peaked_logits, peaked_probs, "Case 1: Peaked Distribution (Confident)")
    analyze_nucleus(flat_logits, flat_probs, "Case 2: Flat Distribution (Uncertain)")

    print("\n" + "=" * 70)
    print("Key insight: Nucleus adapts vocabulary size to confidence level!")
    print("  Peaked → smaller nucleus (3 tokens)")
    print("  Flat → larger nucleus (7-8 tokens)")


def compare_topk_vs_topp():
    """
    Direct comparison of top-k vs top-p on the same distribution.
    """
    vocab_size = 50
    torch.manual_seed(42)
    logits = torch.randn(1, vocab_size)
    probs = F.softmax(logits, dim=-1)

    k = 10
    p = 0.9

    print("\n" + "=" * 70)
    print("Top-k vs Top-p Comparison")
    print("=" * 70)

    # Top-k
    top_k_filtered = top_k_logits_filter(logits, k)
    top_k_probs = F.softmax(top_k_filtered, dim=-1)
    top_k_size = (top_k_probs[0] > 0).sum().item()

    # Top-p
    top_p_filtered = nucleus_logits_filter(logits, p)
    top_p_probs = F.softmax(top_p_filtered, dim=-1)
    top_p_size = (top_p_probs[0] > 0).sum().item()

    print(f"Top-k (k={k}):    {top_k_size} tokens")
    print(f"Top-p (p={p}): {top_p_size} tokens")
    print()
    print("Top-p adapts to the distribution shape,")
    print("while top-k uses fixed size regardless of confidence.")


if __name__ == "__main__":
    demonstrate_nucleus_sampling()
    compare_topk_vs_topp()
```

### Choosing p

**Common values**:
- p = 0.9: Standard for most chatbots (GPT-3.5, GPT-4)
- p = 0.95: Slightly more diverse
- p = 1.0: No filtering (full distribution)

**Interview tip**: Top-p with p=0.9 and temperature=0.7-1.0 is the current industry standard for most production LLMs.

---

## Combining Strategies

In practice, multiple strategies are often combined for better control.

### Temperature + Top-p (Common in Production)

The most common combination:

1. Apply temperature scaling to logits
2. Apply top-p filtering
3. Sample from resulting distribution

```python
def combined_sampling(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = None
) -> torch.Tensor:
    """
    Combined sampling strategy (production-standard).

    Order of operations:
      1. Temperature scaling
      2. (Optional) Top-k filtering
      3. Top-p filtering
      4. Sample

    Args:
        logits: [batch, vocab_size]
        temperature: Temperature scaling
        top_p: Nucleus threshold
        top_k: Optional top-k filtering before top-p

    Returns:
        Sampled tokens [batch]
    """
    # Step 1: Temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Step 2: Optional top-k
    if top_k is not None and top_k > 0:
        top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        min_values = top_k_values[:, -1, None]
        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, float('-inf')),
            logits
        )

    # Step 3: Top-p
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[:, 0] = False

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = logits.masked_fill(mask, float('-inf'))

    # Step 4: Sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return next_token
```

### The Order of Operations Matters

**Correct order**:

```
Logits → Temperature → Top-k → Top-p → Softmax → Sample
```

**Why this order**:

1. **Temperature first**: Modifies the distribution shape
2. **Top-k second** (if used): Removes very unlikely tokens
3. **Top-p third**: Adaptive filtering based on distribution
4. **Softmax**: Convert to probabilities
5. **Sample**: Draw from distribution

**Wrong order example**: Applying top-p before temperature would use the cumulative probabilities from the un-scaled distribution, defeating the purpose of temperature.

### Code Example

```python
def demonstrate_combination_effects():
    """
    Show how different combinations affect generation.
    """
    vocab_size = 100
    torch.manual_seed(42)
    logits = torch.randn(1, vocab_size)

    configs = [
        {"name": "Greedy", "temperature": 0.01, "top_p": 1.0, "top_k": None},
        {"name": "Low temp + top-p", "temperature": 0.5, "top_p": 0.9, "top_k": None},
        {"name": "Balanced", "temperature": 0.8, "top_p": 0.9, "top_k": None},
        {"name": "High diversity", "temperature": 1.2, "top_p": 0.95, "top_k": None},
        {"name": "Top-k + top-p", "temperature": 1.0, "top_p": 0.9, "top_k": 50},
    ]

    print("Combined Sampling Strategies")
    print("=" * 70)

    for config in configs:
        name = config.pop("name")

        # Sample multiple times
        samples = []
        for _ in range(1000):
            token = combined_sampling(logits.clone(), **config)
            samples.append(token.item())

        # Compute statistics
        unique_tokens = len(set(samples))
        sample_counts = torch.bincount(torch.tensor(samples), minlength=vocab_size)
        top_token_freq = sample_counts.max().item() / 1000

        print(f"\n{name}:")
        print(f"  Config: {config}")
        print(f"  Unique tokens sampled: {unique_tokens}")
        print(f"  Most frequent token: {top_token_freq:.1%}")

    print("\n" + "=" * 70)
    print("Observations:")
    print("  - Lower temperature → fewer unique tokens, higher concentration")
    print("  - Top-k + top-p → good balance of quality and diversity")
    print("  - Production systems typically use T=0.7-0.8, p=0.9")


if __name__ == "__main__":
    demonstrate_combination_effects()
```

**Interview insight**: The combination of temperature=0.7-0.8 and top_p=0.9 (without top_k) is what powers most modern chatbots like ChatGPT, Claude, and Gemini.

---

## Beam Search

Beam search maintains multiple candidate sequences and explores them in parallel.

### Maintaining Multiple Candidate Sequences

Unlike sampling methods that generate one sequence at a time, beam search keeps track of the top-k most likely **complete sequences** at each step.

**Algorithm**:

1. Start with initial prompt
2. At each step:
   - For each of the k current sequences
   - Expand with all possible next tokens
   - Compute sequence score for each expansion
   - Keep top k sequences overall
3. Continue until all beams end or max length reached

**Sequence score** (log probability):

```math
\text{score}(x_1, \ldots, x_t) = \sum_{i=1}^t \log P(x_i | x_{<i})
```

### Beam Width and Its Tradeoffs

**Beam width** $k$ controls the number of parallel sequences:

- k = 1: Equivalent to greedy decoding
- k = 5-10: Common for translation and summarization
- k = 50-100: For diverse generation

**Computational cost**: $O(k \times \text{vocab\_size})$ per step

**Memory cost**: Must store $k$ sequences with their KV caches

### Length Normalization

**Problem**: Longer sequences accumulate more log probabilities (all negative), so beam search bias towards shorter sequences.

**Solution**: Normalize by length:

```math
\text{score}(x_1, \ldots, x_t) = \frac{1}{t^\alpha} \sum_{i=1}^t \log P(x_i | x_{<i})
```

where $\alpha \in [0, 1]$ controls the strength of length normalization:
- $\alpha = 0$: No normalization (biased to short)
- $\alpha = 1$: Full normalization (average log probability)
- $\alpha = 0.6$: Common choice

### When to Use Beam Search vs Sampling

**Beam search**:
- ✅ Machine translation
- ✅ Summarization
- ✅ Tasks with "right answer"
- ✅ When you need best/most likely output

**Sampling (temperature/top-p)**:
- ✅ Creative writing
- ✅ Chatbots
- ✅ Tasks requiring diversity
- ✅ When multiple good answers exist

**Why not always use beam search?**

1. **Generic/boring**: Finds most likely sequence, which is often generic
2. **Expensive**: k times the computation
3. **No diversity** within a single beam (all beams are similar)

### Code Example

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple


class BeamSearchNode:
    """Node in beam search tree."""

    def __init__(
        self,
        tokens: torch.Tensor,  # [seq_len]
        log_prob: float,
        length: int
    ):
        self.tokens = tokens
        self.log_prob = log_prob
        self.length = length

    def score(self, alpha: float = 0.6) -> float:
        """Compute length-normalized score."""
        return self.log_prob / (self.length ** alpha)


def beam_search(
    model,
    prompt: torch.Tensor,  # [1, prompt_len]
    beam_width: int = 5,
    max_new_tokens: int = 50,
    length_penalty: float = 0.6,
    eos_token_id: int = None
) -> List[Tuple[torch.Tensor, float]]:
    """
    Beam search decoding.

    Args:
        model: Language model
        prompt: Initial tokens
        beam_width: Number of beams to maintain
        max_new_tokens: Maximum tokens to generate
        length_penalty: Alpha for length normalization
        eos_token_id: End-of-sequence token ID

    Returns:
        List of (sequence, score) tuples, sorted by score
    """
    device = prompt.device
    vocab_size = model.lm_head.out_features  # Assuming model has lm_head

    # Initialize beam with prompt
    beams = [BeamSearchNode(
        tokens=prompt[0],
        log_prob=0.0,
        length=prompt.size(1)
    )]

    completed_beams = []

    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            candidates = []

            # Expand each beam
            for beam in beams:
                # Check if already completed
                if eos_token_id is not None and beam.tokens[-1].item() == eos_token_id:
                    completed_beams.append(beam)
                    continue

                # Forward pass
                input_ids = beam.tokens.unsqueeze(0)  # [1, seq_len]
                logits = model(input_ids)
                next_token_logits = logits[0, -1, :]  # [vocab_size]

                # Get log probabilities
                log_probs = F.log_softmax(next_token_logits, dim=-1)

                # Get top-k next tokens
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                # Create candidate beams
                for log_prob, token_id in zip(top_log_probs, top_indices):
                    new_tokens = torch.cat([beam.tokens, token_id.unsqueeze(0)])
                    new_log_prob = beam.log_prob + log_prob.item()
                    new_length = beam.length + 1

                    candidates.append(BeamSearchNode(
                        tokens=new_tokens,
                        log_prob=new_log_prob,
                        length=new_length
                    ))

            # If no candidates, we're done
            if not candidates:
                break

            # Select top-k candidates by score
            candidates.sort(key=lambda x: x.score(length_penalty), reverse=True)
            beams = candidates[:beam_width]

            # Check if all beams completed
            if len(beams) == 0:
                break

    # Add any remaining beams to completed
    completed_beams.extend(beams)

    # Sort by score
    completed_beams.sort(key=lambda x: x.score(length_penalty), reverse=True)

    # Return top results
    results = [(beam.tokens, beam.score(length_penalty)) for beam in completed_beams]

    return results


def demonstrate_beam_search():
    """
    Demonstrate beam search with a simple example.
    """
    print("Beam Search Demonstration")
    print("=" * 70)

    # This is a simplified demo - in practice, you'd use a real model
    # For illustration, we'll show the concept

    print("Beam search maintains multiple hypotheses:")
    print()
    print("Step 0: [START]")
    print("  Beam 1: [START] (score: 0.0)")
    print()
    print("Step 1: Consider all next tokens for each beam")
    print("  From Beam 1:")
    print("    → [START, 'The'] (score: -0.5)")
    print("    → [START, 'A'] (score: -1.2)")
    print("    → [START, 'In'] (score: -1.5)")
    print("  Keep top 3 (beam_width=3):")
    print("    Beam 1: [START, 'The'] (score: -0.5)")
    print("    Beam 2: [START, 'A'] (score: -1.2)")
    print("    Beam 3: [START, 'In'] (score: -1.5)")
    print()
    print("Step 2: Expand all 3 beams (3 × vocab_size candidates)")
    print("  From Beam 1 [START, 'The']:")
    print("    → [START, 'The', 'cat'] (score: -0.8)")
    print("    → [START, 'The', 'dog'] (score: -1.0)")
    print("  From Beam 2 [START, 'A']:")
    print("    → [START, 'A', 'cat'] (score: -1.5)")
    print("  From Beam 3 [START, 'In']:")
    print("    → [START, 'In', 'the'] (score: -1.7)")
    print("  Keep top 3:")
    print("    Beam 1: [START, 'The', 'cat'] (score: -0.8)")
    print("    Beam 2: [START, 'The', 'dog'] (score: -1.0)")
    print("    Beam 3: [START, 'A', 'cat'] (score: -1.5)")
    print()
    print("... Continue until max length or EOS ...")
    print()
    print("Final beams (sorted by score):")
    print("  1. [START, 'The', 'cat', 'sat', 'down'] (score: -2.1)")
    print("  2. [START, 'The', 'dog', 'ran', 'away'] (score: -2.3)")
    print("  3. [START, 'A', 'cat', 'walked', 'by'] (score: -3.0)")
    print()
    print("Best sequence: 'The cat sat down'")


if __name__ == "__main__":
    demonstrate_beam_search()
```

**Interview tip**: Beam search is deterministic (always returns the same top-k sequences for given beam width). It's fundamentally different from sampling methods which introduce randomness.

---

## Advanced Decoding Methods

Beyond the standard techniques, several advanced methods have been proposed. These are worth knowing for interviews, though they're less commonly used in production.

### Contrastive Decoding

**Motivation**: Use a smaller "amateur" model to contrast with the main "expert" model, amplifying the expert's knowledge.

**Formula**:

```math
P_{\text{CD}}(x_t | x_{<t}) \propto \frac{P_{\text{expert}}(x_t | x_{<t})^\alpha}{P_{\text{amateur}}(x_t | x_{<t})^\beta}
```

where $\alpha > \beta$ (typically $\alpha=1, \beta=0.5$).

**Intuition**: Tokens that both models predict are likely generic/common. Tokens only the expert predicts are likely the distinctive, high-quality choices.

**Use case**: Improving factuality and reducing generic responses.

**Reference**: Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization" (2022)

### Typical Decoding

**Motivation**: Avoid both high-probability (boring) and low-probability (nonsensical) tokens by sampling from the "typical" probability range.

**Method**: Sample tokens close to the conditional entropy:

```math
\text{typicality}(v) = \left| -\log P(v | x_{<t}) - H(P(\cdot | x_{<t})) \right|
```

Keep tokens with low typicality (close to average information content).

**Intuition**: Typical decoding prefers tokens that carry the "expected" amount of information, avoiding both highly predictable and highly surprising tokens.

**Reference**: Meister et al., "Typical Decoding for Natural Language Generation" (2022)

### Mirostat

**Motivation**: Dynamically adjust sampling to maintain a target perplexity, preventing both repetition (too low perplexity) and incoherence (too high perplexity).

**Method**:
1. Set target perplexity $\tau$
2. At each step, adjust sampling threshold to keep running perplexity near $\tau$
3. Uses a feedback control mechanism

**Intuition**: Perplexity measures how "surprised" the model is. Mirostat keeps this surprise level constant.

**Use case**: Long-form generation where repetition is a problem.

**Reference**: Basu et al., "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity" (2020)

### Brief Descriptions

```python
# Contrastive decoding (simplified)
def contrastive_decoding(expert_logits, amateur_logits, alpha=1.0, beta=0.5):
    """
    Amplify expert's advantage over amateur.

    Args:
        expert_logits: Logits from large/expert model
        amateur_logits: Logits from small/amateur model
        alpha, beta: Scaling factors
    """
    expert_probs = F.softmax(expert_logits, dim=-1)
    amateur_probs = F.softmax(amateur_logits, dim=-1)

    # Contrastive distribution
    contrastive_probs = (expert_probs ** alpha) / (amateur_probs ** beta + 1e-10)
    contrastive_probs = contrastive_probs / contrastive_probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(contrastive_probs, num_samples=1)


# Typical decoding (simplified)
def typical_decoding(logits, mass=0.9):
    """
    Sample from tokens near conditional entropy.

    Args:
        logits: Model outputs
        mass: Probability mass to preserve
    """
    probs = F.softmax(logits, dim=-1)

    # Compute entropy
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1, keepdim=True)

    # Compute typicality (distance from entropy)
    log_probs = torch.log(probs + 1e-10)
    typicality = torch.abs(log_probs + entropy)

    # Sort by typicality (most typical first)
    sorted_typicality, sorted_indices = torch.sort(typicality, dim=-1)
    sorted_probs = torch.gather(probs, -1, sorted_indices)

    # Keep tokens until cumulative mass
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative_probs > mass
    cutoff[:, 0] = False

    sorted_probs[cutoff] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_indices, -1, sampled).squeeze(-1)
```

**Interview tip**: You don't need to implement these in detail, but knowing they exist and their motivation shows depth of knowledge.

---

## Repetition Penalties

Repetition is a common problem in autoregressive generation. Repetition penalties modify logits to discourage recently used tokens.

### Frequency Penalty

Reduce logit for each token proportional to how many times it's appeared:

```math
\text{logit}'(v) = \text{logit}(v) - \lambda \times \text{count}(v)
```

where $\text{count}(v)$ is the number of times token $v$ appeared in generated sequence.

**Effect**: Linear penalty based on frequency.

### Presence Penalty

Binary version: penalize any token that has appeared at least once:

```math
\text{logit}'(v) = \text{logit}(v) - \lambda \times \mathbb{1}[v \in \text{generated}]
```

**Effect**: Same penalty whether token appeared once or many times.

### How They Modify Logits

Both penalties are applied **before** temperature scaling and sampling:

```
Logits → Repetition Penalty → Temperature → Top-p → Sample
```

### Code Example

```python
def apply_repetition_penalty(
    logits: torch.Tensor,  # [batch, vocab_size]
    generated_tokens: torch.Tensor,  # [batch, seq_len]
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0
) -> torch.Tensor:
    """
    Apply repetition penalties to logits.

    Args:
        logits: Raw model outputs for next token
        generated_tokens: Previously generated tokens
        frequency_penalty: Penalty proportional to count (e.g., 0.5)
        presence_penalty: Fixed penalty for any occurrence (e.g., 0.5)

    Returns:
        Modified logits
    """
    batch_size, vocab_size = logits.shape

    # Count token frequencies
    for i in range(batch_size):
        tokens = generated_tokens[i]

        if frequency_penalty != 0.0:
            # Frequency penalty: proportional to count
            token_counts = torch.bincount(tokens, minlength=vocab_size).float()
            logits[i] -= frequency_penalty * token_counts

        if presence_penalty != 0.0:
            # Presence penalty: binary
            unique_tokens = torch.unique(tokens)
            logits[i, unique_tokens] -= presence_penalty

    return logits


def demonstrate_repetition_penalties():
    """
    Show how repetition penalties affect generation.
    """
    vocab_size = 20

    # Simulate: token 5 is highly likely but already generated multiple times
    logits = torch.randn(1, vocab_size)
    logits[0, 5] = 10.0  # Token 5 has very high logit

    # Generated sequence has token 5 five times
    generated = torch.tensor([[5, 3, 5, 7, 5, 2, 5, 5]])

    print("Repetition Penalty Demonstration")
    print("=" * 70)
    print(f"Token 5 appears {(generated == 5).sum().item()} times")
    print(f"Original logit for token 5: {logits[0, 5].item():.2f}")
    print()

    # No penalty
    probs_original = F.softmax(logits, dim=-1)
    print(f"No penalty - P(token 5) = {probs_original[0, 5].item():.4f}")

    # Frequency penalty
    logits_freq = apply_repetition_penalty(
        logits.clone(), generated, frequency_penalty=0.5
    )
    probs_freq = F.softmax(logits_freq, dim=-1)
    print(f"Frequency penalty 0.5 - P(token 5) = {probs_freq[0, 5].item():.4f}")
    print(f"  (logit reduced by {0.5 * 5} = 2.5)")

    # Presence penalty
    logits_pres = apply_repetition_penalty(
        logits.clone(), generated, presence_penalty=2.0
    )
    probs_pres = F.softmax(logits_pres, dim=-1)
    print(f"Presence penalty 2.0 - P(token 5) = {probs_pres[0, 5].item():.4f}")
    print(f"  (logit reduced by 2.0 regardless of count)")

    # Both
    logits_both = apply_repetition_penalty(
        logits.clone(), generated, frequency_penalty=0.3, presence_penalty=1.0
    )
    probs_both = F.softmax(logits_both, dim=-1)
    print(f"Both penalties - P(token 5) = {probs_both[0, 5].item():.4f}")
    print(f"  (logit reduced by 0.3×5 + 1.0 = 2.5)")


if __name__ == "__main__":
    demonstrate_repetition_penalties()
```

**Typical values** (OpenAI API):
- `frequency_penalty`: 0.0 to 2.0 (common: 0.3-0.7)
- `presence_penalty`: 0.0 to 2.0 (common: 0.3-0.7)

**Interview tip**: Frequency penalty is more aggressive than presence penalty. Use frequency penalty for creative tasks where you want to strongly discourage repetition, presence penalty for subtle discouragement.

---

## Stop Conditions

Generation must terminate at some point. Several mechanisms control when to stop.

### EOS Token

Most common: generate until the model outputs an end-of-sequence token.

```python
def generate_until_eos(model, prompt, eos_token_id, max_length=1000):
    """Generate until EOS token or max length."""
    generated = prompt.clone()

    for _ in range(max_length):
        logits = model(generated)
        next_token = sample(logits[:, -1, :])  # Some sampling strategy

        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        # Stop if EOS generated
        if next_token.item() == eos_token_id:
            break

    return generated
```

**Common EOS tokens**:
- GPT-2/GPT-3: `<|endoftext|>` (token 50256)
- LLaMA: `</s>` (token 2)
- Custom chat formats: `<|im_end|>`, `</s>`, etc.

### Max Length

Always enforce a maximum length to prevent infinite generation:

```python
max_new_tokens = 100  # or max_length = 2048
```

**Why needed**: Model might never generate EOS, or EOS might be suppressed by sampling strategy.

### Stop Strings/Sequences

For chat models, stop on specific strings:

```python
def generate_until_stop_string(
    model,
    tokenizer,
    prompt,
    stop_strings: List[str],
    max_length: int = 1000
):
    """
    Generate until any stop string appears.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Initial tokens
        stop_strings: List of strings to stop on (e.g., ["\n\n", "User:", "<|im_end|>"])
        max_length: Maximum total length
    """
    generated = prompt.clone()

    for _ in range(max_length - prompt.size(1)):
        logits = model(generated)
        next_token = sample(logits[:, -1, :])
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        # Decode to check for stop strings
        text = tokenizer.decode(generated[0])

        if any(stop_str in text for stop_str in stop_strings):
            break

    return generated
```

**Common stop strings for chat**:
- `"\n\nUser:"` (user's turn)
- `"<|im_end|>"` (ChatML format)
- `"###"` (some instruction formats)

### Code Example

```python
def generate_with_stop_conditions(
    model,
    tokenizer,
    prompt: torch.Tensor,
    eos_token_id: int = None,
    max_new_tokens: int = 100,
    stop_strings: List[str] = None,
    sampling_fn = None
):
    """
    Complete generation function with all stop conditions.

    Stops when:
      1. EOS token generated
      2. Max length reached
      3. Stop string detected

    Args:
        model: Language model
        tokenizer: Tokenizer (needed for stop strings)
        prompt: Initial tokens [1, seq_len]
        eos_token_id: EOS token ID
        max_new_tokens: Maximum tokens to generate
        stop_strings: Optional list of stop strings
        sampling_fn: Function to sample next token (default: greedy)
    """
    if sampling_fn is None:
        sampling_fn = lambda logits: torch.argmax(logits, dim=-1)

    generated = prompt.clone()

    for step in range(max_new_tokens):
        # Forward pass
        logits = model(generated)
        next_token = sampling_fn(logits[:, -1, :])
        next_token = next_token.unsqueeze(1)  # [1, 1]

        # Append
        generated = torch.cat([generated, next_token], dim=1)

        # Check EOS
        if eos_token_id is not None and next_token.item() == eos_token_id:
            print(f"Stopped at step {step + 1}: EOS token")
            break

        # Check stop strings
        if stop_strings is not None:
            text = tokenizer.decode(generated[0])
            for stop_str in stop_strings:
                if stop_str in text:
                    print(f"Stopped at step {step + 1}: Stop string '{stop_str}'")
                    # Optionally trim the stop string
                    return generated

        # Check max length
        if step == max_new_tokens - 1:
            print(f"Stopped at step {step + 1}: Max length reached")

    return generated


def demonstrate_stop_conditions():
    """
    Illustrate different stop conditions.
    """
    print("Stop Conditions Demonstration")
    print("=" * 70)

    print("\nScenario 1: Model generates EOS token")
    print("  Generated: [1, 2, 3, 4, 2]  (token 2 is EOS)")
    print("  Stop reason: EOS token")
    print("  Final sequence: [1, 2, 3, 4, 2]")

    print("\nScenario 2: Max length reached before EOS")
    print("  Generated: [1, 2, 3, 4, 5, 6, 7, 8, ...]")
    print("  Max length: 10 tokens")
    print("  Stop reason: Maximum length")
    print("  Final sequence: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")

    print("\nScenario 3: Stop string detected")
    print("  Decoded text: 'Hello, how can I help?\\n\\nUser:'")
    print("  Stop strings: ['\\n\\nUser:', '###']")
    print("  Stop reason: Stop string '\\n\\nUser:' detected")
    print("  Final text: 'Hello, how can I help?'  (trimmed)")

    print("\nPriority order:")
    print("  1. Check EOS token (every step)")
    print("  2. Check stop strings (every step, expensive)")
    print("  3. Check max length (every step)")


if __name__ == "__main__":
    demonstrate_stop_conditions()
```

**Interview tip**: In production systems, stop strings are checked by decoding periodically (e.g., every 10 tokens) rather than every step to avoid the overhead of decoding.

---

## Connection to Other Topics

Inference strategies interact with several other concepts in the guide.

### Link to KV Cache for Efficient Inference

All sampling strategies benefit from KV caching to avoid recomputing attention for previous tokens.

See [Chapter 15: KV Cache](15-kv-cache.md) for details.

**Key interaction**: Sampling strategy doesn't affect KV cache usage. Cache is updated the same way regardless of whether we use greedy, temperature, top-p, or beam search.

```python
# Pseudo-code showing KV cache with different sampling
for step in range(max_tokens):
    # Forward pass with cache (same for all strategies)
    logits, new_cache = model(next_token, cache=cache)

    # Sampling strategy varies here
    if strategy == "greedy":
        next_token = greedy_decode(logits)
    elif strategy == "top_p":
        next_token = nucleus_sampling(logits, p=0.9)
    # ... etc

    cache = new_cache  # Update cache (same for all)
```

### Link to Speculative Decoding for Faster Generation

Speculative decoding drafts multiple tokens with a smaller model, then verifies with the larger model.

See [Chapter 29: Hardware, Quantization, and Training Optimization](29-hardware-quantization-optimization.md) for speculative decoding.

**Interaction**: Speculative decoding requires **deterministic** sampling (greedy or beam search) or careful handling of randomness to ensure the verified output matches the intended distribution.

### Link to Hardware Chapter for Production Deployment

Production deployment considerations:

1. **Batching**: Can batch greedy/beam search easily; sampling requires careful handling of random seeds
2. **Throughput vs latency**: Greedy is fastest; beam search is slowest
3. **Memory**: Beam search requires k× the memory for KV cache

See [Chapter 29: Hardware, Quantization, and Training Optimization](29-hardware-quantization-optimization.md).

---

## Production Considerations

### Throughput vs Latency

Different strategies have different performance characteristics:

| Strategy | Latency | Throughput | Memory | Deterministic |
|----------|---------|------------|--------|---------------|
| Greedy | Lowest | Highest | Low | Yes |
| Temperature | Low | High | Low | No |
| Top-p | Low | High | Low | No |
| Top-k | Low | High | Low | No |
| Beam search (k=5) | 5× higher | 1/5 | 5× | Yes |

**Production insight**: Most production systems use temperature + top-p for quality/throughput balance.

### Request Batching

Batching multiple requests together improves GPU utilization:

```python
def batch_generate(model, prompts: List[torch.Tensor], **sampling_kwargs):
    """
    Generate for multiple prompts in parallel.

    Requires:
      - Padding prompts to same length
      - Attention masks for padding
      - Separate sampling for each element in batch
    """
    # Pad prompts
    max_len = max(p.size(0) for p in prompts)
    padded = torch.stack([
        F.pad(p, (0, max_len - p.size(0)), value=pad_token_id)
        for p in prompts
    ])

    # Create attention mask
    mask = (padded != pad_token_id)

    # Generate
    # ... (handle per-example sampling)
```

**Challenge**: Different sampling strategies may require different random states per example.

### Caching Strategies

For frequently used prompts (e.g., system prompts in chatbots):

1. **Precompute KV cache** for system prompt
2. **Share across requests** (copy-on-write)
3. Save significant computation

See [Chapter 15: KV Cache - PagedAttention](15-kv-cache.md#memory-management-pagedattention) for implementation.

---

## Common Interview Questions

### Q1: What's the difference between greedy and beam search?

**A**:
- **Greedy**: Selects the single most likely token at each step. Fast but myopic—can miss better sequences.
- **Beam search**: Maintains k candidate sequences, exploring multiple paths. Finds higher-quality sequences but is k× slower.

Key difference: Greedy makes one local decision per step; beam search explores multiple paths and can recover from suboptimal early choices.

### Q2: Why use temperature instead of just top-p?

**A**: They serve different purposes:
- **Temperature**: Controls the sharpness of the entire distribution (affects how peaked vs uniform it is)
- **Top-p**: Truncates the distribution (removes low-probability tail)

Used together: Temperature reshapes, then top-p filters. For example, high temperature (1.5) might make many tokens probable; top-p then ensures we don't sample from the extreme tail.

### Q3: When would you use beam search vs sampling?

**A**:
- **Beam search**: Tasks with a "correct" answer (translation, summarization, factual QA). Finds most likely sequence.
- **Sampling**: Open-ended generation (chatbots, creative writing). Need diversity, multiple valid answers exist.

Rule of thumb: If humans would agree on a single best answer, use beam search. If valid answers vary, use sampling.

### Q4: What temperature should I use?

**A**: Depends on the task:
- **0.1-0.5**: Factual, deterministic tasks (code generation, math)
- **0.7-0.8**: Conversational AI (standard for ChatGPT)
- **1.0**: Default (model's original distribution)
- **1.2-2.0**: Creative writing, brainstorming

Interview tip: Explain that temperature=1.0 is not always best; it depends on whether you want the model's "raw" distribution or a more focused/diverse one.

### Q5: How do repetition penalties work?

**A**:
- **Frequency penalty**: Reduces logit by `λ × count(token)`. Proportional to how often token appeared.
- **Presence penalty**: Reduces logit by `λ` if token appeared at all. Binary.

Applied before sampling. Prevents repetitive loops common in autoregressive models.

Typical values: 0.3-0.7 for most applications.

### Q6: Why is top-p preferred over top-k in production?

**A**: Top-p adapts to the distribution's confidence:
- When model is confident (peaked distribution), top-p uses fewer tokens
- When model is uncertain (flat distribution), top-p uses more tokens

Top-k uses fixed size regardless, which can:
- Waste budget on unlikely tokens (peaked case)
- Be too restrictive (flat case)

Most modern LLMs (GPT-4, Claude, Gemini) use top-p by default.

### Q7: What's the computational cost of different sampling strategies?

**A**:
- **Greedy**: O(vocab_size) for argmax, no sampling
- **Temperature/top-p/top-k**: O(vocab_size) for filtering, O(log vocab_size) for sampling
- **Beam search (width k)**: O(k × vocab_size) per step, k× memory for KV cache

Beam search is k× slower and uses k× memory. Other methods have similar cost.

### Q8: Can you combine multiple sampling strategies? What order?

**A**: Yes, common in production:

**Order**: Logits → Temperature → Top-k (optional) → Top-p → Softmax → Sample

**Why this order**:
1. Temperature first (reshapes distribution)
2. Top-k second (coarse filtering)
3. Top-p third (adaptive filtering)
4. Softmax (normalize)
5. Sample (draw token)

Changing order (e.g., top-p before temperature) would use wrong cumulative probabilities.

### Q9: How do you prevent infinite generation?

**A**: Multiple stop conditions:
1. **EOS token**: Model generates end-of-sequence token
2. **Max length**: Hard limit on tokens generated
3. **Stop strings**: Detect specific strings (e.g., "\n\nUser:" in chat)

Always set `max_length` as a safety fallback, even if expecting EOS.

### Q10: What's the relationship between KV cache and sampling strategy?

**A**: **Independent**. KV cache optimizes inference by storing key/value projections from previous tokens. It's used the same way regardless of sampling strategy (greedy, temperature, beam search, etc.).

The only difference: beam search requires k separate KV caches (one per beam).

---

## Summary

### Key Takeaways

1. **Decoding strategies control how we sample from the model's output distribution**, dramatically affecting generation quality, diversity, and computational cost.

2. **Greedy decoding** (argmax) is fast but myopic and repetitive.

3. **Temperature scaling** controls randomness: lower T (0.3-0.7) for focused output, higher T (1.2-2.0) for diversity.

4. **Top-p (nucleus) sampling** is the industry standard, adapting vocabulary size to confidence level (typical: p=0.9).

5. **Top-k sampling** uses fixed vocabulary size; less adaptive than top-p.

6. **Combining temperature + top-p** gives the best balance for most applications (standard for GPT-4, Claude, etc.).

7. **Beam search** finds high-quality sequences for tasks with "correct" answers but is k× slower than sampling.

8. **Repetition penalties** (frequency and presence) prevent repetitive loops.

9. **Stop conditions** (EOS, max length, stop strings) are essential for reliable generation.

10. **Production systems** typically use: temperature=0.7-0.8, top_p=0.9, frequency_penalty=0.3-0.7, with KV caching for efficiency.

### Mathematical Summary

**Temperature scaling**:
```math
P_T(v | x_{<t}) = \frac{\exp(z_v / T)}{\sum_{v'} \exp(z_{v'} / T)}
```

**Top-p (nucleus)**:
```math
V_p = \min \left\{ V' : \sum_{v \in V'} P(v | x_{<t}) \geq p \right\}
```

**Beam search score**:
```math
\text{score}(x_{1:t}) = \frac{1}{t^\alpha} \sum_{i=1}^t \log P(x_i | x_{<i})
```

**Repetition penalty**:
```math
\text{logit}'(v) = \text{logit}(v) - \lambda_f \cdot \text{count}(v) - \lambda_p \cdot \mathbb{1}[v \in \text{generated}]
```

### Connection to Other Chapters

- [Chapter 15: KV Cache](15-kv-cache.md) - Efficient inference for autoregressive generation
- [Chapter 29: Hardware, Quantization, and Training Optimization](29-hardware-quantization-optimization.md) - Speculative decoding, production deployment
- [Chapter 11: Complete Transformer](11-complete-transformer.md) - The models these strategies operate on

### Interview Talking Points

1. **Temperature is the most commonly asked about**: Understand its effect on distribution shape
2. **Top-p is industry standard**: Know why it's preferred over top-k
3. **Understand the tradeoffs**: Greedy (fast, repetitive) vs sampling (diverse, higher quality) vs beam search (best quality, slow)
4. **Order matters**: Temperature → top-k → top-p → softmax → sample
5. **Production defaults**: T=0.7-0.8, p=0.9, frequency_penalty=0.3-0.7

---

## References

### Foundational Papers

1. **Beam Search**
   - Freitag & Al-Onaizan, "Beam Search Strategies for Neural Machine Translation" (2017)

2. **Temperature Sampling**
   - Ackley et al., "A Learning Algorithm for Boltzmann Machines" (1985)
   - Hinton, "Training Products of Experts by Minimizing Contrastive Divergence" (2002)

3. **Top-k and Top-p Sampling**
   - Holtzman et al., "The Curious Case of Neural Text Degeneration" (2019)
     - Introduced nucleus (top-p) sampling
     - Analyzed issues with likelihood maximization and pure sampling
     - https://arxiv.org/abs/1904.09751

4. **Length Normalization for Beam Search**
   - Wu et al., "Google's Neural Machine Translation System" (2016)

### Advanced Decoding Methods

5. **Contrastive Decoding**
   - Li et al., "Contrastive Decoding: Open-ended Text Generation as Optimization" (2022)
   - https://arxiv.org/abs/2210.15097

6. **Typical Decoding**
   - Meister et al., "Typical Decoding for Natural Language Generation" (2022)
   - https://arxiv.org/abs/2202.00666

7. **Mirostat**
   - Basu et al., "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity" (2020)
   - https://arxiv.org/abs/2007.14966

### Analysis and Best Practices

8. **Analysis of Decoding Strategies**
   - Fan et al., "Hierarchical Neural Story Generation" (2018)
   - Analyzed sampling vs beam search for creative tasks

9. **Repetition in Neural Text Generation**
   - Welleck et al., "Neural Text Generation with Unlikelihood Training" (2019)
   - https://arxiv.org/abs/1908.04319

### Production Documentation

10. **OpenAI API Documentation**
    - Temperature, top_p, frequency_penalty, presence_penalty parameters
    - https://platform.openai.com/docs/api-reference/chat/create

11. **Hugging Face Generation Guide**
    - Comprehensive guide to generation parameters
    - https://huggingface.co/docs/transformers/generation_strategies

### Blog Posts and Tutorials

12. [How to Generate Text: Different Decoding Methods](https://huggingface.co/blog/how-to-generate) - Hugging Face
    - Excellent visual guide to different sampling strategies

13. [Controlling Text Generation](https://blog.fastforwardlabs.com/2019/05/22/automated-text-generation-using-gpt-2.html) - Fast Forward Labs

---

**Previous Chapter**: [Complete Transformer](11-complete-transformer.md) - Building complete transformer architectures

**Next Chapter**: [Flash Attention](13-flash-attention.md) - Memory-efficient attention implementation
