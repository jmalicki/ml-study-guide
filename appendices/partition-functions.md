# Appendix: Partition Functions and Why They're Intractable for LLMs

This appendix explains the partition function, why it appears throughout probabilistic machine learning, and specifically why it becomes computationally intractable in the context of language models.

## Table of Contents

1. [What is a Partition Function?](#what-is-a-partition-function)
2. [The Partition Function in LLMs](#the-partition-function-in-llms)
3. [Why It's Intractable](#why-its-intractable)
4. [Concrete Example: The Scale of the Problem](#concrete-example-the-scale-of-the-problem)
5. [**The Key Breakthrough: How DPO Cancels Z(x)**](#the-key-breakthrough-how-dpo-cancels-zx)
6. [Other Methods for Handling Intractability](#other-methods-for-handling-intractability)
7. [Connection to RLHF and DPO](#connection-to-rlhf-and-dpo)

---

## What is a Partition Function?

In probability theory, when we define an unnormalized probability distribution, the **partition function** is the normalizing constant that makes probabilities sum to 1.

Given an energy function $E(x)$ or score function $s(x)$, we define:

```math
p(x) = \frac{1}{Z} \exp(-E(x)) = \frac{1}{Z} \exp(s(x))
```

where the partition function $Z$ is:

```math
Z = \sum_{x \in \mathcal{X}} \exp(s(x))
```

For continuous distributions, the sum becomes an integral.

**The partition function ensures probabilities sum to 1:**

```math
\sum_{x} p(x) = \sum_{x} \frac{\exp(s(x))}{Z} = \frac{1}{Z} \sum_{x} \exp(s(x)) = \frac{Z}{Z} = 1
```

---

## The Partition Function in LLMs

In the context of language models and RLHF, the partition function appears when we want to define a policy (probability distribution over responses) in terms of a reward function.

### The RLHF Optimal Policy

In RLHF, we optimize:

```math
\max_\pi \mathbb{E}_{y \sim \pi(\cdot|x)}[r(x, y)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})
```

The optimal solution has a closed form:

```math
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right)
```

where:

```math
Z(x) = \sum_{y \in \mathcal{Y}} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right)
```

**Here, $\mathcal{Y}$ is the set of ALL possible responses** - every possible sequence of tokens the model could generate.

---

## Why It's Intractable

The fundamental problem is that **the space of possible responses is astronomically large**.

### The Combinatorial Explosion

For a language model with:
- Vocabulary size $V$ (e.g., 50,000 tokens)
- Maximum response length $L$ (e.g., 1,000 tokens)

The number of possible responses is:

```math
|\mathcal{Y}| = V^L = 50,000^{1000}
```

To put this in perspective:
- The number of atoms in the observable universe is approximately $10^{80}$
- $50,000^{1000} \approx 10^{4,699}$

**You cannot enumerate, sum over, or even sample a meaningful fraction of this space.**

### What the Partition Function Requires

Computing $Z(x)$ requires:

```math
Z(x) = \sum_{y \in \mathcal{Y}} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right)
```

This means:
1. **Enumerate** every possible response $y$ (impossible - there are too many)
2. **Compute** $\pi_{\text{ref}}(y|x)$ for each (requires a forward pass per response)
3. **Compute** $r(x, y)$ for each (requires running reward model)
4. **Sum** all the terms

Even if each computation took 1 nanosecond, computing the partition function for a single prompt would take longer than the age of the universe by a factor of $10^{4,600}$.

---

## Concrete Example: The Scale of the Problem

Let's make this concrete with a tiny example:

### Toy Example: 3-Token Vocabulary, Max Length 4

Even with an absurdly small vocabulary of just 3 tokens (`[A, B, C]`) and maximum length 4:

```
Possible responses:
Length 1: A, B, C                           (3 responses)
Length 2: AA, AB, AC, BA, BB, BC, CA, CB, CC (9 responses)
Length 3: AAA, AAB, ..., CCC                (27 responses)
Length 4: AAAA, AAAB, ..., CCCC             (81 responses)

Total: 3 + 9 + 27 + 81 = 120 responses
```

For this toy case, we could enumerate all 120 responses and compute $Z(x)$.

### Real LLM: 50K Vocabulary, 100 Token Responses

```
Possible responses of exactly 100 tokens: 50,000^100 ≈ 10^470
```

This is **impossible to enumerate**. There is no algorithm, no computer, no amount of time that can sum over this space.

### The Autoregressive Factorization Doesn't Help

You might think: "But LLMs generate token-by-token, so can't we use the chain rule?"

```math
\pi(y|x) = \prod_{t=1}^{T} \pi(y_t | x, y_{<t})
```

This factorization helps us **sample** from the distribution and **compute the probability of a specific sequence**, but it doesn't help compute the partition function because:

```math
Z(x) = \sum_{y_1} \sum_{y_2} \cdots \sum_{y_T} \prod_{t=1}^{T} \pi_{\text{ref}}(y_t | x, y_{<t}) \exp\left(\frac{r(x, y_{1:T})}{\beta}\right)
```

The reward $r(x, y)$ depends on the **complete** sequence, so we can't factorize the exponential term. The sums don't simplify.

---

## The Key Breakthrough: How DPO Cancels Z(x)

This section explains the central mathematical insight that makes Direct Preference Optimization (DPO) possible: **the partition function completely cancels out when we work with preference pairs**.

This is not just a minor optimization—it transforms an utterly intractable problem ($10^{4,699}$ terms to sum) into a simple, tractable supervised learning objective. Understanding this cancellation is essential to understanding why DPO works.

### The Setup: Expressing Reward in Terms of Policy

Recall the optimal RLHF policy:

```math
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right)
```

We can rearrange this to express the reward in terms of the policy:

```math
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
```

**Notice that $Z(x)$ appears as an additive constant.** It depends only on the prompt $x$, not on which response $y$ we're evaluating.

### The Cancellation: Reward Differences Eliminate Z(x)

Now consider two responses to the same prompt: a preferred response $y_w$ (winner) and a dispreferred response $y_l$ (loser).

The reward for each is:

```math
r(x, y_w) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x)
```

```math
r(x, y_l) = \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \beta \log Z(x)
```

When we compute the **difference** in rewards:

```math
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \cancel{\beta \log Z(x)} - \cancel{\beta \log Z(x)}
```

**The $Z(x)$ terms cancel exactly!** The intractable partition function—the sum over $10^{4,699}$ possible sequences—simply disappears from the equation.

### Why This Matters: From Intractable to Tractable

| Without Cancellation | With Cancellation |
|---------------------|-------------------|
| Must compute $Z(x) = \sum_{y \in \mathcal{Y}} \pi_{\text{ref}}(y\|x) \exp(r(x,y)/\beta)$ | Only need $\pi(y_w\|x)$ and $\pi(y_l\|x)$ |
| Sum over $V^L \approx 10^{4,699}$ terms | Evaluate 2 sequences |
| Computationally impossible | Simple forward passes |
| Requires RL sampling loop | Direct supervised learning |

This is why DPO is called "Direct" Preference Optimization—it bypasses the intractable normalization entirely.

### The Bradley-Terry Connection

The reason preferences work is the Bradley-Terry model of pairwise comparison:

```math
p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
```

Human preferences are modeled as depending only on the **difference** in rewards, not their absolute values. This is psychologically motivated—humans compare options relative to each other, not against some absolute scale.

Since the Bradley-Terry model only uses reward differences, and $Z(x)$ cancels in reward differences, we never need to compute the partition function at all!

### The DPO Loss Function

Substituting our reward expression (with $Z(x)$ cancelled) into the Bradley-Terry model, we get the DPO loss:

```math
\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
```

This loss requires only:
1. Forward pass to compute $\pi_\theta(y_w|x)$ and $\pi_\theta(y_l|x)$
2. Forward pass to compute $\pi_{\text{ref}}(y_w|x)$ and $\pi_{\text{ref}}(y_l|x)$
3. Standard cross-entropy-style optimization

**No sampling, no RL, no reward model, no partition function.**

### Significance for Practitioners

The Z(x) cancellation is not just mathematically elegant—it has profound practical implications:

1. **Training simplicity**: DPO is just supervised learning on preference pairs. You can use standard optimizers, learning rate schedules, and training infrastructure.

2. **Computational efficiency**: No need for sampling responses (expensive for long sequences), training a separate reward model, or running PPO with its complex critic and value head.

3. **Stability**: RL training is notoriously unstable (reward hacking, mode collapse, hyperparameter sensitivity). DPO's supervised objective is much more stable.

4. **Sample efficiency**: Every preference pair provides direct supervision. In RLHF, samples are used to estimate gradients with high variance.

---

## Other Methods for Handling Intractability

While DPO's Z-cancellation is the most elegant solution for preference learning, other methods handle partition function intractability differently:

### 1. Avoid Computing Z Directly (Sampling Methods)

**Policy Gradient / REINFORCE / PPO:**

Instead of computing $Z(x)$, we sample responses from the current policy and estimate gradients:

```math
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(y|x) \cdot A(x, y)]
```

This requires only:
- Sampling (which LLMs do naturally via autoregressive generation)
- Computing log probability of the sampled sequence (tractable)
- Estimating advantage $A(x, y)$ for the sampled sequence

**No partition function needed!** But this approach requires sampling many responses and suffers from high variance gradient estimates.

### 2. Approximate Z (Contrastive Methods)

**Noise Contrastive Estimation (NCE):**

Approximate the partition function using negative samples:

```math
Z \approx 1 + \sum_{i=1}^{k} \frac{p_{\text{model}}(y_i)}{p_{\text{noise}}(y_i)}
```

This approach powered Word2Vec's negative sampling, which transformed an intractable $O(V)$ softmax into an $O(k)$ binary classification problem. The same insight—avoiding direct computation of the partition function—laid conceptual groundwork for DPO's more elegant algebraic cancellation. See [Chapter 2: Embeddings](../chapters/02-embeddings.md#word2vec) for the Word2Vec treatment.

### 4. Design Models Where Z is Tractable

**Autoregressive LLMs (for next-token prediction):**

For standard language modeling, the partition function at each step is just a sum over the vocabulary:

```math
Z_t = \sum_{v \in V} \exp(\text{logit}_v)
```

With vocabulary size 50,000, this is completely tractable - it's what softmax computes!

The intractability only arises when we try to define distributions over **complete sequences** with rewards that depend on the full sequence.

---

## Connection to RLHF and DPO

### Why RLHF Uses RL (Not Direct Optimization)

The partition function intractability is **the reason** RLHF uses reinforcement learning:

1. We want: $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$
2. We can't normalize this (intractable $Z$)
3. Instead, we use policy gradient to find $\pi^*$ iteratively via sampling

### Why DPO is "Direct"

DPO circumvents the partition function entirely:

1. Express the implicit reward in terms of policies
2. Use preference pairs where $Z(x)$ cancels in the difference
3. Optimize directly on the resulting loss function

This is why DPO is more computationally efficient than RLHF - it avoids the RL loop entirely by exploiting the cancellation of $Z(x)$.

### The Bradley-Terry Model

The preference model:

```math
p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
```

Only depends on reward **differences**, which is why the partition function cancels. This is the key mathematical insight that makes DPO possible.

---

## Summary

| Aspect | Explanation |
|--------|-------------|
| **What is Z?** | Normalizing constant that makes probabilities sum to 1 |
| **Why intractable for LLMs?** | Sum over all possible token sequences: $V^L$ possibilities |
| **Scale of problem** | $50,000^{1000} \approx 10^{4,699}$ - vastly larger than atoms in universe |
| **How RLHF handles it** | Uses sampling (policy gradient) instead of computing Z |
| **How DPO handles it** | **Preference differences cancel Z out entirely** |
| **When Z is tractable** | Per-token softmax (vocabulary-sized sum) |

**Key Takeaway:** The partition function over complete sequences is fundamentally intractable for language models—there are more possible token sequences than atoms in the universe. DPO's central insight is that by working with preference pairs and reward *differences*, the partition function cancels algebraically. This transforms an impossible problem into straightforward supervised learning, which is why DPO has become the preferred method for preference learning in modern LLMs.

---

## Further Reading

- [Energy-Based Models](https://arxiv.org/abs/2101.03288) - General treatment of partition functions in ML
- [DPO Paper](https://arxiv.org/abs/2305.18290) - Original derivation showing Z cancellation
- [RLHF Paper](https://arxiv.org/abs/2203.02155) - Why RL is used instead of direct optimization
