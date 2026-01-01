# Appendix: Fisher Information In Depth

The Fisher information matrix is one of the most profound concepts connecting statistics, information theory, and optimization. This appendix develops the theory from first principles.

## Historical Context

In 1922, R.A. Fisher introduced the concept of "information" in the context of statistical estimation. His key insight: **different parameterizations of the same model carry different amounts of information about the true parameter value**.

This led to:
- The theory of maximum likelihood estimation
- The Cramér-Rao lower bound on estimator variance
- The foundations of information geometry
- Modern natural gradient methods in machine learning

## Statistical Estimation: The Big Picture

Before diving into Fisher information, we need to understand the problem it solves. This section summarizes key ideas from mathematical statistics that most ML practitioners haven't formally studied.

### The Fundamental Problem

We observe data $X = (X_1, \ldots, X_n)$ generated from some distribution $p(x; \theta^\ast)$ where $\theta^\ast$ is the **true but unknown** parameter. Our goal: estimate $\theta^\ast$ from the data.

**Example**: Coin flips
- Data: $X_1, \ldots, X_n \in \{0, 1\}$ (outcomes of $n$ flips)
- Model: $X_i \sim \text{Bernoulli}(p)$
- Unknown parameter: $p$ = probability of heads
- Goal: Estimate $p$ from the observed flips

### What Is an Estimator?

An **estimator** $\hat{\theta}$ is any function of the data:

$$\hat{\theta} = g(X_1, \ldots, X_n)$$

For coin flips, some estimators of $p$:
- Sample mean: $\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i$
- First observation: $\hat{p} = X_1$
- Constant: $\hat{p} = 0.5$

All are valid estimators; some are better than others.

### Properties of Good Estimators

**1. Unbiasedness**: $\mathbb{E}[\hat{\theta}] = \theta$

The estimator is "right on average." The sample mean is unbiased for Bernoulli $p$:
$$\mathbb{E}\left[\frac{1}{n}\sum X_i\right] = \frac{1}{n} \sum \mathbb{E}[X_i] = \frac{1}{n} \cdot np = p$$

**2. Low Variance**: $\text{Var}(\hat{\theta})$ should be small

Among unbiased estimators, we prefer those with less variability.

**3. Mean Squared Error (MSE)**: $\text{MSE} = \text{Bias}^2 + \text{Variance}$

Trades off bias and variance. Sometimes biased estimators have lower MSE.

**4. Consistency**: $\hat{\theta}_n \xrightarrow{P} \theta$ as $n \to \infty$

With more data, the estimator converges to the truth.

```python
import torch
import numpy as np

def estimator_properties_demo():
    """Demonstrate key estimator properties."""
    torch.manual_seed(42)

    p_true = 0.7  # True probability

    print("Estimator properties for Bernoulli p=0.7")
    print("-" * 50)

    for n in [10, 100, 1000, 10000]:
        n_experiments = 5000
        estimates = []

        for _ in range(n_experiments):
            # Generate n coin flips
            data = torch.bernoulli(torch.full((n,), p_true))
            # Sample mean estimator
            estimate = data.mean().item()
            estimates.append(estimate)

        estimates = np.array(estimates)
        bias = estimates.mean() - p_true
        variance = estimates.var()
        mse = bias**2 + variance

        # Theoretical variance: p(1-p)/n
        var_theory = p_true * (1 - p_true) / n

        print(f"n={n:5d}: Bias={bias:+.4f}, Var={variance:.6f} "
              f"(theory: {var_theory:.6f}), MSE={mse:.6f}")
```

### Sufficiency: Capturing All Information

A statistic $T(X)$ is **sufficient** for $\theta$ if it captures all the information in the data about $\theta$. Formally:

$$p(X | T(X), \theta) = p(X | T(X))$$

Given the sufficient statistic, the full data provides no additional information about $\theta$.

**Example**: For $n$ coin flips, $T = \sum_{i=1}^n X_i$ (total heads) is sufficient for $p$.

Knowing the total is enough—the order doesn't matter for estimating $p$.

**The Neyman-Fisher Factorization Theorem**: $T(X)$ is sufficient iff:

$$p(x; \theta) = g(T(x), \theta) \cdot h(x)$$

The likelihood factors into a part depending on $\theta$ only through $T$, and a part independent of $\theta$.

**Why sufficiency matters for Fisher information**: The Fisher information in a sufficient statistic equals the Fisher information in the full data. Any data reduction beyond sufficiency loses information.

```python
def sufficiency_demo():
    """Demonstrate that sufficient statistics capture all information."""
    torch.manual_seed(42)

    p_true = 0.6
    n = 100
    n_experiments = 10000

    estimates_full = []  # Use full data
    estimates_sufficient = []  # Use only sum (sufficient statistic)

    for _ in range(n_experiments):
        data = torch.bernoulli(torch.full((n,), p_true))

        # Estimator using full data: sample mean
        est_full = data.mean().item()

        # Estimator using only sufficient statistic: sum/n
        sufficient_stat = data.sum().item()
        est_sufficient = sufficient_stat / n

        estimates_full.append(est_full)
        estimates_sufficient.append(est_sufficient)

    var_full = np.var(estimates_full)
    var_sufficient = np.var(estimates_sufficient)

    print("Sufficiency demonstration:")
    print(f"  Variance using full data: {var_full:.6f}")
    print(f"  Variance using sufficient stat: {var_sufficient:.6f}")
    print(f"  (These should be equal!)")
```

### Maximum Likelihood Estimation (MLE)

The **maximum likelihood estimator** is the parameter value that maximizes the probability of the observed data:

$$\hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^n p(X_i; \theta) = \arg\max_\theta \sum_{i=1}^n \log p(X_i; \theta)$$

**Why MLE is special**:

1. **Consistency**: $\hat{\theta}_{MLE} \xrightarrow{P} \theta^\ast$ as $n \to \infty$

2. **Asymptotic normality**:
$$\sqrt{n}(\hat{\theta}_{MLE} - \theta^\ast) \xrightarrow{d} \mathcal{N}(0, I(\theta^\ast)^{-1})$$

3. **Asymptotic efficiency**: MLE achieves the Cramér-Rao bound asymptotically

4. **Invariance**: If $\hat{\theta}$ is MLE for $\theta$, then $g(\hat{\theta})$ is MLE for $g(\theta)$

**The Fisher information appears naturally here**: The asymptotic variance of the MLE is exactly $I(\theta)^{-1}$.

```python
def mle_properties_demo():
    """Demonstrate MLE asymptotic normality."""
    torch.manual_seed(42)

    p_true = 0.3
    n_experiments = 10000

    print("MLE asymptotic normality for Bernoulli")
    print("√n(p̂ - p) should be N(0, 1/I(p)) = N(0, p(1-p))")
    print("-" * 55)

    for n in [100, 1000, 10000]:
        normalized_errors = []

        for _ in range(n_experiments):
            data = torch.bernoulli(torch.full((n,), p_true))
            p_mle = data.mean().item()

            # Normalized error
            error = np.sqrt(n) * (p_mle - p_true)
            normalized_errors.append(error)

        errors = np.array(normalized_errors)

        # Should be N(0, p(1-p))
        theoretical_var = p_true * (1 - p_true)
        empirical_var = errors.var()

        print(f"n={n:5d}: Empirical var={empirical_var:.4f}, "
              f"Theory={theoretical_var:.4f}")
```

### The Central Role of Fisher Information

Fisher information connects all these concepts:

| Concept | Role of Fisher Information |
|---------|---------------------------|
| Cramér-Rao Bound | $\text{Var}(\hat{\theta}) \geq 1/I(\theta)$ |
| MLE Asymptotic Variance | $\text{Var}(\hat{\theta}_{MLE}) \approx 1/(nI(\theta))$ |
| Sufficiency | Fisher info in $T(X)$ = Fisher info in $X$ iff $T$ sufficient |
| Efficiency | Estimator is efficient iff it achieves Cramér-Rao bound |
| Information Geometry | Fisher defines the natural metric on parameter space |

**The deep insight**: Fisher information quantifies **how much the data tells us about the parameter**. This single quantity determines the fundamental limits of statistical estimation.

## The Score Function

### Definition

For a probability distribution $p(x; \theta)$ parameterized by $\theta$, the **score function** is:

$$s(x; \theta) = \nabla_\theta \log p(x; \theta) = \frac{\nabla_\theta p(x; \theta)}{p(x; \theta)}$$

The score measures the sensitivity of the log-likelihood to parameter changes.

### Key Property: Zero Mean

The score has zero expectation under the model:

$$\mathbb{E}_{x \sim p_\theta}[s(x; \theta)] = 0$$

**Proof**:

$$\mathbb{E}[s(x; \theta)] = \int p(x; \theta) \frac{\nabla_\theta p(x; \theta)}{p(x; \theta)} dx = \nabla_\theta \int p(x; \theta) dx = \nabla_\theta 1 = 0$$

This uses the fact that $\int p(x; \theta) dx = 1$ for all $\theta$.

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Callable

def verify_score_zero_mean():
    """Empirically verify that the score has zero mean."""
    torch.manual_seed(42)

    # Gaussian distribution: p(x; μ) = N(x; μ, 1)
    # Score: s(x; μ) = x - μ
    mu_true = 2.0
    n_samples = 100000

    x = torch.randn(n_samples) + mu_true  # Sample from N(μ, 1)
    score = x - mu_true  # Score function

    print(f"Theoretical E[score]: 0")
    print(f"Empirical E[score]: {score.mean().item():.6f}")
    print(f"(Should be close to 0)")
```

## Fisher Information: Definition

### Scalar Parameter Case

For a single parameter $\theta \in \mathbb{R}$, the **Fisher information** is the variance of the score:

$$I(\theta) = \mathbb{E}_{x \sim p_\theta}\left[s(x; \theta)^2\right] = \mathbb{E}\left[\left(\frac{\partial \log p(x; \theta)}{\partial \theta}\right)^2\right]$$

Since the score has zero mean, this equals:

$$I(\theta) = \text{Var}_{x \sim p_\theta}[s(x; \theta)]$$

### Equivalent Form: Negative Expected Hessian

Under [regularity conditions](https://en.wikipedia.org/wiki/Leibniz_integral_rule) (interchanging differentiation and integration):

$$I(\theta) = -\mathbb{E}_{x \sim p_\theta}\left[\frac{\partial^2 \log p(x; \theta)}{\partial \theta^2}\right]$$

**Proof**:

$$\frac{\partial^2 \log p}{\partial \theta^2} = \frac{\partial}{\partial \theta}\left(\frac{1}{p}\frac{\partial p}{\partial \theta}\right) = \frac{1}{p}\frac{\partial^2 p}{\partial \theta^2} - \frac{1}{p^2}\left(\frac{\partial p}{\partial \theta}\right)^2$$

Taking expectations:
$$\mathbb{E}\left[\frac{\partial^2 \log p}{\partial \theta^2}\right] = \int \frac{\partial^2 p}{\partial \theta^2} dx - \mathbb{E}\left[\left(\frac{\partial \log p}{\partial \theta}\right)^2\right] = 0 - I(\theta)$$

```python
def fisher_two_forms():
    """Verify the two equivalent forms of Fisher information."""
    torch.manual_seed(42)

    # Poisson distribution: p(x; λ) = λ^x e^(-λ) / x!
    # log p(x; λ) = x log λ - λ - log(x!)
    # Score: s = x/λ - 1
    # Score²: (x/λ - 1)²
    # -d²log p/dλ² = x/λ²

    lambda_true = 3.0
    n_samples = 100000

    # Sample from Poisson
    x = torch.poisson(torch.full((n_samples,), lambda_true))

    # Form 1: E[score²]
    score = x / lambda_true - 1
    fisher_form1 = (score ** 2).mean()

    # Form 2: -E[d²log p / dλ²] = E[x/λ²]
    fisher_form2 = (x / lambda_true**2).mean()

    # Theoretical: I(λ) = 1/λ
    fisher_theory = 1 / lambda_true

    print(f"Fisher (score variance):    {fisher_form1.item():.4f}")
    print(f"Fisher (neg. Hessian):      {fisher_form2.item():.4f}")
    print(f"Fisher (theoretical 1/λ):   {fisher_theory:.4f}")
```

### Fisher Information Matrix

For vector parameters $\theta \in \mathbb{R}^d$, the Fisher information becomes a matrix:

$$F(\theta) = \mathbb{E}_{x \sim p_\theta}\left[\nabla_\theta \log p(x; \theta) \cdot \nabla_\theta \log p(x; \theta)^T\right]$$

The $(i,j)$ entry is:

$$F_{ij}(\theta) = \mathbb{E}\left[\frac{\partial \log p}{\partial \theta_i} \frac{\partial \log p}{\partial \theta_j}\right]$$

Equivalently:

$$F(\theta) = -\mathbb{E}_{x \sim p_\theta}\left[\nabla_\theta^2 \log p(x; \theta)\right]$$

## Properties of Fisher Information

### 1. Positive Semi-Definiteness

$F(\theta) \succeq 0$ always. For any vector $v$:

$$v^T F(\theta) v = \mathbb{E}\left[(v^T \nabla \log p)^2\right] \geq 0$$

The Fisher matrix is positive **definite** when the model is identifiable (different parameters give different distributions).

### 2. Additivity for Independent Samples

If $x_1, \ldots, x_n$ are i.i.d. from $p_\theta$:

$$I_n(\theta) = n \cdot I(\theta)$$

**Proof**: The log-likelihood of i.i.d. samples is additive:
$$\log p(x_1, \ldots, x_n; \theta) = \sum_{i=1}^n \log p(x_i; \theta)$$

Taking second derivatives and expectations gives the result.

```python
def fisher_additivity():
    """Demonstrate Fisher information additivity."""
    torch.manual_seed(42)

    # Bernoulli with parameter p
    p_true = 0.3

    # Single sample Fisher: I(p) = 1/(p(1-p))
    fisher_single = 1 / (p_true * (1 - p_true))

    # Empirically verify with n samples
    for n in [1, 10, 100]:
        n_trials = 10000
        total_fisher = 0

        for _ in range(n_trials):
            x = torch.bernoulli(torch.full((n,), p_true))
            # Score for Bernoulli: s = x/p - (1-x)/(1-p)
            score = x / p_true - (1 - x) / (1 - p_true)
            # Sum of scores for n samples
            total_score = score.sum()
            total_fisher += total_score ** 2

        empirical_fisher = total_fisher / n_trials
        theoretical_fisher = n * fisher_single

        print(f"n={n:3d}: Empirical={empirical_fisher:.2f}, Theoretical={theoretical_fisher:.2f}")
```

### 3. Sufficiency

If $T(x)$ is a sufficient statistic for $\theta$, then the Fisher information about $\theta$ in $T(x)$ equals the Fisher information in $x$.

This connects Fisher information to the Neyman-Fisher factorization theorem.

### 4. Data Processing Inequality

For any function $g$: $I_\theta(g(x)) \leq I_\theta(x)$

Processing data cannot increase information. Equality holds iff $g$ is a sufficient statistic.

## Fisher Information for Common Distributions

### Bernoulli Distribution

$X \sim \text{Bernoulli}(p)$: $P(X = 1) = p$, $P(X = 0) = 1-p$

$$\log p(x; p) = x \log p + (1-x) \log(1-p)$$

$$\frac{\partial \log p}{\partial p} = \frac{x}{p} - \frac{1-x}{1-p}$$

$$I(p) = \mathbb{E}\left[\left(\frac{X}{p} - \frac{1-X}{1-p}\right)^2\right] = \frac{1}{p(1-p)}$$

**Interpretation**: Information is highest when $p = 0.5$ (most uncertainty), lowest near $p = 0$ or $p = 1$.

```python
def bernoulli_fisher():
    """Fisher information for Bernoulli distribution."""
    ps = torch.linspace(0.01, 0.99, 100)
    fisher = 1 / (ps * (1 - ps))

    print("Bernoulli Fisher Information I(p) = 1/(p(1-p))")
    print(f"  I(0.5) = {1/(0.5*0.5):.2f} (maximum)")
    print(f"  I(0.1) = {1/(0.1*0.9):.2f}")
    print(f"  I(0.9) = {1/(0.9*0.1):.2f}")
```

### Gaussian Distribution: Known Variance

$X \sim \mathcal{N}(\mu, \sigma^2)$ with $\sigma^2$ known:

$$\log p(x; \mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

$$\frac{\partial \log p}{\partial \mu} = \frac{x - \mu}{\sigma^2}$$

$$I(\mu) = \mathbb{E}\left[\frac{(X-\mu)^2}{\sigma^4}\right] = \frac{1}{\sigma^2}$$

**Interpretation**: Smaller variance means more information about the mean.

### Gaussian Distribution: Known Mean

$X \sim \mathcal{N}(\mu, \sigma^2)$ with $\mu$ known:

$$\frac{\partial \log p}{\partial \sigma^2} = -\frac{1}{2\sigma^2} + \frac{(x-\mu)^2}{2\sigma^4}$$

$$I(\sigma^2) = \frac{1}{2\sigma^4}$$

### Gaussian Distribution: Both Unknown

The Fisher information matrix for $\theta = (\mu, \sigma^2)$:

$$F = \begin{bmatrix} \frac{1}{\sigma^2} & 0 \\ 0 & \frac{1}{2\sigma^4} \end{bmatrix}$$

The off-diagonal is zero—$\mu$ and $\sigma^2$ are orthogonal in the Fisher metric!

```python
def gaussian_fisher():
    """Fisher information for Gaussian distribution."""
    torch.manual_seed(42)

    mu_true, sigma_true = 0.0, 2.0
    n_samples = 100000

    x = torch.randn(n_samples) * sigma_true + mu_true

    # Score for mu
    score_mu = (x - mu_true) / sigma_true**2
    fisher_mu_empirical = (score_mu ** 2).mean()
    fisher_mu_theory = 1 / sigma_true**2

    # Score for sigma^2
    score_sigma2 = -1/(2*sigma_true**2) + (x - mu_true)**2 / (2*sigma_true**4)
    fisher_sigma2_empirical = (score_sigma2 ** 2).mean()
    fisher_sigma2_theory = 1 / (2 * sigma_true**4)

    # Cross term
    cross_term = (score_mu * score_sigma2).mean()

    print(f"Fisher for μ:     Empirical={fisher_mu_empirical:.4f}, Theory={fisher_mu_theory:.4f}")
    print(f"Fisher for σ²:    Empirical={fisher_sigma2_empirical:.6f}, Theory={fisher_sigma2_theory:.6f}")
    print(f"Cross term F_μσ²: {cross_term:.6f} (should be ≈0)")
```

### Poisson Distribution

$X \sim \text{Poisson}(\lambda)$:

$$\log p(x; \lambda) = x \log \lambda - \lambda - \log(x!)$$

$$I(\lambda) = \frac{1}{\lambda}$$

### Categorical (Multinomial) Distribution

For $K$ categories with probabilities $\pi = (\pi_1, \ldots, \pi_K)$ where $\sum_k \pi_k = 1$:

Using the constraint to eliminate $\pi_K = 1 - \sum_{k=1}^{K-1} \pi_k$:

$$F_{ij} = \frac{\delta_{ij}}{\pi_i} + \frac{1}{\pi_K}$$

where $\delta_{ij}$ is the Kronecker delta.

```python
def categorical_fisher():
    """Fisher information for categorical distribution."""
    # 3 categories with probabilities (0.2, 0.3, 0.5)
    pi = torch.tensor([0.2, 0.3, 0.5])
    K = len(pi)

    # Fisher matrix (using all K parameters, though redundant)
    F = torch.diag(1 / pi)

    print("Categorical Fisher (diagonal):")
    for i in range(K):
        print(f"  F_{i}{i} = 1/π_{i} = 1/{pi[i]:.1f} = {1/pi[i]:.2f}")
```

### Exponential Family (General Form)

For an exponential family:

$$p(x; \theta) = h(x) \exp(\eta(\theta)^T T(x) - A(\theta))$$

where:
- $\eta(\theta)$: natural parameters
- $T(x)$: sufficient statistics
- $A(\theta)$: log-partition function

The Fisher information is:

$$F(\theta) = \nabla_\theta \eta(\theta)^T \cdot \nabla_\eta^2 A(\eta) \cdot \nabla_\theta \eta(\theta)$$

In the natural parameterization ($\theta = \eta$):

$$F(\eta) = \nabla_\eta^2 A(\eta)$$

The Hessian of the log-partition function **is** the Fisher information!

```python
def exponential_family_fisher():
    """
    Fisher = Hessian of log-partition for exponential families.

    Example: Gaussian with natural parameters.
    p(x; η₁, η₂) ∝ exp(η₁x + η₂x²)
    where η₁ = μ/σ², η₂ = -1/(2σ²)
    """
    # For Gaussian: A(η) = -η₁²/(4η₂) - (1/2)log(-2η₂)
    # This is the log-partition function

    def log_partition(eta1, eta2):
        """Log partition function for Gaussian in natural params."""
        return -eta1**2 / (4 * eta2) - 0.5 * torch.log(-2 * eta2)

    # Verify Hessian = Fisher at some point
    mu, sigma = 1.0, 2.0
    eta1 = torch.tensor(mu / sigma**2, requires_grad=True)
    eta2 = torch.tensor(-1 / (2 * sigma**2), requires_grad=True)

    A = log_partition(eta1, eta2)

    # Compute Hessian via autograd
    grad_A = torch.autograd.grad(A, [eta1, eta2], create_graph=True)

    print("Exponential family: Fisher = Hessian of log-partition")
    print(f"  ∂A/∂η₁ = {grad_A[0].item():.4f}")
    print(f"  ∂A/∂η₂ = {grad_A[1].item():.4f}")
```

## The Cramér-Rao Bound

### Statement

For any unbiased estimator $\hat{\theta}(X)$ of $\theta$:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

**The variance of any unbiased estimator is bounded below by the inverse Fisher information.**

### Multivariate Version

For unbiased $\hat{\theta} \in \mathbb{R}^d$:

$$\text{Cov}(\hat{\theta}) \succeq F(\theta)^{-1}$$

in the positive semi-definite sense.

### Proof (Scalar Case)

Let $s(X) = \partial \log p(X;\theta) / \partial \theta$ be the score.

For any unbiased estimator $\hat{\theta}$:
$$\mathbb{E}[\hat{\theta}] = \theta$$

Differentiating with respect to $\theta$:
$$\frac{\partial}{\partial \theta} \int \hat{\theta}(x) p(x; \theta) dx = 1$$

$$\int \hat{\theta}(x) \frac{\partial p}{\partial \theta} dx = 1$$

$$\int \hat{\theta}(x) p(x) \frac{\partial \log p}{\partial \theta} dx = 1$$

$$\mathbb{E}[\hat{\theta} \cdot s] = 1$$

Since $\mathbb{E}[s] = 0$:
$$\text{Cov}(\hat{\theta}, s) = \mathbb{E}[\hat{\theta} \cdot s] - \mathbb{E}[\hat{\theta}]\mathbb{E}[s] = 1$$

By the Cauchy-Schwarz inequality:
$$|\text{Cov}(\hat{\theta}, s)|^2 \leq \text{Var}(\hat{\theta}) \cdot \text{Var}(s)$$

$$1 \leq \text{Var}(\hat{\theta}) \cdot I(\theta)$$

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

```python
def cramer_rao_demo():
    """Demonstrate the Cramér-Rao bound."""
    torch.manual_seed(42)

    # Estimate mean of Gaussian with known variance
    sigma = 2.0
    mu_true = 3.0

    # Fisher information: I(μ) = 1/σ² = 0.25
    fisher = 1 / sigma**2
    cramer_rao_bound = 1 / fisher  # = σ² = 4.0

    # For n samples, Fisher becomes n/σ², so CR bound becomes σ²/n

    n_experiments = 10000

    for n_samples in [1, 10, 100]:
        estimates = []
        for _ in range(n_experiments):
            x = torch.randn(n_samples) * sigma + mu_true
            estimate = x.mean().item()  # MLE = sample mean
            estimates.append(estimate)

        empirical_var = np.var(estimates)
        cr_bound = sigma**2 / n_samples

        print(f"n={n_samples:3d}: Var(μ̂)={empirical_var:.4f}, CR bound={cr_bound:.4f}")
        print(f"       Ratio: {empirical_var/cr_bound:.3f} (should be ≥ 1)")
```

### Efficiency

An estimator that achieves the Cramér-Rao bound is called **efficient**.

- Maximum likelihood estimators are asymptotically efficient
- The sample mean is efficient for Gaussian mean estimation
- Not all distributions have efficient estimators

## Information Geometry

### The Fisher Metric

The Fisher information matrix defines a **Riemannian metric** on parameter space:

$$ds^2 = d\theta^T F(\theta) d\theta$$

This measures "distance" in terms of how distinguishable the corresponding distributions are.

### Geodesics

The shortest path between two distributions (in KL divergence sense) follows geodesics of the Fisher metric.

For exponential families, these geodesics have elegant closed forms.

### Connection to KL Divergence

For infinitesimal parameter changes:

$$D_{KL}(p_\theta \| p_{\theta + d\theta}) \approx \frac{1}{2} d\theta^T F(\theta) d\theta$$

**Proof**:

$$D_{KL}(p_\theta \| p_{\theta + d\theta}) = \mathbb{E}_{p_\theta}\left[\log \frac{p_\theta(x)}{p_{\theta + d\theta}(x)}\right]$$

Taylor expanding:
$$\log p_{\theta + d\theta}(x) \approx \log p_\theta(x) + d\theta^T \nabla_\theta \log p_\theta(x) + \frac{1}{2} d\theta^T \nabla^2_\theta \log p_\theta(x) d\theta$$

Taking expectations and using $\mathbb{E}[\nabla \log p] = 0$:
$$D_{KL} \approx -\frac{1}{2} d\theta^T \mathbb{E}[\nabla^2 \log p] d\theta = \frac{1}{2} d\theta^T F(\theta) d\theta$$

```python
def kl_fisher_approximation():
    """Verify KL ≈ (1/2) dθᵀ F dθ for small perturbations."""

    # Gaussian example
    mu1, sigma = 0.0, 1.0

    for d_mu in [0.01, 0.1, 0.5, 1.0]:
        mu2 = mu1 + d_mu

        # Exact KL divergence for Gaussians with same variance
        kl_exact = (mu2 - mu1)**2 / (2 * sigma**2)

        # Fisher approximation: (1/2) * dμ² * I(μ) = (1/2) * dμ² / σ²
        fisher = 1 / sigma**2
        kl_approx = 0.5 * d_mu**2 * fisher

        print(f"dμ={d_mu:.2f}: KL_exact={kl_exact:.4f}, KL_approx={kl_approx:.4f}, "
              f"ratio={kl_exact/kl_approx:.3f}")
```

## Fisher Information in Neural Networks

### The Challenge

For a neural network $f(x; \theta)$ with parameters $\theta \in \mathbb{R}^n$, the Fisher matrix is $n \times n$.

For GPT-3 (175B parameters): The full Fisher would require $10^{22}$ entries—impossible to store.

### Empirical Fisher vs True Fisher

**True Fisher**: Sample $y$ from the model's own distribution:

$$F(\theta) = \mathbb{E}_{x \sim \text{data}} \mathbb{E}_{y \sim p(\cdot|x;\theta)}\left[\nabla \log p(y|x;\theta) \nabla \log p(y|x;\theta)^T\right]$$

**Empirical Fisher**: Use the actual label:

$$\hat{F}(\theta) = \mathbb{E}_{(x,y) \sim \text{data}}\left[\nabla \log p(y|x;\theta) \nabla \log p(y|x;\theta)^T\right]$$

The empirical Fisher is easier to compute but loses theoretical properties:
- Not guaranteed positive semi-definite at convergence
- Not equivalent to expected Hessian
- Still used in practice (Adam, etc.)

```python
class FisherComparison:
    """Compare true and empirical Fisher information."""

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_empirical_fisher_diagonal(
        self,
        dataloader,
        n_samples: int = 1000
    ) -> dict:
        """Empirical Fisher using actual labels."""
        fisher = {name: torch.zeros_like(p)
                  for name, p in self.model.named_parameters()}

        count = 0
        for x, y in dataloader:
            if count >= n_samples:
                break

            self.model.zero_grad()
            logits = self.model(x)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Use actual label
            loss = -log_probs.gather(1, y.unsqueeze(1)).mean()
            loss.backward()

            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad ** 2

            count += x.shape[0]

        for name in fisher:
            fisher[name] /= count

        return fisher

    def compute_true_fisher_diagonal(
        self,
        dataloader,
        n_samples: int = 1000
    ) -> dict:
        """True Fisher: sample y from model's distribution."""
        fisher = {name: torch.zeros_like(p)
                  for name, p in self.model.named_parameters()}

        count = 0
        for x, _ in dataloader:
            if count >= n_samples:
                break

            self.model.zero_grad()
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)

            # Sample from model's distribution
            y_sampled = torch.multinomial(probs, 1).squeeze()

            log_probs = torch.log_softmax(logits, dim=-1)
            loss = -log_probs.gather(1, y_sampled.unsqueeze(1)).mean()
            loss.backward()

            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad ** 2

            count += x.shape[0]

        for name in fisher:
            fisher[name] /= count

        return fisher
```

### Connection to Hessian and Gauss-Newton

For negative log-likelihood loss $L(\theta) = -\log p(y|x;\theta)$:

$$\nabla^2 L = -\nabla^2 \log p = \underbrace{\mathbb{E}[\nabla \log p \cdot \nabla \log p^T]}_{\text{Fisher}} + \underbrace{\text{(residual terms)}}_{\to 0 \text{ at optimum}}$$

At the optimum (where residuals are small):
$$\text{Hessian} \approx \text{Fisher}$$

For Gaussian likelihoods:
$$\text{Fisher} = \text{Gauss-Newton matrix}$$

This connects three seemingly different concepts!

### Why Fisher Matters for Optimization

1. **Natural gradient**: $\theta_{t+1} = \theta_t - \eta F^{-1} \nabla L$ is reparameterization-invariant

2. **Adam approximation**: Adam's second moment is approximately diagonal Fisher

3. **K-FAC**: Uses Kronecker-factored Fisher for practical second-order optimization

4. **Elastic Weight Consolidation (EWC)**: Uses Fisher to identify important parameters for continual learning

```python
class ElasticWeightConsolidation:
    """
    EWC uses Fisher information to prevent catastrophic forgetting.

    Parameters with high Fisher information were important for
    previous tasks and should change less.
    """

    def __init__(self, model: nn.Module, fisher_diag: dict,
                 old_params: dict, lambda_ewc: float = 1000):
        self.model = model
        self.fisher_diag = fisher_diag
        self.old_params = old_params
        self.lambda_ewc = lambda_ewc

    def penalty(self) -> torch.Tensor:
        """EWC penalty: Σᵢ Fᵢ(θᵢ - θᵢ*)²"""
        loss = 0
        for name, p in self.model.named_parameters():
            if name in self.fisher_diag:
                loss += (self.fisher_diag[name] *
                        (p - self.old_params[name]) ** 2).sum()
        return 0.5 * self.lambda_ewc * loss
```

## Summary: Why Fisher Information Matters

1. **Fundamental limit**: The Cramér-Rao bound says Fisher determines the best possible estimation accuracy

2. **Natural metric**: Fisher defines the natural geometry of probability distributions

3. **Connects to Hessian**: For ML estimation, Fisher ≈ Hessian near the optimum

4. **Reparameterization invariance**: Natural gradient using Fisher is independent of how we parameterize the model

5. **Practical approximations**: Adam ≈ diagonal Fisher; K-FAC ≈ block-diagonal Kronecker Fisher

6. **Beyond optimization**: Fisher appears in information theory, continual learning, compression, and more

## Exercises

1. **Derive Fisher for Exponential distribution**: $p(x; \lambda) = \lambda e^{-\lambda x}$ for $x \geq 0$. Verify both forms agree.

2. **Multivariate Gaussian Fisher**: Derive the full Fisher matrix for $\mathcal{N}(\mu, \Sigma)$ where both $\mu$ and $\Sigma$ are unknown.

3. **Efficiency of MLE**: Show that for Bernoulli, the MLE $\hat{p} = \bar{X}$ achieves the Cramér-Rao bound.

4. **Empirical vs True Fisher**: Train a small network and compare the diagonal entries of empirical and true Fisher. When do they differ most?

5. **Fisher and curvature**: Visualize how the Fisher metric "stretches" parameter space differently in different directions for a simple 2D example.

## Further Reading

- [Fisher, R.A. "On the Mathematical Foundations of Theoretical Statistics" (1922)](https://royalsocietypublishing.org/doi/10.1098/rsta.1922.0009) — The original paper

- [Amari, S. "Information Geometry and Its Applications" (2016)](https://www.springer.com/gp/book/9784431559771) — Comprehensive treatment of information geometry

- [Ly, A. et al. "A Tutorial on Fisher Information" (2017)](https://arxiv.org/abs/1705.01064) — Accessible introduction

- [Martens, J. "New Insights and Perspectives on the Natural Gradient Method" (2020)](https://jmlr.org/papers/v21/17-678.html) — Deep connections to optimization
