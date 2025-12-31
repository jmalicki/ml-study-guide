# ML Interview Study Guide: LLMs and Beyond

A comprehensive, hands-on study guide for machine learning interviews with a focus on Large Language Models. Each chapter includes theoretical explanations with LaTeX notation and runnable PyTorch implementations.

This is my own study guide, largely written by Claude Opus 4.5.  My philosophy was to give Claude a basic concept of an outline, have it write an outline that I reviewed, and get it to write most of it.

After the basics were there, I have reviewed and studied the guide as I have gone along, but using the source materials and published papers as primary references, using this guide as only "Cliff's Notes" to help understand,
and have been correcting it as I have gone along (and there have been many huge, qualitative even, mistakes!!!).  Yet, I have found it helpful - an attempt at code and commentary in a single style written in a cohesive
narrative to compare papers to, and when the Claude AI output has been wrong, that has even sharpened my skills, as Claude mistakes are reasonably likely to be common misconceptions - testing myself against that
IMO strengthens my knowledge.

Part of the value of this method of study for me has been, rather than merely correcting things I feel are wrong, I have to actively *argue* with the LLM into correcting the guide for me once I find an error.
This, to me, aligns with the model of learning of "see one, do one, teach one" - I have to actively teach and explain to the LLM why it is wrong in a convincing way - if I can't eloquently convince the LLM
to do the right thing, do I actually understand what I am talking about?  To me, that identifies a weakness in my understanding rather than a problem with the LLMs (most of the time, at least).

## What's Covered

This guide covers the complete stack of modern LLM development:

- **Foundations**: Tokenization, embeddings, and the building blocks of language models
- **Attention Mechanisms**: From basic attention to multi-head, cross-attention, and efficient variants like Flash Attention
- **Transformer Architecture**: Layer normalization, activation functions, and building complete transformers
- **Training at Scale**: Distributed training, parallelism strategies, optimizers, and scaling laws
- **Alignment**: SFT, RLHF, DPO, and safety techniques
- **Diffusion Models**: Fundamentals through advanced topics like latent diffusion
- **Advanced Capabilities**: Long context, multimodality, and reasoning
- **Production**: Hardware considerations, quantization, and evaluation

## Table of Contents

See the full **[Table of Contents](TABLE_OF_CONTENTS.md)** for all 33 chapters organized into 12 parts.

## How to Use This Guide

Each chapter is designed to be self-contained but builds on previous concepts. The recommended approach:

1. Read the theoretical explanation and mathematical formulations
2. Study the PyTorch implementation
3. Run the code examples
4. Complete the exercises at the end of each chapter

## Prerequisites

This guide is designed for ML practitioners preparing for interviews or deepening their LLM knowledge. While we build concepts incrementally, certain foundational knowledge is assumed.

### Essential Prerequisites

**Python and PyTorch**
- Comfortable writing Python classes and functions
- PyTorch tensor operations, autograd, and nn.Module
- Building and training basic models (forward/backward passes)
- Used throughout for all implementations

**Linear Algebra**
- Matrix multiplication, transpose operations
- Vector dot products and norms
- Understanding of high-dimensional spaces
- Critical for understanding attention mechanisms and embeddings

**Calculus and Optimization**
- Gradients and partial derivatives
- Chain rule for backpropagation
- Basic gradient descent (SGD)
- Foundation for training procedures and advanced optimizers

**Deep Learning Fundamentals**
- Multi-layer perceptrons (MLPs) and feed-forward networks
- Backpropagation through neural networks
- Loss functions (cross-entropy, MSE)
- Overfitting, underfitting, and regularization
- Train/validation/test splits
- Assumed when discussing transformer FFNs and training loops

### Strongly Recommended

**Probability and Statistics**
- Gaussian/normal distributions
- Expectation and variance
- Maximum likelihood estimation
- KL divergence and entropy
- Required for RLHF, DPO, and diffusion models

**Basic NLP Concepts**
- Text as sequences of tokens
- Language modeling intuition (predicting next words)
- Concept of vocabulary and tokenization
- While we cover tokenization in depth, basic familiarity helps

**Optimization Beyond SGD**
- Momentum and adaptive learning rates (RMSprop basics)
- Understanding of why Adam exists
- We review AdamW in detail, but basic optimizer intuition is assumed

### Helpful but Not Required

**Advanced ML Concepts**
- Attention mechanisms at a high level (we teach this from scratch)
- Variational Autoencoders (VAEs) and ELBO framework
- Helpful for diffusion models section, but not strictly required
- Markov chains (useful for understanding diffusion processes)

**Hardware Awareness**
- Basic understanding of GPU memory and compute
- Helpful for distributed training and optimization chapters
- We explain specifics, but general awareness improves comprehension

### What We Don't Assume

- Prior experience with transformers or attention (taught from fundamentals)
- Knowledge of specific LLM architectures (GPT, LLaMA, etc.)
- Understanding of RLHF, DPO, or alignment techniques
- Diffusion models or generative modeling beyond basics
- Production deployment or serving infrastructure

## Requirements

```bash
pip install torch numpy matplotlib tqdm
```

## Development

This project includes validation tools to ensure quality:

```bash
make validate        # Validate internal links
make check-svg       # Check for inline SVG (not supported on GitHub)
make validate-svg    # Validate SVG files with SVGO
make check-latex     # Validate LaTeX syntax
make check           # Run all checks
```

## License

MIT License
