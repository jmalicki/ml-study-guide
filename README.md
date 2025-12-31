# ML Interview Study Guide: LLMs and Beyond

A comprehensive, hands-on study guide for machine learning interviews with a focus on Large Language Models. Each chapter includes theoretical explanations with LaTeX notation and runnable PyTorch implementations.

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

This guide assumes familiarity with:
- Python and PyTorch basics
- Linear algebra fundamentals
- Basic machine learning concepts (gradient descent, backpropagation)
- Neural network basics (MLPs, CNNs)

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
