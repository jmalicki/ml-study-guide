# Table of Contents

## Part 1: Foundations

1. [Tokenization](chapters/01-tokenization.md)
   - Character-level tokenization
   - Word-level tokenization
   - Subword tokenization (BPE, WordPiece, SentencePiece)
   - Building a tokenizer from scratch

2. [Embeddings](chapters/02-embeddings.md)
   - Word embeddings
   - Learned embeddings
   - Embedding layers in PyTorch

## Part 2: Attention Mechanisms

1. [Basic Attention](chapters/03-basic-attention.md)
   - Intuition behind attention
   - Dot-product attention
   - Scaled dot-product attention
   - Attention weights and visualization

2. [Multi-Head Attention](chapters/04-multi-head-attention.md)
   - Why multiple heads?
   - Implementing multi-head attention
   - Concatenation and projection

3. [Bidirectional vs Causal Attention](chapters/05-bidirectional-causal-attention.md)
   - Bidirectional (full) attention
   - Causal (masked) attention for autoregressive models
   - Attention masks implementation
   - When to use each type

4. [Cross-Attention](chapters/06-cross-attention.md)
   - Cross-attention for encoder-decoder models
   - Query, key, value from different sources
   - Applications in seq2seq and multimodal models

## Part 3: Positional Encoding

1. [Positional Encodings](chapters/07-positional-encodings.md)
   - Why position matters
   - Sinusoidal positional encoding
   - Learned positional embeddings
   - Relative positional encodings

2. [Rotary Position Embeddings (RoPE)](chapters/08-rope.md)
   - Rotary embeddings intuition
   - Mathematical formulation
   - Implementation and benefits
   - RoPE in modern LLMs

## Part 4: Transformer Architecture

1. [The Transformer Block](chapters/09-transformer-block.md)
   - Layer normalization (LayerNorm, RMSNorm)
   - Feed-forward networks
   - Residual connections
   - Pre-norm vs post-norm

2. [Activation Functions](chapters/10-activation-functions.md)
   - GELU and variants
   - SwiGLU and gated linear units
   - SiLU, GeGLU
   - Activation functions in modern LLMs

3. [Building a Complete Transformer](chapters/11-complete-transformer.md)
   - Encoder architecture
   - Decoder architecture
   - Encoder-decoder models
   - Decoder-only models (GPT-style)

## Part 5: Inference and Generation

1. [Inference and Generation Strategies](chapters/12-inference-generation.md)
   - Autoregressive generation basics
   - Greedy decoding
   - Temperature scaling
   - Top-k sampling
   - Nucleus (Top-p) sampling
   - Beam search
   - Repetition penalties
   - Stop conditions
   - Connection to KV Cache and Speculative Decoding

## Part 6: Efficient Attention

1. [Flash Attention](chapters/13-flash-attention.md)
   - Memory bottlenecks in attention
   - IO-aware algorithm design
   - Tiling and recomputation
   - Implementation considerations

2. [Other Efficient Attention Variants](chapters/14-efficient-attention.md)
   - Linear attention
   - Sparse attention
   - Sliding window attention
   - Multi-query and grouped-query attention

3. [KV Cache](chapters/15-kv-cache.md)
   - Why KV cache exists
   - Memory analysis and scaling
   - Basic implementation
   - Position encoding interactions (RoPE)
   - Reducing cache size (MQA, GQA)
   - KV cache quantization (INT8, FP8)
   - Memory management (PagedAttention reference)
   - Streaming and long context
   - Production considerations

## Part 7: Training Large Language Models

1. [Language Model Training](chapters/16-lm-training.md)
   - Causal language modeling objective
   - Training loop implementation
   - Gradient accumulation
   - Mixed precision training

2. [Optimizers and Training Techniques](chapters/17-scaling-optimization.md)
   - AdamW and optimizer choices
   - Learning rate schedules (warmup, cosine, WSD)
   - Gradient clipping
   - Batch size scaling
   - Troubleshooting training issues

3. [Scaling Laws and Training Dynamics](chapters/12-scaling-dynamics.md)
   - Kaplan and Chinchilla scaling laws
   - Compute-optimal training
   - Grokking: delayed generalization
   - Double descent phenomenon
   - Emergent capabilities in LLMs
   - Neural scaling phenomena

## Part 8: Alignment and Fine-tuning

1. [Supervised Fine-tuning (SFT)](chapters/18-sft.md)
   - Instruction tuning
   - Dataset preparation
   - Fine-tuning strategies

2. [LoRA and Parameter-Efficient Fine-tuning](chapters/19-peft.md)
   - Low-Rank Adaptation (LoRA)
   - QLoRA (quantized LoRA)
   - Prefix tuning and prompt tuning
   - Adapters and other PEFT methods
   - When to use PEFT vs full fine-tuning

3. [Reinforcement Learning from Human Feedback (RLHF)](chapters/20-rlhf.md)
   - Reward modeling
   - PPO for language models
   - KL divergence constraints
   - Implementation walkthrough

4. [Direct Preference Optimization (DPO)](chapters/21-dpo.md)
   - From RLHF to DPO
   - Mathematical derivation
   - Implementation
   - Variants (IPO, KTO, etc.)

5. [Safety and Alignment Techniques](chapters/22-safety-alignment.md)
   - Constitutional AI (CAI)
   - Red teaming and adversarial testing
   - Harmlessness training
   - Refusal training and jailbreak prevention
   - Alignment tax and capability tradeoffs

## Part 9: Advanced Capabilities

1. [Long Context Techniques](chapters/23-long-context.md)
   - RoPE scaling methods (NTK-aware, YaRN)
   - Attention sinks and StreamingLLM
   - Memory-augmented architectures
   - Landmark attention
   - Ring attention for distributed long context
   - Evaluation on long-range tasks

2. [Multimodality](chapters/24-multimodality.md)
   - Vision encoders (ViT, SigLIP)
   - Cross-modal attention mechanisms
   - Vision-language models (LLaVA, GPT-4V)
   - Audio and speech integration
   - Multimodal tokenization strategies

3. [In-Context Learning](chapters/25-in-context-learning.md)
   - Zero-shot, one-shot, and few-shot learning
   - Induction heads and mechanistic interpretability
   - Theoretical frameworks (meta-learning, Bayesian inference, implicit gradient descent)
   - How ICL emerges during training
   - ICL vs fine-tuning tradeoffs
   - Advanced ICL techniques
   - Connection to reasoning

4. [Reasoning and Chain-of-Thought](chapters/26-reasoning.md)
   - Chain-of-thought prompting
   - Self-consistency and voting
   - Tree-of-thought reasoning
   - Process reward models
   - Reasoning traces and verification
   - Test-time compute scaling

## Part 10: Model Architectures in Practice

1. [Architecture Comparison: Modern LLMs](chapters/27-model-architectures.md)
   - GPT series (GPT-2, GPT-3, GPT-4)
   - Claude (Anthropic)
   - Gemini (Google DeepMind)
     - MoE architecture, 1M+ token context
     - Gemini 1.5, 2.0, 2.5, 3.0 evolution
   - LLaMA series (LLaMA, LLaMA 2, LLaMA 3, LLaMA 4)
   - Qwen series (Qwen, Qwen 2.5, Qwen 3)
   - Mistral and Mixtral
   - Other notable models (Gemma, Phi, DeepSeek)
   - WeDLM (Tencent)
     - Diffusion language model with causal attention
     - Parallel decoding with KV cache compatibility
   - Comparison table: techniques used by each model
     - Attention type (MHA, MQA, GQA)
     - Positional encoding (learned, RoPE, ALiBi)
     - Normalization (pre-norm, RMSNorm)
     - Activation functions (GELU, SwiGLU)
     - Context length and scaling techniques
     - Mixture of Experts (MoE) vs dense
     - Autoregressive vs diffusion-based generation

2. [Model Merging and Distillation](chapters/28-merging-distillation.md)
   - Knowledge distillation techniques
   - Model merging (TIES, DARE, SLERP)
   - Pruning and sparsification
   - Weight averaging methods
   - Creating specialized models from general ones

## Part 11: Hardware and Optimization

1. [Hardware, Quantization, and Training Optimization](chapters/29-hardware-quantization-optimization.md)
   - **Hardware**
     - NVIDIA GPUs (Ampere, Hopper, Blackwell)
     - Google TPUs (v5, v6 Trillium, v7 Ironwood)
     - Tensor Cores and precision formats
   - **Quantization**
     - Post-training quantization (PTQ) and QAT
     - GPTQ and AWQ for GPU inference
     - GGUF and K-quants for CPU inference (llama.cpp)
     - FP8 and FP4 (Blackwell) formats
   - **Mixed Precision Training**
     - BF16 vs FP16
     - FP8 training (DeepSeek approach)
     - NVIDIA Transformer Engine
   - **Memory Optimization**
     - Flash Attention (1, 2, 3)
     - KV cache quantization
     - PagedAttention (vLLM)
   - **Inference Acceleration**
     - Speculative decoding
     - Continuous batching
   - **Optimizers**
     - AdamW
     - Muon (2x efficiency for hidden layers)
     - Shampoo and SOAP
   - **Learning Rate Schedules**
     - Cosine with warmup
     - Warmup-Stable-Decay (WSD)

## Part 12: Distributed Training

1. [Distributed Training and Parallelism](chapters/30-distributed-training.md)
   - Data Parallelism (DP, DDP)
   - Tensor Parallelism (TP)
   - Pipeline Parallelism (PP)
   - Fully Sharded Data Parallel (FSDP)
   - ZeRO optimization stages
   - 3D parallelism strategies

## Part 13: Evaluation and Deployment

1. [Evaluation and Benchmarks](chapters/31-evaluation-benchmarks.md)
   - Perplexity and language modeling metrics
   - Common benchmarks (MMLU, HellaSwag, GSM8K, HumanEval)
   - Reasoning benchmarks (ARC, MATH, BigBench)
   - Safety and alignment evaluations
   - Human evaluation methods
   - Contamination detection and mitigation

## Part 14: Diffusion Models

1. [Diffusion Model Fundamentals](chapters/32-diffusion-fundamentals.md)
   - Forward diffusion process
   - Reverse denoising process
   - Score matching
   - DDPM formulation

2. [Implementing Diffusion Models](chapters/33-diffusion-implementation.md)
   - U-Net architecture
   - Noise scheduling
   - Training loop
   - Sampling algorithms (DDPM, DDIM)

3. [Advanced Diffusion Topics](chapters/34-diffusion-advanced.md)
   - Classifier-free guidance
   - Latent diffusion (Stable Diffusion)
   - Conditioning mechanisms
   - Recent advances

## Appendices

- [Data Curation and Preprocessing](appendices/data-curation.md)
  - Data collection and filtering
  - Deduplication strategies
  - Quality filtering and scoring
  - Data mixing and curriculum learning
  - Tokenizer training data considerations
