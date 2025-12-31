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

3. [Basic Attention](chapters/03-basic-attention.md)
   - Intuition behind attention
   - Dot-product attention
   - Scaled dot-product attention
   - Attention weights and visualization

4. [Multi-Head Attention](chapters/04-multi-head-attention.md)
   - Why multiple heads?
   - Implementing multi-head attention
   - Concatenation and projection

5. [Bidirectional vs Causal Attention](chapters/05-bidirectional-causal-attention.md)
   - Bidirectional (full) attention
   - Causal (masked) attention for autoregressive models
   - Attention masks implementation
   - When to use each type

6. [Cross-Attention](chapters/06-cross-attention.md)
   - Cross-attention for encoder-decoder models
   - Query, key, value from different sources
   - Applications in seq2seq and multimodal models

## Part 3: Positional Encoding

7. [Positional Encodings](chapters/07-positional-encodings.md)
   - Why position matters
   - Sinusoidal positional encoding
   - Learned positional embeddings
   - Relative positional encodings

8. [Rotary Position Embeddings (RoPE)](chapters/08-rope.md)
   - Rotary embeddings intuition
   - Mathematical formulation
   - Implementation and benefits
   - RoPE in modern LLMs

## Part 4: Transformer Architecture

9. [The Transformer Block](chapters/09-transformer-block.md)
   - Layer normalization (LayerNorm, RMSNorm)
   - Feed-forward networks
   - Residual connections
   - Pre-norm vs post-norm

10. [Activation Functions](chapters/10-activation-functions.md)
    - GELU and variants
    - SwiGLU and gated linear units
    - SiLU, GeGLU
    - Activation functions in modern LLMs

11. [Building a Complete Transformer](chapters/11-complete-transformer.md)
    - Encoder architecture
    - Decoder architecture
    - Encoder-decoder models
    - Decoder-only models (GPT-style)

## Part 5: Efficient Attention

12. [Flash Attention](chapters/12-flash-attention.md)
    - Memory bottlenecks in attention
    - IO-aware algorithm design
    - Tiling and recomputation
    - Implementation considerations

13. [Other Efficient Attention Variants](chapters/13-efficient-attention.md)
    - Linear attention
    - Sparse attention
    - Sliding window attention
    - Multi-query and grouped-query attention

14. [KV Cache](chapters/14-kv-cache.md)
    - Why KV cache exists
    - Memory analysis and scaling
    - Basic implementation
    - Position encoding interactions (RoPE)
    - Reducing cache size (MQA, GQA)
    - KV cache quantization (INT8, FP8)
    - Memory management (PagedAttention reference)
    - Streaming and long context
    - Production considerations

## Part 6: Training Large Language Models

15. [Language Model Training](chapters/15-lm-training.md)
    - Causal language modeling objective
    - Training loop implementation
    - Gradient accumulation
    - Mixed precision training

16. [Distributed Training and Parallelism](chapters/16-distributed-training.md)
    - Data Parallelism (DP, DDP)
    - Tensor Parallelism (TP)
    - Pipeline Parallelism (PP)
    - Fully Sharded Data Parallel (FSDP)
    - ZeRO optimization stages
    - 3D parallelism strategies

17. [Optimizers and Training Techniques](chapters/17-scaling-optimization.md)
    - AdamW and optimizer choices
    - Learning rate schedules (warmup, cosine, WSD)
    - Gradient clipping
    - Batch size scaling
    - Troubleshooting training issues

18. [Scaling Laws and Training Dynamics](chapters/18-scaling-dynamics.md)
    - Kaplan and Chinchilla scaling laws
    - Compute-optimal training
    - Grokking: delayed generalization
    - Double descent phenomenon
    - Emergent capabilities in LLMs
    - Neural scaling phenomena

## Part 7: Alignment and Fine-tuning

19. [Supervised Fine-tuning (SFT)](chapters/19-sft.md)
    - Instruction tuning
    - Dataset preparation
    - Fine-tuning strategies

20. [LoRA and Parameter-Efficient Fine-tuning](chapters/20-peft.md)
    - Low-Rank Adaptation (LoRA)
    - QLoRA (quantized LoRA)
    - Prefix tuning and prompt tuning
    - Adapters and other PEFT methods
    - When to use PEFT vs full fine-tuning

21. [Reinforcement Learning from Human Feedback (RLHF)](chapters/21-rlhf.md)
    - Reward modeling
    - PPO for language models
    - KL divergence constraints
    - Implementation walkthrough

22. [Direct Preference Optimization (DPO)](chapters/22-dpo.md)
    - From RLHF to DPO
    - Mathematical derivation
    - Implementation
    - Variants (IPO, KTO, etc.)

23. [Safety and Alignment Techniques](chapters/23-safety-alignment.md)
    - Constitutional AI (CAI)
    - Red teaming and adversarial testing
    - Harmlessness training
    - Refusal training and jailbreak prevention
    - Alignment tax and capability tradeoffs

## Part 8: Diffusion Models

24. [Diffusion Model Fundamentals](chapters/24-diffusion-fundamentals.md)
    - Forward diffusion process
    - Reverse denoising process
    - Score matching
    - DDPM formulation

25. [Implementing Diffusion Models](chapters/25-diffusion-implementation.md)
    - U-Net architecture
    - Noise scheduling
    - Training loop
    - Sampling algorithms (DDPM, DDIM)

26. [Advanced Diffusion Topics](chapters/26-diffusion-advanced.md)
    - Classifier-free guidance
    - Latent diffusion (Stable Diffusion)
    - Conditioning mechanisms
    - Recent advances

## Part 9: Advanced Capabilities

27. [Long Context Techniques](chapters/27-long-context.md)
    - RoPE scaling methods (NTK-aware, YaRN)
    - Attention sinks and StreamingLLM
    - Memory-augmented architectures
    - Landmark attention
    - Ring attention for distributed long context
    - Evaluation on long-range tasks

28. [Multimodality](chapters/28-multimodality.md)
    - Vision encoders (ViT, SigLIP)
    - Cross-modal attention mechanisms
    - Vision-language models (LLaVA, GPT-4V)
    - Audio and speech integration
    - Multimodal tokenization strategies

29. [Reasoning and Chain-of-Thought](chapters/29-reasoning.md)
    - Chain-of-thought prompting
    - Self-consistency and voting
    - Tree-of-thought reasoning
    - Process reward models
    - Reasoning traces and verification
    - Test-time compute scaling

## Part 10: Model Architectures in Practice

30. [Architecture Comparison: Modern LLMs](chapters/30-model-architectures.md)
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

31. [Model Merging and Distillation](chapters/31-merging-distillation.md)
    - Knowledge distillation techniques
    - Model merging (TIES, DARE, SLERP)
    - Pruning and sparsification
    - Weight averaging methods
    - Creating specialized models from general ones

## Part 11: Hardware and Optimization

32. [Hardware, Quantization, and Training Optimization](chapters/32-hardware-quantization-optimization.md)
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

## Part 12: Evaluation and Deployment

33. [Evaluation and Benchmarks](chapters/33-evaluation-benchmarks.md)
    - Perplexity and language modeling metrics
    - Common benchmarks (MMLU, HellaSwag, GSM8K, HumanEval)
    - Reasoning benchmarks (ARC, MATH, BigBench)
    - Safety and alignment evaluations
    - Human evaluation methods
    - Contamination detection and mitigation

## Appendices

- [Data Curation and Preprocessing](appendices/data-curation.md)
    - Data collection and filtering
    - Deduplication strategies
    - Quality filtering and scoring
    - Data mixing and curriculum learning
    - Tokenizer training data considerations
