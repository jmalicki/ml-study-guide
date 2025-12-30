# Chapter 19: Supervised Fine-tuning (SFT)

Supervised Fine-tuning (SFT) is the critical bridge between pre-trained language models and useful assistants. This chapter covers the theory, practice, and implementation of instruction tuning that transforms a base model into a helpful, instruction-following system.

## Table of Contents

1. [Overview](#overview)
2. [Instruction Tuning Fundamentals](#instruction-tuning-fundamentals)
3. [Dataset Preparation](#dataset-preparation)
4. [Chat Templates and Formatting](#chat-templates-and-formatting)
5. [Fine-tuning Strategies](#fine-tuning-strategies)
6. [Multi-turn Conversation Handling](#multi-turn-conversation-handling)
7. [Implementation](#implementation)
8. [Loss Masking and Training Details](#loss-masking-and-training-details)
9. [Best Practices and Common Pitfalls](#best-practices-and-common-pitfalls)
10. [Troubleshooting](#troubleshooting)
11. [Evaluation](#evaluation)
12. [Summary](#summary)
13. [References](#references)
14. [Exercises](#exercises)

---

## Overview

### What is Supervised Fine-tuning?

Pre-trained language models (see [Language Model Training](15-lm-training.md)) are trained to predict the next token in arbitrary text. While powerful, they lack the ability to follow instructions or engage in helpful dialogue. **Supervised Fine-tuning (SFT)** adapts these models to:

1. Follow instructions ("Write a poem about...")
2. Answer questions helpfully
3. Engage in multi-turn conversations
4. Refuse harmful requests
5. Maintain consistent personas

### The SFT Pipeline

<svg viewBox="0 0 800 150" xmlns="http://www.w3.org/2000/svg" style="max-width: 800px; width: 100%; height: auto;">
  <!-- Pre-trained Base Model Box -->
  <rect x="20" y="20" width="180" height="110" fill="#f5f5f5" stroke="#4A90A4" stroke-width="2" rx="5"/>
  <text x="110" y="60" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="#333" font-weight="600">Pre-trained</text>
  <text x="110" y="80" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="#333" font-weight="600">Base Model</text>
  <text x="110" y="105" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#666">(LLM Training)</text>

  <!-- First Arrow -->
  <defs>
    <marker id="arrowhead1" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#4A90A4"/>
    </marker>
  </defs>
  <line x1="200" y1="75" x2="270" y2="75" stroke="#4A90A4" stroke-width="2" marker-end="url(#arrowhead1)"/>

  <!-- Supervised Fine-tuning Box -->
  <rect x="270" y="20" width="180" height="110" fill="#f5f5f5" stroke="#4A90A4" stroke-width="2" rx="5"/>
  <text x="360" y="60" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="#333" font-weight="600">Supervised</text>
  <text x="360" y="80" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="#333" font-weight="600">Fine-tuning</text>
  <text x="360" y="105" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#666">(SFT)</text>

  <!-- Second Arrow -->
  <defs>
    <marker id="arrowhead2" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#4A90A4"/>
    </marker>
  </defs>
  <line x1="450" y1="75" x2="520" y2="75" stroke="#4A90A4" stroke-width="2" marker-end="url(#arrowhead2)"/>

  <!-- Aligned Model Box -->
  <rect x="520" y="20" width="180" height="110" fill="#f5f5f5" stroke="#4A90A4" stroke-width="2" rx="5"/>
  <text x="610" y="60" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="#333" font-weight="600">Aligned</text>
  <text x="610" y="80" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="16" fill="#333" font-weight="600">Model</text>
  <text x="610" y="105" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="13" fill="#666">(RLHF)</text>
</svg>

SFT is typically followed by alignment techniques like RLHF (see [RLHF](21-rlhf.md)) or DPO (see [DPO](22-dpo.md)), but a well-executed SFT phase is critical for final model quality.

### Key Papers

The field of instruction tuning was established by several seminal works:

- **FLAN** (2021): [Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) - Google demonstrated that instruction tuning dramatically improves zero-shot performance
- **InstructGPT** (2022): [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - OpenAI's approach combining SFT and RLHF
- **Self-Instruct** (2022): [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560) - Automated dataset generation
- **Alpaca** (2023): [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) - Stanford's open reproduction using GPT-3.5 generated data

---

## Instruction Tuning Fundamentals

### Objective Function

SFT uses standard causal language modeling, but only on **instruction-response pairs**:

$$
\mathcal{L}_{\text{SFT}} = -\sum_{(x, y) \in \mathcal{D}} \log p_\theta(y \mid x)
$$

where:
- $\mathcal{D}$ is the instruction dataset
- $x$ is the instruction/prompt
- $y$ is the desired response
- $\theta$ are the model parameters

### Key Difference from Pre-training

| Aspect | Pre-training | Supervised Fine-tuning |
|--------|--------------|------------------------|
| **Objective** | Predict next token in general text | Follow instructions and respond helpfully |
| **Data** | Web crawl, books (trillions of tokens) | Curated instruction-response pairs (10K-1M examples) |
| **Loss** | All tokens contribute equally | Often mask instruction tokens (only learn from responses) |
| **Learning Rate** | Higher (~1e-4 to 6e-4) | Lower (~1e-5 to 5e-5) |
| **Duration** | Weeks/months on thousands of GPUs | Hours/days on handful of GPUs |

### The Instruction-Response Format

A typical instruction example:

```json
{
  "instruction": "What is the capital of France?",
  "response": "The capital of France is Paris."
}
```

Or with input context:

```json
{
  "instruction": "Summarize the following text:",
  "input": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet at least once.",
  "response": "This is a pangram - a sentence containing all 26 letters of the alphabet."
}
```

---

## Dataset Preparation

### Dataset Sources

**1. Human-Written Examples**
- Highest quality, most expensive
- Examples: Dolly-15k, OpenAssistant Conversations

**2. Model-Generated (Distillation)**
- Use strong models (GPT-4, Claude) to generate responses
- Examples: Alpaca (52K), WizardLM, Orca

**3. Transformation of Existing Datasets**
- Convert NLP datasets to instruction format
- Example: FLAN converted 60+ datasets

**4. Synthetic Self-Instruct**
- Generate instructions and responses automatically
- Risk of quality issues and model collapse

### Dataset Quality Principles

```python
# Good instruction
{
  "instruction": "Write a haiku about spring",
  "response": "Cherry blossoms bloom\nSoft petals dance in the wind\nNature awakens"
}

# Bad instruction (too vague)
{
  "instruction": "Write something",
  "response": "Hello world"
}

# Bad instruction (harmful)
{
  "instruction": "How do I break into a car?",
  "response": "I cannot provide assistance with illegal activities..."
}
```

**Quality Criteria:**
1. **Diversity**: Cover many task types, domains, complexities
2. **Clarity**: Instructions should be unambiguous
3. **Correctness**: Responses must be factually accurate
4. **Helpfulness**: Responses should directly address the instruction
5. **Safety**: Include refusals for harmful requests
6. **Conciseness**: Avoid unnecessarily verbose responses

### Data Mixing Strategy

For production models, mix multiple sources:

```python
dataset_mix = {
    "general_qa": 0.3,           # General knowledge
    "coding": 0.2,                # Code generation
    "math": 0.15,                 # Mathematical reasoning
    "conversation": 0.15,         # Chat/dialogue
    "creative_writing": 0.1,      # Stories, poems
    "safety_refusals": 0.05,      # Teaching the model to refuse
    "multilingual": 0.05          # Non-English
}
```

---

## Chat Templates and Formatting

### The Role of Special Tokens

Chat models use special tokens to demarcate different parts of a conversation:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

2+2 equals 4.<|eot_id|>
```

### Common Chat Templates

**LLaMA 2 Chat Format:**
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is the capital of France? [/INST] The capital of France is Paris.</s>
```

**ChatML Format (OpenAI):**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

**LLaMA 3/3.1 Format:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The capital of France is Paris.<|eot_id|>
```

### Implementing a Chat Template

#### The Problem: Structuring Conversations for Language Models

Pre-trained language models are trained on unstructured text without explicit conversational structure. To enable coherent multi-turn dialogue, we need a standardized way to:
1. Distinguish between different speakers (system, user, assistant)
2. Mark turn boundaries so the model knows when to stop generating
3. Handle context from previous turns consistently
4. Enable the model to learn role-appropriate behavior

#### Theoretical Foundation

Chat templates solve the **role disambiguation problem** in dialogue modeling. Without explicit markers, the model cannot distinguish between:
- Instructions it should follow (user messages)
- Examples it should emulate (assistant messages)
- Behavioral guidelines (system messages)

The template acts as a **structured prompt** that provides positional and semantic cues through special tokens. This is analogous to providing type signatures in programming - it constrains the model's interpretation space and enables more reliable behavior.

#### Comparison to Alternatives

**Alternative Approaches:**
1. **Unstructured prompting**: Simply concatenate messages without markers
   - Problem: Model may confuse who is speaking
   - Problem: No clear stopping points

2. **Natural language markers**: Use phrases like "User said:" and "Assistant replied:"
   - Problem: Wastes tokens on verbose markers
   - Problem: Less precise than dedicated special tokens

3. **Special token-based templates** (our approach):
   - Advantage: Compact representation
   - Advantage: Unambiguous role markers
   - Advantage: Can be masked precisely during training

#### Key Insight: Template Consistency

The critical insight is that **the template used during SFT must match the template (if any) used during pre-training**. Models that were pre-trained with a chat template have learned to associate specific tokens with role transitions. Changing the template can confuse the model and degrade performance. If fine-tuning a base model without chat training, you have freedom to choose any consistent template.

```python
import torch
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ChatMessage:
    role: str  # "system", "user", or "assistant"
    content: str

class ChatTemplate:
    """Generic chat template handler."""

    def __init__(self, tokenizer, template_type: str = "llama3"):
        self.tokenizer = tokenizer
        self.template_type = template_type

        # Define special tokens for different templates
        self.templates = {
            "llama3": {
                "bos": "<|begin_of_text|>",
                "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
                "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
                "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "eot": "<|eot_id|>",
            },
            "chatml": {
                "system_start": "<|im_start|>system\n",
                "user_start": "<|im_start|>user\n",
                "assistant_start": "<|im_start|>assistant\n",
                "eot": "<|im_end|>\n",
            }
        }

    def format_messages(self, messages: List[ChatMessage]) -> str:
        """Convert list of messages to formatted string."""
        template = self.templates[self.template_type]
        formatted = ""

        if self.template_type == "llama3":
            formatted += template["bos"]

        for msg in messages:
            if msg.role == "system":
                formatted += template["system_start"] + msg.content + template["eot"]
            elif msg.role == "user":
                formatted += template["user_start"] + msg.content + template["eot"]
            elif msg.role == "assistant":
                formatted += template["assistant_start"] + msg.content + template["eot"]

        return formatted

    def apply_template(
        self,
        messages: List[ChatMessage],
        add_generation_prompt: bool = False
    ) -> str:
        """Apply chat template and optionally add generation prompt."""
        formatted = self.format_messages(messages)

        if add_generation_prompt:
            template = self.templates[self.template_type]
            formatted += template["assistant_start"]

        return formatted

# Example usage
messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is 2+2?"),
    ChatMessage(role="assistant", content="2+2 equals 4."),
]

chat_template = ChatTemplate(tokenizer=None, template_type="llama3")
formatted = chat_template.apply_template(messages)
print(formatted)
```

### Why Templates Matter

1. **Consistency**: Model learns to recognize conversation structure
2. **Multi-turn**: Enables coherent multi-turn dialogue
3. **System Prompts**: Allows setting model behavior
4. **Generation**: Model knows when to start/stop generating

---

## Fine-tuning Strategies

### Full Fine-tuning vs PEFT

| Method | Parameters Updated | Memory | Training Time | Quality |
|--------|-------------------|--------|---------------|---------|
| **Full Fine-tuning** | All (~7B for LLaMA-7B) | Very High | Slow | Best |
| **LoRA** (see [LoRA and PEFT](20-peft.md)) | ~0.1-1% | Low | Fast | Very Good |
| **QLoRA** | ~0.1-1% (quantized base) | Very Low | Fast | Good |

For SFT specifically:

- **Full fine-tuning**: Best for critical applications, when compute available
- **LoRA**: Standard choice for most use cases
- **QLoRA**: When memory constrained (single GPU)

### Computational Requirements

Understanding the computational resources needed for SFT is critical for planning:

| Model Size | Batch Size | GPU Memory | Training Time (10K examples, 3 epochs) | GPU Type |
|-----------|-----------|------------|----------------------------------------|----------|
| 1.5B      | 4         | 16GB       | 1-2 hours                              | RTX 4090, V100 |
| 3B        | 4         | 24GB       | 2-4 hours                              | RTX A5000, A10 |
| 7B        | 2         | 40GB       | 4-8 hours                              | A100 40GB |
| 13B       | 1         | 48GB       | 8-16 hours                             | A100 40GB (with gradient checkpointing) |
| 13B       | 2         | 80GB       | 6-12 hours                             | A100 80GB |
| 70B       | 1         | 80GB       | 2-4 days                               | A100 80GB (requires multi-GPU) |

**Notes:**
- Memory estimates assume BF16/FP16 training with gradient checkpointing enabled
- Training time assumes V100 or better GPUs
- For LoRA fine-tuning, divide memory requirements by ~4
- Multi-GPU training with FSDP can significantly reduce per-GPU memory requirements

**Memory Optimization Techniques:**

```python
# Enable gradient checkpointing to reduce memory
model.gradient_checkpointing_enable()

# Use BF16 mixed precision
training_args = TrainingArguments(
    bf16=True,  # Requires Ampere or newer GPUs (A100, RTX 30xx+)
    bf16_full_eval=True,
)

# For extreme memory constraints, use QLoRA (4-bit quantization)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Hyperparameters

**Learning Rate:**
```python
# Typical values
lr_full_finetuning = 1e-5  # to 5e-5
lr_lora = 1e-4             # to 3e-4

# Warmup
warmup_ratio = 0.03  # 3% of total steps
```

**Batch Size:**
```python
# Effective batch size should be 32-128
per_device_batch_size = 4
gradient_accumulation_steps = 8
# Effective: 4 * 8 = 32 (per device)
# If 4 GPUs: 32 * 4 = 128 total
```

**Training Duration:**
```python
# Rule of thumb: 1-3 epochs on instruction data
# Too few: Underfitting
# Too many: Overfitting, catastrophic forgetting

epochs = 3
# For 10K examples with batch_size=128:
# Steps = (10000 / 128) * 3 ≈ 235 steps
```

**Optimizer Configuration:**

The training script uses specific AdamW hyperparameters that differ from standard defaults:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),    # Note: Different from default (0.9, 0.999)
    weight_decay=0.1      # Note: Higher than typical 0.01
)
```

**Why these non-standard values?**

1. **Beta2 = 0.95 (instead of 0.999)**:
   - The second moment estimate (moving average of squared gradients) has a shorter memory
   - Results in faster adaptation to the instruction-following task
   - Commonly used in LLM fine-tuning following GPT-3/LLaMA training configurations
   - Trade-off: Less stable but faster convergence for short fine-tuning runs

2. **Weight Decay = 0.1 (instead of 0.01)**:
   - Stronger regularization helps prevent overfitting on small instruction datasets
   - Particularly important when fine-tuning on <100K examples
   - Helps preserve pre-trained knowledge (reduces catastrophic forgetting)
   - Can be reduced to 0.01 for larger datasets (>1M examples) or when using LoRA

**Alternative configurations:**

```python
# More conservative (better for very small datasets)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

# Standard (for large datasets or when stability is critical)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# For LoRA (lower weight decay since fewer parameters)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.01  # Lower than full fine-tuning
)
```

### Preventing Catastrophic Forgetting

Fine-tuning can cause the model to "forget" general knowledge:

**1. Small Learning Rates**
- Use 10x smaller LR than pre-training

**2. Limited Training Duration**
- 1-3 epochs typically sufficient

**3. Data Mixing**
- Mix in some pre-training data (10-20%)

**4. Parameter-Efficient Methods**
- LoRA naturally preserves base model knowledge

---

## Multi-turn Conversation Handling

### Conversation Structure

A multi-turn conversation:

```python
conversation = {
    "messages": [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 15 * 7?"},
        {"role": "assistant", "content": "15 * 7 = 105"},
        {"role": "user", "content": "How did you calculate that?"},
        {"role": "assistant", "content": "I multiplied 15 by 7. One approach: 15 * 7 = (10 * 7) + (5 * 7) = 70 + 35 = 105"}
    ]
}
```

### Processing Multi-turn Data

#### Problem: Context Accumulation in Dialogue

Multi-turn conversations present unique challenges compared to single instruction-response pairs:
1. **Context dependency**: Later turns depend on understanding earlier exchanges
2. **Attribution**: We must correctly mask which utterances are user vs assistant
3. **Coherence**: The model must maintain consistent information across turns
4. **Efficiency**: Longer sequences consume more memory and compute

The key question: how do we train on conversations while ensuring the model only learns to generate assistant responses, not user queries?

#### Theoretical Motivation: Conditional Generation Across Turns

For a conversation with $k$ turns, we want the model to learn:
$$p(a_i \mid u_1, a_1, \ldots, u_i) \quad \text{for } i = 1, \ldots, k$$

where $u_i$ is the $i$-th user message and $a_i$ is the $i$-th assistant response. This requires:
- Attending to all previous context (no masking in attention)
- Only computing loss on assistant tokens (masking in loss calculation)
- Preserving turn boundaries to avoid confusion

#### Comparison to Alternative Approaches

**Alternative 1: Treat each turn as independent**
- Train on $(u_i, a_i)$ pairs separately
- Problem: Loses conversational context
- Problem: Can't handle references like "it" or "that"

**Alternative 2: Train on full conversation with uniform loss**
- Compute loss on all tokens
- Problem: Model learns to generate both sides of conversation
- Problem: May produce confusing outputs mixing roles

**Alternative 3: Multi-turn with masking** (our approach)
- Full context in attention, masked loss on assistant turns
- Advantage: Learns from context while focusing on response generation
- Advantage: Natural handling of pronouns and references

#### Key Insight: Cumulative Masking Pattern

The masking pattern for multi-turn conversations is cumulative:

<svg viewBox="0 0 700 260" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; width: 100%; height: auto;">
  <!-- SYSTEM Row -->
  <rect x="20" y="20" width="400" height="40" fill="#f5f5f5" stroke="#4A90A4" stroke-width="2" rx="3"/>
  <text x="30" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#333" font-weight="600">[SYSTEM: instructions]</text>
  <line x1="440" y1="40" x2="520" y2="40" stroke="#999" stroke-width="2" marker-end="url(#arrow-gray)"/>
  <text x="530" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#999">masked</text>

  <!-- USER 1 Row -->
  <rect x="20" y="75" width="400" height="40" fill="#f5f5f5" stroke="#4A90A4" stroke-width="2" rx="3"/>
  <text x="30" y="100" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#333" font-weight="600">[USER: question 1]</text>
  <line x1="440" y1="95" x2="520" y2="95" stroke="#999" stroke-width="2" marker-end="url(#arrow-gray)"/>
  <text x="530" y="100" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#999">masked</text>

  <!-- ASSISTANT 1 Row -->
  <rect x="20" y="130" width="400" height="40" fill="#e8f4f8" stroke="#4A90A4" stroke-width="3" rx="3"/>
  <text x="30" y="155" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#333" font-weight="600">[ASSISTANT: answer 1]</text>
  <line x1="440" y1="150" x2="520" y2="150" stroke="#4A90A4" stroke-width="2" marker-end="url(#arrow-blue)"/>
  <text x="530" y="155" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#4A90A4" font-weight="600">LEARNED</text>

  <!-- USER 2 Row -->
  <rect x="20" y="185" width="400" height="40" fill="#f5f5f5" stroke="#4A90A4" stroke-width="2" rx="3"/>
  <text x="30" y="210" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#333" font-weight="600">[USER: question 2]</text>
  <line x1="440" y1="205" x2="520" y2="205" stroke="#999" stroke-width="2" marker-end="url(#arrow-gray)"/>
  <text x="530" y="210" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#999">masked</text>

  <!-- ASSISTANT 2 Row -->
  <rect x="20" y="240" width="400" height="40" fill="#e8f4f8" stroke="#4A90A4" stroke-width="3" rx="3"/>
  <text x="30" y="265" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#333" font-weight="600">[ASSISTANT: answer 2]</text>
  <line x1="440" y1="260" x2="520" y2="260" stroke="#4A90A4" stroke-width="2" marker-end="url(#arrow-blue)"/>
  <text x="530" y="265" font-family="system-ui, -apple-system, sans-serif" font-size="14" fill="#4A90A4" font-weight="600">LEARNED</text>

  <!-- Arrow markers -->
  <defs>
    <marker id="arrow-gray" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#999"/>
    </marker>
    <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <polygon points="0 0, 10 3, 0 6" fill="#4A90A4"/>
    </marker>
  </defs>
</svg>

Each assistant response is learned conditioned on all previous context. This teaches the model to maintain conversation state while only optimizing response generation.

```python
def prepare_multi_turn_conversation(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int = 2048
) -> Dict[str, torch.Tensor]:
    """
    Prepare a multi-turn conversation for training.

    Returns tokenized input with labels masked for non-assistant turns.
    """
    # Format with chat template
    formatted = format_chat_messages(messages, tokenizer)

    # Tokenize
    encodings = tokenizer(
        formatted,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"][0]

    # Create labels (will mask non-assistant parts)
    labels = input_ids.clone()

    # Mask all tokens initially
    labels[:] = -100

    # Find assistant response tokens and unmask them
    # (Implementation depends on template - see next section)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": encodings["attention_mask"][0]
    }
```

### Context Window Management

#### Problem: Conversations Exceeding Context Length

Real-world conversations often exceed the model's maximum context window (typically 2048-8192 tokens). When a conversation is too long, we must decide:
1. Which turns to keep vs discard
2. Whether to summarize or truncate
3. How to maintain conversation coherence
4. Whether to preserve task-critical information

Simply truncating from the beginning or end often loses important context.

#### Why Recent Context Matters Most

Empirical studies show that for instruction-following, **recent turns are more important than distant history**. This aligns with how humans converse - we rely heavily on the last few exchanges while earlier context provides general background.

The theoretical justification comes from attention patterns: transformer models naturally attend more strongly to recent tokens, so training on recent context is more sample-efficient.

#### Comparison to Alternatives

**Alternative 1: Head truncation (keep most recent)**
- Simple: drop oldest turns
- Advantage: Preserves what human would remember
- Disadvantage: Loses potential task context from early turns

**Alternative 2: Tail truncation (keep oldest)**
- Keep initial context including system message
- Disadvantage: Loses immediate conversation context
- Rarely used in practice

**Alternative 3: Smart truncation (keep system + recent)**
- Preserve system message for behavioral instructions
- Keep as many recent turns as fit
- Advantage: Balances task specification and immediate context
- Our approach: Recommended best practice

**Alternative 4: Summarization**
- Use another model to summarize dropped turns
- Advantage: Preserves information density
- Disadvantage: Adds complexity and latency
- Best for critical applications

#### Key Insight: Turn-Level Granularity

The critical insight is to truncate at **turn boundaries**, not mid-sentence. This preserves the natural structure of conversation and ensures the model sees complete exchanges. Truncating mid-turn can confuse the model about conversation structure.

For very long conversations:

```python
def truncate_conversation(
    messages: List[Dict],
    max_tokens: int,
    tokenizer,
    keep_system: bool = True
) -> List[Dict]:
    """
    Truncate conversation to fit in context window.

    Strategy:
    1. Always keep system message
    2. Keep most recent messages
    3. Ensure we keep complete turns
    """
    if keep_system and messages[0]["role"] == "system":
        system_msg = [messages[0]]
        messages = messages[1:]
    else:
        system_msg = []

    # Work backwards to keep most recent context
    truncated = []
    total_tokens = 0

    for msg in reversed(messages):
        msg_tokens = len(tokenizer.encode(msg["content"]))
        if total_tokens + msg_tokens > max_tokens:
            break
        truncated.insert(0, msg)
        total_tokens += msg_tokens

    return system_msg + truncated
```

---

## Implementation

### Complete SFT Training Script

#### Problem Statement: From Raw Instructions to Training Data

The core challenge in SFT is transforming human-readable instruction-response pairs into the tokenized, masked format required for efficient language model training. We need to:
1. Convert conversations to the appropriate chat template format
2. Tokenize while preserving alignment between tokens and their roles
3. Mask non-assistant tokens to focus learning on response generation
4. Handle variable-length sequences efficiently with padding

#### Theoretical Justification: Masked Language Modeling for Instructions

Standard language model pre-training uses the objective:
$$\mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

For SFT, we modify this to only compute loss on assistant responses. This is justified by several principles:

1. **Task Focus**: We want the model to learn answer generation, not question generation
2. **Gradient Efficiency**: All gradient signal focuses on the desired output distribution
3. **Mode Collapse Prevention**: Prevents the model from learning to repeat instructions verbatim
4. **Sample Efficiency**: With limited instruction data, we can't afford to waste gradient updates on instruction tokens

Mathematically, this becomes:
$$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \mathcal{A}} \log p_\theta(x_t \mid x_{<t})$$

where $\mathcal{A}$ is the set of assistant token positions.

#### Relationship to Alternative Approaches

**Compared to Full-Sequence Training:**
- Without masking, models often learn to echo the instruction before answering
- Masking is 2-3x more sample-efficient in practice
- However, some argue that learning instruction structure helps with reasoning

**Compared to Sequence-to-Sequence Models:**
- Traditional seq2seq uses encoder-decoder architecture
- Decoder-only LMs with masking achieve similar effect while leveraging pre-trained weights
- More parameter-efficient since we don't need separate encoder

**Compared to Prefix-Tuning:**
- Masking modifies the loss function, not the model architecture
- Compatible with both full fine-tuning and PEFT methods
- Simpler implementation than prefix-based methods

#### Key Implementation Insights

1. **Dynamic Masking**: We must identify assistant response regions *after* tokenization, since token boundaries may not align with character boundaries
2. **Efficient Batching**: Padding to maximum length enables efficient GPU utilization, but we must mask pad tokens in both attention and loss
3. **Label Offset**: In causal LM, labels are typically input_ids shifted by one position - the loss for token $t$ predicts token $t+1$
4. **Special Token Handling**: The assistant role marker itself should typically be masked - we only want loss on the actual response content

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import json
from tqdm import tqdm

class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        template_type: str = "llama3"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template = ChatTemplate(tokenizer, template_type)

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Convert to messages format
        if "messages" in item:
            # Already in multi-turn format
            messages = [ChatMessage(**msg) for msg in item["messages"]]
        else:
            # Single instruction-response
            messages = []
            if "system" in item:
                messages.append(ChatMessage(role="system", content=item["system"]))

            # Construct instruction
            instruction = item["instruction"]
            if "input" in item and item["input"]:
                instruction = f"{item['instruction']}\n\n{item['input']}"

            messages.append(ChatMessage(role="user", content=instruction))
            messages.append(ChatMessage(role="assistant", content=item["response"]))

        # Format and tokenize
        formatted = self.chat_template.apply_template(messages)

        encodings = self.tokenizer(
            formatted,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        # Create labels with masking
        labels = input_ids.clone()
        labels = self.mask_non_assistant_tokens(
            input_ids, labels, messages
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def mask_non_assistant_tokens(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        messages: List[ChatMessage]
    ) -> torch.Tensor:
        """
        Mask tokens that are not assistant responses.

        This ensures we only compute loss on the model's outputs,
        not on the instruction or user inputs.
        """
        # Get the full formatted text
        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

        # Mask everything initially
        labels[:] = -100

        # Find assistant response sections
        template = self.chat_template.templates[self.chat_template.template_type]
        assistant_marker = template["assistant_start"]
        eot_marker = template["eot"]

        # Track position in token sequence
        current_pos = 0
        text_so_far = ""

        for i, token_id in enumerate(input_ids):
            # Decode token
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            text_so_far += token_text

            # Check if we're in an assistant response
            # Find all assistant response ranges
            in_assistant_response = False
            search_text = text_so_far

            while assistant_marker in search_text:
                start_idx = search_text.find(assistant_marker)
                search_after = search_text[start_idx + len(assistant_marker):]

                if eot_marker in search_after:
                    end_idx = start_idx + len(assistant_marker) + search_after.find(eot_marker)
                    # Check if current position is in this range
                    if start_idx <= len(text_so_far) - len(token_text) < end_idx:
                        in_assistant_response = True
                        break
                    search_text = search_text[end_idx + len(eot_marker):]
                else:
                    # Ongoing assistant response
                    if start_idx <= len(text_so_far) - len(token_text):
                        in_assistant_response = True
                    break

            # Unmask if in assistant response (but not the marker itself)
            if in_assistant_response and assistant_marker not in token_text:
                labels[i] = input_ids[i]

        return labels

def train_sft(
    model_name: str,
    train_data_path: str,
    output_dir: str,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    max_length: int = 2048,
    warmup_ratio: float = 0.03,
):
    """
    Train a model using supervised fine-tuning.

    Args:
        model_name: Hugging Face model name or path
        train_data_path: Path to instruction dataset (JSON)
        output_dir: Where to save the fine-tuned model
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        max_length: Maximum sequence length
        warmup_ratio: Warmup ratio for learning rate schedule
    """
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    print(f"Loading dataset: {train_data_path}")
    dataset = InstructionDataset(
        train_data_path,
        tokenizer,
        max_length=max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    total_steps = len(dataloader) * epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=learning_rate * 0.1
    )

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Training")

        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            epoch_loss += loss.item()

            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                progress_bar.set_postfix({
                    "loss": loss.item() * gradient_accumulation_steps,
                    "lr": scheduler.get_last_lr()[0]
                })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = f"{output_dir}/checkpoint-epoch-{epoch+1}"
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete! Model saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    train_sft(
        model_name="meta-llama/Llama-3.2-3B",
        train_data_path="data/instructions.json",
        output_dir="models/llama3-sft",
        epochs=3,
        learning_rate=2e-5,
        batch_size=4,
        gradient_accumulation_steps=8
    )
```

### Using HuggingFace TRL Library

For production use, the [TRL](https://github.com/huggingface/trl) library provides optimized implementations:

```python
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

# Load dataset
dataset = load_dataset("json", data_files="instructions.json")

# Define formatting function
def format_instruction(example):
    """Format examples into chat template."""
    messages = example["messages"]
    return tokenizer.apply_chat_template(messages, tokenize=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama3-sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    max_grad_norm=1.0,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    formatting_func=format_instruction,
    max_seq_length=2048,
)

# Train
trainer.train()
```

---

## Loss Masking and Training Details

### Why Mask Instruction Tokens?

Consider this example:

```
User: What is 2+2?
Assistant: 4
```

**Without masking** (loss on all tokens):
- Model learns to predict "What" given context
- Model learns "is" given "What"
- Model learns "2+2?" given "What is"
- Model learns "4" given "What is 2+2?"

**With masking** (loss only on assistant tokens):
- Model only learns "4" given "What is 2+2?"

The second approach is better because:
1. We don't want the model to learn to generate questions
2. We want all gradient signal focused on generating good answers
3. Prevents mode collapse to repeating instructions

### Mathematical Formulation

Standard causal LM loss:
$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t})
$$

Masked SFT loss:
$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{|A|} \sum_{t \in A} \log p_\theta(y_t \mid y_{<t})
$$

where $A$ is the set of assistant token positions.

### Efficient Masking Implementation

#### Problem: Token-Level Role Identification

After tokenization, we have a flat sequence of token IDs. We need to identify which tokens belong to assistant responses and should contribute to the loss. The challenge is that:
1. Templates are applied at the string level, but masking happens at the token level
2. Tokenizers may split special tokens or combine them with adjacent text
3. We need to handle multiple assistant turns in a single sequence
4. The implementation must be efficient for large batches

#### Why This Algorithm Works

The algorithm relies on **state tracking through sequential scanning**. Key principles:

1. **Marker-Based State Machine**: We track whether we're currently inside an assistant response by detecting assistant start markers and end-of-turn markers
2. **Position Correspondence**: Token positions correspond to string positions in the decoded text, allowing us to map from string-level template structure to token-level masks
3. **Conservative Masking**: When in doubt (e.g., unclear token boundaries), we err on the side of masking to avoid training on potentially ambiguous tokens

#### Comparison to Alternative Approaches

**Alternative 1: String-based position tracking**
- Decode each token, track character positions
- Problem: Extremely slow for large sequences
- Our approach: Scan token sequence once

**Alternative 2: Regex-based masking**
- Apply regex to find assistant response spans
- Problem: Doesn't account for tokenization boundaries
- Our approach: Work directly with token IDs

**Alternative 3: Pre-compute masks during dataset creation**
- Store masks alongside input_ids
- Advantage: Faster during training
- Disadvantage: Inflexible, harder to debug
- Our approach: Compute on-the-fly for flexibility

#### Critical Insight: Batched Masking

The key optimization is that we can mask in batches by finding all assistant markers and all EOT markers, then pairing them efficiently. This is O(n) rather than O(n²) that a naive approach might use.

```python
def create_labels_mask(
    input_ids: torch.Tensor,
    tokenizer,
    assistant_token_id: int,
    eot_token_id: int
) -> torch.Tensor:
    """
    Create labels tensor with non-assistant tokens masked.

    Args:
        input_ids: Token IDs (batch, seq_len)
        tokenizer: Tokenizer instance
        assistant_token_id: Token ID marking assistant turn start
        eot_token_id: Token ID marking end of turn

    Returns:
        labels: Same as input_ids but with -100 for masked positions
    """
    labels = input_ids.clone()
    batch_size, seq_len = input_ids.shape

    for i in range(batch_size):
        # Find assistant response sections
        assistant_positions = (input_ids[i] == assistant_token_id).nonzero(as_tuple=True)[0]
        eot_positions = (input_ids[i] == eot_token_id).nonzero(as_tuple=True)[0]

        # Mask all tokens initially
        labels[i, :] = -100

        # Unmask assistant responses
        for asst_pos in assistant_positions:
            # Find the next EOT token
            eot_after = eot_positions[eot_positions > asst_pos]
            if len(eot_after) > 0:
                end_pos = eot_after[0]
                # Unmask from assistant marker to EOT (inclusive)
                # Skip the assistant marker itself
                labels[i, asst_pos+1:end_pos+1] = input_ids[i, asst_pos+1:end_pos+1]

    return labels
```

### Handling Padding

```python
def prepare_labels_with_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int
) -> torch.Tensor:
    """
    Ensure padding tokens are also masked in labels.

    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask (1 for real tokens, 0 for padding)
        labels: Labels tensor
        pad_token_id: Padding token ID

    Returns:
        labels: Labels with padding positions masked
    """
    # Mask padding positions
    labels[attention_mask == 0] = -100

    # Also mask pad tokens that might have been included
    labels[input_ids == pad_token_id] = -100

    return labels
```

---

## Best Practices and Common Pitfalls

### Best Practices

**1. Data Quality Over Quantity**
- 10K high-quality examples > 100K low-quality examples
- Manually review samples from your dataset
- Remove duplicates and near-duplicates

**2. Balanced Dataset**
```python
# Check distribution
from collections import Counter

task_types = [example["task_type"] for example in dataset]
print(Counter(task_types))

# Ensure reasonable balance:
# {
#   "qa": 3000,
#   "summarization": 2500,
#   "code": 2000,
#   "creative": 1500,
#   "math": 1000
# }
```

**3. Use Chat Templates Correctly**
- Always use the same template as the base model (if it was pre-tuned)
- Register custom templates in the tokenizer
- Test template application before training

**4. Monitor Training Metrics**
```python
def compute_metrics(eval_preds):
    """Compute metrics during evaluation."""
    predictions, labels = eval_preds

    # Mask padding
    predictions = predictions[labels != -100]
    labels = labels[labels != -100]

    # Compute accuracy
    accuracy = (predictions.argmax(-1) == labels).float().mean()

    # Compute perplexity
    loss = nn.functional.cross_entropy(
        predictions.view(-1, predictions.shape[-1]),
        labels.view(-1)
    )
    perplexity = torch.exp(loss)

    return {
        "accuracy": accuracy.item(),
        "perplexity": perplexity.item()
    }
```

**5. Learning Rate Scheduling**
```python
# Warmup + Cosine decay is standard
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.03 * total_steps),
    num_training_steps=total_steps
)
```

### Common Pitfalls

**1. Catastrophic Forgetting**
- **Symptom**: Model becomes great at instructions but loses general knowledge
- **Solution**:
  - Use lower learning rates (1e-5 to 5e-5)
  - Train for fewer epochs (1-3)
  - Mix in some pre-training data (10-20%)

**2. Overfitting**
- **Symptom**: Perfect on training data, poor on evaluation
- **Solution**:
  - Early stopping
  - Larger, more diverse dataset
  - Regularization (dropout, weight decay)

**3. Mode Collapse**
- **Symptom**: Model generates repetitive or generic responses
- **Solution**:
  - Increase data diversity
  - Use temperature sampling during evaluation
  - Check for duplicates in training data

**4. Format Confusion**
- **Symptom**: Model includes system/user tokens in responses
- **Solution**:
  - Ensure labels masking is correct
  - Verify chat template implementation
  - Add clear EOT tokens

**5. Imbalanced Loss**
- **Symptom**: Model optimizes for short responses
- **Solution**:
  - Length-normalized loss
  - Ensure diverse response lengths in data
  - Monitor average response length

#### Problem: Length Bias in Standard Cross-Entropy Loss

Standard cross-entropy loss sums over all tokens, which creates an implicit bias toward shorter responses. Why? Consider:
- Short response (10 tokens): Total loss ≈ $10 \times 0.5 = 5.0$
- Long response (100 tokens): Total loss ≈ $100 \times 0.5 = 50.0$

Even with the same per-token loss (0.5), the long response contributes 10x more to the batch loss. During gradient descent, the optimizer is incentivized to reduce loss by making responses shorter.

#### Theoretical Justification for Length Normalization

Length normalization addresses this by computing the **average** loss per response rather than total loss:
$$\mathcal{L}_{\text{normalized}} = \frac{1}{B} \sum_{b=1}^{B} \frac{1}{|A_b|} \sum_{t \in A_b} \log p(x_t \mid x_{<t})$$

where $|A_b|$ is the number of assistant tokens in example $b$.

This ensures that each example contributes equally to the loss regardless of response length, allowing the model to learn both concise and detailed responses without bias.

#### When to Use Length Normalization

**Use length normalization when:**
- Training data has diverse response lengths (10-500 tokens)
- You want the model to generate detailed explanations when appropriate
- You notice the model generating overly terse responses

**Don't use length normalization when:**
- All responses are similar length
- You explicitly want to encourage brevity
- Dataset is small (may reduce learning signal)

#### Key Insight: Per-Example Normalization

The critical detail is normalizing **per example** rather than per batch. If we normalized by total batch tokens, examples with different lengths would still contribute unequally. Per-example normalization ensures each training example has equal influence on the gradient update.

```python
# Length-normalized loss
def length_normalized_loss(logits, labels):
    """Compute loss normalized by sequence length."""
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
        reduction='none'
    )

    # Count non-masked tokens per example
    mask = (labels != -100).float()
    lengths = mask.sum(dim=1)

    # Average per example, then normalize
    loss = loss.view(labels.shape)
    loss = (loss * mask).sum(dim=1) / (lengths + 1e-8)

    return loss.mean()
```

---

## Troubleshooting

This section covers common issues encountered during SFT and their solutions.

### Training Issues

**Problem: Loss becomes NaN after a few steps**

**Symptoms:**
```
Epoch 1, Step 10: loss = 2.341
Epoch 1, Step 11: loss = 5.892
Epoch 1, Step 12: loss = nan
```

**Causes:**
- Learning rate too high
- Gradient explosion
- Numerical instability in mixed precision

**Solutions:**
```python
# 1. Reduce learning rate
learning_rate = 1e-5  # Instead of 5e-5

# 2. Enable gradient clipping (should already be enabled)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Check for numerical stability
# Use FP32 for layer norm
model.to(torch.float32)
# Or use BF16 instead of FP16 (more stable)
training_args = TrainingArguments(bf16=True)  # Not fp16=True

# 4. Reduce batch size
per_device_batch_size = 2  # Instead of 4

# 5. Check for corrupted data
# Add validation in dataset __getitem__
def __getitem__(self, idx):
    item = self.data[idx]
    # Validate
    assert len(item['response']) > 0, f"Empty response at index {idx}"
    assert len(item['instruction']) > 0, f"Empty instruction at index {idx}"
    # ... rest of processing
```

**Problem: Out of Memory (OOM) errors**

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```python
# 1. Reduce batch size
per_device_batch_size = 1
gradient_accumulation_steps = 16  # Maintain effective batch size

# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 3. Reduce sequence length
max_length = 1024  # Instead of 2048

# 4. Use LoRA instead of full fine-tuning
from peft import get_peft_model, LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, config)

# 5. Use DeepSpeed ZeRO Stage 2/3
training_args = TrainingArguments(
    deepspeed="ds_config.json"  # DeepSpeed config
)

# 6. Quantize the base model (QLoRA)
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

**Problem: Training loss not decreasing**

**Symptoms:**
```
Epoch 1: loss = 2.341
Epoch 2: loss = 2.338
Epoch 3: loss = 2.340
```

**Causes:**
- Learning rate too low
- Model already well-aligned
- Data quality issues
- Incorrect loss masking

**Solutions:**
```python
# 1. Increase learning rate
learning_rate = 5e-5  # Instead of 1e-5

# 2. Check loss masking is working
def verify_labels(batch):
    """Verify labels are correctly masked."""
    labels = batch["labels"]
    # Count masked vs unmasked tokens
    masked = (labels == -100).sum()
    unmasked = (labels != -100).sum()
    print(f"Masked: {masked}, Unmasked: {unmasked}")
    # Should see significant masking (typically 40-70%)
    assert unmasked > 0, "All tokens are masked!"
    assert masked > 0, "No tokens are masked!"

# 3. Verify data quality
# Print a few examples from the training set
for i in range(3):
    example = dataset[i]
    print(f"Example {i}:")
    print(f"Input IDs shape: {example['input_ids'].shape}")
    print(f"Labels: {example['labels'][:50]}")  # First 50 tokens

# 4. Check if model is frozen
# Ensure requires_grad is True
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Warning: {name} is frozen!")
```

### Generation Issues

**Problem: Model only generates short responses**

**Symptoms:**
```
Input: "Write a detailed explanation of photosynthesis"
Output: "Photosynthesis."
```

**Causes:**
- Training data has mostly short responses
- EOS token learned too early
- Generation parameters too restrictive

**Solutions:**
```python
# 1. Filter training data for length diversity
def filter_short_responses(dataset, min_length=50):
    """Remove examples with very short responses."""
    return [ex for ex in dataset if len(ex['response']) >= min_length]

# 2. Use length penalty during generation
output = model.generate(
    input_ids,
    max_length=512,
    min_length=100,  # Encourage longer responses
    length_penalty=1.0,  # >1.0 encourages longer, <1.0 shorter
    no_repeat_ngram_size=3,  # Prevent repetition
)

# 3. Adjust temperature and top_p
output = model.generate(
    input_ids,
    max_length=512,
    temperature=0.8,  # Higher = more diverse
    top_p=0.9,
    do_sample=True,
)

# 4. Check EOS token in training
# Ensure EOS is not appearing too early in training data
def check_eos_positions(dataset, tokenizer):
    """Check where EOS tokens appear."""
    eos_id = tokenizer.eos_token_id
    positions = []
    for ex in dataset:
        input_ids = ex['input_ids']
        eos_pos = (input_ids == eos_id).nonzero()
        if len(eos_pos) > 0:
            positions.append(eos_pos[0].item())
    print(f"Average EOS position: {sum(positions)/len(positions)}")
```

**Problem: Model repeats the instruction in its response**

**Symptoms:**
```
Input: "What is 2+2?"
Output: "What is 2+2? What is 2+2? The answer is 4."
```

**Causes:**
- Loss masking not working correctly
- Chat template not properly separating roles

**Solutions:**
```python
# 1. Verify loss masking
def debug_loss_masking(dataset, tokenizer):
    """Debug loss masking by printing decoded tokens."""
    example = dataset[0]
    input_ids = example['input_ids']
    labels = example['labels']

    print("Tokens and Labels:")
    for i, (token_id, label) in enumerate(zip(input_ids, labels)):
        token = tokenizer.decode([token_id])
        masked = "MASKED" if label == -100 else "LEARNED"
        print(f"{i:4d}: {token:20s} -> {masked}")

    # Verify instruction tokens are masked
    # Verify assistant response tokens are NOT masked

# 2. Check chat template
# Ensure clear separation between user and assistant
formatted = chat_template.apply_template(messages)
print(formatted)
# Should see clear markers like:
# <|start_header_id|>user<|end_header_id|>
# <|start_header_id|>assistant<|end_header_id|>

# 3. Ensure labels masking logic is correct
# Assistant responses should NOT include the assistant marker itself
```

**Problem: Model generates gibberish or incoherent text**

**Symptoms:**
```
Input: "Explain quantum computing"
Output: "asdf jkl; qwer tyui opzx cvbn m..."
```

**Causes:**
- Corrupted model checkpoint
- Learning rate too high (divergence)
- Mixed up tokenizers between training and inference

**Solutions:**
```python
# 1. Verify tokenizer consistency
# Save tokenizer with model
tokenizer.save_pretrained(output_dir)
# Load same tokenizer during inference
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# 2. Test on validation set during training
# If validation loss also diverges, training issue
# If only inference is bad, checkpoint/tokenizer issue

# 3. Reduce learning rate and restart
learning_rate = 1e-6  # Very conservative

# 4. Load earlier checkpoint
# Check if earlier epochs produced better outputs

# 5. Verify model isn't quantized incorrectly
# Ensure consistent precision between training and inference
```

### Data Issues

**Problem: Model exhibits bias or generates harmful content**

**Symptoms:**
- Outputs reflect biases in training data
- Fails to refuse harmful requests

**Solutions:**
```python
# 1. Add safety examples to training data
safety_examples = [
    {
        "instruction": "How do I hack into someone's account?",
        "response": "I cannot provide assistance with illegal activities like hacking. This violates computer fraud laws and privacy rights."
    },
    {
        "instruction": "Write something offensive about [group]",
        "response": "I cannot generate offensive or discriminatory content about any group of people."
    },
]

# 2. Filter training data
from detoxify import Detoxify

detoxify_model = Detoxify('original')

def filter_toxic_data(dataset, threshold=0.5):
    """Remove toxic examples from dataset."""
    filtered = []
    for ex in dataset:
        scores = detoxify_model.predict(ex['response'])
        if max(scores.values()) < threshold:
            filtered.append(ex)
    return filtered

# 3. Balance dataset with refusal examples
# Ensure ~5-10% of dataset teaches appropriate refusals
```

**Problem: Mode collapse - all responses are too similar**

**Symptoms:**
- Every response starts with "Sure, I'd be happy to help!"
- Lack of diversity in generated text

#### Why Mode Collapse Happens

Mode collapse in SFT occurs when the training data contains repetitive patterns that the model over-learns. This can happen due to:
1. **Dataset generation artifacts**: Many synthetic datasets have formulaic responses
2. **Overfitting**: Training too long on limited data
3. **High-probability sequences**: The model learns "safe" responses that apply broadly

Mathematically, the model learns to maximize $p(a \mid u)$ but collapses to a small set of high-probability $a$ that work reasonably well for many $u$, rather than learning the full conditional distribution.

#### Why Deduplication Matters

Near-duplicate examples in training data cause the model to see the same patterns repeatedly, essentially acting as implicit upweighting. If "Sure, I'd be happy to help!" appears 1000 times but other opening phrases appear once each, the model learns this opening is 1000x more important.

Deduplication ensures that each unique pattern is weighted equally during training, preventing any single pattern from dominating the learned distribution.

#### The Sequence Similarity Approach

We use sequence matching rather than exact duplicates because:
- Synthetic data often has minor variations ("I'd be happy to help" vs "I'm happy to help")
- These variations don't add meaningful diversity
- Edit distance captures semantic similarity better than token matching

The algorithm computes the **longest common subsequence ratio** between response pairs, which is robust to insertions/deletions while catching near-duplicates.

**Solutions:**
```python
# 1. Increase response diversity in training data
# Remove near-duplicates
from difflib import SequenceMatcher

def remove_duplicates(dataset, similarity_threshold=0.85):
    """Remove near-duplicate responses."""
    unique = []
    for ex in dataset:
        is_unique = True
        for existing in unique:
            similarity = SequenceMatcher(
                None, ex['response'], existing['response']
            ).ratio()
            if similarity > similarity_threshold:
                is_unique = False
                break
        if is_unique:
            unique.append(ex)
    return unique

# 2. Use temperature during training
# Some frameworks support temperature in loss calculation

# 3. Sample from different response styles
# Vary tone, formality, verbosity in training data
```

### Performance Issues

**Problem: Training is very slow**

**Solutions:**
```python
# 1. Reduce number of workers if disk I/O is bottleneck
num_workers = 0  # Try 0, 2, 4, 8 to find optimum

# 2. Use faster data loading
from datasets import load_dataset
# Use Hugging Face datasets library with streaming
dataset = load_dataset("json", data_files="data.json", streaming=True)

# 3. Enable TF32 on Ampere GPUs (A100, RTX 30xx+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 4. Use Flash Attention if available
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",  # Requires flash-attn package
    torch_dtype=torch.bfloat16,
)

# 5. Optimize batch size and gradient accumulation
# Larger batch size = fewer steps = faster (if memory allows)
per_device_batch_size = 8
gradient_accumulation_steps = 4

# 6. Compile model (PyTorch 2.0+)
model = torch.compile(model)
```

**Problem: Inconsistent results between runs**

**Causes:**
- Non-deterministic operations
- Different random seeds

**Solutions:**
```python
# Set all random seeds
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# In TrainingArguments
training_args = TrainingArguments(
    seed=42,
    data_seed=42,
    # ...
)
```

---

## Evaluation

### Quantitative Metrics

#### Problem: Measuring Instruction-Following Capability

While training loss tells us how well the model fits the training data, we need additional metrics to understand:
1. Generalization to unseen instructions
2. Quality of generated responses
3. Whether the model is overfitting
4. Comparison to baseline models

Standard NLP metrics (BLEU, ROUGE) are insufficient for instruction-following because they reward exact n-gram matches, while good instruction responses can vary significantly in wording while being equally correct.

#### Theoretical Foundation: Perplexity as Confidence Measure

Perplexity measures the model's uncertainty about the next token:
$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p(x_i \mid x_{<i})\right)$$

Lower perplexity indicates:
- Higher confidence in token predictions
- Better compression of the validation set
- More aligned with the instruction-response distribution

However, perplexity has limitations:
- Doesn't measure response quality or correctness
- Can be artificially lowered by overfitting
- Doesn't account for instruction-following accuracy

#### Why Multiple Metrics Matter

Different metrics capture different aspects of model quality:

1. **Loss/Perplexity**: Technical fit to distribution
2. **Benchmark Accuracy**: Specific capability measurement
3. **Human Evaluation**: Overall usefulness and safety
4. **Instruction-Following Rate**: Core SFT objective

No single metric is sufficient. The best practice is to track a suite of metrics and understand the trade-offs between them.

#### Key Insight: Validation on Held-Out Instructions

The critical insight is to evaluate on held-out instruction types or domains, not just held-out examples from the same distribution. This tests true generalization rather than memorization. For example:
- Train on math and coding instructions
- Evaluate on science and history questions
- Check if instruction-following transfers across domains

**1. Loss and Perplexity**
```python
@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Count non-masked tokens
        num_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity.item()
    }
```

**2. Benchmark Performance**

Test on standard benchmarks (see [Evaluation and Benchmarks](33-evaluation-benchmarks.md)):
- **MMLU**: Multi-task language understanding
- **HellaSwag**: Commonsense reasoning
- **TruthfulQA**: Truthfulness and informativeness
- **GSM8K**: Math word problems
- **HumanEval**: Code generation

**Expected Performance Improvements:**

Here are typical benchmark results showing the impact of SFT on a 7B parameter model:

| Benchmark | Base Model | After SFT | Improvement |
|-----------|-----------|-----------|-------------|
| **MMLU** (5-shot) | 42.5% | 48.3% | +5.8% |
| **HellaSwag** (10-shot) | 76.2% | 78.9% | +2.7% |
| **TruthfulQA** (0-shot) | 38.1% | 52.7% | +14.6% |
| **GSM8K** (5-shot) | 12.3% | 28.4% | +16.1% |
| **HumanEval** (0-shot) | 13.4% | 24.6% | +11.2% |
| **ARC-Challenge** (25-shot) | 52.8% | 56.2% | +3.4% |

**Notes on Performance:**
- TruthfulQA shows largest gains because SFT teaches models to refuse uncertain answers
- GSM8K improves significantly as models learn to format step-by-step reasoning
- HumanEval benefits from code-focused instruction data
- MMLU and HellaSwag show modest gains as these rely heavily on pre-training knowledge
- Results vary significantly based on instruction dataset composition

**Perplexity Ranges:**

| Metric | Base Model | After SFT |
|--------|-----------|-----------|
| **Training Loss** (start) | 1.8-2.2 | N/A |
| **Training Loss** (end) | N/A | 0.8-1.2 |
| **Validation Perplexity** | 8-12 | 3-6 |
| **Instruction Perplexity** | 15-25 | 2.5-4.5 |

Lower perplexity on instruction data indicates better fit, but watch for overfitting if validation perplexity increases while training perplexity decreases.

**Using lm-evaluation-harness:**

```python
from lm_eval import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your SFT model
model = AutoModelForCausalLM.from_pretrained("path/to/sft-model")
tokenizer = AutoTokenizer.from_pretrained("path/to/sft-model")

# Evaluate on multiple benchmarks
results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained=path/to/sft-model,dtype=bfloat16",
    tasks=["mmlu", "hellaswag", "truthfulqa", "gsm8k", "arc_challenge"],
    num_fewshot=5,
    batch_size=8,
    device="cuda"
)

# Print results
for task, scores in results["results"].items():
    print(f"{task}: {scores}")

# Example output:
# mmlu: {'acc': 0.483, 'acc_stderr': 0.008}
# hellaswag: {'acc': 0.789, 'acc_norm': 0.821}
# truthfulqa: {'mc1': 0.527, 'mc2': 0.682}
# gsm8k: {'acc': 0.284}
```

**3. Instruction-Following Metrics**

#### Problem: Beyond Standard NLP Metrics

Traditional metrics like BLEU or ROUGE measure token overlap with reference responses, but instruction-following has unique requirements:
1. Format compliance (e.g., "respond in JSON format")
2. Constraint satisfaction (e.g., "use exactly 50 words")
3. Multi-aspect quality (helpfulness + accuracy + clarity)
4. Task completion rather than text similarity

#### Approach: Programmatic Verification

For many instruction types, we can programmatically verify compliance:
- JSON format → Parse and validate
- Length constraints → Count words/characters
- Code output → Execute and check correctness
- Factual claims → Compare against knowledge base

This provides objective, reproducible measurements that complement human evaluation.

#### Why This Matters

Instruction-following metrics directly measure the core objective of SFT: can the model understand and execute diverse instructions? Unlike perplexity (which measures fit) or benchmark accuracy (which measures specific skills), these metrics assess general instruction compliance across varying task types.

```python
def evaluate_instruction_following(model, tokenizer, test_cases):
    """
    Evaluate model's ability to follow specific instructions.

    Returns:
        dict: Scores for different instruction types
    """
    results = {
        "format_following": 0,
        "constraint_satisfaction": 0,
        "factual_accuracy": 0
    }

    for case in test_cases:
        prompt = case["instruction"]
        expected_format = case.get("expected_format")
        constraints = case.get("constraints", [])

        # Generate response
        response = generate_response(model, tokenizer, prompt)

        # Check format
        if expected_format and check_format(response, expected_format):
            results["format_following"] += 1

        # Check constraints
        if all(check_constraint(response, c) for c in constraints):
            results["constraint_satisfaction"] += 1

    # Normalize
    for key in results:
        results[key] /= len(test_cases)

    return results
```

### Qualitative Evaluation

**Human Evaluation is Critical:**

```python
# Example evaluation template
evaluation_criteria = {
    "helpfulness": "Is the response helpful and relevant?",
    "accuracy": "Is the information factually correct?",
    "clarity": "Is the response clear and well-organized?",
    "safety": "Does the response avoid harmful content?",
    "instruction_following": "Does it follow the specific instruction?"
}

# Scale: 1-5 for each criterion
def collect_human_ratings(model_responses, criteria):
    """Collect human ratings for model outputs."""
    ratings = []
    for response in model_responses:
        rating = {}
        for criterion, description in criteria.items():
            print(f"\n{description}")
            print(f"Response: {response}")
            rating[criterion] = int(input(f"Rate {criterion} (1-5): "))
        ratings.append(rating)
    return ratings
```

### A/B Testing

#### Problem: Quantifying Subjective Quality Improvements

While metrics like perplexity and benchmark accuracy are valuable, they don't fully capture whether one model is "better" than another for real-world use. Questions like "Which response is more helpful?" or "Which model would you prefer to use?" require human judgment.

A/B testing provides a rigorous framework for collecting these judgments and determining whether observed preferences are statistically significant.

#### Why A/B Testing Works

A/B testing is the gold standard for measuring user preferences because:
1. **Comparative judgment is easier**: Humans are better at comparing two options than rating one in isolation
2. **Reduces bias**: Randomizing presentation order prevents position bias
3. **Statistical validity**: With enough samples, we can compute confidence intervals
4. **Captures real preferences**: Measures what users actually prefer, not what correlates with preference

The theoretical foundation is **pairwise preference modeling**:
$$P(\text{prefer } A) = \frac{1}{1 + e^{-(q_A - q_B)}}$$

where $q_A$ and $q_B$ are latent quality scores. A/B testing estimates these preferences empirically.

#### Relationship to Other Evaluation Methods

**Compared to absolute rating:**
- A/B testing: "Which is better?"
- Absolute rating: "Rate this 1-5"
- A/B is more reliable because humans are better at comparisons

**Compared to automated metrics:**
- Metrics measure specific aspects (perplexity, accuracy)
- A/B testing measures overall preference
- Both are needed: metrics for debugging, A/B for final validation

**Compared to RLHF reward modeling:**
- A/B testing directly measures preferences
- Reward models try to predict A/B results
- A/B is ground truth; reward models approximate it

#### Key Insight: Randomization and Sample Size

Two critical elements for valid A/B testing:
1. **Randomize presentation order**: Prevents bias toward first/second position
2. **Sufficient sample size**: Need enough comparisons for statistical significance (typically 100+ per model pair)

Without randomization, results are confounded by presentation bias. Without sufficient samples, random noise dominates signal.

Compare your SFT model against baselines:

```python
def ab_test(model_a, model_b, test_prompts, evaluators):
    """
    A/B test between two models.

    Args:
        model_a: First model
        model_b: Second model
        test_prompts: List of test prompts
        evaluators: Human evaluators

    Returns:
        dict: Win rates and preferences
    """
    results = {"a_wins": 0, "b_wins": 0, "ties": 0}

    for prompt in test_prompts:
        response_a = generate_response(model_a, prompt)
        response_b = generate_response(model_b, prompt)

        # Randomize order to avoid bias
        if random.random() < 0.5:
            shown_order = [("A", response_a), ("B", response_b)]
        else:
            shown_order = [("B", response_b), ("A", response_a)]

        # Collect preference
        print(f"\nPrompt: {prompt}")
        for label, response in shown_order:
            print(f"\n{label}: {response}")

        preference = input("Which is better? (1/2/tie): ")

        # Record result
        if preference == "1":
            winner = shown_order[0][0]
        elif preference == "2":
            winner = shown_order[1][0]
        else:
            results["ties"] += 1
            continue

        if winner == "A":
            results["a_wins"] += 1
        else:
            results["b_wins"] += 1

    return results
```

---

## Summary

### Key Takeaways

1. **SFT transforms base models into instruction-following assistants**
   - Bridge between pre-training and alignment
   - Critical phase in the LLM pipeline

2. **Data quality is paramount**
   - 10K high-quality examples often sufficient
   - Diversity across tasks and domains essential
   - Balance between different capabilities

3. **Chat templates structure conversations**
   - Consistent formatting enables multi-turn dialogue
   - Special tokens demarcate roles and turns
   - Template must match base model's expectations

4. **Loss masking focuses learning**
   - Only compute loss on assistant responses
   - Prevents learning to generate instructions
   - Improves training efficiency

5. **Hyperparameters matter**
   - Lower learning rate than pre-training (1e-5 to 5e-5)
   - Small number of epochs (1-3) to prevent forgetting
   - Warmup + cosine decay scheduler

6. **Evaluation requires multiple approaches**
   - Quantitative: loss, perplexity, benchmarks
   - Qualitative: human evaluation, A/B testing
   - Instruction-following specific metrics

### When to Use SFT

**Use SFT when:**
- You have high-quality instruction-response data
- You need task-specific behavior (customer support, coding, etc.)
- You want to establish a helpful, harmless baseline before RLHF

**Consider alternatives when:**
- Data is limited → Few-shot prompting, in-context learning
- Need alignment without data → DPO with preference data
- Want to preserve base model → Inference-time steering, prompt engineering

### What's Next?

After SFT, models can be further improved through:

- **LoRA/PEFT** (see [LoRA and Parameter-Efficient Fine-tuning](20-peft.md)): Efficient fine-tuning methods
- **RLHF** (see [RLHF](21-rlhf.md)): Optimize for human preferences using reinforcement learning
- **DPO** (see [DPO](22-dpo.md)): Direct preference optimization without RL

The SFT model serves as the **reference model** or **initialization** for these subsequent alignment techniques.

---

## References

### Foundational Papers

1. **FLAN** - [Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (Wei et al., 2021)
   - Demonstrated that instruction tuning improves zero-shot performance
   - Showed effectiveness across many NLP tasks

2. **InstructGPT** - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022)
   - OpenAI's approach combining SFT with RLHF
   - Established the SFT → RLHF pipeline

3. **Self-Instruct** - [Self-Instruct: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560) (Wang et al., 2022)
   - Automated instruction dataset generation
   - Bootstrapping from small seed set

4. **Alpaca** - [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) (Taori et al., 2023)
   - Reproduced InstructGPT results with LLaMA
   - 52K GPT-3.5 generated instruction examples

5. **Vicuna** - [Vicuna: An Open-Source Chatbot Impressing GPT-4](https://lmsys.org/blog/2023-03-30-vicuna/) (Chiang et al., 2023)
   - Fine-tuned LLaMA on ShareGPT conversations
   - Demonstrated importance of multi-turn dialogue data

### Dataset Papers

6. **Dolly** - [Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) (Databricks, 2023)
   - 15K human-written instruction examples
   - Commercial-friendly license

7. **OpenAssistant** - [OpenAssistant Conversations](https://arxiv.org/abs/2304.07327) (Köpf et al., 2023)
   - Crowdsourced conversation trees
   - Multiple responses per prompt with rankings

8. **LIMA** - [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) (Zhou et al., 2023)
   - Strong results with only 1K examples
   - Emphasis on data quality over quantity

### Scaling and Analysis

9. **Scaling Instruction-Finetuned Language Models** - [Flan-PaLM](https://arxiv.org/abs/2210.11416) (Chung et al., 2022)
   - Comprehensive study of instruction tuning at scale
   - Analysis of task mixing and few-shot performance

10. **The Flan Collection** - [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688) (Longpre et al., 2023)
    - Curated collection of instruction datasets
    - Best practices for dataset mixing

### Technical Guides

11. **Hugging Face TRL** - [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)
    - Library for SFT, reward modeling, and RLHF
    - Production-ready implementations

12. **Axolotl** - [Axolotl: Streamlined Fine-tuning Tool](https://github.com/OpenAccess-AI-Collective/axolotl)
    - Tool for efficient fine-tuning
    - Supports multiple training strategies

---

## Exercises

### Exercise 1: Dataset Analysis

Given a dataset of instruction-response pairs:

```python
dataset = [
    {"instruction": "What is 2+2?", "response": "4"},
    {"instruction": "What is 2+2?", "response": "The answer is 4"},
    {"instruction": "What is 2+2?", "response": "2+2=4"},
    # ... more examples
]
```

**Tasks:**
1. Compute dataset diversity metrics (unique instructions, avg response length)
2. Detect near-duplicate instructions
3. Analyze response length distribution
4. Identify potential quality issues

### Exercise 2: Implement Chat Template

Implement a chat template for a custom format:

```python
# Your custom format:
# [SYSTEM] <system message> [/SYSTEM]
# [USER] <user message> [/USER]
# [ASSISTANT] <assistant message> [/ASSISTANT]

class CustomChatTemplate:
    def format_messages(self, messages: List[Dict]) -> str:
        """Implement this method."""
        pass

    def parse_messages(self, formatted: str) -> List[Dict]:
        """Parse formatted string back to messages."""
        pass
```

**Test cases:**
```python
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"}
]

template = CustomChatTemplate()
formatted = template.format_messages(messages)
parsed = template.parse_messages(formatted)
assert parsed == messages
```

### Exercise 3: Loss Masking

Implement proper loss masking for multi-turn conversations:

```python
def create_sft_labels(
    input_ids: torch.Tensor,
    messages: List[Dict[str, str]],
    tokenizer
) -> torch.Tensor:
    """
    Create labels tensor with only assistant responses unmasked.

    Args:
        input_ids: Tokenized conversation
        messages: Original message structure
        tokenizer: Tokenizer used

    Returns:
        labels: Tensor with -100 for masked positions
    """
    # Your implementation here
    pass

# Test
messages = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is artificial intelligence."}
]
# Verify only the assistant response contributes to loss
```

### Exercise 4: Evaluate Catastrophic Forgetting

Design an experiment to measure catastrophic forgetting:

```python
def measure_forgetting(base_model, finetuned_model, general_tasks, instruction_tasks):
    """
    Measure how much general knowledge is lost during SFT.

    Args:
        base_model: Pre-trained model before SFT
        finetuned_model: Model after SFT
        general_tasks: List of general knowledge questions
        instruction_tasks: List of instruction-following tasks

    Returns:
        dict: Performance on both types of tasks
    """
    # Your implementation
    pass
```

**Report:**
- Performance delta on general tasks
- Performance gain on instruction tasks
- Analysis of trade-offs

### Exercise 5: Hyperparameter Tuning

Run an ablation study on SFT hyperparameters:

1. **Learning Rate**: Compare [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
2. **Epochs**: Compare [1, 2, 3, 5]
3. **Batch Size**: Compare [16, 32, 64, 128] (effective)

**Plot:**
- Training loss curves
- Validation performance
- Overfitting indicators

**Conclusion:**
- Recommended hyperparameters for your dataset
- How sensitive is SFT to each hyperparameter?

### Exercise 6: Multi-turn Context Management

Implement a strategy for handling conversations exceeding context length:

```python
def smart_truncate_conversation(
    messages: List[Dict],
    max_tokens: int,
    tokenizer
) -> List[Dict]:
    """
    Intelligently truncate a conversation to fit context window.

    Strategies to consider:
    1. Always keep system message
    2. Keep most recent turns
    3. Summarize middle turns
    4. Preserve task-critical context

    Args:
        messages: Full conversation history
        max_tokens: Maximum token budget
        tokenizer: Tokenizer

    Returns:
        Truncated messages that fit in context
    """
    # Your implementation
    pass
```

### Exercise 7: Build a Complete Pipeline

Create an end-to-end SFT pipeline:

```python
class SFTPipeline:
    """Complete SFT pipeline from data to evaluation."""

    def __init__(self, base_model_name: str):
        self.model_name = base_model_name
        # Initialize components

    def prepare_data(self, raw_data_path: str) -> Dataset:
        """Load and preprocess instruction data."""
        pass

    def train(self, dataset: Dataset, config: Dict):
        """Train the model with SFT."""
        pass

    def evaluate(self, test_dataset: Dataset) -> Dict[str, float]:
        """Evaluate on multiple metrics."""
        pass

    def generate(self, instruction: str) -> str:
        """Generate response for a new instruction."""
        pass

# Use the pipeline
pipeline = SFTPipeline("meta-llama/Llama-3.2-3B")
dataset = pipeline.prepare_data("data/instructions.json")
pipeline.train(dataset, config={"lr": 2e-5, "epochs": 3})
results = pipeline.evaluate(test_dataset)
response = pipeline.generate("What is machine learning?")
```

### Exercise 8: Data Augmentation

Implement data augmentation techniques for instruction datasets:

```python
def augment_instruction_data(
    examples: List[Dict],
    augmentation_factor: int = 2
) -> List[Dict]:
    """
    Augment instruction dataset while preserving semantic meaning.

    Techniques:
    1. Paraphrase instructions (using another LLM)
    2. Rephrase responses (while maintaining correctness)
    3. Add clarifying context
    4. Create multi-turn versions of single-turn examples

    Args:
        examples: Original instruction examples
        augmentation_factor: How many augmented versions per example

    Returns:
        Augmented dataset
    """
    # Your implementation
    pass
```

**Evaluation:**
- Does augmentation improve model performance?
- Which augmentation techniques are most effective?
- Is there a risk of reducing data quality?

---

**Next Chapter:** [LoRA and Parameter-Efficient Fine-tuning](20-peft.md) - Learn efficient alternatives to full fine-tuning that preserve model quality while drastically reducing memory requirements.
