# Chapter 18: Supervised Fine-tuning (SFT)

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
10. [Evaluation](#evaluation)
11. [Summary](#summary)
12. [References](#references)
13. [Exercises](#exercises)

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

```
┌─────────────────┐      ┌──────────────┐      ┌─────────────┐
│   Pre-trained   │ ──>  │  Supervised  │ ──>  │  Aligned    │
│   Base Model    │      │  Fine-tuning │      │   Model     │
│  (LLM Training) │      │    (SFT)     │      │   (RLHF)    │
└─────────────────┘      └──────────────┘      └─────────────┘
```

SFT is typically followed by alignment techniques like RLHF (see [RLHF](20-rlhf.md)) or DPO (see [DPO](21-dpo.md)), but a well-executed SFT phase is critical for final model quality.

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
| **LoRA** (see [LoRA and PEFT](19-peft.md)) | ~0.1-1% | Low | Fast | Very Good |
| **QLoRA** | ~0.1-1% (quantized base) | Very Low | Fast | Good |

For SFT specifically:

- **Full fine-tuning**: Best for critical applications, when compute available
- **LoRA**: Standard choice for most use cases
- **QLoRA**: When memory constrained (single GPU)

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

## Evaluation

### Quantitative Metrics

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

Test on standard benchmarks (see [Evaluation and Benchmarks](32-evaluation-benchmarks.md)):
- **MMLU**: Multi-task language understanding
- **HellaSwag**: Commonsense reasoning
- **TruthfulQA**: Truthfulness and informativeness
- **GSM8K**: Math word problems
- **HumanEval**: Code generation

**3. Instruction-Following Metrics**
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

- **LoRA/PEFT** (see [LoRA and Parameter-Efficient Fine-tuning](19-peft.md)): Efficient fine-tuning methods
- **RLHF** (see [RLHF](20-rlhf.md)): Optimize for human preferences using reinforcement learning
- **DPO** (see [DPO](21-dpo.md)): Direct preference optimization without RL

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

**Next Chapter:** [LoRA and Parameter-Efficient Fine-tuning](19-peft.md) - Learn efficient alternatives to full fine-tuning that preserve model quality while drastically reducing memory requirements.
