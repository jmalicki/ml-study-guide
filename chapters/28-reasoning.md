# Chapter 28: Reasoning and Chain-of-Thought

Reasoning is a critical capability for large language models to solve complex problems, especially in domains like mathematics, programming, and logical inference. This chapter explores how LLMs can be prompted or trained to exhibit improved reasoning through explicit intermediate steps, verification mechanisms, and test-time compute scaling.

## Table of Contents

1. [Introduction to Reasoning in LLMs](#introduction)
2. [Chain-of-Thought Prompting](#chain-of-thought-prompting)
3. [Self-Consistency and Voting](#self-consistency)
4. [Tree-of-Thought Reasoning](#tree-of-thought)
5. [Process Reward Models (PRMs)](#process-reward-models)
6. [Reasoning Traces and Verification](#reasoning-traces)
7. [Test-Time Compute Scaling](#test-time-compute)
8. [Implementation: Building a Reasoning System](#implementation)
9. [Exercises](#exercises)

## Introduction to Reasoning in LLMs {#introduction}

Traditional language models generate responses in a single forward pass, potentially missing intermediate reasoning steps that would lead to better answers. Reasoning techniques address this by:

1. **Explicit intermediate steps**: Breaking down complex problems into manageable sub-problems
2. **Self-verification**: Checking the validity of reasoning steps
3. **Search and planning**: Exploring multiple reasoning paths
4. **Process supervision**: Rewarding correct reasoning processes, not just final answers

### Why Reasoning Matters

For complex tasks like multi-step mathematics, code generation, or logical puzzles, direct answer prediction often fails. Reasoning allows models to:

- **Decompose** problems into simpler steps
- **Backtrack** when encountering errors
- **Verify** intermediate results
- **Scale compute** at test time (more computation → better results)

## Chain-of-Thought Prompting {#chain-of-thought-prompting}

Chain-of-Thought (CoT) prompting, introduced by Wei et al. (2022), demonstrates that LLMs can be prompted to generate intermediate reasoning steps before producing a final answer.

### Basic Chain-of-Thought

The key insight: Simply adding "Let's think step by step" or providing examples with reasoning steps dramatically improves performance on reasoning tasks.

**Standard Prompting:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: 11
```

**Chain-of-Thought Prompting:**
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
5 + 6 = 11. The answer is 11.
```

### Zero-Shot CoT

Kojima et al. (2022) showed that simply appending "Let's think step by step" to the prompt enables zero-shot reasoning:

```python
def zero_shot_cot(model, tokenizer, question, device="cuda"):
    """
    Zero-shot Chain-of-Thought prompting.

    Args:
        model: Pre-trained language model
        tokenizer: Corresponding tokenizer
        question: The question to answer
        device: Device to run on

    Returns:
        reasoning: The reasoning trace
        answer: The extracted answer
    """
    # Step 1: Generate reasoning with CoT prompt
    cot_prompt = f"{question}\nLet's think step by step."

    inputs = tokenizer(cot_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        reasoning_output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    reasoning = tokenizer.decode(reasoning_output[0], skip_special_tokens=True)

    # Step 2: Extract answer with second prompt
    extract_prompt = f"{reasoning}\nTherefore, the answer is:"

    inputs = tokenizer(extract_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        answer_output = model.generate(
            **inputs,
            max_new_tokens=32,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(answer_output[0], skip_special_tokens=True)

    return reasoning, answer


# Example usage
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a model (e.g., Llama, Mistral, etc.)
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to("cuda")

question = "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?"
reasoning, answer = zero_shot_cot(model, tokenizer, question)
print(f"Reasoning: {reasoning}")
print(f"Answer: {answer}")
```

### Few-Shot CoT

Few-shot CoT provides examples with reasoning steps in the prompt:

```python
def few_shot_cot(model, tokenizer, question, examples, device="cuda"):
    """
    Few-shot Chain-of-Thought prompting with examples.

    Args:
        model: Pre-trained language model
        tokenizer: Corresponding tokenizer
        question: The question to answer
        examples: List of (question, reasoning, answer) tuples
        device: Device to run on

    Returns:
        reasoning: The reasoning trace
        answer: The final answer
    """
    # Build prompt with examples
    prompt_parts = []

    for ex_q, ex_reasoning, ex_answer in examples:
        prompt_parts.append(f"Q: {ex_q}")
        prompt_parts.append(f"A: {ex_reasoning} The answer is {ex_answer}.")
        prompt_parts.append("")

    prompt_parts.append(f"Q: {question}")
    prompt_parts.append("A:")

    prompt = "\n".join(prompt_parts)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the answer part after the last "A:"
    response = full_response.split("A:")[-1].strip()

    # Try to extract final answer
    if "The answer is" in response:
        reasoning = response.split("The answer is")[0].strip()
        answer = response.split("The answer is")[1].strip().rstrip(".")
    else:
        reasoning = response
        answer = ""

    return reasoning, answer


# Example with few-shot examples
examples = [
    (
        "John has 3 apples. He gives 1 to Mary. How many does he have?",
        "John started with 3 apples. He gave away 1 apple. 3 - 1 = 2.",
        "2"
    ),
    (
        "A box contains 12 red balls and 8 blue balls. What fraction are red?",
        "Total balls = 12 + 8 = 20. Red balls = 12. Fraction = 12/20 = 3/5.",
        "3/5"
    )
]

question = "Sarah has $50. She spends $12 on lunch and $15 on a book. How much does she have left?"
reasoning, answer = few_shot_cot(model, tokenizer, question, examples)
print(f"Reasoning: {reasoning}")
print(f"Answer: {answer}")
```

### Mathematical Formulation

Chain-of-thought can be viewed as decomposing the probability distribution:

$$P(a|q) = \sum_{r \in \mathcal{R}} P(a|r, q) P(r|q)$$

Where:
- $q$ is the question
- $a$ is the answer
- $r$ is the reasoning trace
- $\mathcal{R}$ is the space of all possible reasoning traces

In practice, we approximate by sampling or greedily generating a single reasoning trace $r^*$:

$$a^* \approx \arg\max_a P(a|r^*, q) \text{ where } r^* = \arg\max_r P(r|q)$$

**References:**
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)
- [Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916)

## Self-Consistency and Voting {#self-consistency}

Self-consistency, proposed by Wang et al. (2022), improves CoT by sampling multiple reasoning paths and taking a majority vote on the final answer.

### Algorithm

1. Generate $N$ diverse reasoning paths using temperature sampling
2. Extract the final answer from each path
3. Return the most common answer (majority vote)

### Why It Works

Different reasoning paths may make different mistakes, but correct reasoning is more likely to converge to the same answer. This is related to ensemble methods in machine learning.

### Implementation

```python
import torch
from collections import Counter
import re

def extract_answer(text):
    """
    Extract numerical answer from reasoning text.
    Handles various formats like "The answer is X" or just a number.
    """
    # Try to find "answer is X" pattern
    patterns = [
        r"[Tt]he answer is[:\s]+([0-9.,]+)",
        r"[Aa]nswer:[:\s]+([0-9.,]+)",
        r"=\s*([0-9.,]+)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(",", "")

    # Fallback: find last number in text
    numbers = re.findall(r"([0-9.,]+)", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def self_consistency(model, tokenizer, question, num_samples=5, temperature=0.7, device="cuda"):
    """
    Self-consistency with Chain-of-Thought.

    Args:
        model: Pre-trained language model
        tokenizer: Corresponding tokenizer
        question: The question to answer
        num_samples: Number of reasoning paths to sample
        temperature: Sampling temperature (higher = more diverse)
        device: Device to run on

    Returns:
        final_answer: The majority-voted answer
        all_answers: List of all sampled answers with their reasoning
        vote_counts: Counter of answer frequencies
    """
    prompt = f"{question}\nLet's think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    all_answers = []

    # Sample multiple reasoning paths
    for i in range(num_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.95,  # Nucleus sampling for diversity
            )

        reasoning = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = extract_answer(reasoning)

        all_answers.append({
            "reasoning": reasoning,
            "answer": answer,
            "sample_id": i
        })

    # Extract valid answers and vote
    valid_answers = [a["answer"] for a in all_answers if a["answer"] is not None]

    if not valid_answers:
        return None, all_answers, Counter()

    vote_counts = Counter(valid_answers)
    final_answer = vote_counts.most_common(1)[0][0]

    return final_answer, all_answers, vote_counts


# Example usage
question = "If a store sells 3 apples for $2, how much do 12 apples cost?"

final_answer, all_answers, vote_counts = self_consistency(
    model, tokenizer, question, num_samples=10, temperature=0.8
)

print(f"Question: {question}")
print(f"\nSampled answers and their frequencies:")
for answer, count in vote_counts.most_common():
    print(f"  {answer}: {count} votes")

print(f"\nFinal answer (majority vote): {final_answer}")

# Show some reasoning paths
print("\nExample reasoning paths:")
for i, sample in enumerate(all_answers[:3]):
    print(f"\nPath {i+1}:")
    print(f"  Answer: {sample['answer']}")
    print(f"  Reasoning: {sample['reasoning'][:200]}...")
```

### Performance Analysis

Self-consistency typically improves accuracy by 5-20% over single-path CoT, especially on tasks where:
- Multiple valid reasoning approaches exist
- The model has sufficient capability but is prone to occasional errors
- Answer space is discrete (e.g., multiple choice, numerical)

**Reference:**
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)

## Tree-of-Thought Reasoning {#tree-of-thought}

Tree-of-Thought (ToT), proposed by Yao et al. (2023), generalizes CoT by exploring a tree of reasoning steps rather than a single chain.

### Key Concepts

1. **Thought decomposition**: Break problem into intermediate thought steps
2. **Thought generator**: Generate multiple candidate next thoughts at each step
3. **State evaluator**: Evaluate the promise of each thought
4. **Search algorithm**: Explore the tree (BFS, DFS, beam search)

### Algorithm Structure

```
                    Question
                       |
            /---------+----------\
        Thought 1   Thought 2   Thought 3
           |           |            |
       /---+---\   /---+---\    /---+---\
      T1.1  T1.2 T2.1  T2.2  T3.1  T3.2
```

At each level, the model:
1. Generates $k$ possible next thoughts
2. Evaluates each thought (heuristic score)
3. Prunes low-scoring branches
4. Continues search from promising branches

### Implementation

```python
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
import heapq

@dataclass
class ThoughtNode:
    """A node in the tree of thoughts."""
    thought: str
    parent: Optional['ThoughtNode']
    depth: int
    score: float
    children: List['ThoughtNode']

    def __lt__(self, other):
        # For heap operations (higher score = better)
        return self.score > other.score

    def get_path(self) -> List[str]:
        """Get the path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node.thought)
            node = node.parent
        return list(reversed(path))


class TreeOfThought:
    """
    Tree-of-Thought reasoning with beam search.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_thoughts(self, question: str, current_path: List[str],
                         num_thoughts: int = 3) -> List[str]:
        """
        Generate multiple candidate next thoughts.

        Args:
            question: The original question
            current_path: List of thoughts so far
            num_thoughts: Number of thoughts to generate

        Returns:
            List of candidate thought strings
        """
        # Build prompt with current path
        prompt_parts = [f"Question: {question}"]
        prompt_parts.append("Let's solve this step by step:")

        for i, thought in enumerate(current_path, 1):
            prompt_parts.append(f"Step {i}: {thought}")

        prompt_parts.append(f"Step {len(current_path) + 1}:")
        prompt = "\n".join(prompt_parts)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        thoughts = []
        for _ in range(num_thoughts):
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9,
                )

            full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # Extract just the new thought
            thought = full_text.split(f"Step {len(current_path) + 1}:")[-1].strip()
            # Take first sentence or line
            thought = thought.split("\n")[0].split(".")[0] + "."
            thoughts.append(thought)

        return thoughts

    def evaluate_thought(self, question: str, path: List[str]) -> float:
        """
        Evaluate how promising a reasoning path is.

        Returns a score between 0 and 1, where higher is better.
        This is a simplified heuristic; in practice, you might:
        - Use a separate value model
        - Use the LLM to self-evaluate
        - Use domain-specific heuristics
        """
        # Build evaluation prompt
        prompt = f"""Question: {question}

Reasoning so far:
{chr(10).join(f"{i+1}. {t}" for i, t in enumerate(path))}

On a scale of 0-10, how likely is this reasoning path to lead to the correct answer?
Consider: logical consistency, relevance, and progress toward solution.

Score:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract score
        try:
            score_text = response.split("Score:")[-1].strip()
            score = float(re.findall(r"\d+", score_text)[0])
            return score / 10.0  # Normalize to [0, 1]
        except (ValueError, IndexError):
            return 0.5  # Default medium score if parsing fails

    def search(self, question: str, max_depth: int = 4,
               beam_width: int = 3, thoughts_per_step: int = 3) -> Tuple[List[str], float]:
        """
        Perform beam search over the tree of thoughts.

        Args:
            question: The question to solve
            max_depth: Maximum depth of reasoning
            beam_width: Number of best paths to keep at each level
            thoughts_per_step: Number of thoughts to generate at each step

        Returns:
            best_path: List of thoughts in the best reasoning path
            best_score: Score of the best path
        """
        # Initialize with root node
        root = ThoughtNode(
            thought="[START]",
            parent=None,
            depth=0,
            score=1.0,
            children=[]
        )

        # Beam: keep top-k nodes at current frontier
        beam = [root]

        for depth in range(max_depth):
            all_candidates = []

            # Expand each node in current beam
            for node in beam:
                path = node.get_path()[1:]  # Exclude [START]

                # Generate candidate thoughts
                new_thoughts = self.generate_thoughts(
                    question, path, num_thoughts=thoughts_per_step
                )

                # Create child nodes and evaluate
                for thought in new_thoughts:
                    new_path = path + [thought]
                    score = self.evaluate_thought(question, new_path)

                    child = ThoughtNode(
                        thought=thought,
                        parent=node,
                        depth=depth + 1,
                        score=score,
                        children=[]
                    )
                    node.children.append(child)
                    all_candidates.append(child)

            # Select top-k candidates for next beam
            if not all_candidates:
                break

            beam = heapq.nlargest(beam_width, all_candidates, key=lambda n: n.score)

        # Return best path
        if beam:
            best_node = max(beam, key=lambda n: n.score)
            best_path = best_node.get_path()[1:]  # Exclude [START]
            return best_path, best_node.score
        else:
            return [], 0.0


# Example usage
tot = TreeOfThought(model, tokenizer, device="cuda")

question = """You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons?"""

best_path, score = tot.search(
    question,
    max_depth=5,
    beam_width=3,
    thoughts_per_step=3
)

print(f"Question: {question}\n")
print("Best reasoning path:")
for i, thought in enumerate(best_path, 1):
    print(f"  {i}. {thought}")
print(f"\nConfidence score: {score:.2f}")
```

### ToT vs CoT Comparison

| Aspect | Chain-of-Thought | Tree-of-Thought |
|--------|------------------|-----------------|
| **Structure** | Linear sequence | Tree with branching |
| **Exploration** | Single path | Multiple paths |
| **Backtracking** | No | Yes |
| **Compute cost** | $O(n)$ | $O(b^d)$ where $b$ is branching, $d$ is depth |
| **Best for** | Most tasks | Complex planning, search problems |

**Reference:**
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)

## Process Reward Models (PRMs) {#process-reward-models}

Process Reward Models (PRMs), introduced by OpenAI for improving mathematical reasoning, reward each step of the reasoning process rather than just the final answer.

### Outcome vs Process Supervision

**Outcome Reward Model (ORM):**
$$r_{\text{outcome}}(q, r, a) = \mathbb{1}[a = a^*]$$

Only rewards correct final answers, regardless of reasoning quality.

**Process Reward Model (PRM):**
$$r_{\text{process}}(q, r, a) = \sum_{i=1}^{n} w_i \cdot \text{score}(r_i | r_{<i}, q)$$

Rewards correct reasoning at each step $r_i$.

### Advantages of PRMs

1. **Fine-grained feedback**: Identifies exactly where reasoning goes wrong
2. **Better credit assignment**: Rewards partial progress
3. **Improved exploration**: Encourages diverse correct reasoning paths
4. **Robustness**: Less sensitive to lucky guesses

### Training a PRM

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ProcessRewardModel(nn.Module):
    """
    A model that scores individual reasoning steps.

    Takes as input a question and a partial reasoning trace,
    outputs a score for the next step.
    """

    def __init__(self, base_model):
        """
        Args:
            base_model: Pre-trained language model (e.g., Llama)
        """
        super().__init__()
        self.base_model = base_model

        # Add a reward head
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)  # Single scalar reward
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Tokenized question + partial reasoning [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            rewards: Scalar reward for each step [batch, seq_len]
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Take last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

        # Compute reward at each position
        rewards = self.reward_head(hidden_states).squeeze(-1)  # [batch, seq_len]

        return rewards


class ReasoningStepDataset(Dataset):
    """
    Dataset of reasoning steps with human labels.

    Each example contains:
    - question: The problem
    - steps: List of reasoning steps
    - labels: List of labels (1 = correct step, 0 = incorrect, -1 = neutral)
    """

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        steps = item["steps"]
        labels = item["labels"]

        # Format: "Question: {q}\nStep 1: {s1}\nStep 2: {s2}\n..."
        text_parts = [f"Question: {question}"]

        # Track which positions correspond to step boundaries
        step_positions = []

        for i, step in enumerate(steps):
            step_text = f"\nStep {i+1}: {step}"
            text_parts.append(step_text)

        text = "".join(text_parts)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # For simplicity, we'll assign the label to the last token of each step
        # In practice, you'd want more sophisticated alignment
        step_labels = torch.tensor(labels, dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": step_labels,
        }


def train_prm(model, train_dataloader, num_epochs=3, lr=1e-5, device="cuda"):
    """
    Train a Process Reward Model.

    Args:
        model: ProcessRewardModel
        train_dataloader: DataLoader with reasoning step data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            rewards = model(input_ids, attention_mask)

            # For simplicity, take mean reward per sequence
            # In practice, align rewards with step positions
            step_rewards = rewards.mean(dim=1)
            target_labels = labels.mean(dim=1)

            # Compute loss
            loss = loss_fn(step_rewards, target_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


# Example: Create synthetic training data
train_data = [
    {
        "question": "What is 15 + 27?",
        "steps": [
            "First, I'll add the ones place: 5 + 7 = 12",
            "Write down 2, carry 1",
            "Then add the tens place: 1 + 2 + 1 = 4",
            "So the answer is 42"
        ],
        "labels": [1.0, 1.0, 1.0, 1.0]  # All correct
    },
    {
        "question": "What is 8 * 6?",
        "steps": [
            "8 * 6 is the same as 8 + 8 + 8 + 8 + 8 + 8",
            "Let me add: 8 + 8 = 16, 16 + 8 = 24",
            "24 + 8 = 32, 32 + 8 = 40",
            "40 + 8 = 48, so the answer is 48"
        ],
        "labels": [1.0, 1.0, 1.0, 1.0]  # All correct
    },
    {
        "question": "What is 100 - 37?",
        "steps": [
            "I'll subtract: 0 - 7 is not possible",
            "So I'll borrow: 10 - 7 = 3",
            "Then 9 - 3 = 6",  # Error: should be 10 - 1 - 3 = 6
            "The answer is 63"
        ],
        "labels": [1.0, 1.0, 0.0, 1.0]  # Step 3 is incorrect reasoning
    },
]

# Create dataset and dataloader
dataset = ReasoningStepDataset(train_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize and train PRM
prm = ProcessRewardModel(model)
trained_prm = train_prm(prm, dataloader, num_epochs=3, lr=1e-5)
```

### Using PRMs for Best-of-N Sampling

Once trained, PRMs can be used to select the best reasoning path:

```python
def best_of_n_with_prm(model, prm, tokenizer, question, n=8, device="cuda"):
    """
    Generate N reasoning paths and select the best according to PRM.

    Args:
        model: Base generation model
        prm: Trained Process Reward Model
        tokenizer: Tokenizer
        question: Question to answer
        n: Number of paths to sample

    Returns:
        best_reasoning: The highest-scoring reasoning path
        best_score: The PRM score
    """
    prompt = f"{question}\nLet's think step by step."

    candidates = []

    # Generate N reasoning paths
    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        reasoning = tokenizer.decode(output[0], skip_special_tokens=True)
        candidates.append((reasoning, output))

    # Score each path with PRM
    scores = []
    for reasoning, output_ids in candidates:
        with torch.no_grad():
            rewards = prm(output_ids, attention_mask=torch.ones_like(output_ids))
            # Take mean reward as overall score
            score = rewards.mean().item()
            scores.append(score)

    # Select best
    best_idx = torch.tensor(scores).argmax().item()
    best_reasoning = candidates[best_idx][0]
    best_score = scores[best_idx]

    return best_reasoning, best_score
```

**Reference:**
- [Let's Verify Step by Step (OpenAI, Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)

## Reasoning Traces and Verification {#reasoning-traces}

Reasoning traces are the explicit intermediate steps a model generates. Verification involves checking whether these steps are correct.

### Types of Verification

1. **Self-verification**: Model checks its own work
2. **External verification**: Use a separate model or tool
3. **Formal verification**: Prove correctness using formal methods

### Self-Verification Implementation

```python
def self_verify_reasoning(model, tokenizer, question, reasoning, device="cuda"):
    """
    Ask the model to verify its own reasoning.

    Args:
        model: Language model
        tokenizer: Tokenizer
        question: Original question
        reasoning: The reasoning trace to verify
        device: Device to use

    Returns:
        is_correct: Boolean or confidence score
        verification_explanation: Why it's correct/incorrect
    """
    verification_prompt = f"""Question: {question}

Proposed reasoning:
{reasoning}

Is this reasoning correct? Please check each step carefully.
Answer with "CORRECT" or "INCORRECT" and explain why.

Verification:"""

    inputs = tokenizer(verification_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    verification = tokenizer.decode(output[0], skip_special_tokens=True)
    verification_text = verification.split("Verification:")[-1].strip()

    # Check for CORRECT/INCORRECT
    is_correct = "CORRECT" in verification_text.upper() and "INCORRECT" not in verification_text.upper()

    return is_correct, verification_text


# Example: Generate and verify
question = "If a car travels 60 km in 45 minutes, what is its speed in km/h?"

# Generate reasoning
reasoning, _ = zero_shot_cot(model, tokenizer, question)

# Verify
is_correct, explanation = self_verify_reasoning(model, tokenizer, question, reasoning)

print(f"Question: {question}")
print(f"\nReasoning:\n{reasoning}")
print(f"\nVerification: {'CORRECT' if is_correct else 'INCORRECT'}")
print(f"Explanation: {explanation}")
```

### Code Execution for Verification

For mathematical problems, we can execute generated code to verify answers:

```python
import re
from typing import Optional

def extract_and_execute_code(reasoning: str) -> Optional[float]:
    """
    Extract Python code from reasoning and execute it safely.

    Args:
        reasoning: Text that may contain Python code blocks

    Returns:
        result: The computed result, or None if extraction/execution fails
    """
    # Look for code blocks in markdown format
    code_pattern = r"```python\s*(.*?)```"
    matches = re.findall(code_pattern, reasoning, re.DOTALL)

    if not matches:
        # Try without markdown
        code_pattern = r"((?:^|\n)(?:result|answer)\s*=.*?)(?:\n|$)"
        matches = re.findall(code_pattern, reasoning, re.IGNORECASE)

    if not matches:
        return None

    code = matches[0].strip()

    # Execute in restricted namespace for safety
    namespace = {"__builtins__": {"abs": abs, "round": round, "max": max, "min": min}}

    try:
        exec(code, namespace)
        # Look for common variable names
        for var in ["result", "answer", "output"]:
            if var in namespace:
                return float(namespace[var])
        # If no explicit variable, return last assignment
        return None
    except Exception as e:
        print(f"Execution error: {e}")
        return None


def verify_with_code_execution(model, tokenizer, question, device="cuda"):
    """
    Generate reasoning with code and verify by executing it.
    """
    prompt = f"""{question}

Let's solve this step by step and write Python code to verify.

Solution:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    reasoning = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract answer from text
    text_answer = extract_answer(reasoning)

    # Execute code if present
    code_answer = extract_and_execute_code(reasoning)

    # Check consistency
    if text_answer and code_answer:
        try:
            text_val = float(text_answer)
            verified = abs(text_val - code_answer) < 0.01
        except ValueError:
            verified = False
    else:
        verified = False

    return {
        "reasoning": reasoning,
        "text_answer": text_answer,
        "code_answer": code_answer,
        "verified": verified,
    }


# Example
question = "A rectangle has length 8 meters and width 5 meters. What is its area?"
result = verify_with_code_execution(model, tokenizer, question)

print(f"Question: {question}")
print(f"\nReasoning:\n{result['reasoning']}")
print(f"\nText answer: {result['text_answer']}")
print(f"Code answer: {result['code_answer']}")
print(f"Verified: {result['verified']}")
```

## Test-Time Compute Scaling {#test-time-compute}

Test-time compute scaling refers to using more computation at inference time to improve answer quality. This is a key principle behind models like OpenAI's o1.

### The Scaling Principle

Traditional scaling: Better models require more **training** compute.

Test-time scaling: Better answers require more **inference** compute.

$$\text{Quality}(a) \propto f(\text{compute}_{\text{test}})$$

### Methods for Test-Time Scaling

1. **Best-of-N sampling**: Generate N answers, pick best
2. **Search**: Explore reasoning tree (beam search, MCTS)
3. **Iterative refinement**: Generate, critique, regenerate
4. **Self-play**: Generate multiple attempts and learn from them

### Implementing Test-Time Scaling

```python
import torch
import numpy as np
from typing import List, Callable

class TestTimeScaler:
    """
    Test-time compute scaling for reasoning tasks.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def compute_log_prob(self, question: str, reasoning: str, answer: str) -> float:
        """
        Compute log probability of answer given question and reasoning.
        """
        full_text = f"{question}\n{reasoning}\nThe answer is {answer}"
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            # Negative loss is log probability (approximately)
            log_prob = -outputs.loss.item()

        return log_prob

    def best_of_n(self, question: str, n: int = 8,
                  verifier: Optional[Callable] = None) -> dict:
        """
        Sample N reasoning paths and select best.

        Args:
            question: Question to answer
            n: Number of samples
            verifier: Optional function to score/verify reasoning

        Returns:
            Dictionary with best reasoning, answer, and score
        """
        candidates = []

        prompt = f"{question}\nLet's think step by step."

        for i in range(n):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            reasoning = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = extract_answer(reasoning)

            # Score with verifier or use log probability
            if verifier:
                score = verifier(question, reasoning, answer)
            else:
                score = self.compute_log_prob(question, reasoning, answer or "")

            candidates.append({
                "reasoning": reasoning,
                "answer": answer,
                "score": score,
                "sample_id": i,
            })

        # Select best
        best = max(candidates, key=lambda x: x["score"])

        return {
            "best_reasoning": best["reasoning"],
            "best_answer": best["answer"],
            "best_score": best["score"],
            "all_candidates": candidates,
            "compute_used": n,
        }

    def iterative_refinement(self, question: str, num_iterations: int = 3) -> dict:
        """
        Iteratively refine the answer through critique and regeneration.

        Args:
            question: Question to answer
            num_iterations: Number of refinement iterations

        Returns:
            Final refined answer with history
        """
        # Initial generation
        prompt = f"{question}\nLet's think step by step."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        current_reasoning = self.tokenizer.decode(output[0], skip_special_tokens=True)
        history = [{"iteration": 0, "reasoning": current_reasoning}]

        # Iterative refinement
        for i in range(num_iterations):
            # Generate critique
            critique_prompt = f"""Question: {question}

Current reasoning:
{current_reasoning}

Please critique this reasoning. What could be improved? Are there any errors?

Critique:"""

            inputs = self.tokenizer(critique_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                critique_output = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            critique = self.tokenizer.decode(critique_output[0], skip_special_tokens=True)
            critique_text = critique.split("Critique:")[-1].strip()

            # Generate refined reasoning
            refine_prompt = f"""Question: {question}

Previous reasoning:
{current_reasoning}

Critique:
{critique_text}

Based on this critique, provide improved reasoning:

Improved reasoning:"""

            inputs = self.tokenizer(refine_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                refined_output = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            refined = self.tokenizer.decode(refined_output[0], skip_special_tokens=True)
            current_reasoning = refined.split("Improved reasoning:")[-1].strip()

            history.append({
                "iteration": i + 1,
                "critique": critique_text,
                "reasoning": current_reasoning,
            })

        final_answer = extract_answer(current_reasoning)

        return {
            "final_reasoning": current_reasoning,
            "final_answer": final_answer,
            "history": history,
            "compute_used": 1 + 2 * num_iterations,  # Initial + (critique + refine) * n
        }


# Example: Test-time compute scaling
scaler = TestTimeScaler(model, tokenizer)

question = "A snail climbs 3 meters up a wall during the day and slides 2 meters down at night. If the wall is 10 meters high, how many days does it take to reach the top?"

# Method 1: Best-of-N
print("=== Best-of-N Sampling ===")
result_bon = scaler.best_of_n(question, n=10)
print(f"Best answer: {result_bon['best_answer']}")
print(f"Score: {result_bon['best_score']:.4f}")
print(f"Compute used: {result_bon['compute_used']} forward passes")

# Method 2: Iterative refinement
print("\n=== Iterative Refinement ===")
result_refine = scaler.iterative_refinement(question, num_iterations=3)
print(f"Final answer: {result_refine['final_answer']}")
print(f"Compute used: {result_refine['compute_used']} forward passes")
print("\nRefinement history:")
for entry in result_refine['history']:
    print(f"  Iteration {entry['iteration']}:")
    if 'critique' in entry:
        print(f"    Critique: {entry['critique'][:100]}...")
    print(f"    Answer: {extract_answer(entry['reasoning'])}")
```

### Compute-Optimal Test-Time Scaling

There's a tradeoff between compute and quality:

$$\text{Quality} = f(\text{compute}) - \text{cost} \cdot \text{compute}$$

Find the optimal compute budget:

```python
def compute_optimal_n(model, tokenizer, question,
                      cost_per_sample=1.0,
                      value_per_accuracy=100.0,
                      max_n=50):
    """
    Find the optimal number of samples balancing quality and cost.

    Args:
        question: Question to solve
        cost_per_sample: Cost per generated sample
        value_per_accuracy: Value of improving accuracy by 1%
        max_n: Maximum N to try

    Returns:
        optimal_n: Best number of samples
        results: Results for different N values
    """
    results = []

    # Try different values of N
    for n in [1, 2, 4, 8, 16, 32, max_n]:
        if n > max_n:
            break

        # Estimate accuracy (in practice, use validation set)
        # Here we use a simplified heuristic
        estimated_accuracy = 1.0 - (0.5 ** (n / 8))  # Diminishing returns

        # Calculate value
        total_cost = n * cost_per_sample
        total_value = estimated_accuracy * value_per_accuracy
        net_value = total_value - total_cost

        results.append({
            "n": n,
            "accuracy": estimated_accuracy,
            "cost": total_cost,
            "value": total_value,
            "net_value": net_value,
        })

    # Find optimal
    optimal = max(results, key=lambda x: x["net_value"])

    return optimal["n"], results


# Example
optimal_n, results = compute_optimal_n(
    model, tokenizer, question,
    cost_per_sample=0.01,
    value_per_accuracy=1.0,
)

print("Compute-optimal analysis:")
for r in results:
    print(f"N={r['n']:2d}: Accuracy={r['accuracy']:.1%}, Cost=${r['cost']:.2f}, Net Value=${r['net_value']:.2f}")
print(f"\nOptimal N: {optimal_n}")
```

### O1-Style Reasoning

OpenAI's o1 model uses long internal reasoning traces with reinforcement learning:

1. **Long CoT**: Generate very long reasoning traces (thousands of tokens)
2. **RL training**: Train with RL to maximize correctness
3. **Test-time search**: Use search/tree exploration at inference
4. **Compute scaling**: More compute → better results

Key principles:
- Train models to "think longer" before answering
- Use process rewards to guide reasoning
- Allow backtracking and self-correction
- Scale compute at test time, not just training time

**References:**
- [Learning to Reason with LLMs (OpenAI o1 announcement)](https://openai.com/index/learning-to-reason-with-llms/)
- [Scaling LLM Test-Time Compute Optimally (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)

## Implementation: Building a Reasoning System {#implementation}

Let's build a complete reasoning system that combines multiple techniques:

```python
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class ReasoningStrategy(Enum):
    """Different reasoning strategies."""
    ZERO_SHOT_COT = "zero_shot_cot"
    SELF_CONSISTENCY = "self_consistency"
    TREE_OF_THOUGHT = "tree_of_thought"
    BEST_OF_N = "best_of_n"
    ITERATIVE_REFINE = "iterative_refine"


@dataclass
class ReasoningConfig:
    """Configuration for reasoning system."""
    strategy: ReasoningStrategy
    temperature: float = 0.7
    max_tokens: int = 256
    num_samples: int = 5  # For self-consistency, best-of-N
    beam_width: int = 3  # For tree-of-thought
    max_depth: int = 4  # For tree-of-thought
    num_refinements: int = 2  # For iterative refinement
    use_verification: bool = True
    use_code_execution: bool = False


class ReasoningSystem:
    """
    A complete reasoning system combining multiple techniques.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: ReasoningConfig,
        reward_model: Optional[nn.Module] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_model = reward_model
        self.device = device

        # Initialize components
        if config.strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            self.tot = TreeOfThought(model, tokenizer, device)

        self.scaler = TestTimeScaler(model, tokenizer, device)

    def reason(self, question: str) -> Dict:
        """
        Main reasoning function that routes to appropriate strategy.

        Args:
            question: The question to answer

        Returns:
            Dictionary with reasoning, answer, confidence, and metadata
        """
        strategy = self.config.strategy

        if strategy == ReasoningStrategy.ZERO_SHOT_COT:
            return self._zero_shot_cot(question)

        elif strategy == ReasoningStrategy.SELF_CONSISTENCY:
            return self._self_consistency(question)

        elif strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            return self._tree_of_thought(question)

        elif strategy == ReasoningStrategy.BEST_OF_N:
            return self._best_of_n(question)

        elif strategy == ReasoningStrategy.ITERATIVE_REFINE:
            return self._iterative_refine(question)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _zero_shot_cot(self, question: str) -> Dict:
        """Zero-shot chain-of-thought reasoning."""
        reasoning, answer = zero_shot_cot(
            self.model, self.tokenizer, question, device=self.device
        )

        # Optional verification
        verified = None
        if self.config.use_verification:
            verified, _ = self_verify_reasoning(
                self.model, self.tokenizer, question, reasoning, device=self.device
            )

        return {
            "strategy": "zero_shot_cot",
            "reasoning": reasoning,
            "answer": answer,
            "verified": verified,
            "confidence": 0.5,  # No confidence estimate for single path
            "compute_used": 1,
        }

    def _self_consistency(self, question: str) -> Dict:
        """Self-consistency with voting."""
        final_answer, all_answers, vote_counts = self_consistency(
            self.model,
            self.tokenizer,
            question,
            num_samples=self.config.num_samples,
            temperature=self.config.temperature,
            device=self.device,
        )

        # Confidence based on vote proportion
        total_votes = sum(vote_counts.values())
        confidence = vote_counts[final_answer] / total_votes if final_answer else 0.0

        return {
            "strategy": "self_consistency",
            "reasoning": all_answers[0]["reasoning"] if all_answers else "",
            "answer": final_answer,
            "all_answers": all_answers,
            "vote_counts": dict(vote_counts),
            "confidence": confidence,
            "compute_used": self.config.num_samples,
        }

    def _tree_of_thought(self, question: str) -> Dict:
        """Tree-of-thought reasoning."""
        best_path, score = self.tot.search(
            question,
            max_depth=self.config.max_depth,
            beam_width=self.config.beam_width,
        )

        # Extract final answer
        reasoning = "\n".join(f"Step {i+1}: {thought}" for i, thought in enumerate(best_path))
        answer = extract_answer(reasoning) if reasoning else None

        return {
            "strategy": "tree_of_thought",
            "reasoning": reasoning,
            "reasoning_path": best_path,
            "answer": answer,
            "confidence": score,
            "compute_used": self.config.beam_width * self.config.max_depth,
        }

    def _best_of_n(self, question: str) -> Dict:
        """Best-of-N sampling with optional reward model."""
        verifier = None
        if self.reward_model:
            def prm_verifier(q, r, a):
                inputs = self.tokenizer(f"{q}\n{r}", return_tensors="pt").to(self.device)
                with torch.no_grad():
                    rewards = self.reward_model(**inputs)
                    return rewards.mean().item()
            verifier = prm_verifier

        result = self.scaler.best_of_n(
            question,
            n=self.config.num_samples,
            verifier=verifier,
        )

        return {
            "strategy": "best_of_n",
            "reasoning": result["best_reasoning"],
            "answer": result["best_answer"],
            "confidence": result["best_score"],
            "all_candidates": result["all_candidates"],
            "compute_used": result["compute_used"],
        }

    def _iterative_refine(self, question: str) -> Dict:
        """Iterative refinement through critique."""
        result = self.scaler.iterative_refinement(
            question,
            num_iterations=self.config.num_refinements,
        )

        return {
            "strategy": "iterative_refine",
            "reasoning": result["final_reasoning"],
            "answer": result["final_answer"],
            "history": result["history"],
            "confidence": 0.7,  # Heuristic: refined answers are more reliable
            "compute_used": result["compute_used"],
        }

    def batch_reason(self, questions: List[str]) -> List[Dict]:
        """
        Reason over a batch of questions.

        Args:
            questions: List of questions

        Returns:
            List of reasoning results
        """
        results = []
        for question in questions:
            result = self.reason(question)
            results.append(result)
        return results


# Example: Using the complete reasoning system

# Configuration for different scenarios
configs = {
    "fast": ReasoningConfig(
        strategy=ReasoningStrategy.ZERO_SHOT_COT,
        temperature=0.5,
    ),
    "accurate": ReasoningConfig(
        strategy=ReasoningStrategy.SELF_CONSISTENCY,
        num_samples=10,
        temperature=0.8,
    ),
    "complex": ReasoningConfig(
        strategy=ReasoningStrategy.TREE_OF_THOUGHT,
        beam_width=5,
        max_depth=6,
    ),
}

# Initialize system
reasoning_system = ReasoningSystem(
    model=model,
    tokenizer=tokenizer,
    config=configs["accurate"],
    device="cuda",
)

# Test on various questions
questions = [
    "What is 17 * 24?",
    "If all roses are flowers and some flowers are red, are all roses red?",
    "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
]

print("=" * 80)
print("REASONING SYSTEM EVALUATION")
print("=" * 80)

for i, question in enumerate(questions, 1):
    print(f"\n[Question {i}]")
    print(question)
    print()

    result = reasoning_system.reason(question)

    print(f"Strategy: {result['strategy']}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result.get('confidence', 0):.2%}")
    print(f"Compute used: {result['compute_used']} forward passes")

    if result.get('vote_counts'):
        print(f"Vote distribution: {result['vote_counts']}")

    print(f"\nReasoning excerpt:")
    reasoning_text = result['reasoning']
    print(reasoning_text[:300] + "..." if len(reasoning_text) > 300 else reasoning_text)
    print()
```

## Exercises {#exercises}

### Exercise 1: Implement Answer Extraction

Improve the `extract_answer()` function to handle more formats:
- Fractions (e.g., "3/4", "2 1/2")
- Percentages (e.g., "25%")
- Multiple choice (e.g., "A", "B", "C", "D")
- Yes/No questions
- Ranges (e.g., "between 5 and 10")

```python
def robust_extract_answer(text: str, answer_type: str = "numeric") -> str:
    """
    Extract answer from text based on expected answer type.

    Args:
        text: The reasoning text
        answer_type: One of "numeric", "multiple_choice", "boolean", "text"

    Returns:
        The extracted answer
    """
    # TODO: Implement this
    pass
```

### Exercise 2: Implement Monte Carlo Tree Search

Implement MCTS for reasoning, similar to AlphaGo but for problem-solving:

```python
class MCTSNode:
    def __init__(self, thought, parent=None):
        self.thought = thought
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def uct_score(self, exploration_constant=1.4):
        """
        Calculate UCT (Upper Confidence Bound for Trees) score.

        UCT = value + exploration_constant * sqrt(ln(parent_visits) / visits)
        """
        # TODO: Implement UCT formula
        pass


class MCTSReasoning:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def search(self, question: str, num_simulations: int = 100):
        """
        Perform MCTS to find best reasoning path.

        Steps:
        1. Selection: Choose most promising node using UCT
        2. Expansion: Generate new thoughts from selected node
        3. Simulation: Rollout to completion
        4. Backpropagation: Update values along path
        """
        # TODO: Implement MCTS
        pass
```

### Exercise 3: Build a Multi-Modal Reasoner

Extend the reasoning system to handle images (e.g., geometry problems with diagrams):

```python
class MultiModalReasoning:
    def __init__(self, vision_model, language_model, tokenizer):
        self.vision_model = vision_model
        self.language_model = language_model
        self.tokenizer = tokenizer

    def reason(self, question: str, image_path: Optional[str] = None):
        """
        Reason over question with optional image.

        If image provided:
        1. Extract visual features
        2. Generate image description
        3. Combine with question for reasoning
        """
        # TODO: Implement multi-modal reasoning
        pass
```

### Exercise 4: Reasoning Benchmark Evaluation

Evaluate your reasoning system on standard benchmarks:

```python
def evaluate_on_gsm8k(reasoning_system, num_samples=100):
    """
    Evaluate on GSM8K (Grade School Math) dataset.

    Dataset: https://github.com/openai/grade-school-math

    Metrics to compute:
    - Accuracy
    - Average confidence on correct/incorrect answers
    - Compute efficiency (accuracy per forward pass)
    """
    # TODO: Implement evaluation
    pass

def evaluate_on_math(reasoning_system, num_samples=100):
    """
    Evaluate on MATH dataset (competition mathematics).

    Dataset: https://github.com/hendrycks/math
    """
    # TODO: Implement evaluation
    pass
```

### Exercise 5: Reasoning Distillation

Train a smaller model to mimic the reasoning of a larger model:

```python
def distill_reasoning(teacher_model, student_model, dataset, num_epochs=3):
    """
    Distill reasoning ability from teacher to student.

    Approach:
    1. Generate reasoning traces with teacher
    2. Train student to predict both reasoning and answer
    3. Use KL divergence loss on intermediate steps

    Returns:
        trained_student: The distilled student model
    """
    # TODO: Implement distillation
    pass
```

## Summary

Reasoning is a critical capability for LLMs tackling complex problems:

1. **Chain-of-Thought**: Simple prompting technique that dramatically improves reasoning
2. **Self-Consistency**: Ensemble method using majority voting over multiple reasoning paths
3. **Tree-of-Thought**: Search over reasoning trees for complex planning problems
4. **Process Reward Models**: Reward correct reasoning steps, not just final answers
5. **Verification**: Check reasoning correctness through self-verification or code execution
6. **Test-Time Compute Scaling**: Use more computation at inference for better results

Key takeaways:
- Explicit reasoning steps improve accuracy and interpretability
- Multiple paths with voting reduces errors
- Process supervision is better than outcome supervision
- Test-time compute can be scaled like training compute
- Verification and self-correction are crucial for reliability

For training reasoning models, see [Chapter 20: RLHF](20-rlhf.md) for reward modeling techniques.

For evaluating reasoning capabilities, see [Chapter 32: Evaluation and Benchmarks](32-evaluation-benchmarks.md).

## Additional Resources

### Papers
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) - Wei et al., 2022
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/abs/2305.10601) - Yao et al., 2023
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - Lightman et al., 2023 (OpenAI)
- [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465) - Zelikman et al., 2022
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) - Snell et al., 2024

### Datasets
- [GSM8K](https://github.com/openai/grade-school-math) - Grade school math problems
- [MATH](https://github.com/hendrycks/math) - Competition mathematics
- [PRM800K](https://github.com/openai/prm800k) - Process supervision dataset

### Code Resources
- [Guidance](https://github.com/guidance-ai/guidance) - Structured generation for reasoning
- [LangChain](https://github.com/langchain-ai/langchain) - Reasoning chains and agents
- [DSPy](https://github.com/stanfordnlp/dspy) - Programming with foundation models
