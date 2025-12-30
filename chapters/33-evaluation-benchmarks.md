# Chapter 33: Evaluation and Benchmarks

## Introduction

Evaluating large language models is a critical challenge in modern NLP. Unlike traditional ML tasks with clear metrics, LLMs must be evaluated across multiple dimensions: language understanding, reasoning, safety, factuality, and more. This chapter covers the metrics, benchmarks, and methodologies used to evaluate LLMs comprehensively.

## 32.1 Language Modeling Metrics

### 32.1.1 Perplexity

Perplexity is the fundamental metric for language models, measuring how "surprised" a model is by a test sequence.

**Definition**: For a sequence of tokens $x_1, x_2, \ldots, x_N$:

$$
\text{PPL}(x) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})\right)
$$

Lower perplexity indicates better performance. A perplexity of $k$ means the model is as uncertain as if it had to choose uniformly among $k$ possibilities at each step.

**Why Perplexity Matters**: Perplexity is the single most important intrinsic metric for language models because it directly measures the model's predictive uncertainty on held-out data. Unlike task-specific metrics, perplexity is universal across all language modeling tasks and correlates strongly with downstream performance. It's the LM equivalent of loss, but exponentiated to be more interpretable.

**Theoretical Foundation**: Perplexity is the exponential of cross-entropy, which measures the expected number of bits needed to encode data using the model's predicted distribution. It's rooted in information theory:
- Cross-entropy: $H(p, q) = -\sum_x p(x) \log q(x)$
- For language modeling: $H = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})$
- Perplexity: $\text{PPL} = 2^H = \exp(H)$

**Relationship to Alternatives**:
- **vs. Raw Loss**: Perplexity is more interpretable than cross-entropy loss because it represents the effective vocabulary size the model is choosing from at each step.
- **vs. Accuracy**: Unlike accuracy, perplexity is continuous and measures confidence in predictions, not just correctness.
- **vs. BLEU/ROUGE**: Perplexity is an intrinsic metric (model-only) while BLEU/ROUGE are extrinsic (require references), making perplexity faster and more general.

**Key Implementation Insights**:
1. **Sliding Window**: For sequences longer than the model's context window, we use a sliding window approach with stride. This ensures all tokens are evaluated while respecting context limits.
2. **Masking Previous Tokens**: The critical line `target_ids[:, :-trg_len] = -100` prevents counting the same token multiple times across windows, which would artificially lower perplexity.
3. **Per-Token vs. Per-Sequence**: We can compute both aggregate perplexity and per-token perplexities for analysis. High per-token perplexity identifies specific words the model struggles with.

**Implementation**:

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple

class PerplexityCalculator:
    """Calculate perplexity for language models."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def calculate_perplexity(
        self,
        text: str,
        stride: int = 512,
        max_length: int = 1024
    ) -> Tuple[float, float]:
        """
        Calculate perplexity using sliding window approach.

        Args:
            text: Input text to evaluate
            stride: Stride for sliding window
            max_length: Maximum sequence length

        Returns:
            perplexity: The perplexity score
            nll: Negative log-likelihood
        """
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.device)

        seq_len = input_ids.size(1)
        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_chunk.clone()
            target_ids[:, :-trg_len] = -100  # Ignore previously seen tokens

            with torch.no_grad():
                outputs = self.model(input_ids_chunk, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc

            if end_loc == seq_len:
                break

        total_nll = torch.stack(nlls).sum()
        perplexity = torch.exp(total_nll / seq_len)

        return perplexity.item(), (total_nll / seq_len).item()

    def calculate_token_perplexities(
        self,
        text: str
    ) -> List[Tuple[str, float]]:
        """Calculate per-token perplexities for analysis."""
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.device)

        token_perplexities = []

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

            for i in range(1, input_ids.size(1)):
                # Get probability of next token
                probs = F.softmax(logits[0, i-1], dim=-1)
                target_prob = probs[input_ids[0, i]]
                token_ppl = 1.0 / target_prob.item()

                token = self.tokenizer.decode([input_ids[0, i].item()])
                token_perplexities.append((token, token_ppl))

        return token_perplexities


# Example usage
def example_perplexity():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    calculator = PerplexityCalculator(model, tokenizer, device='cpu')

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Colorless green ideas sleep furiously.",  # Grammatical but semantically odd
        "Dog lazy the over jumps fox brown quick the."  # Ungrammatical
    ]

    for text in test_texts:
        ppl, nll = calculator.calculate_perplexity(text)
        print(f"Text: {text}")
        print(f"Perplexity: {ppl:.2f}, NLL: {nll:.4f}\n")
```

### 32.1.2 Bits Per Byte/Character

For multilingual or byte-level models, bits per byte (BPB) or bits per character (BPC) normalizes across different tokenizations:

$$
\text{BPB} = \frac{-\sum_{i=1}^{N} \log_2 P(x_i | x_{<i})}{\text{num\_bytes}}
$$

**Why Bits Per Byte Matters**: When comparing models with different tokenization schemes (e.g., BPE with different vocabulary sizes, character-level, byte-level), perplexity becomes incomparable because it depends on token granularity. BPB provides a tokenization-agnostic metric by normalizing to the underlying byte representation, making it essential for multilingual model evaluation and cross-model comparisons.

**Theoretical Foundation**: BPB is grounded in Shannon's source coding theorem from information theory:
- The cross-entropy in bits represents the expected number of bits needed to encode each symbol using the model's predicted distribution.
- By normalizing to bytes (the actual data representation), we get a hardware-independent, universal compression metric.
- A model with BPB of $b$ could theoretically compress the data to $b$ bits per byte using arithmetic coding.
- The relationship to perplexity: $\text{BPB} = \log_2(\text{PPL}) \times \frac{\text{tokens}}{\text{bytes}}$

**Relationship to Alternatives**:
- **vs. Perplexity**: BPB is comparable across different tokenizations, while perplexity is not. A character-level model will have much lower perplexity than a word-level model on the same data, but similar BPB.
- **vs. Bits Per Character**: BPC is better for multilingual evaluation because byte-level is universal (all Unicode maps to bytes), while character boundaries depend on encoding.
- **vs. Compression Ratio**: BPB directly relates to lossless compression efficiency, making it interpretable for practitioners.

**Key Implementation Insights**:
1. **Logarithm Base Conversion**: Models typically output natural log probabilities (nats), so we divide by $\log(2)$ to convert to bits.
2. **UTF-8 Encoding**: We use `text.encode('utf-8')` to get the byte representation. This handles multilingual text correctly, where characters may be 1-4 bytes.
3. **Normalization**: The critical step is dividing total bits by number of bytes, not tokens. This makes the metric independent of vocabulary size.

```python
import math

def calculate_bits_per_byte(text: str, model, tokenizer, device='cpu'):
    """Calculate bits per byte metric."""
    # Get model loss
    encodings = tokenizer(text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)
        nll = outputs.loss.item()

    # Convert from nats to bits and normalize by bytes
    num_bytes = len(text.encode('utf-8'))
    num_tokens = encodings.input_ids.size(1)

    # NLL is per-token, convert to per-byte
    total_bits = nll * num_tokens / math.log(2)  # Convert nats to bits
    bpb = total_bits / num_bytes

    return bpb


# Example
text = "Hello, world! 你好世界"
# bpb = calculate_bits_per_byte(text, model, tokenizer)
# print(f"Bits per byte: {bpb:.3f}")
```

## 32.2 Common Benchmarks

### 32.2.1 MMLU (Massive Multitask Language Understanding)

MMLU ([Hendrycks et al., 2021](https://arxiv.org/abs/2009.03300)) tests knowledge across 57 subjects including STEM, humanities, and social sciences. It uses multiple-choice questions.

**Why MMLU Matters**: MMLU is the gold standard for measuring broad knowledge in LLMs because it covers 57 diverse subjects from elementary to professional levels. Unlike domain-specific benchmarks, MMLU reveals whether a model has truly learned general knowledge or just specialized capabilities. It's particularly important for:
- Comparing models of different sizes (shows scaling laws in action)
- Detecting knowledge gaps across domains (e.g., strong in STEM, weak in humanities)
- Measuring the impact of training data curation and alignment

**Theoretical Foundation**: MMLU evaluation is based on **log-probability ranking** rather than generation:
- For each multiple choice question with options A, B, C, D, we compute $P(\text{token}_A | \text{prompt})$, $P(\text{token}_B | \text{prompt})$, etc.
- The model predicts the option with highest conditional probability: $\arg\max_{o \in \{A,B,C,D\}} P(o | \text{question})$
- This is more robust than generation-based evaluation because it's deterministic and not affected by decoding strategies.
- Few-shot prompting provides in-context learning examples that help the model understand the task format.

**Relationship to Alternatives**:
- **vs. TruthfulQA**: MMLU measures factual knowledge breadth, while TruthfulQA measures resistance to common misconceptions.
- **vs. ARC/OpenBookQA**: MMLU covers broader academic domains with more challenging graduate-level questions.
- **vs. Generation-based QA**: Log-probability evaluation is more consistent and doesn't require complex answer extraction, but may not capture the model's ability to explain reasoning.

**Key Implementation Insights**:
1. **Tokenization of Answer Choices**: The critical challenge is that "A", " A", and "A." may tokenize differently. We must ensure we're comparing the correct token IDs for each choice letter.
2. **Few-Shot Selection**: Using questions from the same subject as few-shot examples (typically 5) improves accuracy significantly by providing domain-specific context and format examples.
3. **Log-Probability Extraction**: We use `log_softmax` on the final token's logits to get normalized probabilities for each answer token, avoiding numerical instability.
4. **Subject-Level vs. Overall**: MMLU performance varies dramatically by subject (e.g., 95% on elementary math, 30% on philosophy), so reporting per-subject scores is crucial for understanding model capabilities.

**Implementation**:

```python
import torch
from torch.nn.functional import softmax
from typing import List, Dict
import json

class MMLUEvaluator:
    """Evaluator for MMLU benchmark."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def format_question(
        self,
        question: str,
        choices: List[str],
        few_shot_examples: List[Dict] = None
    ) -> str:
        """Format question with few-shot examples."""
        prompt = ""

        # Add few-shot examples if provided
        if few_shot_examples:
            for example in few_shot_examples:
                prompt += self._format_single_question(
                    example['question'],
                    example['choices'],
                    example['answer']
                )
                prompt += "\n\n"

        # Add the actual question
        prompt += self._format_single_question(question, choices)
        return prompt

    def _format_single_question(
        self,
        question: str,
        choices: List[str],
        answer: str = None
    ) -> str:
        """Format a single question."""
        formatted = f"{question}\n"
        for i, choice in enumerate(choices):
            label = chr(65 + i)  # A, B, C, D
            formatted += f"{label}. {choice}\n"

        if answer is not None:
            formatted += f"Answer: {answer}"
        else:
            formatted += "Answer:"

        return formatted

    def get_answer_logprobs(
        self,
        prompt: str,
        num_choices: int = 4
    ) -> List[float]:
        """Get log probabilities for each answer choice (A, B, C, D)."""
        # Tokenize prompt
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)

        # Get tokens for A, B, C, D
        choice_tokens = []
        for i in range(num_choices):
            label = chr(65 + i)
            # Handle both " A" and "A" tokenizations
            token_id = self.tokenizer.encode(f" {label}", add_special_tokens=False)[-1]
            choice_tokens.append(token_id)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1]  # Last token logits
            log_probs = torch.log_softmax(logits, dim=-1)

        # Get log probabilities for each choice
        answer_logprobs = [log_probs[token_id].item() for token_id in choice_tokens]
        return answer_logprobs

    def evaluate_question(
        self,
        question: str,
        choices: List[str],
        correct_answer: str,
        few_shot_examples: List[Dict] = None
    ) -> Dict:
        """Evaluate a single question."""
        prompt = self.format_question(question, choices, few_shot_examples)
        logprobs = self.get_answer_logprobs(prompt, len(choices))

        # Get predicted answer
        predicted_idx = logprobs.index(max(logprobs))
        predicted_answer = chr(65 + predicted_idx)

        correct = predicted_answer == correct_answer

        return {
            'question': question,
            'predicted': predicted_answer,
            'correct': correct_answer,
            'is_correct': correct,
            'logprobs': dict(zip([chr(65+i) for i in range(len(choices))], logprobs))
        }

    def evaluate_subject(
        self,
        questions: List[Dict],
        few_shot_examples: List[Dict] = None,
        num_shots: int = 5
    ) -> Dict:
        """Evaluate all questions in a subject."""
        # Use first few questions as few-shot examples if not provided
        if few_shot_examples is None and num_shots > 0:
            few_shot_examples = questions[:num_shots]
            test_questions = questions[num_shots:]
        else:
            test_questions = questions

        results = []
        for q in test_questions:
            result = self.evaluate_question(
                q['question'],
                q['choices'],
                q['answer'],
                few_shot_examples
            )
            results.append(result)

        accuracy = sum(r['is_correct'] for r in results) / len(results)

        return {
            'accuracy': accuracy,
            'num_correct': sum(r['is_correct'] for r in results),
            'num_total': len(results),
            'results': results
        }


# Example MMLU-style questions
example_questions = {
    'abstract_algebra': [
        {
            'question': 'Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.',
            'choices': ['0', '4', '2', '6'],
            'answer': 'B'
        },
        {
            'question': 'Let A be a Hermitian matrix. Which of the following is always true?',
            'choices': [
                'All eigenvalues of A are real',
                'All eigenvalues of A are imaginary',
                'A is diagonal',
                'A is invertible'
            ],
            'answer': 'A'
        }
    ],
    'anatomy': [
        {
            'question': 'Which of the following bones is part of the axial skeleton?',
            'choices': ['Femur', 'Humerus', 'Sternum', 'Tibia'],
            'answer': 'C'
        }
    ]
}


def run_mmlu_evaluation():
    """Example of running MMLU evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = 'gpt2'  # Replace with your model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    evaluator = MMLUEvaluator(model, tokenizer, device='cpu')

    # Evaluate each subject
    all_results = {}
    for subject, questions in example_questions.items():
        print(f"\nEvaluating {subject}...")
        results = evaluator.evaluate_subject(questions, num_shots=1)
        all_results[subject] = results
        print(f"Accuracy: {results['accuracy']:.2%}")

    # Calculate overall accuracy
    total_correct = sum(r['num_correct'] for r in all_results.values())
    total_questions = sum(r['num_total'] for r in all_results.values())
    overall_accuracy = total_correct / total_questions

    print(f"\nOverall MMLU Accuracy: {overall_accuracy:.2%}")
    return all_results
```

### 32.2.2 HellaSwag

HellaSwag ([Zellers et al., 2019](https://arxiv.org/abs/1905.07830)) tests commonsense reasoning through sentence completion. Given a context, the model must choose the most plausible continuation.

**Why HellaSwag Matters**: HellaSwag is specifically designed to be adversarially difficult for models while remaining easy for humans (95% human accuracy). It tests whether models have genuine commonsense understanding or are just pattern matching on superficial features. The benchmark uses **Adversarial Filtering** where wrong answers are generated by a language model and then filtered to be challenging, making it harder to game than earlier benchmarks.

**Theoretical Foundation**: HellaSwag evaluation is based on **conditional likelihood scoring**:
- For each ending $e_i$, we compute the conditional probability: $P(e_i | \text{context}) = \prod_{t=1}^{|e_i|} P(w_t | \text{context}, e_{i,<t})$
- We take the log to avoid numerical underflow: $\log P(e_i | \text{context}) = \sum_{t=1}^{|e_i|} \log P(w_t | \text{context}, e_{i,<t})$
- The model selects: $\arg\max_{i} \log P(e_i | \text{context})$
- Critically, we must **length-normalize** because language models naturally assign higher probability to shorter sequences (fewer opportunities for low-probability tokens).

**Relationship to Alternatives**:
- **vs. COPA/StoryCloze**: HellaSwag is much larger (70k examples) and adversarially filtered, making it more challenging.
- **vs. MMLU**: HellaSwag tests implicit commonsense reasoning rather than explicit factual knowledge.
- **vs. WinoGrande**: Both test commonsense, but HellaSwag focuses on physical/temporal reasoning while WinoGrande tests coreference resolution.

**Key Implementation Insights**:
1. **Conditional Probability Computation**: Unlike MMLU where we only check the first token, here we must compute the full probability of multi-token endings conditioned on the context.
2. **Length Normalization**: The line `normalized_logprob = log_prob / max(1, len(full_ids) - ending_start_idx)` is critical. Without it, the model would always prefer shorter endings regardless of plausibility.
3. **Tokenization Alignment**: We must carefully handle the boundary between context and ending. The `-1` in `ending_start_idx = len(context_ids) - 1` prevents double-counting the token where context and ending join.
4. **Autoregressive Evaluation**: Each token in the ending is predicted conditioned on both the context AND the previous tokens in the ending, mimicking how the model would actually generate text.

```python
class HellaSwagEvaluator:
    """Evaluator for HellaSwag benchmark."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def evaluate_example(
        self,
        context: str,
        endings: List[str],
        correct_idx: int
    ) -> Dict:
        """
        Evaluate a single HellaSwag example.

        Uses conditional log-likelihood: log P(ending | context)
        """
        logprobs = []

        for ending in endings:
            # Combine context and ending
            full_text = context + " " + ending

            # Tokenize
            context_ids = self.tokenizer.encode(context, add_special_tokens=True)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

            # Get the tokens that correspond to the ending
            ending_start_idx = len(context_ids) - 1  # -1 because we don't double count

            input_ids = torch.tensor([full_ids]).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0]

            # Calculate log probability of the ending given context
            log_prob = 0.0
            for i in range(ending_start_idx, len(full_ids) - 1):
                token_logprobs = torch.log_softmax(logits[i], dim=-1)
                target_token = full_ids[i + 1]
                log_prob += token_logprobs[target_token].item()

            # Normalize by length to avoid bias toward shorter endings
            normalized_logprob = log_prob / max(1, len(full_ids) - ending_start_idx)
            logprobs.append(normalized_logprob)

        predicted_idx = logprobs.index(max(logprobs))
        is_correct = predicted_idx == correct_idx

        return {
            'context': context,
            'predicted_idx': predicted_idx,
            'correct_idx': correct_idx,
            'is_correct': is_correct,
            'logprobs': logprobs
        }


# Example HellaSwag-style questions
hellaswag_examples = [
    {
        'context': 'A man is sitting on a roof. He',
        'endings': [
            'is using wrap to wrap a pair of skis.',
            'is ripping level tiles off.',
            'is holding a rubik\'s cube.',
            'starts pulling up roofing on a roof.'
        ],
        'correct': 3
    },
    {
        'context': 'A woman is outside with a bucket and a dog. The dog',
        'endings': [
            'is running around trying to avoid a ball.',
            'washes his head in the bucket.',
            'gets into the bucket.',
            'is being sprayed with a hose.'
        ],
        'correct': 2
    }
]
```

### 32.2.3 GSM8K (Grade School Math)

GSM8K ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)) contains grade-school level math word problems requiring multi-step reasoning.

**Why GSM8K Matters**: GSM8K is the canonical benchmark for testing multi-step reasoning in LLMs. Unlike knowledge-based tasks (MMLU) or single-step reasoning (HellaSwag), GSM8K requires models to:
1. Decompose complex problems into subtasks
2. Perform arithmetic operations correctly
3. Chain reasoning steps coherently
4. Avoid accumulating errors across steps

It's particularly valuable because it's **verifiable** (answers are numerical) and reveals the limits of reasoning capabilities in current models. Performance on GSM8K correlates strongly with general reasoning ability.

**Theoretical Foundation**: GSM8K evaluation leverages **Chain-of-Thought (CoT) prompting**:
- Standard prompting: Model directly generates answer $P(\text{answer} | \text{question})$
- CoT prompting: Model generates reasoning steps first: $P(\text{reasoning}, \text{answer} | \text{question})$
- The factorization becomes: $P(\text{answer} | \text{question}) = \sum_{\text{reasoning}} P(\text{answer} | \text{reasoning}, \text{question}) \cdot P(\text{reasoning} | \text{question})$
- By explicitly generating reasoning, we bias the model toward the most likely reasoning path, dramatically improving accuracy (often 2-3x improvement).
- Few-shot examples teach the model the format and reasoning style expected.

**Relationship to Alternatives**:
- **vs. MATH Dataset**: GSM8K is elementary-level while MATH contains competition-level problems requiring advanced mathematics. GSM8K is better for evaluating general reasoning.
- **vs. AQuA/MAWPS**: GSM8K has more natural language questions and requires longer reasoning chains (avg. 5-7 steps vs. 1-3 steps).
- **vs. HumanEval**: Both test multi-step reasoning, but GSM8K is language-based while HumanEval is code-based, revealing different cognitive capabilities.

**Key Implementation Insights**:
1. **Chain-of-Thought Prompting**: The `use_cot` flag enables intermediate reasoning steps. This is not just a nice-to-have—it's essential for good performance (20% → 60%+ accuracy for many models).
2. **Greedy Decoding**: We use `temperature=0.0` for deterministic, reproducible results. Since answers are verifiable, we don't need sampling diversity.
3. **Answer Extraction**: The critical challenge is extracting the final numerical answer from free-form text. We use multiple regex patterns (GSM8K format `#### 42`, natural language "the answer is 42", fallback to last number).
4. **Numerical Tolerance**: We use `abs(predicted - correct_answer) < 1e-4` instead of exact equality to handle floating-point precision issues while still being strict.

```python
import re
from typing import Optional

class GSM8KEvaluator:
    """Evaluator for GSM8K benchmark."""

    def __init__(self, model, tokenizer, device='cuda', use_cot=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_cot = use_cot  # Chain-of-thought prompting

    def create_prompt(
        self,
        question: str,
        few_shot_examples: List[Dict] = None
    ) -> str:
        """Create prompt with optional few-shot examples."""
        if self.use_cot:
            instruction = "Solve the following math problem step by step.\n\n"
        else:
            instruction = "Solve the following math problem.\n\n"

        prompt = instruction

        if few_shot_examples:
            for example in few_shot_examples:
                prompt += f"Question: {example['question']}\n"
                if self.use_cot and 'reasoning' in example:
                    prompt += f"Answer: {example['reasoning']}\n"
                    prompt += f"Therefore, the answer is {example['answer']}.\n\n"
                else:
                    prompt += f"Answer: {example['answer']}\n\n"

        prompt += f"Question: {question}\nAnswer:"
        return prompt

    def generate_answer(
        self,
        question: str,
        few_shot_examples: List[Dict] = None,
        max_new_tokens: int = 256
    ) -> str:
        """Generate answer for a question."""
        prompt = self.create_prompt(question, few_shot_examples)

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # Greedy decoding for consistency
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return generated_text

    def extract_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from generated text."""
        # Look for patterns like "#### 42" (GSM8K format)
        gsm8k_pattern = r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)'
        match = re.search(gsm8k_pattern, text)
        if match:
            number_str = match.group(1).replace(',', '')
            return float(number_str)

        # Look for "the answer is X" pattern
        answer_pattern = r'(?:answer is|equals?)\s*\$?(-?\d+(?:,\d+)*(?:\.\d+)?)'
        match = re.search(answer_pattern, text.lower())
        if match:
            number_str = match.group(1).replace(',', '')
            return float(number_str)

        # Try to find last number in the text
        numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
        if numbers:
            return float(numbers[-1].replace(',', ''))

        return None

    def evaluate_question(
        self,
        question: str,
        correct_answer: float,
        few_shot_examples: List[Dict] = None
    ) -> Dict:
        """Evaluate a single question."""
        generated = self.generate_answer(question, few_shot_examples)
        predicted = self.extract_answer(generated)

        is_correct = False
        if predicted is not None:
            # Allow small numerical tolerance
            is_correct = abs(predicted - correct_answer) < 1e-4

        return {
            'question': question,
            'generated': generated,
            'predicted': predicted,
            'correct': correct_answer,
            'is_correct': is_correct
        }


# Example GSM8K problems
gsm8k_examples = [
    {
        'question': 'Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers\' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers\' market?',
        'reasoning': 'Janet sells 16 - 3 - 4 = 9 duck eggs per day. She makes 9 * 2 = $18 every day at the farmers\' market.',
        'answer': 18.0
    },
    {
        'question': 'A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?',
        'reasoning': 'It takes 2/2 = 1 bolt of white fiber. So in total it takes 2 + 1 = 3 bolts.',
        'answer': 3.0
    }
]
```

### 32.2.4 HumanEval (Code Generation)

HumanEval ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)) evaluates code generation capabilities through Python programming problems.

**Why HumanEval Matters**: HumanEval is the industry-standard benchmark for code generation because it tests **functional correctness** via actual code execution, not just syntactic similarity. This is fundamentally different from language tasks—there's no ambiguity in whether code works. Key aspects:
- Tests both basic programming (loops, conditionals) and algorithmic thinking
- Requires understanding docstrings and translating intent to code
- Used to evaluate Codex, GPT-4, Claude, and all major code models
- Correlates with real-world coding ability better than text-similarity metrics

**Theoretical Foundation**: HumanEval uses **pass@k** as the primary metric:
- Generate $n$ samples for each problem
- A problem is "solved" if at least $k$ of the $n$ samples pass all tests
- Mathematically: $\text{pass@k} = \mathbb{E}_{\text{problems}} [1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}]$ where $c$ is the number of correct samples
- For $k=1$, this simplifies to: fraction of problems with at least 1 correct sample out of $n$ attempts
- Common configurations: pass@1 (best single attempt), pass@10, pass@100

The theoretical justification is that code generation often requires multiple attempts even for humans (debugging cycle), and sampling diversity can find correct solutions that greedy decoding misses.

**Relationship to Alternatives**:
- **vs. MBPP** (Mostly Basic Programming Problems): MBPP has simpler problems (3 lines avg vs. 7-8 for HumanEval) and is better for weaker models.
- **vs. CodeContests/APPS**: These contain competitive programming problems requiring complex algorithms, while HumanEval tests practical programming.
- **vs. Unit Test Generation**: HumanEval provides tests and requires code; the reverse task tests different skills.

**Key Implementation Insights**:
1. **Execution-Based Evaluation**: The critical innovation is running code in a sandboxed environment with unit tests. This is objective and unambiguous, unlike BLEU for code.
2. **Temperature vs. Greedy**: For pass@k with k>1, we use `temperature=0.2` for diversity. Too high leads to syntax errors; too low reduces coverage of solution space.
3. **Stop Tokens**: The `extract_code` function stops at `\nclass`, `\ndef`, etc. to prevent the model from generating additional code beyond the function being completed.
4. **Timeout Protection**: The `timeout=5` parameter prevents infinite loops from hanging evaluation. This is essential for untrusted code execution.
5. **Pass@k vs. Success Rate**: We track both individual sample success and problem-level pass@k. The former shows sample quality; the latter shows problem coverage.

```python
import subprocess
import tempfile
import os
from typing import List

class HumanEvalEvaluator:
    """Evaluator for HumanEval benchmark."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_code(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        num_samples: int = 1
    ) -> List[str]:
        """Generate code completions."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        completions = []
        for _ in range(num_samples):
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            completions.append(generated)

        return completions

    def extract_code(self, completion: str, stop_tokens: List[str] = None) -> str:
        """Extract code from completion, stopping at certain tokens."""
        if stop_tokens is None:
            stop_tokens = ['\nclass', '\ndef', '\n#', '\nif __name__']

        for stop in stop_tokens:
            if stop in completion:
                completion = completion[:completion.index(stop)]

        return completion.strip()

    def execute_code(
        self,
        prompt: str,
        completion: str,
        test_cases: str,
        timeout: int = 5
    ) -> Dict:
        """Execute code with test cases."""
        # Combine prompt, completion, and tests
        full_code = prompt + completion + "\n\n" + test_cases

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(full_code)
            temp_file = f.name

        try:
            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            success = result.returncode == 0
            output = result.stdout
            error = result.stderr

        except subprocess.TimeoutExpired:
            success = False
            output = ""
            error = f"Execution timed out after {timeout} seconds"

        except Exception as e:
            success = False
            output = ""
            error = str(e)

        finally:
            # Clean up
            os.unlink(temp_file)

        return {
            'success': success,
            'output': output,
            'error': error
        }

    def evaluate_problem(
        self,
        prompt: str,
        test_cases: str,
        num_samples: int = 1,
        k: int = 1
    ) -> Dict:
        """
        Evaluate a single problem.

        Args:
            prompt: Function signature and docstring
            test_cases: Test cases to run
            num_samples: Number of completions to generate
            k: Parameter for pass@k metric
        """
        completions = self.generate_code(prompt, num_samples=num_samples)

        results = []
        num_passed = 0

        for i, completion in enumerate(completions):
            code = self.extract_code(completion)
            exec_result = self.execute_code(prompt, code, test_cases)

            if exec_result['success']:
                num_passed += 1

            results.append({
                'completion': code,
                'passed': exec_result['success'],
                'output': exec_result['output'],
                'error': exec_result['error']
            })

        # Calculate pass@k
        pass_at_k = num_passed >= k

        return {
            'results': results,
            'num_passed': num_passed,
            'num_samples': num_samples,
            'pass@k': pass_at_k
        }


# Example HumanEval problem
humaneval_example = {
    'prompt': '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
    'test_cases': '''
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True

check(has_close_elements)
print("All tests passed!")
'''
}
```

### 32.2.5 GPQA (Graduate-Level Google-Proof QA)

GPQA ([Rein et al., 2023](https://arxiv.org/abs/2311.12022)) contains graduate-level questions in biology, physics, and chemistry that are designed to be difficult even for experts and resistant to web search.

**Why GPQA Matters**: GPQA represents the frontier of knowledge evaluation for LLMs. Unlike MMLU which tests breadth, GPQA tests **depth** of expert-level understanding:
- Questions are validated by PhD-level experts who confirm they cannot be solved via web search
- Average expert accuracy is only ~65% (vs. 95%+ for MMLU), with non-expert accuracy ~35%
- Specifically designed to avoid "contamination" from web-scraped training data
- Tests whether models truly understand complex scientific concepts or just pattern match

This benchmark is crucial for evaluating the most advanced models (GPT-4, Claude Opus, etc.) where simpler benchmarks saturate.

**Theoretical Foundation**: GPQA uses the same **log-probability ranking** as MMLU but with important differences:
- Questions require multi-hop reasoning across concepts (e.g., combining quantum mechanics + thermodynamics)
- Distractors are carefully crafted to be plausible to non-experts but wrong
- The evaluation methodology is identical to MMLU: $\arg\max_{o} P(o | \text{question}, \text{domain context})$
- Domain context in the prompt provides specialist framing, improving performance by ~5-10%

**Relationship to Alternatives**:
- **vs. MMLU**: GPQA is much harder (state-of-the-art ~50% vs. ~90% on MMLU) and tests expert knowledge vs. general knowledge.
- **vs. Google-Proof QA**: GPQA extends this concept by explicitly validating that questions cannot be solved via search engines.
- **vs. STEM Benchmarks**: Most STEM benchmarks test undergraduate level; GPQA requires graduate-level expertise.

**Key Implementation Insights**:
1. **Domain Prompting**: The `_format_gpqa_question` includes domain context ("This is a graduate-level physics question"). This primes the model's expert knowledge and improves accuracy.
2. **Expert Validation**: Unlike other benchmarks, GPQA questions go through multiple rounds of expert review to ensure correctness and difficulty.
3. **Resistance to Contamination**: By being "Google-proof," GPQA minimizes the risk that models have seen answers during training, making it more reliable for benchmarking over time.
4. **Per-Domain Analysis**: Performance varies dramatically by domain (e.g., physics > biology > chemistry for many models), revealing specific knowledge gaps.

```python
class GPQAEvaluator:
    """Evaluator for GPQA benchmark."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def evaluate_question(
        self,
        question: str,
        choices: List[str],
        correct_answer: str,
        domain: str,
        difficulty: str = 'expert'
    ) -> Dict:
        """
        Evaluate a GPQA question.

        Args:
            question: The question text
            choices: List of answer choices
            correct_answer: Correct answer (A, B, C, or D)
            domain: Domain (biology, physics, chemistry)
            difficulty: Difficulty level
        """
        # Format similar to MMLU but with more detailed prompts
        prompt = self._format_gpqa_question(question, choices, domain)

        # Get answer probabilities
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        choice_tokens = []
        for i in range(len(choices)):
            label = chr(65 + i)
            token_id = self.tokenizer.encode(f" {label}", add_special_tokens=False)[-1]
            choice_tokens.append(token_id)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1]
            log_probs = torch.log_softmax(logits, dim=-1)

        answer_logprobs = [log_probs[token_id].item() for token_id in choice_tokens]
        predicted_idx = answer_logprobs.index(max(answer_logprobs))
        predicted_answer = chr(65 + predicted_idx)

        return {
            'question': question,
            'domain': domain,
            'difficulty': difficulty,
            'predicted': predicted_answer,
            'correct': correct_answer,
            'is_correct': predicted_answer == correct_answer,
            'logprobs': dict(zip([chr(65+i) for i in range(len(choices))], answer_logprobs))
        }

    def _format_gpqa_question(
        self,
        question: str,
        choices: List[str],
        domain: str
    ) -> str:
        """Format GPQA question with domain context."""
        prompt = f"The following is a graduate-level {domain} question.\n\n"
        prompt += f"Question: {question}\n\n"

        for i, choice in enumerate(choices):
            label = chr(65 + i)
            prompt += f"{label}. {choice}\n"

        prompt += "\nAnswer:"
        return prompt


# Example GPQA question
gpqa_example = {
    'question': 'In a double-slit experiment with electrons, if the distance between the slits is decreased while keeping all other parameters constant, what happens to the interference pattern on the screen?',
    'choices': [
        'The fringe spacing increases',
        'The fringe spacing decreases',
        'The fringe spacing remains constant',
        'The interference pattern disappears'
    ],
    'correct': 'A',
    'domain': 'physics'
}
```

### 32.2.6 MT-Bench (Multi-Turn Benchmark)

MT-Bench ([Zheng et al., 2023](https://arxiv.org/abs/2403.04132)) evaluates multi-turn conversation abilities using LLM judges.

**Why MT-Bench Matters**: Most benchmarks test single-turn performance, but real-world LLM usage involves multi-turn conversations where models must:
- Maintain context across turns
- Handle follow-up questions and clarifications
- Adapt responses based on previous interactions
- Demonstrate coherent long-form reasoning

MT-Bench is critical because it revealed that **model rankings change significantly** in multi-turn settings. Some models excel at single-turn but degrade in conversations, while others maintain or improve performance. It's also the first major benchmark to use **LLM-as-judge** evaluation, which correlates better with human preference than automated metrics.

**Theoretical Foundation**: MT-Bench uses **LLM-based evaluation** rather than automated metrics:
- A strong LLM (e.g., GPT-4) acts as a judge, scoring responses on a 1-10 scale
- The judge evaluates: relevance, accuracy, depth, coherence, and helpfulness
- Scoring prompt: $\text{score} = \text{LLM}_{\text{judge}}(\text{question}, \text{response}, \text{reference (optional)})$
- This approach leverages the LLM's ability to understand nuance, context, and quality in ways that BLEU/ROUGE cannot
- Correlation with human judgments: ~0.8-0.9 (much higher than traditional metrics at ~0.3-0.5)

The theoretical justification is that for open-ended generation tasks, automated metrics fail to capture quality. LLM judges can evaluate reasoning, style, factuality, and helpfulness holistically.

**Relationship to Alternatives**:
- **vs. Chatbot Arena**: MT-Bench provides standardized questions with reproducible evaluation, while Arena uses crowdsourced comparisons on user queries.
- **vs. Single-Turn Benchmarks**: MT-Bench explicitly tests conversation continuity and context tracking.
- **vs. AlpacaEval**: Both use LLM judges, but MT-Bench focuses on multi-turn while AlpacaEval is single-turn instruction following.

**Key Implementation Insights**:
1. **Conversation State Management**: The `conversation_history` tracks all previous turns. Each new response must be generated conditioned on the full history, testing the model's ability to maintain long context.
2. **LLM Judge Prompting**: The `_judge_response` function creates a detailed prompt for the judge, including evaluation criteria. The quality of this prompt directly affects judgment reliability.
3. **Temperature Selection**: We use `temperature=0.7` for generation to get natural, diverse responses. Too low makes responses robotic; too high makes them incoherent.
4. **Turn-by-Turn Scoring**: We score each turn individually AND compute average score. This reveals whether models maintain quality across turns or degrade.
5. **Category-Specific Evaluation**: Different categories (writing, math, coding) require different evaluation criteria, which the judge prompt must specify.

```python
class MTBenchEvaluator:
    """Evaluator for MT-Bench (Multi-Turn Benchmark)."""

    def __init__(self, model, tokenizer, judge_model, judge_tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.device = device

    def evaluate_conversation(
        self,
        turns: List[str],
        category: str,
        reference_answers: List[str] = None
    ) -> Dict:
        """
        Evaluate a multi-turn conversation.

        Args:
            turns: List of user prompts for each turn
            category: Category (writing, roleplay, reasoning, math, coding, etc.)
            reference_answers: Optional reference answers
        """
        conversation_history = []
        turn_scores = []

        for i, user_prompt in enumerate(turns):
            # Generate model response
            response = self._generate_turn(conversation_history, user_prompt)

            # Update conversation history
            conversation_history.append({
                'role': 'user',
                'content': user_prompt
            })
            conversation_history.append({
                'role': 'assistant',
                'content': response
            })

            # Judge the response
            score = self._judge_response(
                user_prompt,
                response,
                reference_answers[i] if reference_answers else None,
                category,
                turn_number=i+1
            )

            turn_scores.append({
                'turn': i + 1,
                'user_prompt': user_prompt,
                'response': response,
                'score': score
            })

        avg_score = sum(t['score'] for t in turn_scores) / len(turn_scores)

        return {
            'category': category,
            'turn_scores': turn_scores,
            'average_score': avg_score,
            'conversation': conversation_history
        }

    def _generate_turn(
        self,
        conversation_history: List[Dict],
        user_prompt: str,
        max_new_tokens: int = 512
    ) -> str:
        """Generate response for a conversation turn."""
        # Format conversation history
        prompt = ""
        for turn in conversation_history:
            if turn['role'] == 'user':
                prompt += f"User: {turn['content']}\n"
            else:
                prompt += f"Assistant: {turn['content']}\n"

        prompt += f"User: {user_prompt}\nAssistant:"

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _judge_response(
        self,
        user_prompt: str,
        response: str,
        reference_answer: str,
        category: str,
        turn_number: int
    ) -> float:
        """Use LLM judge to score response on 1-10 scale."""
        judge_prompt = f"""You are an expert evaluator for AI assistant responses.

Category: {category}
Turn: {turn_number}

User Question:
{user_prompt}

Assistant Response:
{response}

Please evaluate this response on a scale of 1-10 considering:
- Helpfulness: Does it address the user's question?
- Relevance: Is it on-topic?
- Accuracy: Is the information correct?
- Depth: Does it provide sufficient detail?
- Clarity: Is it well-written and easy to understand?

Provide your rating as a single number between 1 and 10.

Rating:"""

        input_ids = self.judge_tokenizer.encode(
            judge_prompt,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.judge_model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False
            )

        output = self.judge_tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Extract score
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', output)
        if match:
            score = float(match.group(1))
            return min(10.0, max(1.0, score))

        return 5.0  # Default if parsing fails


# Example MT-Bench conversation
mt_bench_example = {
    'category': 'writing',
    'turns': [
        'Compose a short story about a time traveler who accidentally changes history.',
        'Now rewrite the ending to make it more uplifting and optimistic.'
    ]
}
```

### 32.2.7 IFEval (Instruction Following Evaluation)

IFEval ([Zhou et al., 2023](https://arxiv.org/abs/2311.07911)) tests models' ability to follow specific formatting and constraint instructions.

**Why IFEval Matters**: Most benchmarks test what models know or can reason about, but IFEval tests whether models can **precisely follow instructions**, which is critical for:
- Controlled generation (e.g., "write exactly 200 words")
- Format compliance (e.g., "include 3 paragraphs")
- Constraint satisfaction (e.g., "don't mention X")
- Real-world applications where outputs must meet specific requirements

IFEval revealed a surprising insight: **instruction following ability doesn't correlate strongly with model size or general intelligence**. Some smaller, well-aligned models outperform larger models, suggesting that instruction following is a distinct capability requiring specific training.

**Theoretical Foundation**: IFEval uses **verifiable constraint checking**:
- Instructions are decomposed into formal constraints: $C = \{c_1, c_2, \ldots, c_n\}$
- Each constraint is programmatically checkable: $c_i : \text{string} \rightarrow \{\text{true}, \text{false}\}$
- Success requires satisfying ALL constraints: $\text{success} = \bigwedge_{i=1}^{n} c_i(\text{response})$
- Metrics: strict accuracy (all constraints met) and loose accuracy (relaxed variants)

This differs from LLM-judge evaluation because constraints are objective and verifiable, eliminating subjective judgment.

**Relationship to Alternatives**:
- **vs. MT-Bench**: MT-Bench evaluates quality; IFEval evaluates compliance. A response can be high-quality but non-compliant.
- **vs. FollowBench**: IFEval has more diverse constraint types (formatting, content, structure) while FollowBench focuses on multi-step reasoning.
- **vs. Traditional NLG Metrics**: BLEU/ROUGE measure similarity to references; IFEval measures adherence to explicit constraints.

**Key Implementation Insights**:
1. **Programmatic Constraint Checking**: Each constraint type (`word_count`, `contains_keyword`, `forbidden_words`, etc.) has a dedicated checker function. This makes evaluation objective and reproducible.
2. **Constraint Composition**: A single instruction may have multiple constraints (e.g., "write 200 words without mentioning 'the'"). All must be satisfied for success.
3. **Temperature Trade-off**: We use `temperature=0.7` to allow natural language while still following constraints. Higher temps improve fluency but reduce constraint adherence.
4. **Regex-Based Parsing**: Constraints like `num_sentences` use regex to parse structure. The quality of these parsers directly affects evaluation accuracy.
5. **Strict vs. Loose Evaluation**: The benchmark supports both strict (exact match) and loose (relaxed) evaluation modes for constraints like word count.

```python
class IFEvalEvaluator:
    """Evaluator for IFEval (Instruction Following Eval)."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_instruction(
        self,
        prompt: str,
        constraints: List[Dict],
        instruction_type: str
    ) -> Dict:
        """
        Evaluate instruction following.

        Args:
            prompt: The instruction to follow
            constraints: List of constraint dicts with 'type' and 'params'
            instruction_type: Type of instruction
        """
        # Generate response
        response = self._generate_response(prompt)

        # Check each constraint
        constraint_results = []
        all_satisfied = True

        for constraint in constraints:
            satisfied = self._check_constraint(response, constraint)
            constraint_results.append({
                'type': constraint['type'],
                'satisfied': satisfied,
                'params': constraint.get('params', {})
            })
            all_satisfied = all_satisfied and satisfied

        return {
            'prompt': prompt,
            'response': response,
            'instruction_type': instruction_type,
            'constraints': constraint_results,
            'all_satisfied': all_satisfied,
            'num_satisfied': sum(c['satisfied'] for c in constraint_results),
            'num_constraints': len(constraints)
        }

    def _generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512
    ) -> str:
        """Generate response to instruction."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _check_constraint(self, response: str, constraint: Dict) -> bool:
        """Check if response satisfies a constraint."""
        constraint_type = constraint['type']
        params = constraint.get('params', {})

        if constraint_type == 'word_count':
            words = response.split()
            min_words = params.get('min', 0)
            max_words = params.get('max', float('inf'))
            return min_words <= len(words) <= max_words

        elif constraint_type == 'contains_keyword':
            keyword = params['keyword']
            case_sensitive = params.get('case_sensitive', False)
            if case_sensitive:
                return keyword in response
            else:
                return keyword.lower() in response.lower()

        elif constraint_type == 'num_sentences':
            import re
            sentences = re.split(r'[.!?]+', response)
            sentences = [s.strip() for s in sentences if s.strip()]
            return len(sentences) == params['count']

        elif constraint_type == 'num_paragraphs':
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            return len(paragraphs) == params['count']

        elif constraint_type == 'forbidden_words':
            forbidden = params['words']
            response_lower = response.lower()
            return not any(word.lower() in response_lower for word in forbidden)

        elif constraint_type == 'starts_with':
            prefix = params['prefix']
            return response.startswith(prefix)

        elif constraint_type == 'ends_with':
            suffix = params['suffix']
            return response.endswith(suffix)

        elif constraint_type == 'all_caps':
            return response.isupper()

        elif constraint_type == 'no_punctuation':
            import string
            return not any(c in string.punctuation for c in response)

        else:
            # Unknown constraint type
            return False


# Example IFEval instructions
ifeval_examples = [
    {
        'prompt': 'Write a description of a sunset in exactly 3 sentences. Your response must contain the word "horizon".',
        'constraints': [
            {'type': 'num_sentences', 'params': {'count': 3}},
            {'type': 'contains_keyword', 'params': {'keyword': 'horizon'}}
        ],
        'instruction_type': 'format_and_keyword'
    },
    {
        'prompt': 'Write a brief poem about coding. Your response should be between 50 and 100 words and must not use the word "computer".',
        'constraints': [
            {'type': 'word_count', 'params': {'min': 50, 'max': 100}},
            {'type': 'forbidden_words', 'params': {'words': ['computer']}}
        ],
        'instruction_type': 'length_and_constraint'
    }
]
```

### 32.2.8 SimpleQA

SimpleQA ([OpenAI, 2024](https://openai.com/index/introducing-simpleqa/)) is a factuality benchmark with short, fact-seeking questions that have clear, unambiguous answers.

**Why SimpleQA Matters**: While many benchmarks test reasoning or knowledge breadth, SimpleQA specifically tests **factual accuracy** on straightforward questions where the model should either know the answer or acknowledge uncertainty. This is critical for:
- Measuring hallucination rates on simple facts
- Testing calibration (does the model know when it doesn't know?)
- Evaluating real-world reliability for fact-seeking queries
- Distinguishing between "plausible-sounding but wrong" and "correct" answers

SimpleQA revealed that even state-of-the-art models struggle with simple factuality, often confidently stating incorrect information rather than admitting uncertainty.

**Theoretical Foundation**: SimpleQA uses a **three-way classification** system:
- **Correct**: Response contains the correct answer
- **Incorrect**: Response contains a wrong answer (hallucination)
- **Not Attempted**: Model refuses to answer or expresses uncertainty

The key metrics are:
- Accuracy: $\frac{\text{correct}}{\text{correct} + \text{incorrect}}$ (excludes not attempted)
- Correctness rate: $\frac{\text{correct}}{\text{total}}$ (includes all responses)
- Hallucination rate: $\frac{\text{incorrect}}{\text{total}}$ (the critical safety metric)

This framework distinguishes between "I don't know" (good calibration) and making up facts (hallucination), which is crucial for trustworthy AI.

**Relationship to Alternatives**:
- **vs. TriviaQA**: SimpleQA focuses on factuality and calibration, while TriviaQA measures knowledge breadth without penalizing hallucinations.
- **vs. TruthfulQA**: Both test factuality, but TruthfulQA focuses on resisting common misconceptions while SimpleQA tests straightforward facts.
- **vs. MMLU**: MMLU provides answer choices (guessing is possible); SimpleQA requires generation (reveals true knowledge vs. guessing).

**Key Implementation Insights**:
1. **Short Answer Generation**: We use `max_new_tokens=50` to force concise answers. This prevents the model from hedging or generating verbose non-answers.
2. **Greedy Decoding**: `temperature=0.0` ensures deterministic, confident answers. We want to see if the model will hallucinate when forced to commit.
3. **Answer Matching**: The `_check_correctness` function handles variants (e.g., "Harper Lee" vs. "Harper") and partial matches, accounting for different phrasings of correct answers.
4. **Refusal Detection**: The `_classify_response` function identifies refusal patterns ("I don't know", "I'm not sure") to separate uncertainty from hallucination.
5. **Three Metrics**: Tracking correct, incorrect, and not-attempted separately reveals model calibration—the best models have high correctness and low hallucination, using "I don't know" appropriately.

```python
class SimpleQAEvaluator:
    """Evaluator for SimpleQA benchmark."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_question(
        self,
        question: str,
        correct_answer: str,
        answer_variants: List[str] = None
    ) -> Dict:
        """
        Evaluate a SimpleQA question.

        Args:
            question: The factual question
            correct_answer: The correct answer
            answer_variants: Alternative phrasings of correct answer
        """
        # Generate model response
        response = self._generate_answer(question)

        # Check correctness
        is_correct = self._check_correctness(
            response,
            correct_answer,
            answer_variants
        )

        # Classify response type
        response_type = self._classify_response(response, correct_answer)

        return {
            'question': question,
            'response': response,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'response_type': response_type
        }

    def _generate_answer(
        self,
        question: str,
        max_new_tokens: int = 50
    ) -> str:
        """Generate short answer to question."""
        prompt = f"Answer the following question concisely:\n\nQuestion: {question}\nAnswer:"

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _check_correctness(
        self,
        response: str,
        correct_answer: str,
        answer_variants: List[str] = None
    ) -> bool:
        """Check if response contains correct answer."""
        response_lower = response.lower().strip()
        correct_lower = correct_answer.lower().strip()

        # Direct match
        if correct_lower in response_lower:
            return True

        # Check variants
        if answer_variants:
            for variant in answer_variants:
                if variant.lower().strip() in response_lower:
                    return True

        return False

    def _classify_response(
        self,
        response: str,
        correct_answer: str
    ) -> str:
        """
        Classify response type:
        - 'correct': Contains correct answer
        - 'incorrect': Contains wrong answer
        - 'not_attempted': Refuses to answer or says "I don't know"
        """
        response_lower = response.lower()

        # Check for refusal patterns
        refusal_patterns = [
            "i don't know",
            "i'm not sure",
            "i cannot",
            "i can't",
            "not enough information",
            "unable to answer"
        ]

        if any(pattern in response_lower for pattern in refusal_patterns):
            return 'not_attempted'

        # Check if correct
        if self._check_correctness(response, correct_answer):
            return 'correct'

        return 'incorrect'


# Example SimpleQA questions
simpleqa_examples = [
    {
        'question': 'What is the capital of France?',
        'correct_answer': 'Paris',
        'answer_variants': []
    },
    {
        'question': 'Who wrote "To Kill a Mockingbird"?',
        'correct_answer': 'Harper Lee',
        'answer_variants': ['Harper']
    },
    {
        'question': 'What year did World War II end?',
        'correct_answer': '1945',
        'answer_variants': []
    }
]
```

## 32.3 Reasoning Benchmarks

### 32.3.1 ARC (AI2 Reasoning Challenge)

ARC ([Clark et al., 2018](https://arxiv.org/abs/1803.05457)) contains science questions requiring reasoning. It has two sets: Easy and Challenge.

**Why ARC Matters**: ARC was designed to test genuine reasoning rather than simple fact retrieval. Key features:
- Questions require multi-hop reasoning across scientific concepts
- ARC-Challenge specifically filters out questions answerable by simple retrieval or co-occurrence
- Many questions require understanding causality, not just correlation
- Performance gap between humans (>95%) and models reveals reasoning limitations

ARC is particularly valuable because it distinguishes between models that have memorized facts and models that can reason about scientific principles. The Challenge set remains difficult even for large models.

**Theoretical Foundation**: ARC combines multiple evaluation approaches:
- **Direct answering**: $P(\text{answer} | \text{question}, \text{choices})$ using log-probability ranking
- **Chain-of-Thought**: $P(\text{reasoning}, \text{answer} | \text{question})$ to expose intermediate steps
- **Retrieval-augmented**: $P(\text{answer} | \text{question}, \text{retrieved facts})$ to test reasoning over provided knowledge

The retrieval component is critical: it separates "can the model reason with facts" from "does the model know the facts." Many questions become solvable with the right supporting facts, testing pure reasoning ability.

**Relationship to Alternatives**:
- **vs. MMLU Science**: ARC requires deeper reasoning; MMLU science is more fact-based
- **vs. GSM8K**: Both test reasoning, but ARC is science-focused while GSM8K is math-focused
- **vs. CommonsenseQA**: ARC tests scientific reasoning; CommonsenseQA tests everyday reasoning

**Key Implementation Insights**:
1. **Chain-of-Thought**: The `use_cot` flag enables reasoning steps before the answer. For ARC-Challenge, this is nearly essential (often 20-30% accuracy gain).
2. **Retrieval Integration**: The `evaluate_with_retrieval` method tests whether the model can reason when given supporting facts, isolating reasoning from knowledge.
3. **Two-Set Design**: ARC-Easy tests basic competence; ARC-Challenge tests genuine reasoning. Models should score >80% on Easy, and Challenge scores reveal reasoning capability.
4. **Few-Shot Selection**: Using science-domain few-shot examples helps the model activate relevant knowledge and reasoning patterns.

```python
class ARCEvaluator:
    """Evaluator for ARC benchmark."""

    def __init__(self, model, tokenizer, device='cuda', use_cot=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_cot = use_cot

    def format_question(
        self,
        question: str,
        choices: Dict[str, str],
        few_shot: List[Dict] = None
    ) -> str:
        """Format ARC question with optional chain-of-thought."""
        prompt = ""

        if few_shot:
            for ex in few_shot:
                prompt += self._format_single(
                    ex['question'],
                    ex['choices'],
                    ex.get('reasoning', ''),
                    ex['answer']
                )
                prompt += "\n\n"

        prompt += self._format_single(question, choices)
        return prompt

    def _format_single(
        self,
        question: str,
        choices: Dict[str, str],
        reasoning: str = '',
        answer: str = None
    ) -> str:
        """Format a single question."""
        formatted = f"Question: {question}\n"
        for key in sorted(choices.keys()):
            formatted += f"{key}) {choices[key]}\n"

        if self.use_cot and reasoning:
            formatted += f"Reasoning: {reasoning}\n"

        if answer:
            formatted += f"Answer: {answer}"
        else:
            if self.use_cot:
                formatted += "Reasoning:"
            else:
                formatted += "Answer:"

        return formatted

    def evaluate_with_retrieval(
        self,
        question: str,
        choices: Dict[str, str],
        correct_answer: str,
        knowledge_base: List[str] = None
    ) -> Dict:
        """
        Evaluate with optional knowledge retrieval.

        Some ARC questions benefit from external knowledge.
        """
        prompt = self.format_question(question, choices)

        # Add retrieved knowledge if available
        if knowledge_base:
            context = "Relevant information:\n"
            for fact in knowledge_base[:3]:  # Top 3 facts
                context += f"- {fact}\n"
            prompt = context + "\n" + prompt

        # Get answer (similar to MMLU)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Get probabilities for each choice
        choice_tokens = [
            self.tokenizer.encode(f" {key}", add_special_tokens=False)[-1]
            for key in sorted(choices.keys())
        ]

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1]
            log_probs = torch.log_softmax(logits, dim=-1)

        logprobs = [log_probs[token].item() for token in choice_tokens]
        predicted_idx = logprobs.index(max(logprobs))
        predicted = list(sorted(choices.keys()))[predicted_idx]

        return {
            'question': question,
            'predicted': predicted,
            'correct': correct_answer,
            'is_correct': predicted == correct_answer
        }
```

### 32.3.2 MATH Dataset

The MATH dataset ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874)) contains challenging competition mathematics problems. Related to [Reasoning and Chain-of-Thought](29-reasoning.md).

**Why MATH Dataset Matters**: While GSM8K tests elementary arithmetic reasoning, MATH tests **advanced mathematical problem-solving** at the competition level (AMC, AIME, etc.). This is important because:
- It requires deep mathematical knowledge (algebra, geometry, calculus, number theory)
- Problems demand creative problem-solving, not just step-by-step procedures
- Success rate correlates with general reasoning ability and complex task performance
- It's one of the few benchmarks where GPT-4 still struggles (<50% accuracy)

MATH is a frontier benchmark that reveals the limits of current LLM reasoning capabilities.

**Theoretical Foundation**: MATH evaluation faces unique challenges due to mathematical equivalence:
- Unlike text, many mathematical expressions are equivalent: $\frac{1}{2} = 0.5 = \frac{2}{4}$
- The solution must handle: symbolic expressions, fractions, decimals, multiple representations
- Evaluation: $\text{correct} = \text{equiv}(\text{normalize}(\text{predicted}), \text{normalize}(\text{ground truth}))$
- Common format: answers in `\boxed{...}` notation from LaTeX

The key challenge is **answer extraction and normalization**—models may get the math right but format answers incorrectly.

**Relationship to Alternatives**:
- **vs. GSM8K**: MATH is much harder (competition-level vs. elementary) and requires specialized mathematical knowledge.
- **vs. Minerva/TheoremQA**: These test formal theorem proving; MATH tests problem-solving intuition.
- **vs. ARC**: Both test reasoning, but MATH requires mathematical creativity while ARC tests scientific reasoning.

**Key Implementation Insights**:
1. **Answer Normalization**: The `normalize_answer` function handles LaTeX formatting, whitespace, and different representations. This is critical because models often format correctly but wrap answers differently.
2. **Boxed Answer Extraction**: The MATH dataset uses `\boxed{answer}` format. The regex pattern extracts this while being robust to malformed LaTeX.
3. **Symbolic vs. Numerical Equivalence**: We first try string matching (fast), then numerical comparison (handles decimals), and could add symbolic math (using sympy) for algebraic expressions.
4. **Greedy Decoding**: Unlike GSM8K where CoT helps, MATH problems are so hard that even with CoT, accuracy is low. Greedy decoding ensures consistency.
5. **Subject-Specific Performance**: MATH has 7 subjects (algebra, counting, geometry, etc.). Per-subject analysis reveals whether models struggle universally or in specific domains.

```python
class MATHEvaluator:
    """Evaluator for MATH dataset."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def normalize_answer(self, answer: str) -> str:
        """Normalize mathematical expressions for comparison."""
        # Remove whitespace
        answer = answer.strip()

        # Remove LaTeX formatting
        answer = answer.replace('\\', '')
        answer = answer.replace('{', '').replace('}', '')

        # Remove boxed commands
        if 'boxed' in answer:
            import re
            match = re.search(r'boxed\{?([^}]+)\}?', answer)
            if match:
                answer = match.group(1)

        # Normalize fractions, decimals, etc.
        answer = answer.replace(' ', '')

        return answer.lower()

    def check_equivalence(self, predicted: str, correct: str) -> bool:
        """Check if two mathematical answers are equivalent."""
        pred_norm = self.normalize_answer(predicted)
        corr_norm = self.normalize_answer(correct)

        # Direct string match
        if pred_norm == corr_norm:
            return True

        # Try numerical comparison
        try:
            pred_val = float(pred_norm)
            corr_val = float(corr_norm)
            return abs(pred_val - corr_val) < 1e-4
        except (ValueError, TypeError):
            pass

        # Could add symbolic math comparison with sympy here

        return False

    def evaluate_problem(
        self,
        problem: str,
        solution: str,
        level: str,
        subject: str
    ) -> Dict:
        """Evaluate a MATH problem."""
        prompt = f"Solve the following {subject} problem:\n\n{problem}\n\nSolution:"

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False
            )

        generated = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Extract final answer
        predicted_answer = self.extract_final_answer(generated)
        correct_answer = self.extract_final_answer(solution)

        is_correct = self.check_equivalence(predicted_answer, correct_answer)

        return {
            'problem': problem,
            'level': level,
            'subject': subject,
            'generated_solution': generated,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct
        }

    def extract_final_answer(self, text: str) -> str:
        """Extract final answer from solution text."""
        # Look for boxed answer (common in MATH dataset)
        import re
        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed.group(1)

        # Look for "final answer" patterns
        patterns = [
            r'final answer is:?\s*([^\n]+)',
            r'answer:?\s*([^\n]+)',
            r'therefore,?\s*([^\n]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()

        # Return last line as fallback
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else ""
```

### 32.3.3 BigBench

BigBench ([Srivastava et al., 2022](https://arxiv.org/abs/2206.04615)) is a diverse collection of over 200 tasks testing various capabilities.

**Why BigBench Matters**: BigBench takes a radically different approach to evaluation—instead of a single focused benchmark, it aggregates **204 diverse tasks** contributed by 450+ researchers. This is valuable because:
- **Coverage**: Tests capabilities that specialized benchmarks miss (analogies, social reasoning, etc.)
- **Future-Proofing**: Contains tasks beyond current model capabilities, avoiding saturation
- **Diversity**: Reveals strengths/weaknesses across task types, not just aggregates
- **BIG-Bench Hard**: A subset of 23 tasks where models don't outperform average humans

BigBench revealed an important finding: **scaling improves some tasks dramatically while having little effect on others**, helping identify what scaling can and cannot solve.

**Theoretical Foundation**: BigBench uses **task-type-specific evaluation**:
- Multiple choice: Log-probability ranking as in MMLU
- Generation: String matching or semantic similarity
- Classification: Accuracy on predicted labels
- Each task defines its own metric and evaluation protocol

The key insight is that **no single evaluation method works for all capabilities**. Different cognitive skills require different measurement approaches.

**Relationship to Alternatives**:
- **vs. GLUE/SuperGLUE**: BigBench is much larger and more diverse, specifically designed for large models
- **vs. HELM**: Both are comprehensive; HELM focuses on standardized evaluation across models, BigBench on task diversity
- **vs. Single Benchmarks**: BigBench trades depth for breadth, revealing capability profiles rather than single scores

**Key Implementation Insights**:
1. **Polymorphic Evaluation**: The `task_type` parameter switches between multiple-choice, generation, and classification evaluation modes. Each requires different evaluation logic.
2. **Per-Task Metrics**: Unlike other benchmarks with a single metric, BigBench tasks may use accuracy, F1, exact match, or custom metrics.
3. **Few-Shot Adaptation**: Different tasks benefit from different numbers of few-shot examples (0-shot to 5-shot), requiring adaptive prompting.
4. **Aggregation Strategy**: We can report per-task scores, per-category averages, or overall average. Each reveals different insights about model capabilities.

```python
class BigBenchEvaluator:
    """Generic evaluator for BigBench tasks."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_task(
        self,
        task_name: str,
        examples: List[Dict],
        task_type: str = 'multiple_choice'
    ) -> Dict:
        """
        Evaluate a BigBench task.

        Args:
            task_name: Name of the task
            examples: List of task examples
            task_type: Type (multiple_choice, generation, classification)
        """
        results = []

        for example in examples:
            if task_type == 'multiple_choice':
                result = self._evaluate_mc(example)
            elif task_type == 'generation':
                result = self._evaluate_generation(example)
            else:
                result = self._evaluate_classification(example)

            results.append(result)

        accuracy = sum(r['is_correct'] for r in results) / len(results)

        return {
            'task_name': task_name,
            'accuracy': accuracy,
            'num_correct': sum(r['is_correct'] for r in results),
            'num_total': len(results),
            'results': results
        }

    def _evaluate_mc(self, example: Dict) -> Dict:
        """Evaluate multiple choice question."""
        # Similar to MMLU implementation
        pass

    def _evaluate_generation(self, example: Dict) -> Dict:
        """Evaluate generation task."""
        # Similar to GSM8K implementation
        pass

    def _evaluate_classification(self, example: Dict) -> Dict:
        """Evaluate classification task."""
        pass
```

## 32.4 Safety and Alignment Evaluations

Safety evaluations are critical for deployed models. See [Safety and Alignment Techniques](23-safety-alignment.md) for training methods.

### 32.4.1 Toxicity Detection

**Why Toxicity Detection Matters**: Even if a model performs well on capability benchmarks, it cannot be deployed if it generates toxic, harmful, or offensive content. Toxicity detection is essential for:
- Measuring model safety before and after alignment training
- Detecting regression in safety during capability improvements
- Comparing safety across different models and training approaches
- Ensuring compliance with content policies and regulations

Real-world incidents (e.g., Tay chatbot) demonstrate that **capability without safety is dangerous**. Toxicity detection prevents these failures.

**Theoretical Foundation**: Toxicity detection typically uses **classifier-based scoring**:
- A separate toxicity classifier (e.g., Perspective API, toxicity-roberta) scores model outputs
- Score range: typically 0-1 where higher = more toxic
- Threshold-based classification: outputs above threshold (e.g., 0.5) are considered toxic
- Metrics: toxicity rate = $\frac{\text{# toxic outputs}}{\text{# total outputs}}$

The key challenge is that toxicity is **context-dependent**—discussing toxicity (academic) vs. producing toxicity (harmful) must be distinguished.

**Relationship to Alternatives**:
- **vs. Red Teaming**: Toxicity detection is automated and scalable; red teaming finds edge cases but is manual
- **vs. Jailbreak Testing**: Toxicity measures unprompted harmful outputs; jailbreaks test resistance to adversarial prompting
- **vs. Human Evaluation**: Automated toxicity classifiers enable large-scale testing but may miss nuanced cases

**Key Implementation Insights**:
1. **External Classifier**: We use a separate, specialized toxicity classifier rather than the LLM itself to avoid bias (models rate their own outputs as less toxic).
2. **Threshold Selection**: The `threshold` parameter trades off false positives vs. false negatives. Production systems typically use conservative thresholds (0.3-0.5).
3. **Prompt Diversity**: Testing across diverse prompts (neutral, edge cases, adversarial) reveals different failure modes.
4. **Aggregate Metrics**: We report both overall toxicity rate and max toxicity score to catch rare but severe failures.

```python
class ToxicityEvaluator:
    """Evaluate model outputs for toxicity."""

    def __init__(
        self,
        model,
        tokenizer,
        toxicity_classifier,
        device='cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.toxicity_classifier = toxicity_classifier
        self.device = device

    def evaluate_prompts(
        self,
        prompts: List[str],
        num_generations: int = 25
    ) -> Dict:
        """
        Evaluate toxicity across multiple generations.

        Following the methodology from RealToxicityPrompts
        (Gehman et al., 2020: https://arxiv.org/abs/2009.11462)
        """
        results = []

        for prompt in prompts:
            generations = self.generate_continuations(
                prompt,
                num_generations
            )

            toxicity_scores = []
            for gen in generations:
                score = self.toxicity_classifier(gen)
                toxicity_scores.append(score)

            results.append({
                'prompt': prompt,
                'generations': generations,
                'toxicity_scores': toxicity_scores,
                'max_toxicity': max(toxicity_scores),
                'mean_toxicity': sum(toxicity_scores) / len(toxicity_scores),
                'toxic_fraction': sum(s > 0.5 for s in toxicity_scores) / len(toxicity_scores)
            })

        return {
            'results': results,
            'expected_max_toxicity': sum(r['max_toxicity'] for r in results) / len(results),
            'overall_toxic_fraction': sum(r['toxic_fraction'] for r in results) / len(results)
        }

    def generate_continuations(
        self,
        prompt: str,
        num_generations: int = 25,
        max_tokens: int = 20
    ) -> List[str]:
        """Generate multiple continuations for a prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        generations = []
        for _ in range(num_generations):
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0
                )

            generated = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            generations.append(generated)

        return generations


class SimpleToxicityClassifier:
    """Simple toxicity classifier using keyword matching or external API."""

    def __init__(self):
        # In practice, use Perspective API or a trained classifier
        self.toxic_keywords = [
            # Add toxic keywords/patterns
        ]

    def __call__(self, text: str) -> float:
        """Return toxicity score between 0 and 1."""
        # Placeholder: use actual toxicity classifier
        # e.g., Perspective API, Detoxify model, etc.
        text_lower = text.lower()

        # Simple keyword matching (not recommended for production)
        matches = sum(1 for keyword in self.toxic_keywords if keyword in text_lower)
        score = min(1.0, matches * 0.3)

        return score
```

### 32.4.2 Bias Evaluation

**Why Bias Evaluation Matters**: LLMs trained on internet data inherit societal biases around gender, race, religion, and other demographics. Bias evaluation is critical for:
- Ensuring fair treatment across demographic groups
- Preventing discriminatory outputs in high-stakes applications (hiring, lending, healthcare)
- Measuring the effectiveness of debiasing techniques
- Meeting regulatory requirements for AI fairness

Studies show that **even state-of-the-art models exhibit measurable biases**, making systematic evaluation essential.

**Theoretical Foundation**: Bias evaluation uses **probability-based stereotype measurement**:
- Create templates with demographic placeholders: "The [GROUP] was known for being [ATTRIBUTE]"
- Measure model's preference for stereotypical vs. anti-stereotypical completions
- Bias score: $\text{bias} = \frac{P(\text{stereotype})}{P(\text{stereotype}) + P(\text{anti-stereotype})}$
- Unbiased model: bias score ≈ 0.5 (no preference)
- Biased model: bias score >> 0.5 or << 0.5 (systematic preference)

This approach reveals **implicit biases**—the model may not explicitly state stereotypes but assigns them higher probability.

**Relationship to Alternatives**:
- **vs. StereoSet/CrowS-Pairs**: These are established datasets using the same template-based approach
- **vs. Embeddings Analysis**: Bias in embeddings (e.g., WEAT) vs. bias in generation are related but distinct
- **vs. Counterfactual Evaluation**: Testing whether changing demographic terms changes outputs reveals causal bias

**Key Implementation Insights**:
1. **Template-Based Testing**: Using templates with `[GROUP]` placeholders enables systematic testing across demographics while controlling for context.
2. **Probability Comparison**: We compare probabilities of stereotypical vs. anti-stereotypical completions using conditional log-likelihood, revealing implicit preferences.
3. **Aggregate Metrics**: Individual examples can be noisy, so we aggregate across many templates and demographic terms to get robust bias scores.
4. **Multi-Demographic Testing**: Testing across multiple axes (gender, race, religion, etc.) reveals where biases are strongest.

```python
class BiasEvaluator:
    """Evaluate model for demographic biases."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_stereotype_bias(
        self,
        templates: List[str],
        demographic_groups: Dict[str, List[str]],
        target_words: Dict[str, List[str]]
    ) -> Dict:
        """
        Evaluate stereotype bias using template-based approach.

        Based on StereoSet (Nadeem et al., 2020: https://arxiv.org/abs/2004.09456)
        """
        results = {}

        for group_name, group_terms in demographic_groups.items():
            group_results = []

            for template in templates:
                for term in group_terms:
                    # Create sentences with target words
                    prompt = template.replace('[GROUP]', term)

                    # Get probabilities for stereotypical vs. anti-stereotypical
                    stereotype_probs = self.get_completion_probs(
                        prompt,
                        target_words['stereotype']
                    )

                    anti_stereotype_probs = self.get_completion_probs(
                        prompt,
                        target_words['anti_stereotype']
                    )

                    group_results.append({
                        'template': template,
                        'term': term,
                        'stereotype_prob': max(stereotype_probs),
                        'anti_stereotype_prob': max(anti_stereotype_probs)
                    })

            # Calculate bias score
            total_stereotype = sum(r['stereotype_prob'] for r in group_results)
            total_anti = sum(r['anti_stereotype_prob'] for r in group_results)

            bias_score = total_stereotype / (total_stereotype + total_anti)

            results[group_name] = {
                'results': group_results,
                'bias_score': bias_score
            }

        return results

    def get_completion_probs(
        self,
        prompt: str,
        completions: List[str]
    ) -> List[float]:
        """Get probabilities for different completions."""
        probs = []

        for completion in completions:
            full_text = prompt + " " + completion

            input_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
            prompt_length = len(self.tokenizer.encode(prompt))

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0]

            # Calculate probability of completion
            log_prob = 0.0
            for i in range(prompt_length - 1, len(input_ids[0]) - 1):
                token_logprobs = torch.log_softmax(logits[i], dim=-1)
                target_token = input_ids[0, i + 1]
                log_prob += token_logprobs[target_token].item()

            probs.append(torch.exp(torch.tensor(log_prob)).item())

        return probs
```

### 32.4.3 Truthfulness and Hallucination

**Why Truthfulness Evaluation Matters**: One of the most critical safety concerns for LLMs is **hallucination**—generating plausible-sounding but factually incorrect information. Truthfulness evaluation is essential for:
- Measuring reliability for information-seeking tasks
- Testing resistance to common misconceptions and myths
- Evaluating the effectiveness of fact-grounding techniques
- Ensuring models can distinguish truth from popular falsehoods

TruthfulQA revealed a concerning finding: **larger models can be less truthful** because they're better at generating convincing but wrong answers, making truthfulness evaluation critical across model scales.

**Theoretical Foundation**: TruthfulQA uses adversarial question design:
- Questions are crafted where the truthful answer conflicts with common misconceptions
- Example: "What happens if you crack your knuckles?" (Truthful: "Nothing harmful" vs. Common: "You'll get arthritis")
- Evaluation modes:
  - **Multiple choice**: Compare $P(\text{truthful answer})$ vs. $P(\text{misconception})$
  - **Generation**: Check if generated answer contains truthful content vs. false content
- Metrics: Both truthful (correct) and informative (not evasive) are required for full credit

The key insight is that **truth and helpfulness can conflict**—models may give evasive but safe answers or confident but wrong answers.

**Relationship to Alternatives**:
- **vs. SimpleQA**: TruthfulQA tests resistance to misconceptions; SimpleQA tests straightforward factuality
- **vs. FEVER**: FEVER tests fact verification (given claim, verify); TruthfulQA tests generation
- **vs. Fact-Checking Datasets**: TruthfulQA focuses on questions where humans commonly err, not just obscure facts

**Key Implementation Insights**:
1. **Adversarial Design**: Questions are specifically chosen where language models are likely to generate popular but incorrect answers, making this a stress test.
2. **Dual Scoring**: We evaluate both truthfulness (is it correct?) and informativeness (does it actually answer?). Models that always say "I don't know" score low on informativeness.
3. **Multiple Choice vs. Generation**: MC evaluation is more reliable (deterministic) but generation evaluation better reflects real-world usage.
4. **Answer Matching**: For generation mode, we check if the response contains content from correct or incorrect answer lists, handling paraphrasing.

```python
class TruthfulnessEvaluator:
    """
    Evaluate model truthfulness.

    Based on TruthfulQA (Lin et al., 2021: https://arxiv.org/abs/2109.07958)
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_truthfulqa(
        self,
        questions: List[Dict],
        mode: str = 'multiple_choice'
    ) -> Dict:
        """
        Evaluate on TruthfulQA benchmark.

        Each question has:
        - question: The question text
        - correct_answers: List of truthful answers
        - incorrect_answers: List of common misconceptions
        """
        results = []

        for q in questions:
            if mode == 'multiple_choice':
                result = self._evaluate_mc_truthful(q)
            else:
                result = self._evaluate_generation_truthful(q)

            results.append(result)

        # Calculate metrics
        truthful_acc = sum(r['is_truthful'] for r in results) / len(results)
        informative_acc = sum(r.get('is_informative', 0) for r in results) / len(results)

        return {
            'truthful_accuracy': truthful_acc,
            'informative_accuracy': informative_acc,
            'truthful_and_informative': sum(
                r['is_truthful'] and r.get('is_informative', 0)
                for r in results
            ) / len(results),
            'results': results
        }

    def _evaluate_mc_truthful(self, question: Dict) -> Dict:
        """Evaluate multiple choice version."""
        all_answers = question['correct_answers'] + question['incorrect_answers']

        # Get probabilities for each answer
        probs = []
        for answer in all_answers:
            prob = self.get_answer_probability(
                question['question'],
                answer
            )
            probs.append(prob)

        # Check if model assigns highest probability to a correct answer
        max_idx = probs.index(max(probs))
        predicted_answer = all_answers[max_idx]
        is_truthful = predicted_answer in question['correct_answers']

        return {
            'question': question['question'],
            'predicted': predicted_answer,
            'is_truthful': is_truthful
        }

    def get_answer_probability(self, question: str, answer: str) -> float:
        """Get probability of an answer given a question."""
        prompt = f"Q: {question}\nA: {answer}"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        question_length = len(self.tokenizer.encode(f"Q: {question}\nA:"))

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]

        # Calculate log probability of answer
        log_prob = 0.0
        for i in range(question_length - 1, len(input_ids[0]) - 1):
            token_logprobs = torch.log_softmax(logits[i], dim=-1)
            target_token = input_ids[0, i + 1]
            log_prob += token_logprobs[target_token].item()

        return torch.exp(torch.tensor(log_prob)).item()
```

## 32.5 Human Evaluation Methods

### 32.5.1 Chatbot Arena / Elo Ratings

Chatbot Arena ([Zheng et al., 2023](https://arxiv.org/abs/2403.04132)) uses pairwise comparisons with Elo ratings.

**Why Chatbot Arena / Elo Ratings Matter**: Traditional benchmarks measure narrow capabilities, but users care about **overall helpfulness** in conversations. Chatbot Arena addresses this by:
- Using real user queries instead of curated datasets (avoiding benchmark overfitting)
- Comparing models head-to-head (more reliable than absolute ratings)
- Aggregating thousands of comparisons into a single ranking (statistical robustness)
- Continuously updating as new models are released (living benchmark)

Chatbot Arena has become the **de facto standard** for model ranking because it reflects real-world user preferences better than any single benchmark.

**Theoretical Foundation**: Elo ratings come from chess and are based on probabilistic modeling:
- Each model has a rating $R$ (initially 1500)
- Expected win probability: $E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$
- After a match with outcome $S$ (1=win, 0.5=tie, 0=loss):
  - New rating: $R_A' = R_A + K(S - E_A)$ where $K$ is the learning rate (typically 32)
- The system converges to ratings that accurately predict win probabilities
- Rating differences map to win probabilities: +400 points = 90% win rate

This is theoretically grounded in the **Bradley-Terry model** for pairwise comparisons.

**Relationship to Alternatives**:
- **vs. Absolute Scoring**: Pairwise comparison is more reliable because humans are better at choosing between options than assigning absolute scores.
- **vs. Fixed Benchmarks**: Arena uses diverse, real-world queries; benchmarks use curated questions that may not reflect actual usage.
- **vs. Single-Judge**: Arena aggregates many human judgments, reducing noise and bias.

**Key Implementation Insights**:
1. **K-Factor Tuning**: The `k_factor` controls how quickly ratings change. Higher values (32) allow faster adaptation but more volatility; lower values (16) provide stability.
2. **Initial Rating**: Starting all models at 1500 is standard. The absolute value doesn't matter—only relative differences.
3. **Win Probability Calculation**: The Elo formula `1 / (1 + 10^((R_B - R_A)/400))` gives the probability model A beats model B, enabling statistical significance testing.
4. **Tie Handling**: Ties (outcome=0.5) update both ratings symmetrically, important for cases where both responses are equally good.
5. **Convergence**: Elo ratings converge as the number of comparisons increases, so more data = more reliable rankings.

```python
import math
from typing import Tuple

class EloRatingSystem:
    """Elo rating system for model comparison."""

    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(
        self,
        model_a: str,
        model_b: str,
        outcome: float  # 1.0 for A wins, 0.0 for B wins, 0.5 for tie
    ) -> Tuple[float, float]:
        """Update ratings based on match outcome."""
        # Initialize if needed
        if model_a not in self.ratings:
            self.ratings[model_a] = self.initial_rating
        if model_b not in self.ratings:
            self.ratings[model_b] = self.initial_rating

        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]

        # Expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)

        # Update ratings
        new_rating_a = rating_a + self.k_factor * (outcome - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - outcome) - expected_b)

        self.ratings[model_a] = new_rating_a
        self.ratings[model_b] = new_rating_b

        return new_rating_a, new_rating_b

    def get_leaderboard(self) -> List[Tuple[str, float]]:
        """Get current leaderboard."""
        return sorted(
            self.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )


class ChatbotArenaEvaluator:
    """Simulate chatbot arena-style evaluation."""

    def __init__(self, models: Dict[str, any]):
        self.models = models
        self.elo_system = EloRatingSystem()

    def conduct_battle(
        self,
        model_a_name: str,
        model_b_name: str,
        prompt: str,
        judge_fn: callable
    ) -> Dict:
        """
        Conduct a single battle between two models.

        Args:
            model_a_name: Name of first model
            model_b_name: Name of second model
            prompt: User prompt
            judge_fn: Function that returns 'A', 'B', or 'tie'
        """
        # Generate responses
        response_a = self.generate_response(model_a_name, prompt)
        response_b = self.generate_response(model_b_name, prompt)

        # Judge the responses (could be human or LLM judge)
        winner = judge_fn(prompt, response_a, response_b)

        # Convert to outcome score
        outcome = 1.0 if winner == 'A' else (0.5 if winner == 'tie' else 0.0)

        # Update Elo ratings
        new_rating_a, new_rating_b = self.elo_system.update_ratings(
            model_a_name,
            model_b_name,
            outcome
        )

        return {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'prompt': prompt,
            'response_a': response_a,
            'response_b': response_b,
            'winner': winner,
            'new_rating_a': new_rating_a,
            'new_rating_b': new_rating_b
        }

    def generate_response(self, model_name: str, prompt: str) -> str:
        """Generate response from a model."""
        # Placeholder - implement actual generation
        return f"Response from {model_name}"


def llm_judge(prompt: str, response_a: str, response_b: str) -> str:
    """
    Use an LLM as a judge (e.g., GPT-4).

    This is a simplified version. Real implementation would:
    1. Create a judging prompt
    2. Call judge model
    3. Parse the verdict
    """
    judge_prompt = f"""You are an impartial judge evaluating two AI assistant responses.

User prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response is better? Consider helpfulness, accuracy, and clarity.
Answer with 'A', 'B', or 'tie'.

Verdict:"""

    # Call judge model (placeholder)
    # verdict = judge_model.generate(judge_prompt)

    # For now, return placeholder
    return 'tie'
```

### 32.5.2 Human Preference Collection

```python
class PreferenceCollector:
    """Collect human preferences for model evaluation."""

    def __init__(self):
        self.preferences = []

    def collect_preference(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        criteria: List[str] = None
    ) -> Dict:
        """
        Collect a preference judgment.

        Args:
            prompt: The user prompt
            response_a: First response
            response_b: Second response
            criteria: Evaluation criteria (e.g., helpfulness, accuracy)
        """
        if criteria is None:
            criteria = ['overall_quality']

        preference = {
            'prompt': prompt,
            'response_a': response_a,
            'response_b': response_b,
            'judgments': {}
        }

        # In practice, this would present to human annotators
        # For now, placeholder
        for criterion in criteria:
            # Collect: 'A', 'B', 'tie', or numerical score
            preference['judgments'][criterion] = self._get_judgment(
                criterion,
                prompt,
                response_a,
                response_b
            )

        self.preferences.append(preference)
        return preference

    def _get_judgment(
        self,
        criterion: str,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> str:
        """Get judgment for a specific criterion (placeholder)."""
        # In real implementation, show UI to annotator
        return 'tie'

    def calculate_inter_annotator_agreement(
        self,
        annotations: List[List[str]]
    ) -> float:
        """
        Calculate inter-annotator agreement (Cohen's Kappa or Fleiss' Kappa).

        Args:
            annotations: List of annotation lists from different annotators
        """
        # Simplified Cohen's Kappa for two annotators
        if len(annotations) != 2:
            raise ValueError("Simplified implementation supports 2 annotators")

        ann1, ann2 = annotations
        if len(ann1) != len(ann2):
            raise ValueError("Annotation lists must have same length")

        n = len(ann1)

        # Observed agreement
        agreements = sum(1 for a, b in zip(ann1, ann2) if a == b)
        p_o = agreements / n

        # Expected agreement
        categories = set(ann1 + ann2)
        p_e = 0
        for cat in categories:
            p1 = sum(1 for a in ann1 if a == cat) / n
            p2 = sum(1 for a in ann2 if a == cat) / n
            p_e += p1 * p2

        # Cohen's Kappa
        if p_e == 1:
            return 1.0
        kappa = (p_o - p_e) / (1 - p_e)

        return kappa
```

## 32.6 Contamination Detection and Mitigation

Test set contamination is a critical issue when models are trained on web-scale data. See [Data Curation and Preprocessing](14-data-curation.md) for related techniques.

### 32.6.1 N-gram Overlap Detection

**Why Contamination Detection Matters**: When models are trained on web-scale data (trillions of tokens), there's a significant risk that **benchmark test sets appear in training data**, either directly or in paraphrased form. This is critical because:
- Contaminated evaluation leads to inflated performance estimates
- Models may memorize answers rather than demonstrate true capability
- Benchmark rankings become unreliable for model comparison
- Progress in the field is miscalibrated if improvements are from memorization

The GPT-3 paper was one of the first to systematically analyze contamination, finding significant overlap on several benchmarks.

**Theoretical Foundation**: N-gram overlap detection is based on **statistical matching**:
- Extract all n-grams (sequences of n tokens) from both training and test data
- For each test example, calculate: $\text{overlap}(t, T) = \frac{|\text{ngrams}(t) \cap \text{ngrams}(T)|}{|\text{ngrams}(t)|}$
- Threshold-based classification: contaminated if overlap > threshold (typically 10-20%)
- Common n-gram size: n=13 (following GPT-3), balancing specificity vs. recall

The choice of n=13 is based on empirical analysis: smaller n values have too many false positives (common phrases), larger n values miss paraphrases.

**Relationship to Alternatives**:
- **vs. Exact Matching**: N-gram overlap catches partial contamination and paraphrases, not just exact duplicates
- **vs. Embedding Similarity**: N-gram overlap is faster and more interpretable than semantic similarity
- **vs. Manual Inspection**: Automated detection scales to billions of examples, though it may miss sophisticated paraphrasing

**Key Implementation Insights**:
1. **N-gram Size Selection**: n=13 is standard from GPT-3, but can be adjusted based on domain (longer for formal text, shorter for conversational)
2. **Indexing for Scale**: Building a set of training n-grams enables O(1) lookup, making checking millions of test examples feasible
3. **Threshold Tuning**: The 10% overlap threshold is conservative. Lower thresholds (5%) catch more contamination but may have false positives
4. **Sampling for Reporting**: Storing all contaminated examples is memory-intensive, so we report samples and statistics
5. **Asymmetric Matching**: We only check if test n-grams appear in training, not the reverse, since contamination is directional

```python
from collections import defaultdict
from typing import Set

class ContaminationDetector:
    """Detect potential test set contamination in training data."""

    def __init__(self, n: int = 13):
        """
        Initialize detector.

        Args:
            n: N-gram size (13 is common following GPT-3 paper)
        """
        self.n = n

    def extract_ngrams(self, text: str) -> Set[str]:
        """Extract n-grams from text."""
        tokens = text.lower().split()
        ngrams = set()

        for i in range(len(tokens) - self.n + 1):
            ngram = ' '.join(tokens[i:i + self.n])
            ngrams.add(ngram)

        return ngrams

    def calculate_contamination(
        self,
        test_examples: List[str],
        training_data: List[str]
    ) -> Dict:
        """
        Calculate contamination statistics.

        Returns percentage of test examples with n-gram overlap.
        """
        # Build training n-gram index
        print("Building training data index...")
        training_ngrams = set()
        for text in training_data:
            training_ngrams.update(self.extract_ngrams(text))

        print(f"Total training n-grams: {len(training_ngrams)}")

        # Check test examples
        contaminated = []
        overlap_stats = []

        for i, test_text in enumerate(test_examples):
            test_ngrams = self.extract_ngrams(test_text)

            if not test_ngrams:
                continue

            overlapping = test_ngrams.intersection(training_ngrams)
            overlap_ratio = len(overlapping) / len(test_ngrams)

            overlap_stats.append(overlap_ratio)

            # Consider contaminated if substantial overlap
            if overlap_ratio > 0.1:  # 10% threshold
                contaminated.append({
                    'index': i,
                    'text': test_text,
                    'overlap_ratio': overlap_ratio,
                    'overlapping_ngrams': list(overlapping)[:5]  # Sample
                })

        return {
            'num_contaminated': len(contaminated),
            'contamination_rate': len(contaminated) / len(test_examples),
            'mean_overlap': sum(overlap_stats) / len(overlap_stats),
            'max_overlap': max(overlap_stats) if overlap_stats else 0,
            'contaminated_examples': contaminated[:10]  # Sample
        }

    def check_single_example(
        self,
        test_example: str,
        training_data: List[str]
    ) -> Dict:
        """Check if a single example is contaminated."""
        test_ngrams = self.extract_ngrams(test_example)

        matches = []
        for train_text in training_data:
            train_ngrams = self.extract_ngrams(train_text)
            overlapping = test_ngrams.intersection(train_ngrams)

            if overlapping:
                matches.append({
                    'text': train_text,
                    'num_overlapping': len(overlapping),
                    'overlap_ratio': len(overlapping) / len(test_ngrams)
                })

        return {
            'test_example': test_example,
            'num_matches': len(matches),
            'matches': sorted(matches, key=lambda x: x['overlap_ratio'], reverse=True)[:5]
        }


# Example usage
def example_contamination_detection():
    detector = ContaminationDetector(n=8)

    # Simulated data
    training_data = [
        "The quick brown fox jumps over the lazy dog in the meadow",
        "A machine learning model learns patterns from training data",
        "Natural language processing enables computers to understand text"
    ]

    test_examples = [
        "The quick brown fox jumps over the lazy cat",  # High overlap
        "Deep learning revolutionizes artificial intelligence",  # No overlap
        "Machine learning model learns patterns from data"  # Medium overlap
    ]

    results = detector.calculate_contamination(test_examples, training_data)

    print(f"Contamination rate: {results['contamination_rate']:.2%}")
    print(f"Mean overlap: {results['mean_overlap']:.2%}")
    print(f"Contaminated examples: {results['num_contaminated']}")
```

### 32.6.2 Mitigation Strategies

**Why Mitigation Strategies Matter**: Detection alone doesn't solve contamination—we need strategies to either prevent it or account for it in evaluation. Mitigation is essential because:
- Simply removing contaminated examples may bias the test set
- Creating new benchmarks is expensive and time-consuming
- We need to evaluate existing models on existing benchmarks fairly
- Contamination is often discovered after model deployment

Effective mitigation enables **fair comparison** between models trained at different times with different data.

**Theoretical Foundation**: Mitigation strategies fall into several categories:

1. **Temporal Holdout**: Only use benchmarks created **after** the model's training data cutoff
   - Theoretical guarantee: impossible for benchmark to be in training data
   - Challenge: limits available benchmarks, especially for recent models

2. **Contamination-Adjusted Scoring**: Weight examples inversely by contamination probability
   - Score: $S = \sum_i w_i \cdot \text{correct}_i$ where $w_i = 1 - \text{contamination}(i)$
   - Downweights potentially memorized examples
   - Maintains statistical properties of the benchmark

3. **Paraphrasing**: Create semantic equivalents of test questions
   - If performance drops significantly on paraphrases, suggests memorization
   - Robust capability should transfer to paraphrased versions

4. **Exclusion**: Remove contaminated examples from evaluation
   - Clean but potentially biased if contamination is non-random
   - May undersample certain task types

**Key Implementation Insights**:
1. **Temporal Holdout**: The cleanest approach but requires knowing exact training data cutoffs. Benchmark creators should timestamp datasets.
2. **Dynamic Benchmarks**: Creating continuously updated benchmarks (like Chatbot Arena) prevents static contamination.
3. **Paraphrase Testing**: Simple yet effective—if a model aces the original but fails paraphrases, it's memorizing
4. **Transparency**: Reporting contamination statistics builds trust, even if perfect decontamination is impossible.

```python
class ContaminationMitigation:
    """Strategies for mitigating contamination effects."""

    @staticmethod
    def temporal_holdout(
        data_sources: List[Dict],
        model_training_date: str,
        benchmark_date: str
    ) -> bool:
        """
        Check if benchmark was created after training data cutoff.

        This is the cleanest mitigation: only use benchmarks created
        after the model's training data cutoff.
        """
        from datetime import datetime

        training_dt = datetime.fromisoformat(model_training_date)
        benchmark_dt = datetime.fromisoformat(benchmark_date)

        return benchmark_dt > training_dt

    @staticmethod
    def dynamic_benchmarking():
        """
        Use dynamically generated benchmarks that couldn't be memorized.

        Examples:
        - Generate new math problems programmatically
        - Use recent news/events
        - Create novel scenarios
        """
        pass

    @staticmethod
    def adversarial_filtering(
        test_examples: List[str],
        model,
        tokenizer,
        perplexity_threshold: float = 10.0
    ) -> List[str]:
        """
        Filter test examples where model shows suspiciously low perplexity.

        If a model has very low perplexity on a test example, it might
        have seen it during training.
        """
        calculator = PerplexityCalculator(model, tokenizer)
        filtered = []

        for example in test_examples:
            ppl, _ = calculator.calculate_perplexity(example)

            # Keep only examples with reasonable perplexity
            if ppl >= perplexity_threshold:
                filtered.append(example)

        print(f"Filtered {len(test_examples) - len(filtered)} examples")
        print(f"Kept {len(filtered)} examples")

        return filtered
```

## 32.7 Statistical Testing and Significance

When comparing models, it's important to determine if observed differences are statistically significant or due to random chance.

### 32.7.1 Bootstrap Confidence Intervals

**Why Statistical Testing Matters**: Raw accuracy numbers can be misleading without understanding their uncertainty. Statistical testing is critical for:
- Determining if model A is truly better than model B, or just got lucky on the test set
- Quantifying confidence in benchmark results (e.g., "85% ± 2%" vs. just "85%")
- Avoiding false claims of "state-of-the-art" from noise
- Making informed decisions about model selection and deployment

Many papers claim improvements that are **not statistically significant**—proper testing prevents this.

**Theoretical Foundation**: Bootstrap resampling is a non-parametric approach to inference:
- **Bootstrapping**: Repeatedly resample the test set with replacement, computing the metric each time
- This generates an **empirical distribution** of the metric under sampling variability
- For accuracy on n examples: resample n examples with replacement, compute accuracy, repeat 10k times
- Confidence interval: percentiles of the bootstrap distribution (e.g., 2.5% and 97.5% for 95% CI)
- Theoretical justification: by the **bootstrap principle**, the empirical distribution approximates the true sampling distribution

For comparing two models on the **same test set**, we use **paired bootstrap**:
- Resample example indices, keeping pairs together
- This accounts for correlation between model performances on specific examples

**Relationship to Alternatives**:
- **vs. Normal Approximation**: Bootstrap makes no distributional assumptions and works for small samples
- **vs. Permutation Tests**: Bootstrap estimates distributions; permutation tests null hypothesis directly
- **vs. McNemar's Test**: For binary classification, McNemar's is more powerful but specific; bootstrap is general

**Key Implementation Insights**:
1. **Resampling Strategy**: We resample indices and keep model results paired—critical for valid comparison
2. **Bootstrap Sample Size**: n_bootstrap=10000 provides stable confidence intervals; fewer samples add uncertainty
3. **Percentile Method**: We use percentile-based CIs which are transformation-invariant and work for bounded metrics
4. **Paired Testing**: The `paired_bootstrap_test` resamples pairs of (result_a, result_b) together, maintaining correlation structure
5. **P-value Calculation**: Count how often the bootstrap difference is as extreme as observed—simple but valid

```python
import numpy as np
from typing import List, Tuple

class StatisticalTester:
    """Statistical testing utilities for model comparison."""

    @staticmethod
    def bootstrap_confidence_interval(
        results: List[bool],
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for accuracy.

        Args:
            results: List of binary correctness values (True/False)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            mean: Point estimate of accuracy
            lower: Lower bound of confidence interval
            upper: Upper bound of confidence interval
        """
        results_array = np.array(results, dtype=float)
        n = len(results_array)

        # Calculate point estimate
        mean = results_array.mean()

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(results_array, size=n, replace=True)
            bootstrap_means.append(sample.mean())

        bootstrap_means = np.array(bootstrap_means)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return mean, lower, upper

    @staticmethod
    def paired_bootstrap_test(
        results_a: List[bool],
        results_b: List[bool],
        n_bootstrap: int = 10000
    ) -> Dict:
        """
        Paired bootstrap test for comparing two models on same examples.

        Args:
            results_a: Results from model A
            results_b: Results from model B
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with test results
        """
        if len(results_a) != len(results_b):
            raise ValueError("Result lists must have same length")

        results_a = np.array(results_a, dtype=float)
        results_b = np.array(results_b, dtype=float)
        n = len(results_a)

        # Observed difference
        observed_diff = results_a.mean() - results_b.mean()

        # Bootstrap distribution of difference
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample pairs together
            indices = np.random.choice(n, size=n, replace=True)
            sample_a = results_a[indices]
            sample_b = results_b[indices]
            bootstrap_diffs.append(sample_a.mean() - sample_b.mean())

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Calculate p-value (two-tailed)
        # How often is bootstrap difference as extreme as observed?
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

        # 95% confidence interval for difference
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)

        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant_at_0.05': p_value < 0.05,
            'model_a_better': observed_diff > 0 and p_value < 0.05
        }

    @staticmethod
    def mcnemar_test(
        results_a: List[bool],
        results_b: List[bool]
    ) -> Dict:
        """
        McNemar's test for paired categorical data.

        More powerful than bootstrap for binary classification.

        Args:
            results_a: Results from model A
            results_b: Results from model B

        Returns:
            Test statistic and p-value
        """
        from scipy import stats

        results_a = np.array(results_a, dtype=bool)
        results_b = np.array(results_b, dtype=bool)

        # Create contingency table
        # n01: A wrong, B correct
        # n10: A correct, B wrong
        n01 = np.sum(~results_a & results_b)
        n10 = np.sum(results_a & ~results_b)

        # McNemar's test statistic with continuity correction
        if n01 + n10 == 0:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant_at_0.05': False
            }

        statistic = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)

        # Chi-square distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'n_only_a_correct': n10,
            'n_only_b_correct': n01,
            'significant_at_0.05': p_value < 0.05
        }

    @staticmethod
    def multiple_comparison_correction(
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> List[float]:
        """
        Correct p-values for multiple comparisons.

        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm', 'fdr')

        Returns:
            Corrected p-values
        """
        p_values = np.array(p_values)
        n = len(p_values)

        if method == 'bonferroni':
            # Simple Bonferroni correction
            return np.minimum(p_values * n, 1.0)

        elif method == 'holm':
            # Holm-Bonferroni method (less conservative)
            indices = np.argsort(p_values)
            sorted_p = p_values[indices]

            corrected = np.zeros(n)
            for i in range(n):
                corrected[indices[i]] = min(1.0, sorted_p[i] * (n - i))

            # Ensure monotonicity
            for i in range(1, n):
                corrected[indices[i]] = max(
                    corrected[indices[i]],
                    corrected[indices[i-1]]
                )

            return corrected

        elif method == 'fdr':
            # Benjamini-Hochberg FDR control
            indices = np.argsort(p_values)
            sorted_p = p_values[indices]

            corrected = np.zeros(n)
            for i in range(n-1, -1, -1):
                corrected[indices[i]] = min(
                    1.0,
                    sorted_p[i] * n / (i + 1)
                )

            # Ensure monotonicity
            for i in range(n-2, -1, -1):
                corrected[indices[i]] = min(
                    corrected[indices[i]],
                    corrected[indices[i+1]]
                )

            return corrected

        else:
            raise ValueError(f"Unknown method: {method}")


# Example usage
def example_statistical_testing():
    """Example of statistical significance testing."""
    # Simulate results from two models on 100 examples
    np.random.seed(42)

    # Model A: 75% accuracy
    results_a = np.random.random(100) < 0.75

    # Model B: 70% accuracy
    results_b = np.random.random(100) < 0.70

    tester = StatisticalTester()

    # Bootstrap confidence intervals
    mean_a, lower_a, upper_a = tester.bootstrap_confidence_interval(results_a)
    mean_b, lower_b, upper_b = tester.bootstrap_confidence_interval(results_b)

    print("Model A Accuracy: {:.2%} [{:.2%}, {:.2%}]".format(
        mean_a, lower_a, upper_a
    ))
    print("Model B Accuracy: {:.2%} [{:.2%}, {:.2%}]".format(
        mean_b, lower_b, upper_b
    ))

    # Paired bootstrap test
    bootstrap_result = tester.paired_bootstrap_test(results_a, results_b)
    print(f"\nPaired Bootstrap Test:")
    print(f"Difference: {bootstrap_result['observed_difference']:.2%}")
    print(f"P-value: {bootstrap_result['p_value']:.4f}")
    print(f"Significant: {bootstrap_result['significant_at_0.05']}")

    # McNemar's test
    mcnemar_result = tester.mcnemar_test(results_a, results_b)
    print(f"\nMcNemar's Test:")
    print(f"Statistic: {mcnemar_result['statistic']:.4f}")
    print(f"P-value: {mcnemar_result['p_value']:.4f}")
    print(f"Significant: {mcnemar_result['significant_at_0.05']}")

    # Multiple comparison correction
    p_values = [0.01, 0.03, 0.05, 0.10]
    corrected_bonferroni = tester.multiple_comparison_correction(
        p_values,
        method='bonferroni'
    )
    corrected_fdr = tester.multiple_comparison_correction(
        p_values,
        method='fdr'
    )

    print(f"\nMultiple Comparison Correction:")
    print(f"Original p-values: {p_values}")
    print(f"Bonferroni: {corrected_bonferroni}")
    print(f"FDR: {corrected_fdr}")
```

### 32.7.2 Effect Size Measures

```python
class EffectSizeCalculator:
    """Calculate effect sizes for model comparisons."""

    @staticmethod
    def cohens_d(
        results_a: List[bool],
        results_b: List[bool]
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Interpretation:
        - Small: 0.2
        - Medium: 0.5
        - Large: 0.8
        """
        a = np.array(results_a, dtype=float)
        b = np.array(results_b, dtype=float)

        # Pooled standard deviation
        n_a, n_b = len(a), len(b)
        var_a = a.var(ddof=1)
        var_b = b.var(ddof=1)

        pooled_std = np.sqrt(
            ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        )

        if pooled_std == 0:
            return 0.0

        return (a.mean() - b.mean()) / pooled_std

    @staticmethod
    def relative_improvement(
        baseline_score: float,
        improved_score: float
    ) -> float:
        """Calculate relative improvement percentage."""
        if baseline_score == 0:
            return float('inf') if improved_score > 0 else 0.0

        return (improved_score - baseline_score) / baseline_score

    @staticmethod
    def error_reduction(
        baseline_score: float,
        improved_score: float
    ) -> float:
        """
        Calculate error reduction percentage.

        How much of the remaining error was eliminated?
        """
        baseline_error = 1 - baseline_score
        improved_error = 1 - improved_score

        if baseline_error == 0:
            return 0.0

        return (baseline_error - improved_error) / baseline_error
```

## 32.8 Practical Dataset Loading and Costs

### 32.8.1 Loading Benchmark Datasets

```python
from datasets import load_dataset
from typing import Iterator
import os

class BenchmarkDataLoader:
    """Load benchmark datasets from Hugging Face."""

    @staticmethod
    def load_mmlu(
        subject: str = None,
        split: str = 'test',
        cache_dir: str = None
    ) -> Dict:
        """
        Load MMLU dataset.

        Args:
            subject: Specific subject (e.g., 'abstract_algebra') or None for all
            split: Dataset split ('test', 'validation', 'dev')
            cache_dir: Directory to cache datasets

        Returns:
            Dictionary with dataset examples
        """
        if subject is None:
            # Load all subjects
            dataset = load_dataset('cais/mmlu', 'all', cache_dir=cache_dir)
        else:
            dataset = load_dataset(
                'cais/mmlu',
                subject,
                cache_dir=cache_dir
            )

        examples = []
        for item in dataset[split]:
            examples.append({
                'question': item['question'],
                'choices': item['choices'],
                'answer': item['answer'],
                'subject': item.get('subject', subject)
            })

        return {
            'name': f'MMLU-{subject if subject else "all"}',
            'split': split,
            'num_examples': len(examples),
            'examples': examples
        }

    @staticmethod
    def load_hellaswag(
        split: str = 'validation',
        cache_dir: str = None
    ) -> Dict:
        """Load HellaSwag dataset."""
        dataset = load_dataset('hellaswag', cache_dir=cache_dir)

        examples = []
        for item in dataset[split]:
            examples.append({
                'context': item['ctx'],
                'endings': item['endings'],
                'correct_idx': int(item['label']),
                'activity_label': item.get('activity_label', '')
            })

        return {
            'name': 'HellaSwag',
            'split': split,
            'num_examples': len(examples),
            'examples': examples
        }

    @staticmethod
    def load_gsm8k(
        split: str = 'test',
        cache_dir: str = None
    ) -> Dict:
        """Load GSM8K dataset."""
        dataset = load_dataset('gsm8k', 'main', cache_dir=cache_dir)

        examples = []
        for item in dataset[split]:
            # Extract answer from the answer field
            answer_text = item['answer']
            # Answer is typically in format "#### NUMBER"
            import re
            match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', answer_text)
            if match:
                answer = float(match.group(1).replace(',', ''))
            else:
                answer = None

            examples.append({
                'question': item['question'],
                'answer': answer,
                'full_solution': answer_text
            })

        return {
            'name': 'GSM8K',
            'split': split,
            'num_examples': len(examples),
            'examples': examples
        }

    @staticmethod
    def load_humaneval(cache_dir: str = None) -> Dict:
        """Load HumanEval dataset."""
        dataset = load_dataset('openai_humaneval', cache_dir=cache_dir)

        examples = []
        for item in dataset['test']:
            examples.append({
                'task_id': item['task_id'],
                'prompt': item['prompt'],
                'test': item['test'],
                'entry_point': item['entry_point'],
                'canonical_solution': item.get('canonical_solution', '')
            })

        return {
            'name': 'HumanEval',
            'split': 'test',
            'num_examples': len(examples),
            'examples': examples
        }

    @staticmethod
    def stream_dataset(
        dataset_name: str,
        batch_size: int = 32
    ) -> Iterator[List[Dict]]:
        """
        Stream dataset in batches for memory efficiency.

        Useful for large datasets that don't fit in memory.
        """
        dataset = load_dataset(dataset_name, streaming=True)

        batch = []
        for item in dataset['test']:
            batch.append(item)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


# Example usage
def example_dataset_loading():
    """Example of loading benchmark datasets."""
    loader = BenchmarkDataLoader()

    # Load MMLU abstract algebra
    mmlu_data = loader.load_mmlu(subject='abstract_algebra', split='test')
    print(f"Loaded {mmlu_data['num_examples']} MMLU examples")

    # Load HellaSwag
    hellaswag_data = loader.load_hellaswag(split='validation')
    print(f"Loaded {hellaswag_data['num_examples']} HellaSwag examples")

    # Load GSM8K
    gsm8k_data = loader.load_gsm8k(split='test')
    print(f"Loaded {gsm8k_data['num_examples']} GSM8K examples")

    # Example: Access first question
    first_example = mmlu_data['examples'][0]
    print(f"\nExample question: {first_example['question']}")
    print(f"Choices: {first_example['choices']}")
    print(f"Answer: {first_example['answer']}")
```

### 32.8.2 Cost Estimation

```python
class EvaluationCostEstimator:
    """Estimate costs for running evaluations."""

    # Approximate costs per 1M tokens (as of 2024)
    COSTS_PER_MILLION_TOKENS = {
        'gpt-4': {'input': 30.0, 'output': 60.0},
        'gpt-3.5-turbo': {'input': 0.5, 'output': 1.5},
        'claude-3-opus': {'input': 15.0, 'output': 75.0},
        'claude-3-sonnet': {'input': 3.0, 'output': 15.0},
        'claude-3-haiku': {'input': 0.25, 'output': 1.25},
    }

    def __init__(self, model_name: str = 'gpt-3.5-turbo'):
        self.model_name = model_name
        self.costs = self.COSTS_PER_MILLION_TOKENS.get(
            model_name,
            {'input': 0.0, 'output': 0.0}
        )

    def estimate_benchmark_cost(
        self,
        benchmark_name: str,
        num_examples: int,
        avg_prompt_tokens: int,
        avg_completion_tokens: int,
        num_samples_per_example: int = 1
    ) -> Dict:
        """
        Estimate cost for running a benchmark.

        Args:
            benchmark_name: Name of benchmark
            num_examples: Number of examples to evaluate
            avg_prompt_tokens: Average tokens in prompt
            avg_completion_tokens: Average tokens in completion
            num_samples_per_example: Number of samples per example

        Returns:
            Cost breakdown
        """
        total_examples = num_examples * num_samples_per_example

        total_input_tokens = total_examples * avg_prompt_tokens
        total_output_tokens = total_examples * avg_completion_tokens

        input_cost = (total_input_tokens / 1_000_000) * self.costs['input']
        output_cost = (total_output_tokens / 1_000_000) * self.costs['output']
        total_cost = input_cost + output_cost

        return {
            'benchmark': benchmark_name,
            'model': self.model_name,
            'num_examples': num_examples,
            'samples_per_example': num_samples_per_example,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'input_cost_usd': input_cost,
            'output_cost_usd': output_cost,
            'total_cost_usd': total_cost,
            'cost_per_example_usd': total_cost / num_examples
        }

    def estimate_standard_suite(self) -> Dict:
        """Estimate cost for standard evaluation suite."""
        benchmarks = {
            'MMLU': {
                'num_examples': 14042,
                'avg_prompt_tokens': 500,  # 5-shot examples + question
                'avg_completion_tokens': 5,  # Just A/B/C/D
                'samples': 1
            },
            'HellaSwag': {
                'num_examples': 10042,
                'avg_prompt_tokens': 200,
                'avg_completion_tokens': 50,
                'samples': 1
            },
            'GSM8K': {
                'num_examples': 1319,
                'avg_prompt_tokens': 300,
                'avg_completion_tokens': 200,  # Chain-of-thought
                'samples': 1
            },
            'HumanEval': {
                'num_examples': 164,
                'avg_prompt_tokens': 150,
                'avg_completion_tokens': 150,
                'samples': 1  # or 10-100 for pass@k
            },
            'TruthfulQA': {
                'num_examples': 817,
                'avg_prompt_tokens': 150,
                'avg_completion_tokens': 100,
                'samples': 1
            }
        }

        results = {}
        total_cost = 0.0

        for name, params in benchmarks.items():
            cost = self.estimate_benchmark_cost(
                name,
                params['num_examples'],
                params['avg_prompt_tokens'],
                params['avg_completion_tokens'],
                params['samples']
            )
            results[name] = cost
            total_cost += cost['total_cost_usd']

        results['total_suite_cost_usd'] = total_cost

        return results

    def estimate_time(
        self,
        num_examples: int,
        avg_tokens_per_second: float = 50,
        avg_completion_tokens: int = 100,
        parallelism: int = 1
    ) -> Dict:
        """
        Estimate time to run evaluation.

        Args:
            num_examples: Number of examples
            avg_tokens_per_second: Model generation speed
            avg_completion_tokens: Average completion length
            parallelism: Number of parallel requests

        Returns:
            Time estimates
        """
        seconds_per_example = avg_completion_tokens / avg_tokens_per_second
        total_seconds = (num_examples * seconds_per_example) / parallelism

        return {
            'num_examples': num_examples,
            'tokens_per_second': avg_tokens_per_second,
            'parallelism': parallelism,
            'total_seconds': total_seconds,
            'total_minutes': total_seconds / 60,
            'total_hours': total_seconds / 3600
        }


# Example usage
def example_cost_estimation():
    """Example of cost estimation."""
    # For API-based model
    estimator = EvaluationCostEstimator(model_name='gpt-3.5-turbo')

    # Estimate MMLU cost
    mmlu_cost = estimator.estimate_benchmark_cost(
        'MMLU',
        num_examples=14042,
        avg_prompt_tokens=500,
        avg_completion_tokens=5
    )

    print(f"MMLU Evaluation Cost:")
    print(f"Total cost: ${mmlu_cost['total_cost_usd']:.2f}")
    print(f"Cost per example: ${mmlu_cost['cost_per_example_usd']:.4f}")

    # Estimate full suite
    suite_costs = estimator.estimate_standard_suite()
    print(f"\nFull evaluation suite cost: ${suite_costs['total_suite_cost_usd']:.2f}")

    for benchmark, cost in suite_costs.items():
        if benchmark != 'total_suite_cost_usd':
            print(f"{benchmark}: ${cost['total_cost_usd']:.2f}")

    # Estimate time for local model
    time_est = estimator.estimate_time(
        num_examples=14042,
        avg_tokens_per_second=50,
        avg_completion_tokens=5,
        parallelism=8
    )

    print(f"\nEstimated evaluation time:")
    print(f"{time_est['total_hours']:.2f} hours")
```

## 32.9 Comprehensive Evaluation Framework

```python
from dataclasses import dataclass
from typing import Any
import json

@dataclass
class EvaluationResult:
    """Store evaluation results."""
    benchmark_name: str
    score: float
    num_examples: int
    metadata: Dict[str, Any]

class ComprehensiveEvaluator:
    """Run comprehensive evaluation across multiple benchmarks."""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Initialize individual evaluators
        self.mmlu = MMLUEvaluator(model, tokenizer, device)
        self.hellaswag = HellaSwagEvaluator(model, tokenizer, device)
        self.gsm8k = GSM8KEvaluator(model, tokenizer, device)
        self.humaneval = HumanEvalEvaluator(model, tokenizer, device)
        self.truthful = TruthfulnessEvaluator(model, tokenizer, device)

    def run_full_evaluation(
        self,
        benchmarks: List[str] = None,
        save_path: str = None
    ) -> Dict[str, EvaluationResult]:
        """
        Run evaluation on specified benchmarks.

        Args:
            benchmarks: List of benchmark names to run
            save_path: Optional path to save results
        """
        if benchmarks is None:
            benchmarks = ['mmlu', 'hellaswag', 'gsm8k', 'humaneval', 'truthfulqa']

        results = {}

        for benchmark in benchmarks:
            print(f"\n{'='*50}")
            print(f"Evaluating on {benchmark.upper()}")
            print(f"{'='*50}\n")

            if benchmark == 'mmlu':
                result = self._run_mmlu()
            elif benchmark == 'hellaswag':
                result = self._run_hellaswag()
            elif benchmark == 'gsm8k':
                result = self._run_gsm8k()
            elif benchmark == 'humaneval':
                result = self._run_humaneval()
            elif benchmark == 'truthfulqa':
                result = self._run_truthfulqa()
            else:
                print(f"Unknown benchmark: {benchmark}")
                continue

            results[benchmark] = result
            print(f"{benchmark.upper()} Score: {result.score:.2%}")

        # Save results
        if save_path:
            self.save_results(results, save_path)

        # Print summary
        self.print_summary(results)

        return results

    def _run_mmlu(self) -> EvaluationResult:
        """Run MMLU evaluation."""
        # Load MMLU data (placeholder)
        # In practice, load from datasets library
        all_results = run_mmlu_evaluation()  # Using function from earlier

        overall_acc = sum(
            r['num_correct'] for r in all_results.values()
        ) / sum(
            r['num_total'] for r in all_results.values()
        )

        return EvaluationResult(
            benchmark_name='MMLU',
            score=overall_acc,
            num_examples=sum(r['num_total'] for r in all_results.values()),
            metadata=all_results
        )

    def _run_hellaswag(self) -> EvaluationResult:
        """Run HellaSwag evaluation."""
        # Placeholder
        return EvaluationResult(
            benchmark_name='HellaSwag',
            score=0.0,
            num_examples=0,
            metadata={}
        )

    def _run_gsm8k(self) -> EvaluationResult:
        """Run GSM8K evaluation."""
        # Placeholder
        return EvaluationResult(
            benchmark_name='GSM8K',
            score=0.0,
            num_examples=0,
            metadata={}
        )

    def _run_humaneval(self) -> EvaluationResult:
        """Run HumanEval evaluation."""
        # Placeholder
        return EvaluationResult(
            benchmark_name='HumanEval',
            score=0.0,
            num_examples=0,
            metadata={}
        )

    def _run_truthfulqa(self) -> EvaluationResult:
        """Run TruthfulQA evaluation."""
        # Placeholder
        return EvaluationResult(
            benchmark_name='TruthfulQA',
            score=0.0,
            num_examples=0,
            metadata={}
        )

    def save_results(self, results: Dict[str, EvaluationResult], path: str):
        """Save evaluation results to file."""
        serializable = {
            name: {
                'benchmark_name': result.benchmark_name,
                'score': result.score,
                'num_examples': result.num_examples,
                'metadata': result.metadata
            }
            for name, result in results.items()
        }

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\nResults saved to {path}")

    def print_summary(self, results: Dict[str, EvaluationResult]):
        """Print summary table of results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"{'Benchmark':<20} {'Score':<15} {'Num Examples':<15}")
        print("-"*60)

        for name, result in results.items():
            print(
                f"{result.benchmark_name:<20} "
                f"{result.score:>6.2%}        "
                f"{result.num_examples:>8}"
            )

        print("="*60)
```

## 32.10 Exercises

1. **Perplexity Analysis**: Implement a function that visualizes per-token perplexity for a given text. Identify which tokens the model finds most surprising.

2. **Custom Benchmark**: Create a small custom benchmark (10-20 questions) for a specific domain (e.g., your field of expertise). Evaluate a model on it.

3. **Contamination Study**: Take a well-known benchmark and check for contamination in a public dataset (e.g., C4, Wikipedia). Report your findings.

4. **Preference Collection**: Design a simple web interface for collecting human preferences between model outputs. What criteria should annotators use?

5. **LLM-as-Judge**: Implement an LLM-based judge for comparing model outputs. Compare its judgments to human judgments. Calculate agreement.

6. **Bias Detection**: Choose a demographic dimension (gender, race, age, etc.) and design a set of templates to test for bias. Evaluate a model.

7. **Multi-metric Evaluation**: For a specific task (e.g., summarization), implement evaluation using multiple metrics (ROUGE, BERTScore, human eval). How do they correlate?

8. **Dynamic Benchmark**: Create a system that generates novel math problems programmatically to avoid contamination issues.

9. **Statistical Significance**: Implement statistical significance testing (e.g., bootstrap resampling) to determine if the difference between two models is significant.

10. **Error Analysis**: Take a model's outputs on a benchmark and categorize the errors (reasoning errors, knowledge gaps, instruction following, etc.). What patterns emerge?

## Summary

Evaluating LLMs requires a multifaceted approach:

- **Language modeling metrics** (perplexity, BPB) measure basic modeling capability
- **Task benchmarks** (MMLU, HellaSwag, GSM8K, HumanEval) test specific capabilities
- **Reasoning benchmarks** (ARC, MATH, BigBench) evaluate complex reasoning
- **Safety evaluations** measure toxicity, bias, and truthfulness
- **Human evaluation** through platforms like Chatbot Arena provides real-world signal
- **Contamination detection** ensures benchmark validity

No single metric captures all aspects of model quality. Comprehensive evaluation requires combining automated benchmarks, human evaluation, and careful analysis of model behaviors. As models improve, benchmarks must evolve to remain challenging and meaningful.

## References

1. Hendrycks et al. (2021). "Measuring Massive Multitask Language Understanding (MMLU)". [https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)

2. Zellers et al. (2019). "HellaSwag: Can a Machine Really Finish Your Sentence?". [https://arxiv.org/abs/1905.07830](https://arxiv.org/abs/1905.07830)

3. Cobbe et al. (2021). "Training Verifiers to Solve Math Word Problems (GSM8K)". [https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)

4. Chen et al. (2021). "Evaluating Large Language Models Trained on Code (HumanEval)". [https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)

5. Rein et al. (2023). "GPQA: A Graduate-Level Google-Proof Q&A Benchmark". [https://arxiv.org/abs/2311.12022](https://arxiv.org/abs/2311.12022)

6. Zhou et al. (2023). "Instruction-Following Evaluation for Large Language Models". [https://arxiv.org/abs/2311.07911](https://arxiv.org/abs/2311.07911)

7. OpenAI (2024). "Introducing SimpleQA". [https://openai.com/index/introducing-simpleqa/](https://openai.com/index/introducing-simpleqa/)

8. Clark et al. (2018). "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge". [https://arxiv.org/abs/1803.05457](https://arxiv.org/abs/1803.05457)

9. Hendrycks et al. (2021). "Measuring Mathematical Problem Solving With the MATH Dataset". [https://arxiv.org/abs/2103.03874](https://arxiv.org/abs/2103.03874)

10. Srivastava et al. (2022). "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models". [https://arxiv.org/abs/2206.04615](https://arxiv.org/abs/2206.04615)

11. Gehman et al. (2020). "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models". [https://arxiv.org/abs/2009.11462](https://arxiv.org/abs/2009.11462)

12. Nadeem et al. (2020). "StereoSet: Measuring stereotypical bias in pretrained language models". [https://arxiv.org/abs/2004.09456](https://arxiv.org/abs/2004.09456)

13. Lin et al. (2021). "TruthfulQA: Measuring How Models Mimic Human Falsehoods". [https://arxiv.org/abs/2109.07958](https://arxiv.org/abs/2109.07958)

14. Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena". [https://arxiv.org/abs/2403.04132](https://arxiv.org/abs/2403.04132)
