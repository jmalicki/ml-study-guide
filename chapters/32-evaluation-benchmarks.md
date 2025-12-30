# Chapter 32: Evaluation and Benchmarks

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

## 32.3 Reasoning Benchmarks

### 32.3.1 ARC (AI2 Reasoning Challenge)

ARC ([Clark et al., 2018](https://arxiv.org/abs/1803.05457)) contains science questions requiring reasoning. It has two sets: Easy and Challenge.

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

The MATH dataset ([Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874)) contains challenging competition mathematics problems. Related to [Reasoning and Chain-of-Thought](28-reasoning.md).

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

Safety evaluations are critical for deployed models. See [Safety and Alignment Techniques](22-safety-alignment.md) for training methods.

### 32.4.1 Toxicity Detection

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

## 32.7 Comprehensive Evaluation Framework

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

## 32.8 Exercises

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

5. Clark et al. (2018). "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge". [https://arxiv.org/abs/1803.05457](https://arxiv.org/abs/1803.05457)

6. Hendrycks et al. (2021). "Measuring Mathematical Problem Solving With the MATH Dataset". [https://arxiv.org/abs/2103.03874](https://arxiv.org/abs/2103.03874)

7. Srivastava et al. (2022). "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models". [https://arxiv.org/abs/2206.04615](https://arxiv.org/abs/2206.04615)

8. Gehman et al. (2020). "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models". [https://arxiv.org/abs/2009.11462](https://arxiv.org/abs/2009.11462)

9. Nadeem et al. (2020). "StereoSet: Measuring stereotypical bias in pretrained language models". [https://arxiv.org/abs/2004.09456](https://arxiv.org/abs/2004.09456)

10. Lin et al. (2021). "TruthfulQA: Measuring How Models Mimic Human Falsehoods". [https://arxiv.org/abs/2109.07958](https://arxiv.org/abs/2109.07958)

11. Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena". [https://arxiv.org/abs/2403.04132](https://arxiv.org/abs/2403.04132)
