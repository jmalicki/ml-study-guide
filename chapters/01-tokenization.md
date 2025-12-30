# Chapter 1: Tokenization

## Introduction

Tokenization is the first and foundational step in any natural language processing pipeline, especially for Large Language Models (LLMs). It converts raw text into discrete units (tokens) that can be processed by neural networks. The choice of tokenization strategy significantly impacts model performance, vocabulary size, training efficiency, and the ability to handle rare words and out-of-vocabulary (OOV) terms.

In this chapter, we'll explore various tokenization approaches, implement them from scratch using PyTorch, and understand their mathematical foundations and practical trade-offs.

**Learning Objectives:**
- Understand different tokenization strategies and their trade-offs
- Implement character-level, word-level, and subword tokenizers from scratch
- Learn Byte Pair Encoding (BPE), WordPiece, and SentencePiece algorithms
- Build practical tokenizers for modern LLM applications

## 1.1 Why Tokenization Matters

Before diving into implementations, let's understand why tokenization is critical:

1. **Vocabulary Size**: Determines model parameters and computational cost
2. **Semantic Representation**: Affects how well the model captures meaning
3. **Rare Words**: Impacts handling of infrequent terms and morphological variations
4. **Multilingual Support**: Critical for cross-lingual applications
5. **Compression**: Influences sequence length and memory requirements

The fundamental tension in tokenization is between **granularity** and **vocabulary size**:
- Fine-grained (character-level): Small vocabulary, long sequences
- Coarse-grained (word-level): Large vocabulary, short sequences
- Middle ground (subword): Balanced approach used by modern LLMs

## 1.2 Character-Level Tokenization

Character-level tokenization splits text into individual characters. It has the smallest possible vocabulary size but produces the longest sequences.

### 1.2.1 Mathematical Formulation

Given a text string $T$ of length $n$ over an alphabet $\Sigma$:

$$T = c_1c_2...c_n, \quad c_i \in \Sigma$$

The tokenization function $\tau: T \rightarrow \mathbb{Z}^n$ maps each character to an integer ID:

$$\tau(T) = [id(c_1), id(c_2), ..., id(c_n)]$$

where $id: \Sigma \rightarrow \{0, 1, ..., |\Sigma|-1\}$ is a bijective mapping.

### 1.2.2 Advantages and Disadvantages

**Advantages:**
- Very small vocabulary size (typically < 256 for ASCII, < 150K for Unicode)
- No OOV words - can handle any character
- Useful for tasks like spelling correction or morphological analysis

**Disadvantages:**
- Long sequences increase computational cost ($O(n^2)$ for attention mechanisms)
- Difficult to capture long-range semantic dependencies
- Each token carries less semantic information

### 1.2.3 Implementation

```python
import torch
from typing import List, Dict, Tuple
from collections import Counter

class CharTokenizer:
    """Character-level tokenizer implementation."""

    def __init__(self, special_tokens: List[str] = None):
        """
        Initialize character tokenizer.

        Args:
            special_tokens: List of special tokens like [PAD], [UNK], [BOS], [EOS]
        """
        self.special_tokens = special_tokens or ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size = 0

        # Initialize with special tokens
        for token in self.special_tokens:
            self._add_char(token)

    def _add_char(self, char: str) -> int:
        """Add a character to the vocabulary."""
        if char not in self.char_to_id:
            char_id = len(self.char_to_id)
            self.char_to_id[char] = char_id
            self.id_to_char[char_id] = char
            self.vocab_size = len(self.char_to_id)
        return self.char_to_id[char]

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of training texts
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        # Add all unique characters to vocabulary
        for char in sorted(unique_chars):
            self._add_char(char)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_special_tokens: Whether to add [BOS] and [EOS] tokens

        Returns:
            List of token IDs
        """
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.char_to_id['[BOS]'])

        for char in text:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                token_ids.append(self.char_to_id['[UNK]'])

        if add_special_tokens:
            token_ids.append(self.char_to_id['[EOS]'])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        chars = []
        for token_id in token_ids:
            char = self.id_to_char.get(token_id, '[UNK]')
            if skip_special_tokens and char in self.special_tokens:
                continue
            chars.append(char)
        return ''.join(chars)

    def encode_batch(self, texts: List[str],
                     max_length: int = None,
                     padding: bool = True) -> torch.Tensor:
        """
        Encode a batch of texts to padded tensor.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length (truncate if longer)
            padding: Whether to pad sequences to same length

        Returns:
            Tensor of shape (batch_size, seq_length)
        """
        encoded = [self.encode(text) for text in texts]

        if max_length is not None:
            encoded = [seq[:max_length] for seq in encoded]

        if padding:
            max_len = max(len(seq) for seq in encoded)
            pad_id = self.char_to_id['[PAD]']
            padded = [seq + [pad_id] * (max_len - len(seq)) for seq in encoded]
            return torch.tensor(padded, dtype=torch.long)
        else:
            return encoded


# Example Usage
if __name__ == "__main__":
    # Sample texts
    texts = [
        "Hello, world!",
        "Tokenization is fundamental to NLP.",
        "Characters are the building blocks."
    ]

    # Initialize and build vocabulary
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample characters: {list(tokenizer.char_to_id.keys())[:20]}")

    # Encode a text
    sample_text = "Hello!"
    encoded = tokenizer.encode(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")

    # Batch encoding
    batch_tensor = tokenizer.encode_batch(texts)
    print(f"\nBatch shape: {batch_tensor.shape}")
    print(f"Batch tensor:\n{batch_tensor}")
```

## 1.3 Word-Level Tokenization

Word-level tokenization splits text into words, typically using whitespace and punctuation as delimiters. This was the dominant approach before the rise of subword methods.

### 1.3.1 Mathematical Formulation

Given text $T$, we split it into words $w_1, w_2, ..., w_m$ where $m \leq n$ and each word $w_i$ is a sequence of characters:

$$T = w_1 \, w_2 \, ... \, w_m$$

The vocabulary $V$ is the set of all unique words in the training corpus:

$$V = \{w_1, w_2, ..., w_{|V|}\}$$

Tokenization maps each word to its index in $V$:

$$\tau(w_i) = j \quad \text{where} \quad w_i = V[j]$$

### 1.3.2 Advantages and Disadvantages

**Advantages:**
- Tokens carry semantic meaning
- Shorter sequences than character-level
- Intuitive and interpretable

**Disadvantages:**
- Large vocabulary size (often 50K-500K words)
- OOV problem for rare or unseen words
- Doesn't handle morphological variations well
- Language-dependent (challenges with Chinese, Japanese, etc.)

### 1.3.3 Implementation

```python
import re
from typing import List, Dict, Optional
import torch

class WordTokenizer:
    """Word-level tokenizer with frequency-based vocabulary."""

    def __init__(self,
                 vocab_size: int = 10000,
                 special_tokens: List[str] = None,
                 lowercase: bool = True):
        """
        Initialize word tokenizer.

        Args:
            vocab_size: Maximum vocabulary size
            special_tokens: Special tokens to include
            lowercase: Whether to lowercase all text
        """
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.special_tokens = special_tokens or ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()

        # Initialize special tokens
        for token in self.special_tokens:
            self._add_word(token)

    def _add_word(self, word: str) -> int:
        """Add a word to vocabulary."""
        if word not in self.word_to_id:
            word_id = len(self.word_to_id)
            self.word_to_id[word] = word_id
            self.id_to_word[word_id] = word
        return self.word_to_id[word]

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Split text into words using regex.

        This is a simple tokenizer. Production systems use more sophisticated
        approaches like spaCy or NLTK.
        """
        if self.lowercase:
            text = text.lower()

        # Split on whitespace and punctuation, but keep punctuation
        words = re.findall(r'\w+|[^\w\s]', text)
        return words

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts, keeping most frequent words.

        Args:
            texts: List of training texts
        """
        # Count word frequencies
        for text in texts:
            words = self._tokenize_text(text)
            self.word_freq.update(words)

        # Get most common words (excluding special tokens count)
        num_special = len(self.special_tokens)
        most_common = self.word_freq.most_common(self.vocab_size - num_special)

        # Add to vocabulary
        for word, freq in most_common:
            self._add_word(word)

        print(f"Built vocabulary with {len(self.word_to_id)} words")
        print(f"Corpus vocabulary: {len(self.word_freq)} unique words")
        if len(self.word_freq) > self.vocab_size:
            print(f"Truncated to top {self.vocab_size} most frequent words")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        words = self._tokenize_text(text)
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.word_to_id['[BOS]'])

        unk_id = self.word_to_id['[UNK]']
        for word in words:
            token_ids.append(self.word_to_id.get(word, unk_id))

        if add_special_tokens:
            token_ids.append(self.word_to_id['[EOS]'])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        words = []
        for token_id in token_ids:
            word = self.id_to_word.get(token_id, '[UNK]')
            if skip_special_tokens and word in self.special_tokens:
                continue
            words.append(word)

        # Simple detokenization: join with spaces, but handle punctuation
        text = ' '.join(words)
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text

    def encode_batch(self, texts: List[str],
                     max_length: int = None,
                     padding: bool = True) -> torch.Tensor:
        """Encode batch of texts to padded tensor."""
        encoded = [self.encode(text) for text in texts]

        if max_length is not None:
            encoded = [seq[:max_length] for seq in encoded]

        if padding:
            max_len = max(len(seq) for seq in encoded)
            pad_id = self.word_to_id['[PAD]']
            padded = [seq + [pad_id] * (max_len - len(seq)) for seq in encoded]
            return torch.tensor(padded, dtype=torch.long)
        else:
            return encoded


# Example Usage
if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fascinating!",
        "Tokenization is the first step in NLP pipelines.",
        "The tokenizer converts text into tokens."
    ]

    tokenizer = WordTokenizer(vocab_size=50, lowercase=True)
    tokenizer.build_vocab(texts)

    print(f"\nVocabulary size: {len(tokenizer.word_to_id)}")
    print(f"Most common words: {tokenizer.word_freq.most_common(10)}")

    sample = "The tokenizer is working!"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print(f"\nOriginal: {sample}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Demonstrate OOV handling
    oov_text = "Supercalifragilisticexpialidocious!"
    encoded_oov = tokenizer.encode(oov_text)
    print(f"\nOOV text: {oov_text}")
    print(f"Encoded (note [UNK] tokens): {encoded_oov}")
```

## 1.4 Subword Tokenization

Subword tokenization represents the sweet spot between character and word-level approaches. It splits rare words into meaningful subword units while keeping common words intact. This is the dominant approach in modern LLMs.

**Key Insight**: Frequent words remain whole, rare words are split into subwords, and all words can be represented without OOV issues.

### 1.4.1 Byte Pair Encoding (BPE)

BPE, originally a data compression algorithm, was adapted for NLP by Sennrich et al. (2016) in ["Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909). It's used in GPT-2, GPT-3, RoBERTa, and many other models.

#### Algorithm

BPE iteratively merges the most frequent pair of consecutive tokens:

**Training Algorithm:**
1. Initialize vocabulary with all characters in the corpus
2. Repeat for $k$ iterations (or until desired vocab size):
   - Count all adjacent token pairs in the corpus
   - Find the most frequent pair $(a, b)$
   - Merge all occurrences of $(a, b)$ into a new token $ab$
   - Add $ab$ to the vocabulary

**Encoding Algorithm:**
1. Start with character-level tokenization
2. Iteratively apply learned merge rules until no more merges possible

#### Mathematical Formulation

Let $V_0$ be the initial character vocabulary. At each step $i$:

$$V_{i+1} = V_i \cup \{a \circ b\}$$

where $(a, b) = \arg\max_{(x,y) \in V_i \times V_i} \text{count}(xy)$ and $\circ$ denotes concatenation.

The final vocabulary after $k$ merges:

$$V_k = V_0 \cup \{m_1, m_2, ..., m_k\}$$

where $m_i$ is the $i$-th merge operation.

#### Implementation

```python
import re
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import torch

class BPETokenizer:
    """Byte Pair Encoding tokenizer implementation."""

    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']

        # Initialize special tokens
        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        """Add token to vocabulary."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def _get_stats(self, words: Dict[str, int]) -> Counter:
        """
        Count frequency of adjacent token pairs.

        Args:
            words: Dictionary mapping word (as space-separated tokens) to frequency

        Returns:
            Counter of token pair frequencies
        """
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair: Tuple[str, str], words: Dict[str, int]) -> Dict[str, int]:
        """
        Merge all occurrences of a token pair in the vocabulary.

        Args:
            pair: Tuple of (token1, token2) to merge
            words: Current word vocabulary

        Returns:
            Updated word vocabulary with merged pairs
        """
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = words[word]

        return new_words

    def train(self, texts: List[str], verbose: bool = False) -> None:
        """
        Train BPE tokenizer on texts.

        Args:
            texts: List of training texts
            verbose: Whether to print training progress
        """
        # Preprocess: simple tokenization and character splitting
        word_freqs = Counter()
        for text in texts:
            words = text.lower().split()
            word_freqs.update(words)

        # Convert words to character sequences with space separators
        # e.g., "hello" -> "h e l l o"
        words = {' '.join(word): freq for word, freq in word_freqs.items()}

        # Add all individual characters to vocabulary
        chars = set()
        for word in words.keys():
            chars.update(word.split())

        for char in sorted(chars):
            self._add_token(char)

        # Perform BPE merges
        num_merges = self.vocab_size - len(self.token_to_id)

        for i in range(num_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break

            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)

            if verbose and i % 100 == 0:
                print(f"Merge {i}: {best_pair} (freq: {pairs[best_pair]})")

            # Merge the pair
            words = self._merge_pair(best_pair, words)

            # Record merge operation
            self.merges.append(best_pair)

            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            self._add_token(merged_token)

        print(f"Training complete. Vocabulary size: {len(self.token_to_id)}")
        print(f"Number of merges: {len(self.merges)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned BPE merges.

        Args:
            word: Word to tokenize

        Returns:
            List of subword tokens
        """
        # Start with character-level tokenization
        tokens = list(word)

        # Apply each merge operation in order
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == pair:
                    # Merge the pair
                    tokens = tokens[:i] + [''.join(pair)] + tokens[i+2:]
                else:
                    i += 1

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add [BOS] and [EOS]

        Returns:
            List of token IDs
        """
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.token_to_id['[BOS]'])

        # Simple word splitting
        words = text.lower().split()
        unk_id = self.token_to_id['[UNK]']

        for word in words:
            # Tokenize word into subwords
            subwords = self._tokenize_word(word)

            for subword in subwords:
                token_ids.append(self.token_to_id.get(subword, unk_id))

        if add_special_tokens:
            token_ids.append(self.token_to_id['[EOS]'])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '[UNK]')
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)

        # Join tokens and add spaces between words (heuristic)
        # In practice, you'd want more sophisticated detokenization
        text = ''.join(tokens)
        # Add space after common word endings
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        return text

    def encode_batch(self, texts: List[str],
                     max_length: int = None,
                     padding: bool = True) -> torch.Tensor:
        """Encode batch of texts."""
        encoded = [self.encode(text) for text in texts]

        if max_length is not None:
            encoded = [seq[:max_length] for seq in encoded]

        if padding:
            max_len = max(len(seq) for seq in encoded)
            pad_id = self.token_to_id['[PAD]']
            padded = [seq + [pad_id] * (max_len - len(seq)) for seq in encoded]
            return torch.tensor(padded, dtype=torch.long)
        else:
            return encoded


# Example Usage
if __name__ == "__main__":
    # Training corpus
    corpus = [
        "low lower lowest",
        "new newer newest",
        "wide wider widest",
        "the quick brown fox",
        "the lazy dog jumped",
        "tokenization is important for language models"
    ]

    # Train BPE
    bpe = BPETokenizer(vocab_size=200)
    bpe.train(corpus, verbose=True)

    # Show some learned merges
    print(f"\nFirst 10 merges: {bpe.merges[:10]}")
    print(f"\nVocabulary sample: {list(bpe.token_to_id.keys())[:30]}")

    # Test encoding
    test_text = "the lowest"
    encoded = bpe.encode(test_text)
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Tokens: {[bpe.id_to_token[i] for i in encoded]}")
    print(f"Decoded: {bpe.decode(encoded)}")
```

### 1.4.2 WordPiece

WordPiece, developed by Google and used in BERT, is similar to BPE but uses a different merge criterion. Instead of frequency, it chooses merges that maximize the likelihood of the training data.

#### Algorithm Difference

While BPE uses frequency:
$$\text{score}_{\text{BPE}}(a, b) = \text{count}(ab)$$

WordPiece uses likelihood:
$$\text{score}_{\text{WordPiece}}(a, b) = \frac{P(ab)}{P(a)P(b)} = \frac{\text{count}(ab)}{\text{count}(a) \cdot \text{count}(b)}$$

This scores pairs by how much their joint probability exceeds their independent probabilities (mutual information).

#### Implementation

```python
import math
from typing import List, Dict, Tuple
from collections import Counter
import torch

class WordPieceTokenizer:
    """WordPiece tokenizer implementation (used in BERT)."""

    def __init__(self, vocab_size: int = 1000, prefix: str = "##"):
        """
        Initialize WordPiece tokenizer.

        Args:
            vocab_size: Target vocabulary size
            prefix: Prefix for continuation tokens (e.g., "##" in BERT)
        """
        self.vocab_size = vocab_size
        self.prefix = prefix
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

        # Initialize special tokens
        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        """Add token to vocabulary."""
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    def _get_pair_scores(self, words: Dict[str, int]) -> Dict[Tuple[str, str], float]:
        """
        Calculate scores for all token pairs using mutual information.

        Args:
            words: Dictionary of words (as token sequences) to frequencies

        Returns:
            Dictionary mapping pairs to their scores
        """
        pair_counts = Counter()
        token_counts = Counter()

        # Count pairs and individual tokens
        for word, freq in words.items():
            tokens = word.split()
            for i in range(len(tokens)):
                token_counts[tokens[i]] += freq
                if i < len(tokens) - 1:
                    pair_counts[(tokens[i], tokens[i+1])] += freq

        # Calculate mutual information score
        scores = {}
        for pair, pair_freq in pair_counts.items():
            token1_freq = token_counts[pair[0]]
            token2_freq = token_counts[pair[1]]

            # Score = P(pair) / (P(token1) * P(token2))
            # Using counts as proxy for probabilities
            score = pair_freq / (token1_freq * token2_freq)
            scores[pair] = score

        return scores

    def _merge_pair(self, pair: Tuple[str, str], words: Dict[str, int]) -> Dict[str, int]:
        """Merge a token pair in the vocabulary."""
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = words[word]

        return new_words

    def train(self, texts: List[str], verbose: bool = False) -> None:
        """
        Train WordPiece tokenizer on texts.

        Args:
            texts: List of training texts
            verbose: Whether to print training progress
        """
        # Build initial word vocabulary
        word_freqs = Counter()
        for text in texts:
            words = text.lower().split()
            word_freqs.update(words)

        # Initialize with character-level tokens
        # First character has no prefix, subsequent chars have prefix
        words = {}
        for word, freq in word_freqs.items():
            if len(word) > 0:
                # First char without prefix, rest with prefix
                tokens = [word[0]] + [self.prefix + c for c in word[1:]]
                words[' '.join(tokens)] = freq

        # Collect all unique base tokens
        chars = set()
        for word in words.keys():
            chars.update(word.split())

        for char in sorted(chars):
            self._add_token(char)

        # Perform merges
        num_merges = self.vocab_size - len(self.token_to_id)

        for i in range(num_merges):
            scores = self._get_pair_scores(words)
            if not scores:
                break

            # Get best pair by score
            best_pair = max(scores, key=scores.get)

            if verbose and i % 100 == 0:
                print(f"Merge {i}: {best_pair} (score: {scores[best_pair]:.6f})")

            # Merge the pair
            words = self._merge_pair(best_pair, words)

            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            self._add_token(merged_token)

        print(f"Training complete. Vocabulary size: {len(self.token_to_id)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a word using greedy longest-match-first algorithm.

        This is the standard WordPiece encoding algorithm.
        """
        if not word:
            return []

        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            found = False

            # Try to find longest matching subword
            while start < end:
                substr = word[start:end]
                # Add prefix if not at word start
                if start > 0:
                    substr = self.prefix + substr

                if substr in self.token_to_id:
                    tokens.append(substr)
                    found = True
                    break

                end -= 1

            if not found:
                # Unknown character, use [UNK]
                tokens.append('[UNK]')
                start += 1
            else:
                start = end

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add [CLS] and [SEP]

        Returns:
            List of token IDs
        """
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.token_to_id['[CLS]'])

        words = text.lower().split()

        for word in words:
            subwords = self._tokenize_word(word)
            for subword in subwords:
                token_ids.append(self.token_to_id.get(subword,
                                                      self.token_to_id['[UNK]']))

        if add_special_tokens:
            token_ids.append(self.token_to_id['[SEP]'])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '[UNK]')
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)

        # Remove prefix markers and join
        text = ''.join(tokens).replace(self.prefix, '')

        return text

    def encode_batch(self, texts: List[str],
                     max_length: int = None,
                     padding: bool = True) -> torch.Tensor:
        """Encode batch of texts."""
        encoded = [self.encode(text) for text in texts]

        if max_length is not None:
            encoded = [seq[:max_length] for seq in encoded]

        if padding:
            max_len = max(len(seq) for seq in encoded)
            pad_id = self.token_to_id['[PAD]']
            padded = [seq + [pad_id] * (max_len - len(seq)) for seq in encoded]
            return torch.tensor(padded, dtype=torch.long)
        else:
            return encoded


# Example Usage
if __name__ == "__main__":
    corpus = [
        "playing played player",
        "running runner run",
        "tokenization tokenizer tokenize",
        "the model is training on data"
    ]

    # Train WordPiece
    wp = WordPieceTokenizer(vocab_size=150, prefix="##")
    wp.train(corpus, verbose=True)

    # Test encoding
    test_text = "the player is running"
    encoded = wp.encode(test_text)

    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Tokens: {[wp.id_to_token[i] for i in encoded]}")
    print(f"Decoded: {wp.decode(encoded)}")

    # Show vocabulary sample
    print(f"\nVocabulary size: {len(wp.token_to_id)}")
    print(f"Sample tokens: {list(wp.token_to_id.keys())[:30]}")
```

### 1.4.3 SentencePiece

SentencePiece, developed by Google, is a language-independent tokenizer that treats the input as a raw stream of Unicode characters. Unlike BPE and WordPiece which require pre-tokenization, SentencePiece works directly on raw text.

**Key Features:**
- Language-agnostic (no pre-tokenization required)
- Treats whitespace as a normal character (using ▁ symbol)
- Supports both BPE and Unigram Language Model algorithms
- Used in T5, ALBERT, XLNet, and many multilingual models

**Unigram Language Model**: Unlike BPE (which starts small and grows), the unigram model starts with a large vocabulary and iteratively removes tokens that minimize the loss in likelihood.

The objective is to find the tokenization that maximizes:

$$P(x) = \prod_{s \in S(x)} P(s)$$

where $S(x)$ is the set of all possible segmentations of input $x$, and we choose:

$$x^* = \arg\max_{x \in S(x)} P(x)$$

Since we don't implement SentencePiece from scratch here (it's quite complex), we show how to use the library:

```python
# Installation: pip install sentencepiece

import sentencepiece as spm
import torch
from typing import List

class SentencePieceTokenizer:
    """Wrapper around SentencePiece for consistent interface."""

    def __init__(self, model_path: str = None):
        """
        Initialize SentencePiece tokenizer.

        Args:
            model_path: Path to trained SentencePiece model
        """
        self.model_path = model_path
        self.sp = None
        if model_path:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)

    def train(self,
              input_file: str,
              model_prefix: str,
              vocab_size: int = 8000,
              model_type: str = 'bpe',
              character_coverage: float = 0.9995,
              special_tokens: List[str] = None):
        """
        Train SentencePiece model.

        Args:
            input_file: Path to training text file
            model_prefix: Prefix for output model files
            vocab_size: Target vocabulary size
            model_type: 'bpe' or 'unigram'
            character_coverage: Character coverage (1.0 for langs with small charset)
            special_tokens: Additional special tokens
        """
        if special_tokens is None:
            special_tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', '[MASK]']

        # Build training command
        train_args = (
            f'--input={input_file} '
            f'--model_prefix={model_prefix} '
            f'--vocab_size={vocab_size} '
            f'--model_type={model_type} '
            f'--character_coverage={character_coverage} '
            f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
            f'--user_defined_symbols={",".join(special_tokens[4:])}'
        )

        spm.SentencePieceTrainer.train(train_args)

        # Load trained model
        self.model_path = f'{model_prefix}.model'
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)

        print(f"Trained SentencePiece model: {self.model_path}")
        print(f"Vocabulary size: {self.sp.vocab_size()}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if not self.sp:
            raise ValueError("Model not loaded. Train or load a model first.")

        if add_special_tokens:
            # Add BOS and EOS
            ids = [self.sp.bos_id()] + self.sp.encode(text) + [self.sp.eos_id()]
        else:
            ids = self.sp.encode(text)

        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if not self.sp:
            raise ValueError("Model not loaded. Train or load a model first.")

        if skip_special_tokens:
            # Filter special tokens
            special_ids = {self.sp.pad_id(), self.sp.unk_id(),
                          self.sp.bos_id(), self.sp.eos_id()}
            token_ids = [id for id in token_ids if id not in special_ids]

        return self.sp.decode(token_ids)

    def encode_batch(self, texts: List[str],
                     max_length: int = None,
                     padding: bool = True) -> torch.Tensor:
        """Encode batch of texts."""
        encoded = [self.encode(text) for text in texts]

        if max_length is not None:
            encoded = [seq[:max_length] for seq in encoded]

        if padding:
            max_len = max(len(seq) for seq in encoded)
            pad_id = self.sp.pad_id()
            padded = [seq + [pad_id] * (max_len - len(seq)) for seq in encoded]
            return torch.tensor(padded, dtype=torch.long)
        else:
            return encoded

    def get_vocab(self) -> Dict[int, str]:
        """Get vocabulary mapping."""
        if not self.sp:
            raise ValueError("Model not loaded.")

        return {i: self.sp.id_to_piece(i) for i in range(self.sp.vocab_size())}


# Example Usage (requires training file)
if __name__ == "__main__":
    # Create sample training file
    with open('/tmp/train.txt', 'w', encoding='utf-8') as f:
        f.write("The quick brown fox jumps over the lazy dog.\n")
        f.write("Natural language processing with transformers.\n")
        f.write("Tokenization is the foundation of NLP.\n")
        f.write("SentencePiece is language agnostic.\n")

    # Train tokenizer
    sp_tokenizer = SentencePieceTokenizer()
    sp_tokenizer.train(
        input_file='/tmp/train.txt',
        model_prefix='/tmp/sp_model',
        vocab_size=100,
        model_type='bpe'
    )

    # Test encoding
    test_text = "Tokenization with SentencePiece"
    encoded = sp_tokenizer.encode(test_text)
    decoded = sp_tokenizer.decode(encoded)

    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Show vocabulary
    vocab = sp_tokenizer.get_vocab()
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Sample tokens: {[(i, vocab[i]) for i in range(20)]}")
```

## 1.5 Comparison of Tokenization Methods

| Method | Vocab Size | Seq Length | OOV Handling | Pros | Cons | Used In |
|--------|-----------|------------|--------------|------|------|---------|
| **Character** | ~256-150K | Long | Perfect | Simple, no OOV | Long sequences, weak semantics | Some RNNs, character-level models |
| **Word** | 50K-500K | Short | Poor | Semantic units, short | Large vocab, OOV problem | Early NLP, simple tasks |
| **BPE** | 10K-50K | Medium | Good | Balance, popular | Frequency-based, no linguistic | GPT-2/3, RoBERTa, BART |
| **WordPiece** | 10K-50K | Medium | Good | Likelihood-based | Requires pre-tokenization | BERT, DistilBERT, ELECTRA |
| **SentencePiece** | 8K-32K | Medium | Perfect | Language-agnostic, clean | Complex implementation | T5, ALBERT, XLNet, mT5 |

### Vocabulary Size Impact

The relationship between vocabulary size and sequence length for a corpus of size $N$ tokens:

$$\text{Sequence Length} \approx \frac{N}{\text{Avg Tokens per Word} \times \text{Compression Ratio}}$$

For a fixed corpus:
- Character-level: $V \approx 256$, sequences ~10x longer than words
- Word-level: $V \approx 50K$, shortest sequences but OOV issues
- Subword: $V \approx 32K$, sequences ~1.5-2x longer than words, no OOV

## 1.6 Modern Tokenizers: Practical Usage

In practice, you'll often use existing tokenizer libraries rather than implementing from scratch. Here's how to use the most common ones:

```python
# Using Hugging Face Transformers
from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    BertTokenizer,
    T5Tokenizer
)
import torch

# GPT-2 (BPE)
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
text = "Tokenization is fundamental to NLP!"
encoded = gpt2_tokenizer(text, return_tensors='pt')
print(f"GPT-2 encoding: {encoded['input_ids']}")
print(f"Tokens: {gpt2_tokenizer.tokenize(text)}")

# BERT (WordPiece)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoded = bert_tokenizer(text, return_tensors='pt')
print(f"\nBERT encoding: {encoded['input_ids']}")
print(f"Tokens: {bert_tokenizer.tokenize(text)}")

# T5 (SentencePiece)
t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
encoded = t5_tokenizer(text, return_tensors='pt')
print(f"\nT5 encoding: {encoded['input_ids']}")
print(f"Tokens: {t5_tokenizer.tokenize(text)}")

# Batch processing with padding and attention masks
texts = [
    "Short text",
    "This is a much longer text that will require padding in the batch"
]

batch_encoded = gpt2_tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=20,
    return_tensors='pt'
)

print(f"\nBatch input_ids shape: {batch_encoded['input_ids'].shape}")
print(f"Batch attention_mask:\n{batch_encoded['attention_mask']}")
```

## 1.7 Advanced Topics

### 1.7.1 Byte-Level BPE

Used in GPT-2 and later models, Byte-Level BPE operates on bytes rather than characters, ensuring that any text can be encoded without UNK tokens.

**Key insight**: Map all 256 possible byte values to a base vocabulary, then apply BPE on top.

```python
# GPT-2 uses byte-level BPE
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Can handle any Unicode, emojis, etc.
text = "Hello 世界 🌍!"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
print(f"IDs: {tokenizer.encode(text)}")

# The tokenizer can handle absolutely any byte sequence
binary_text = "\\x00\\xff\\xfe"
print(f"Binary tokens: {tokenizer.tokenize(binary_text)}")
```

### 1.7.2 Tokenization and Model Performance

Tokenization choices affect:

1. **Training Speed**: Vocabulary size impacts embedding layer size
   $$\text{Embedding Params} = V \times d_{model}$$
   where $V$ is vocab size and $d_{model}$ is embedding dimension.

2. **Inference Speed**: Longer sequences → more compute
   $$\text{Attention Cost} = O(n^2 d)$$
   where $n$ is sequence length.

3. **Generalization**: Subword tokenization improves handling of rare words through compositional understanding.

### 1.7.3 Handling Special Tokens

Special tokens serve various purposes:

```python
class TokenizerWithSpecialTokens:
    """Demonstrates special token handling."""

    def __init__(self):
        self.special_tokens = {
            '[PAD]': 0,   # Padding for batch processing
            '[UNK]': 1,   # Unknown tokens
            '[BOS]': 2,   # Beginning of sequence
            '[EOS]': 3,   # End of sequence
            '[CLS]': 4,   # Classification token (BERT)
            '[SEP]': 5,   # Separator token (BERT)
            '[MASK]': 6,  # Masking token (BERT MLM)
        }

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask (1 for real tokens, 0 for padding).

        Args:
            input_ids: Tensor of shape (batch_size, seq_length)

        Returns:
            Attention mask of same shape
        """
        pad_token_id = self.special_tokens['[PAD]']
        return (input_ids != pad_token_id).long()

    def create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create position IDs for positional encoding.

        Args:
            input_ids: Tensor of shape (batch_size, seq_length)

        Returns:
            Position IDs of same shape
        """
        attention_mask = self.create_attention_mask(input_ids)
        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids


# Example
tokenizer = TokenizerWithSpecialTokens()
input_ids = torch.tensor([
    [2, 10, 20, 30, 3, 0, 0],  # Sequence with padding
    [2, 15, 25, 35, 40, 45, 3]  # Full sequence
])

attention_mask = tokenizer.create_attention_mask(input_ids)
position_ids = tokenizer.create_position_ids(input_ids)

print(f"Input IDs:\n{input_ids}")
print(f"Attention Mask:\n{attention_mask}")
print(f"Position IDs:\n{position_ids}")
```

## 1.8 Building a Complete Tokenizer Pipeline

Let's build a production-ready tokenizer with all features:

```python
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union
import torch

class ProductionTokenizer:
    """Production-ready tokenizer with full feature set."""

    def __init__(self,
                 vocab_size: int = 10000,
                 tokenization_type: str = 'bpe',
                 special_tokens: Dict[str, int] = None):
        """
        Initialize production tokenizer.

        Args:
            vocab_size: Target vocabulary size
            tokenization_type: 'char', 'word', or 'bpe'
            special_tokens: Dict mapping special token strings to IDs
        """
        self.vocab_size = vocab_size
        self.tokenization_type = tokenization_type

        # Default special tokens
        if special_tokens is None:
            self.special_tokens = {
                '[PAD]': 0,
                '[UNK]': 1,
                '[BOS]': 2,
                '[EOS]': 3,
                '[MASK]': 4
            }
        else:
            self.special_tokens = special_tokens

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Initialize with special tokens
        for token, token_id in self.special_tokens.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

        # For BPE
        self.merges: List[Tuple[str, str]] = []

    def train(self, texts: List[str], verbose: bool = False) -> None:
        """Train tokenizer on corpus."""
        if self.tokenization_type == 'bpe':
            self._train_bpe(texts, verbose)
        elif self.tokenization_type == 'word':
            self._train_word(texts, verbose)
        elif self.tokenization_type == 'char':
            self._train_char(texts, verbose)
        else:
            raise ValueError(f"Unknown tokenization type: {self.tokenization_type}")

    def _train_bpe(self, texts: List[str], verbose: bool) -> None:
        """Train BPE tokenizer (simplified version from earlier)."""
        # Implementation from BPETokenizer class above
        # (omitted for brevity - use the earlier implementation)
        pass

    def _train_word(self, texts: List[str], verbose: bool) -> None:
        """Train word-level tokenizer."""
        # Implementation from WordTokenizer class above
        pass

    def _train_char(self, texts: List[str], verbose: bool) -> None:
        """Train character-level tokenizer."""
        # Implementation from CharTokenizer class above
        pass

    def encode(self,
               text: Union[str, List[str]],
               add_special_tokens: bool = True,
               max_length: Optional[int] = None,
               padding: bool = False,
               truncation: bool = False,
               return_tensors: Optional[str] = None) -> Union[List[int], Dict[str, torch.Tensor]]:
        """
        Encode text(s) to token IDs with full feature set.

        Args:
            text: Single string or list of strings
            add_special_tokens: Add [BOS] and [EOS]
            max_length: Maximum sequence length
            padding: Pad to max_length or longest in batch
            truncation: Truncate sequences longer than max_length
            return_tensors: 'pt' for PyTorch tensors, None for lists

        Returns:
            Encoded token IDs as list or dict with tensors
        """
        # Handle single text vs batch
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]

        # Encode each text
        all_token_ids = []
        for txt in texts:
            token_ids = self._encode_single(txt, add_special_tokens)

            # Truncation
            if truncation and max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            all_token_ids.append(token_ids)

        # Padding
        if padding:
            target_length = max_length if max_length else max(len(ids) for ids in all_token_ids)
            pad_id = self.special_tokens['[PAD]']
            all_token_ids = [
                ids + [pad_id] * (target_length - len(ids))
                for ids in all_token_ids
            ]

        # Return format
        if return_tensors == 'pt':
            input_ids = torch.tensor(all_token_ids, dtype=torch.long)
            attention_mask = self._create_attention_mask(input_ids)

            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

            if not is_batch:
                # Return single-item batch
                result = {k: v for k, v in result.items()}

            return result
        else:
            return all_token_ids if is_batch else all_token_ids[0]

    def _encode_single(self, text: str, add_special_tokens: bool) -> List[int]:
        """Encode a single text string."""
        # Implementation depends on tokenization_type
        # (use implementations from earlier classes)
        pass

    def _create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask (1 for real tokens, 0 for padding)."""
        pad_id = self.special_tokens['[PAD]']
        return (input_ids != pad_id).long()

    def decode(self,
               token_ids: Union[List[int], torch.Tensor],
               skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '[UNK]')

            if skip_special_tokens and token in self.special_tokens:
                continue

            tokens.append(token)

        # Join tokens (implementation depends on tokenization type)
        return ' '.join(tokens)

    def save(self, save_path: Union[str, Path]) -> None:
        """Save tokenizer to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            'vocab_size': self.vocab_size,
            'tokenization_type': self.tokenization_type,
            'special_tokens': self.special_tokens
        }

        with open(save_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Save vocabulary
        with open(save_path / 'vocab.json', 'w') as f:
            json.dump(self.token_to_id, f, indent=2)

        # Save merges (for BPE)
        if self.tokenization_type == 'bpe':
            with open(save_path / 'merges.pkl', 'wb') as f:
                pickle.dump(self.merges, f)

        print(f"Tokenizer saved to {save_path}")

    @classmethod
    def load(cls, load_path: Union[str, Path]) -> 'ProductionTokenizer':
        """Load tokenizer from disk."""
        load_path = Path(load_path)

        # Load configuration
        with open(load_path / 'config.json', 'r') as f:
            config = json.load(f)

        # Create tokenizer instance
        tokenizer = cls(**config)

        # Load vocabulary
        with open(load_path / 'vocab.json', 'r') as f:
            tokenizer.token_to_id = json.load(f)
            tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}

        # Load merges (for BPE)
        if tokenizer.tokenization_type == 'bpe':
            merge_path = load_path / 'merges.pkl'
            if merge_path.exists():
                with open(merge_path, 'rb') as f:
                    tokenizer.merges = pickle.load(f)

        print(f"Tokenizer loaded from {load_path}")
        return tokenizer
```

## 1.9 Exercises

### Exercise 1: Vocabulary Size Analysis
Implement a function to analyze how vocabulary size affects sequence length:

```python
def analyze_vocab_size_impact(text: str, vocab_sizes: List[int]) -> Dict:
    """
    Analyze how different vocabulary sizes affect tokenization.

    Args:
        text: Sample text
        vocab_sizes: List of vocabulary sizes to test

    Returns:
        Dictionary with analysis results
    """
    # TODO: Implement this
    # 1. Train BPE tokenizers with different vocab sizes
    # 2. Encode the same text with each
    # 3. Record sequence length, compression ratio, etc.
    pass
```

### Exercise 2: OOV Analysis
Compare how different tokenization methods handle out-of-vocabulary words:

```python
def compare_oov_handling(train_texts: List[str], test_texts: List[str]) -> Dict:
    """
    Compare OOV handling across tokenization methods.

    Args:
        train_texts: Training corpus
        test_texts: Test corpus with OOV words

    Returns:
        Comparison results
    """
    # TODO: Implement this
    # 1. Train char, word, and BPE tokenizers
    # 2. Test on OOV-heavy test set
    # 3. Measure UNK token frequency
    pass
```

### Exercise 3: Multilingual Tokenization
Build a tokenizer that handles multiple languages:

```python
def build_multilingual_tokenizer(texts_by_language: Dict[str, List[str]]) -> ProductionTokenizer:
    """
    Build a tokenizer for multiple languages.

    Args:
        texts_by_language: Dict mapping language code to texts

    Returns:
        Trained multilingual tokenizer
    """
    # TODO: Implement this
    # Consider: character coverage, script diversity, shared subwords
    pass
```

### Exercise 4: Custom Merging Strategy
Implement a custom merge criterion for BPE:

```python
def custom_merge_criterion(pair: Tuple[str, str],
                          pair_freq: int,
                          token_freqs: Dict[str, int]) -> float:
    """
    Custom scoring function for BPE merges.

    Args:
        pair: Token pair
        pair_freq: Frequency of the pair
        token_freqs: Frequencies of individual tokens

    Returns:
        Score for this merge (higher = better)
    """
    # TODO: Implement a custom criterion
    # Ideas: consider token length, entropy, linguistic features
    pass
```

## 1.10 Key Takeaways

1. **Tokenization is fundamental**: It bridges raw text and neural network processing
2. **Trade-offs are everywhere**: Vocabulary size vs sequence length, semantic meaning vs granularity
3. **Subword methods dominate**: BPE, WordPiece, and SentencePiece offer the best balance
4. **Language matters**: Different languages benefit from different tokenization strategies
5. **Special tokens are crucial**: They enable models to learn task-specific behaviors
6. **No one-size-fits-all**: Choose tokenization based on your task, data, and constraints

## 1.11 Further Reading

**Foundational Papers:**
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) - Sennrich et al., 2016
- [Japanese and Korean Voice Search (WordPiece)](https://research.google/pubs/pub37842/) - Schuster & Nakajima, 2012
- [SentencePiece: A simple and language independent approach to subword tokenization](https://arxiv.org/abs/1808.06226) - Kudo & Richardson, 2018
- [Language Models are Unsupervised Multitask Learners (GPT-2/Byte-level BPE)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al., 2019

**Libraries and Tools:**
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) - Fast, production-ready tokenizers
- [SentencePiece](https://github.com/google/sentencepiece) - Google's language-independent tokenizer
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI's fast BPE tokenizer

**Next Chapter:**
Continue to [Chapter 2: Embeddings](02-embeddings.md) to learn how tokens are converted into dense vector representations.

---

*This chapter provided a comprehensive introduction to tokenization. You should now understand the algorithms, be able to implement them from scratch, and know how to choose the right approach for your application. The implementations here are educational; for production, use battle-tested libraries like Hugging Face Tokenizers or SentencePiece.*
