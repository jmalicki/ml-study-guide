# Chapter 14: Data Curation and Preprocessing

## Introduction

Data quality is often the most important factor in training high-performing language models. As the saying goes in machine learning: "garbage in, garbage out." This chapter covers the entire pipeline of preparing data for large language model training, from raw web scrapes to carefully curated and filtered datasets.

Modern LLMs are typically trained on trillions of tokens from diverse sources. The [GPT-3 paper](https://arxiv.org/abs/2005.14165) used 300B tokens from various sources, while more recent models like [LLaMA 2](https://arxiv.org/abs/2307.09288) used 2T tokens and [Qwen 2.5](https://arxiv.org/abs/2412.15115) used over 18T tokens. The curation process for these datasets involves sophisticated filtering, deduplication, and quality scoring techniques.

Key datasets and papers we'll reference:
- [**The Pile**](https://arxiv.org/abs/2101.00027) (EleutherAI): 825GB diverse dataset with 22 high-quality sources
- [**RefinedWeb**](https://arxiv.org/abs/2306.01116) (Falcon): Massive-scale web data with careful filtering
- [**RedPajama**](https://github.com/togethercomputer/RedPajama-Data): Open reproduction of LLaMA's training data
- [**Dolma**](https://arxiv.org/abs/2402.00159) (AI2): 3T token dataset with detailed curation documentation
- [**FineWeb**](https://huggingface.co/datasets/HuggingFaceFW/fineweb): 15T token CommonCrawl-based dataset
- [**C4**](https://arxiv.org/abs/1910.10683) (Colossal Clean Crawled Corpus): Filtered CommonCrawl data

## Table of Contents

1. [Data Collection and Filtering](#data-collection-and-filtering)
2. [Deduplication Strategies](#deduplication-strategies)
3. [Quality Filtering and Scoring](#quality-filtering-and-scoring)
4. [Data Mixing and Curriculum Learning](#data-mixing-and-curriculum-learning)
5. [Tokenizer Training Data Considerations](#tokenizer-training-data-considerations)
6. [PII Removal and Safety Filtering](#pii-removal-and-safety-filtering)
7. [Complete Pipeline Implementation](#complete-pipeline-implementation)
8. [Exercises](#exercises)

## Data Collection and Filtering

### Data Sources

Common data sources for LLM training include:

1. **Web Crawls**: CommonCrawl, web scraping
2. **Code Repositories**: GitHub, GitLab, StackOverflow
3. **Books**: Project Gutenberg, various book corpora
4. **Scientific Papers**: arXiv, PubMed, S2ORC
5. **Wikipedia and Encyclopedias**
6. **News and Media**: News archives, blogs
7. **Social Media**: Reddit, Twitter (with filtering)
8. **Specialized Domains**: Legal documents, medical texts

### CommonCrawl Processing

[CommonCrawl](https://commoncrawl.org/) is a massive web archive containing petabytes of data collected since 2008. It's the foundation for many LLM training datasets.

**CommonCrawl Format**: WARC (Web ARChive) files containing:
- Raw HTML
- HTTP headers
- Metadata

**Key Challenges**:
- Noisy content (ads, navigation, boilerplate)
- Duplicate content
- Low-quality text
- Non-natural language
- Harmful content

### Text Extraction

The first step is extracting clean text from HTML:

```python
import re
from typing import Dict, Optional
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class TextExtractor:
    """Extract clean text from HTML content."""

    def __init__(self):
        # Tags to remove entirely (navigation, scripts, etc.)
        self.remove_tags = [
            'script', 'style', 'nav', 'footer', 'header',
            'aside', 'form', 'iframe', 'noscript'
        ]

        # Block-level tags that should add newlines
        self.block_tags = [
            'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'li', 'blockquote', 'pre', 'article', 'section'
        ]

    def extract(self, html: str, url: Optional[str] = None) -> Dict[str, str]:
        """
        Extract text from HTML.

        Args:
            html: Raw HTML content
            url: URL of the page (for metadata)

        Returns:
            Dictionary with extracted text and metadata
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Extract metadata
        metadata = self._extract_metadata(soup, url)

        # Remove unwanted tags
        for tag in self.remove_tags:
            for element in soup.find_all(tag):
                element.decompose()

        # Extract main content (prioritize article/main tags)
        main_content = (
            soup.find('article') or
            soup.find('main') or
            soup.find('body') or
            soup
        )

        # Get text with line breaks preserved
        text = self._get_text_with_breaks(main_content)

        # Clean up whitespace
        text = self._clean_whitespace(text)

        return {
            'text': text,
            'title': metadata['title'],
            'url': metadata['url'],
            'domain': metadata['domain'],
        }

    def _extract_metadata(self, soup: BeautifulSoup, url: Optional[str]) -> Dict[str, str]:
        """Extract metadata from HTML."""
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ''

        domain = ''
        if url:
            parsed = urlparse(url)
            domain = parsed.netloc

        return {
            'title': title,
            'url': url or '',
            'domain': domain,
        }

    def _get_text_with_breaks(self, element) -> str:
        """Get text while preserving paragraph breaks."""
        text_parts = []

        for child in element.descendants:
            if isinstance(child, str):
                text_parts.append(child)
            elif child.name in self.block_tags:
                text_parts.append('\n')

        return ''.join(text_parts)

    def _clean_whitespace(self, text: str) -> str:
        """Clean up excessive whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()


# Example usage
extractor = TextExtractor()

html_example = """
<html>
<head><title>Machine Learning Basics</title></head>
<body>
    <nav>Navigation menu</nav>
    <article>
        <h1>Introduction to Neural Networks</h1>
        <p>Neural networks are computing systems inspired by biological neural networks.</p>
        <p>They consist of layers of interconnected nodes called neurons.</p>
        <script>trackingCode();</script>
    </article>
    <footer>Copyright 2024</footer>
</body>
</html>
"""

result = extractor.extract(html_example, "https://example.com/ml-basics")
print("Title:", result['title'])
print("Domain:", result['domain'])
print("\nText:")
print(result['text'])
```

### Basic Filtering Rules

After extraction, apply basic filters to remove low-quality content:

```python
import re
from typing import List, Dict
import unicodedata

class BasicFilter:
    """Apply basic quality filters to documents."""

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 1_000_000,
        min_avg_word_length: float = 3.0,
        max_avg_word_length: float = 10.0,
        max_symbol_ratio: float = 0.15,
        max_uppercase_ratio: float = 0.3,
        max_bullet_ratio: float = 0.5,
        max_ellipsis_ratio: float = 0.1,
        min_alpha_ratio: float = 0.6,
        blocklist_words: Optional[List[str]] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_symbol_ratio = max_symbol_ratio
        self.max_uppercase_ratio = max_uppercase_ratio
        self.max_bullet_ratio = max_bullet_ratio
        self.max_ellipsis_ratio = max_ellipsis_ratio
        self.min_alpha_ratio = min_alpha_ratio

        # Words that indicate low-quality content
        self.blocklist_words = blocklist_words or [
            'lorem ipsum', 'click here', 'subscribe now',
            'buy now', 'sponsored content'
        ]

    def filter(self, text: str) -> Dict[str, any]:
        """
        Apply all filters to text.

        Returns:
            Dictionary with 'passed' boolean and 'reasons' list
        """
        reasons = []

        # Length checks
        if len(text) < self.min_length:
            reasons.append(f"Too short ({len(text)} chars)")
        if len(text) > self.max_length:
            reasons.append(f"Too long ({len(text)} chars)")

        # Word-level checks
        words = text.split()
        if len(words) > 0:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < self.min_avg_word_length:
                reasons.append(f"Words too short (avg {avg_word_len:.1f})")
            if avg_word_len > self.max_avg_word_length:
                reasons.append(f"Words too long (avg {avg_word_len:.1f})")

        # Character-level checks
        alpha_count = sum(c.isalpha() for c in text)
        upper_count = sum(c.isupper() for c in text)
        symbol_count = sum(c in '!@#$%^&*()' for c in text)

        if len(text) > 0:
            alpha_ratio = alpha_count / len(text)
            if alpha_ratio < self.min_alpha_ratio:
                reasons.append(f"Too few letters ({alpha_ratio:.2%})")

            if alpha_count > 0:
                uppercase_ratio = upper_count / alpha_count
                if uppercase_ratio > self.max_uppercase_ratio:
                    reasons.append(f"Too much uppercase ({uppercase_ratio:.2%})")

            symbol_ratio = symbol_count / len(text)
            if symbol_ratio > self.max_symbol_ratio:
                reasons.append(f"Too many symbols ({symbol_ratio:.2%})")

        # Detect list-heavy content (likely navigation, not prose)
        lines = text.split('\n')
        bullet_lines = sum(1 for line in lines if line.strip().startswith(('•', '-', '*', '·')))
        if len(lines) > 0:
            bullet_ratio = bullet_lines / len(lines)
            if bullet_ratio > self.max_bullet_ratio:
                reasons.append(f"Too many bullet points ({bullet_ratio:.2%})")

        # Detect excessive ellipsis (often indicates truncated/low-quality content)
        ellipsis_count = text.count('...') + text.count('…')
        if len(words) > 0:
            ellipsis_ratio = ellipsis_count / len(words)
            if ellipsis_ratio > self.max_ellipsis_ratio:
                reasons.append(f"Too many ellipses ({ellipsis_count})")

        # Blocklist check
        text_lower = text.lower()
        for word in self.blocklist_words:
            if word in text_lower:
                reasons.append(f"Contains blocklisted phrase: '{word}'")
                break

        return {
            'passed': len(reasons) == 0,
            'reasons': reasons,
        }


# Example usage
filter_engine = BasicFilter()

# Good example
good_text = """
Neural networks are a fundamental component of modern machine learning systems.
They consist of layers of interconnected nodes that process information in a
hierarchical manner, allowing them to learn complex patterns from data.
"""

result = filter_engine.filter(good_text)
print("Good text passed:", result['passed'])

# Bad examples
bad_examples = [
    "Click here!!! Buy now!!! AMAZING DEALS!!!",  # Too much uppercase, symbols
    "a b c d e f",  # Too short, weird word length
    "• Item 1\n• Item 2\n• Item 3\n• Item 4\n• Item 5",  # Too many bullets
]

for i, bad_text in enumerate(bad_examples, 1):
    result = filter_engine.filter(bad_text)
    print(f"\nBad example {i} passed: {result['passed']}")
    if not result['passed']:
        print(f"Reasons: {', '.join(result['reasons'])}")
```

### Language Identification

Filter for specific languages using language detection:

```python
try:
    from typing import Optional
    # In practice, use: pip install langdetect or fasttext
    # For this example, we'll create a simple heuristic-based detector

    class SimpleLanguageDetector:
        """Simple language detector based on character ranges."""

        def __init__(self):
            # Character ranges for different scripts
            self.ranges = {
                'latin': (0x0041, 0x024F),  # Basic Latin + Latin Extended
                'cyrillic': (0x0400, 0x04FF),
                'arabic': (0x0600, 0x06FF),
                'chinese': (0x4E00, 0x9FFF),  # CJK Unified Ideographs
                'japanese_hiragana': (0x3040, 0x309F),
                'japanese_katakana': (0x30A0, 0x30FF),
                'korean': (0xAC00, 0xD7AF),  # Hangul
            }

        def detect(self, text: str, min_confidence: float = 0.5) -> Optional[str]:
            """
            Detect language based on character distribution.

            Returns:
                Language code or None if uncertain
            """
            if not text:
                return None

            # Count characters in each script
            script_counts = {script: 0 for script in self.ranges}
            total_chars = 0

            for char in text:
                code_point = ord(char)
                total_chars += 1

                for script, (start, end) in self.ranges.items():
                    if start <= code_point <= end:
                        script_counts[script] += 1
                        break

            if total_chars == 0:
                return None

            # Find dominant script
            max_script = max(script_counts, key=script_counts.get)
            confidence = script_counts[max_script] / total_chars

            if confidence < min_confidence:
                return None

            # Map scripts to language codes (simplified)
            script_to_lang = {
                'latin': 'en',  # Assume English for Latin (could be many languages)
                'cyrillic': 'ru',
                'arabic': 'ar',
                'chinese': 'zh',
                'japanese_hiragana': 'ja',
                'japanese_katakana': 'ja',
                'korean': 'ko',
            }

            return script_to_lang.get(max_script)


    detector = SimpleLanguageDetector()

    examples = [
        ("This is an English sentence.", "en"),
        ("これは日本語の文です。", "ja"),
        ("这是中文句子。", "zh"),
        ("Это русский текст.", "ru"),
    ]

    for text, expected in examples:
        detected = detector.detect(text)
        print(f"Text: {text[:30]}... -> Detected: {detected} (Expected: {expected})")

except Exception as e:
    print(f"Language detection example skipped: {e}")
```

For production systems, use established libraries:
- **langdetect**: Python port of Google's language-detection library
- **fastText**: Facebook's language identification model (very fast)
- **pycld2/pycld3**: Python bindings for Chrome's Compact Language Detector

## Deduplication Strategies

Deduplication is critical for several reasons:
1. **Reduces memorization**: Models memorize frequent sequences
2. **Improves generalization**: Less repetition means broader coverage
3. **Ethical concerns**: Reduces privacy risks from repeated PII
4. **Efficiency**: Smaller datasets train faster

The [Dolma paper](https://arxiv.org/abs/2402.00159) found that aggressive deduplication improved downstream performance significantly.

### Exact Deduplication

The simplest approach: remove exact duplicates using hashing.

```python
import hashlib
from typing import Set, List, Dict
from collections import defaultdict

class ExactDeduplicator:
    """Remove exact duplicate documents using hashing."""

    def __init__(self, hash_algorithm: str = 'sha256'):
        self.hash_algorithm = hash_algorithm
        self.seen_hashes: Set[str] = set()
        self.duplicate_count = 0

    def compute_hash(self, text: str) -> str:
        """Compute hash of text."""
        # Normalize whitespace before hashing
        normalized = ' '.join(text.split())

        if self.hash_algorithm == 'sha256':
            return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        elif self.hash_algorithm == 'md5':
            return hashlib.md5(normalized.encode('utf-8')).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")

    def is_duplicate(self, text: str) -> bool:
        """Check if document is a duplicate."""
        doc_hash = self.compute_hash(text)

        if doc_hash in self.seen_hashes:
            self.duplicate_count += 1
            return True

        self.seen_hashes.add(doc_hash)
        return False

    def deduplicate(self, documents: List[str]) -> List[str]:
        """Remove duplicates from document list."""
        unique_docs = []

        for doc in documents:
            if not self.is_duplicate(doc):
                unique_docs.append(doc)

        return unique_docs

    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return {
            'unique_documents': len(self.seen_hashes),
            'duplicates_found': self.duplicate_count,
        }


# Example usage
deduplicator = ExactDeduplicator()

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "The quick brown fox jumps over the lazy dog.",  # Exact duplicate
    "Neural networks learn from data.",
    "The   quick   brown   fox   jumps   over   the   lazy   dog.",  # Whitespace variant
]

unique_docs = deduplicator.deduplicate(documents)
print(f"Original: {len(documents)} documents")
print(f"After dedup: {len(unique_docs)} documents")
print(f"Stats: {deduplicator.get_stats()}")
print("\nUnique documents:")
for i, doc in enumerate(unique_docs, 1):
    print(f"{i}. {doc[:60]}...")
```

### MinHash for Near-Duplicate Detection

MinHash enables efficient detection of near-duplicates using Jaccard similarity. This is crucial for finding documents that are slightly modified versions of each other.

**Theory**: MinHash provides an approximation of Jaccard similarity between sets:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

For sets of n-grams, MinHash signatures allow us to estimate similarity in $O(k)$ time instead of $O(|A| + |B|)$, where $k$ is the signature size.

```python
import hashlib
from typing import List, Set, Tuple
import random

class MinHash:
    """MinHash for near-duplicate detection."""

    def __init__(self, num_perm: int = 128, ngram_size: int = 5, seed: int = 42):
        """
        Initialize MinHash.

        Args:
            num_perm: Number of permutations (signature size)
            ngram_size: Size of character n-grams
            seed: Random seed for hash functions
        """
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.seed = seed

        # Generate hash functions (using different seeds)
        random.seed(seed)
        self.hash_seeds = [random.randint(0, 2**32 - 1) for _ in range(num_perm)]

    def _get_ngrams(self, text: str) -> Set[str]:
        """Extract character n-grams from text."""
        # Normalize whitespace
        text = ' '.join(text.split())

        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])

        return ngrams

    def _hash_ngram(self, ngram: str, seed: int) -> int:
        """Hash an n-gram with a specific seed."""
        # Combine ngram with seed
        combined = f"{ngram}_{seed}"
        hash_val = hashlib.sha256(combined.encode('utf-8')).digest()
        # Convert first 8 bytes to integer
        return int.from_bytes(hash_val[:8], byteorder='big')

    def compute_signature(self, text: str) -> List[int]:
        """
        Compute MinHash signature for text.

        Returns:
            List of hash values (signature)
        """
        ngrams = self._get_ngrams(text)

        if not ngrams:
            return [0] * self.num_perm

        signature = []

        # For each hash function (permutation)
        for seed in self.hash_seeds:
            # Find minimum hash value across all n-grams
            min_hash = min(self._hash_ngram(ngram, seed) for ngram in ngrams)
            signature.append(min_hash)

        return signature

    def estimate_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Estimate Jaccard similarity from signatures.

        Args:
            sig1, sig2: MinHash signatures

        Returns:
            Estimated Jaccard similarity [0, 1]
        """
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have same length")

        matches = sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2)
        return matches / len(sig1)


class LSH:
    """Locality Sensitive Hashing for efficient near-duplicate detection."""

    def __init__(self, num_bands: int = 16, rows_per_band: int = 8):
        """
        Initialize LSH.

        Args:
            num_bands: Number of bands (more = faster but less sensitive)
            rows_per_band: Rows per band (signature must be divisible)
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.signature_size = num_bands * rows_per_band

        # Store buckets: band_id -> band_hash -> list of doc_ids
        self.buckets = [defaultdict(list) for _ in range(num_bands)]

    def _hash_band(self, band: List[int]) -> int:
        """Hash a band (subset of signature)."""
        band_str = ','.join(map(str, band))
        hash_val = hashlib.sha256(band_str.encode('utf-8')).digest()
        return int.from_bytes(hash_val[:8], byteorder='big')

    def add(self, doc_id: int, signature: List[int]):
        """Add document signature to LSH index."""
        if len(signature) != self.signature_size:
            raise ValueError(f"Signature size must be {self.signature_size}")

        # Split signature into bands
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]

            # Hash the band and add to bucket
            band_hash = self._hash_band(band)
            self.buckets[band_idx][band_hash].append(doc_id)

    def query(self, signature: List[int]) -> Set[int]:
        """
        Find candidate duplicates for a signature.

        Returns:
            Set of document IDs that are candidates
        """
        candidates = set()

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]

            band_hash = self._hash_band(band)

            # Get all documents in same bucket
            if band_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_hash])

        return candidates


# Example usage
print("=== MinHash Example ===\n")

minhash = MinHash(num_perm=128, ngram_size=5)

doc1 = "The quick brown fox jumps over the lazy dog."
doc2 = "The quick brown fox jumps over a lazy dog."  # Very similar
doc3 = "Machine learning is revolutionizing artificial intelligence."  # Different

sig1 = minhash.compute_signature(doc1)
sig2 = minhash.compute_signature(doc2)
sig3 = minhash.compute_signature(doc3)

print(f"Similarity(doc1, doc2): {minhash.estimate_jaccard(sig1, sig2):.3f}")
print(f"Similarity(doc1, doc3): {minhash.estimate_jaccard(sig1, sig3):.3f}")

print("\n=== LSH Example ===\n")

# Build LSH index
lsh = LSH(num_bands=16, rows_per_band=8)

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox leaps over the lazy dog.",
    "Machine learning models require large datasets.",
    "Deep neural networks are powerful function approximators.",
    "The fast brown fox jumps over the sleepy dog.",
]

signatures = [minhash.compute_signature(doc) for doc in documents]

# Add to LSH index
for doc_id, sig in enumerate(signatures):
    lsh.add(doc_id, sig)

# Query for duplicates of first document
candidates = lsh.query(signatures[0])
print(f"Candidates for document 0: {candidates}")

# Verify with actual similarity
print("\nActual similarities:")
for candidate in candidates:
    if candidate != 0:
        sim = minhash.estimate_jaccard(signatures[0], signatures[candidate])
        print(f"  Doc 0 vs Doc {candidate}: {sim:.3f}")
        print(f"    Doc {candidate}: {documents[candidate][:60]}...")
```

### Fuzzy Deduplication Implementation

Combining MinHash and LSH for efficient fuzzy deduplication:

```python
from typing import List, Dict, Tuple, Optional

class FuzzyDeduplicator:
    """Fuzzy deduplication using MinHash + LSH."""

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        num_bands: int = 16,
        ngram_size: int = 5,
    ):
        """
        Initialize fuzzy deduplicator.

        Args:
            threshold: Jaccard similarity threshold for duplicates
            num_perm: Number of MinHash permutations
            num_bands: Number of LSH bands
            ngram_size: Size of character n-grams
        """
        self.threshold = threshold
        self.minhash = MinHash(num_perm=num_perm, ngram_size=ngram_size)

        rows_per_band = num_perm // num_bands
        self.lsh = LSH(num_bands=num_bands, rows_per_band=rows_per_band)

        self.signatures: List[List[int]] = []
        self.documents: List[str] = []
        self.is_duplicate: List[bool] = []

    def add_document(self, text: str) -> Tuple[int, bool]:
        """
        Add document and check if it's a duplicate.

        Returns:
            (doc_id, is_duplicate)
        """
        doc_id = len(self.documents)
        signature = self.minhash.compute_signature(text)

        # Find candidates
        candidates = self.lsh.query(signature)

        # Check actual similarity with candidates
        is_dup = False
        for candidate_id in candidates:
            similarity = self.minhash.estimate_jaccard(
                signature, self.signatures[candidate_id]
            )

            if similarity >= self.threshold:
                is_dup = True
                break

        # Store document
        self.documents.append(text)
        self.signatures.append(signature)
        self.is_duplicate.append(is_dup)

        # Add to LSH index if not duplicate
        if not is_dup:
            self.lsh.add(doc_id, signature)

        return doc_id, is_dup

    def get_unique_documents(self) -> List[str]:
        """Get all unique documents."""
        return [
            doc for doc, is_dup in zip(self.documents, self.is_duplicate)
            if not is_dup
        ]

    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return {
            'total_documents': len(self.documents),
            'unique_documents': sum(1 for d in self.is_duplicate if not d),
            'duplicates': sum(1 for d in self.is_duplicate if d),
        }


# Example usage
print("=== Fuzzy Deduplication Example ===\n")

deduplicator = FuzzyDeduplicator(threshold=0.8)

test_docs = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "The quick brown fox leaps over the lazy dog.",  # Similar to doc 0
    "Neural networks consist of layers of interconnected nodes.",
    "The fast brown fox jumps over the sleepy dog.",  # Similar to doc 0
    "Deep learning uses neural networks with many layers.",
    "Machine learning is a branch of AI.",  # Similar to doc 1
]

for i, doc in enumerate(test_docs):
    doc_id, is_dup = deduplicator.add_document(doc)
    status = "DUPLICATE" if is_dup else "UNIQUE"
    print(f"Doc {doc_id} [{status}]: {doc[:60]}...")

print(f"\nStats: {deduplicator.get_stats()}")
print(f"\nUnique documents: {len(deduplicator.get_unique_documents())}")
```

**Note**: For production systems at scale, consider:
- **[datasketch](https://github.com/ekzhu/datasketch)**: Efficient MinHash/LSH implementation
- **Distributed processing**: Use Spark or Dask for large-scale deduplication
- **Disk-based storage**: Store signatures on disk for datasets larger than memory

## Quality Filtering and Scoring

Beyond basic filtering, we need sophisticated quality scoring to rank documents.

### Perplexity-Based Filtering

Use a language model to score text quality. High-quality text has lower perplexity under a well-trained LM.

**Perplexity** is defined as:

$$\text{PPL}(X) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i | x_{<i})\right)$$

where $N$ is the number of tokens and $P(x_i | x_{<i})$ is the model's predicted probability of token $x_i$ given context.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math

class PerplexityFilter:
    """Filter documents based on perplexity from a language model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: any,
        max_perplexity: float = 1500.0,
        device: str = 'cpu',
    ):
        """
        Initialize perplexity filter.

        Args:
            model: Pre-trained language model
            tokenizer: Tokenizer for the model
            max_perplexity: Maximum acceptable perplexity
            device: Device to run model on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_perplexity = max_perplexity
        self.device = device

    @torch.no_grad()
    def compute_perplexity(self, text: str, stride: int = 512) -> float:
        """
        Compute perplexity using sliding window.

        Args:
            text: Input text
            stride: Stride for sliding window (for long texts)

        Returns:
            Perplexity score
        """
        # Tokenize
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings['input_ids'].to(self.device)

        seq_len = input_ids.size(1)

        if seq_len == 0:
            return float('inf')

        # For short sequences, compute directly
        if seq_len <= self.model.config.max_position_embeddings:
            outputs = self.model(input_ids, labels=input_ids)
            return math.exp(outputs.loss.item())

        # For long sequences, use sliding window
        max_length = self.model.config.max_position_embeddings
        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids_window = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_window.clone()
            target_ids[:, :-trg_len] = -100  # Ignore context

            outputs = self.model(input_ids_window, labels=target_ids)
            nlls.append(outputs.loss * trg_len)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        perplexity = math.exp(sum(nlls) / end_loc)
        return perplexity

    def should_keep(self, text: str) -> tuple[bool, float]:
        """
        Determine if document should be kept.

        Returns:
            (should_keep, perplexity)
        """
        perplexity = self.compute_perplexity(text)
        return perplexity <= self.max_perplexity, perplexity


# Example usage (with dummy model for illustration)
print("=== Perplexity Filtering Example ===\n")

# For a real implementation, use a pre-trained model like:
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Simple dummy example showing the concept
class DummyTokenizer:
    def __call__(self, text, return_tensors='pt'):
        # Simple word-based tokenization for demo
        tokens = text.split()
        token_ids = [hash(token) % 1000 for token in tokens]
        return {'input_ids': torch.tensor([token_ids])}

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('Config', (), {'max_position_embeddings': 1024})()
        self.embedding = nn.Embedding(1000, 128)
        self.lm_head = nn.Linear(128, 1000)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        logits = self.lm_head(x)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return type('Output', (), {'loss': loss})()

        return type('Output', (), {'logits': logits})()

# Demo with dummy model
dummy_model = DummyModel()
dummy_tokenizer = DummyTokenizer()

perplexity_filter = PerplexityFilter(
    model=dummy_model,
    tokenizer=dummy_tokenizer,
    max_perplexity=1500.0,
)

good_text = "Machine learning is a rapidly evolving field with applications in computer vision and natural language processing"
random_text = "asdfkj qwer zxcv tyui bnm hjkl vbnm"

for label, text in [("Good", good_text), ("Random", random_text)]:
    should_keep, ppl = perplexity_filter.should_keep(text)
    print(f"{label} text (PPL={ppl:.1f}): {'KEEP' if should_keep else 'REJECT'}")
```

### Classifier-Based Quality Scoring

Train a classifier to predict document quality based on human-labeled examples.

The [RefinedWeb paper](https://arxiv.org/abs/2306.01116) used a fastText classifier trained on curated vs. random web data.

```python
import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np

class QualityClassifier(nn.Module):
    """Neural classifier for document quality."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            Quality scores [batch_size] in range [0, 1]
        """
        # Embed: [batch_size, seq_len, embedding_dim]
        x = self.embedding(input_ids)

        # LSTM: output is [batch_size, seq_len, hidden_dim * 2]
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use final hidden state from both directions
        # hidden: [2, batch_size, hidden_dim]
        forward_hidden = hidden[0]  # [batch_size, hidden_dim]
        backward_hidden = hidden[1]  # [batch_size, hidden_dim]

        # Concatenate: [batch_size, hidden_dim * 2]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Classify: [batch_size, 1] -> [batch_size]
        logits = self.fc(combined).squeeze(1)
        scores = self.sigmoid(logits)

        return scores


class QualityFilter:
    """Filter documents using quality classifier."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: any,
        threshold: float = 0.5,
        device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.device = device

    @torch.no_grad()
    def score(self, text: str) -> float:
        """Score document quality."""
        # Tokenize
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = tokens['input_ids'].to(self.device)

        # Get quality score
        score = self.model(input_ids).item()
        return score

    def should_keep(self, text: str) -> tuple[bool, float]:
        """Determine if document should be kept."""
        score = self.score(text)
        return score >= self.threshold, score


# Example: Training a quality classifier
def train_quality_classifier(
    train_data: List[tuple[str, int]],  # (text, label) pairs
    vocab_size: int = 10000,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
):
    """
    Train quality classifier.

    Args:
        train_data: List of (text, label) where label is 0 (low quality) or 1 (high quality)
    """
    # Initialize model
    model = QualityClassifier(vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Simple tokenizer (in practice, use proper tokenizer)
    def simple_tokenize(text, max_len=512):
        words = text.lower().split()[:max_len]
        ids = [hash(word) % vocab_size for word in words]
        # Pad to max_len
        ids = ids + [0] * (max_len - len(ids))
        return torch.tensor(ids)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        # Shuffle data
        np.random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            # Prepare batch
            texts, labels = zip(*batch)
            input_ids = torch.stack([simple_tokenize(t) for t in texts])
            labels = torch.tensor(labels, dtype=torch.float32)

            # Forward pass
            scores = model(input_ids)
            loss = criterion(scores, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item()
            predictions = (scores > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

        accuracy = correct / total
        avg_loss = total_loss / (len(train_data) / batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model


# Example usage
print("\n=== Quality Classifier Training Example ===\n")

# Example training data (in practice, need much more data)
training_data = [
    ("Machine learning is transforming how we process data and make predictions.", 1),
    ("Neural networks are powerful function approximators used in deep learning.", 1),
    ("asdf qwer zxcv asdf qwer", 0),
    ("Click here!!! Buy now!!! Amazing deals!!!", 0),
    ("The Transformer architecture revolutionized natural language processing.", 1),
    ("lorem ipsum dolor sit amet consectetur", 0),
] * 20  # Replicate for demo

# Train classifier
trained_model = train_quality_classifier(training_data, epochs=5)

# Use for filtering
quality_filter = QualityFilter(model=trained_model, tokenizer=lambda x, **kwargs: {'input_ids': torch.tensor([[hash(w) % 10000 for w in x.split()[:512]]])}, threshold=0.5)

test_texts = [
    "Deep learning models require large amounts of training data and computational resources.",
    "asdfasdf qwerqwer zxcvzxcv",
]

for text in test_texts:
    should_keep, score = quality_filter.should_keep(text)
    print(f"Score: {score:.3f} - {'KEEP' if should_keep else 'REJECT'}: {text[:60]}...")
```

### Heuristic-Based Scoring

The [Gopher paper](https://arxiv.org/abs/2112.11446) and [C4](https://arxiv.org/abs/1910.10683) used various heuristic scores:

```python
import re
from typing import Dict
import string

class HeuristicScorer:
    """Compute various heuristic quality scores for documents."""

    def compute_scores(self, text: str) -> Dict[str, float]:
        """Compute all heuristic scores."""
        words = text.split()
        lines = text.split('\n')

        scores = {}

        # 1. Word count
        scores['word_count'] = len(words)

        # 2. Mean word length
        if words:
            scores['mean_word_length'] = sum(len(w) for w in words) / len(words)
        else:
            scores['mean_word_length'] = 0

        # 3. Fraction of alphabetic characters
        alpha_chars = sum(c.isalpha() for c in text)
        scores['alpha_ratio'] = alpha_chars / max(len(text), 1)

        # 4. Fraction of uppercase words
        if words:
            uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            scores['uppercase_ratio'] = uppercase_words / len(words)
        else:
            scores['uppercase_ratio'] = 0

        # 5. Fraction of lines ending with punctuation
        if lines:
            lines_with_punct = sum(
                1 for line in lines
                if line.strip() and line.strip()[-1] in '.!?'
            )
            scores['ending_punct_ratio'] = lines_with_punct / len(lines)
        else:
            scores['ending_punct_ratio'] = 0

        # 6. Symbol-to-word ratio
        if words:
            symbols = sum(1 for c in text if c in string.punctuation)
            scores['symbol_to_word_ratio'] = symbols / len(words)
        else:
            scores['symbol_to_word_ratio'] = 0

        # 7. Fraction of lines starting with bullet points
        if lines:
            bullet_lines = sum(
                1 for line in lines
                if line.strip() and line.strip()[0] in ['•', '-', '*', '·']
            )
            scores['bullet_ratio'] = bullet_lines / len(lines)
        else:
            scores['bullet_ratio'] = 0

        # 8. Stop word ratio (simple check for common words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        }
        if words:
            stop_word_count = sum(1 for w in words if w.lower() in stop_words)
            scores['stop_word_ratio'] = stop_word_count / len(words)
        else:
            scores['stop_word_ratio'] = 0

        # 9. Fraction of lines with "..."
        if lines:
            ellipsis_lines = sum(1 for line in lines if '...' in line or '…' in line)
            scores['ellipsis_ratio'] = ellipsis_lines / max(len(lines), 1)
        else:
            scores['ellipsis_ratio'] = 0

        # 10. Duplicate line ratio
        if lines:
            unique_lines = len(set(line.strip() for line in lines if line.strip()))
            non_empty_lines = sum(1 for line in lines if line.strip())
            scores['duplicate_line_ratio'] = 1 - (unique_lines / max(non_empty_lines, 1))
        else:
            scores['duplicate_line_ratio'] = 0

        return scores

    def compute_quality_score(self, text: str) -> float:
        """
        Compute overall quality score using weighted heuristics.

        Returns:
            Score in range [0, 1] where higher is better
        """
        scores = self.compute_scores(text)

        # Weights tuned based on empirical analysis
        # These are example weights - tune for your use case
        quality = 0.0

        # Positive indicators
        quality += 0.2 * min(scores['alpha_ratio'], 1.0)
        quality += 0.15 * min(scores['stop_word_ratio'] / 0.3, 1.0)  # ~30% is good
        quality += 0.15 * min(scores['ending_punct_ratio'], 1.0)

        # Length (prefer medium-length words)
        word_len = scores['mean_word_length']
        if 4.0 <= word_len <= 6.0:
            quality += 0.2
        elif 3.0 <= word_len <= 7.0:
            quality += 0.1

        # Negative indicators (subtract from quality)
        quality -= 0.2 * scores['uppercase_ratio']
        quality -= 0.15 * scores['bullet_ratio']
        quality -= 0.1 * scores['ellipsis_ratio']
        quality -= 0.15 * scores['duplicate_line_ratio']

        # Clamp to [0, 1]
        return max(0.0, min(1.0, quality))


# Example usage
print("\n=== Heuristic Scoring Example ===\n")

scorer = HeuristicScorer()

examples = [
    (
        "good",
        "Machine learning has revolutionized many fields of computer science. "
        "Neural networks, in particular, have shown remarkable success in tasks "
        "such as image recognition and natural language processing. The transformer "
        "architecture introduced attention mechanisms that significantly improved performance."
    ),
    (
        "bad",
        "CLICK HERE NOW!!! BUY BUY BUY!!!\n"
        "• Item 1\n• Item 2\n• Item 3\n• Item 4\n• Item 5\n"
        "...\n...\n..."
    ),
]

for label, text in examples:
    scores = scorer.compute_scores(text)
    quality = scorer.compute_quality_score(text)

    print(f"{label.upper()} example (Quality: {quality:.3f}):")
    print(f"  Alpha ratio: {scores['alpha_ratio']:.3f}")
    print(f"  Stop word ratio: {scores['stop_word_ratio']:.3f}")
    print(f"  Uppercase ratio: {scores['uppercase_ratio']:.3f}")
    print(f"  Bullet ratio: {scores['bullet_ratio']:.3f}")
    print()
```

## Data Mixing and Curriculum Learning

When training on multiple data sources, careful mixing is important for model quality.

### Data Mixing Strategies

**Temperature Sampling**: Sample from different sources with temperature-adjusted probabilities.

Given source sizes $n_1, n_2, \ldots, n_k$, the sampling probability for source $i$ is:

$$p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}$$

where $T$ is temperature:
- $T = 1$: Proportional sampling
- $T \to 0$: Uniform sampling
- $T \to \infty$: Sample only from largest source

```python
import numpy as np
from typing import List, Dict, Iterator
import random

class DataMixer:
    """Mix data from multiple sources using temperature sampling."""

    def __init__(
        self,
        sources: Dict[str, List[str]],  # source_name -> documents
        temperature: float = 1.0,
        seed: int = 42,
    ):
        """
        Initialize data mixer.

        Args:
            sources: Dictionary mapping source names to document lists
            temperature: Sampling temperature (0 = uniform, 1 = proportional)
            seed: Random seed
        """
        self.sources = sources
        self.temperature = temperature
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # Compute sampling probabilities
        self.sampling_probs = self._compute_sampling_probs()

        # Track current indices for each source
        self.indices = {name: 0 for name in sources}

        # Shuffle each source
        for name in sources:
            random.shuffle(sources[name])

    def _compute_sampling_probs(self) -> Dict[str, float]:
        """Compute temperature-adjusted sampling probabilities."""
        sizes = {name: len(docs) for name, docs in self.sources.items()}

        # Apply temperature
        if self.temperature == 0:
            # Uniform sampling
            prob = 1.0 / len(sizes)
            return {name: prob for name in sizes}
        else:
            # Temperature-adjusted
            adjusted_sizes = {
                name: size ** (1.0 / self.temperature)
                for name, size in sizes.items()
            }
            total = sum(adjusted_sizes.values())
            return {
                name: adj_size / total
                for name, adj_size in adjusted_sizes.items()
            }

    def sample_document(self) -> tuple[str, str]:
        """
        Sample a document from mixed sources.

        Returns:
            (source_name, document)
        """
        # Sample source
        source_names = list(self.sources.keys())
        probs = [self.sampling_probs[name] for name in source_names]
        source_name = np.random.choice(source_names, p=probs)

        # Get document from source
        docs = self.sources[source_name]
        idx = self.indices[source_name]

        # Wrap around if needed
        if idx >= len(docs):
            random.shuffle(docs)
            idx = 0

        document = docs[idx]
        self.indices[source_name] = idx + 1

        return source_name, document

    def iterate(self, num_documents: int) -> Iterator[tuple[str, str]]:
        """Iterate over mixed documents."""
        for _ in range(num_documents):
            yield self.sample_document()

    def get_stats(self, num_samples: int = 10000) -> Dict[str, float]:
        """Get empirical sampling statistics."""
        counts = {name: 0 for name in self.sources}

        for _ in range(num_samples):
            source_name, _ = self.sample_document()
            counts[source_name] += 1

        return {
            name: count / num_samples
            for name, count in counts.items()
        }


# Example usage
print("=== Data Mixing Example ===\n")

# Create example sources
sources = {
    'web': ['web doc ' + str(i) for i in range(10000)],
    'books': ['book doc ' + str(i) for i in range(3000)],
    'code': ['code doc ' + str(i) for i in range(2000)],
    'papers': ['paper doc ' + str(i) for i in range(1000)],
}

print("Source sizes:")
for name, docs in sources.items():
    print(f"  {name}: {len(docs):,} documents")

# Try different temperatures
for temp in [0.0, 0.5, 1.0]:
    print(f"\nTemperature = {temp}")
    mixer = DataMixer(sources, temperature=temp)

    print("  Theoretical probabilities:")
    for name, prob in mixer.sampling_probs.items():
        print(f"    {name}: {prob:.3f}")

    print("  Empirical probabilities (10k samples):")
    stats = mixer.get_stats(num_samples=10000)
    for name, prob in stats.items():
        print(f"    {name}: {prob:.3f}")
```

### Curriculum Learning

**Curriculum learning** progressively increases data difficulty during training.

Strategies:
1. **Start with high-quality data**: Begin training on curated sources (books, Wikipedia)
2. **Add web data gradually**: Introduce noisier web data later
3. **Increase sequence length**: Start with shorter sequences, increase over time
4. **Domain progression**: General → Specialized domains

```python
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CurriculumStage:
    """Defines a stage in curriculum learning."""
    name: str
    duration_tokens: int  # How many tokens to train in this stage
    sources: Dict[str, float]  # source_name -> weight
    max_seq_length: int = 2048

class CurriculumScheduler:
    """Schedule curriculum learning stages."""

    def __init__(self, stages: List[CurriculumStage]):
        self.stages = stages
        self.current_stage_idx = 0
        self.tokens_in_stage = 0

    def update(self, num_tokens: int) -> bool:
        """
        Update with number of tokens processed.

        Returns:
            True if stage changed, False otherwise
        """
        self.tokens_in_stage += num_tokens

        current_stage = self.stages[self.current_stage_idx]

        # Check if we should move to next stage
        if (self.tokens_in_stage >= current_stage.duration_tokens and
            self.current_stage_idx < len(self.stages) - 1):
            self.current_stage_idx += 1
            self.tokens_in_stage = 0
            return True

        return False

    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]

    def get_progress(self) -> Dict[str, any]:
        """Get training progress."""
        current_stage = self.get_current_stage()

        return {
            'stage': self.current_stage_idx + 1,
            'total_stages': len(self.stages),
            'stage_name': current_stage.name,
            'tokens_in_stage': self.tokens_in_stage,
            'stage_duration': current_stage.duration_tokens,
            'progress': self.tokens_in_stage / current_stage.duration_tokens,
        }


# Example usage
print("\n=== Curriculum Learning Example ===\n")

# Define curriculum stages
curriculum_stages = [
    CurriculumStage(
        name="High-quality foundation",
        duration_tokens=50_000_000_000,  # 50B tokens
        sources={'books': 0.5, 'wikipedia': 0.3, 'papers': 0.2},
        max_seq_length=1024,
    ),
    CurriculumStage(
        name="Mixed quality",
        duration_tokens=100_000_000_000,  # 100B tokens
        sources={'web': 0.4, 'books': 0.3, 'code': 0.2, 'papers': 0.1},
        max_seq_length=2048,
    ),
    CurriculumStage(
        name="Web-heavy",
        duration_tokens=150_000_000_000,  # 150B tokens
        sources={'web': 0.6, 'code': 0.2, 'books': 0.1, 'papers': 0.1},
        max_seq_length=4096,
    ),
]

scheduler = CurriculumScheduler(curriculum_stages)

# Simulate training
total_tokens_processed = 0
batch_size = 1_000_000  # 1M tokens per batch

print("Training simulation:")
for batch_idx in range(300):  # Process 300 batches
    # Process batch
    stage_changed = scheduler.update(batch_size)
    total_tokens_processed += batch_size

    if stage_changed or batch_idx % 50 == 0:
        progress = scheduler.get_progress()
        print(f"\nBatch {batch_idx}, Total tokens: {total_tokens_processed / 1e9:.1f}B")
        print(f"  Stage {progress['stage']}/{progress['total_stages']}: {progress['stage_name']}")
        print(f"  Progress: {progress['progress']:.1%}")
        print(f"  Max seq length: {scheduler.get_current_stage().max_seq_length}")
        print(f"  Sources: {scheduler.get_current_stage().sources}")
```

## Tokenizer Training Data Considerations

The data used to train your tokenizer affects model performance. See [Chapter 1: Tokenization](01-tokenization.md) for implementation details.

### Key Considerations

1. **Representative sample**: Tokenizer training data should represent the distribution of training data
2. **Size**: Typically 1-10GB of text is sufficient
3. **Multilingual**: Include all languages in appropriate proportions
4. **Vocabulary size**: Balance between coverage and efficiency (32k-100k common)

```python
from typing import List, Dict
import random
from collections import Counter

class TokenizerTrainingDataSelector:
    """Select representative sample for tokenizer training."""

    def __init__(
        self,
        target_size_mb: float = 1000,  # 1GB
        sample_method: str = 'stratified',  # 'stratified' or 'random'
    ):
        self.target_size_bytes = int(target_size_mb * 1024 * 1024)
        self.sample_method = sample_method

    def select_sample(
        self,
        sources: Dict[str, List[str]],
        source_weights: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Select representative sample for tokenizer training.

        Args:
            sources: Dict mapping source names to document lists
            source_weights: Optional weights for each source

        Returns:
            List of sampled documents
        """
        if source_weights is None:
            # Use proportional weights
            total_docs = sum(len(docs) for docs in sources.values())
            source_weights = {
                name: len(docs) / total_docs
                for name, docs in sources.items()
            }

        # Normalize weights
        total_weight = sum(source_weights.values())
        source_weights = {
            name: weight / total_weight
            for name, weight in source_weights.items()
        }

        # Calculate target size per source
        target_sizes = {
            name: int(self.target_size_bytes * weight)
            for name, weight in source_weights.items()
        }

        selected = []

        for source_name, target_size in target_sizes.items():
            docs = sources[source_name].copy()
            random.shuffle(docs)

            current_size = 0
            source_selected = []

            for doc in docs:
                doc_size = len(doc.encode('utf-8'))
                if current_size + doc_size > target_size:
                    break
                source_selected.append(doc)
                current_size += doc_size

            selected.extend(source_selected)
            print(f"  {source_name}: {len(source_selected):,} docs, "
                  f"{current_size / 1024 / 1024:.1f} MB")

        random.shuffle(selected)
        return selected

    def analyze_sample(self, documents: List[str]) -> Dict[str, any]:
        """Analyze tokenizer training sample."""
        total_size = sum(len(doc.encode('utf-8')) for doc in documents)
        total_chars = sum(len(doc) for doc in documents)

        # Character distribution
        char_counts = Counter()
        for doc in documents:
            char_counts.update(doc)

        return {
            'num_documents': len(documents),
            'total_size_mb': total_size / 1024 / 1024,
            'total_characters': total_chars,
            'unique_characters': len(char_counts),
            'most_common_chars': char_counts.most_common(20),
        }


# Example usage
print("\n=== Tokenizer Training Data Selection ===\n")

# Create example sources
tokenizer_sources = {
    'english_web': ['english doc ' + str(i) + ' with some content' for i in range(10000)],
    'code': ['def func' + str(i) + '(): pass' for i in range(5000)],
    'multilingual': ['texto en español ' + str(i) for i in range(3000)],
}

selector = TokenizerTrainingDataSelector(target_size_mb=1.0)  # Small for demo

print("Selecting tokenizer training sample:")
sample = selector.select_sample(
    tokenizer_sources,
    source_weights={'english_web': 0.6, 'code': 0.3, 'multilingual': 0.1}
)

print(f"\nTotal sample: {len(sample):,} documents")

analysis = selector.analyze_sample(sample)
print(f"\nSample analysis:")
print(f"  Total size: {analysis['total_size_mb']:.2f} MB")
print(f"  Total characters: {analysis['total_characters']:,}")
print(f"  Unique characters: {analysis['unique_characters']}")
```

### Vocabulary Coverage Analysis

After training a tokenizer, analyze coverage on your training data:

```python
from typing import List, Dict
from collections import defaultdict

class VocabularyAnalyzer:
    """Analyze tokenizer vocabulary coverage."""

    def __init__(self, tokenizer):
        """
        Initialize analyzer.

        Args:
            tokenizer: Trained tokenizer with encode() method
        """
        self.tokenizer = tokenizer

    def analyze_coverage(
        self,
        documents: List[str],
        max_docs: int = 1000,
    ) -> Dict[str, any]:
        """
        Analyze vocabulary coverage on documents.

        Returns:
            Dictionary with coverage statistics
        """
        token_counts = defaultdict(int)
        total_tokens = 0
        unk_tokens = 0

        for doc in documents[:max_docs]:
            # Tokenize
            tokens = self.tokenizer.encode(doc)

            for token_id in tokens:
                token_counts[token_id] += 1
                total_tokens += 1

                # Check if unknown token (depends on tokenizer)
                # Most tokenizers use specific IDs for UNK
                if token_id == self.tokenizer.unk_token_id:
                    unk_tokens += 1

        # Calculate statistics
        unique_tokens = len(token_counts)
        unk_rate = unk_tokens / max(total_tokens, 1)

        # Token frequency distribution
        freq_sorted = sorted(token_counts.values(), reverse=True)
        top10_coverage = sum(freq_sorted[:10]) / max(total_tokens, 1)
        top100_coverage = sum(freq_sorted[:100]) / max(total_tokens, 1)
        top1000_coverage = sum(freq_sorted[:1000]) / max(total_tokens, 1)

        return {
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'unk_rate': unk_rate,
            'top10_coverage': top10_coverage,
            'top100_coverage': top100_coverage,
            'top1000_coverage': top1000_coverage,
        }


# Example with dummy tokenizer
class DummyTokenizer:
    def __init__(self):
        self.unk_token_id = 0
        self.vocab = {'[UNK]': 0, 'the': 1, 'a': 2, 'is': 3}
        self.next_id = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        tokens = []
        for word in text.lower().split():
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.next_id += 1
            tokens.append(self.vocab[word])
        return tokens

print("\n=== Vocabulary Coverage Analysis ===\n")

tokenizer = DummyTokenizer()
test_docs = [
    "the quick brown fox jumps over the lazy dog",
    "a neural network is a computational model",
    "the transformer architecture is revolutionary",
] * 10

analyzer = VocabularyAnalyzer(tokenizer)
stats = analyzer.analyze_coverage(test_docs)

print("Coverage statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2%}")
    else:
        print(f"  {key}: {value:,}")
```

## PII Removal and Safety Filtering

Remove personally identifiable information (PII) and harmful content from training data.

### PII Detection and Removal

Common PII types:
- Email addresses
- Phone numbers
- IP addresses
- Social Security Numbers (SSN)
- Credit card numbers
- Addresses
- Names (harder - requires NER)

```python
import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

@dataclass
class PIIMatch:
    """Represents a detected PII instance."""
    pii_type: str
    start: int
    end: int
    text: str

class PIIDetector:
    """Detect and remove PII from text."""

    def __init__(self):
        # Compile regex patterns for different PII types
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            # URL pattern (can contain usernames/tokens)
            'url_with_credentials': re.compile(
                r'https?://[^:]+:[^@]+@[^\s]+'
            ),
        }

        # Common PII keywords (for context-based detection)
        self.pii_keywords = {
            'ssn', 'social security', 'credit card', 'password',
            'passport', "driver's license", 'license plate',
        }

    def detect(self, text: str) -> List[PIIMatch]:
        """Detect PII in text."""
        matches = []

        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                ))

        return matches

    def remove(
        self,
        text: str,
        replacement: str = '[REDACTED]',
        min_matches: int = 1,
    ) -> Tuple[str, List[PIIMatch]]:
        """
        Remove PII from text.

        Args:
            text: Input text
            replacement: Replacement string for PII
            min_matches: Minimum matches to trigger redaction

        Returns:
            (cleaned_text, matches)
        """
        matches = self.detect(text)

        if len(matches) < min_matches:
            return text, matches

        # Sort matches by position (reverse order for safe replacement)
        matches_sorted = sorted(matches, key=lambda m: m.start, reverse=True)

        cleaned = text
        for match in matches_sorted:
            cleaned = (
                cleaned[:match.start] +
                replacement +
                cleaned[match.end:]
            )

        return cleaned, matches

    def should_filter(
        self,
        text: str,
        max_pii_count: int = 5,
        max_pii_density: float = 0.01,  # PII matches per 100 chars
    ) -> Tuple[bool, str]:
        """
        Determine if document should be filtered out due to excessive PII.

        Returns:
            (should_filter, reason)
        """
        matches = self.detect(text)

        if len(matches) > max_pii_count:
            return True, f"Too many PII instances ({len(matches)})"

        if len(text) > 0:
            density = len(matches) / (len(text) / 100)
            if density > max_pii_density:
                return True, f"PII density too high ({density:.3f})"

        return False, ""


# Example usage
print("=== PII Detection and Removal ===\n")

pii_detector = PIIDetector()

test_texts = [
    "Contact me at john.doe@example.com or call 555-123-4567.",
    "My SSN is 123-45-6789 and credit card is 1234-5678-9012-3456.",
    "Visit https://user:password@secret-site.com for more info.",
    "The server IP is 192.168.1.1 and the code is available on GitHub.",
]

for i, text in enumerate(test_texts, 1):
    print(f"Example {i}:")
    print(f"  Original: {text}")

    # Detect PII
    matches = pii_detector.detect(text)
    if matches:
        print(f"  PII found: {[m.pii_type for m in matches]}")

    # Remove PII
    cleaned, _ = pii_detector.remove(text)
    print(f"  Cleaned: {cleaned}")

    # Check if should filter
    should_filter, reason = pii_detector.should_filter(text, max_pii_count=2)
    if should_filter:
        print(f"  FILTER: {reason}")

    print()
```

### Content Safety Filtering

Filter harmful content including:
- Hate speech
- Violence
- Adult content
- Toxicity

For production systems, use specialized tools:
- **[Perspective API](https://perspectiveapi.com/)**: Google's toxicity detection
- **OpenAI Moderation API**: Multi-category content moderation
- **Custom classifiers**: Train on domain-specific harmful content

```python
from typing import List, Dict, Set
import re

class SafetyFilter:
    """Filter potentially harmful content."""

    def __init__(self):
        # Blocklists (in production, use comprehensive lists)
        # These are heavily simplified examples
        self.profanity_words = {
            # Add actual profanity list
            'badword1', 'badword2',
        }

        self.hate_speech_patterns = [
            # Add actual hate speech patterns
            # These would be carefully curated regex patterns
        ]

        # Violence indicators
        self.violence_keywords = {
            'kill', 'murder', 'weapon', 'blood', 'violence',
            # In practice, need context-aware detection
        }

    def detect_profanity(self, text: str) -> int:
        """Count profanity instances."""
        words = set(text.lower().split())
        return len(words & self.profanity_words)

    def detect_violence(self, text: str) -> float:
        """Compute violence score based on keyword density."""
        words = text.lower().split()
        if not words:
            return 0.0

        violence_count = sum(1 for word in words if word in self.violence_keywords)
        return violence_count / len(words)

    def should_filter(
        self,
        text: str,
        max_profanity: int = 3,
        max_violence_score: float = 0.05,
    ) -> Tuple[bool, List[str]]:
        """
        Determine if content should be filtered.

        Returns:
            (should_filter, reasons)
        """
        reasons = []

        # Check profanity
        profanity_count = self.detect_profanity(text)
        if profanity_count > max_profanity:
            reasons.append(f"Excessive profanity ({profanity_count} instances)")

        # Check violence
        violence_score = self.detect_violence(text)
        if violence_score > max_violence_score:
            reasons.append(f"Violence score too high ({violence_score:.3f})")

        return len(reasons) > 0, reasons


# Example usage
print("=== Safety Filtering Example ===\n")

safety_filter = SafetyFilter()

test_cases = [
    "This is a normal document about machine learning and neural networks.",
    "This text contains repeated mentions of kill, murder, weapon, and violence.",
]

for i, text in enumerate(test_cases, 1):
    should_filter, reasons = safety_filter.should_filter(text)
    print(f"Example {i}: {'FILTER' if should_filter else 'KEEP'}")
    if reasons:
        print(f"  Reasons: {', '.join(reasons)}")
    print()
```

**Important Note**: Safety filtering is nuanced and context-dependent. In production:
1. Use established APIs or well-tested libraries
2. Consider context (educational content about violence vs. promoting violence)
3. Have human review for edge cases
4. Document filtering decisions for transparency
5. Consider multilingual content
6. Regular audits and updates

## Complete Pipeline Implementation

Putting it all together into a complete data curation pipeline:

```python
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
import json

@dataclass
class Document:
    """Represents a document with metadata."""
    text: str
    url: Optional[str] = None
    source: Optional[str] = None
    quality_score: float = 0.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DataCurationPipeline:
    """Complete data curation pipeline."""

    def __init__(
        self,
        # Component configurations
        enable_exact_dedup: bool = True,
        enable_fuzzy_dedup: bool = True,
        fuzzy_dedup_threshold: float = 0.8,
        enable_pii_removal: bool = True,
        enable_safety_filter: bool = True,
        min_quality_score: float = 0.5,
    ):
        # Initialize components
        self.text_extractor = TextExtractor()
        self.basic_filter = BasicFilter()

        if enable_exact_dedup:
            self.exact_dedup = ExactDeduplicator()
        else:
            self.exact_dedup = None

        if enable_fuzzy_dedup:
            self.fuzzy_dedup = FuzzyDeduplicator(threshold=fuzzy_dedup_threshold)
        else:
            self.fuzzy_dedup = None

        if enable_pii_removal:
            self.pii_detector = PIIDetector()
        else:
            self.pii_detector = None

        if enable_safety_filter:
            self.safety_filter = SafetyFilter()
        else:
            self.safety_filter = None

        self.heuristic_scorer = HeuristicScorer()
        self.min_quality_score = min_quality_score

        # Statistics
        self.stats = {
            'total_processed': 0,
            'passed_basic_filter': 0,
            'passed_exact_dedup': 0,
            'passed_fuzzy_dedup': 0,
            'passed_pii_filter': 0,
            'passed_safety_filter': 0,
            'passed_quality_filter': 0,
            'final_output': 0,
        }

    def process_html(self, html: str, url: Optional[str] = None) -> Optional[Document]:
        """Process HTML document through pipeline."""
        self.stats['total_processed'] += 1

        # 1. Extract text
        extracted = self.text_extractor.extract(html, url)
        text = extracted['text']

        # 2. Basic filtering
        basic_result = self.basic_filter.filter(text)
        if not basic_result['passed']:
            return None
        self.stats['passed_basic_filter'] += 1

        # 3. Exact deduplication
        if self.exact_dedup and self.exact_dedup.is_duplicate(text):
            return None
        self.stats['passed_exact_dedup'] += 1

        # 4. Fuzzy deduplication
        if self.fuzzy_dedup:
            _, is_dup = self.fuzzy_dedup.add_document(text)
            if is_dup:
                return None
        self.stats['passed_fuzzy_dedup'] += 1

        # 5. PII removal
        if self.pii_detector:
            should_filter, reason = self.pii_detector.should_filter(text)
            if should_filter:
                return None
            # Remove PII but keep document
            text, pii_matches = self.pii_detector.remove(text)
        self.stats['passed_pii_filter'] += 1

        # 6. Safety filtering
        if self.safety_filter:
            should_filter, reasons = self.safety_filter.should_filter(text)
            if should_filter:
                return None
        self.stats['passed_safety_filter'] += 1

        # 7. Quality scoring
        quality_score = self.heuristic_scorer.compute_quality_score(text)
        if quality_score < self.min_quality_score:
            return None
        self.stats['passed_quality_filter'] += 1

        # Create document
        doc = Document(
            text=text,
            url=url,
            source=extracted.get('domain'),
            quality_score=quality_score,
            metadata={
                'title': extracted.get('title'),
                'length': len(text),
            }
        )

        self.stats['final_output'] += 1
        return doc

    def process_batch(
        self,
        html_documents: List[tuple[str, Optional[str]]],  # (html, url) pairs
    ) -> List[Document]:
        """Process batch of HTML documents."""
        results = []

        for html, url in html_documents:
            doc = self.process_html(html, url)
            if doc is not None:
                results.append(doc)

        return results

    def get_stats(self) -> Dict[str, any]:
        """Get pipeline statistics."""
        if self.stats['total_processed'] == 0:
            return self.stats

        total = self.stats['total_processed']
        return {
            **self.stats,
            'pass_rates': {
                'basic_filter': self.stats['passed_basic_filter'] / total,
                'exact_dedup': self.stats['passed_exact_dedup'] / total,
                'fuzzy_dedup': self.stats['passed_fuzzy_dedup'] / total,
                'pii_filter': self.stats['passed_pii_filter'] / total,
                'safety_filter': self.stats['passed_safety_filter'] / total,
                'quality_filter': self.stats['passed_quality_filter'] / total,
                'final_output': self.stats['final_output'] / total,
            }
        }


# Example usage
print("=== Complete Pipeline Example ===\n")

pipeline = DataCurationPipeline(
    enable_fuzzy_dedup=True,
    min_quality_score=0.3,
)

# Example HTML documents
html_docs = [
    ("""<html><body><article>
        <h1>Introduction to Machine Learning</h1>
        <p>Machine learning is a subset of artificial intelligence that focuses on
        building systems that learn from data. Neural networks are a key component.</p>
        <p>Applications include computer vision, natural language processing, and
        recommendation systems.</p>
    </article></body></html>""", "https://example.com/ml-intro"),

    ("""<html><body>
        <p>CLICK HERE!!! BUY NOW!!! AMAZING DEALS!!!</p>
        <p>Contact: spam@example.com or 555-0000</p>
    </body></html>""", "https://spam-site.com"),

    ("""<html><body><article>
        <h1>Deep Learning Fundamentals</h1>
        <p>Deep learning uses neural networks with multiple layers to learn
        hierarchical representations of data. Convolutional networks excel at
        image tasks while transformers dominate natural language processing.</p>
    </article></body></html>""", "https://example.com/dl-fundamentals"),
]

# Process documents
results = pipeline.process_batch(html_docs)

print(f"Processed {len(html_docs)} documents")
print(f"Output: {len(results)} documents\n")

for i, doc in enumerate(results, 1):
    print(f"Document {i}:")
    print(f"  URL: {doc.url}")
    print(f"  Quality: {doc.quality_score:.3f}")
    print(f"  Length: {doc.metadata['length']} chars")
    print(f"  Text preview: {doc.text[:100]}...")
    print()

# Print statistics
print("Pipeline Statistics:")
stats = pipeline.get_stats()
print(f"  Total processed: {stats['total_processed']}")
print(f"  Final output: {stats['final_output']}")
print("\nPass rates:")
for stage, rate in stats['pass_rates'].items():
    print(f"  {stage}: {rate:.1%}")
```

## Best Practices Summary

1. **Start with quality over quantity**: Better to have 100B high-quality tokens than 1T noisy tokens
2. **Aggressive deduplication**: Both exact and fuzzy deduplication significantly improve models
3. **Multiple filtering stages**: Combine heuristic, perplexity, and classifier-based filtering
4. **PII removal is critical**: Both for privacy and to reduce memorization
5. **Data mixing matters**: Carefully balance different data sources
6. **Document everything**: Keep detailed logs of filtering decisions for reproducibility
7. **Iterative improvement**: Analyze model failures and improve data filters accordingly
8. **Contamination awareness**: Track test set overlaps (see [Chapter 32: Evaluation and Benchmarks](32-evaluation-benchmarks.md))

## Exercises

### Exercise 1: Build a Web Scraper

Implement a web scraper that downloads pages from a list of URLs and processes them through the extraction pipeline.

**Requirements**:
- Use `requests` and `BeautifulSoup`
- Handle errors gracefully (timeouts, 404s, etc.)
- Respect robots.txt
- Add rate limiting

### Exercise 2: MinHash Performance Analysis

Compare exact vs. approximate deduplication:
1. Generate 10,000 synthetic documents with varying similarity
2. Measure precision/recall of MinHash vs. exact matching
3. Analyze the speed/quality tradeoff with different signature sizes

### Exercise 3: Quality Classifier Training

Train a quality classifier:
1. Download examples from Wikipedia (high quality) and a random web corpus (mixed quality)
2. Label a dataset of 10,000 examples
3. Train a classifier using the QualityClassifier architecture
4. Evaluate on a held-out test set
5. Analyze what features the model learns

### Exercise 4: Data Mixing Experiment

Implement a training experiment:
1. Create 3 data sources with different characteristics
2. Train small language models with different mixing ratios
3. Evaluate perplexity on held-out data from each source
4. Find optimal mixing ratios

### Exercise 5: PII Detection Improvement

Extend the PIIDetector:
1. Add detection for additional PII types (addresses, names using NER)
2. Implement context-aware filtering (e.g., "My name is [NAME]" vs. product names)
3. Add privacy-preserving alternatives (replace specific PII with generic placeholders)
4. Measure false positive and false negative rates

### Exercise 6: End-to-End Pipeline

Build a complete pipeline:
1. Scrape 1,000 web pages from diverse domains
2. Process through the complete curation pipeline
3. Generate statistics at each stage
4. Analyze what types of content pass/fail each filter
5. Create visualizations of the data quality distribution

## References and Further Reading

### Papers

1. **Gopher** (Rae et al., 2021): [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446)
   - Detailed data curation pipeline for a 280B parameter model

2. **The Pile** (Gao et al., 2020): [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)
   - Comprehensive multi-source dataset

3. **RefinedWeb** (Penedo et al., 2023): [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)
   - State-of-the-art web data filtering

4. **Dolma** (Soldaini et al., 2024): [Dolma: An Open Corpus of 3 Trillion Tokens](https://arxiv.org/abs/2402.00159)
   - Open dataset with extensive documentation

5. **C4** (Raffel et al., 2019): [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)
   - CommonCrawl filtering methodology

6. **Deduplication** (Lee et al., 2021): [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
   - Analysis of deduplication impact

7. **Data Quality** (Longpre et al., 2023): [The Data Provenance Initiative](https://arxiv.org/abs/2310.16787)
   - Analysis of training data quality and documentation

### Datasets

- [CommonCrawl](https://commoncrawl.org/): Petabyte-scale web archive
- [The Pile](https://pile.eleuther.ai/): 825GB diverse dataset
- [RedPajama](https://github.com/togethercomputer/RedPajama-Data): Open LLaMA replication
- [Dolma](https://huggingface.co/datasets/allenai/dolma): 3T token open corpus
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb): 15T token filtered CommonCrawl

### Tools

- [trafilatura](https://github.com/adbar/trafilatura): Web scraping and text extraction
- [datasketch](https://github.com/ekzhu/datasketch): MinHash and LSH
- [fastText](https://fasttext.cc/): Language identification and text classification
- [Perspective API](https://perspectiveapi.com/): Toxicity detection

---

**Next Chapter**: [Language Model Training](15-lm-training.md) - Learn how to train language models on curated data

**Previous Chapter**: [Other Efficient Attention Variants](13-efficient-attention.md) - Efficient attention mechanisms
