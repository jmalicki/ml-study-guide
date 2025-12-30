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
4. [Data Contamination and Test Set Leakage](#data-contamination-and-test-set-leakage)
5. [Data Mixing and Curriculum Learning](#data-mixing-and-curriculum-learning)
6. [Tokenizer Training Data Considerations](#tokenizer-training-data-considerations)
7. [PII Removal and Safety Filtering](#pii-removal-and-safety-filtering)
8. [Scaling and Distributed Processing](#scaling-and-distributed-processing)
9. [Legal and Ethical Considerations](#legal-and-ethical-considerations)
10. [Complete Pipeline Implementation](#complete-pipeline-implementation)
11. [Exercises](#exercises)

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

**The Problem**: Web crawl data like CommonCrawl comes in raw HTML format, which contains a mixture of actual content and non-content elements (navigation menus, advertisements, JavaScript code, CSS styling, etc.). For LLM training, we need pure text that represents the actual information on the page.

**Why This Matters**: Training on raw HTML would waste model capacity learning boilerplate patterns instead of natural language. Modern LLMs trained on trillions of tokens need clean text to maximize the value of each training token. The [C4 dataset paper](https://arxiv.org/abs/1910.10683) showed that aggressive HTML cleaning significantly improved downstream performance.

**Theoretical Justification**: The goal is to extract the **main content** while removing **boilerplate**. This is essentially a signal-to-noise problem. Research on content extraction includes:
- **DOM-based methods**: Identify semantic HTML5 tags (`<article>`, `<main>`) that typically contain primary content
- **Density-based methods**: Content regions have higher text density than navigation/ads
- **Template detection**: Identify and remove repeated structures across pages

**How This Relates to Alternatives**:
- **Simple tag stripping** (e.g., `html2text`): Fast but keeps navigation, ads, etc.
- **Boilerplate detection** (e.g., `jusText`, `Readability`): More sophisticated but slower
- **Machine learning approaches**: Can be most accurate but require training data
- **Our approach**: Balance between simplicity and effectiveness using semantic HTML tags

**Key Insights**:
1. **Prioritize semantic tags**: Modern HTML5's `<article>`, `<main>` tags are strong signals
2. **Remove known noise tags**: Scripts, styles, navigation are never useful for LLM training
3. **Preserve structure**: Keep paragraph breaks - they're meaningful for coherence
4. **Normalize whitespace**: HTML's whitespace is presentational, not semantic

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

**The Problem**: Even after extracting clean text from HTML, web crawl data contains enormous amounts of low-quality content: spam pages, error messages, navigation-heavy pages, cookie notices, auto-generated content, and more. This "noise" can degrade model performance if included in training data.

**Why This Matters**: The [Gopher paper](https://arxiv.org/abs/2112.11446) found that training on carefully filtered data substantially outperformed training on raw web crawls of the same size. Quality filtering allows models to learn from better examples in fewer training steps. Since compute is expensive, it's more efficient to filter aggressively than to train on noisy data.

**Theoretical Justification**: These filters are based on **distributional assumptions about natural language**:
- **Length constraints**: Natural documents have reasonable length ranges. Too short suggests fragments or navigation; too long suggests concatenated pages or auto-generated content
- **Word statistics**: Natural language has characteristic word length distributions. Abnormal distributions indicate generated content, URLs, or code
- **Character composition**: Real text is mostly alphabetic. High symbol/number ratios suggest technical content, spam, or errors
- **Case patterns**: Excessive uppercase indicates shouting, headlines, or low-quality content
- **Structural patterns**: Excessive bullets/lists suggest navigation menus rather than prose

**How This Relates to Alternatives**:
- **No filtering**: Simple but includes massive amounts of noise, hurting model quality
- **Manual curation**: Highest quality but completely infeasible at scale
- **ML-based filtering**: More accurate (discussed later) but requires labeled data and is computationally expensive
- **Rule-based filtering (our approach)**: Fast, interpretable, requires no training data, catches most low-quality content

**Key Insights**:
1. **Multiple weak signals are stronger than one**: No single heuristic is perfect, but combining many catches most low-quality content
2. **Conservative thresholds**: False negatives (keeping some bad data) are usually acceptable, but false positives (removing good data) are costly
3. **Domain-specific tuning**: Thresholds should be adjusted based on data source (news vs. social media vs. technical docs)
4. **Fast pre-filtering**: These cheap heuristics can quickly reduce dataset size 10-100x before expensive filtering

```python
import re
from typing import List, Dict, Optional, Any
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

    def filter(self, text: str) -> Dict[str, Any]:
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

**The Problem**: Web crawl data contains text in hundreds of languages. For most LLM training scenarios, we want to focus on specific languages (often primarily English, with some multilingual content). Training on random languages dilutes the model's capacity and makes the training data less efficient.

**Why This Matters**: Language composition directly affects model capabilities. The [LLaMA 2 paper](https://arxiv.org/abs/2307.09288) used 89.7% English data, while multilingual models like [BLOOM](https://arxiv.org/abs/2211.05100) carefully balanced 46 languages. Incorrect language filtering can accidentally exclude important multilingual content or waste capacity on unintended languages.

**Theoretical Justification**: Language identification exploits the fact that different languages have **distinctive character and n-gram distributions**:
- **Character-level**: Different scripts (Latin, Cyrillic, Arabic, CJK) have non-overlapping Unicode ranges
- **Byte-level**: Character encoding patterns differ by language
- **N-gram level**: Languages have distinctive sequences (e.g., "th" is common in English, rare in Spanish)
- **Statistical models**: Modern detectors use neural networks or compressed models trained on multilingual text

**How This Relates to Alternatives**:
- **Unicode script detection (our simple approach)**: Very fast, works for script-level detection, but can't distinguish languages with same script (English vs. French)
- **langdetect**: Python port of Google's detector, uses character n-grams, good accuracy, moderate speed
- **fastText language ID**: Facebook's neural model, extremely fast (milliseconds for documents), very accurate, supports 176 languages
- **pycld2/pycld3**: Chrome's Compact Language Detector, highly optimized, good for production
- **GPT-based detection**: Use an LLM to identify language - accurate but slow and expensive

**Key Insights**:
1. **Script detection is cheapest**: For many use cases, detecting script (Latin vs. Cyrillic vs. CJK) is sufficient
2. **Context length matters**: Language detectors need enough text (50-100+ characters) for reliable detection
3. **Code-switching**: Real web data contains mixed languages within documents - handle with confidence thresholds
4. **False positives**: Short texts can be ambiguous ("OK", "STOP") - use minimum length requirements

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

**The Problem**: Web crawl data contains massive amounts of exact duplicates - the same document appears multiple times across different URLs, pages are mirrored, content is syndicated, and scrapers revisit the same pages. Without deduplication, models waste capacity memorizing these repeated examples instead of learning from diverse data.

**Why This Matters**: The [GPT-3 paper](https://arxiv.org/abs/2005.14165) found that exact deduplication was essential for preventing models from overfitting to repeated content. The [Scaling Laws paper](https://arxiv.org/abs/2001.08361) showed that diverse, non-repeated data is much more valuable than redundant data. Training on duplicates causes:
- **Memorization**: Models overfit to frequently-seen sequences
- **Reduced effective dataset size**: 1000 copies of a document ≠ 1000 unique documents
- **Privacy concerns**: Repeated PII amplifies privacy risks
- **Biased representations**: Overrepresented content dominates model behavior

**Theoretical Justification**: Exact deduplication is a **set membership problem** solved efficiently with hashing:
- **Cryptographic hashes** (SHA-256, MD5) map arbitrary text to fixed-size fingerprints
- **Collision resistance**: Different documents (with high probability) produce different hashes
- **Constant-time lookup**: Hash table membership tests are O(1)
- **Memory efficient**: Store hashes (32 bytes) instead of full documents

**How This Relates to Alternatives**:
- **Naive comparison**: Compare every document to every other - O(n²) time, infeasible at scale
- **Exact string matching**: Same complexity as naive comparison
- **Hash-based deduplication (our approach)**: O(n) time and space, simple and reliable
- **Near-duplicate detection**: More sophisticated (see next section) but not needed for exact matches
- **Bloom filters**: More memory-efficient but allows false positives (acceptable for some use cases)

**Key Insights**:
1. **Normalize before hashing**: Whitespace differences shouldn't prevent duplicate detection
2. **Hash quality matters**: Cryptographic hashes prevent accidental collisions
3. **Memory is the bottleneck**: Storing billions of hashes requires gigabytes of RAM
4. **This catches only exact duplicates**: Near-duplicates (slightly modified versions) need different approaches

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

### LSH Parameter Selection

Choosing the right LSH parameters (number of bands and rows per band) is crucial for balancing precision and recall in duplicate detection.

**Theory**: For a similarity threshold $s$ and LSH configuration with $b$ bands and $r$ rows per band, the probability that two documents with Jaccard similarity $s$ become candidates is:

$$P(\text{candidate}|s) = 1 - (1 - s^r)^b$$

This forms an **S-curve**: low similarity pairs have low probability of being candidates, and high similarity pairs have high probability.

**Key insights**:
- **Similarity threshold** $t$: We want $P(\text{candidate}|t) \approx 0.5$ (the "knee" of the S-curve)
- At the threshold: $t^r \approx 0.5 \Rightarrow r \approx \frac{\log(0.5)}{\log(t)}$
- Number of bands: $b = \frac{\text{num\_perm}}{r}$

```python
import math
import numpy as np
import matplotlib.pyplot as plt

def lsh_probability(similarity: float, num_bands: int, rows_per_band: int) -> float:
    """
    Compute probability that documents become candidates.

    Args:
        similarity: Jaccard similarity between documents
        num_bands: Number of LSH bands
        rows_per_band: Rows per band

    Returns:
        Probability of being candidates [0, 1]
    """
    return 1 - (1 - similarity ** rows_per_band) ** num_bands


def choose_lsh_params(
    threshold: float,
    num_perm: int,
    optimize_for: str = 'balance',  # 'precision' or 'recall' or 'balance'
) -> Tuple[int, int]:
    """
    Choose optimal LSH parameters for a similarity threshold.

    Args:
        threshold: Desired similarity threshold (e.g., 0.8)
        num_perm: Total number of permutations (signature size)
        optimize_for: Whether to optimize for precision, recall, or balance

    Returns:
        (num_bands, rows_per_band)
    """
    # Rule of thumb: Set r such that threshold^r ≈ 0.5
    # This puts the threshold at the "knee" of the S-curve

    if optimize_for == 'balance':
        target_prob = 0.5
    elif optimize_for == 'precision':
        target_prob = 0.3  # Steeper curve, fewer false positives
    elif optimize_for == 'recall':
        target_prob = 0.7  # Shallower curve, fewer false negatives
    else:
        raise ValueError(f"Unknown optimization target: {optimize_for}")

    # Solve for r: threshold^r = target_prob
    optimal_r = int(math.log(target_prob) / math.log(threshold))
    optimal_r = max(1, optimal_r)  # At least 1

    # Compute b
    optimal_b = num_perm // optimal_r

    # Adjust to use all permutations
    if optimal_b * optimal_r != num_perm:
        # Try to find divisors of num_perm close to our target
        best_b, best_r = optimal_b, optimal_r
        best_diff = abs(lsh_probability(threshold, best_b, best_r) - target_prob)

        for r in range(1, num_perm + 1):
            if num_perm % r == 0:
                b = num_perm // r
                prob = lsh_probability(threshold, b, r)
                diff = abs(prob - target_prob)

                if diff < best_diff:
                    best_b, best_r = b, r
                    best_diff = diff

        optimal_b, optimal_r = best_b, best_r

    return optimal_b, optimal_r


def plot_lsh_curves(num_perm: int = 128):
    """Plot LSH probability curves for different parameter settings."""
    similarities = np.linspace(0, 1, 100)

    # Different parameter choices
    configs = [
        (16, 8, "16 bands, 8 rows (threshold ~0.8)"),
        (32, 4, "32 bands, 4 rows (threshold ~0.84)"),
        (8, 16, "8 bands, 16 rows (threshold ~0.73)"),
    ]

    plt.figure(figsize=(10, 6))

    for num_bands, rows_per_band, label in configs:
        probs = [
            lsh_probability(s, num_bands, rows_per_band)
            for s in similarities
        ]
        plt.plot(similarities, probs, label=label, linewidth=2)

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='P=0.5')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Probability of Becoming Candidate')
    plt.title('LSH S-Curves for Different Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Note: In practice, you'd save or show this plot
    # plt.savefig('lsh_curves.png')
    print("LSH curve plot generated (would display in interactive environment)")


# Example usage
print("=== LSH Parameter Selection ===\n")

# For different similarity thresholds
thresholds = [0.7, 0.8, 0.9]
num_perm = 128

print("Optimal LSH parameters for different thresholds:\n")
for threshold in thresholds:
    num_bands, rows_per_band = choose_lsh_params(threshold, num_perm)

    # Compute actual probability at threshold
    prob = lsh_probability(threshold, num_bands, rows_per_band)

    print(f"Threshold: {threshold}")
    print(f"  Bands: {num_bands}, Rows/band: {rows_per_band}")
    print(f"  P(candidate | s={threshold}): {prob:.3f}")
    print(f"  Total signature size: {num_bands * rows_per_band}")

    # Show probabilities at different similarities
    print("  Probabilities:")
    for s in [threshold - 0.1, threshold, threshold + 0.1]:
        if 0 <= s <= 1:
            p = lsh_probability(s, num_bands, rows_per_band)
            print(f"    P(candidate | s={s:.1f}): {p:.3f}")
    print()

# Demonstrate impact of optimization target
print("Impact of optimization target (threshold=0.8):")
for target in ['precision', 'balance', 'recall']:
    num_bands, rows_per_band = choose_lsh_params(0.8, num_perm, optimize_for=target)
    prob = lsh_probability(0.8, num_bands, rows_per_band)

    print(f"\n{target.capitalize()}:")
    print(f"  Bands: {num_bands}, Rows/band: {rows_per_band}")
    print(f"  P(candidate | s=0.8): {prob:.3f}")

    # False positive rate (at s=0.5)
    fp_prob = lsh_probability(0.5, num_bands, rows_per_band)
    print(f"  P(candidate | s=0.5): {fp_prob:.3f} (lower is better for precision)")

# Plot curves (if in interactive environment)
try:
    plot_lsh_curves(num_perm)
except:
    pass
```

**Practical Guidelines**:

1. **Standard configuration** (threshold ~0.8):
   - Use `num_bands=16, rows_per_band=8` for 128 permutations
   - Good balance between precision and recall

2. **High precision** (few false positives):
   - Increase rows per band (steeper S-curve)
   - Example: `num_bands=32, rows_per_band=4` for threshold ~0.84

3. **High recall** (few false negatives):
   - Decrease rows per band (shallower S-curve)
   - Example: `num_bands=8, rows_per_band=16` for threshold ~0.73

4. **Memory vs. Accuracy trade-off**:
   - More bands = more memory (more hash tables)
   - More permutations = better accuracy but slower
   - Typical range: 64-256 permutations

**Note**: For production systems at scale, consider:
- **[datasketch](https://github.com/ekzhu/datasketch)**: Efficient MinHash/LSH implementation
- **Distributed processing**: Use Spark or Dask for large-scale deduplication
- **Disk-based storage**: Store signatures on disk for datasets larger than memory

## Quality Filtering and Scoring

Beyond basic filtering, we need sophisticated quality scoring to rank documents.

### Perplexity-Based Filtering

**The Problem**: Rule-based filters (character ratios, length constraints) can't distinguish subtle quality differences. We need a way to measure how "natural" or "coherent" text is - whether it resembles the kind of high-quality text we want the model to learn from.

**Why This Matters**: The [CCNet paper](https://arxiv.org/abs/1911.00359) (Facebook's CommonCrawl filtering) showed that perplexity-based filtering dramatically improved downstream task performance. The [Gopher paper](https://arxiv.org/abs/2112.11446) used perplexity filtering and found it essential for removing garbled, auto-generated, and low-quality text that passed rule-based filters. High-quality datasets like Wikipedia have much lower average perplexity than random web text.

**Theoretical Justification**: Perplexity measures **how surprising text is to a language model**:
- **Well-formed text**: Natural, grammatical text is predictable to a good language model → low perplexity
- **Garbled text**: Random characters, corrupted encoding, spam → high perplexity
- **Auto-generated text**: Often has unusual patterns that increase perplexity
- **Domain mismatch**: Technical jargon, specialized terminology → higher perplexity (but may still be valuable)

Use a language model to score text quality. High-quality text has lower perplexity under a well-trained LM.

**Perplexity** is defined as:

$$\text{PPL}(X) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i | x_{<i})\right)$$

where $N$ is the number of tokens and $P(x_i | x_{<i})$ is the model's predicted probability of token $x_i$ given context.

**How This Relates to Alternatives**:
- **Rule-based filtering**: Fast but misses subtle quality issues; perplexity catches what rules miss
- **Classifier-based filtering**: More targeted but requires labeled data (see next section)
- **Human annotation**: Most accurate but completely infeasible at scale
- **Perplexity filtering (our approach)**: Uses an already-trained LM to score quality without requiring labeled data

**Key Insights**:
1. **Model choice matters**: Use a high-quality LM trained on diverse, clean data (e.g., GPT-2, KenLM on Wikipedia)
2. **Sliding window for long texts**: Models have context limits; use overlapping windows for fair scoring
3. **Threshold tuning**: Optimal perplexity threshold depends on your model and target domain (typical: 500-1500)
4. **Computational cost**: This is expensive - use it after cheap filters reduce dataset size
5. **Not perfect**: Low perplexity can also indicate memorized/repeated content

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

**The Problem**: Perplexity is a general measure of coherence, but it doesn't capture domain-specific notions of "quality." We might want to filter for specific attributes: informative vs. promotional content, factual vs. opinion, well-sourced vs. unsourced, etc. Generic language models can't make these distinctions.

**Why This Matters**: The [RefinedWeb paper](https://arxiv.org/abs/2306.01116) (Falcon LLM's training data) showed that training a classifier on high-quality vs. low-quality examples, then using it to filter billions of documents, substantially improved model performance. The [C4 paper](https://arxiv.org/abs/1910.10683) used a classifier trained to distinguish Wikipedia/books from random web pages. This approach allows you to define "quality" based on your specific use case.

**Theoretical Justification**: Quality classification is a **supervised learning problem**:
- **Labeled data**: Collect examples of "high quality" (e.g., Wikipedia, curated sources) vs. "low quality" (random web scrapes)
- **Feature learning**: A neural network learns features that distinguish quality levels
- **Transfer learning**: The classifier generalizes to new documents
- **Binary or multi-class**: Can be simple binary (good/bad) or multi-level (excellent/good/medium/poor)

Train a classifier to predict document quality based on human-labeled examples.

The [RefinedWeb paper](https://arxiv.org/abs/2306.01116) used a fastText classifier trained on curated vs. random web data.

**How This Relates to Alternatives**:
- **Rule-based filters**: No learning, can't capture complex quality patterns
- **Perplexity filtering**: Measures coherence but not domain-specific quality
- **Classifier-based (our approach)**: Learns from examples, can capture nuanced quality concepts, but requires labeled data
- **LLM-as-judge**: Use GPT-4 to score quality - very accurate but extremely expensive at scale
- **Ensemble methods**: Combine multiple signals (rules + perplexity + classifier) for best results

**Key Insights**:
1. **Training data is critical**: Your classifier learns from examples - garbage in, garbage out
2. **Simple models can work**: FastText or small BiLSTM often sufficient; don't need large transformers
3. **Speed matters**: Must be fast enough to score billions of documents (milliseconds per document)
4. **Threshold tuning**: Choose score threshold based on your quality/quantity trade-off
5. **Domain-specific**: A classifier trained for news may not work for technical documentation

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

**The Problem**: While we have binary filters (pass/fail), we often want to **rank** documents by quality for curriculum learning or to keep only the top-k% of data. We need interpretable quality scores that correlate with document quality without requiring a trained model.

**Why This Matters**: The [Gopher paper](https://arxiv.org/abs/2112.11446) found that combining multiple heuristic signals into a quality score was highly effective. The [C4 dataset](https://arxiv.org/abs/1910.10683) used similar heuristics to rank and filter 156B tokens from 806B tokens of raw web data. These heuristics are fast (no model inference), interpretable (you can see why a document scored poorly), and customizable (adjust weights for your domain).

**Theoretical Justification**: Quality scoring combines **multiple weak signals into a strong signal**:
- **Ensemble approach**: Each heuristic captures one aspect of quality; combined they're more robust
- **Statistical regularities**: High-quality text (books, Wikipedia, news) has characteristic statistical patterns
- **Weighted combination**: Learn weights from a small labeled dataset or use domain knowledge
- **Rank correlation**: The goal is ranking accuracy, not absolute score accuracy

The [Gopher paper](https://arxiv.org/abs/2112.11446) and [C4](https://arxiv.org/abs/1910.10683) used various heuristic scores:

**How This Relates to Alternatives**:
- **Single heuristic**: Too weak, high false positive/negative rates
- **Binary filters**: Only give yes/no, not relative quality
- **ML-based scoring**: More accurate but requires training data and is slower
- **Heuristic scoring (our approach)**: Fast, interpretable, good enough for ranking
- **Hybrid approach**: Use heuristics for initial filtering, ML for fine-grained ranking

**Key Insights**:
1. **More signals = better**: Combine 10+ heuristics for robust scores
2. **Outlier detection**: Quality is often about detecting outliers (too many bullets, too much uppercase)
3. **Domain matters**: Heuristics for news articles differ from those for code documentation
4. **Weight tuning**: Use a small validation set to optimize heuristic weights
5. **Interpretability value**: Unlike neural models, you can debug why documents scored low

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

## Data Contamination and Test Set Leakage

One of the most critical concerns in modern LLM training is **data contamination**: when training data includes examples from evaluation benchmarks, artificially inflating performance metrics. This has become a major issue as models are trained on ever-larger web scrapes that may inadvertently include benchmark datasets.

### Why Contamination Matters

When a model has seen test examples during training, it can memorize answers rather than demonstrate true capability. This:
- **Inflates benchmark scores** without improving real-world performance
- **Makes model comparison difficult** when contamination levels differ
- **Undermines scientific progress** by providing misleading signals
- **Damages trust** in published results

Notable examples:
- GPT-3.5 showed suspiciously high performance on some benchmarks, leading to contamination investigations
- Several models have been found to have exact or near-exact matches to popular benchmark questions
- The community now demands contamination analysis in research papers

### N-gram Overlap Detection

The most common contamination detection method is **n-gram overlap**: computing the fraction of test example n-grams that appear in training data.

For a test example $x_{test}$ and training corpus $D_{train}$, the n-gram overlap is:

$$\text{Overlap}_n(x_{test}, D_{train}) = \frac{|\text{ngrams}_n(x_{test}) \cap \text{ngrams}_n(D_{train})|}{|\text{ngrams}_n(x_{test})|}$$

**Common practices**:
- Use 13-grams (balances specificity and coverage)
- Flag examples with >50% overlap as potentially contaminated
- Some studies use stricter thresholds (80% for high confidence)

```python
from typing import List, Set, Dict, Tuple
from collections import defaultdict
import hashlib

class ContaminationDetector:
    """Detect test set contamination in training data."""

    def __init__(self, n: int = 13):
        """
        Initialize contamination detector.

        Args:
            n: Size of n-grams to use (13 is standard)
        """
        self.n = n
        # Store n-gram hashes from training data
        self.train_ngram_hashes: Set[str] = set()
        self.train_doc_count = 0

    def _get_ngrams(self, text: str) -> Set[str]:
        """Extract word-level n-grams from text."""
        words = text.split()
        ngrams = set()

        for i in range(len(words) - self.n + 1):
            ngram = ' '.join(words[i:i + self.n])
            ngrams.add(ngram)

        return ngrams

    def _hash_ngram(self, ngram: str) -> str:
        """Hash an n-gram for efficient storage."""
        return hashlib.md5(ngram.encode('utf-8')).hexdigest()

    def add_train_document(self, text: str):
        """Add training document to contamination index."""
        ngrams = self._get_ngrams(text)

        for ngram in ngrams:
            ngram_hash = self._hash_ngram(ngram)
            self.train_ngram_hashes.add(ngram_hash)

        self.train_doc_count += 1

    def check_contamination(
        self,
        test_text: str,
        threshold: float = 0.5,
    ) -> Tuple[bool, float, Dict[str, any]]:
        """
        Check if test example is contaminated.

        Args:
            test_text: Test example text
            threshold: Overlap threshold for flagging contamination

        Returns:
            (is_contaminated, overlap_ratio, details)
        """
        test_ngrams = self._get_ngrams(test_text)

        if not test_ngrams:
            return False, 0.0, {'reason': 'No n-grams in test text'}

        # Count overlapping n-grams
        overlapping = 0
        for ngram in test_ngrams:
            ngram_hash = self._hash_ngram(ngram)
            if ngram_hash in self.train_ngram_hashes:
                overlapping += 1

        overlap_ratio = overlapping / len(test_ngrams)
        is_contaminated = overlap_ratio >= threshold

        details = {
            'total_ngrams': len(test_ngrams),
            'overlapping_ngrams': overlapping,
            'overlap_ratio': overlap_ratio,
            'threshold': threshold,
        }

        return is_contaminated, overlap_ratio, details

    def analyze_test_set(
        self,
        test_examples: List[str],
        threshold: float = 0.5,
    ) -> Dict[str, any]:
        """
        Analyze contamination across entire test set.

        Returns:
            Statistics about contamination levels
        """
        contaminated_count = 0
        overlap_ratios = []

        for test_text in test_examples:
            is_contaminated, overlap, _ = self.check_contamination(
                test_text, threshold
            )

            overlap_ratios.append(overlap)
            if is_contaminated:
                contaminated_count += 1

        return {
            'total_examples': len(test_examples),
            'contaminated_examples': contaminated_count,
            'contamination_rate': contaminated_count / max(len(test_examples), 1),
            'mean_overlap': sum(overlap_ratios) / max(len(overlap_ratios), 1),
            'max_overlap': max(overlap_ratios) if overlap_ratios else 0.0,
            'min_overlap': min(overlap_ratios) if overlap_ratios else 0.0,
        }


# Example usage
print("=== Contamination Detection Example ===\n")

detector = ContaminationDetector(n=5)  # Using 5-grams for demo (13 in practice)

# Simulate training corpus
training_docs = [
    "Machine learning is a method of data analysis that automates analytical model building.",
    "Neural networks are computing systems inspired by biological neural networks.",
    "Deep learning is part of a broader family of machine learning methods.",
    "Transformers use self-attention mechanisms to process sequential data.",
]

for doc in training_docs:
    detector.add_train_document(doc)

print(f"Indexed {detector.train_doc_count} training documents")
print(f"Total unique n-grams: {len(detector.train_ngram_hashes):,}\n")

# Test examples
test_examples = [
    # High contamination - nearly exact match
    "Machine learning is a method of data analysis that automates analytical model building using algorithms.",

    # Medium contamination - partial match
    "Neural networks and deep learning are important for machine learning methods.",

    # Low contamination - different content
    "Quantum computing leverages quantum mechanical phenomena for computation.",
]

print("Individual test example analysis:")
for i, test_text in enumerate(test_examples, 1):
    is_cont, overlap, details = detector.check_contamination(test_text, threshold=0.5)

    print(f"\nExample {i}:")
    print(f"  Text: {test_text[:70]}...")
    print(f"  Overlap: {overlap:.1%}")
    print(f"  Contaminated: {'YES' if is_cont else 'NO'}")
    print(f"  Details: {details['overlapping_ngrams']}/{details['total_ngrams']} n-grams overlap")

# Analyze full test set
print("\n" + "="*60)
print("Test set analysis:")
stats = detector.analyze_test_set(test_examples, threshold=0.5)
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2%}" if value < 1 else f"  {key}: {value:.3f}")
    else:
        print(f"  {key}: {value}")
```

### Substring Matching for Exact Contamination

For detecting exact contamination (copy-pasted benchmark examples), use substring matching:

```python
from typing import List, Dict, Set
import re

class ExactContaminationDetector:
    """Detect exact or near-exact benchmark contamination."""

    def __init__(self, min_match_length: int = 50):
        """
        Args:
            min_match_length: Minimum substring length to flag
        """
        self.min_match_length = min_match_length
        # Store training document substrings
        self.train_substrings: Set[str] = set()

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase and remove extra whitespace
        text = text.lower()
        text = ' '.join(text.split())
        # Remove punctuation for fuzzy matching
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def add_train_document(self, text: str):
        """Add training document substrings to index."""
        normalized = self.normalize_text(text)

        # Store all substrings of minimum length
        for i in range(len(normalized) - self.min_match_length + 1):
            substring = normalized[i:i + self.min_match_length]
            self.train_substrings.add(substring)

    def find_matches(self, test_text: str) -> List[Dict[str, any]]:
        """Find exact matches between test and training data."""
        normalized = self.normalize_text(test_text)
        matches = []

        # Check all substrings
        for i in range(len(normalized) - self.min_match_length + 1):
            substring = normalized[i:i + self.min_match_length]

            if substring in self.train_substrings:
                matches.append({
                    'start': i,
                    'length': self.min_match_length,
                    'text': substring,
                })

        return matches

    def check_contamination(
        self,
        test_text: str,
        max_matches: int = 5,
    ) -> Tuple[bool, List[Dict]]:
        """
        Check for exact contamination.

        Args:
            test_text: Test example
            max_matches: Maximum number of matches before flagging

        Returns:
            (is_contaminated, match_list)
        """
        matches = self.find_matches(test_text)
        is_contaminated = len(matches) > max_matches

        return is_contaminated, matches


# Example usage
print("\n=== Exact Contamination Detection ===\n")

exact_detector = ExactContaminationDetector(min_match_length=20)

# Add training data
train_texts = [
    "What is the capital of France? The capital of France is Paris.",
    "Who wrote Romeo and Juliet? William Shakespeare wrote Romeo and Juliet.",
]

for text in train_texts:
    exact_detector.add_train_document(text)

# Test examples
test_texts = [
    "What is the capital of France? (A) London (B) Paris (C) Berlin",  # Contaminated
    "What is the capital of Germany?",  # Not contaminated
]

for i, test_text in enumerate(test_texts, 1):
    is_cont, matches = exact_detector.check_contamination(test_text, max_matches=0)
    print(f"Test {i}: {test_text}")
    print(f"  Contaminated: {'YES' if is_cont else 'NO'}")
    print(f"  Matches found: {len(matches)}")
    if matches:
        print(f"  Sample match: '{matches[0]['text'][:40]}...'")
    print()
```

### Best Practices for Contamination Prevention

1. **Proactive Filtering**:
   - Maintain a list of known benchmark datasets
   - Filter URLs from benchmark hosting sites (e.g., huggingface.co/datasets)
   - Remove documents containing benchmark dataset names

2. **Post-Training Analysis**:
   - Check n-gram overlap with major benchmarks (MMLU, HellaSwag, etc.)
   - Report contamination rates in research papers
   - Discount scores on contaminated benchmarks

3. **Temporal Separation**:
   - Use benchmarks created after training data cutoff
   - Prefer newly released benchmarks for evaluation
   - Track benchmark release dates

4. **Dynamic Benchmarks**:
   - Use generative benchmarks that can't be memorized
   - Create private evaluation sets
   - Use contamination-resistant metrics

```python
from typing import List, Set, Dict
import re
from urllib.parse import urlparse

class BenchmarkFilter:
    """Filter out known benchmark datasets from training data."""

    def __init__(self):
        # Known benchmark hosting domains
        self.blocked_domains = {
            'huggingface.co/datasets',
            'github.com/openai/grade-school-math',
            'rajpurkar.github.io/SQuAD-explorer',
            'super.gluebenchmark.com',
            # Add more as needed
        }

        # Benchmark dataset names to filter
        self.benchmark_keywords = {
            'mmlu', 'hellaswag', 'winogrande', 'arc', 'truthfulqa',
            'gsm8k', 'math', 'humaneval', 'mbpp', 'squad', 'glue',
            'super_glue', 'boolq', 'piqa', 'siqa', 'copa',
        }

        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://[^\s]+')

    def should_filter(self, text: str, url: str = None) -> Tuple[bool, str]:
        """
        Check if document should be filtered due to benchmark content.

        Returns:
            (should_filter, reason)
        """
        # Check URL
        if url:
            parsed = urlparse(url)
            domain_path = parsed.netloc + parsed.path

            for blocked in self.blocked_domains:
                if blocked in domain_path:
                    return True, f"URL from benchmark domain: {blocked}"

        # Check for benchmark keywords in text
        text_lower = text.lower()
        for keyword in self.benchmark_keywords:
            # Look for dataset references
            patterns = [
                f'{keyword} dataset',
                f'{keyword} benchmark',
                f'evaluate on {keyword}',
                f'{keyword} test set',
            ]

            for pattern in patterns:
                if pattern in text_lower:
                    return True, f"Contains benchmark reference: {keyword}"

        # Check for URLs in text that might point to benchmarks
        urls_in_text = self.url_pattern.findall(text)
        for found_url in urls_in_text:
            parsed = urlparse(found_url)
            domain_path = parsed.netloc + parsed.path

            for blocked in self.blocked_domains:
                if blocked in domain_path:
                    return True, f"Text contains benchmark URL: {blocked}"

        return False, ""


# Example usage
print("=== Benchmark Filtering Example ===\n")

benchmark_filter = BenchmarkFilter()

test_cases = [
    ("Clean document about machine learning techniques.", None),
    ("We evaluate our model on the MMLU benchmark dataset.", None),
    ("Tutorial on how to use datasets from HuggingFace.", "https://example.com"),
    ("Research paper text.", "https://huggingface.co/datasets/mmlu"),
]

for i, (text, url) in enumerate(test_cases, 1):
    should_filter, reason = benchmark_filter.should_filter(text, url)
    print(f"Case {i}:")
    print(f"  Text: {text[:60]}...")
    if url:
        print(f"  URL: {url}")
    print(f"  Filter: {'YES' if should_filter else 'NO'}")
    if reason:
        print(f"  Reason: {reason}")
    print()
```

### Multilingual and Temporal Considerations

**Multilingual Contamination**:
- Benchmarks often have translations in multiple languages
- Check contamination in all languages present in training data
- Cross-lingual contamination: test sets in one language leaked via translations

```python
class MultilingualContaminationDetector:
    """Detect contamination across multiple languages."""

    def __init__(self, languages: List[str] = None):
        """
        Args:
            languages: List of language codes to track
        """
        self.languages = languages or ['en', 'es', 'fr', 'de', 'zh', 'ja']
        # Separate detectors per language
        self.detectors = {
            lang: ContaminationDetector(n=13)
            for lang in self.languages
        }

    def add_train_document(self, text: str, language: str):
        """Add training document with language tag."""
        if language in self.detectors:
            self.detectors[language].add_train_document(text)

    def check_contamination(
        self,
        test_text: str,
        language: str,
        threshold: float = 0.5,
    ) -> Tuple[bool, float, Dict]:
        """Check contamination for specific language."""
        if language not in self.detectors:
            return False, 0.0, {'error': f'Language {language} not supported'}

        return self.detectors[language].check_contamination(
            test_text, threshold
        )

    def cross_lingual_check(
        self,
        test_text: str,
        source_language: str,
        threshold: float = 0.3,  # Lower threshold for cross-lingual
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Check if test example appears in training data of other languages.

        This can detect cases where benchmarks are translated.
        """
        results = {}

        for lang in self.languages:
            if lang != source_language:
                is_cont, overlap, _ = self.detectors[lang].check_contamination(
                    test_text, threshold
                )
                results[lang] = (is_cont, overlap)

        return results


print("\n=== Multilingual Contamination Detection ===\n")

multi_detector = MultilingualContaminationDetector()

# Add training docs in different languages
multi_detector.add_train_document(
    "Machine learning is a subset of artificial intelligence.",
    language='en'
)
multi_detector.add_train_document(
    "El aprendizaje automático es un subconjunto de la inteligencia artificial.",
    language='es'
)

# Check English test example
en_test = "Machine learning is a subset of artificial intelligence and deep learning."
is_cont, overlap, _ = multi_detector.check_contamination(en_test, 'en', threshold=0.5)
print(f"English test contamination: {is_cont} (overlap: {overlap:.1%})")

# Check for cross-lingual contamination
cross_lingual = multi_detector.cross_lingual_check(en_test, 'en', threshold=0.1)
print(f"Cross-lingual check: {cross_lingual}")
```

**Temporal Data Considerations**:
- Training data freshness affects model knowledge
- Test sets should reflect current knowledge, not outdated information
- Consider data staleness in domains that change rapidly (news, science)

### Reporting Contamination in Research

When publishing model results:

1. **Report methodology**: Describe contamination detection methods used
2. **Provide statistics**: Report overlap percentages for major benchmarks
3. **Flag affected benchmarks**: Clearly mark which benchmarks may be contaminated
4. **Alternative evaluation**: Provide results on clean benchmarks or newly created tests
5. **Release contamination data**: Share contamination analysis for reproducibility

## Data Mixing and Curriculum Learning

When training on multiple data sources, careful mixing is important for model quality.

### Data Mixing Strategies

**The Problem**: LLMs are typically trained on data from multiple sources: web scrapes, books, code, Wikipedia, etc. Each source has different sizes and quality levels. Simply concatenating all data means the largest (often lowest-quality) source dominates. We need a principled way to balance sources.

**Why This Matters**: The [GLaM paper](https://arxiv.org/abs/2112.06905) showed that careful data mixing significantly improved performance. The [LLaMA 2 paper](https://arxiv.org/abs/2307.09288) carefully designed mixture proportions: 89.7% web, 4.5% code, 2.5% Wikipedia, etc. Poor mixing leads to:
- **Domain imbalance**: Large low-quality sources overwhelm small high-quality ones
- **Capability gaps**: Under-represented domains (e.g., code, math) lead to weak capabilities
- **Inefficient learning**: Proportional sampling wastes tokens on redundant data

**Theoretical Justification**: Temperature sampling is a **power law interpolation** between uniform and proportional sampling:

**Temperature Sampling**: Sample from different sources with temperature-adjusted probabilities.

Given source sizes $n_1, n_2, \ldots, n_k$, the sampling probability for source $i$ is:

$$p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}$$

where $T$ is temperature:
- $T = 1$: Proportional sampling (matches source sizes)
- $T \to 0$: Uniform sampling (equal probability per source)
- $T \to \infty$: Sample only from largest source

The temperature parameter provides a **smooth trade-off** between:
- **Diversity** (low T): See all sources equally, but over-sample small sources
- **Naturalness** (high T): Reflect true data distribution, but under-sample small sources

**How This Relates to Alternatives**:
- **Simple concatenation**: Equivalent to T=1, large sources dominate
- **Uniform per-source**: Equivalent to T→0, small sources over-represented
- **Manual mixing ratios**: Requires domain expertise and experimentation
- **Temperature sampling (our approach)**: Single parameter to tune, interpretable, widely used
- **Learned mixing**: Use validation performance to optimize ratios (more complex)

**Key Insights**:
1. **Typical value**: T ≈ 0.3-0.7 balances diversity and naturalness
2. **Source quality matters**: High-quality small sources (Wikipedia) benefit from low T
3. **Curriculum learning**: Can change T during training (start low for diversity, increase for naturalness)
4. **Per-domain temperature**: Different sources can have different sampling temperatures

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

**The Problem**: Training on a random mixture of data treats all examples equally, but not all data is equally valuable at all stages of training. Early in training, models need to learn basic language patterns; later they can benefit from more complex or noisy data. Curriculum learning strategically orders training data.

**Why This Matters**: The [Curriculum Learning paper](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf) showed that training order matters. Recent work on LLMs found curriculum learning can improve convergence speed and final performance. For example:
- **Early training**: Models struggle with noisy/complex data, waste time on noise
- **Foundation building**: Clean, structured data helps establish basic language understanding
- **Progressive complexity**: Gradually introducing harder examples improves generalization

**Theoretical Justification**: Curriculum learning is inspired by **human learning** - we learn easier concepts before harder ones:
- **Progressive abstraction**: Simple patterns → Complex patterns
- **Signal-to-noise optimization**: Start with high signal (Wikipedia), add noise (web) later when model is robust
- **Staged objectives**: Different training stages optimize different capabilities
- **Overfitting prevention**: Early exposure to diverse, clean data prevents fixation on noise patterns

**How This Relates to Alternatives**:
- **Random sampling**: Simple but treats all data equally, suboptimal learning
- **Static mixing**: Fixed mixture throughout training, misses opportunity for curriculum
- **Curriculum learning (our approach)**: Adapts data distribution during training, empirically better
- **Reinforcement learning curriculum**: Use model performance to dynamically adjust data (advanced)

**Key Insights**:
1. **Quality first**: Start with highest-quality data (Wikipedia, books, curated sources)
2. **Gradual noise introduction**: Add web data once basic patterns are learned
3. **Sequence length progression**: Shorter sequences early (faster, easier), longer later
4. **Not just difficulty**: Also consider diversity, domain coverage, data efficiency
5. **Monitoring required**: Track performance to ensure curriculum is helping

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

**The Problem**: The tokenizer is the first step in the LLM pipeline, converting text to tokens. A poorly-trained tokenizer creates inefficiencies that propagate through the entire model: longer sequences, higher compute costs, worse performance on underrepresented content.

**Why This Matters**: The [SentencePiece paper](https://arxiv.org/abs/1808.06226) and [BPE paper](https://arxiv.org/abs/1508.07909) established that tokenizer training data distribution critically affects tokenization efficiency. Poor tokenizer choices hurt model performance:
- **Domain mismatch**: Code tokenized with a news-trained tokenizer fragments into many tokens
- **Language imbalance**: Underrepresented languages get inefficient tokenization (more tokens = less context)
- **Rare characters**: Missing characters create unknown tokens, breaking the model
- **Efficiency**: GPT-3's tokenizer uses ~1.3 tokens/word on average; poor tokenizers use 2+ tokens/word

**Theoretical Justification**: Tokenizer training learns a **vocabulary** and **merging rules** from a corpus:
- **BPE/SentencePiece**: Iteratively merge most frequent character pairs → learns common subwords
- **Coverage**: Vocabulary must cover characters/patterns in training data
- **Frequency-based**: Common sequences get dedicated tokens; rare sequences are decomposed
- **Sample representativeness**: Training sample statistics should match full dataset statistics

The data used to train your tokenizer affects model performance. See [Chapter 1: Tokenization](01-tokenization.md) for implementation details.

### Key Considerations

1. **Representative sample**: Tokenizer training data should represent the distribution of training data
2. **Size**: Typically 1-10GB of text is sufficient
3. **Multilingual**: Include all languages in appropriate proportions
4. **Vocabulary size**: Balance between coverage and efficiency (32k-100k common)

**How This Relates to Alternatives**:
- **Character-level tokenization**: No training needed but extremely long sequences (inefficient)
- **Word-level tokenization**: Fixed vocabulary, can't handle rare words or new words
- **BPE/SentencePiece (standard approach)**: Learns from data, handles rare words via subwords
- **Reusing existing tokenizer**: Fast but may not match your data distribution

**Key Insights**:
1. **Match training distribution**: Tokenizer sample should mirror full training data distribution
2. **Sample size matters**: Too small misses rare patterns; too large wastes time (1-10GB sweet spot)
3. **Multilingual balance**: Languages with <1% of data still need representation in tokenizer training
4. **Vocabulary size trade-off**: Larger vocab = better compression but bigger embedding table
5. **Test coverage**: Measure unknown token rate on validation data (should be <1%)

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

**The Problem**: Web crawl data inadvertently contains personally identifiable information (PII): email addresses, phone numbers, social security numbers, credit card numbers, addresses, and names. Training on this data raises serious privacy and legal concerns, and models can memorize and regurgitate PII during generation.

**Why This Matters**: The [Extracting Training Data from Large Language Models paper](https://arxiv.org/abs/2012.07805) demonstrated that models can memorize and leak training data, including PII. Legal frameworks (GDPR in EU, CCPA in California) require protecting personal data. The [LLaMA 2 paper](https://arxiv.org/abs/2307.09288) explicitly mentions PII removal as part of their data pipeline. Failure to remove PII can lead to:
- **Privacy violations**: Models leak personal information
- **Legal liability**: GDPR fines up to 4% of global revenue
- **Reputational damage**: Public trust erosion
- **Security risks**: Exposure of credentials, tokens, API keys

**Theoretical Justification**: PII detection combines **pattern matching and entity recognition**:
- **Regex patterns**: Structured PII (emails, phone numbers, SSNs) follows predictable formats
- **Validation rules**: Check digit algorithms for credit cards, SSNs
- **Named Entity Recognition (NER)**: ML models detect person names, locations
- **Context clues**: Keywords like "email:", "call me at" help identify PII
- **Statistical anomalies**: Unusual character patterns in URLs (credentials in URLs)

Common PII types:
- Email addresses
- Phone numbers
- IP addresses
- Social Security Numbers (SSN)
- Credit card numbers
- Addresses
- Names (harder - requires NER)

**How This Relates to Alternatives**:
- **No PII removal**: Legally and ethically problematic, high risk
- **Regex-based removal (our approach)**: Fast, catches structured PII, but misses names and complex cases
- **NER-based removal**: Catches names and locations, but slower and requires ML model
- **LLM-based detection**: Most comprehensive but extremely expensive at scale
- **Hybrid approach**: Regex for structured PII + NER for names (recommended for production)

**Key Insights**:
1. **False positives are acceptable**: Better to over-redact than leak PII
2. **Context matters**: "John Smith" in a novel is different from a contact list
3. **International formats**: Phone/address formats vary by country
4. **Embedded PII**: Look for credentials in URLs, code snippets
5. **Validation helps**: Check digit algorithms reduce false positives for credit cards

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

**The Problem**: Web crawl data contains harmful content - hate speech, graphic violence, adult content, harassment, extremist content, and toxicity. Training models on this content can cause them to generate harmful outputs, creating safety, legal, and ethical issues.

**Why This Matters**: The [RealToxicityPrompts paper](https://arxiv.org/abs/2009.11462) demonstrated that models trained on toxic data generate toxic outputs. The [LLaMA 2 paper](https://arxiv.org/abs/2307.09288) emphasizes safety filtering in their data pipeline. Companies face:
- **Brand risk**: Models generating offensive content damage reputation
- **User harm**: Toxic outputs can harm vulnerable users
- **Legal liability**: In some jurisdictions, harmful content generation has legal consequences
- **Platform violations**: Content policies on distribution platforms

**Theoretical Justification**: Safety filtering combines **multiple detection approaches**:
- **Keyword/pattern matching**: Fast but limited, catches obvious cases
- **Toxicity classifiers**: ML models trained on labeled toxic/non-toxic data
- **Ensemble methods**: Combine multiple signals for better accuracy
- **Contextual understanding**: Advanced models understand context (news about violence ≠ promoting violence)

Filter harmful content including:
- Hate speech
- Violence
- Adult content
- Toxicity

**How This Relates to Alternatives**:
- **No filtering**: Unacceptable for deployed systems, creates safety risks
- **Simple keyword blocklists (our simple approach)**: Fast but many false positives/negatives
- **ML classifiers** (Perspective API, OpenAI Moderation): More accurate, context-aware
- **LLM-based filtering**: Most nuanced but expensive
- **Human review**: Most accurate but completely infeasible at scale
- **Hybrid approach**: Keyword pre-filter + ML classifier for candidates (recommended)

**Key Insights**:
1. **Context is critical**: "Violence" in news is different from promotion of violence
2. **False positives acceptable**: Better to over-filter than include harmful content
3. **Multi-category**: Hate, violence, adult content require different detection approaches
4. **Cultural sensitivity**: What's harmful varies by culture and context
5. **Use specialized tools**: Don't build from scratch - use Perspective API, OpenAI Moderation, etc.

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

## Scaling and Distributed Processing

Moving from prototype to production-scale data curation requires handling billions of documents and trillions of tokens. This necessitates distributed computing frameworks and careful optimization.

### Scaling Challenges

**Dataset Scale**:
- Modern LLMs train on 1-15 trillion tokens
- Raw CommonCrawl is hundreds of terabytes
- Deduplication requires comparing billions of document pairs
- Memory-intensive operations (LSH, quality scoring) don't fit on single machines

**Computational Requirements**:
- Processing CommonCrawl: weeks to months on single machine
- Fuzzy deduplication: $O(n^2)$ naive comparison
- Quality scoring with LMs: GPU-intensive
- Cost: thousands to millions of dollars in compute

### Distributed Processing Frameworks

**The Problem**: Single-machine processing is infeasible at LLM scale. Processing CommonCrawl on one machine would take months or years. Distributed computing allows us to parallelize across hundreds or thousands of machines, but requires careful design.

**Why This Matters**: The [RefinedWeb paper](https://arxiv.org/abs/2306.01116) processed 5 trillion tokens from CommonCrawl - impossible on a single machine. The [RedPajama project](https://github.com/togethercomputer/RedPajama-Data) open-sourced their distributed pipeline. Modern data curation requires:
- **Massive parallelism**: 100-1000+ workers processing simultaneously
- **Fault tolerance**: Days-long jobs must survive machine failures
- **Efficient shuffling**: Deduplication requires comparing documents across workers
- **Cost optimization**: Cloud compute costs can reach millions for full pipelines

**Theoretical Justification**: Distributed data processing follows the **MapReduce paradigm**:
- **Map**: Apply transformations to partitions independently (embarrassingly parallel)
- **Shuffle**: Redistribute data for operations requiring coordination (e.g., deduplication)
- **Reduce**: Aggregate results across partitions
- **Fault tolerance**: Store intermediate results, replay failures
- **Data locality**: Move computation to data when possible

Three main options for distributed data processing:

1. **Apache Spark**: Most common for LLM data pipelines
2. **Dask**: Python-native, similar to pandas
3. **Ray**: Modern framework with good ML integration

**How This Relates to Alternatives**:
- **Single-machine processing**: Fine for small datasets (<100GB), completely infeasible at scale
- **Custom distributed system**: Maximum control but requires months of engineering
- **Apache Spark (our approach)**: Industry standard, mature, well-documented, rich ecosystem
- **Dask**: Better for Python-heavy workloads, easier learning curve, less mature
- **Ray**: Good for ML workloads, modern API, smaller ecosystem than Spark

**Key Insights**:
1. **Embarrassingly parallel operations first**: Filtering, extraction, scoring don't need coordination
2. **Minimize shuffles**: Deduplication requires shuffling - most expensive operation
3. **Partition wisely**: Partition by domain/URL for better data locality
4. **Checkpoint frequently**: Save intermediate results to avoid recomputation
5. **Monitor resource usage**: Memory/disk bottlenecks can kill jobs

```python
"""
Example: Distributed data curation with PySpark

Note: This example requires PySpark installation:
    pip install pyspark

For production, run on a cluster (AWS EMR, Databricks, etc.)
"""

from typing import Iterator, List, Dict
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import StringType, FloatType, BooleanType, StructType, StructField
import hashlib

class DistributedDataCurator:
    """Data curation pipeline using Apache Spark."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def extract_text_from_html(self, html_content: str) -> str:
        """Extract text from HTML (simplified)."""
        # In production, use trafilatura or similar
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove scripts, styles
            for tag in soup(['script', 'style']):
                tag.decompose()
            return soup.get_text(separator=' ', strip=True)
        except:
            return ""

    def compute_quality_score(self, text: str) -> float:
        """Compute simple quality score."""
        if not text:
            return 0.0

        words = text.split()
        if len(words) < 10:
            return 0.0

        # Simple heuristics
        avg_word_len = sum(len(w) for w in words) / len(words)
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)

        score = 0.0
        if 3.0 <= avg_word_len <= 8.0:
            score += 0.5
        if alpha_ratio > 0.7:
            score += 0.5

        return score

    def exact_duplicate_hash(self, text: str) -> str:
        """Compute hash for exact deduplication."""
        normalized = ' '.join(text.split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def process_html_documents(
        self,
        input_path: str,
        output_path: str,
        min_quality: float = 0.5,
    ):
        """
        Process HTML documents through curation pipeline.

        Args:
            input_path: Path to input data (can be S3, HDFS, etc.)
            output_path: Path to output data
            min_quality: Minimum quality score threshold
        """
        # Register UDFs
        extract_udf = udf(self.extract_text_from_html, StringType())
        quality_udf = udf(self.compute_quality_score, FloatType())
        hash_udf = udf(self.exact_duplicate_hash, StringType())

        # Read input data
        # Assuming JSON format with 'html' and 'url' fields
        df = self.spark.read.json(input_path)

        print(f"Initial documents: {df.count():,}")

        # Step 1: Extract text
        df = df.withColumn('text', extract_udf(col('html')))

        # Step 2: Filter by length
        df = df.filter(col('text').isNotNull())
        df = df.filter(length(col('text')) >= 100)
        print(f"After length filter: {df.count():,}")

        # Step 3: Compute quality scores
        df = df.withColumn('quality_score', quality_udf(col('text')))

        # Step 4: Filter by quality
        df = df.filter(col('quality_score') >= min_quality)
        print(f"After quality filter: {df.count():,}")

        # Step 5: Exact deduplication
        df = df.withColumn('content_hash', hash_udf(col('text')))
        df = df.dropDuplicates(['content_hash'])
        print(f"After deduplication: {df.count():,}")

        # Step 6: Save results
        df.select('url', 'text', 'quality_score').write.parquet(
            output_path,
            mode='overwrite',
            compression='snappy',
        )

        print(f"Saved to: {output_path}")


# Example usage (requires Spark cluster)
def example_spark_pipeline():
    """
    Example of running distributed curation pipeline.

    In production, run with:
        spark-submit --master yarn --num-executors 100 \\
                     --executor-memory 16g --executor-cores 4 \\
                     curation_script.py
    """
    # Create Spark session
    spark = SparkSession.builder \\
        .appName("DataCuration") \\
        .config("spark.sql.shuffle.partitions", "1000") \\
        .config("spark.driver.memory", "8g") \\
        .config("spark.executor.memory", "16g") \\
        .getOrCreate()

    curator = DistributedDataCurator(spark)

    # Process data
    curator.process_html_documents(
        input_path="s3://my-bucket/raw-crawl/*.json",
        output_path="s3://my-bucket/curated-data/",
        min_quality=0.5,
    )

    spark.stop()


print("=== Distributed Processing Example ===\n")
print("Note: Full example requires Spark cluster.")
print("See code comments for production deployment.")
```

### Distributed Fuzzy Deduplication

Fuzzy deduplication is challenging to distribute due to the all-pairs comparison problem. Here's a scalable approach using LSH:

```python
"""
Scalable fuzzy deduplication strategy:

1. Compute MinHash signatures (embarrassingly parallel)
2. Use LSH to generate candidate pairs (distributed hash tables)
3. Verify candidates with exact Jaccard (parallel comparison)
4. Mark duplicates and filter

Key insight: LSH reduces comparisons from O(n^2) to O(n * c) where c is
average candidates per document (typically << n).
"""

from typing import List, Tuple, Set
import hashlib

class DistributedFuzzyDeduplicator:
    """Distributed fuzzy deduplication using Spark."""

    def __init__(self, spark: SparkSession, num_perm: int = 128):
        self.spark = spark
        self.num_perm = num_perm

    def compute_minhash(self, text: str) -> List[int]:
        """Compute MinHash signature (simplified)."""
        # In production, use datasketch library
        import random
        random.seed(42)

        # Character n-grams
        ngrams = set()
        for i in range(len(text) - 5 + 1):
            ngrams.add(text[i:i+5])

        if not ngrams:
            return [0] * self.num_perm

        signature = []
        for seed in range(self.num_perm):
            min_hash = min(
                int(hashlib.md5(f"{ng}_{seed}".encode()).hexdigest(), 16)
                for ng in ngrams
            )
            signature.append(min_hash)

        return signature

    def lsh_buckets(
        self,
        signature: List[int],
        num_bands: int = 16,
    ) -> List[Tuple[int, str]]:
        """Generate LSH bucket IDs for signature."""
        rows_per_band = len(signature) // num_bands
        buckets = []

        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band = signature[start:end]

            # Hash the band
            band_str = ','.join(map(str, band))
            bucket_id = hashlib.md5(band_str.encode()).hexdigest()

            buckets.append((band_idx, bucket_id))

        return buckets

    def deduplicate(
        self,
        input_df: DataFrame,
        text_column: str = 'text',
        threshold: float = 0.8,
    ) -> DataFrame:
        """
        Perform distributed fuzzy deduplication.

        Args:
            input_df: Input DataFrame with text
            text_column: Name of text column
            threshold: Jaccard similarity threshold

        Returns:
            DataFrame with duplicates removed
        """
        from pyspark.sql.functions import monotonically_increasing_id

        # Add unique ID to each document
        df = input_df.withColumn('doc_id', monotonically_increasing_id())

        # Step 1: Compute MinHash signatures (parallel)
        minhash_udf = udf(self.compute_minhash, ArrayType(IntegerType()))
        df = df.withColumn('signature', minhash_udf(col(text_column)))

        # Step 2: Generate LSH buckets (parallel)
        lsh_udf = udf(
            self.lsh_buckets,
            ArrayType(StructType([
                StructField('band', IntegerType()),
                StructField('bucket', StringType())
            ]))
        )
        df = df.withColumn('buckets', lsh_udf(col('signature')))

        # Step 3: Explode buckets to find candidates
        from pyspark.sql.functions import explode
        df_buckets = df.select('doc_id', explode('buckets').alias('bucket_info'))
        df_buckets = df_buckets.select(
            'doc_id',
            col('bucket_info.band').alias('band'),
            col('bucket_info.bucket').alias('bucket')
        )

        # Step 4: Self-join to find candidate pairs
        candidates = df_buckets.alias('a').join(
            df_buckets.alias('b'),
            (col('a.band') == col('b.band')) &
            (col('a.bucket') == col('b.bucket')) &
            (col('a.doc_id') < col('b.doc_id'))  # Avoid duplicates
        ).select(
            col('a.doc_id').alias('doc1'),
            col('b.doc_id').alias('doc2')
        ).distinct()

        # Step 5: Verify candidates with exact Jaccard
        # (In production, implement actual Jaccard computation)
        # For now, mark all candidates as duplicates

        # Step 6: Keep only one document from each duplicate cluster
        # Simple approach: keep document with lower ID
        duplicate_ids = candidates.select('doc2').distinct()

        # Step 7: Filter out duplicates
        result_df = df.join(
            duplicate_ids,
            df.doc_id == duplicate_ids.doc2,
            'left_anti'  # Keep rows not in duplicate_ids
        )

        return result_df.drop('doc_id', 'signature', 'buckets')


print("\n=== Distributed Fuzzy Deduplication ===\n")
print("Strategy: MinHash + LSH + Spark")
print("Reduces complexity from O(n^2) to O(n * candidates)")
print("Scales to billions of documents")
```

### Sharding and Streaming Strategies

For trillion-token datasets, use sharding to process data incrementally:

```python
class ShardedPipeline:
    """Process data in shards for memory efficiency."""

    def __init__(self, shard_size_gb: int = 10):
        self.shard_size_gb = shard_size_gb
        self.shard_stats = []

    def process_in_shards(
        self,
        input_files: List[str],
        output_dir: str,
        process_func,
    ):
        """
        Process large dataset in shards.

        Args:
            input_files: List of input file paths
            output_dir: Output directory
            process_func: Function to process each shard
        """
        current_shard = []
        current_size = 0
        shard_idx = 0

        for file_path in input_files:
            file_size = get_file_size(file_path)  # In GB

            if current_size + file_size > self.shard_size_gb and current_shard:
                # Process current shard
                output_path = f"{output_dir}/shard_{shard_idx:05d}"
                stats = process_func(current_shard, output_path)
                self.shard_stats.append(stats)

                # Start new shard
                current_shard = []
                current_size = 0
                shard_idx += 1

            current_shard.append(file_path)
            current_size += file_size

        # Process final shard
        if current_shard:
            output_path = f"{output_dir}/shard_{shard_idx:05d}"
            stats = process_func(current_shard, output_path)
            self.shard_stats.append(stats)

        return self.aggregate_stats()

    def aggregate_stats(self) -> Dict:
        """Aggregate statistics across shards."""
        total_docs = sum(s.get('documents', 0) for s in self.shard_stats)
        total_filtered = sum(s.get('filtered', 0) for s in self.shard_stats)

        return {
            'total_shards': len(self.shard_stats),
            'total_documents': total_docs,
            'total_filtered': total_filtered,
            'pass_rate': (total_docs - total_filtered) / max(total_docs, 1),
        }


def get_file_size(path: str) -> float:
    """Get file size in GB."""
    import os
    return os.path.getsize(path) / (1024 ** 3)


print("\n=== Sharding Strategy ===\n")
print("Process data in manageable chunks")
print("Reduces memory requirements")
print("Enables incremental progress tracking")
```

### Cost Optimization

**Compute vs. Training Tradeoffs**:

```python
def estimate_curation_cost(
    dataset_size_tb: float,
    cost_per_cpu_hour: float = 0.10,
    cost_per_gpu_hour: float = 1.00,
    processing_rate_gb_per_hour: float = 100,
) -> Dict[str, float]:
    """
    Estimate data curation costs.

    Args:
        dataset_size_tb: Dataset size in terabytes
        cost_per_cpu_hour: Cost per CPU hour
        cost_per_gpu_hour: Cost per GPU hour (for quality filtering)
        processing_rate_gb_per_hour: Processing throughput

    Returns:
        Cost breakdown
    """
    dataset_size_gb = dataset_size_tb * 1024

    # Basic filtering (CPU)
    cpu_hours = dataset_size_gb / processing_rate_gb_per_hour
    cpu_cost = cpu_hours * cost_per_cpu_hour

    # Quality scoring with LM (GPU, slower)
    # Assume 10x slower than basic filtering
    gpu_hours = cpu_hours * 10
    gpu_cost = gpu_hours * cost_per_gpu_hour

    # Deduplication (CPU, memory intensive)
    # Assume 2x time of basic filtering
    dedup_hours = cpu_hours * 2
    dedup_cost = dedup_hours * cost_per_cpu_hour

    total_cost = cpu_cost + gpu_cost + dedup_cost

    return {
        'basic_filtering_cost': cpu_cost,
        'quality_scoring_cost': gpu_cost,
        'deduplication_cost': dedup_cost,
        'total_cost': total_cost,
        'cost_per_gb': total_cost / dataset_size_gb,
    }


# Example: CommonCrawl processing cost
print("\n=== Cost Estimation ===\n")

# CommonCrawl is ~400TB raw
costs = estimate_curation_cost(dataset_size_tb=400)

print("Estimated costs for 400TB dataset:")
for key, value in costs.items():
    print(f"  {key}: ${value:,.2f}")

print("\nKey insight: Data curation costs are often 1-10% of training costs")
print("But impact on model quality can be enormous")
```

### Best Practices for Scaling

1. **Start small, scale gradually**:
   - Prototype on 1GB sample
   - Test on 100GB subset
   - Scale to full dataset

2. **Use appropriate tools**:
   - <1TB: Single machine with multiprocessing
   - 1-100TB: Spark cluster
   - >100TB: Specialized systems (Snowflake, Databricks)

3. **Optimize for I/O**:
   - Use columnar formats (Parquet, ORC)
   - Compress data (snappy, zstd)
   - Partition by domain or date

4. **Monitor and checkpoint**:
   - Save intermediate results
   - Track progress per shard
   - Enable resume from failure

5. **Profile and optimize**:
   - Identify bottlenecks
   - Optimize hot paths
   - Consider approximate methods for massive scale

## Legal and Ethical Considerations

Data curation for LLMs raises important legal and ethical questions that must be addressed responsibly.

### Copyright and Fair Use

**Legal Landscape**:
- Web scraping exists in a legal gray area
- Copyright law varies by jurisdiction
- Fair use doctrine (US) vs. other frameworks
- Recent lawsuits against AI companies for training data use

**Key Issues**:

1. **Copyright Infringement**:
   - Training on copyrighted works without permission
   - Outputting memorized copyrighted content
   - "Transformative use" defense (debated)

2. **Terms of Service Violations**:
   - Many websites prohibit scraping in ToS
   - Legal enforceability varies
   - Risk of being blocked or sued

3. **Fair Use Considerations** (US law):
   - Purpose (commercial vs. research)
   - Nature of copyrighted work
   - Amount used
   - Effect on market value

```python
class LicenseChecker:
    """Check for restrictive licenses in training data."""

    def __init__(self):
        # Common restrictive license indicators
        self.restrictive_terms = {
            'all rights reserved',
            'proprietary',
            'confidential',
            'do not distribute',
            'internal use only',
            'copyright',  # May indicate full copyright
        }

        # Creative Commons non-commercial licenses
        self.nc_licenses = {
            'cc by-nc',
            'cc by-nc-sa',
            'cc by-nc-nd',
        }

    def check_license(self, text: str, metadata: Dict = None) -> Dict[str, any]:
        """
        Check text for license restrictions.

        Returns:
            Dictionary with license info and recommendations
        """
        text_lower = text.lower()
        issues = []

        # Check for copyright notices
        if 'copyright' in text_lower or '©' in text:
            issues.append('Contains copyright notice')

        # Check for restrictive terms
        for term in self.restrictive_terms:
            if term in text_lower:
                issues.append(f'Contains restrictive term: {term}')

        # Check for NC licenses
        for license_type in self.nc_licenses:
            if license_type in text_lower:
                issues.append(f'Non-commercial license: {license_type}')

        # Check metadata if available
        if metadata:
            if metadata.get('license') in ['proprietary', 'all-rights-reserved']:
                issues.append(f"Restrictive license in metadata: {metadata['license']}")

        risk_level = 'high' if issues else 'low'

        return {
            'risk_level': risk_level,
            'issues': issues,
            'recommendation': 'Filter if commercial use' if issues else 'Likely acceptable',
        }


print("=== License Checking Example ===\n")

checker = LicenseChecker()

test_documents = [
    "This tutorial explains machine learning basics.",
    "Copyright © 2024 ExampleCorp. All rights reserved. Proprietary material.",
    "This work is licensed under CC BY-NC-SA 4.0.",
]

for i, doc in enumerate(test_documents, 1):
    result = checker.check_license(doc)
    print(f"Document {i}:")
    print(f"  Risk: {result['risk_level']}")
    print(f"  Recommendation: {result['recommendation']}")
    if result['issues']:
        print(f"  Issues: {result['issues']}")
    print()
```

### Privacy and Consent

**Personal Data Concerns**:
- GDPR (Europe) requires consent for personal data processing
- CCPA (California) grants data privacy rights
- Training data may contain personal information
- Right to be forgotten vs. model retraining costs

**Best Practices**:

1. **Aggressive PII Removal**:
   - Go beyond basic detection
   - Use context-aware methods
   - Err on the side of caution

2. **Respect robots.txt and opt-outs**:
   - Honor website scraping preferences
   - Implement opt-out mechanisms
   - Maintain blocklists of prohibited sites

3. **Data minimization**:
   - Only collect necessary data
   - Delete raw data after processing
   - Retain only curated datasets

```python
class PrivacyChecker:
    """Check for privacy concerns in training data."""

    def __init__(self):
        # Privacy-sensitive content indicators
        self.sensitive_phrases = {
            'patient record',
            'medical history',
            'social security',
            'bank account',
            'credit card',
            'passport number',
            'driver license',
            'patient',
            'diagnosis',
            'treatment plan',
        }

        # GDPR special categories
        self.gdpr_special_categories = {
            'racial origin',
            'ethnic origin',
            'political opinion',
            'religious belief',
            'trade union',
            'genetic data',
            'biometric data',
            'health data',
            'sex life',
            'sexual orientation',
        }

    def assess_privacy_risk(self, text: str) -> Dict[str, any]:
        """Assess privacy risks in text."""
        text_lower = text.lower()
        risks = []

        # Check for sensitive phrases
        for phrase in self.sensitive_phrases:
            if phrase in text_lower:
                risks.append(f'Contains sensitive data: {phrase}')

        # Check for GDPR special categories
        for category in self.gdpr_special_categories:
            if category in text_lower:
                risks.append(f'GDPR special category: {category}')

        # Estimate risk level
        if len(risks) >= 3:
            risk_level = 'high'
        elif len(risks) >= 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'risk_level': risk_level,
            'risks': risks,
            'recommendation': 'Filter' if risk_level == 'high' else 'Review' if risk_level == 'medium' else 'OK',
        }


print("\n=== Privacy Risk Assessment ===\n")

privacy_checker = PrivacyChecker()

test_texts = [
    "Machine learning models can be trained on large datasets.",
    "Patient record shows diagnosis of diabetes with treatment plan.",
]

for i, text in enumerate(test_texts, 1):
    result = privacy_checker.assess_privacy_risk(text)
    print(f"Text {i}:")
    print(f"  Risk level: {result['risk_level']}")
    print(f"  Recommendation: {result['recommendation']}")
    if result['risks']:
        print(f"  Risks: {result['risks'][:3]}")
    print()
```

### Transparency and Documentation

**Research Ethics**:
- Document data sources and curation processes
- Report data composition and statistics
- Acknowledge limitations and biases
- Enable reproducibility

**Data Cards and Model Cards**:
- Describe dataset composition
- Document curation decisions
- Report demographic distributions
- Disclose known biases

```python
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class DatasetCard:
    """Documentation for a curated dataset."""

    # Basic info
    name: str
    version: str
    creation_date: str
    creators: List[str]

    # Data composition
    total_documents: int
    total_tokens: int
    sources: Dict[str, float]  # source -> percentage
    languages: Dict[str, float]  # language -> percentage

    # Curation details
    filters_applied: List[str]
    deduplication_method: str
    quality_threshold: float

    # Privacy and legal
    pii_removed: bool
    license_filtering: bool
    known_sensitive_content: List[str]

    # Limitations
    known_biases: List[str]
    excluded_domains: List[str]
    data_cutoff_date: str

    def to_json(self, filepath: str):
        """Save dataset card as JSON."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str):
        """Load dataset card from JSON."""
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)


# Example dataset card
print("\n=== Dataset Card Example ===\n")

card = DatasetCard(
    name="ExampleCuratedWeb",
    version="1.0",
    creation_date=datetime.now().isoformat(),
    creators=["Research Team"],
    total_documents=10_000_000,
    total_tokens=50_000_000_000,
    sources={
        "web": 0.7,
        "books": 0.15,
        "code": 0.10,
        "papers": 0.05,
    },
    languages={
        "en": 0.8,
        "es": 0.1,
        "fr": 0.05,
        "other": 0.05,
    },
    filters_applied=[
        "Basic quality (length, language)",
        "Exact deduplication (SHA-256)",
        "Fuzzy deduplication (MinHash, threshold=0.8)",
        "PII removal (emails, phones, SSN)",
        "Safety filtering (toxicity, hate speech)",
    ],
    deduplication_method="MinHash + LSH (128 perms, 16 bands)",
    quality_threshold=0.5,
    pii_removed=True,
    license_filtering=True,
    known_sensitive_content=["medical", "legal"],
    known_biases=[
        "English-language bias",
        "Western cultural bias",
        "Temporal bias (data from 2020-2024)",
    ],
    excluded_domains=[
        "Social media (privacy concerns)",
        "Paywalled content",
        "Known benchmark datasets",
    ],
    data_cutoff_date="2024-06-01",
)

print("Dataset Card:")
print(json.dumps(asdict(card), indent=2))
```

### Ethical Guidelines

1. **Do No Harm**:
   - Consider downstream impacts
   - Avoid amplifying harmful biases
   - Protect vulnerable groups

2. **Respect Rights**:
   - Honor copyright and licenses
   - Respect privacy and consent
   - Follow platform ToS where reasonable

3. **Transparency**:
   - Document data sources
   - Disclose curation methods
   - Acknowledge limitations

4. **Accountability**:
   - Enable opt-outs
   - Respond to takedown requests
   - Regular audits

5. **Continuous Improvement**:
   - Monitor for emerging issues
   - Update practices based on feedback
   - Engage with affected communities

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
5. **Data mixing matters**: Carefully balance different data sources with temperature sampling
6. **Document everything**: Keep detailed logs of filtering decisions for reproducibility
7. **Iterative improvement**: Analyze model failures and improve data filters accordingly
8. **Contamination prevention**: Proactively detect and remove test set leakage using n-gram overlap
9. **Scale gradually**: Start with prototypes on small data, optimize, then scale to full datasets
10. **Legal and ethical compliance**: Consider copyright, privacy laws, and ethical implications
11. **Transparency**: Create dataset cards documenting sources, filters, and known limitations
12. **LSH parameter tuning**: Choose bands and rows to optimize precision/recall tradeoff for your threshold
13. **Multilingual considerations**: Handle language-specific filtering and cross-lingual contamination
14. **Cost optimization**: Balance curation costs against training costs and model quality improvements

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

8. **Chinchilla** (Hoffmann et al., 2022): [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
   - Data scaling laws and optimal dataset sizes

9. **DataComp** (Gadre et al., 2023): [DataComp: In search of the next generation of multimodal datasets](https://arxiv.org/abs/2304.14108)
   - Large-scale study of data quality impact on model performance

10. **Contamination Analysis** (Various): Studies on benchmark contamination in GPT-3, GPT-4, and other models
    - Important for understanding evaluation validity

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
