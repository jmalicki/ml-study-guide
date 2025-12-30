# Chapter 12: Flash Attention

Flash Attention is an IO-aware attention algorithm that achieves 2-4x speedup and reduces memory usage from O(N²) to O(N) for sequence length N. Understanding Flash Attention is essential for ML interviews because it demonstrates how algorithm design must consider hardware characteristics, not just theoretical complexity.

This chapter covers the fundamental problem Flash Attention solves, the algorithmic techniques it uses, and practical implementation considerations.

## Table of Contents

1. [The Memory Bottleneck Problem](#the-memory-bottleneck-problem)
2. [GPU Memory Hierarchy](#gpu-memory-hierarchy)
3. [Standard Attention Analysis](#standard-attention-analysis)
4. [IO-Aware Algorithm Design](#io-aware-algorithm-design)
5. [Tiling Strategy](#tiling-strategy)
6. [Online Softmax Algorithm](#online-softmax-algorithm)
7. [Forward Pass Algorithm](#forward-pass-algorithm)
8. [Backward Pass and Recomputation](#backward-pass-and-recomputation)
9. [Implementation Considerations](#implementation-considerations)
10. [FlashAttention Versions](#flashattention-versions)
11. [Using Flash Attention in Practice](#using-flash-attention-in-practice)
12. [Theoretical Analysis](#theoretical-analysis)
13. [Extensions and Variants](#extensions-and-variants)
14. [Summary](#summary)

---

## The Memory Bottleneck Problem

### The O(N²) Problem

Standard attention has a fundamental memory problem that becomes critical for long sequences:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The attention matrix $S = QK^T$ has shape $(N, N)$ where $N$ is the sequence length. For modern LLMs with long contexts:

- 32K context: $32{,}768^2 = 1{,}073{,}741{,}824$ elements
- 100K context: $100{,}000^2 = 10{,}000{,}000{,}000$ elements
- Each element in FP16: 2 bytes

**Memory requirements:**
- 32K context: 2 GB per attention head
- 100K context: 20 GB per attention head
- With 64 heads: 1.28 TB (!)

This is clearly infeasible. But the problem is not just memory size—it's memory bandwidth.

```python
import torch
import time

def demonstrate_memory_bottleneck():
    """
    Demonstrate that attention is memory-bound, not compute-bound.

    On modern GPUs, the bottleneck is moving data between HBM and SRAM,
    not the actual computation.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configuration
    batch_size = 8
    n_heads = 8
    seq_len = 4096
    head_dim = 64

    # Create random Q, K, V
    Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(10):
        _ = torch.matmul(Q, K.transpose(-2, -1))

    torch.cuda.synchronize()

    # Time just QK^T (compute)
    start = time.time()
    for _ in range(100):
        scores = torch.matmul(Q, K.transpose(-2, -1))
    torch.cuda.synchronize()
    compute_time = (time.time() - start) / 100

    # Time full attention (compute + memory)
    start = time.time()
    for _ in range(100):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
    torch.cuda.synchronize()
    full_time = (time.time() - start) / 100

    print(f"Sequence length: {seq_len}")
    print(f"QK^T compute time: {compute_time*1000:.2f} ms")
    print(f"Full attention time: {full_time*1000:.2f} ms")
    print(f"Memory overhead ratio: {full_time/compute_time:.2f}x")
    print(f"\nAttention matrix size: {seq_len * seq_len * 2 / 1e6:.2f} MB per head")
```

**Key insight:** On modern GPUs with high FLOPS but limited memory bandwidth, the bottleneck is not computation but moving data between memory levels.

---

## GPU Memory Hierarchy

Understanding GPU memory hierarchy is critical for understanding Flash Attention's design.

### Memory Levels

```python
class GPUMemoryHierarchy:
    """
    GPU memory hierarchy (NVIDIA Ampere/Hopper architecture).

    Flash Attention's core insight: Minimize HBM ↔ SRAM transfers.
    """

    SRAM = {
        'name': 'On-chip SRAM (Shared Memory + L1/L2 Cache)',
        'size_A100': '20 MB per SM × 108 SMs = ~2 GB total',
        'size_H100': '32 MB per SM × 132 SMs = ~4 GB total',
        'bandwidth': '~20 TB/s (A100), ~30 TB/s (H100)',
        'latency': '~1 cycle',
        'notes': 'Fast but tiny. Must fit working set here for efficiency.'
    }

    HBM = {
        'name': 'High Bandwidth Memory (DRAM)',
        'size_A100': '40 GB or 80 GB',
        'size_H100': '80 GB',
        'bandwidth': '1.5-2 TB/s (A100), 3 TB/s (H100)',
        'latency': '~100s of cycles',
        'notes': 'Large but slow. This is the bottleneck!'
    }

    @staticmethod
    def bandwidth_ratio() -> float:
        """
        SRAM is ~10-15x faster than HBM.

        This means: Reading 1 MB from HBM costs as much time as
        performing ~10-15 TFLOPS of computation on data in SRAM.

        Implication: We should do maximum computation on data
        while it's in SRAM, even if it means redundant computation.
        """
        return 20 / 1.5  # ~13x on A100


def calculate_memory_bandwidth_cost():
    """
    Calculate the cost of memory transfers vs computation.

    This shows why Flash Attention's recomputation strategy makes sense.
    """
    # Constants for A100
    hbm_bandwidth = 1.5e12  # 1.5 TB/s = 1.5e12 bytes/s
    compute_throughput = 312e12  # 312 TFLOPS FP16

    # For a sequence of length N with head dimension d
    N = 4096
    d = 64

    # Standard attention memory transfers (bytes)
    # Read Q, K, V: 3 × N × d × 2 bytes (FP16)
    # Write S = QK^T: N × N × 2 bytes
    # Read S for softmax: N × N × 2 bytes
    # Write P = softmax(S): N × N × 2 bytes
    # Read P, V for output: (N × N + N × d) × 2 bytes

    total_memory_bytes = (
        3 * N * d * 2 +      # Read Q, K, V
        N * N * 2 +          # Write S
        N * N * 2 +          # Read S
        N * N * 2 +          # Write P
        (N * N + N * d) * 2  # Read P, V
    )

    memory_time = total_memory_bytes / hbm_bandwidth

    # Computation (FLOPs)
    # QK^T: 2 × N × N × d (matmul)
    # Softmax: ~5 × N × N (exp, sum, divide)
    # PV: 2 × N × N × d (matmul)

    total_flops = (
        2 * N * N * d +  # QK^T
        5 * N * N +      # Softmax
        2 * N * N * d    # PV
    )

    compute_time = total_flops / compute_throughput

    print(f"Sequence length: {N}")
    print(f"Memory transfer time: {memory_time*1000:.3f} ms")
    print(f"Compute time: {compute_time*1000:.3f} ms")
    print(f"Memory bound by: {memory_time/compute_time:.2f}x")
    print(f"\nTotal memory accessed: {total_memory_bytes/1e9:.2f} GB")
    print(f"Attention is {'memory-bound' if memory_time > compute_time else 'compute-bound'}")

    return memory_time, compute_time
```

**Key takeaway:** For attention, memory transfers dominate computation time. Flash Attention addresses this by minimizing HBM access.

---

## Standard Attention Analysis

Let's analyze standard attention implementation to understand where the bottleneck is.

```python
import torch
import torch.nn.functional as F

class StandardAttention:
    """
    Standard attention implementation with detailed memory analysis.

    This is what we want to improve.
    """

    @staticmethod
    def forward(
        Q: torch.Tensor,  # (batch, n_heads, seq_len, head_dim)
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Standard attention forward pass.

        Memory complexity: O(N²) due to storing attention matrix.
        """
        batch, n_heads, N, d = Q.shape

        # Step 1: Compute attention scores S = QK^T / √d
        # Memory: Allocate N×N matrix, write to HBM
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
        # HBM writes: N² × batch × n_heads elements

        # Step 2: Apply causal mask if needed
        if causal:
            mask = torch.triu(torch.ones(N, N, device=Q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            # HBM reads: N² (mask), HBM writes: N²

        # Step 3: Softmax
        # Memory: Read N² from HBM, write N² back
        attention_weights = F.softmax(scores, dim=-1)
        # HBM reads: N², HBM writes: N²

        # Step 4: Weighted sum of values
        # Memory: Read N² (attention) + Nd (V), write Nd (output)
        output = torch.matmul(attention_weights, V)
        # HBM reads: N² + Nd, HBM writes: Nd

        return output

    @staticmethod
    def memory_analysis(batch: int, n_heads: int, seq_len: int, head_dim: int) -> dict:
        """
        Analyze memory requirements and HBM accesses.

        Returns dictionary with memory statistics.
        """
        N = seq_len
        d = head_dim

        # Memory storage (in elements, multiply by 2 for FP16 bytes)
        attention_matrix = batch * n_heads * N * N
        inputs_outputs = batch * n_heads * 3 * N * d  # Q, K, V

        total_elements = attention_matrix + inputs_outputs
        total_bytes = total_elements * 2  # FP16

        # HBM accesses (read + write operations)
        # Forward pass
        hbm_reads_fwd = (
            3 * N * d +      # Read Q, K, V
            N * N +          # Read scores for softmax
            N * N +          # Read attention for matmul with V
            N * d            # Read V
        ) * batch * n_heads * 2  # FP16 bytes

        hbm_writes_fwd = (
            N * N +          # Write scores
            N * N +          # Write attention weights
            N * d            # Write output
        ) * batch * n_heads * 2

        total_hbm_access = hbm_reads_fwd + hbm_writes_fwd

        return {
            'peak_memory_gb': total_bytes / 1e9,
            'attention_matrix_gb': attention_matrix * 2 / 1e9,
            'hbm_access_gb': total_hbm_access / 1e9,
            'memory_complexity': 'O(N²)',
            'hbm_complexity': 'O(N² + Nd)',
        }


# Example analysis
def compare_sequence_lengths():
    """Compare memory requirements for different sequence lengths."""
    batch = 8
    n_heads = 32
    head_dim = 128

    for seq_len in [1024, 4096, 16384, 65536]:
        stats = StandardAttention.memory_analysis(batch, n_heads, seq_len, head_dim)
        print(f"\nSequence length: {seq_len}")
        print(f"  Peak memory: {stats['peak_memory_gb']:.2f} GB")
        print(f"  Attention matrix: {stats['attention_matrix_gb']:.2f} GB")
        print(f"  HBM access: {stats['hbm_access_gb']:.2f} GB")
```

**Problems with standard attention:**
1. **O(N²) memory:** Must store full attention matrix
2. **Excessive HBM access:** Attention matrix written/read multiple times
3. **Limited sequence length:** Cannot fit long sequences in memory
4. **Memory-bound:** Spends more time moving data than computing

---

## IO-Aware Algorithm Design

Flash Attention is designed around the principle: **minimize HBM access, even at the cost of extra computation**.

### Core Principles

```python
class FlashAttentionPrinciples:
    """
    Core design principles of Flash Attention.
    """

    @staticmethod
    def principle_1_tiling():
        """
        Principle 1: Tiling (Kernel Fusion)

        Standard approach (3 separate kernels):
        - Kernel 1: Compute QK^T, write to HBM
        - Kernel 2: Softmax, read from HBM, write to HBM
        - Kernel 3: Multiply by V, read from HBM

        Flash Attention (1 fused kernel):
        - Process Q, K, V in blocks that fit in SRAM
        - Never write intermediate attention matrix to HBM
        - Output final result directly

        Key: Do all operations on a tile while it's in SRAM.
        """
        pass

    @staticmethod
    def principle_2_recomputation():
        """
        Principle 2: Recomputation in Backward Pass

        Standard approach:
        - Store attention matrix for backward pass
        - Cost: O(N²) memory

        Flash Attention:
        - Don't store attention matrix
        - Recompute it on-the-fly during backward pass
        - Cost: O(N) memory, but extra computation

        Trade-off: Memory bandwidth vs. compute
        - Modern GPUs have excess compute capacity
        - But limited memory bandwidth
        - Therefore: Recomputation is actually faster!
        """
        pass

    @staticmethod
    def principle_3_online_softmax():
        """
        Principle 3: Online (Incremental) Softmax

        Challenge: How to compute softmax over blocks?

        Softmax requires global statistics (max and sum over all elements).
        But we're processing in blocks!

        Solution: Online softmax algorithm
        - Maintain running max and sum
        - Update incrementally as we process each block
        - Mathematically exact, no approximation

        This is the key algorithmic innovation.
        """
        pass


def visualize_standard_vs_flash():
    """
    Visualize data movement: Standard vs Flash Attention.

    Standard Attention:
    ┌─────────┐
    │   HBM   │  Q, K, V stored here
    │         │  ↓ Read Q, K
    │         │  ↓ Compute QK^T
    │  (Slow) │  ↓ Write S to HBM
    │         │  ↓ Read S
    │         │  ↓ Softmax
    │         │  ↓ Write P to HBM
    │         │  ↓ Read P, V
    └─────────┘  ↓ Write Output

    Flash Attention:
    ┌──────────┐
    │   SRAM   │  Load block of Q, K, V
    │          │  ↓ Compute QK^T (stays in SRAM)
    │  (Fast)  │  ↓ Softmax (stays in SRAM)
    │          │  ↓ Multiply V (stays in SRAM)
    │          │  ↓ Accumulate output
    └──────────┘  ↓ Write final block to HBM

    Key difference: Intermediate results never leave SRAM!
    """
    pass
```

---

## Tiling Strategy

Flash Attention divides Q, K, V into blocks (tiles) that fit in SRAM.

### Block Size Selection

```python
import math

class TilingStrategy:
    """
    Tiling strategy for Flash Attention.

    Key question: What block size to use?
    Answer: As large as possible while fitting in SRAM.
    """

    @staticmethod
    def compute_block_size(
        sram_size: int,           # SRAM available (bytes)
        head_dim: int,            # Head dimension d
        dtype_bytes: int = 2      # FP16 = 2 bytes
    ) -> tuple[int, int]:
        """
        Compute optimal block sizes Bc (columns) and Br (rows).

        We need to fit in SRAM:
        - Q block: Br × d
        - K block: Bc × d
        - V block: Bc × d
        - S block: Br × Bc (attention scores)
        - O block: Br × d (output accumulator)

        Total: (3Br + 3Bc)d + BrBc elements

        Constraint: (3Br + 3Bc)d + BrBc ≤ SRAM_size / dtype_bytes

        For Flash Attention, they use:
        - Bc = ⌊SRAM_size / (4d)⌋
        - Br = min(Bc, d)
        """
        # Simplified: Assume Bc = Br for simplicity
        # Constraint: 6Bd + B² ≤ M where M = SRAM_size / dtype_bytes
        # Approximately: B ≈ √(M) when B² dominates
        # Or: B ≈ M/(6d) when 6Bd dominates

        M = sram_size // dtype_bytes

        # Use the formula from the paper
        Bc = M // (4 * head_dim)
        Br = min(Bc, head_dim)

        return Bc, Br

    @staticmethod
    def example_block_sizes():
        """
        Example block sizes for typical configurations.
        """
        # A100: ~20 MB SRAM per SM
        # But we can't use all of it (need space for other things)
        # Flash Attention uses ~16 KB - 64 KB per block

        configs = [
            {'gpu': 'A100', 'sram_kb': 32, 'head_dim': 64},
            {'gpu': 'A100', 'sram_kb': 32, 'head_dim': 128},
            {'gpu': 'H100', 'sram_kb': 64, 'head_dim': 128},
        ]

        for config in configs:
            sram_bytes = config['sram_kb'] * 1024
            d = config['head_dim']
            Bc, Br = TilingStrategy.compute_block_size(sram_bytes, d)

            print(f"{config['gpu']}, d={d}, SRAM={config['sram_kb']}KB:")
            print(f"  Bc={Bc}, Br={Br}")
            print(f"  Memory used: {(6*Br*d + Br*Bc)*2/1024:.1f} KB\n")


def visualize_tiling():
    """
    Visualize how matrices are divided into blocks.

    Q (N × d):  K (N × d):  V (N × d):
    ┌─┬─┬─┐    ┌─┬─┬─┐    ┌─┬─┬─┐
    ├─┼─┼─┤    ├─┼─┼─┤    ├─┼─┼─┤
    ├─┼─┼─┤    ├─┼─┼─┤    ├─┼─┼─┤
    └─┴─┴─┘    └─┴─┴─┘    └─┴─┴─┘
    Br×d blocks Bc×d blocks Bc×d blocks

    Process:
    1. Outer loop over K, V blocks (Tc = ⌈N/Bc⌉ iterations)
    2. Inner loop over Q blocks (Tr = ⌈N/Br⌉ iterations)
    3. For each (Q_block, K_block, V_block):
       - Compute attention for this block
       - Accumulate to output

    Total iterations: Tr × Tc
    But each iteration operates on small blocks in SRAM!
    """
    pass
```

---

## Online Softmax Algorithm

The key algorithmic innovation in Flash Attention is computing softmax incrementally over blocks.

### Standard Softmax

Standard softmax requires two passes over the data:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

**Two-pass algorithm:**
1. First pass: Find $m = \max_j x_j$ (for numerical stability)
2. Second pass: Compute $\sum_j e^{x_j - m}$ and $\frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$

**Problem:** We're processing in blocks, so we can't make two passes!

### Online Softmax (Safe Softmax)

```python
import torch
import math

class OnlineSoftmax:
    """
    Online (incremental) softmax algorithm.

    Key idea: Update running statistics as we see new blocks.

    Reference: "Online normalizer calculation for softmax" (Milakov & Gimelshein, 2018)
    """

    @staticmethod
    def standard_softmax(x: torch.Tensor) -> torch.Tensor:
        """
        Standard softmax (two-pass).

        Args:
            x: Input tensor [..., n]

        Returns:
            Softmax output [..., n]
        """
        # Pass 1: Find max for numerical stability
        m = x.max(dim=-1, keepdim=True).values

        # Pass 2: Compute exp and sum
        exp_x = torch.exp(x - m)
        sum_exp = exp_x.sum(dim=-1, keepdim=True)

        return exp_x / sum_exp

    @staticmethod
    def online_softmax_step(
        m_old: torch.Tensor,     # Previous max
        l_old: torch.Tensor,     # Previous sum of exp
        x_new: torch.Tensor      # New block of values
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update softmax statistics with new block.

        Mathematical derivation:

        Old state:
          m_old = max(x_1, ..., x_k)
          l_old = Σ exp(x_i - m_old)

        New values: x_{k+1}, ..., x_{k+j}

        New max:
          m_new = max(m_old, max(x_new))

        Key insight: We can update l_old to account for new max!

          l_new = l_old × exp(m_old - m_new) + Σ exp(x_new - m_new)

        This rescaling ensures numerical stability.

        Args:
            m_old: Current max value
            l_old: Current sum of exponentials (scaled by old max)
            x_new: New block of values

        Returns:
            m_new: Updated max
            l_new: Updated sum
            weights: Softmax weights for x_new
        """
        # Find max of new block
        m_new_block = x_new.max(dim=-1, keepdim=True).values

        # Global max
        m_new = torch.maximum(m_old, m_new_block)

        # Rescale old sum with new max
        # l_old was computed with m_old, now we use m_new
        l_old_rescaled = l_old * torch.exp(m_old - m_new)

        # Compute exp for new block with new max
        exp_new = torch.exp(x_new - m_new)
        l_new_block = exp_new.sum(dim=-1, keepdim=True)

        # Updated sum
        l_new = l_old_rescaled + l_new_block

        return m_new, l_new, exp_new

    @staticmethod
    def online_softmax_demo():
        """
        Demonstrate online softmax matches standard softmax.
        """
        # Generate random scores
        x = torch.randn(4, 100)

        # Standard softmax
        standard_result = OnlineSoftmax.standard_softmax(x)

        # Online softmax (process in blocks of 20)
        block_size = 20
        n_blocks = x.shape[-1] // block_size

        # Initialize
        m = torch.full((4, 1), float('-inf'))
        l = torch.zeros(4, 1)

        exp_blocks = []

        for i in range(n_blocks):
            block = x[:, i*block_size:(i+1)*block_size]
            m, l, exp_block = OnlineSoftmax.online_softmax_step(m, l, block)
            exp_blocks.append(exp_block)

        # Final normalization
        online_result = torch.cat(exp_blocks, dim=-1) / l

        # Verify they match
        print("Max difference:", (standard_result - online_result).abs().max().item())
        print("Results match:", torch.allclose(standard_result, online_result, atol=1e-5))

        return standard_result, online_result


class FlashAttentionSoftmax:
    """
    Softmax as used in Flash Attention.

    We need to compute softmax over rows of the attention matrix,
    but we're processing K, V in blocks.
    """

    @staticmethod
    def flash_attention_softmax_update(
        m_old: torch.Tensor,     # (Br, 1) - row-wise max so far
        l_old: torch.Tensor,     # (Br, 1) - row-wise sum so far
        O_old: torch.Tensor,     # (Br, d) - output accumulator
        Q_block: torch.Tensor,   # (Br, d) - query block
        K_block: torch.Tensor,   # (Bc, d) - key block
        V_block: torch.Tensor    # (Bc, d) - value block
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update Flash Attention state with new K, V block.

        This is the core of the Flash Attention algorithm.

        Args:
            m_old: Previous row-wise max
            l_old: Previous row-wise sum of exp
            O_old: Previous output accumulator
            Q_block: Current query block (Br × d)
            K_block: Current key block (Bc × d)
            V_block: Current value block (Bc × d)

        Returns:
            m_new: Updated max
            l_new: Updated sum
            O_new: Updated output
        """
        d = Q_block.shape[-1]

        # Compute attention scores for this block
        # S_block: (Br, Bc)
        S_block = torch.matmul(Q_block, K_block.T) / math.sqrt(d)

        # Update max and sum using online softmax
        m_new_block = S_block.max(dim=-1, keepdim=True).values  # (Br, 1)
        m_new = torch.maximum(m_old, m_new_block)

        # Rescale old statistics
        exp_correction = torch.exp(m_old - m_new)  # (Br, 1)
        l_old_rescaled = l_old * exp_correction

        # Compute exp for new block
        exp_S_block = torch.exp(S_block - m_new)  # (Br, Bc)
        l_new_block = exp_S_block.sum(dim=-1, keepdim=True)  # (Br, 1)
        l_new = l_old_rescaled + l_new_block

        # Update output accumulator
        # Key insight: Need to rescale old output because max changed!
        O_old_rescaled = O_old * exp_correction  # (Br, d)
        O_new_block = torch.matmul(exp_S_block, V_block)  # (Br, d)
        O_new = O_old_rescaled + O_new_block

        return m_new, l_new, O_new
```

**Key mathematical insight:**

When the max changes from $m_{old}$ to $m_{new}$, we need to rescale everything:

$$
\begin{align}
l_{new} &= l_{old} \cdot e^{m_{old} - m_{new}} + \sum_{j \in \text{new block}} e^{S_{ij} - m_{new}} \\
O_{new} &= O_{old} \cdot e^{m_{old} - m_{new}} + \sum_{j \in \text{new block}} e^{S_{ij} - m_{new}} V_j
\end{align}
$$

This allows us to maintain exact softmax while processing in blocks!

---

## Forward Pass Algorithm

Now we can put together the complete Flash Attention forward pass.

```python
import torch
import math

class FlashAttentionForward:
    """
    Complete Flash Attention forward pass implementation.

    This is a simplified but functional version for educational purposes.
    The actual implementation uses CUDA kernels for efficiency.
    """

    @staticmethod
    def forward(
        Q: torch.Tensor,     # (batch, n_heads, N, d)
        K: torch.Tensor,     # (batch, n_heads, N, d)
        V: torch.Tensor,     # (batch, n_heads, N, d)
        block_size_c: int = 64,  # Bc: block size for K, V
        block_size_r: int = 64,  # Br: block size for Q
        causal: bool = False
    ) -> torch.Tensor:
        """
        Flash Attention forward pass.

        Algorithm (simplified):

        1. Divide Q into Tr = ⌈N/Br⌉ blocks: Q_1, ..., Q_Tr
        2. Divide K, V into Tc = ⌈N/Bc⌉ blocks: K_1, V_1, ..., K_Tc, V_Tc
        3. For each Q block i:
             Initialize O_i = 0, l_i = 0, m_i = -∞
             For each K, V block j:
               If causal and j > i: skip
               Compute S_ij = Q_i K_j^T / √d
               Update m_i, l_i, O_i using online softmax
             Normalize: O_i = O_i / l_i
        4. Concatenate all O_i blocks

        Memory complexity: O(N) instead of O(N²)
        - Never store full N×N attention matrix
        - Only store Br×Bc blocks in SRAM at a time
        """
        batch, n_heads, N, d = Q.shape

        # Number of blocks
        Tr = math.ceil(N / block_size_r)
        Tc = math.ceil(N / block_size_c)

        # Initialize output
        O = torch.zeros_like(Q)

        # Process each Q block
        for i in range(Tr):
            # Extract Q block
            start_r = i * block_size_r
            end_r = min((i + 1) * block_size_r, N)
            Q_block = Q[:, :, start_r:end_r, :]  # (batch, n_heads, Br, d)

            Br_actual = end_r - start_r

            # Initialize statistics for this Q block
            m_i = torch.full(
                (batch, n_heads, Br_actual, 1),
                float('-inf'),
                device=Q.device,
                dtype=Q.dtype
            )
            l_i = torch.zeros(
                (batch, n_heads, Br_actual, 1),
                device=Q.device,
                dtype=Q.dtype
            )
            O_i = torch.zeros(
                (batch, n_heads, Br_actual, d),
                device=Q.device,
                dtype=Q.dtype
            )

            # Process each K, V block
            for j in range(Tc):
                # Causal masking: skip future blocks
                if causal and j > i:
                    break

                # Extract K, V block
                start_c = j * block_size_c
                end_c = min((j + 1) * block_size_c, N)
                K_block = K[:, :, start_c:end_c, :]  # (batch, n_heads, Bc, d)
                V_block = V[:, :, start_c:end_c, :]

                Bc_actual = end_c - start_c

                # Compute attention scores for this block
                S_ij = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(d)
                # S_ij: (batch, n_heads, Br, Bc)

                # Apply causal mask within block if needed
                if causal and i == j:
                    # Within-block causal mask
                    mask = torch.triu(
                        torch.ones(Br_actual, Bc_actual, device=Q.device),
                        diagonal=1
                    ).bool()
                    S_ij = S_ij.masked_fill(mask, float('-inf'))

                # Update statistics using online softmax
                m_ij = S_ij.max(dim=-1, keepdim=True).values  # (batch, n_heads, Br, 1)
                m_i_new = torch.maximum(m_i, m_ij)

                # Rescale old values
                alpha = torch.exp(m_i - m_i_new)  # (batch, n_heads, Br, 1)

                # Compute new exp values
                exp_S_ij = torch.exp(S_ij - m_i_new)  # (batch, n_heads, Br, Bc)

                # Update sum
                l_i = l_i * alpha + exp_S_ij.sum(dim=-1, keepdim=True)

                # Update output
                O_i = O_i * alpha + torch.matmul(exp_S_ij, V_block)

                # Update max
                m_i = m_i_new

            # Final normalization for this Q block
            O_i = O_i / l_i

            # Store result
            O[:, :, start_r:end_r, :] = O_i

        return O

    @staticmethod
    def test_correctness():
        """
        Test Flash Attention matches standard attention.
        """
        torch.manual_seed(42)

        batch = 2
        n_heads = 4
        seq_len = 256
        head_dim = 64

        Q = torch.randn(batch, n_heads, seq_len, head_dim)
        K = torch.randn(batch, n_heads, seq_len, head_dim)
        V = torch.randn(batch, n_heads, seq_len, head_dim)

        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1)
        output_standard = torch.matmul(attn, V)

        # Flash attention
        output_flash = FlashAttentionForward.forward(
            Q, K, V,
            block_size_c=64,
            block_size_r=64
        )

        # Compare
        max_diff = (output_standard - output_flash).abs().max().item()
        print(f"Max difference: {max_diff}")
        print(f"Results match: {torch.allclose(output_standard, output_flash, atol=1e-4)}")

        return output_standard, output_flash


# Example usage
if __name__ == "__main__":
    print("Testing Flash Attention forward pass...")
    FlashAttentionForward.test_correctness()
```

**Complexity analysis:**

- **Memory:** O(N) - only store O_i blocks, not full N×N matrix
- **HBM reads:** O(N²d) - still need to read Q, K, V multiple times across blocks
- **HBM writes:** O(Nd) - only write final output
- **FLOPs:** O(N²d) - same as standard attention (no approximation!)

The key win is in HBM access pattern, not FLOP count.

---

## Backward Pass and Recomputation

Flash Attention achieves O(N) memory in the backward pass by recomputing attention on-the-fly.

```python
class FlashAttentionBackward:
    """
    Flash Attention backward pass with recomputation.

    Key insight: Don't store attention matrix, recompute it!

    Standard attention backward:
    - Store attention matrix P = softmax(QK^T / √d)
    - Memory: O(N²)

    Flash attention backward:
    - Store only O, m, l (the softmax statistics)
    - Recompute attention during backward
    - Memory: O(N)

    Why recomputation is faster:
    - Saving O(N²) memory saves HBM bandwidth
    - Recomputation happens in SRAM (fast)
    - Modern GPUs are compute-bound → free compute!
    """

    @staticmethod
    def forward_with_cache(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        block_size: int = 64
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass that saves minimal state for backward.

        Saved state:
        - O: Output (Nd)
        - m: Row-wise max (N)
        - l: Row-wise sum (N)
        - Q, K, V: Inputs (3Nd)

        Total: O(Nd) memory vs O(N²) for standard attention

        Returns:
            output: Attention output
            cache: Dictionary with saved tensors
        """
        # Run forward (implementation same as before)
        output = FlashAttentionForward.forward(Q, K, V, block_size, block_size)

        # In actual implementation, we'd save m and l computed during forward
        # For this demo, we'll recompute them
        batch, n_heads, N, d = Q.shape

        # Compute and save statistics (in real impl, these come from forward)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        m = scores.max(dim=-1, keepdim=True).values  # (batch, n_heads, N, 1)
        exp_scores = torch.exp(scores - m)
        l = exp_scores.sum(dim=-1, keepdim=True)  # (batch, n_heads, N, 1)

        cache = {
            'Q': Q,
            'K': K,
            'V': V,
            'O': output,
            'm': m,
            'l': l,
            'block_size': block_size
        }

        return output, cache

    @staticmethod
    def backward(
        dO: torch.Tensor,     # Gradient w.r.t. output
        cache: dict           # Saved tensors from forward
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass with recomputation.

        Given dL/dO, compute dL/dQ, dL/dK, dL/dV.

        Key equations (from standard attention backward):

        Let P = softmax(S) where S = QK^T / √d

        dL/dV = P^T @ dL/dO
        dL/dP = dL/dO @ V^T
        dL/dS = P ⊙ (dL/dP - (dL/dP ⊙ P).sum(axis=-1, keepdims=True))
        dL/dQ = dL/dS @ K / √d
        dL/dK = dL/dS^T @ Q / √d

        The trick: We recompute P from saved m, l instead of storing it!
        """
        Q = cache['Q']
        K = cache['K']
        V = cache['V']
        m = cache['m']
        l = cache['l']

        batch, n_heads, N, d = Q.shape

        # Recompute attention matrix P from saved statistics
        # This is the key: we're trading compute for memory
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        P = torch.exp(scores - m) / l  # Recomputed from m, l

        # Now compute gradients using standard attention backward
        # dV = P^T @ dO
        dV = torch.matmul(P.transpose(-2, -1), dO)

        # dP = dO @ V^T
        dP = torch.matmul(dO, V.transpose(-2, -1))

        # Softmax backward: dS = P ⊙ (dP - (dP ⊙ P).sum(-1, keepdim=True))
        dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))

        # dQ = dS @ K / √d
        dQ = torch.matmul(dS, K) / math.sqrt(d)

        # dK = dS^T @ Q / √d
        dK = torch.matmul(dS.transpose(-2, -1), Q) / math.sqrt(d)

        return dQ, dK, dV

    @staticmethod
    def memory_comparison():
        """
        Compare memory requirements: standard vs flash backward.
        """
        batch = 8
        n_heads = 32
        seq_len = 4096
        head_dim = 128

        # Standard attention backward
        attention_matrix = batch * n_heads * seq_len * seq_len * 2  # FP16
        inputs = batch * n_heads * 3 * seq_len * head_dim * 2
        standard_memory = (attention_matrix + inputs) / 1e9

        # Flash attention backward
        # Only need to store: Q, K, V, O, m, l
        flash_inputs = batch * n_heads * 3 * seq_len * head_dim * 2  # Q, K, V
        flash_output = batch * n_heads * seq_len * head_dim * 2      # O
        flash_stats = batch * n_heads * seq_len * 2 * 2              # m, l
        flash_memory = (flash_inputs + flash_output + flash_stats) / 1e9

        print(f"Sequence length: {seq_len}")
        print(f"Standard backward memory: {standard_memory:.2f} GB")
        print(f"Flash backward memory: {flash_memory:.2f} GB")
        print(f"Memory reduction: {standard_memory / flash_memory:.1f}x")
```

**Why recomputation is faster:**

1. **Memory bandwidth bound:** Saving/loading O(N²) attention matrix saturates HBM
2. **Compute abundance:** Modern GPUs have excess compute capacity
3. **SRAM recomputation:** Recomputing in SRAM is much faster than HBM access
4. **Net effect:** 2-4x speedup despite doing more FLOPs!

---

## Implementation Considerations

### CUDA Implementation

The Python implementations above are for understanding. Real Flash Attention is implemented in CUDA for performance.

```python
class CUDAImplementationNotes:
    """
    Notes on CUDA implementation of Flash Attention.

    The actual implementation requires careful optimization:
    """

    @staticmethod
    def kernel_fusion():
        """
        Kernel fusion strategy.

        Standard attention: 3+ kernel launches
        - MatMul (QK^T)
        - Softmax
        - MatMul (PV)

        Flash Attention: 1 kernel launch
        - All operations fused
        - No intermediate HBM writes

        Benefits:
        - Fewer kernel launch overheads
        - No HBM writes between operations
        - Better instruction-level parallelism
        """
        pass

    @staticmethod
    def memory_layout():
        """
        Optimal memory layout for SRAM usage.

        Challenges:
        - SRAM is tiny (~20 MB per SM on A100)
        - Need to fit Q, K, V, S blocks simultaneously
        - Layout matters for memory access patterns

        Techniques:
        - Use shared memory for blocks
        - Careful padding to avoid bank conflicts
        - Row-major vs column-major layout choices
        """
        pass

    @staticmethod
    def thread_block_mapping():
        """
        Mapping computation to CUDA thread blocks.

        Flash Attention 1:
        - One thread block per attention head per Q block
        - Parallelize over batch and heads
        - Sequential over K, V blocks (online softmax)

        Flash Attention 2:
        - Improved parallelization across sequence length
        - Non-matmul operations minimized
        - Better work partitioning

        Flash Attention 3 (Hopper-specific):
        - Warp specialization
        - Asynchronous GEMM operations
        - Overlapping compute and memory
        """
        pass

    @staticmethod
    def numerical_precision():
        """
        Numerical precision considerations.

        Challenges:
        - FP16/BF16 have limited dynamic range
        - Softmax requires exp (can overflow/underflow)
        - Accumulation errors with long sequences

        Solutions:
        - Always compute max in FP32 for stability
        - Use higher precision for accumulation
        - Careful order of operations

        Flash Attention 3 adds:
        - FP8 support (E4M3 and E5M2)
        - Block-wise scaling for FP8
        """
        pass


class PerformanceOptimizations:
    """
    Performance optimization techniques in Flash Attention.
    """

    @staticmethod
    def block_size_tuning():
        """
        Auto-tuning block sizes for different GPUs.

        Factors:
        - Available SRAM per SM
        - Number of SMs
        - Head dimension
        - Sequence length

        Flash Attention uses heuristics to select optimal Br and Bc.
        Different for different GPU architectures (Ampere vs Hopper).
        """
        pass

    @staticmethod
    def causal_masking_optimization():
        """
        Optimizing causal (autoregressive) attention.

        Standard approach: Apply mask to full N×N matrix

        Flash Attention optimization:
        - Skip unnecessary K, V blocks entirely
        - For Q block i, only process K, V blocks 0..i
        - Reduces computation by ~50% for causal attention
        """
        pass

    @staticmethod
    def multi_query_attention():
        """
        Flash Attention with Multi-Query and Grouped-Query Attention.

        MQA/GQA: Fewer K, V heads than Q heads (see [Multi-Head Attention](04-multi-head-attention.md))

        Flash Attention adapts:
        - K, V blocks are smaller
        - Can process more K, V per iteration
        - Even better memory efficiency
        """
        pass
```

---

## FlashAttention Versions

Flash Attention has evolved through three major versions, each with significant improvements.

### FlashAttention 1 (2022)

```python
class FlashAttention1:
    """
    FlashAttention 1: Original algorithm.

    Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    Authors: Dao, Fu, Ermon, Rudra, Ré (Stanford, 2022)

    Key contributions:
    1. IO-aware algorithm design
    2. Tiling strategy
    3. Online softmax
    4. Recomputation in backward

    Performance:
    - 2-4x faster than PyTorch attention
    - O(N) memory vs O(N²)
    - Exact (no approximation)

    Limitations:
    - Head dimension limited to 64 or 128
    - Not fully optimized for all sequence lengths
    - Ampere architecture (A100) focus
    """

    speedup_vs_pytorch = "2-4x"
    memory_reduction = "N² → N"
    supported_head_dims = [64, 128]
    target_architecture = "NVIDIA Ampere (A100)"


class FlashAttention2:
    """
    FlashAttention 2: Better parallelism and work partitioning.

    Paper: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
    Authors: Dao (Together AI, 2023)

    Improvements over FA1:

    1. Better parallelization:
       - Parallelize over sequence length, not just batch/heads
       - Reduces number of non-matmul FLOPs
       - Better GPU utilization

    2. Work partitioning:
       - Better distribution of work across SMs
       - Reduced thread block synchronization

    3. Support for longer head dimensions:
       - Up to 256 (vs 128 in FA1)
       - Important for models with larger heads

    4. Causal attention optimization:
       - More efficient masking
       - Skips unnecessary blocks

    Performance:
    - ~2x faster than FA1
    - 4-8x faster than PyTorch
    - Up to 225 TFLOPs/s on A100 (vs 115 for FA1)
    """

    speedup_vs_fa1 = "~2x"
    speedup_vs_pytorch = "4-8x"
    supported_head_dims = [64, 128, 256]
    target_architecture = "NVIDIA Ampere (A100, RTX 3090, RTX 4090)"

    @staticmethod
    def improvements_summary():
        """
        Summary of improvements in FA2 vs FA1.

        | Metric | FA1 | FA2 | Improvement |
        |--------|-----|-----|-------------|
        | TFLOPs/s (A100, seq=2K, d=64) | 115 | 225 | 1.96x |
        | TFLOPs/s (A100, seq=8K, d=64) | 120 | 230 | 1.92x |
        | Non-matmul FLOPs reduction | - | 2x | Better |
        | Parallelism | Batch/head | + Sequence | More |
        | Max head dim | 128 | 256 | 2x |
        """
        pass


class FlashAttention3:
    """
    FlashAttention 3: Hopper-optimized with asynchronous operations.

    Paper: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
    Authors: Shah, Dao, et al. (Together AI & Colfax Research, 2024)

    Hopper-specific optimizations:

    1. Warp specialization:
       - Different warps do different tasks
       - Producer warps: Load data
       - Consumer warps: Compute
       - Overlapping for better throughput

    2. Asynchronous GEMM:
       - Use Hopper's async Tensor Core instructions
       - Pipeline data loading and computation
       - Hide memory latency

    3. Low-precision support:
       - FP8 (E4M3 and E5M2)
       - Block-wise scaling
       - Maintains accuracy

    4. New Hopper instructions:
       - WGMMA (Warp Group Matrix Multiply-Accumulate)
       - TMA (Tensor Memory Accelerator)
       - Async barrier synchronization

    Performance:
    - ~75% of Tensor Core theoretical max (vs 35% for FA2 on H100)
    - 1.5-2x faster than FA2 on H100
    - FP8: 2.6x faster than FA2 BF16

    Limitations:
    - Requires Hopper architecture (H100, H200)
    - More complex implementation
    """

    speedup_vs_fa2_h100 = "1.5-2x"
    tensor_core_utilization = "~75%"
    supported_precisions = ["FP16", "BF16", "FP8_E4M3", "FP8_E5M2"]
    target_architecture = "NVIDIA Hopper (H100, H200)"

    @staticmethod
    def performance_comparison():
        """
        Performance comparison on H100.

        Configuration: Batch=1, Heads=32, Seq=8192, Head_dim=128

        | Implementation | Precision | TFLOPs/s | % of Peak |
        |----------------|-----------|----------|-----------|
        | PyTorch SDPA   | BF16      | 450      | 22%       |
        | FlashAttention 2 | BF16    | 700      | 35%       |
        | FlashAttention 3 | BF16    | 1,500    | 75%       |
        | FlashAttention 3 | FP8     | 2,800    | 70% (of FP8 peak) |

        Note: FA3 achieves near-optimal hardware utilization!
        """
        pass


def flashattention_version_comparison():
    """
    Summary comparison of FlashAttention versions.

    | Feature | FA1 (2022) | FA2 (2023) | FA3 (2024) |
    |---------|------------|------------|------------|
    | **Target GPU** | A100 | A100/RTX | H100 |
    | **Speedup vs PyTorch** | 2-4x | 4-8x | 8-15x |
    | **Max head dim** | 128 | 256 | 256 |
    | **Precision** | FP16/BF16 | FP16/BF16 | FP16/BF16/FP8 |
    | **Tensor Core util** | ~30% | ~35% | ~75% |
    | **Key innovation** | Tiling + online softmax | Better parallelism | Async + warp specialization |
    | **Backward pass** | Recomputation | Improved recomputation | Optimized recomputation |
    | **Production ready** | Yes | Yes | Yes (H100 only) |
    """
    pass
```

**Key papers:**
- [FlashAttention (v1)](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608) (Shah et al., 2024)

---

## Using Flash Attention in Practice

### PyTorch Integration

```python
import torch
import torch.nn.functional as F

def use_flash_attention_pytorch():
    """
    Using Flash Attention in PyTorch 2.0+.

    PyTorch 2.0+ includes scaled_dot_product_attention (SDPA)
    which automatically uses Flash Attention when available.
    """

    # Create sample Q, K, V
    batch = 4
    n_heads = 8
    seq_len = 2048
    head_dim = 64

    Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

    # Method 1: Use F.scaled_dot_product_attention (automatic backend selection)
    output = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True  # For autoregressive models
    )

    # Method 2: Force Flash Attention backend
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,       # Use Flash Attention
        enable_math=False,       # Disable slow math backend
        enable_mem_efficient=False  # Disable memory-efficient attention
    ):
        output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    return output


def check_flash_attention_available():
    """
    Check if Flash Attention is available in your PyTorch installation.
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return False

    # Check if SDPA is available
    if not hasattr(F, 'scaled_dot_product_attention'):
        print("scaled_dot_product_attention not available. Update PyTorch to 2.0+")
        return False

    # Try to use Flash Attention
    try:
        Q = torch.randn(1, 1, 128, 64, device='cuda', dtype=torch.float16)
        K = torch.randn(1, 1, 128, 64, device='cuda', dtype=torch.float16)
        V = torch.randn(1, 1, 128, 64, device='cuda', dtype=torch.float16)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            _ = F.scaled_dot_product_attention(Q, K, V)

        print("Flash Attention is available!")
        return True
    except Exception as e:
        print(f"Flash Attention not available: {e}")
        return False


class FlashAttentionModule(torch.nn.Module):
    """
    Multi-head attention module using Flash Attention.

    Drop-in replacement for standard MultiheadAttention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Projections
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass using Flash Attention.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            is_causal: Whether to apply causal masking
            attn_mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply Flash Attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output


# Example usage
def example_usage():
    """Example of using FlashAttentionModule."""
    model = FlashAttentionModule(
        d_model=512,
        n_heads=8,
        dropout=0.1
    ).cuda()

    x = torch.randn(4, 1024, 512).cuda()

    # Forward pass
    output = model(x, is_causal=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
```

### Using the Official Flash Attention Library

```python
def use_official_flash_attention():
    """
    Using the official Flash Attention library.

    Installation:
        pip install flash-attn --no-build-isolation

    Note: Requires CUDA toolkit and compatible GPU (Ampere or newer).
    """
    try:
        from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

        # Example 1: Separate Q, K, V tensors
        batch = 4
        seqlen = 2048
        n_heads = 8
        head_dim = 64

        Q = torch.randn(batch, seqlen, n_heads, head_dim, device='cuda', dtype=torch.float16)
        K = torch.randn(batch, seqlen, n_heads, head_dim, device='cuda', dtype=torch.float16)
        V = torch.randn(batch, seqlen, n_heads, head_dim, device='cuda', dtype=torch.float16)

        # Flash Attention forward
        output = flash_attn_func(
            Q, K, V,
            dropout_p=0.0,
            softmax_scale=None,  # Defaults to 1/√d
            causal=True,
            return_attn_probs=False
        )

        print(f"Output shape: {output.shape}")

        # Example 2: Packed QKV (more efficient)
        qkv = torch.randn(batch, seqlen, 3, n_heads, head_dim, device='cuda', dtype=torch.float16)

        output_packed = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            causal=True
        )

        print("Flash Attention executed successfully!")

    except ImportError:
        print("flash-attn not installed. Install with: pip install flash-attn --no-build-isolation")
```

---

## Theoretical Analysis

### IO Complexity Analysis

```python
class IOComplexityAnalysis:
    """
    Formal IO complexity analysis of Flash Attention.

    Model: We analyze the number of HBM accesses (reads + writes).
    This is the dominant cost on modern GPUs.
    """

    @staticmethod
    def standard_attention_io(N: int, d: int, M: int) -> tuple[int, str]:
        """
        IO complexity of standard attention.

        Args:
            N: Sequence length
            d: Head dimension
            M: SRAM size (in elements)

        Returns:
            Number of HBM accesses and complexity expression
        """
        # Forward pass:
        # - Read Q, K: 2Nd
        # - Write S = QK^T: N²
        # - Read S: N²
        # - Write P = softmax(S): N²
        # - Read P, V: N² + Nd
        # - Write O: Nd

        hbm_accesses = 2*N*d + N**2 + N**2 + N**2 + N**2 + N*d + N*d
        hbm_accesses = 4*N*d + 4*N**2

        return hbm_accesses, "Θ(Nd + N²)"

    @staticmethod
    def flash_attention_io(N: int, d: int, M: int) -> tuple[int, str]:
        """
        IO complexity of Flash Attention.

        Key insight: With block size B = Θ(√M), we achieve O(N²d²/M) HBM accesses.

        For typical M = Θ(d), this becomes O(N²d/√M) = O(N²√d).

        Since d << N, this is much better than standard Θ(N²).
        """
        import math

        # Block size
        B = int(math.sqrt(M))

        # Number of blocks
        num_blocks = math.ceil(N / B)

        # For each of Tr = N/B row blocks of Q:
        #   For each of Tc = N/B column blocks of K, V:
        #     Read Q block: Bd
        #     Read K block: Bd
        #     Read V block: Bd
        #     Write O block: Bd (partial, but amortized)

        # Total: Tr × Tc × (3Bd + Bd) = (N/B)² × 4Bd
        hbm_accesses = (N / B) ** 2 * 4 * B * d
        hbm_accesses = 4 * N**2 * d / B

        # With B = Θ(√M) and M = Θ(d):
        # HBM = Θ(N²d / √d) = Θ(N²√d)

        return hbm_accesses, f"Θ(N²d²/M) = Θ(N²√d) when M=Θ(d)"

    @staticmethod
    def comparison(N: int = 4096, d: int = 64, M: int = None):
        """
        Compare IO complexity of standard vs Flash Attention.
        """
        import math

        if M is None:
            M = d  # Typical case

        standard_io, standard_expr = IOComplexityAnalysis.standard_attention_io(N, d, M)
        flash_io, flash_expr = IOComplexityAnalysis.flash_attention_io(N, d, M)

        print(f"Sequence length N = {N}, Head dimension d = {d}, SRAM M = {M}")
        print(f"\nStandard Attention:")
        print(f"  HBM accesses: {standard_io:,}")
        print(f"  Complexity: {standard_expr}")
        print(f"\nFlash Attention:")
        print(f"  HBM accesses: {flash_io:,.0f}")
        print(f"  Complexity: {flash_expr}")
        print(f"\nReduction factor: {standard_io / flash_io:.2f}x")

        # For long sequences, the gap grows
        print(f"\nAs N grows (with d={d} fixed):")
        for seq_len in [1024, 4096, 16384, 65536]:
            std_io, _ = IOComplexityAnalysis.standard_attention_io(seq_len, d, M)
            fa_io, _ = IOComplexityAnalysis.flash_attention_io(seq_len, d, M)
            print(f"  N={seq_len:>5}: Standard={std_io:>15,}, Flash={fa_io:>15,.0f}, Ratio={std_io/fa_io:>5.1f}x")


# Run analysis
IOComplexityAnalysis.comparison()
```

**Theoretical result:**

For standard attention:
- **IO complexity:** $\Theta(N^2 + Nd)$ where $N$ is sequence length, $d$ is head dimension
- **Dominated by:** $\Theta(N^2)$ for long sequences

For Flash Attention:
- **IO complexity:** $\Theta\left(\frac{N^2 d^2}{M}\right)$ where $M$ is SRAM size
- **With** $M = \Theta(d)$: $\Theta(N^2 \sqrt{d})$
- **Reduction:** $\Theta(\sqrt{d})$ improvement (e.g., 8x for $d=64$)

---

## Extensions and Variants

Flash Attention has inspired many extensions and variants.

```python
class FlashAttentionExtensions:
    """
    Extensions and variants of Flash Attention.
    """

    @staticmethod
    def flash_decoding():
        """
        Flash-Decoding: Optimized for generation (decoding).

        Problem: During generation, we append one token at a time.
        - Query: 1 token
        - Key, Value: All previous tokens (growing)

        Standard Flash Attention parallelizes over queries,
        but with 1 query token, we have no parallelism!

        Flash-Decoding solution:
        - Parallelize over K, V instead
        - Split K, V into blocks
        - Compute attention for each block in parallel
        - Reduce using online softmax

        Result: Up to 8x faster decoding for long contexts.

        Paper: "Flash-Decoding for long-context inference" (2023)
        """
        pass

    @staticmethod
    def paged_flash_attention():
        """
        Paged Flash Attention: Combining Flash Attention with PagedAttention.

        PagedAttention (from vLLM) manages KV cache in pages (see [Hardware, Quantization, and Training Optimization](31-hardware-quantization-optimization.md))

        Paged Flash Attention:
        - Use Flash Attention for computation
        - Use paged memory management for KV cache
        - Best of both worlds!

        Enables:
        - Long context with Flash Attention efficiency
        - Memory efficient KV cache management
        - Continuous batching for inference

        Used in: vLLM, TensorRT-LLM
        """
        pass

    @staticmethod
    def block_sparse_flash_attention():
        """
        Block-Sparse Flash Attention.

        Idea: Combine Flash Attention with block sparsity patterns.

        Applications:
        - Longformer-style attention (local + global)
        - BigBird attention patterns
        - Custom sparsity patterns

        Implementation:
        - Skip blocks that are masked out
        - Only process non-zero blocks
        - Maintains Flash Attention efficiency

        Paper: "Flash-Attention with Block-Sparse Attention" (Dao et al.)
        """
        pass

    @staticmethod
    def multi_query_flash_attention():
        """
        Flash Attention for Multi-Query and Grouped-Query Attention.

        MQA/GQA: Fewer K, V heads than Q heads (see [Multi-Head Attention](04-multi-head-attention.md))

        Optimizations:
        - K, V blocks are smaller
        - Can fit more K, V in SRAM
        - Better K, V reuse across Q heads

        Result: Even faster than standard Flash Attention!
        """
        pass

    @staticmethod
    def ring_attention():
        """
        Ring Attention: Distributed Flash Attention.

        For sequences longer than single-GPU memory:
        - Split sequence across multiple GPUs
        - Pass K, V blocks in a ring
        - Each GPU processes its Q block with all K, V

        Enables:
        - Multi-million token contexts
        - Distributed training on long sequences

        Paper: "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)
        """
        pass


class FlashAttentionVariants:
    """
    Attention mechanisms inspired by Flash Attention principles.
    """

    @staticmethod
    def memory_efficient_attention():
        """
        Memory-Efficient Attention (Xformers).

        Similar to Flash Attention but different approach:
        - Focuses on reducing peak memory
        - Uses gradient checkpointing strategically
        - Supports more flexible attention masks

        Trade-offs vs Flash Attention:
        - More flexible (arbitrary masks)
        - Slightly slower
        - Lower memory peak

        Used in: Stable Diffusion, many vision models
        """
        pass

    @staticmethod
    def fused_attention():
        """
        Fused Attention (NVIDIA Apex).

        NVIDIA's optimized attention implementation:
        - Fused CUDA kernels
        - Similar principles to Flash Attention
        - Integrated with Apex mixed precision training

        Superseded by Flash Attention in most use cases.
        """
        pass
```

---

## Summary

### Key Takeaways for Interviews

1. **The Core Problem**
   - Standard attention is **memory-bound**, not compute-bound
   - O(N²) attention matrix is too large for long sequences
   - HBM bandwidth is the bottleneck, not FLOPs

2. **Flash Attention's Solution**
   - **Tiling:** Process Q, K, V in blocks that fit in SRAM
   - **Kernel fusion:** Never write intermediate attention matrix to HBM
   - **Online softmax:** Compute softmax incrementally across blocks
   - **Recomputation:** Recompute attention in backward pass (faster than storing!)

3. **Key Benefits**
   - **Memory:** O(N) instead of O(N²)
   - **Speed:** 2-8x faster than standard attention
   - **Exact:** No approximation, mathematically identical
   - **Long context:** Enables sequences up to 100K+ tokens

4. **Three Versions**
   - **FA1 (2022):** Original, 2-4x speedup
   - **FA2 (2023):** Better parallelism, 4-8x speedup
   - **FA3 (2024):** Hopper-optimized, 8-15x speedup, FP8 support

5. **Practical Considerations**
   - Use PyTorch 2.0+ `F.scaled_dot_product_attention` (automatic Flash Attention)
   - Requires modern GPU (Ampere or newer)
   - Critical for long-context models (>4K tokens)
   - Combines well with other optimizations (PagedAttention, quantization)

### Mental Model

Think of Flash Attention as:
- **Compiler optimization** for attention: Kernel fusion + memory optimization
- **Streaming algorithm:** Process data in chunks, maintain running statistics
- **Hardware-aware design:** Designed around GPU memory hierarchy, not just algorithmic complexity

### Interview Questions You Should Be Able to Answer

1. Why is standard attention slow despite having the same O(N²d) FLOP count as Flash Attention?
2. What is online softmax and why is it necessary for Flash Attention?
3. Why does Flash Attention recompute attention in the backward pass instead of storing it?
4. What's the difference between FlashAttention 2 and 3?
5. How does Flash Attention enable longer context lengths?

---

## References

### Core Papers

1. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao, Fu, Ermon, Rudra, Ré, 2022)
2. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, 2023)
3. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (Shah, Dao, et al., 2024)

### Related Work

4. [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, 2018) - Online softmax algorithm
5. [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682) (Rabe & Staats, 2021) - Earlier memory-efficient attention
6. [Memory-Efficient Attention (Xformers)](https://github.com/facebookresearch/xformers) - Alternative approach

### Extensions

7. [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) (2023)
8. [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (Liu et al., 2023)
9. [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023)

### Implementation Resources

10. [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention) - Official implementation
11. [PyTorch SDPA Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
12. [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) - Excellent explanation of GPU optimization

---

## Exercises

1. **Memory Calculation**: Calculate the memory saved by Flash Attention vs standard attention for:
   - Sequence length: 16K
   - Batch size: 8
   - Number of heads: 32
   - Head dimension: 128
   - Precision: FP16

2. **Online Softmax**: Implement online softmax from scratch and verify it matches standard softmax for random inputs.

3. **Block Size Selection**: Given SRAM size of 64 KB and head dimension of 128, calculate the optimal block sizes Bc and Br. How many elements can you fit?

4. **IO Analysis**: For sequence length 8K and head dimension 64, calculate the number of HBM accesses for:
   - Standard attention (show all reads and writes)
   - Flash Attention with block size 128
   - What's the reduction factor?

5. **Causal Attention Optimization**: Explain how Flash Attention can skip computation for causal attention. For sequence length N and block size B, how many blocks can be skipped?

6. **Backward Pass**: Why is recomputation in the backward pass actually faster than storing the attention matrix? Calculate the memory bandwidth savings.

7. **Comparison**: Compare Flash Attention with other attention optimization techniques (sparse attention, linear attention). When would you use each?

8. **Long Context**: You need to process 1M token context. Standard attention requires 1TB of memory for the attention matrix. How does Flash Attention make this feasible? What are the remaining bottlenecks?
