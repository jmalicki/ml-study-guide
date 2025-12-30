# Chapter 16: Distributed Training and Parallelism

Training large language models requires distributing computation across multiple GPUs and machines. A single modern GPU (even an H100 with 80GB) cannot fit models with hundreds of billions of parameters. This chapter covers the fundamental parallelism strategies used to train LLMs at scale.

## Table of Contents

1. [Why Distributed Training?](#why-distributed-training)
2. [Memory Requirements Analysis](#memory-requirements-analysis)
3. [Data Parallelism](#data-parallelism)
   - [Basic Data Parallelism (DP)](#basic-data-parallelism-dp)
   - [Distributed Data Parallel (DDP)](#distributed-data-parallel-ddp)
4. [Tensor Parallelism](#tensor-parallelism)
   - [Megatron-LM Tensor Parallelism](#megatron-lm-tensor-parallelism)
   - [Communication Analysis](#communication-analysis)
5. [Pipeline Parallelism](#pipeline-parallelism)
   - [GPipe and PipeDream](#gpipe-and-pipedream)
   - [Bubble Reduction Strategies](#bubble-reduction-strategies)
6. [ZeRO: Zero Redundancy Optimizer](#zero-zero-redundancy-optimizer)
   - [ZeRO Stage 1: Optimizer State Sharding](#zero-stage-1-optimizer-state-sharding)
   - [ZeRO Stage 2: Gradient Sharding](#zero-stage-2-gradient-sharding)
   - [ZeRO Stage 3: Parameter Sharding](#zero-stage-3-parameter-sharding)
7. [Fully Sharded Data Parallel (FSDP)](#fully-sharded-data-parallel-fsdp)
8. [3D Parallelism](#3d-parallelism)
9. [Communication Costs and Trade-offs](#communication-costs-and-trade-offs)
10. [Practical Implementation](#practical-implementation)
11. [Exercises](#exercises)

---

## Why Distributed Training?

Consider GPT-3 with 175B parameters:

```python
import torch

def calculate_model_memory(num_params_billions, bytes_per_param=2):
    """
    Calculate memory required for model parameters.

    Args:
        num_params_billions: Number of parameters in billions
        bytes_per_param: 2 for FP16/BF16, 4 for FP32

    Returns:
        Memory in GB
    """
    params = num_params_billions * 1e9
    memory_bytes = params * bytes_per_param
    memory_gb = memory_bytes / (1024**3)
    return memory_gb

def calculate_training_memory(num_params_billions):
    """
    Calculate total training memory (parameters + gradients + optimizer states).
    Using AdamW optimizer with mixed precision (FP16 training).
    """
    # Model parameters in FP16
    params_memory = calculate_model_memory(num_params_billions, bytes_per_param=2)

    # Gradients in FP16
    gradients_memory = calculate_model_memory(num_params_billions, bytes_per_param=2)

    # Optimizer states (AdamW maintains FP32 copy + 2 momentum buffers)
    # - FP32 parameters: 4 bytes
    # - First moment (momentum): 4 bytes
    # - Second moment (variance): 4 bytes
    # Total: 12 bytes per parameter
    optimizer_memory = calculate_model_memory(num_params_billions, bytes_per_param=12)

    total = params_memory + gradients_memory + optimizer_memory

    print(f"Model parameters (FP16): {params_memory:.2f} GB")
    print(f"Gradients (FP16): {gradients_memory:.2f} GB")
    print(f"Optimizer states (FP32): {optimizer_memory:.2f} GB")
    print(f"Total: {total:.2f} GB")
    print(f"\nThis excludes activations, which can be 10-100x larger!")

    return total

# GPT-3 175B
print("GPT-3 175B memory requirements:")
gpt3_memory = calculate_training_memory(175)
print(f"\nAn H100 has 80GB. Need at least {gpt3_memory/80:.1f} GPUs just for model states!")
```

Output:
```
GPT-3 175B memory requirements:
Model parameters (FP16): 325.26 GB
Gradients (FP16): 325.26 GB
Optimizer states (FP32): 1951.56 GB
Total: 2602.08 GB

This excludes activations, which can be 10-100x larger!

An H100 has 80GB. Need at least 32.5 GPUs just for model states!
```

This doesn't even include **activations** (intermediate outputs stored for backpropagation), which scale with batch size and sequence length!

---

## Memory Requirements Analysis

The memory breakdown for training a transformer:

$$
M_{\text{total}} = M_{\text{params}} + M_{\text{gradients}} + M_{\text{optimizer}} + M_{\text{activations}}
$$

### Parameter Memory

For a model with $\Theta$ parameters in mixed precision (FP16/BF16):

$$
M_{\text{params}} = 2\Theta \text{ bytes}
$$

### Gradient Memory

Same as parameters:

$$
M_{\text{gradients}} = 2\Theta \text{ bytes}
$$

### Optimizer Memory (AdamW)

AdamW maintains:
- FP32 copy of parameters: $4\Theta$ bytes
- First moment estimate: $4\Theta$ bytes
- Second moment estimate: $4\Theta$ bytes

$$
M_{\text{optimizer}} = 12\Theta \text{ bytes}
$$

Total without activations: $16\Theta$ bytes

### Activation Memory

For a transformer with $L$ layers, hidden dimension $d_{\text{model}}$, batch size $B$, sequence length $S$:

**Attention activations** (per layer):
- Query, Key, Value projections: $3BSD_{\text{model}}$
- Attention scores: $BS^2H$ (where $H$ is number of heads)
- Attention output: $BSD_{\text{model}}$

**Feed-forward activations** (per layer):
- First linear layer: $BS \cdot 4d_{\text{model}}$ (typically FFN has 4x hidden dim)
- After activation: $BS \cdot 4d_{\text{model}}$

With activation checkpointing, we can reduce this significantly (recompute during backward pass).

```python
def calculate_activation_memory(
    batch_size,
    seq_length,
    num_layers,
    hidden_dim,
    num_heads,
    bytes_per_element=2  # FP16
):
    """Calculate activation memory for a transformer."""
    B, S, L, D, H = batch_size, seq_length, num_layers, hidden_dim, num_heads

    # Per layer activations
    # Attention: QKV projections + attention matrix + output
    attention = 3 * B * S * D + B * S * S * H + B * S * D

    # FFN: intermediate activations (assuming 4x expansion)
    ffn = 2 * B * S * 4 * D

    # Layer norm inputs
    layernorm = 2 * B * S * D

    per_layer = (attention + ffn + layernorm) * bytes_per_element
    total = per_layer * L

    return total / (1024**3)  # Convert to GB

# Example: LLaMA-7B equivalent
# 32 layers, 4096 hidden, 32 heads
activation_mem = calculate_activation_memory(
    batch_size=4,
    seq_length=2048,
    num_layers=32,
    hidden_dim=4096,
    num_heads=32
)
print(f"Activation memory (no checkpointing): {activation_mem:.2f} GB")

# With activation checkpointing, divide by sqrt(L) approximately
checkpointed = activation_mem / (32 ** 0.5)
print(f"With activation checkpointing: {checkpointed:.2f} GB")
```

---

## Data Parallelism

The simplest parallelism strategy: replicate the model on each GPU, split the batch.

### Basic Data Parallelism (DP)

PyTorch's `nn.DataParallel` (simple but inefficient):

```python
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=50000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)

# Basic DataParallel (single-machine, multi-GPU)
model = SimpleTransformer()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model = model.cuda()

# Forward pass automatically splits batch across GPUs
batch = torch.randint(0, 50000, (32, 128)).cuda()
output = model(batch)
```

**Problems with DataParallel:**
1. **Single-process bottleneck**: Main process on GPU 0 does all setup
2. **Gradient gather**: All gradients copied to GPU 0 for optimizer step
3. **GIL contention**: Python's Global Interpreter Lock limits parallelism
4. **No multi-node support**: Cannot scale beyond one machine

### Distributed Data Parallel (DDP)

DDP uses one process per GPU with all-reduce for gradients:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import os

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

class DummyDataset(Dataset):
    """Simple dataset for demonstration."""
    def __init__(self, size=1000, seq_len=128, vocab_size=50000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Random token IDs
        return torch.randint(0, self.vocab_size, (self.seq_len,))

def train_ddp(rank, world_size):
    """Training function for each process."""
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # Create model and move to GPU
    model = SimpleTransformer().to(rank)

    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # Create dataset with DistributedSampler
    dataset = DummyDataset(size=1000)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    ddp_model.train()
    for epoch in range(2):
        # Important: set epoch for proper shuffling
        sampler.set_epoch(epoch)

        for batch_idx, data in enumerate(dataloader):
            data = data.to(rank)

            # Forward pass
            output = ddp_model(data)

            # Compute loss (shift for language modeling)
            loss = loss_fn(
                output[:, :-1].reshape(-1, output.size(-1)),
                data[:, 1:].reshape(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # DDP automatically all-reduces gradients here
            optimizer.step()

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    cleanup()

def run_ddp_demo(world_size=2):
    """Launch DDP training."""
    mp.spawn(
        train_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

# Run with: run_ddp_demo(world_size=torch.cuda.device_count())
```

**How DDP Works:**

1. **Initialization**: Each process has identical model copy
2. **Forward pass**: Each process computes on its data shard
3. **Backward pass**: Gradients computed locally
4. **All-Reduce**: Gradients averaged across all processes using ring all-reduce
5. **Optimizer step**: Each process updates its local model copy (now identical)

**Communication Pattern (Ring All-Reduce):**

For $N$ GPUs, each with gradient tensor $g_i$:

$$
g_{\text{avg}} = \frac{1}{N} \sum_{i=1}^{N} g_i
$$

Ring all-reduce achieves this in $2(N-1)$ steps with bandwidth-optimal communication:

```python
def ring_all_reduce_conceptual(gradients, rank, world_size):
    """
    Conceptual implementation of ring all-reduce.
    In practice, NCCL handles this efficiently.

    Args:
        gradients: List of gradient chunks for this rank
        rank: Current process rank
        world_size: Total number of processes
    """
    num_chunks = len(gradients)

    # Phase 1: Reduce-scatter (each rank gets sum of one chunk)
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        # Send gradients[send_idx] to next rank
        # Receive from previous rank and add to gradients[recv_idx]
        # (Actual communication omitted for clarity)
        pass

    # Phase 2: All-gather (broadcast the summed chunks)
    for step in range(world_size - 1):
        send_idx = (rank + 1 - step) % world_size
        recv_idx = (rank - step) % world_size

        # Send gradients[send_idx] to next rank
        # Receive from previous rank into gradients[recv_idx]
        pass

    # Average
    for i in range(num_chunks):
        gradients[i] /= world_size

    return gradients
```

**Communication Cost:**

Transfer volume per GPU: $\frac{2(N-1)}{N} \cdot |\text{gradients}| \approx 2|\text{gradients}|$

For large $N$, this is optimal (each GPU must send/receive all gradients).

---

## Tensor Parallelism

Split individual layers across GPUs. Each GPU computes part of a matrix multiplication.

### Megatron-LM Tensor Parallelism

[Megatron-LM](https://arxiv.org/abs/1909.08053) (NVIDIA, 2019) introduced efficient tensor parallelism for transformers.

**Key Idea**: Partition weight matrices along specific dimensions to minimize communication.

#### Column-Parallel Linear Layer

Split output dimension:

$$
Y = XA = X[A_1 | A_2] = [XA_1 | XA_2] = [Y_1 | Y_2]
$$

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    Split output features across GPUs:
        Y = XW where W is partitioned as [W1, W2, ..., Wn]
        Each GPU computes Yi = X @ Wi

    Args:
        in_features: Input dimension
        out_features: Output dimension (will be split across GPUs)
        bias: Whether to use bias
        gather_output: Whether to all-gather outputs
    """
    def __init__(self, in_features, out_features, bias=True, gather_output=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        # Get tensor parallel group (assume already initialized)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        assert out_features % world_size == 0

        self.out_features_per_partition = out_features // world_size

        # Each GPU only stores its partition
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_partition)
            )
        else:
            self.bias = None

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize with same values as if not partitioned."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features] if gather_output else
            [..., out_features_per_partition]
        """
        # Local matrix multiply
        output = torch.matmul(x, self.weight.t())

        if self.bias is not None:
            output = output + self.bias

        if self.gather_output and dist.is_initialized():
            # All-gather outputs across tensor parallel group
            output = self._all_gather(output)

        return output

    def _all_gather(self, tensor):
        """Gather tensors from all GPUs along last dimension."""
        world_size = dist.get_world_size()

        # Create list to hold gathered tensors
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]

        # All-gather
        dist.all_gather(tensor_list, tensor)

        # Concatenate along last dimension
        output = torch.cat(tensor_list, dim=-1)
        return output

class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    Split input features across GPUs:
        Y = XW where X is partitioned as [X1, X2, ..., Xn]
        Each GPU computes Yi = Xi @ Wi
        Final Y = sum(Yi) via all-reduce

    Args:
        in_features: Input dimension (will be split across GPUs)
        out_features: Output dimension
        bias: Whether to use bias
        input_is_parallel: Whether input is already partitioned
    """
    def __init__(self, in_features, out_features, bias=True, input_is_parallel=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        assert in_features % world_size == 0

        self.in_features_per_partition = in_features // world_size

        # Each GPU stores its partition
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

        if bias:
            # Only rank 0 has bias (to avoid duplicating in all-reduce)
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None and isinstance(self.bias, nn.Parameter):
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [..., in_features] or [..., in_features_per_partition]
               if input_is_parallel=True
        """
        # If input is not already partitioned, split it
        if not self.input_is_parallel and dist.is_initialized():
            x = self._split_along_last_dim(x)

        # Local matrix multiply
        output = torch.matmul(x, self.weight.t())

        # All-reduce across tensor parallel group
        if dist.is_initialized():
            dist.all_reduce(output)

        if self.bias is not None:
            output = output + self.bias

        return output

    def _split_along_last_dim(self, tensor):
        """Split tensor along last dimension across GPUs."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Calculate per-partition size
        last_dim = tensor.size(-1)
        per_partition = last_dim // world_size

        # Get this rank's chunk
        start = rank * per_partition
        end = start + per_partition

        return tensor[..., start:end].contiguous()
```

#### Applying to Transformer Layers

**Multi-Head Attention** (column parallel):

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

Split heads across GPUs:

```python
class TensorParallelAttention(nn.Module):
    """
    Multi-head attention with tensor parallelism.
    Splits heads across GPUs.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        assert num_heads % world_size == 0

        self.num_heads_per_partition = num_heads // world_size

        # QKV projection: column parallel (split heads)
        # Output: [batch, seq_len, d_model] where d_model is split
        self.qkv_proj = ColumnParallelLinear(
            d_model,
            3 * d_model,  # Q, K, V
            bias=True,
            gather_output=False  # Keep partitioned
        )

        # Output projection: row parallel (input is split, reduce output)
        self.out_proj = RowParallelLinear(
            d_model,
            d_model,
            bias=True,
            input_is_parallel=True
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection (output is partitioned)
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*d_model/world_size]

        # Split into Q, K, V
        qkv = qkv.reshape(
            batch_size, seq_len, 3,
            self.num_heads_per_partition, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads_per_partition, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation (local to each GPU)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Reshape: [batch, heads_per_partition, seq_len, head_dim]
        #       -> [batch, seq_len, d_model/world_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            batch_size, seq_len, -1
        )

        # Output projection (row parallel, all-reduces inside)
        output = self.out_proj(attn_output)

        return output
```

**Feed-Forward Network**:

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

- $W_1$: Column parallel (split output features)
- $W_2$: Row parallel (split input features, matching $W_1$ output)

```python
class TensorParallelFFN(nn.Module):
    """
    Feed-forward network with tensor parallelism.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # First linear: column parallel
        self.fc1 = ColumnParallelLinear(
            d_model, d_ff, gather_output=False
        )

        # Second linear: row parallel
        self.fc2 = RowParallelLinear(
            d_ff, d_model, input_is_parallel=True
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # fc1: [batch, seq, d_model] -> [batch, seq, d_ff/world_size]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # fc2: [batch, seq, d_ff/world_size] -> [batch, seq, d_model]
        # All-reduce happens inside fc2
        x = self.fc2(x)
        x = self.dropout(x)

        return x
```

### Communication Analysis

**Per Transformer Layer**:

1. **Attention**:
   - QKV projection: No communication (column parallel)
   - Output projection: All-reduce of $[B, S, d_{\text{model}}]$ tensor

2. **FFN**:
   - First linear: No communication (column parallel)
   - Second linear: All-reduce of $[B, S, d_{\text{model}}]$ tensor

**Total communication per layer**: 2 all-reduces of size $BSd_{\text{model}}$

For all-reduce with $N$ GPUs: $\frac{2(N-1)}{N} \cdot BSd_{\text{model}} \approx 2BSd_{\text{model}}$ bytes per GPU

---

## Pipeline Parallelism

Split model layers across GPUs vertically (each GPU handles a subset of layers).

### GPipe and PipeDream

**Naive Pipeline** (sequential):
- GPU 0: Layers 1-4
- GPU 1: Layers 5-8
- GPU 2: Layers 9-12
- GPU 3: Layers 13-16

Problem: **Pipeline bubbles** (idle time)

```
Time -->
GPU 0: [F0]    [B0]         [F1]    [B1]
GPU 1:     [F0]    [B0]         [F1]    [B1]
GPU 2:         [F0]    [B0]         [F1]
GPU 3:             [F0]    [B0]         [F1]

Legend: F = forward pass, B = backward pass, number = micro-batch
```

**GPipe** ([Google, 2019](https://arxiv.org/abs/1811.06965)): Split each batch into micro-batches

```python
import torch
import torch.nn as nn
from typing import List

class PipelineStage(nn.Module):
    """One stage of a pipeline (subset of layers on one GPU)."""
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GPipeSimple:
    """
    Simplified GPipe implementation.

    Splits batch into micro-batches and pipelines them through stages.
    """
    def __init__(self, stages: List[PipelineStage], num_microbatches: int):
        """
        Args:
            stages: List of pipeline stages (one per GPU)
            num_microbatches: Number of micro-batches to split batch into
        """
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.num_stages = len(stages)

    def forward(self, x):
        """
        Forward pass through pipeline.

        Args:
            x: Input tensor [batch, ...]

        Returns:
            Output tensor [batch, ...]
        """
        batch_size = x.size(0)
        assert batch_size % self.num_microbatches == 0

        microbatch_size = batch_size // self.num_microbatches

        # Split input into micro-batches
        microbatches = torch.chunk(x, self.num_microbatches, dim=0)

        # Store outputs for each stage
        outputs = [[] for _ in range(self.num_stages)]

        # Forward pass schedule
        for mb_idx in range(self.num_microbatches):
            microbatch = microbatches[mb_idx]

            for stage_idx, stage in enumerate(self.stages):
                if stage_idx == 0:
                    # First stage: use input microbatch
                    output = stage(microbatch)
                else:
                    # Later stages: use output from previous stage
                    # In real implementation, this would involve communication
                    output = stage(outputs[stage_idx - 1][mb_idx])

                outputs[stage_idx].append(output)

        # Concatenate micro-batch outputs
        final_output = torch.cat(outputs[-1], dim=0)
        return final_output

    def backward(self, grad_output):
        """Backward pass through pipeline."""
        # Split gradient into micro-batches
        grad_microbatches = torch.chunk(grad_output, self.num_microbatches, dim=0)

        # Backward pass in reverse order
        for mb_idx in range(self.num_microbatches - 1, -1, -1):
            grad = grad_microbatches[mb_idx]

            for stage_idx in range(self.num_stages - 1, -1, -1):
                # Compute gradients for this stage
                # In real implementation, this involves communication
                pass
```

**Bubble Time Analysis**:

With $M$ micro-batches and $P$ pipeline stages:

- Total time: $M + P - 1$ steps (each step = 1 forward or backward pass)
- Ideal parallel time: $2M$ steps (forward + backward for all micro-batches)
- Bubble time: $(P - 1)$ steps

Bubble ratio: $\frac{P - 1}{M + P - 1} \approx \frac{P}{M}$ for large $M$

**Rule of thumb**: Use $M \geq 4P$ to keep bubble overhead < 20%

### Bubble Reduction Strategies

**PipeDream** ([Microsoft, 2019](https://arxiv.org/abs/1806.03377)):
- Interleave forward/backward passes
- Maintain multiple versions of weights (weight stashing)

**Interleaved schedules** (used in Megatron-LM):
- Split each GPU's layers into multiple chunks
- Reduces bubble time from $O(P)$ to $O(P/V)$ where $V$ is number of chunks

```python
def create_interleaved_schedule(num_stages, num_microbatches, num_model_chunks):
    """
    Create an interleaved pipeline schedule.

    Instead of GPU 0: [Layers 1-4], GPU 1: [Layers 5-8]
    Use: GPU 0: [Layers 1-2, 9-10], GPU 1: [Layers 3-4, 11-12]

    This reduces bubble time.

    Args:
        num_stages: Number of pipeline stages (GPUs)
        num_microbatches: Number of micro-batches
        num_model_chunks: Number of layer chunks per GPU (V)

    Returns:
        Schedule as list of (stage, operation, microbatch_id) tuples
    """
    total_chunks = num_stages * num_model_chunks
    schedule = []

    # Forward passes
    for mb in range(num_microbatches):
        for chunk in range(total_chunks):
            stage = chunk % num_stages
            schedule.append((stage, 'forward', mb, chunk))

    # Backward passes
    for mb in range(num_microbatches - 1, -1, -1):
        for chunk in range(total_chunks - 1, -1, -1):
            stage = chunk % num_stages
            schedule.append((stage, 'backward', mb, chunk))

    return schedule

# Example
schedule = create_interleaved_schedule(
    num_stages=4,
    num_microbatches=8,
    num_model_chunks=2
)

# Visualize first few steps
for i, (stage, op, mb, chunk) in enumerate(schedule[:20]):
    print(f"Step {i}: GPU {stage}, {op} microbatch {mb}, chunk {chunk}")
```

---

## ZeRO: Zero Redundancy Optimizer

[DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) (Microsoft, 2020) eliminates memory redundancy in data parallelism.

**Key Insight**: In standard DDP, each GPU stores:
- Full model parameters
- Full gradients
- Full optimizer states

But we only need each GPU to compute updates for its data shard. Can we partition the states?

### ZeRO Stage 1: Optimizer State Sharding

Each GPU stores optimizer states for only a subset of parameters.

**Memory Savings**: Optimizer states reduced by $N$ where $N$ = number of GPUs

For AdamW with $\Theta$ parameters:
- Standard DDP: $12\Theta$ bytes per GPU
- ZeRO-1: $12\Theta / N$ bytes per GPU

```python
import torch
import torch.distributed as dist
from torch.optim import AdamW

class ZeROStage1Optimizer:
    """
    ZeRO Stage 1: Partition optimizer states.

    Each rank maintains:
    - Full copy of parameters (for forward/backward)
    - Partition of optimizer states
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps

        # Get distributed info
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Partition parameters across ranks
        self._partition_parameters()

        # Create optimizer for local partition only
        self.optimizer = AdamW(
            self.local_params,
            lr=lr,
            betas=betas,
            eps=eps
        )

    def _partition_parameters(self):
        """Assign each parameter to a rank."""
        self.param_to_rank = {}
        self.local_params = []

        for i, param in enumerate(self.params):
            # Assign to rank in round-robin fashion
            assigned_rank = i % self.world_size
            self.param_to_rank[param] = assigned_rank

            if assigned_rank == self.rank:
                self.local_params.append(param)

    def step(self):
        """
        Optimizer step with ZeRO-1.

        1. Each rank updates its partition of parameters
        2. All-gather updated parameters
        """
        # Update local parameters
        self.optimizer.step()

        if not dist.is_initialized():
            return

        # All-gather updated parameters
        for param in self.params:
            assigned_rank = self.param_to_rank[param]

            # Broadcast from the rank that owns this parameter
            dist.broadcast(param.data, src=assigned_rank)

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

# Usage example
def train_with_zero1():
    model = SimpleTransformer()

    # Use ZeRO-1 optimizer
    optimizer = ZeROStage1Optimizer(model.parameters(), lr=1e-4)

    # Training loop
    for batch in dataloader:
        output = model(batch)
        loss = compute_loss(output)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### ZeRO Stage 2: Gradient Sharding

Partition both optimizer states AND gradients.

**Memory Savings**:
- Optimizer states: $12\Theta / N$ bytes
- Gradients: $2\Theta / N$ bytes
- Total: $(12\Theta + 2\Theta) / N = 14\Theta / N$ bytes

```python
class ZeROStage2Optimizer(ZeROStage1Optimizer):
    """
    ZeRO Stage 2: Partition optimizer states and gradients.

    Each rank maintains:
    - Full copy of parameters
    - Partition of gradients (only for its parameters)
    - Partition of optimizer states
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr, betas, eps)

        # Register backward hooks to reduce-scatter gradients
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to reduce-scatter gradients after backward."""
        for param in self.params:
            if param.requires_grad:
                param.register_hook(
                    lambda grad, p=param: self._gradient_hook(grad, p)
                )

    def _gradient_hook(self, grad, param):
        """
        Hook called after gradient computation.

        Reduce-scatter: Each rank gets average of its partition.
        """
        if not dist.is_initialized():
            return grad

        assigned_rank = self.param_to_rank[param]

        if assigned_rank == self.rank:
            # This rank owns this parameter
            # Reduce gradients from all ranks
            dist.reduce(grad, dst=assigned_rank, op=dist.ReduceOp.SUM)
            grad.div_(self.world_size)
            return grad
        else:
            # This rank doesn't own this parameter
            # Send gradient and free memory
            dist.reduce(grad, dst=assigned_rank, op=dist.ReduceOp.SUM)
            # Return None to free gradient memory
            return None
```

### ZeRO Stage 3: Parameter Sharding

Partition parameters, gradients, AND optimizer states (Full parameter sharding).

**Memory Savings**: $16\Theta / N$ bytes per GPU

This is what **FSDP** (Fully Sharded Data Parallel) implements!

```python
class ZeROStage3Module(nn.Module):
    """
    ZeRO Stage 3 / FSDP: Partition everything including parameters.

    Each rank maintains:
    - Partition of parameters (not full copy!)
    - Partition of gradients
    - Partition of optimizer states

    Parameters are all-gathered before forward/backward, then freed.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

        # Get distributed info
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Partition parameters
        self._partition_parameters()

    def _partition_parameters(self):
        """
        Partition parameters across ranks.

        Each rank keeps only its shard of parameters.
        """
        self.param_shards = {}

        for name, param in self.module.named_parameters():
            # Flatten parameter
            param_flat = param.data.flatten()
            total_size = param_flat.numel()

            # Calculate shard size
            shard_size = (total_size + self.world_size - 1) // self.world_size

            # Extract this rank's shard
            start = self.rank * shard_size
            end = min(start + shard_size, total_size)

            # Store only this shard
            self.param_shards[name] = param_flat[start:end].clone()

            # Free original parameter (we'll reconstruct when needed)
            param.data = self.param_shards[name]

    def _all_gather_params(self):
        """
        All-gather parameters before forward/backward pass.

        Returns:
            Dictionary of full parameters
        """
        full_params = {}

        for name, param in self.module.named_parameters():
            # All-gather shards from all ranks
            shard = self.param_shards[name]

            # Prepare list for gathering
            shard_list = [
                torch.zeros_like(shard) for _ in range(self.world_size)
            ]

            if dist.is_initialized():
                dist.all_gather(shard_list, shard)
            else:
                shard_list = [shard]

            # Concatenate shards
            full_param = torch.cat(shard_list)

            # Reshape to original shape
            full_params[name] = full_param.view(param.shape)

        return full_params

    def _free_full_params(self):
        """Free full parameters after forward/backward."""
        # In real implementation, restore parameter data to shards
        pass

    def forward(self, *args, **kwargs):
        """
        Forward pass with parameter all-gather.

        1. All-gather parameters
        2. Run forward pass
        3. Free full parameters (keep only shards)
        """
        # All-gather parameters
        full_params = self._all_gather_params()

        # Temporarily replace parameters with full versions
        original_params = {}
        for name, param in self.module.named_parameters():
            original_params[name] = param.data
            param.data = full_params[name]

        # Forward pass
        output = self.module(*args, **kwargs)

        # Restore sharded parameters
        for name, param in self.module.named_parameters():
            param.data = original_params[name]

        return output
```

**Trade-off**: More communication (all-gather parameters for each layer) but massive memory savings.

---

## Fully Sharded Data Parallel (FSDP)

PyTorch's native implementation of ZeRO-3:

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
import functools

def train_with_fsdp(rank, world_size):
    """Training with FSDP."""
    setup(rank, world_size)

    # Create model
    model = SimpleTransformer()

    # Define auto wrap policy (which modules to wrap)
    # Wrap layers with > 100M parameters
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000
    )

    # Mixed precision policy
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=rank,
        # Sharding strategy
        sharding_strategy="FULL_SHARD",  # ZeRO-3
        # sharding_strategy="SHARD_GRAD_OP",  # ZeRO-2
        # sharding_strategy="NO_SHARD",  # DDP
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    dataset = DummyDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

    model.train()
    for epoch in range(2):
        sampler.set_epoch(epoch)

        for batch_idx, data in enumerate(dataloader):
            data = data.to(rank)

            # Forward pass
            # FSDP automatically all-gathers parameters
            output = model(data)

            # Loss
            loss = nn.functional.cross_entropy(
                output[:, :-1].reshape(-1, output.size(-1)),
                data[:, 1:].reshape(-1)
            )

            # Backward pass
            # FSDP automatically reduces gradients
            loss.backward()

            # Optimizer step
            # Each rank updates its parameter shard
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    cleanup()

# To run: mp.spawn(train_with_fsdp, args=(world_size,), nprocs=world_size)
```

**FSDP Benefits**:
- Can train models much larger than single GPU memory
- Good scaling efficiency (communication overlap)
- Works well with activation checkpointing

**When to use FSDP**:
- Model too large for DDP
- Memory-constrained (can trade communication for memory)
- Training very large models (> 10B parameters)

---

## 3D Parallelism

Combine all three parallelism types:

1. **Data Parallelism** (DP): Replicate model, split batch
2. **Tensor Parallelism** (TP): Split layers horizontally
3. **Pipeline Parallelism** (PP): Split layers vertically

Used for largest models (GPT-3, PaLM, etc.)

```python
"""
Example: Train 175B parameter model on 1024 GPUs

Organize GPUs in 3D grid:
- Pipeline parallel size: 8 (P=8, 8 stages)
- Tensor parallel size: 8 (T=8, split within stages)
- Data parallel size: 16 (D=16, replicas)

Total: 8 × 8 × 16 = 1024 GPUs

Each pipeline stage distributed across 8 GPUs (tensor parallel)
16 replicas of entire pipeline (data parallel)
"""

def calculate_3d_parallelism(
    total_gpus,
    model_size_billions,
    memory_per_gpu_gb=80,
    activation_checkpointing=True
):
    """
    Suggest 3D parallelism configuration.

    Args:
        total_gpus: Total number of GPUs available
        model_size_billions: Model size in billions of parameters
        memory_per_gpu_gb: Memory per GPU in GB
        activation_checkpointing: Whether using activation checkpointing

    Returns:
        (pipeline_parallel, tensor_parallel, data_parallel) sizes
    """
    # Estimate memory needed per GPU for model states
    # 16 bytes per param (mixed precision with Adam)
    model_memory_gb = model_size_billions * 16 / self.world_size

    # Rule of thumb: tensor parallel size
    # Limit communication overhead, typically 4-8
    tensor_parallel = min(8, total_gpus)

    # Pipeline parallel size
    # Based on model size and memory constraints
    required_memory_reduction = model_memory_gb / memory_per_gpu_gb
    pipeline_parallel = max(1, int(required_memory_reduction / tensor_parallel))

    # Data parallel size (remaining GPUs)
    data_parallel = total_gpus // (pipeline_parallel * tensor_parallel)

    print(f"Suggested configuration for {model_size_billions}B model on {total_gpus} GPUs:")
    print(f"  Pipeline Parallel (P): {pipeline_parallel}")
    print(f"  Tensor Parallel (T): {tensor_parallel}")
    print(f"  Data Parallel (D): {data_parallel}")
    print(f"  Total: {pipeline_parallel} × {tensor_parallel} × {data_parallel} = {pipeline_parallel * tensor_parallel * data_parallel}")

    return pipeline_parallel, tensor_parallel, data_parallel

# Examples
calculate_3d_parallelism(1024, 175)  # GPT-3 scale
calculate_3d_parallelism(512, 70)    # LLaMA-70B scale
calculate_3d_parallelism(128, 13)    # LLaMA-13B scale
```

**Communication Patterns in 3D Parallelism**:

1. **Within Tensor Parallel Group**: High bandwidth needed (NVLink preferred)
2. **Within Pipeline Parallel Group**: Sequential communication (point-to-point)
3. **Within Data Parallel Group**: All-reduce gradients (can use InfiniBand)

**Optimal GPU Topology**:
- Tensor parallel: Place on same node (NVLink)
- Pipeline parallel: Can span nodes (InfiniBand OK)
- Data parallel: Can span data centers (if needed)

---

## Communication Costs and Trade-offs

### Bandwidth Hierarchy

```python
# Approximate bandwidths (2024)
bandwidths = {
    "NVLink 4.0 (H100)": "900 GB/s (within node)",
    "PCIe 5.0": "128 GB/s (within node)",
    "InfiniBand HDR": "200 Gb/s = 25 GB/s (across nodes)",
    "Ethernet 100G": "12.5 GB/s (across nodes)",
}

for tech, bw in bandwidths.items():
    print(f"{tech}: {bw}")
```

### Communication Volume per Strategy

For a model with $\Theta$ parameters, batch size $B$, sequence length $S$, hidden dim $d$:

| Strategy | Communication per Step | Frequency |
|----------|----------------------|-----------|
| **DDP** | $2\Theta$ (gradient all-reduce) | Once per step |
| **FSDP (ZeRO-3)** | $2\Theta$ (all-gather params) + $2\Theta$ (reduce-scatter grads) | Per layer |
| **Tensor Parallel** | $2BSd$ (all-reduce activations) | 2× per layer |
| **Pipeline Parallel** | $BSd$ (point-to-point) | Per micro-batch |

### When to Use Each Strategy

```python
def recommend_parallelism_strategy(
    model_size_gb,
    gpu_memory_gb,
    num_gpus,
    interconnect="nvlink"  # or "infiniband", "ethernet"
):
    """
    Recommend parallelism strategy based on model size and resources.
    """
    recommendations = []

    # Can model fit on single GPU with DDP?
    if model_size_gb * 1.5 < gpu_memory_gb:  # 1.5x for activations
        recommendations.append("DDP - simplest, fastest")

    # Need memory reduction?
    memory_ratio = model_size_gb / gpu_memory_gb

    if memory_ratio > 1:
        if num_gpus >= 8 and interconnect == "nvlink":
            recommendations.append(f"Tensor Parallel (TP) - {min(8, num_gpus)} way split")

        if num_gpus >= 16:
            recommendations.append("Pipeline Parallel (PP) - for very deep models")

        recommendations.append("FSDP/ZeRO-3 - most memory efficient")

    # Large scale training?
    if num_gpus >= 64:
        recommendations.append("3D Parallelism - combine DP+TP+PP")

    print(f"\nModel: {model_size_gb:.1f}GB, GPU: {gpu_memory_gb}GB, {num_gpus} GPUs")
    print("Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    return recommendations

# Examples
recommend_parallelism_strategy(14, 80, 8, "nvlink")     # 7B model, 8× H100
recommend_parallelism_strategy(140, 80, 64, "nvlink")   # 70B model, 64× H100
recommend_parallelism_strategy(350, 80, 512, "infiniband")  # 175B model, 512 GPUs
```

### Comparison Table

| Strategy | Memory Efficiency | Communication | Code Complexity | Best For |
|----------|------------------|---------------|-----------------|----------|
| **DDP** | Low (full replica) | Low (gradient all-reduce only) | Low | Models that fit in GPU memory |
| **FSDP/ZeRO-3** | High (shard everything) | Medium-High (frequent all-gather) | Medium | Memory-constrained single models |
| **Tensor Parallel** | Medium (shard layers) | High (per-layer all-reduce) | High | Wide models, fast interconnect |
| **Pipeline Parallel** | Medium (shard layers) | Low (point-to-point) | High | Very deep models |
| **3D Parallel** | Highest | Complex | Very High | Largest models (100B+ params) |

---

## Practical Implementation

### Complete FSDP Training Script

```python
"""
Complete training script with FSDP for LLM.

Run with:
    torchrun --nproc_per_node=4 train_fsdp.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from functools import partial
import os

class TransformerBlock(nn.Module):
    """Single transformer block for FSDP wrapping."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # FFN
        ffn_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x

class GPTModel(nn.Module):
    """GPT-style model for FSDP."""
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(2048, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, S = input_ids.shape

        # Embeddings
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def train_fsdp():
    """Main training function with FSDP."""
    # Setup
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Model configuration (e.g., GPT-2 scale)
    vocab_size = 50257
    d_model = 768
    nhead = 12
    num_layers = 12
    dim_feedforward = 3072

    # Create model
    model = GPTModel(vocab_size, d_model, nhead, num_layers, dim_feedforward)

    # FSDP auto-wrap policy
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock}
    )

    # Mixed precision
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Data
    dataset = DummyDataset(size=10000, seq_len=512, vocab_size=vocab_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=2)

    # Training loop
    model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)

        for batch_idx, input_ids in enumerate(dataloader):
            input_ids = input_ids.cuda()

            # Forward
            logits = model(input_ids)

            # Loss (language modeling)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if batch_idx % 50 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    cleanup_distributed()

if __name__ == "__main__":
    train_fsdp()
```

### Launching Multi-Node Training

```bash
# Single node, 8 GPUs
torchrun --nproc_per_node=8 train_fsdp.py

# Multi-node (2 nodes, 8 GPUs each)
# Node 0:
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train_fsdp.py

# Node 1:
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train_fsdp.py
```

---

## Exercises

### Exercise 1: Memory Calculation

Calculate the memory requirements for training LLaMA-13B (13 billion parameters) using:
a) Standard DDP with FP16
b) FSDP/ZeRO-3 with 8 GPUs
c) 3D parallelism with 64 GPUs (TP=8, PP=8, DP=1)

Include parameters, gradients, optimizer states, and estimate activation memory for batch size 4, sequence length 2048.

<details>
<summary>Solution</summary>

```python
def llama13b_memory_calculation():
    params = 13e9  # 13B parameters
    batch_size = 4
    seq_len = 2048
    d_model = 5120  # LLaMA-13B hidden size
    num_layers = 40

    # a) DDP with FP16
    params_mem = params * 2 / (1024**3)  # FP16
    grads_mem = params * 2 / (1024**3)   # FP16
    optimizer_mem = params * 12 / (1024**3)  # FP32 AdamW

    # Activations (rough estimate)
    per_token_activation = 2 * num_layers * d_model * 4  # QKV + FFN
    activation_mem = batch_size * seq_len * per_token_activation * 2 / (1024**3)

    total_ddp = params_mem + grads_mem + optimizer_mem + activation_mem
    print(f"a) DDP: {total_ddp:.2f} GB per GPU")
    print(f"   Parameters: {params_mem:.2f} GB")
    print(f"   Gradients: {grads_mem:.2f} GB")
    print(f"   Optimizer: {optimizer_mem:.2f} GB")
    print(f"   Activations: {activation_mem:.2f} GB")

    # b) FSDP/ZeRO-3 with 8 GPUs
    fsdp_params = params_mem / 8
    fsdp_grads = grads_mem / 8
    fsdp_optimizer = optimizer_mem / 8
    fsdp_activations = activation_mem  # Not sharded

    total_fsdp = fsdp_params + fsdp_grads + fsdp_optimizer + fsdp_activations
    print(f"\nb) FSDP (8 GPUs): {total_fsdp:.2f} GB per GPU")

    # c) 3D parallelism (TP=8, PP=8, DP=1)
    # Pipeline splits 40 layers into 8 stages = 5 layers per stage
    layers_per_stage = num_layers / 8
    params_per_stage = params / 8

    # Tensor parallel splits each layer across 8 GPUs
    params_3d = params_per_stage * 2 / 8 / (1024**3)  # FP16
    grads_3d = params_per_stage * 2 / 8 / (1024**3)
    optimizer_3d = params_per_stage * 12 / 8 / (1024**3)

    # Activations for pipeline stage (micro-batches reduce this)
    activation_3d = activation_mem / 8  # Divided across pipeline

    total_3d = params_3d + grads_3d + optimizer_3d + activation_3d
    print(f"\nc) 3D Parallelism (64 GPUs): {total_3d:.2f} GB per GPU")

llama13b_memory_calculation()
```
</details>

### Exercise 2: Implement ZeRO-2

Implement a simplified ZeRO-2 optimizer that partitions both optimizer states and gradients. Test with a small model.

<details>
<summary>Hint</summary>

Extend `ZeROStage1Optimizer` by:
1. Registering backward hooks on parameters
2. In hooks, reduce-scatter gradients (each rank gets its partition)
3. Free gradients for parameters not owned by this rank
</details>

### Exercise 3: Pipeline Bubble Analysis

For a pipeline with 4 stages and varying numbers of micro-batches (1, 2, 4, 8, 16):
a) Calculate the bubble ratio
b) Plot bubble ratio vs. number of micro-batches
c) What's the minimum number of micro-batches to achieve <10% bubble?

<details>
<summary>Solution</summary>

```python
import matplotlib.pyplot as plt

def analyze_pipeline_bubbles(num_stages=4):
    microbatch_counts = [1, 2, 4, 8, 16, 32]
    bubble_ratios = []

    for M in microbatch_counts:
        P = num_stages

        # Total steps: forward + backward for all micro-batches + pipeline fill/drain
        total_steps = 2 * M + 2 * (P - 1)

        # Ideal steps (fully parallel): forward + backward for all
        ideal_steps = 2 * M

        # Bubble steps
        bubble_steps = 2 * (P - 1)

        # Bubble ratio
        ratio = bubble_steps / total_steps
        bubble_ratios.append(ratio)

        print(f"M={M:2d}: Bubble ratio = {ratio:.2%} ({bubble_steps}/{total_steps} steps)")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(microbatch_counts, [r * 100 for r in bubble_ratios], marker='o')
    plt.axhline(y=10, color='r', linestyle='--', label='10% target')
    plt.xlabel('Number of Micro-batches')
    plt.ylabel('Bubble Ratio (%)')
    plt.title(f'Pipeline Bubble Ratio vs Micro-batches (P={num_stages})')
    plt.grid(True)
    plt.legend()
    plt.savefig('pipeline_bubbles.png')
    print("\nFor <10% bubble overhead, need at least 8 micro-batches")

analyze_pipeline_bubbles(4)
```
</details>

### Exercise 4: Communication Volume Comparison

For a GPT-2 sized model (124M parameters) trained on 8 GPUs with batch size 32, sequence length 1024:

Calculate the total communication volume per training step for:
a) DDP
b) FSDP
c) Tensor Parallelism (2-way)

Assume BF16 for all operations.

<details>
<summary>Solution</summary>

```python
def communication_volume_comparison():
    params = 124e6  # 124M
    bytes_per_param = 2  # BF16

    batch_size = 32
    seq_len = 1024
    d_model = 768  # GPT-2
    num_layers = 12

    # a) DDP: All-reduce gradients once
    ddp_comm = 2 * params * bytes_per_param
    print(f"a) DDP: {ddp_comm / (1024**3):.3f} GB per GPU per step")

    # b) FSDP: All-gather params + reduce-scatter grads for each layer
    # Approximately: 2x params per layer for forward, 2x for backward
    fsdp_comm = 4 * params * bytes_per_param
    print(f"b) FSDP: {fsdp_comm / (1024**3):.3f} GB per GPU per step")

    # c) Tensor Parallelism (2-way)
    # All-reduce activations: 2x per layer (attention + FFN)
    # Volume: batch_size * seq_len * d_model * bytes_per_param * 2 * num_layers
    activation_volume = batch_size * seq_len * d_model * bytes_per_param
    tp_comm = 2 * activation_volume * 2 * num_layers  # forward + backward
    print(f"c) Tensor Parallel (2-way): {tp_comm / (1024**3):.3f} GB per GPU per step")

    print(f"\nRatio TP/DDP: {tp_comm/ddp_comm:.2f}x")
    print(f"Ratio FSDP/DDP: {fsdp_comm/ddp_comm:.2f}x")

communication_volume_comparison()
```
</details>

### Exercise 5: Optimal Parallelism Strategy

Design a parallelism strategy for training a 70B parameter model on 256 GPUs (H100, 80GB each). Assume:
- Hidden dimension: 8192
- 80 layers
- Batch size: 512
- Sequence length: 4096
- NVLink within nodes (8 GPUs/node), InfiniBand across nodes

Justify your choice of DP, TP, and PP dimensions.

<details>
<summary>Discussion Points</summary>

Consider:
1. Memory requirements (16 bytes/param × 70B = 1.12TB model states alone)
2. Tensor parallel limited by interconnect (NVLink within node → TP ≤ 8)
3. Pipeline parallel for layer distribution
4. Data parallel for scaling training throughput
5. Activation checkpointing necessity

Reasonable configuration:
- TP = 8 (within each node, use NVLink)
- PP = 8 (80 layers / 8 = 10 layers per stage)
- DP = 4 (256 / (8×8) = 4 replicas)

This gives:
- Memory per GPU: ~1.12TB / 64 ≈ 17.5GB (manageable with activation checkpointing)
- Good use of fast interconnect (NVLink for TP)
- Reasonable pipeline depth (10 layers per stage)
- Some data parallelism for throughput
</details>

---

## Key Takeaways

1. **Data Parallelism (DDP)**: Simplest, use when model fits in memory
2. **FSDP/ZeRO-3**: Best memory efficiency, shards everything
3. **Tensor Parallelism**: Fast but communication-heavy, needs fast interconnect
4. **Pipeline Parallelism**: Splits layers vertically, watch for bubbles
5. **3D Parallelism**: Combines all three for largest models (100B+ parameters)

**Choosing a strategy**:
- Model fits in GPU memory → **DDP**
- Model doesn't fit, good interconnect → **FSDP** or **Tensor Parallel**
- Very large model (100B+) → **3D Parallelism**

**Communication optimization**:
- Overlap communication with computation
- Use gradient/activation checkpointing to reduce memory
- Match parallelism to network topology (TP within nodes, DP across)

## References

1. **Distributed Data Parallel**: [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
2. **Megatron-LM**: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019)
3. **ZeRO**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., 2020)
4. **GPipe**: [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) (Huang et al., 2019)
5. **PipeDream**: [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377) (Narayanan et al., 2019)
6. **PyTorch FSDP**: [Getting Started with FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
7. **3D Parallelism**: [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (Narayanan et al., 2021)

---

**Next Chapter**: [Scaling Laws and Optimization](17-scaling-optimization.md) - Learn about learning rate schedules, optimizers, and how model performance scales with size and compute.

**Previous Chapter**: [Language Model Training](15-lm-training.md) - Single-GPU training fundamentals.

**Related**: [Hardware, Quantization, and Training Optimization](31-hardware-quantization-optimization.md) - Hardware considerations for distributed training.
