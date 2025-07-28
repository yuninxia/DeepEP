# Available Tests Analysis for DeepEP

## Test Overview

### 1. ✅ test_intranode.py (ALREADY RAN - WORKS)
- **Purpose**: Tests NVLink-based GPU-to-GPU communication within a single node
- **Requirements**: Multiple GPUs with NVLink
- **Status**: ✅ FULLY FUNCTIONAL on this node
- **What it tests**:
  - BF16 and FP8 dispatch/combine operations
  - Async and sync modes
  - Performance tuning for different chunk sizes
  - NVLink bandwidth optimization

### 2. ❌ test_internode.py (LIMITED/WON'T WORK FULLY)
- **Purpose**: Tests RDMA-based communication across multiple nodes
- **Requirements**: 
  - Multiple nodes
  - InfiniBand/RDMA network
  - Proper MPI/distributed setup
- **Status**: ❌ Won't work for multi-node features
- **What we CAN test**:
  - Single-node mode (will behave similarly to intranode)
  - Low-latency compatibility mode with `--test-ll-compatibility`
- **Limitations**: No actual inter-node communication due to lack of RDMA

### 3. ⚠️ test_low_latency.py (PARTIALLY WORKS)
- **Purpose**: Tests ultra-low latency kernels for inference decoding
- **Requirements**:
  - NVSHMEM (✅ we have it)
  - RDMA for inter-node (❌ we don't have it)
  - Multiple QPs per rank
- **Status**: ⚠️ May work for intra-node scenarios
- **What it tests**:
  - Small batch sizes (default 128 tokens vs 4096 for others)
  - Hook-based overlapping
  - CUDA graph compatibility
  - Pure RDMA mode (won't work) vs NVLink mode (might work)

## Running Additional Tests

### Option 1: Try Low-Latency Test (Intra-node Mode)
```bash
source .venv/bin/activate
PYTHONPATH=. python tests/test_low_latency.py --num-processes 8
```

This might work because:
- We have NVSHMEM installed
- Can use NVLink instead of RDMA for intra-node
- The `--disable-nvlink` flag suggests NVLink mode exists

### Option 2: Try Internode Test (Single-node Mode)
```bash
source .venv/bin/activate
PYTHONPATH=. python tests/test_internode.py --num-processes 8
```

This will likely:
- Run in degraded single-node mode
- Show warnings about missing RDMA
- Still test some code paths

### Option 3: Low-Latency Compatibility Test
```bash
source .venv/bin/activate
PYTHONPATH=. python tests/test_internode.py --num-processes 8 --test-ll-compatibility
```

This tests compatibility between normal and low-latency kernels.

## Recommended Test Order

1. **test_low_latency.py** - Most likely to provide new insights
2. **test_internode.py --test-ll-compatibility** - Tests kernel compatibility
3. **test_internode.py** - Least likely to work properly

## Expected Outcomes

- **Low-latency test**: May work partially, showing NVLink-based low-latency performance
- **Internode test**: Will fail for actual inter-node ops but may test single-node fallbacks
- Both will show InfiniBand/RDMA warnings (as we saw with intranode)

## Performance Expectations

- Low-latency: Lower throughput but better latency than normal kernels
- Internode: Similar to intranode performance when running single-node
- No multi-node scaling tests possible without RDMA