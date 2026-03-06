# mio - Memory I/O Microbenchmark

A tool for measuring memory bandwidth across various access patterns on Linux.

Supports both **NUMA system memory** (`--membind`) and **CXL/DevDAX devices** (`--devdax`).

## Requirements

- Linux (Ubuntu recommended)
- x86-64 CPU with AVX2 support
- Python 3.10+
- Root/sudo privileges
- **One of the following memory targets:**
  - NUMA node (any multi-socket or CXL-attached memory)
  - CXL/DevDAX device (e.g., `/dev/dax0.0`)

## Setup

### Build only (no root required)

```bash
./build.sh
```

### Full environment setup

```bash
sudo ./scripts/setup.sh                  # install deps + build
sudo ./scripts/setup.sh --build-only     # build without environment changes
sudo ./scripts/setup.sh --with-cxl       # also configure CXL device to DevDAX mode
```

The setup script can:
1. Install build tools (`cmake`, `ninja-build`, `g++`, `numactl`, `ndctl`, etc.)
2. Load MSR kernel module
3. *(Optional, `--with-cxl`)* Convert CXL device to DevDAX mode
4. Disable NUMA auto balancing
5. Disable swap
6. Build the benchmark binary

To verify the environment without making changes:

```bash
./scripts/setup.sh --check
```

If your DevDAX device is not `dax0.0`:

```bash
sudo ./scripts/setup.sh --with-cxl --devdax dax1.0
```

## Recommended: DevDAX Mode

For best results, we recommend running benchmarks in **DevDAX mode** (`--devdax`).

## Quick Start

### Using CXL/DevDAX device (recommended)

```bash
# Sequential read on DevDAX device
sudo ./build/microbench --mode seq_read --threads 16 --memory-per-thread 65536 --devdax /dev/dax0.0

# Run full automated benchmark
sudo python3 scripts/run_benchmark.py --devdax /dev/dax0.0
```

## Benchmark Modes

### Automated benchmark (`run_benchmark.py`)

```bash
# Full benchmark (both large and small memory sizes)
sudo python3 scripts/run_benchmark.py --devdax /dev/dax0.0
sudo python3 scripts/run_benchmark.py --membind 4

# Large memory only (cache miss scenario)
sudo python3 scripts/run_benchmark.py --devdax /dev/dax0.0 --large-mem

# Small memory only (cache hit scenario)
sudo python3 scripts/run_benchmark.py --membind 4 --small-mem

# Quick mode (minimal test set)
sudo python3 scripts/run_benchmark.py --devdax /dev/dax0.0 4
```

Full mode tests:
- `seq_read`, `seq_write`
- `random_read` x3 block sizes (512K, 1M, 2M)
- `zipfian_read` x3 block sizes (512K, 1M, 2M)
- `random_write` (1M)

Quick mode tests (append `4`):
- `seq_read`, `seq_write`, `random_read` (1M), `random_write` (1M)

Memory size presets (with 16 threads):

| Option | Per Thread | Total 
|--------|-----------|-------
| `--large-mem` | 65536 MiB | 1 TiB 
| `--small-mem` | 8192 MiB | 128 GiB 

### Single test (`run.sh`)

```bash
./scripts/run.sh seq_read                        # default: 16 threads, 1MiB block
./scripts/run.sh random_read 32                  # 32 threads
./scripts/run.sh random_read 16 2097152          # block-size=2MiB
./scripts/run.sh --init seq_read                 # initialize memory with seq_write first
./scripts/run.sh --help                          # show all options
```

> **Note:** When using DevDAX with certain devices, you may need the `--init` flag on the first run to initialize the device with a sequential write.

Supported modes: `seq_read`, `seq_write`, `random_read`, `random_write`, `stride_read`, `stride_write`, `zipfian_read`

### Units

| Parameter | Unit | Example |
|-----------|------|---------|
| `block-size` | Bytes | `1048576` = 1 MiB, `2097152` = 2 MiB, `524288` = 512 KiB |
| `memory-per-thread` | MiB | `65536` = 64 GiB, `8192` = 8 GiB |
| `total-mem` (run.sh) | MiB | `1048576` = 1 TiB |

See [scripts/commands.md](scripts/commands.md) for a full list of copy-pasteable commands.

## Output

Results are organized by timestamp:

```
result/
  2025-01-15_14-30-00/
    00_init_seq_write/
    mem_65536/
      01_cache_flush_1/
      02_seq_read/
      ...
    mem_8192/
      ...
summary/
  2025-01-15_14-30-00/
    benchmark_summary.txt
```

The summary file contains bandwidth results in a table format:

```
MemPerThread Mode                 Block Size       Bandwidth
------------ -------------------- ------------ ---------------
       65536 seq_read                        -      12.34 GB/s
       65536 random_read           1048576 B        8.56 GB/s
       ...
```

## Advanced Options

For direct use of the benchmark binary:

```bash
sudo ./build/microbench [OPTIONS]
```

Key options:
| Option | Description |
|--------|-------------|
| `--mode <mode>` | Access pattern (e.g., `seq_read`, `random_write`) |
| `--threads <n>` | Number of threads (default: 1) |
| `--memory-per-thread <MiB>` | Memory per thread in MiB |
| `--block-size <bytes>` | Block size for random/stride access |
| `--membind <node>` | NUMA memory node (e.g., `0`, `4`) |
| `--devdax <device>` | DevDAX device path (e.g., `/dev/dax0.0`) |
| `--offset <hex>` | Memory offset for DevDAX |
| `--cpu-affinity <nodes>` | Pin threads to CPUs of NUMA node(s) |
| `--prefetch <ON\|OFF>` | Control HW prefetcher (requires Secure Boot disabled) |
| `--bypass-cache` | Use non-temporal stores |
| `--hugepage` | Use 2MB huge pages |
| `--result-dir <dir>` | Output directory |
| `--help` | Show all options |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
