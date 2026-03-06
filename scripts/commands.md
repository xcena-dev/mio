# Benchmark Commands Reference

Raw `./build/microbench` commands. All commands should be run from the repository root.

## Units

| Parameter | Unit | Example |
|-----------|------|---------|
| `--block-size` | Bytes | `524288` = 512 KiB, `1048576` = 1 MiB, `2097152` = 2 MiB |
| `--memory-per-thread` | MiB | `65536` = 64 GiB, `8192` = 8 GiB |

## Using NUMA Memory (`--membind`)

### Sequential Access

```bash
# Sequential read (NUMA node 0)
sudo ./build/microbench --mode seq_read --threads 4 --memory-per-thread 1024 --membind 0

# Sequential write (NUMA node 0)
sudo ./build/microbench --mode seq_write --threads 4 --memory-per-thread 1024 --membind 0
```

### Random Access

```bash
# Random read (NUMA node 0, block-size=1MiB)
sudo ./build/microbench --mode random_read --threads 8 --memory-per-thread 512 --block-size 1048576 --membind 0

# Random write (NUMA node 0, block-size=1MiB)
sudo ./build/microbench --mode random_write --threads 8 --memory-per-thread 512 --block-size 1048576 --membind 0
```

## Using DevDAX Device (`--devdax`)

### Sequential Access

```bash
# Sequential read
sudo ./build/microbench --mode seq_read --threads 16 --memory-per-thread 65536 --devdax /dev/dax0.0

# Sequential write
sudo ./build/microbench --mode seq_write --threads 16 --memory-per-thread 65536 --devdax /dev/dax0.0
```

### Random Access

```bash
# Random read (block-size=512KiB)
sudo ./build/microbench --mode random_read --threads 16 --memory-per-thread 65536 --block-size 524288 --devdax /dev/dax0.0

# Random read (block-size=1MiB)
sudo ./build/microbench --mode random_read --threads 16 --memory-per-thread 65536 --block-size 1048576 --devdax /dev/dax0.0

# Random read (block-size=2MiB)
sudo ./build/microbench --mode random_read --threads 16 --memory-per-thread 65536 --block-size 2097152 --devdax /dev/dax0.0

# Random write (block-size=1MiB)
sudo ./build/microbench --mode random_write --threads 16 --memory-per-thread 65536 --block-size 1048576 --devdax /dev/dax0.0
```

### Zipfian Access

```bash
# Zipfian read (block-size=512KiB)
sudo ./build/microbench --mode zipfian_read --threads 16 --memory-per-thread 65536 --block-size 524288 --devdax /dev/dax0.0

# Zipfian read (block-size=1MiB)
sudo ./build/microbench --mode zipfian_read --threads 16 --memory-per-thread 65536 --block-size 1048576 --devdax /dev/dax0.0

# Zipfian read (block-size=2MiB)
sudo ./build/microbench --mode zipfian_read --threads 16 --memory-per-thread 65536 --block-size 2097152 --devdax /dev/dax0.0
```

### Stride Access

```bash
# Stride read
sudo ./build/microbench --mode stride_read --threads 16 --memory-per-thread 65536 --devdax /dev/dax0.0

# Stride write
sudo ./build/microbench --mode stride_write --threads 16 --memory-per-thread 65536 --devdax /dev/dax0.0
```

## Thread Count Variations

```bash
# 32 threads, sequential read (DevDAX)
sudo ./build/microbench --mode seq_read --threads 32 --memory-per-thread 65536 --devdax /dev/dax0.0

# 64 threads, random read (DevDAX)
sudo ./build/microbench --mode random_read --threads 64 --memory-per-thread 65536 --block-size 1048576 --devdax /dev/dax0.0

# 4 threads, sequential read (NUMA)
sudo ./build/microbench --mode seq_read --threads 4 --memory-per-thread 2048 --membind 0
```
