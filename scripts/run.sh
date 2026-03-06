#!/bin/bash
INIT=false
MEMBIND=""

ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--init" ]; then
    INIT=true
  elif [ "$arg" = "--membind" ]; then
    MEMBIND="next"
  elif [ "$MEMBIND" = "next" ]; then
    MEMBIND="$arg"
  else
    ARGS+=("$arg")
  fi
done

MODE=${ARGS[0]}
THREADS=${ARGS[1]:-16}
BLOCK_SIZE=${ARGS[2]:-1048576}
TOTAL_MEM=${ARGS[3]:-1048576}   # memory-per-thread = total-mem / threads
DEV_DAX=${ARGS[4]:-}

if [ -z "$MODE" ] || [ "$MODE" = "--help" ] || [ "$MODE" = "-h" ]; then
  echo "Usage: $0 [--init] [--membind <node>] <mode> [threads] [block-size] [total-mem] [devdax]"
  echo ""
  echo "Supported modes: seq_read, seq_write, random_read, random_write, stride_read, stride_write, zipfian_read"
  echo ""
  echo "Memory target (one required):"
  echo "  --membind <node>    Use NUMA memory node (e.g., 0, 4)"
  echo "  [devdax]            DevDAX device path as 5th positional arg (e.g., /dev/dax0.0)"
  echo ""
  echo "Options:"
  echo "  --init    Run seq_write initialization before benchmark"
  echo ""
  echo "Defaults: threads=16, block-size=1048576(1MiB), total-mem=1048576(1TiB)"
  echo ""
  echo "Examples:"
  echo "  $0 seq_read 16 1048576 1048576 /dev/dax0.0    # DevDAX"
  echo "  $0 --membind 0 seq_read 4 1048576 4096         # NUMA node 0"
  exit 1
fi

# Determine memory binding args
if [ -n "$MEMBIND" ]; then
  MEM_ARGS="--membind $MEMBIND"
elif [ -n "$DEV_DAX" ]; then
  MEM_ARGS="--devdax $DEV_DAX"
else
  echo "ERROR: Must specify either --membind <node> or devdax device path as 5th arg"
  exit 1
fi

if [ $((TOTAL_MEM % THREADS)) -ne 0 ]; then
  echo "ERROR: TOTAL_MEM($TOTAL_MEM) is not divisible by threads($THREADS)"
  exit 1
fi

MEM_PER_THREAD=$((TOTAL_MEM / THREADS))

echo "mode: $MODE"
echo "threads: $THREADS"
echo "memory-per-thread: $MEM_PER_THREAD"
echo "block-size: $BLOCK_SIZE"
echo "memory target: $MEM_ARGS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MICROBENCH="$REPO_ROOT/build/microbench"

# init
if [ "$INIT" = true ]; then
  echo "[INIT] Running seq_write initialization"
  sudo "$MICROBENCH" \
    --mode "seq_write" \
    --threads "$THREADS" \
    --memory-per-thread "$MEM_PER_THREAD" \
    --block-size "$BLOCK_SIZE" \
    $MEM_ARGS
fi

sudo "$MICROBENCH" \
  --mode "seq_read" \
  --threads "16" \
  --memory-per-thread "65536" \
  --block-size "1048576" \
  $MEM_ARGS \
  --offset "0x10000000000"

sudo "$MICROBENCH" \
  --mode "$MODE" \
  --threads "$THREADS" \
  --memory-per-thread "$MEM_PER_THREAD" \
  --block-size "$BLOCK_SIZE" \
  $MEM_ARGS
