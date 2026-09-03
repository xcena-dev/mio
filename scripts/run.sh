#!/bin/bash
# The flush matches the bench: a read bench flushes with seq_read, a write bench
# with seq_write. A read-only flush leaves the device DRAM cache clean, and then
# the opening segment of a write bench evicts clean lines with no SSD write-back
# and measures too fast. See flush_stages() in run_benchmark.py for the same
# logic.

FLUSH_THREADS=8         # flush always runs with 8 threads
FLUSH_MEM=1048576       # MiB total; 1 TiB. Override with --flush-mem
FLUSH_BLOCK_SIZE=1048576

INIT=false
MEMBIND=""
FLUSH_MEM_ARG=""

ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--init" ]; then
    INIT=true
  elif [ "$arg" = "--membind" ]; then
    MEMBIND="next"
  elif [ "$MEMBIND" = "next" ]; then
    MEMBIND="$arg"
  elif [ "$arg" = "--flush-mem" ]; then
    FLUSH_MEM_ARG="next"
  elif [ "$FLUSH_MEM_ARG" = "next" ]; then
    FLUSH_MEM_ARG="$arg"
  else
    ARGS+=("$arg")
  fi
done

if [ -n "$FLUSH_MEM_ARG" ] && [ "$FLUSH_MEM_ARG" != "next" ]; then
  FLUSH_MEM="$FLUSH_MEM_ARG"
fi

MODE=${ARGS[0]}
THREADS=${ARGS[1]:-8}
BLOCK_SIZE=${ARGS[2]:-1048576}
TOTAL_MEM=${ARGS[3]:-1048576}   # memory-per-thread = total-mem / threads
DEV_DAX=${ARGS[4]:-}

if [ -z "$MODE" ] || [ "$MODE" = "--help" ] || [ "$MODE" = "-h" ]; then
  echo "Usage: $0 [--init] [--membind <node>] [--flush-mem <MiB>] <mode> [threads] [block-size] [total-mem] [devdax]"
  echo ""
  echo "Flushes before measuring: seq_read for a read mode, seq_write for a write"
  echo "mode so the device cache is left dirty. The flush region starts right"
  echo "after the bench region, at total-mem MiB."
  echo ""
  echo "Supported modes: seq_read, seq_write, random_read, random_write, stride_read, stride_write, zipfian_read"
  echo ""
  echo "Memory target (one required):"
  echo "  --membind <node>    Use NUMA memory node (e.g., 0, 4)"
  echo "  [devdax]            DevDAX device path as 5th positional arg (e.g., /dev/dax0.0)"
  echo ""
  echo "Options:"
  echo "  --init                            Run seq_write initialization before benchmark"
  echo "  --flush-mem <MiB>                 Total flush size, split over ${FLUSH_THREADS} threads (default: $FLUSH_MEM = 1 TiB)"
  echo ""
  echo "Defaults: threads=8, block-size=1048576(1MiB), total-mem=1048576(1TiB)"
  echo ""
  echo "Examples:"
  echo "  $0 seq_read 8 1048576 1048576 /dev/dax0.0      # DevDAX"
  echo "  $0 --membind 0 seq_read 4 1048576 4096         # NUMA node 0"
  echo "  $0 --membind 0 random_write 8 1048576 524288   # write bench, write flush"
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

if [ $((FLUSH_MEM % FLUSH_THREADS)) -ne 0 ]; then
  echo "ERROR: FLUSH_MEM($FLUSH_MEM) is not divisible by flush threads($FLUSH_THREADS)"
  exit 1
fi

FLUSH_MEM_PER_THREAD=$((FLUSH_MEM / FLUSH_THREADS))

MEM_PER_THREAD=$((TOTAL_MEM / THREADS))

case "$MODE" in
  seq_write|random_write|stride_write)          FLUSH_MODE="seq_write" ;;
  seq_read|random_read|stride_read|zipfian_read) FLUSH_MODE="seq_read" ;;
  *)
    echo "ERROR: unsupported mode '$MODE'"
    echo "Supported modes: seq_read, seq_write, random_read, random_write, stride_read, stride_write, zipfian_read"
    echo "Aggregate modes are not supported; run once per mode so each gets its own flush."
    exit 1
    ;;
esac

# The flush region sits right after the bench region, which starts at offset 0.
FLUSH_OFFSET=$(printf "0x%x" $((TOTAL_MEM * 1024 * 1024)))

echo "mode: $MODE"
echo "threads: $THREADS"
echo "memory-per-thread: $MEM_PER_THREAD"
echo "block-size: $BLOCK_SIZE"
echo "memory target: $MEM_ARGS"
echo "flush: $FLUSH_MODE $FLUSH_MEM MiB @ $FLUSH_OFFSET"

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
  --mode "$FLUSH_MODE" \
  --threads "$FLUSH_THREADS" \
  --memory-per-thread "$FLUSH_MEM_PER_THREAD" \
  --block-size "$FLUSH_BLOCK_SIZE" \
  $MEM_ARGS \
  --offset "$FLUSH_OFFSET"

sudo "$MICROBENCH" \
  --mode "$MODE" \
  --threads "$THREADS" \
  --memory-per-thread "$MEM_PER_THREAD" \
  --block-size "$BLOCK_SIZE" \
  --detail \
  $MEM_ARGS
