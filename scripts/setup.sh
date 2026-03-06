#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  echo "Usage: sudo $0 [OPTIONS]"
  echo ""
  echo "Benchmark environment setup script"
  echo ""
  echo "Options:"
  echo "  --build-only        Build only (skip environment setup)"
  echo "  --check             Check current environment status only"
  echo "  --with-cxl          Configure CXL device to DevDAX mode"
  echo "  --devdax <device>   Device to convert (default: dax0.0, requires --with-cxl)"
  echo ""
  echo "Setup steps:"
  echo "  1. Install dependencies"
  echo "  2. Load MSR kernel module"
  echo "  3. (Optional) Convert CXL device to devdax mode (--with-cxl)"
  echo "  4. Disable NUMA auto balancing"
  echo "  5. Disable swap"
  echo "  6. Build benchmark"
}

DEV_DAX="dax0.0"
BUILD_ONLY=false
CHECK_ONLY=false
WITH_CXL=false

while [ $# -gt 0 ]; do
  case "$1" in
    --devdax)    DEV_DAX="$2"; shift 2 ;;
    --build-only) BUILD_ONLY=true; shift ;;
    --check)     CHECK_ONLY=true; shift ;;
    --with-cxl)  WITH_CXL=true; shift ;;
    --help|-h)   usage; exit 0 ;;
    *)           echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

check_env() {
  echo "=== Environment Status ==="
  echo ""

  # devdax
  if [ -e "/dev/$DEV_DAX" ]; then
    echo "[OK] /dev/$DEV_DAX exists"
  else
    echo "[--] /dev/$DEV_DAX not found (use --with-cxl to configure)"
  fi

  # NUMA balancing
  if [ -f /proc/sys/kernel/numa_balancing ]; then
    val=$(cat /proc/sys/kernel/numa_balancing)
    if [ "$val" = "0" ]; then
      echo "[OK] NUMA auto balancing OFF"
    else
      echo "[--] NUMA auto balancing ON (current: $val)"
    fi
  else
    echo "[OK] NUMA balancing not available in kernel (disabled)"
  fi

  # Swap
  swap_count=$(swapon --show --noheadings 2>/dev/null | wc -l)
  if [ "$swap_count" -eq 0 ]; then
    echo "[OK] Swap OFF"
  else
    echo "[--] Swap ON ($swap_count active)"
  fi

  # MSR module
  if lsmod | grep -q "^msr"; then
    echo "[OK] MSR module loaded"
  else
    echo "[--] MSR module not loaded"
  fi

  # Build
  if [ -x "$REPO_ROOT/build/microbench" ]; then
    echo "[OK] microbench built"
  else
    echo "[--] microbench not built"
  fi

  echo ""
}

if [ "$CHECK_ONLY" = true ]; then
  check_env
  exit 0
fi

if [ "$EUID" -ne 0 ] && [ "$BUILD_ONLY" = false ]; then
  echo "ERROR: Environment setup requires root. Run with sudo."
  echo "To build only: $0 --build-only"
  exit 1
fi

if [ "$BUILD_ONLY" = false ]; then
  echo "=== 1. Install dependencies ==="
  apt-get update -qq
  apt-get install -y -qq cmake ninja-build g++ gnuplot msr-tools numactl ndctl
  echo ""

  echo "=== 2. Load MSR kernel module ==="
  modprobe msr
  echo "done"
  echo ""

  if [ "$WITH_CXL" = true ]; then
    echo "=== 3. Convert CXL device to devdax mode ==="
    if [ -e "/dev/$DEV_DAX" ]; then
      echo "/dev/$DEV_DAX already exists, skipping"
    else
      echo "daxctl reconfigure-device --human --mode=devdax --force $DEV_DAX"
      daxctl reconfigure-device --human --mode=devdax --force "$DEV_DAX"
    fi
    echo ""
  else
    echo "=== 3. Skipping CXL device setup (use --with-cxl to enable) ==="
    echo ""
  fi

  echo "=== 4. Disable NUMA auto balancing ==="
  if [ -f /proc/sys/kernel/numa_balancing ]; then
    echo 0 | tee /proc/sys/kernel/numa_balancing
    echo "done"
  else
    echo "numa_balancing not available in kernel (already disabled)"
  fi
  echo ""

  echo "=== 5. Disable swap ==="
  swapoff -a
  echo "swap status:"
  cat /proc/swaps
  echo ""
fi

echo "=== 6. Build benchmark ==="
cd "$REPO_ROOT"
./build.sh
echo ""

check_env

echo "Setup complete."
