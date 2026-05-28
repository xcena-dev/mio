#!/usr/bin/env python3
"""
Memory Benchmark Automation Script

Usage:
  python3 run_benchmark.py --devdax /dev/dax0.0          # full benchmark using DevDAX
  python3 run_benchmark.py --membind 4                   # full benchmark using NUMA node
  python3 run_benchmark.py --devdax /dev/dax0.0 --large-mem   # large memory only (cache miss)
  python3 run_benchmark.py --membind 4 --small-mem            # small memory only (cache hit)
  python3 run_benchmark.py --devdax /dev/dax0.0 4        # quick mode

Cache behavior:
  --large-mem (65536 MiB/thread):  Dummy read uses large offset -> cache flush (miss)
  --small-mem (4096 MiB/thread):   Dummy read uses offset 0     -> cache warmup (hit)

Repetition:
  Only the small (cache-hit) region is jittery, so it runs NUM_SETS
  (warmup + bench) sets per mode and records the median bandwidth (raw
  values printed for variance inspection). The large region is stable and
  runs a single (flush + bench) set.

Full mode:
  seq_read, seq_write,
  random_read  x3 block sizes (2M, 1M, 512K),
  random_write x3 block sizes,
  zipfian_read x3 block sizes

Quick mode (4):
  seq_read, seq_write, random_read (1MiB), random_write (1MiB)
"""

import argparse
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MICROBENCH = REPO_ROOT / "build" / "microbench"
SUMMARY_DIR = REPO_ROOT / "summary"
RESULT_DIR = REPO_ROOT / "result"

LARGE_MEM = 65536  # MiB per thread, cache-miss scenario
SMALL_MEM = 4096   # MiB per thread, cache-hit scenario
MEMORY_PER_THREAD = [LARGE_MEM, SMALL_MEM]

BLOCK_SIZES = [524288, 1048576, 2097152]
FLUSH_OFFSET = "0x10000000000"

NUM_SETS = 5  # small-mem repetitions; median is taken there. large-mem always runs once.


def build_tests(mode_count: int | None) -> list[tuple[str, int | None]]:
    """Return the list of real (mode, block_size) measurements. No dummies."""
    if mode_count == 4:
        return [
            ("seq_read", None),
            ("seq_write", None),
            ("random_read", 1048576),
            ("random_write", 1048576),
        ]
    tests: list[tuple[str, int | None]] = [
        ("seq_read", None),
        ("seq_write", None),
    ]
    for mode in ("random_read", "zipfian_read"):
        for bs in BLOCK_SIZES:
            tests.append((mode, bs))
    tests.append(("random_write", 1048576))
    return tests


def parse_bandwidth(output: str) -> float | None:
    m = re.search(r"Bandwidth:\s+([\d.]+)\s+GB/s", output)
    return float(m.group(1)) if m else None


def command_rt(cmd_list, is_exit=True):
    """Run a command and print stdout in real-time."""
    out = ""
    print(f"{' '.join(cmd_list)}")

    proc = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            out += line
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Terminating subprocess...")
        proc.send_signal(signal.SIGINT)
        proc.wait()
        sys.exit(-1)
    finally:
        proc.stdout.close()

    ret = proc.wait()

    if ret != 0 and is_exit:
        print(f"\nfailed {ret}")
        sys.exit(-1)

    return ret, out


def build_cmd(mode: str, memory_per_thread: int, mem_args: list[str],
              offset: str = "0x0",
              block_size: int | None = None, result_dir: Path | None = None) -> list[str]:
    cmd = [
        "sudo", str(MICROBENCH),
        "--mode", mode,
        "--memory-per-thread", str(memory_per_thread),
        "--threads", "16",
        *mem_args,
        "--offset", offset,
    ]
    if block_size:
        cmd += ["--block-size", str(block_size)]
    if result_dir:
        cmd += ["--result-dir", str(result_dir)]
    return cmd


def run_init(base_result_dir: Path, memory_per_thread: int, mem_args: list[str]):
    """Run initial seq_write to initialize memory region."""
    print(f"\n{'#'*60}")
    print(f"  [INIT] seq_write to initialize memory")
    print(f"{'#'*60}")

    init_dir = base_result_dir / "00_init_seq_write"
    init_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd("seq_write", memory_per_thread, mem_args, result_dir=init_dir)
    ret, _ = command_rt(cmd)
    if ret != 0:
        print("Init seq_write failed. Aborting.")
        sys.exit(1)
    print("  Init done.\n")


def run_step(cmd: list[str], label: str, step_dir: Path) -> tuple[bool, float | None]:
    """Run one microbench invocation. Returns (ok, bandwidth_gbps)."""
    step_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Result: {step_dir}")
    print(f"{'='*60}")

    start = time.time()
    ret, out = command_rt(cmd)
    elapsed = time.time() - start

    if ret != 0:
        print(f"  FAILED (exit code {ret}, {elapsed:.1f}s)")
        return False, None
    bw = parse_bandwidth(out)
    print(f"  Done ({elapsed:.1f}s)" + (f"  BW={bw:.2f} GB/s" if bw is not None else ""))
    return True, bw


def parse_args():
    parser = argparse.ArgumentParser(description="Memory Benchmark Automation Script")
    parser.add_argument(
        "mode_count", nargs="?", type=int, default=None,
        help="Quick mode: 4 for seq r/w + random r/w (1MiB block)",
    )

    mem_group = parser.add_mutually_exclusive_group(required=True)
    mem_group.add_argument("--devdax", type=str, metavar="DEVICE",
                           help="DevDAX device path (e.g., /dev/dax0.0)")
    mem_group.add_argument("--membind", type=int, metavar="NODE",
                           help="NUMA memory node (e.g., 0, 4)")

    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--small-mem", action="store_true",
                            help=f"{SMALL_MEM} MiB/thread x 16 threads — cache hit scenario")
    size_group.add_argument("--large-mem", action="store_true",
                            help=f"{LARGE_MEM} MiB/thread x 16 threads — cache miss scenario")

    parser.add_argument("--no-init", action="store_true",
                        help="Skip initial seq_write (memory already initialized)")

    return parser.parse_args()


def main():
    if not MICROBENCH.exists():
        print(f"Error: microbench binary not found at {MICROBENCH}")
        print("Run build.sh first.")
        sys.exit(1)

    args = parse_args()

    if args.devdax:
        mem_args = ["--devdax", args.devdax]
    else:
        mem_args = ["--membind", str(args.membind)]

    if args.small_mem:
        memory_sizes = [SMALL_MEM]
    elif args.large_mem:
        memory_sizes = [LARGE_MEM]
    else:
        memory_sizes = MEMORY_PER_THREAD

    tests = build_tests(args.mode_count)
    total_modes = len(tests)

    # Aggregated medians for the final stdout summary
    results: dict[int, list[tuple[str, int | None, float]]] = {}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_result_dir = RESULT_DIR / timestamp

    summary_subdir = SUMMARY_DIR / timestamp
    summary_subdir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_subdir / "benchmark_summary.txt"

    preset = f"quick ({args.mode_count})" if args.mode_count else "full"
    mem_target = f"devdax={args.devdax}" if args.devdax else f"membind={args.membind}"
    print(f"Memory Benchmark Automation  [{preset}]")
    print(f"Memory target: {mem_target}")
    print(f"Base results directory: {base_result_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Memory per thread sizes: {memory_sizes}")
    print(f"Modes per memory size: {total_modes}  "
          f"(small={NUM_SETS} sets/median, large=1 set)")

    with open(summary_file, "w") as f:
        f.write(f"Memory Benchmark Summary  ({timestamp})  [{preset}]\n")
        f.write(f"{'='*70}\n")
        f.write(f"{'MemPerThread':>12} {'Mode':<20} {'Block Size':>12} {'Bandwidth':>15}\n")
        f.write(f"{'-'*12} {'-'*20} {'-'*12} {'-'*15}\n")

    if not args.no_init:
        run_init(base_result_dir, memory_sizes[0], mem_args)

    for mem_size in memory_sizes:
        print(f"\n{'#'*60}")
        print(f"  Running with --memory-per-thread {mem_size}")
        print(f"{'#'*60}")

        result_dir = base_result_dir / f"mem_{mem_size}"
        cache_hit = mem_size == SMALL_MEM
        # Only the small (cache-hit) region is jittery, so repeat + median there.
        # Large region is stable, so a single set is enough.
        num_sets = NUM_SETS if cache_hit else 1
        dummy_tag = "warmup" if cache_hit else "flush"
        dummy_offset = "0x0" if cache_hit else FLUSH_OFFSET
        results[mem_size] = []

        for mode_idx, (mode, block_size) in enumerate(tests, start=1):
            mode_label = mode + (f"_bs{block_size}" if block_size else "")
            mode_dir = result_dir / f"{mode_idx:02d}_{mode_label}"

            print(f"\n{'#'*60}")
            print(f"  Mode {mode_idx}/{total_modes}: {mode_label}  ({num_sets} sets)")
            print(f"{'#'*60}")

            bws: list[float] = []
            for set_idx in range(1, num_sets + 1):
                # warmup (cache_hit) or flush (cache_miss) dummy
                dummy_dir = mode_dir / f"set{set_idx}_{dummy_tag}"
                dummy_cmd = build_cmd("seq_read", mem_size, mem_args,
                                      dummy_offset, None, dummy_dir)
                ok, _ = run_step(
                    dummy_cmd,
                    f"[{dummy_tag.upper()}] mode {mode_idx}/{total_modes} {mode_label}  set {set_idx}/{num_sets}",
                    dummy_dir,
                )
                if not ok:
                    print(f"\n{mode_label} set{set_idx} {dummy_tag} failed. Aborting.")
                    sys.exit(1)

                # bench
                bench_dir = mode_dir / f"set{set_idx}_bench"
                bench_cmd = build_cmd(mode, mem_size, mem_args,
                                      "0x0", block_size, bench_dir)
                ok, bw = run_step(
                    bench_cmd,
                    f"[BENCH]  mode {mode_idx}/{total_modes} {mode_label}  set {set_idx}/{num_sets}",
                    bench_dir,
                )
                if not ok:
                    print(f"\n{mode_label} set{set_idx} bench failed. Aborting.")
                    sys.exit(1)
                if bw is not None:
                    bws.append(bw)

            if bws:
                med = statistics.median(bws)
                bs_str = f"{block_size} B" if block_size else "-"
                bw_str = f"{med:.2f} GB/s"
                line = f"{mem_size:>12} {mode:<20} {bs_str:>12} {bw_str:>15}"
                if cache_hit:
                    raw_str = "[" + ", ".join(f"{x:.2f}" for x in bws) + "]"
                    print(f"\n  >> MEDIAN  {line}  raw={raw_str}")
                else:
                    print(f"\n  >> RESULT  {line}")
                with open(summary_file, "a") as f:
                    f.write(line + "\n")
                    f.flush()
                results[mem_size].append((mode, block_size, med))
            else:
                print(f"  WARNING: no bandwidth parsed for {mode_label}")

        with open(summary_file, "a") as f:
            f.write(f"{'-'*70}\n")

    with open(summary_file, "a") as f:
        f.write(f"{'='*70}\n")

    print(f"\n{'='*60}")
    print(f"  All steps completed successfully!")
    print(f"  Results: {base_result_dir}")
    print(f"  Summary: {summary_file}")
    print(f"{'='*60}")

    # Final aggregated summary on stdout (median per mode, grouped by mem size)
    print(f"\n{'='*60}")
    print(f"  Final Summary")
    print(f"{'='*60}")
    for mem_size in memory_sizes:
        if mem_size == SMALL_MEM:
            header = "[Small size]"
        elif mem_size == LARGE_MEM:
            header = "[Large size]"
        else:
            header = f"[mem {mem_size}]"
        print(f"\n{header}")
        for mode, block_size, med in results.get(mem_size, []):
            bs_str = f"{block_size} B" if block_size else "-"
            bw_str = f"{med:.2f} GB/s"
            print(f"  {mode:<20} {bs_str:>12} {bw_str:>15}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
