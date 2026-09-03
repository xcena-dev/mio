#include <errno.h>
#include <fcntl.h>
#include <numa.h>
#include <numaif.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <sstream>
#include <string>
#include <vector>

#include "benchmark.h"
#include "hw_prefetcher.h"
#include "numa_affinity.h"
#include "output.h"
#ifdef ENABLE_TRACING
#include "tracing.h"
#endif

#ifdef ENABLE_LATENCY_MEASURE
#include "latency_measure.h"
#endif

using namespace std;

static void printDetail(const BandwidthResult &r)
{
  if (!g_detail_enabled)
    return;
  const RegionBreakdown &g = r.regions;
  auto line = [](const char *name, const char *span, double gbps, double bytes,
                 double ms)
  {
    printf("  %-6s %s: %6.2f GB/s (%8.2f GiB in %9.2f ms)\n", name, span, gbps,
           bytes / (1024.0 * 1024.0 * 1024.0), ms);
  };
  if (!g.valid)
  {
    printf("  (too few trace samples to split into regions; the figure above "
           "is the whole run)\n");
    return;
  }

  double lo = g_steady_lo_frac * 100.0, hi = g_steady_hi_frac * 100.0;
  char head_span[16], steady_span[16], tail_span[16];
  snprintf(head_span, sizeof(head_span), "[%3.0f-%3.0f%%]", 0.0, lo);
  snprintf(steady_span, sizeof(steady_span), "[%3.0f-%3.0f%%]", lo, hi);
  snprintf(tail_span, sizeof(tail_span), "[%3.0f-%3.0f%%]", hi, 100.0);
  line("head", head_span, g.head_gbps, g.head_bytes, g.head_ms);
  line("steady", steady_span, g.steady_gbps, g.steady_bytes, g.steady_ms);
  line("tail", tail_span, g.tail_gbps, g.tail_bytes, g.tail_ms);
}

void printUsage(const char *prog_name)
{
  printf("Usage: %s [OPTIONS]\n", prog_name);
  printf("\n");
  printf("Core Options:\n");
  printf(
      "  --mode <mode>                 Test modes (comma-separated, default: "
      "seq,random):\n");
  printf(
      "                                  Detailed: seq_read, seq_write, "
      "random_read, random_write, stride_read, stride_write, zipfian_read, pointer_chase\n");
  printf(
      "                                  Combined: seq (=seq_read+seq_write), "
      "random, stride, zipfian, all\n");
  printf(
      "  --threads <num>               Number of threads to use (default: "
      "1)\n");
  printf(
      "  --memory-per-thread <MiB>     Memory size per thread in MiB (default: "
      "100)\n");
  printf(
      "  --block-size <bytes>          Block size for access (default: 64)\n");
  printf(
      "  --stride-size <bytes>         Stride size for stride access (default: "
      "64)\n");
  printf(
      "  --inject-delay <cycles>       Inject delay for pointer_chase mode "
      "(default: 0)\n");
  printf(
      "  --zipfian-alpha <value>       Zipfian skew parameter (default: 0.99)\n");
  printf("\n");
  printf("Memory Binding (mutually exclusive):\n");
  printf("  --devdax <device>      DevDAX device path (e.g., /dev/dax0.0)\n");
  printf(
      "  --offset <bytes>       Start offset for devdax mmap (default: 0)\n");
  printf("  --membind <node>       NUMA memory node (e.g., 4)\n");
  printf("\n");
  printf("CPU Affinity:\n");
  printf(
      "  --cpu-affinity <nodes> Pin threads to CPUs of NUMA node(s) (e.g., 0 "
      "or 0,2,4)\n");
  printf("\n");
  printf("Performance Control:\n");
  printf(
      "  --prefetch <ON|OFF>    Hardware prefetcher control (optional, "
      "requires Secure Boot disabled)\n");
  printf(
      "  --bypass-cache         Enable non-temporal loads/stores (cache bypass)\n");
  printf(
      "  --hugepage             Use 2MB huge pages for allocation (requires "
      "huge pages reserved)\n");
  printf("\n");
  printf("Output Control:\n");
  printf(
      "  --result-dir <path>    Custom result directory (default: auto-create "
      "result/YYYY-MM-DD_HH-MM-SS/)\n");
  printf(
      "  --no-progress          Suppress the live progress lines (output only; "
      "sampling still runs)\n");
  printf(
      "  --detail               Also print the head/steady/tail breakdown "
      "under each result\n");
  printf(
      "                         (regions split at the --steady-range "
      "boundaries of total bytes transferred)\n");
  printf(
      "  --steady-range <lo>:<hi>  Steady region boundaries as a percentage of "
      "total bytes (default: 40:60)\n");
  printf(
      "                         Requires 0 < lo < hi < 100. A boundary on 0 or "
      "100 empties a region, which makes every measurement fall back to the "
      "whole-run rate\n");
  printf(
      "  --trace-interval-ms <ms>  Bandwidth time-series tick written to the "
      "thread_trace CSV (default: 50)\n");
  printf(
      "                         A tick longer than the run leaves too few "
      "samples to split regions, so the reported figure falls back to the "
      "whole-run rate\n");
  printf("\n");
  printf("Reported bandwidth:\n");
  printf(
      "  The GB/s figure per mode is the STEADY region -- the byte window set "
      "by --steady-range -- so\n");
  printf(
      "  device warm-up and the uneven thread-completion drain are excluded. "
      "The time in\n");
  printf(
      "  parentheses is the WHOLE run, i.e. how long the measurement took.\n");
  printf("\n");
  printf("Other:\n");
  printf("  --help, -h             Show this help message\n");
  printf("\n");
  printf("Examples:\n");
  printf("  sudo %s --mode seq,random --threads 4 --membind 4\n", prog_name);
  printf(
      "  sudo %s --mode seq_read --threads 128 --prefetch ON --devdax "
      "/dev/dax0.0\n",
      prog_name);
  printf(
      "  sudo %s --mode stride_read,stride_write --threads 64 --stride-size "
      "4096 --membind 4\n",
      prog_name);
  printf("\n");
  printf("Note:\n");
  printf("  - Run with specific thread count (no iteration)\n");
  printf("  - Results saved in timestamped CSV file with dynamic headers\n");
  printf(
      "  - --prefetch option requires MSR access (disable Secure Boot in "
      "BIOS)\n");
  printf("  - Without --prefetch, uses system default prefetcher settings\n");
}

int main(int argc, char *argv[])
{
  int memory_per_thread_mib =
      -1;
  string mode = "seq,random";
  int num_threads = 1;
  const char *result_dir_path = nullptr;
  const char *devdax_path = nullptr;
  int membind_node = -1;
  string prefetch_option =
      "";
  bool bypass_cache = false;
  const char *cpu_affinity_str = nullptr;
  uint64_t inject_delay_cycles =
      0;
  bool use_hugepage = false;
  size_t devdax_offset = 0;
  double zipfian_alpha = 0.99;

  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGSEGV, signal_handler);
  signal(SIGBUS, signal_handler);

  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--memory-per-thread") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --memory-per-thread requires a value\n");
        return 1;
      }
      memory_per_thread_mib = atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "--mode") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --mode requires a value\n");
        return 1;
      }
      mode = argv[++i];
    }
    else if (strcmp(argv[i], "--threads") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --threads requires a value\n");
        return 1;
      }
      num_threads = atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "--result-dir") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --result-dir requires a value\n");
        return 1;
      }
      result_dir_path = argv[++i];
    }
    else if (strcmp(argv[i], "--devdax") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --devdax requires a value\n");
        return 1;
      }
      devdax_path = argv[++i];
    }
    else if (strcmp(argv[i], "--membind") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --membind requires a value\n");
        return 1;
      }
      membind_node = atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "--prefetch") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --prefetch requires a value\n");
        return 1;
      }
      prefetch_option = argv[++i];
      if (prefetch_option != "ON" && prefetch_option != "OFF")
      {
        fprintf(stderr, "Error: --prefetch must be ON or OFF\n");
        return 1;
      }
    }
    else if (strcmp(argv[i], "--block-size") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --block-size requires a value\n");
        return 1;
      }
      BLOCK_SIZE = atoi(argv[++i]);
      if (BLOCK_SIZE == 0)
      {
        fprintf(stderr, "Error: --block-size must be greater than 0\n");
        return 1;
      }
    }
    else if (strcmp(argv[i], "--stride-size") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --stride-size requires a value\n");
        return 1;
      }
      STRIDE_SIZE = atoi(argv[++i]);
      if (STRIDE_SIZE == 0)
      {
        fprintf(stderr, "Error: --stride-size must be greater than 0\n");
        return 1;
      }
    }
    else if (strcmp(argv[i], "--bypass-cache") == 0)
    {
      bypass_cache = true;
    }
    else if (strcmp(argv[i], "--no-progress") == 0)
    {
      g_progress_enabled = false;
    }
    else if (strcmp(argv[i], "--detail") == 0)
    {
      g_detail_enabled = true;
    }
    else if (strcmp(argv[i], "--steady-range") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --steady-range requires a value\n");
        return 1;
      }

      double lo = 0, hi = 0;
      char trailing = 0;
      if (sscanf(argv[++i], "%lf:%lf%c", &lo, &hi, &trailing) != 2)
      {
        fprintf(stderr,
                "Error: --steady-range expects <lo>:<hi> in percent, e.g. "
                "40:60 (got '%s')\n",
                argv[i]);
        return 1;
      }

      if (!(lo > 0.0) || !(hi > lo) || !(hi < 100.0))
      {
        fprintf(stderr,
                "Error: --steady-range requires 0 < lo < hi < 100 (got "
                "%g:%g)\n",
                lo, hi);
        return 1;
      }
      g_steady_lo_frac = lo / 100.0;
      g_steady_hi_frac = hi / 100.0;
    }
    else if (strcmp(argv[i], "--trace-interval-ms") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --trace-interval-ms requires a value\n");
        return 1;
      }
      g_thread_trace_interval_ms = atoi(argv[++i]);
      if (g_thread_trace_interval_ms <= 0)
      {
        fprintf(stderr, "Error: --trace-interval-ms must be greater than 0\n");
        return 1;
      }
    }
    else if (strcmp(argv[i], "--cpu-affinity") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --cpu-affinity requires a value\n");
        return 1;
      }
      cpu_affinity_str = argv[++i];
    }
    else if (strcmp(argv[i], "--inject-delay") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --inject-delay requires a value\n");
        return 1;
      }
      inject_delay_cycles = atoll(argv[++i]);
    }
    else if (strcmp(argv[i], "--hugepage") == 0)
    {
      use_hugepage = true;
    }
    else if (strcmp(argv[i], "--offset") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --offset requires a value\n");
        return 1;
      }
      devdax_offset = strtoull(argv[++i], nullptr, 0);
    }
    else if (strcmp(argv[i], "--zipfian-alpha") == 0)
    {
      if (i + 1 >= argc)
      {
        fprintf(stderr, "Error: --zipfian-alpha requires a value\n");
        return 1;
      }
      zipfian_alpha = atof(argv[++i]);
      if (zipfian_alpha <= 0)
      {
        fprintf(stderr, "Error: --zipfian-alpha must be greater than 0\n");
        return 1;
      }
    }
    else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
    {
      printUsage(argv[0]);
      return 0;
    }
    else
    {
      fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
      printUsage(argv[0]);
      return 1;
    }
  }

  if (devdax_path && membind_node >= 0)
  {
    fprintf(stderr, "Error: --devdax and --membind cannot be used together\n");
    return 1;
  }

  if (!devdax_path && membind_node < 0)
  {
    fprintf(stderr, "Error: You must specify either --devdax or --membind\n");
    return 1;
  }

  if (memory_per_thread_mib <= 0)
  {
    memory_per_thread_mib = 500;
  }

  vector<string> modes;
  if (mode == "all")
  {
    modes.push_back("seq");
    modes.push_back("random");
    modes.push_back("stride");
  }
  else
  {
    stringstream ss(mode);
    string token;
    while (getline(ss, token, ','))
    {
      if (token != "seq" && token != "random" && token != "stride" &&
          token != "seq_read" && token != "seq_write" &&
          token != "random_read" && token != "random_write" &&
          token != "stride_read" && token != "stride_write" &&
          token != "pointer_chase" &&
          token != "zipfian" && token != "zipfian_read")
      {
        fprintf(stderr, "Error: Invalid mode '%s'\n", token.c_str());
        fprintf(stderr,
                "Valid modes: seq_read, seq_write, random_read, random_write, "
                "stride_read, stride_write, zipfian_read, pointer_chase, seq, "
                "random, stride, zipfian, all\n");
        return 1;
      }
      modes.push_back(token);
    }
  }

  if (modes.empty())
  {
    fprintf(stderr, "Error: No valid modes specified\n");
    return 1;
  }

  vector<string> expanded_modes;
  for (const auto &m : modes)
  {
    if (m == "seq")
    {
      expanded_modes.push_back("seq_read");
      expanded_modes.push_back("seq_write");
    }
    else if (m == "random")
    {
      expanded_modes.push_back("random_read");
      expanded_modes.push_back("random_write");
    }
    else if (m == "stride")
    {
      expanded_modes.push_back("stride_read");
      expanded_modes.push_back("stride_write");
    }
    else if (m == "zipfian")
    {
      expanded_modes.push_back("zipfian_read");
    }
    else if (m == "all")
    {
      expanded_modes.push_back("seq_read");
      expanded_modes.push_back("seq_write");
      expanded_modes.push_back("random_read");
      expanded_modes.push_back("random_write");
      expanded_modes.push_back("stride_read");
      expanded_modes.push_back("stride_write");
    }
    else
    {
      expanded_modes.push_back(m);
    }
  }

  if (num_threads <= 0)
  {
    fprintf(stderr, "Error: threads must be positive\n");
    return 1;
  }

  vector<int> numa_nodes;
  vector<int> cpu_list;
  if (cpu_affinity_str)
  {
    numa_nodes = parseNumaNodes(cpu_affinity_str);
    if (numa_nodes.empty())
    {
      fprintf(stderr, "Error: Failed to parse --cpu-affinity argument\n");
      return 1;
    }

    if (!validateNumaNodes(numa_nodes))
    {
      return 1;
    }

    cpu_list = getCpusFromNumaNodes(numa_nodes);
    if (cpu_list.empty())
    {
      fprintf(stderr, "Error: No CPUs found in specified NUMA nodes\n");
      return 1;
    }

    printf("CPU affinity enabled: Using %zu CPUs from NUMA node(s): ",
           cpu_list.size());
    for (size_t i = 0; i < numa_nodes.size(); i++)
    {
      if (i > 0)
        printf(",");
      printf("%d", numa_nodes[i]);
    }
    printf("\n");
  }

  if (!prefetch_option.empty())
  {
    if (!save_prefetcher_state())
    {
      fprintf(stderr, "Error: Failed to read hardware prefetcher state.\n");
      fprintf(stderr,
              "Please disable Secure Boot in BIOS to enable MSR access.\n");
      return 1;
    }
    printf("Saved hardware prefetcher state\n");
    if (!set_prefetcher_state(prefetch_option == "ON"))
    {
      return 1;
    }
    printf("Hardware prefetcher: %s\n", prefetch_option.c_str());
  }
  else
  {
    printf("Hardware prefetcher: not controlled (using system default)\n");
  }

  if (membind_node >= 0)
  {
    if (numa_available() < 0)
    {
      fprintf(stderr, "Error: NUMA is not available\n");
      restore_prefetcher_state();
      return 1;
    }

    int max_node = numa_max_node();
    if (membind_node > max_node)
    {
      fprintf(stderr, "Error: NUMA node %d does not exist (max: %d)\n",
              membind_node, max_node);
      restore_prefetcher_state();
      return 1;
    }
    printf("Will allocate memory on NUMA node %d\n", membind_node);
  }

  char result_dir[512];
  if (result_dir_path)
  {
    snprintf(result_dir, sizeof(result_dir), "%s", result_dir_path);
  }
  else
  {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", t);
    snprintf(result_dir, sizeof(result_dir), "result/%s", timestamp);
  }

  char mkdir_cmd[1024];
  snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", result_dir);
  int mkdir_ret = system(mkdir_cmd);
  (void)mkdir_ret;

  g_thread_trace_dir = result_dir;

  char output_file_path_buf[1024];
  time_t now = time(NULL);
  struct tm *t = localtime(&now);
  char timestamp[64];
  strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", t);
  snprintf(output_file_path_buf, sizeof(output_file_path_buf), "%s/%s.txt",
           result_dir, timestamp);

  printf("\n");
  printf("Memory Bandwidth Measurement\n");
  printf("============================\n");
  printf("Memory per thread: %d MiB\n", memory_per_thread_mib);
  printf("Threads: %d\n", num_threads);
  printf("Mode: %s\n", mode.c_str());
  printf("Result Directory: %s\n", result_dir);
  printf("Output File: %s\n", output_file_path_buf);
  printf("\n");

  int total_memory_mib = memory_per_thread_mib * num_threads;
  size_t memory_size = (size_t)total_memory_mib * 1024 * 1024;

  void *data;
  int devdax_fd = -1;
  bool using_devdax = (devdax_path != nullptr);

  if (using_devdax)
  {
    devdax_fd = open(devdax_path, O_RDWR);
    if (devdax_fd < 0)
    {
      fprintf(stderr, "Error: Failed to open devdax device %s: %s\n",
              devdax_path, strerror(errno));
      restore_prefetcher_state();
      return 1;
    }

    data = mmap(NULL, memory_size, PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_POPULATE, devdax_fd, devdax_offset);
    if (data == MAP_FAILED)
    {
      fprintf(stderr, "Error: Failed to mmap %d MiB from %s: %s\n",
              total_memory_mib, devdax_path, strerror(errno));
      close(devdax_fd);
      restore_prefetcher_state();
      return 1;
    }
    printf("Using devdax device: %s (offset: %zu bytes, 0x%zx)\n", devdax_path,
           devdax_offset, devdax_offset);
  }
  else if (membind_node >= 0)
  {
    if (use_hugepage)
    {
      data = mmap(
          NULL, memory_size, PROT_READ | PROT_WRITE,
          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (21 << MAP_HUGE_SHIFT),
          -1, 0);

      if (data != MAP_FAILED)
      {
        unsigned long nodemask = 1UL << membind_node;
        if (mbind(data, memory_size, MPOL_BIND, &nodemask, sizeof(nodemask) * 8,
                  0) != 0)
        {
          fprintf(stderr,
                  "Warning: mbind failed, huge pages may not be on correct "
                  "NUMA node\n");
        }
        printf("Using NUMA node %d (2MB huge pages)\n", membind_node);
      }
      else
      {
        fprintf(stderr,
                "Error: Failed to allocate huge pages. Check: cat "
                "/proc/meminfo | grep Huge\n");
        fprintf(stderr,
                "Reserve huge pages: echo N | sudo tee "
                "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages\n");
        restore_prefetcher_state();
        return 1;
      }
    }
    else
    {
      data = numa_alloc_onnode(memory_size, membind_node);
      if (data == NULL)
      {
        fprintf(stderr, "Error: Failed to allocate %d MiB on NUMA node %d\n",
                total_memory_mib, membind_node);
        restore_prefetcher_state();
        return 1;
      }
      printf("Using NUMA node %d (regular pages)\n", membind_node);
    }
  }
  else
  {
    if (posix_memalign(&data, BLOCK_SIZE, memory_size) != 0)
    {
      fprintf(stderr, "Error: Failed to allocate %d MiB of memory\n",
              total_memory_mib);
      restore_prefetcher_state();
      return 1;
    }
    printf("Using system memory (posix_memalign)\n");
  }

  printf("Memory allocated successfully\n\n");

  g_cpu_affinity_list = cpu_list;

  saveEnvironmentJson(result_dir, mode, BLOCK_SIZE, STRIDE_SIZE, bypass_cache,
                      devdax_path);

#ifdef ENABLE_TRACING

  char tracing_dir[512];
  snprintf(tracing_dir, sizeof(tracing_dir), "%s/tracing", result_dir);
  mkdir(tracing_dir, 0755);
#endif

  bool run_seq_read = false, run_seq_write = false;
  bool run_random_read = false, run_random_write = false;
  bool run_stride_read = false, run_stride_write = false;
  bool run_pointer_chase = false;
  bool run_zipfian_read = false;
  for (const auto &m : expanded_modes)
  {
    if (m == "seq_read")
      run_seq_read = true;
    if (m == "seq_write")
      run_seq_write = true;
    if (m == "random_read")
      run_random_read = true;
    if (m == "random_write")
      run_random_write = true;
    if (m == "stride_read")
      run_stride_read = true;
    if (m == "stride_write")
      run_stride_write = true;
    if (m == "zipfian_read")
      run_zipfian_read = true;
    if (m == "pointer_chase")
      run_pointer_chase = true;
  }

  bool file_exists = false;
  bool has_header = false;
  FILE *check_file = fopen(output_file_path_buf, "r");
  if (check_file)
  {
    file_exists = true;
    char first_line[1024];
    if (fgets(first_line, sizeof(first_line), check_file))
    {
      if (strncmp(first_line, "# Threads", 9) == 0)
      {
        has_header = true;
      }
    }
    fclose(check_file);
  }

  FILE *output_file = fopen(output_file_path_buf, file_exists ? "a" : "w");
  if (!output_file)
  {
    fprintf(stderr, "Error: Failed to create output file '%s'\n",
            output_file_path_buf);
    if (using_devdax)
    {
      munmap(data, memory_size);
      close(devdax_fd);
    }
    else
    {
      free(data);
    }
    restore_prefetcher_state();
    return 1;
  }

  if (!has_header)
  {
    fprintf(output_file, "# Threads");
    if (run_seq_read)
    {
      fprintf(output_file, " SeqRead(GB/s) SeqRead(ms)");
    }
    if (run_seq_write)
    {
      fprintf(output_file, " SeqWrite(GB/s) SeqWrite(ms)");
    }
    if (run_random_read)
    {
      fprintf(output_file, " RandRead(GB/s) RandRead(ms)");
    }
    if (run_random_write)
    {
      fprintf(output_file, " RandWrite(GB/s) RandWrite(ms)");
    }
    if (run_stride_read)
    {
      fprintf(output_file, " StrideRead(GB/s) StrideRead(ms)");
    }
    if (run_stride_write)
    {
      fprintf(output_file, " StrideWrite(GB/s) StrideWrite(ms)");
    }
    if (run_zipfian_read)
    {
      fprintf(output_file, " ZipfianRead(GB/s) ZipfianRead(ms)");
    }
    if (run_pointer_chase)
    {
      fprintf(output_file, " PointerChase(GB/s) PointerChase(ms)");
    }
    fprintf(output_file, "\n");
  }

  printf("\n========================================\n");
  printf("Testing with %d threads\n", num_threads);
  printf("========================================\n");

  size_t memory_to_use =
      (size_t)num_threads * memory_per_thread_mib * 1024 * 1024;

  double seq_read_bw = 0, seq_write_bw = 0, rand_read_bw = 0, rand_write_bw = 0;
  double seq_read_ms = 0, seq_write_ms = 0, rand_read_ms = 0, rand_write_ms = 0;
  double stride_read_bw = 0, stride_write_bw = 0;
  double stride_read_ms = 0, stride_write_ms = 0;
  double zipfian_read_bw = 0;
  double zipfian_read_ms = 0;
  double pointer_chase_bw = 0;
  double pointer_chase_ms = 0;

  if (run_seq_read || run_seq_write)
  {
    printf("[Sequential Tests]\n");
  }

  if (run_seq_read)
  {
    printf("Running Sequential Read test...\n");
#ifdef ENABLE_TRACING
    clearTraceBuffers();
#endif
#ifdef ENABLE_LATENCY_MEASURE
    clearLatencyBuffers();
#endif
    auto seq_read_result =
        measureSequentialRead(data, memory_to_use, num_threads, bypass_cache);
    seq_read_bw = seq_read_result.bandwidth_gbps;
    seq_read_ms = seq_read_result.elapsed_ms;
    printf("Read Bandwidth:  %.2f GB/s (%.2f ms)\n",
           seq_read_result.bandwidth_gbps, seq_read_result.elapsed_ms);
    printDetail(seq_read_result);
#ifdef ENABLE_TRACING
    char test_dir[1024];
    snprintf(test_dir, sizeof(test_dir), "%s/seq_read", tracing_dir);
    mkdir(test_dir, 0755);
    saveMetadata(test_dir, data, memory_to_use, num_threads);
    saveTrace(test_dir, (uintptr_t)data);
#endif
#ifdef ENABLE_LATENCY_MEASURE
    {
      char latency_file[1024];
      snprintf(latency_file, sizeof(latency_file), "%s/seq_read_latency.log",
               result_dir);
      double cpu_freq_ghz = getCpuFrequencyGHz();
      saveLatencyLog(latency_file, cpu_freq_ghz);
    }
#endif
  }

  if (run_seq_write)
  {
    printf("Running Sequential Write test...\n");
#ifdef ENABLE_TRACING
    clearTraceBuffers();
#endif
    auto seq_write_result =
        measureSequentialWrite(data, memory_to_use, num_threads, bypass_cache);
    seq_write_bw = seq_write_result.bandwidth_gbps;
    seq_write_ms = seq_write_result.elapsed_ms;
    printf("Write Bandwidth: %.2f GB/s (%.2f ms)\n",
           seq_write_result.bandwidth_gbps, seq_write_result.elapsed_ms);
    printDetail(seq_write_result);
#ifdef ENABLE_TRACING
    char test_dir[1024];
    snprintf(test_dir, sizeof(test_dir), "%s/seq_write", tracing_dir);
    mkdir(test_dir, 0755);
    saveMetadata(test_dir, data, memory_to_use, num_threads);
    saveTrace(test_dir, (uintptr_t)data);
#endif
  }

  if (run_seq_read || run_seq_write)
  {
    printf("\n");
  }

  if (run_random_read || run_random_write)
  {
    printf("[Random Tests (%zu bytes blocks)]\n", BLOCK_SIZE);
  }

  if (run_random_read)
  {
    printf("Running Random Read test...\n");
#ifdef ENABLE_TRACING
    clearTraceBuffers();
#endif
#ifdef ENABLE_LATENCY_MEASURE
    clearLatencyBuffers();
#endif
    auto rand_read_result = measureRandomRead(data, memory_to_use, num_threads, bypass_cache);
    rand_read_bw = rand_read_result.bandwidth_gbps;
    rand_read_ms = rand_read_result.elapsed_ms;
    printf("Read Bandwidth:  %.2f GB/s (%.2f ms)\n",
           rand_read_result.bandwidth_gbps, rand_read_result.elapsed_ms);
    printDetail(rand_read_result);
#ifdef ENABLE_TRACING
    char test_dir[1024];
    snprintf(test_dir, sizeof(test_dir), "%s/random_read", tracing_dir);
    mkdir(test_dir, 0755);
    saveMetadata(test_dir, data, memory_to_use, num_threads);
    saveTrace(test_dir, (uintptr_t)data);
#endif
#ifdef ENABLE_LATENCY_MEASURE
    {
      char latency_file[1024];
      snprintf(latency_file, sizeof(latency_file), "%s/random_read_latency.log",
               result_dir);
      double cpu_freq_ghz = getCpuFrequencyGHz();
      saveLatencyLog(latency_file, cpu_freq_ghz);
    }
#endif
  }

  if (run_random_write)
  {
    printf("Running Random Write test...\n");
#ifdef ENABLE_TRACING
    clearTraceBuffers();
#endif
    auto rand_write_result =
        measureRandomWrite(data, memory_to_use, num_threads, bypass_cache);
    rand_write_bw = rand_write_result.bandwidth_gbps;
    rand_write_ms = rand_write_result.elapsed_ms;
    printf("Write Bandwidth: %.2f GB/s (%.2f ms)\n",
           rand_write_result.bandwidth_gbps, rand_write_result.elapsed_ms);
    printDetail(rand_write_result);
#ifdef ENABLE_TRACING
    char test_dir[1024];
    snprintf(test_dir, sizeof(test_dir), "%s/random_write", tracing_dir);
    mkdir(test_dir, 0755);
    saveMetadata(test_dir, data, memory_to_use, num_threads);
    saveTrace(test_dir, (uintptr_t)data);
#endif
  }

  if (run_random_read || run_random_write)
  {
    printf("\n");
  }

  if (run_zipfian_read)
  {
    printf("[Zipfian Tests (%zu bytes blocks, alpha=%.2f)]\n", BLOCK_SIZE, zipfian_alpha);
    printf("Running Zipfian Read test...\n");
#ifdef ENABLE_TRACING
    clearTraceBuffers();
#endif
#ifdef ENABLE_LATENCY_MEASURE
    clearLatencyBuffers();
#endif
    auto zipfian_read_result = measureZipfianRead(data, memory_to_use, num_threads, zipfian_alpha, bypass_cache);
    zipfian_read_bw = zipfian_read_result.bandwidth_gbps;
    zipfian_read_ms = zipfian_read_result.elapsed_ms;
    printf("Read Bandwidth:  %.2f GB/s (%.2f ms)\n",
           zipfian_read_result.bandwidth_gbps, zipfian_read_result.elapsed_ms);
    printDetail(zipfian_read_result);
#ifdef ENABLE_TRACING
    char test_dir[1024];
    snprintf(test_dir, sizeof(test_dir), "%s/zipfian_read", tracing_dir);
    mkdir(test_dir, 0755);
    saveMetadata(test_dir, data, memory_to_use, num_threads);
    saveTrace(test_dir, (uintptr_t)data);
#endif
#ifdef ENABLE_LATENCY_MEASURE
    {
      char latency_file[1024];
      snprintf(latency_file, sizeof(latency_file), "%s/zipfian_read_latency.log",
               result_dir);
      double cpu_freq_ghz = getCpuFrequencyGHz();
      saveLatencyLog(latency_file, cpu_freq_ghz);
    }
#endif
    printf("\n");
  }

  if (run_stride_read || run_stride_write)
  {
    printf("[Stride Tests (stride size: %zu bytes, block size: %zu bytes)]\n",
           STRIDE_SIZE, BLOCK_SIZE);
  }

  if (run_stride_read)
  {
    printf("Running Stride Read test...\n");
#ifdef ENABLE_TRACING
    clearTraceBuffers();
#endif
    auto stride_read_result =
        measureStrideRead(data, memory_to_use, num_threads, STRIDE_SIZE, bypass_cache);
    stride_read_bw = stride_read_result.bandwidth_gbps;
    stride_read_ms = stride_read_result.elapsed_ms;
    printf("Read Bandwidth:  %.2f GB/s (%.2f ms)\n",
           stride_read_result.bandwidth_gbps, stride_read_result.elapsed_ms);
    printDetail(stride_read_result);
#ifdef ENABLE_TRACING
    char test_dir[1024];
    snprintf(test_dir, sizeof(test_dir), "%s/stride_read", tracing_dir);
    mkdir(test_dir, 0755);
    saveMetadata(test_dir, data, memory_to_use, num_threads);
    saveTrace(test_dir, (uintptr_t)data);
#endif
  }

  if (run_stride_write)
  {
    printf("Running Stride Write test...\n");
#ifdef ENABLE_TRACING
    clearTraceBuffers();
#endif
    auto stride_write_result = measureStrideWrite(
        data, memory_to_use, num_threads, STRIDE_SIZE, bypass_cache);
    stride_write_bw = stride_write_result.bandwidth_gbps;
    stride_write_ms = stride_write_result.elapsed_ms;
    printf("Write Bandwidth: %.2f GB/s (%.2f ms)\n",
           stride_write_result.bandwidth_gbps, stride_write_result.elapsed_ms);
    printDetail(stride_write_result);
#ifdef ENABLE_TRACING
    char test_dir[1024];
    snprintf(test_dir, sizeof(test_dir), "%s/stride_write", tracing_dir);
    mkdir(test_dir, 0755);
    saveMetadata(test_dir, data, memory_to_use, num_threads);
    saveTrace(test_dir, (uintptr_t)data);
#endif
  }

  printf("Waiting for 1.5 second after testing...\n");
  sleep(2.0);

  if (run_stride_read || run_stride_write)
  {
    printf("\n");
  }

  if (run_pointer_chase)
  {
    printf("[Pointer Chase with Load (inject delay: %lu cycles)]\n",
           inject_delay_cycles);
    printf("Running Pointer Chase test...\n");
#ifdef ENABLE_LATENCY_MEASURE
    clearLatencyBuffers();
#endif
    auto pointer_chase_result = measurePointerChaseWithLoad(
        data, memory_to_use, num_threads, inject_delay_cycles, membind_node);
    pointer_chase_bw = pointer_chase_result.bandwidth_gbps;
    pointer_chase_ms = pointer_chase_result.elapsed_ms;
    printf("Load Bandwidth: %.2f GB/s (%.2f ms)\n",
           pointer_chase_result.bandwidth_gbps,
           pointer_chase_result.elapsed_ms);
    printDetail(pointer_chase_result);
#ifdef ENABLE_LATENCY_MEASURE
    {
      char latency_file[1024];
      snprintf(latency_file, sizeof(latency_file),
               "%s/pointer_chase_latency.log", result_dir);
      double cpu_freq_ghz = getCpuFrequencyGHz();
      saveLatencyLog(latency_file, cpu_freq_ghz);
    }
#endif
    printf("\n");
  }

  fprintf(output_file, "%d", num_threads);
  if (run_seq_read)
  {
    fprintf(output_file, " %.2f %.2f", seq_read_bw, seq_read_ms);
  }
  if (run_seq_write)
  {
    fprintf(output_file, " %.2f %.2f", seq_write_bw, seq_write_ms);
  }
  if (run_random_read)
  {
    fprintf(output_file, " %.2f %.2f", rand_read_bw, rand_read_ms);
  }
  if (run_random_write)
  {
    fprintf(output_file, " %.2f %.2f", rand_write_bw, rand_write_ms);
  }
  if (run_stride_read)
  {
    fprintf(output_file, " %.2f %.2f", stride_read_bw, stride_read_ms);
  }
  if (run_stride_write)
  {
    fprintf(output_file, " %.2f %.2f", stride_write_bw, stride_write_ms);
  }
  if (run_zipfian_read)
  {
    fprintf(output_file, " %.2f %.2f", zipfian_read_bw, zipfian_read_ms);
  }
  if (run_pointer_chase)
  {
    fprintf(output_file, " %.2f %.2f", pointer_chase_bw, pointer_chase_ms);
  }
  fprintf(output_file, "\n");
  fflush(output_file);

  fclose(output_file);
  printf("\n========================================\n");
  printf("All tests completed!\n");
  printf("Results saved to: %s\n", output_file_path_buf);
  printf("========================================\n\n");

  if (!result_dir_path)
  {
    const char *dir_name = strrchr(result_dir, '/');
    if (dir_name)
    {
      dir_name++;
      char symlink_cmd[1024];
      snprintf(symlink_cmd, sizeof(symlink_cmd),
               "ln -sfn %s result/latest_result", dir_name);
      int symlink_ret = system(symlink_cmd);
      (void)symlink_ret;
      printf("Updated result/latest_result -> %s\n", dir_name);
    }
  }

  if (using_devdax)
  {
    munmap(data, memory_size);
    close(devdax_fd);
    printf("Memory unmapped from devdax device\n");
  }
  else if (membind_node >= 0)
  {
    numa_free(data, memory_size);
    printf("Memory freed (numa_free from node %d)\n", membind_node);
  }
  else
  {
    free(data);
    printf("Memory freed\n");
  }

  restore_prefetcher_state();
  printf("Hardware prefetcher state restored\n");

  printf("\nTest completed successfully!\n");

  return 0;
}
