#include "benchmark.h"

#include <immintrin.h>
#include <numa.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <random>
#include <thread>
#include <vector>

#include "numa_affinity.h"

#include <pxl/memory.hpp>

#ifdef ENABLE_TRACING
#include "tracing.h"
#endif

#ifdef ENABLE_LATENCY_MEASURE
#include "latency_measure.h"
#endif

using namespace std;
using namespace chrono;

// Issue an explicit PXL device prefetch for [addr, addr+size).
static inline void devicePrefetch(void *addr, size_t size)
{
  pxl::prefetchMemory(addr, size);
}

// rdtscp inline function for delay injection (always available)
#ifndef ENABLE_LATENCY_MEASURE
static inline uint64_t rdtscp()
{
  uint32_t aux;
  uint64_t rax, rdx;
  asm volatile("rdtscp" : "=a"(rax), "=d"(rdx), "=c"(aux));
  return (rdx << 32) | rax;
}
#endif

// Global configuration variables
size_t BLOCK_SIZE = 64;  // Default: 64 bytes
size_t STRIDE_SIZE = 64; // Default: 64 bytes

// Progress update threshold (bytes accumulated per worker before publishing
// to the shared atomic). Byte-based instead of block-count based so the
// progress bar updates smoothly regardless of BLOCK_SIZE — with large
// BLOCK_SIZE (e.g. 32 MiB) a count-based threshold could exceed the total
// blocks per thread and leave the bar stuck at 0% until threads finish.
#define PROGRESS_UPDATE_BYTES (256ULL * 1024 * 1024)

// Global CPU affinity list
std::vector<int> g_cpu_affinity_list;

// Live progress reporting toggle (see benchmark.h for rationale).
bool g_progress_enabled = true;

// Progress monitor: main thread polls the shared counter on a fixed sleep
// cadence (api_test pattern). poll_ms bounds the worst-case padding between
// actual worker completion and the polling exit (≤ poll_ms). print cadence
// is throttled separately so screen output isn't too noisy when poll_ms is
// short. When g_progress_enabled is false, wait_until_done is a no-op and
// the caller's subsequent join() bounds elapsed_ms within join overhead.
class ProgressMonitor
{
public:
  ProgressMonitor(size_t total_bytes, const char *operation_name)
      : total_bytes_(total_bytes),
        bytes_processed_(0),
        name_(operation_name) {}

  // Block on the main thread until workers report >= total_bytes via
  // add_bytes(). Polls every poll_ms (≤ poll_ms padding on completion);
  // prints at most every print_ms.
  void wait_until_done(int print_ms = 1000, int poll_ms = 100)
  {
    if (!g_progress_enabled)
    {
      return;
    }
    printf("\n");
    auto last_print = steady_clock::now() - chrono::milliseconds(print_ms);
    while (bytes_processed_.load(memory_order_relaxed) < total_bytes_)
    {
      auto now = steady_clock::now();
      if (now - last_print >= chrono::milliseconds(print_ms))
      {
        print_progress();
        last_print = now;
      }
      this_thread::sleep_for(chrono::milliseconds(poll_ms));
    }
    print_progress();
  }

  void add_bytes(size_t bytes)
  {
    if (!g_progress_enabled) return;
    bytes_processed_.fetch_add(bytes, memory_order_relaxed);
  }

  atomic<size_t> &get_counter() { return bytes_processed_; }

private:
  void print_progress()
  {
    // Get current time
    auto now = system_clock::now();
    time_t now_time = system_clock::to_time_t(now);
    struct tm *tm_info = localtime(&now_time);
    char time_buf[16];
    strftime(time_buf, sizeof(time_buf), "%H:%M:%S", tm_info);

    size_t processed = bytes_processed_.load();
    double progress = (double)processed / total_bytes_ * 100.0;
    if (progress > 100.0)
      progress = 100.0;
    double processed_gib = (double)processed / (1024.0 * 1024.0 * 1024.0);
    double total_gib = (double)total_bytes_ / (1024.0 * 1024.0 * 1024.0);
    printf("  [%s] [%s] %.1f%% (%.2f / %.2f GiB)\n", time_buf, name_, progress,
           processed_gib, total_gib);
    fflush(stdout);
  }

  size_t total_bytes_;
  atomic<size_t> bytes_processed_;
  const char *name_;
};

void flushHostCache(void *hostVirtualPtr, size_t size)
{
  const uint64_t CacheLineSize = 64;
  uint64_t hostVirtualPtrStart = reinterpret_cast<uint64_t>(hostVirtualPtr);
  uint64_t hostVirtualPtrEnd = hostVirtualPtrStart + size - 1;

  hostVirtualPtrStart = hostVirtualPtrStart & ~(CacheLineSize - 1);
  hostVirtualPtrEnd = (hostVirtualPtrEnd | (CacheLineSize - 1)) + 1;

  for (uint64_t address = hostVirtualPtrStart; address < hostVirtualPtrEnd;
       address += CacheLineSize)
  {
    _mm_clflushopt(reinterpret_cast<void *>(address));
  }
  _mm_sfence();
}

// Sequential Read Bandwidth Test
BandwidthResult measureSequentialRead(void *data, size_t size,
                                      int num_threads, bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  ProgressMonitor progress(size, "SeqRead");

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
                         {
                           // Set CPU affinity if enabled
                           if (!g_cpu_affinity_list.empty())
                           {
                             int cpu_id = g_cpu_affinity_list[t % g_cpu_affinity_list.size()];
                             setThreadCpuAffinity(cpu_id);
                           }

                           size_t per_thread_size = size / num_threads;
                           int32_t *thread_data =
                               reinterpret_cast<int32_t *>((char *)data + t * per_thread_size);
                           size_t per_thread_ints = per_thread_size / sizeof(int32_t);

#ifdef ENABLE_TRACING
                           reserveTraceBuffer(num_threads);
#endif
#ifdef ENABLE_LATENCY_MEASURE
                           // Reserve latency buffer with 2x expected samples to avoid reallocation
                           size_t expected_samples =
                               (per_thread_ints / 8 / LATENCY_SAMPLE_INTERVAL) * 2;
                           reserveLatencyBuffer(expected_samples);
#endif

                           // Flush host CPU cache (skip when bypass_cache - non-temporal loads bypass cache)
                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           // Signal that this thread is ready
                           ready_count.fetch_add(1);

                           // Wait for all threads to be ready
                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           // Sequential read with BLOCK_SIZE outer loop + 32B AVX2 inner loop.
                           // Progress hook lives on the outer loop so atomic update runs once per
                           // block (avoids cache-line ping-pong on `progress.add_bytes()` atomic).
                           __m256i acc = _mm256_setzero_si256();
                           size_t local_bytes = 0;
                           const size_t block_ints = BLOCK_SIZE / sizeof(int32_t);
                           const size_t num_full_blocks = per_thread_ints / block_ints;
#ifdef ENABLE_LATENCY_MEASURE
                           // For sequential, measure every LATENCY_SAMPLE_INTERVAL cache lines (64
                           // bytes = 2 AVX loads) This ensures we measure cold cache line access
                           size_t cache_line_count = 0;
                           const size_t AVX_LOADS_PER_CACHELINE = 2; // 64B cache line / 32B AVX = 2
#endif

                           for (size_t blk = 0; blk < num_full_blocks; blk++)
                           {
                             const size_t base_ints = blk * block_ints;
                             for (size_t j = 0; j + 8 <= block_ints; j += 8)
                             {
                               size_t i = base_ints + j;
#ifdef ENABLE_TRACING
                               uint64_t ts = rdtsc();
#endif
#ifdef ENABLE_LATENCY_MEASURE
                               if (cache_line_count % LATENCY_SAMPLE_INTERVAL == 0 &&
                                   (j / 8) % AVX_LOADS_PER_CACHELINE == 0)
                               {
                                 uint64_t start = rdtscp();
                                 __m256i data_vec = bypass_cache
                                                        ? _mm256_stream_load_si256(
                                                              reinterpret_cast<__m256i *>(&thread_data[i]))
                                                        : _mm256_loadu_si256(
                                                              reinterpret_cast<const __m256i *>(&thread_data[i]));
                                 uint64_t end = rdtscp();
                                 latency_buffer.push_back(end - start);
                                 acc = _mm256_add_epi32(acc, data_vec);
                               }
                               else
                               {
                                 __m256i data_vec = bypass_cache
                                                        ? _mm256_stream_load_si256(
                                                              reinterpret_cast<__m256i *>(&thread_data[i]))
                                                        : _mm256_loadu_si256(
                                                              reinterpret_cast<const __m256i *>(&thread_data[i]));
                                 acc = _mm256_add_epi32(acc, data_vec);
                               }
                               if ((j / 8) % AVX_LOADS_PER_CACHELINE == 1)
                               {
                                 cache_line_count++;
                               }
#else
                               __m256i data_vec = bypass_cache
                                                      ? _mm256_stream_load_si256(
                                                            reinterpret_cast<__m256i *>(&thread_data[i]))
                                                      : _mm256_loadu_si256(
                                                            reinterpret_cast<const __m256i *>(&thread_data[i]));
                               acc = _mm256_add_epi32(acc, data_vec);
#endif
#ifdef ENABLE_TRACING
                               trace_buffer.push_back({ts, (uintptr_t)&thread_data[i], 32, 0});
#endif
                             }

                             local_bytes += BLOCK_SIZE;
                             if (local_bytes >= PROGRESS_UPDATE_BYTES)
                             {
                               progress.add_bytes(local_bytes);
                               local_bytes = 0;
                             }
                           }

                           // Drain accumulator
                           volatile long long sum = 0;
                           int32_t temp[8];
                           _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc);
                           for (int j = 0; j < 8; j++)
                             sum += temp[j];

                           // Leftover (last partial block: scalar fallback)
                           size_t i = num_full_blocks * block_ints;
                           for (; i < per_thread_ints; i++)
                           {
#ifdef ENABLE_TRACING
                             uint64_t ts = rdtsc();
#endif
                             sum += thread_data[i];
#ifdef ENABLE_TRACING
                             trace_buffer.push_back({ts, (uintptr_t)&thread_data[i], 4, 0});
#endif
                             local_bytes += 4;
                           }
                           progress.add_bytes(local_bytes);

                           auto thread_end = steady_clock::now();
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
#ifdef ENABLE_LATENCY_MEASURE
                           collectLatencyBuffers();
#endif
                         });
  }

  // Wait for all threads to finish setup
  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  // Start all threads simultaneously
  start_flag.store(true);

  // Drive the progress monitor on this (main) thread — no separate monitor
  // thread is spawned, so worker count == active thread count and there is
  // no CPU-share bias affecting low-thread-count measurements.
  progress.wait_until_done();

  // Workers are essentially done by the time wait_until_done() returns.
  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = steady_clock::now();
  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  size_t per_thread_size = size / num_threads;
  for (uint64_t t = 0; t < num_threads; t++)
  {
    double thread_bandwidth_gbps =
        (per_thread_size / (1000.0 * 1000.0 * 1000.0)) / (thread_times[t] / 1000.0);
    printf("Thread %lu time: %f ms, bandwidth: %f GB/s\n", t, thread_times[t], thread_bandwidth_gbps);
  }

  return {bandwidth_gbps, elapsed_ms};
}

// Sequential Write Bandwidth Test
BandwidthResult measureSequentialWrite(void *data, size_t size, int num_threads,
                                       bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  ProgressMonitor progress(size, "SeqWrite");

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
                         {
                           // Set CPU affinity if enabled
                           if (!g_cpu_affinity_list.empty())
                           {
                             int cpu_id = g_cpu_affinity_list[t % g_cpu_affinity_list.size()];
                             setThreadCpuAffinity(cpu_id);
                           }

                           size_t per_thread_size = size / num_threads;
                           int32_t *thread_data =
                               reinterpret_cast<int32_t *>((char *)data + t * per_thread_size);
                           size_t per_thread_ints = per_thread_size / sizeof(int32_t);

#ifdef ENABLE_TRACING
                           reserveTraceBuffer(num_threads);
#endif

                           // Flush and warmup this thread's memory region (skip when bypass_cache)
                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           // Signal that this thread is ready
                           ready_count.fetch_add(1);

                           // Wait for all threads to be ready
                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           // Sequential write with BLOCK_SIZE outer loop + 32B AVX2 inner loop.
                           // Progress hook lives on the outer loop so atomic update runs once per
                           // block (avoids cache-line ping-pong on `bytes_processed_`).
                           size_t local_bytes = 0;
                           const size_t block_ints = BLOCK_SIZE / sizeof(int32_t);
                           const size_t num_full_blocks = per_thread_ints / block_ints;

                           for (size_t blk = 0; blk < num_full_blocks; blk++)
                           {
                             const size_t base_ints = blk * block_ints;
                             for (size_t j = 0; j + 8 <= block_ints; j += 8)
                             {
                               size_t i = base_ints + j;
#ifdef ENABLE_TRACING
                               uint64_t ts = rdtsc();
#endif
                               __m256i data_vec = _mm256_set_epi32(i + 7, i + 6, i + 5, i + 4,
                                                                   i + 3, i + 2, i + 1, i);
                               if (bypass_cache)
                               {
                                 _mm256_stream_si256(reinterpret_cast<__m256i *>(&thread_data[i]),
                                                     data_vec);
                               }
                               else
                               {
                                 _mm256_storeu_si256(reinterpret_cast<__m256i *>(&thread_data[i]),
                                                     data_vec);
                               }
#ifdef ENABLE_TRACING
                               trace_buffer.push_back({ts, (uintptr_t)&thread_data[i], 32, 1});
#endif
                             }

                             local_bytes += BLOCK_SIZE;
                             if (local_bytes >= PROGRESS_UPDATE_BYTES)
                             {
                               progress.add_bytes(local_bytes);
                               local_bytes = 0;
                             }
                           }

                           // Ensure all non-temporal stores complete
                           if (bypass_cache)
                           {
                             _mm_sfence();
                           }

                           // Leftover (last partial block: scalar fallback)
                           size_t i = num_full_blocks * block_ints;
                           for (; i < per_thread_ints; i++)
                           {
#ifdef ENABLE_TRACING
                             uint64_t ts = rdtsc();
#endif
                             thread_data[i] = static_cast<int32_t>(i);
#ifdef ENABLE_TRACING
                             trace_buffer.push_back({ts, (uintptr_t)&thread_data[i], 4, 1});
#endif
                             local_bytes += 4;
                           }
                           progress.add_bytes(local_bytes);

                           auto thread_end = steady_clock::now();
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
                           flushHostCache(thread_data, per_thread_size);
#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
                         });
  }

  // Wait for all threads to finish setup
  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  // Start all threads simultaneously
  start_flag.store(true);

  // Drive the progress monitor on this (main) thread — no separate monitor
  // thread is spawned, so worker count == active thread count and there is
  // no CPU-share bias affecting low-thread-count measurements.
  progress.wait_until_done();

  // Workers are essentially done by the time wait_until_done() returns.
  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = steady_clock::now();
  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  return {bandwidth_gbps, elapsed_ms};
}

// Zipfian CDF builder: cdf[k] = Σ(1/i^alpha) for i=1..k+1, normalized to [0,1]
static void buildZipfianCDF(vector<double> &cdf, size_t n, double alpha)
{
  cdf.resize(n);
  double sum = 0;
  for (size_t i = 0; i < n; i++)
  {
    sum += 1.0 / pow((double)(i + 1), alpha);
    cdf[i] = sum;
  }
  for (size_t i = 0; i < n; i++)
  {
    cdf[i] /= sum;
  }
}

// Uniform random [0,1) → Zipfian rank (0-indexed) via binary search
static size_t sampleZipfian(const vector<double> &cdf, double u)
{
  return lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin();
}

// Random Read Bandwidth Test
BandwidthResult measureRandomRead(void *data, size_t size, int num_threads,
                                  bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  ProgressMonitor progress(size, "RandRead");

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
                         {
                           // Set CPU affinity if enabled
                           if (!g_cpu_affinity_list.empty())
                           {
                             int cpu_id = g_cpu_affinity_list[t % g_cpu_affinity_list.size()];
                             setThreadCpuAffinity(cpu_id);
                           }

                           size_t per_thread_size = size / num_threads;
                           int32_t *thread_data =
                               reinterpret_cast<int32_t *>((char *)data + t * per_thread_size);
                           size_t num_blocks = per_thread_size / BLOCK_SIZE;

#ifdef ENABLE_TRACING
                           reserveTraceBuffer(num_threads);
#endif
#ifdef ENABLE_LATENCY_MEASURE
                           // Reserve latency buffer with 2x expected samples to avoid reallocation
                           size_t expected_samples =
                               (num_blocks * (BLOCK_SIZE / 32) / LATENCY_SAMPLE_INTERVAL) * 2;
                           reserveLatencyBuffer(expected_samples);
#endif

                           // Flush host CPU cache (skip when bypass_cache - non-temporal loads bypass cache)
                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           // Generate random indices
                           vector<size_t> indices(num_blocks);
                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             indices[i] = i;
                           }

                           // Shuffle indices for random access
                           random_device rd;
                           mt19937 gen(rd() + t);
                           shuffle(indices.begin(), indices.end(), gen);

                           // Signal that this thread is ready
                           ready_count.fetch_add(1);

                           // Wait for all threads to be ready
                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           // Random read with per-block outer loop + 32B AVX2 inner loop.
                           // Per-block progress hook avoids cache-line ping-pong on the atomic.
                           __m256i acc = _mm256_setzero_si256();
                           size_t local_bytes = 0;
                           const size_t block_ints = BLOCK_SIZE / sizeof(int32_t);
#ifdef ENABLE_LATENCY_MEASURE
                           size_t block_count = 0;
#endif

                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             const size_t offset_ints = (indices[i] * BLOCK_SIZE) / sizeof(int32_t);
                             size_t j = 0;
#ifdef ENABLE_LATENCY_MEASURE
                             // Measure latency only on first access of each sampled block (cold
                             // cache line)
                             if (block_count % LATENCY_SAMPLE_INTERVAL == 0 && j + 8 <= block_ints)
                             {
                               uint64_t start = rdtscp();
                               __m256i data_vec = bypass_cache
                                                      ? _mm256_stream_load_si256(
                                                            reinterpret_cast<__m256i *>(&thread_data[offset_ints + j]))
                                                      : _mm256_loadu_si256(
                                                            reinterpret_cast<const __m256i *>(&thread_data[offset_ints + j]));
                               uint64_t end = rdtscp();
                               latency_buffer.push_back(end - start);
                               acc = _mm256_add_epi32(acc, data_vec);
                               j += 8;
                             }
                             block_count++;
#endif
                             for (; j + 8 <= block_ints; j += 8)
                             {
#ifdef ENABLE_TRACING
                               uint64_t ts = rdtsc();
#endif
                               __m256i data_vec = bypass_cache
                                                      ? _mm256_stream_load_si256(
                                                            reinterpret_cast<__m256i *>(&thread_data[offset_ints + j]))
                                                      : _mm256_loadu_si256(
                                                            reinterpret_cast<const __m256i *>(&thread_data[offset_ints + j]));
                               acc = _mm256_add_epi32(acc, data_vec);
#ifdef ENABLE_TRACING
                               trace_buffer.push_back(
                                   {ts, (uintptr_t)&thread_data[offset_ints + j], 32, 0});
#endif
                             }
                             local_bytes += BLOCK_SIZE;
                             if (local_bytes >= PROGRESS_UPDATE_BYTES)
                             {
                               progress.add_bytes(local_bytes);
                               local_bytes = 0;
                             }
                           }
                           progress.add_bytes(local_bytes);

                           // Store accumulator to prevent optimization
                           volatile long long sum = 0;
                           int32_t temp[8];
                           _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc);
                           for (int j = 0; j < 8; j++)
                             sum += temp[j];

                           auto thread_end = steady_clock::now();
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
#ifdef ENABLE_LATENCY_MEASURE
                           collectLatencyBuffers();
#endif
                         });
  }

  // Wait for all threads to finish setup (including shuffling)
  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  // Start all threads simultaneously
  start_flag.store(true);

  // Drive the progress monitor on this (main) thread — no separate monitor
  // thread is spawned, so worker count == active thread count and there is
  // no CPU-share bias affecting low-thread-count measurements.
  progress.wait_until_done();

  // Workers are essentially done by the time wait_until_done() returns.
  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = steady_clock::now();
  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  return {bandwidth_gbps, elapsed_ms};
}

// Random Write Bandwidth Test
BandwidthResult measureRandomWrite(void *data, size_t size, int num_threads,
                                   bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  ProgressMonitor progress(size, "RandWrite");

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
                         {
                           // Set CPU affinity if enabled
                           if (!g_cpu_affinity_list.empty())
                           {
                             int cpu_id = g_cpu_affinity_list[t % g_cpu_affinity_list.size()];
                             setThreadCpuAffinity(cpu_id);
                           }

                           size_t per_thread_size = size / num_threads;
                           int32_t *thread_data =
                               reinterpret_cast<int32_t *>((char *)data + t * per_thread_size);
                           size_t num_blocks = per_thread_size / BLOCK_SIZE;

#ifdef ENABLE_TRACING
                           reserveTraceBuffer(num_threads);
#endif

                           // Flush and warmup this thread's memory region (skip when bypass_cache)
                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           // Generate random indices
                           vector<size_t> indices(num_blocks);
                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             indices[i] = i;
                           }

                           // Shuffle indices for random access
                           random_device rd;
                           mt19937 gen(rd() + t);
                           shuffle(indices.begin(), indices.end(), gen);

                           // Signal that this thread is ready
                           ready_count.fetch_add(1);

                           // Wait for all threads to be ready
                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           // Random write with per-block outer loop + 32B AVX2 inner loop.
                           // Same structure as the read counterpart; progress hook fires once
                           // per block to avoid cache-line ping-pong on the atomic counter.
                           size_t local_bytes = 0;
                           const size_t block_ints = BLOCK_SIZE / sizeof(int32_t);

                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             const size_t offset_ints = (indices[i] * BLOCK_SIZE) / sizeof(int32_t);

                             // Write entire block using AVX2
                             __m256i value = _mm256_set1_epi32(static_cast<int32_t>(i));
                             for (size_t j = 0; j + 8 <= block_ints; j += 8)
                             {
#ifdef ENABLE_TRACING
                               uint64_t ts = rdtsc();
#endif
                               if (bypass_cache)
                               {
                                 _mm256_stream_si256(
                                     reinterpret_cast<__m256i *>(&thread_data[offset_ints + j]),
                                     value);
                               }
                               else
                               {
                                 _mm256_storeu_si256(
                                     reinterpret_cast<__m256i *>(&thread_data[offset_ints + j]),
                                     value);
                               }
#ifdef ENABLE_TRACING
                               trace_buffer.push_back(
                                   {ts, (uintptr_t)&thread_data[offset_ints + j], 32, 1});
#endif
                             }
                             local_bytes += BLOCK_SIZE;
                             if (local_bytes >= PROGRESS_UPDATE_BYTES)
                             {
                               progress.add_bytes(local_bytes);
                               local_bytes = 0;
                             }
                           }
                           progress.add_bytes(local_bytes);

                           // Ensure all non-temporal stores complete
                           if (bypass_cache)
                           {
                             _mm_sfence();
                           }

                           auto thread_end = steady_clock::now();
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
                           flushHostCache(thread_data, per_thread_size);

#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
                         });
  }

  // Wait for all threads to finish setup (including shuffling)
  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  // Start all threads simultaneously
  start_flag.store(true);

  // Drive the progress monitor on this (main) thread — no separate monitor
  // thread is spawned, so worker count == active thread count and there is
  // no CPU-share bias affecting low-thread-count measurements.
  progress.wait_until_done();

  // Workers are essentially done by the time wait_until_done() returns.
  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = steady_clock::now();
  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  return {bandwidth_gbps, elapsed_ms};
}

// Zipfian Read Bandwidth Test
// Access pattern: Zipfian distribution over shuffled block indices
// Total accesses = num_blocks (same total data volume as random read)
BandwidthResult measureZipfianRead(void *data, size_t size, int num_threads,
                                   double zipfian_alpha, bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  ProgressMonitor progress(size, "ZipfRead");

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache, zipfian_alpha]()
                         {
                           // Set CPU affinity if enabled
                           if (!g_cpu_affinity_list.empty())
                           {
                             int cpu_id = g_cpu_affinity_list[t % g_cpu_affinity_list.size()];
                             setThreadCpuAffinity(cpu_id);
                           }

                           size_t per_thread_size = size / num_threads;
                           int32_t *thread_data =
                               reinterpret_cast<int32_t *>((char *)data + t * per_thread_size);
                           size_t num_blocks = per_thread_size / BLOCK_SIZE;

#ifdef ENABLE_TRACING
                           reserveTraceBuffer(num_threads);
#endif
#ifdef ENABLE_LATENCY_MEASURE
                           size_t expected_samples =
                               (num_blocks * (BLOCK_SIZE / 32) / LATENCY_SAMPLE_INTERVAL) * 2;
                           reserveLatencyBuffer(expected_samples);
#endif

                           // Flush host CPU cache (skip when bypass_cache)
                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           // Generate and shuffle block indices (same as random read)
                           // indices[0] = hottest block, indices[num_blocks-1] = coldest block
                           vector<size_t> indices(num_blocks);
                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             indices[i] = i;
                           }
                           random_device rd;
                           mt19937 gen(rd() + t);
                           shuffle(indices.begin(), indices.end(), gen);

                           // Build Zipfian CDF and generate access sequence
                           // Total accesses = num_blocks (guarantees same total read volume)
                           vector<double> cdf;
                           buildZipfianCDF(cdf, num_blocks, zipfian_alpha);

                           vector<size_t> access_sequence(num_blocks);
                           uniform_real_distribution<double> udist(0.0, 1.0);
                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             size_t rank = sampleZipfian(cdf, udist(gen));
                             access_sequence[i] = indices[rank];
                           }

                           // Signal that this thread is ready
                           ready_count.fetch_add(1);

                           // Wait for all threads to be ready
                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           // Zipfian read using AVX2
                           __m256i acc = _mm256_setzero_si256();
                           size_t local_bytes = 0;
#ifdef ENABLE_LATENCY_MEASURE
                           size_t block_count = 0;
#endif
                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             size_t offset_ints = (access_sequence[i] * BLOCK_SIZE) / sizeof(int32_t);
                             size_t block_ints = BLOCK_SIZE / sizeof(int32_t);
                             int32_t *block_ptr = &thread_data[offset_ints];

                             // Read entire block using AVX2
                             size_t j = 0;
#ifdef ENABLE_LATENCY_MEASURE
                             if (block_count % LATENCY_SAMPLE_INTERVAL == 0 && j + 8 <= block_ints)
                             {
                               uint64_t start = rdtscp();
                               __m256i data_vec = bypass_cache
                                                      ? _mm256_stream_load_si256(
                                                            reinterpret_cast<__m256i *>(&thread_data[offset_ints + j]))
                                                      : _mm256_loadu_si256(
                                                            reinterpret_cast<const __m256i *>(&thread_data[offset_ints + j]));
                               uint64_t end = rdtscp();
                               latency_buffer.push_back(end - start);
                               acc = _mm256_add_epi32(acc, data_vec);
                               j += 8;
                             }
                             block_count++;
#endif
                             for (; j + 8 <= block_ints; j += 8)
                             {
#ifdef ENABLE_TRACING
                               uint64_t ts = rdtsc();
#endif
                               __m256i data_vec = bypass_cache
                                                      ? _mm256_stream_load_si256(
                                                            reinterpret_cast<__m256i *>(&thread_data[offset_ints + j]))
                                                      : _mm256_loadu_si256(
                                                            reinterpret_cast<const __m256i *>(&thread_data[offset_ints + j]));
                               acc = _mm256_add_epi32(acc, data_vec);
#ifdef ENABLE_TRACING
                               trace_buffer.push_back(
                                   {ts, (uintptr_t)&thread_data[offset_ints + j], 32, 0});
#endif
                             }
                             if (!bypass_cache)
                             {
                               flushHostCache(block_ptr, BLOCK_SIZE);
                             }
                             local_bytes += BLOCK_SIZE;
                             if (local_bytes >= PROGRESS_UPDATE_BYTES)
                             {
                               progress.add_bytes(local_bytes);
                               local_bytes = 0;
                             }
                           }
                           progress.add_bytes(local_bytes);

                           // Store accumulator to prevent optimization
                           volatile long long sum = 0;
                           int32_t temp[8];
                           _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc);
                           for (int j = 0; j < 8; j++)
                             sum += temp[j];

                           auto thread_end = steady_clock::now();
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
#ifdef ENABLE_LATENCY_MEASURE
                           collectLatencyBuffers();
#endif
                         });
  }

  // Wait for all threads to finish setup
  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  // Start all threads simultaneously
  start_flag.store(true);

  // Drive the progress monitor on this (main) thread — no separate monitor
  // thread is spawned, so worker count == active thread count and there is
  // no CPU-share bias affecting low-thread-count measurements.
  progress.wait_until_done();

  // Workers are essentially done by the time wait_until_done() returns.
  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = steady_clock::now();
  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  return {bandwidth_gbps, elapsed_ms};
}

// Stride Read Bandwidth Test
// Access pattern: phase-based strided access to cover all memory
// Example: stride=64, block_size=64: Phase 0 reads 0-63, 128-191, 256-319, ...
//                                     Phase 64 reads 64-127, 192-255, 320-383,
//                                     ...
// Jump interval: stride + block_size
BandwidthResult measureStrideRead(void *data, size_t size, int num_threads,
                                  size_t stride, bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  ProgressMonitor progress(size, "StrideRead");

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
                         {
                           // Set CPU affinity if enabled
                           if (!g_cpu_affinity_list.empty())
                           {
                             int cpu_id = g_cpu_affinity_list[t % g_cpu_affinity_list.size()];
                             setThreadCpuAffinity(cpu_id);
                           }

                           size_t per_thread_size = size / num_threads;
                           char *thread_data = reinterpret_cast<char *>(data) + t * per_thread_size;

#ifdef ENABLE_TRACING
                           reserveTraceBuffer(num_threads);
#endif

                           // Flush host CPU cache (skip when bypass_cache - non-temporal loads bypass cache)
                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           // Signal that this thread is ready
                           ready_count.fetch_add(1);

                           // Wait for all threads to be ready
                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           // Stride read using AVX2 with phase-based access pattern
                           __m256i acc = _mm256_setzero_si256();
                           size_t local_bytes = 0;
                           size_t access_count = 0;

                           // Outer loop: iterate through phases (0, block_size, 2*block_size, ...,
                           // stride+block_size-block_size) Number of phases: (stride + block_size) /
                           // block_size
                           for (size_t phase = 0; phase < stride + BLOCK_SIZE; phase += BLOCK_SIZE)
                           {
                             // Inner loop: stride access starting from phase, jumping by (stride +
                             // block_size)
                             for (size_t offset = phase; offset < per_thread_size;
                                  offset += (stride + BLOCK_SIZE))
                             {
                               // Read block_size bytes at each stride position (32 bytes at a time)
                               for (size_t b = 0; b < BLOCK_SIZE; b += 32)
                               {
                                 if (offset + b + 32 <= per_thread_size)
                                 {
#ifdef ENABLE_TRACING
                                   uint64_t ts = rdtsc();
#endif
                                   __m256i data_vec = bypass_cache
                                                          ? _mm256_stream_load_si256(
                                                                reinterpret_cast<__m256i *>(thread_data + offset + b))
                                                          : _mm256_loadu_si256(
                                                                reinterpret_cast<const __m256i *>(thread_data + offset + b));
                                   acc = _mm256_add_epi32(acc, data_vec);
#ifdef ENABLE_TRACING
                                   trace_buffer.push_back(
                                       {ts, (uintptr_t)(thread_data + offset + b), 32, 0});
#endif
                                   local_bytes += 32;
                                   access_count++;
                                   if (local_bytes >= PROGRESS_UPDATE_BYTES)
                                   {
                                     progress.add_bytes(local_bytes);
                                     local_bytes = 0;
                                   }
                                 }
                               }
                             }
                           }
                           progress.add_bytes(local_bytes);

                           // Store accumulator to prevent optimization
                           volatile long long sum = 0;
                           int32_t temp[8];
                           _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc);
                           for (int j = 0; j < 8; j++)
                             sum += temp[j];

                           auto thread_end = steady_clock::now();
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
                         });
  }

  // Wait for all threads to finish setup
  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  // Start all threads simultaneously
  start_flag.store(true);

  // Drive the progress monitor on this (main) thread — no separate monitor
  // thread is spawned, so worker count == active thread count and there is
  // no CPU-share bias affecting low-thread-count measurements.
  progress.wait_until_done();

  // Workers are essentially done by the time wait_until_done() returns.
  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = steady_clock::now();
  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

  // Use total memory size for bandwidth calculation (all memory accessed once)
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  return {bandwidth_gbps, elapsed_ms};
}

// Stride Write Bandwidth Test
// Access pattern: phase-based strided access to cover all memory
// Example: stride=64, block_size=64: Phase 0 writes 0-63, 128-191, 256-319, ...
//                                     Phase 64 writes 64-127, 192-255, 320-383,
//                                     ...
// Jump interval: stride + block_size
BandwidthResult measureStrideWrite(void *data, size_t size, int num_threads,
                                   size_t stride, bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  ProgressMonitor progress(size, "StrideWrite");

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
                         {
                           // Set CPU affinity if enabled
                           if (!g_cpu_affinity_list.empty())
                           {
                             int cpu_id = g_cpu_affinity_list[t % g_cpu_affinity_list.size()];
                             setThreadCpuAffinity(cpu_id);
                           }

                           size_t per_thread_size = size / num_threads;
                           char *thread_data = reinterpret_cast<char *>(data) + t * per_thread_size;

#ifdef ENABLE_TRACING
                           reserveTraceBuffer(num_threads);
#endif

                           // Flush and warmup this thread's memory region (skip when bypass_cache)
                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           // Signal that this thread is ready
                           ready_count.fetch_add(1);

                           // Wait for all threads to be ready
                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           // Stride write using AVX2 with phase-based access pattern
                           __m256i value = _mm256_set1_epi32(t);
                           size_t local_bytes = 0;
                           size_t access_count = 0;

                           // Outer loop: iterate through phases (0, block_size, 2*block_size, ...,
                           // stride+block_size-block_size) Number of phases: (stride + block_size) /
                           // block_size
                           for (size_t phase = 0; phase < stride + BLOCK_SIZE; phase += BLOCK_SIZE)
                           {
                             // Inner loop: stride access starting from phase, jumping by (stride +
                             // block_size)
                             for (size_t offset = phase; offset < per_thread_size;
                                  offset += (stride + BLOCK_SIZE))
                             {
                               // Write block_size bytes at each stride position (32 bytes at a time)
                               for (size_t b = 0; b < BLOCK_SIZE; b += 32)
                               {
                                 if (offset + b + 32 <= per_thread_size)
                                 {
#ifdef ENABLE_TRACING
                                   uint64_t ts = rdtsc();
#endif
                                   if (bypass_cache)
                                   {
                                     _mm256_stream_si256(
                                         reinterpret_cast<__m256i *>(thread_data + offset + b),
                                         value);
                                   }
                                   else
                                   {
                                     _mm256_storeu_si256(
                                         reinterpret_cast<__m256i *>(thread_data + offset + b),
                                         value);
                                   }
#ifdef ENABLE_TRACING
                                   trace_buffer.push_back(
                                       {ts, (uintptr_t)(thread_data + offset + b), 32, 1});
#endif
                                   local_bytes += 32;
                                   access_count++;
                                   if (local_bytes >= PROGRESS_UPDATE_BYTES)
                                   {
                                     progress.add_bytes(local_bytes);
                                     local_bytes = 0;
                                   }
                                 }
                               }
                             }
                           }
                           progress.add_bytes(local_bytes);

                           // Ensure all non-temporal stores complete
                           if (bypass_cache)
                           {
                             _mm_sfence();
                           }

                           auto thread_end = steady_clock::now();
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
                         });
  }

  // Wait for all threads to finish setup
  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  // Start all threads simultaneously
  start_flag.store(true);

  // Drive the progress monitor on this (main) thread — no separate monitor
  // thread is spawned, so worker count == active thread count and there is
  // no CPU-share bias affecting low-thread-count measurements.
  progress.wait_until_done();

  // Workers are essentially done by the time wait_until_done() returns.
  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = steady_clock::now();
  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

  // Use total memory size for bandwidth calculation (all memory accessed once)
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  return {bandwidth_gbps, elapsed_ms};
}

// Pointer Chase with Load Test
// Measures latency under memory bandwidth pressure
// - 1 measurement thread: performs pointer chase and measures latency
// - N load threads: generate memory traffic (half read, half write) with inject
// delay
// - membind_node: NUMA node where data is allocated (-1 for devdax/default)
BandwidthResult measurePointerChaseWithLoad(void *data, size_t size,
                                            int num_load_threads,
                                            uint64_t inject_delay_cycles,
                                            int membind_node)
{
  const size_t NODE_SIZE = 64;          // 256 bytes per node
  const size_t NUM_SAMPLES = 200000;    // Number of latency samples to collect
  const size_t WARMUP_SAMPLES = 100000; // Skip first N samples for warmup

  // Fixed 2GB chase region for L2 STLB coverage (~2GB with 2MB huge pages)
  const size_t CHASE_SIZE_BYTES = 2ULL * 1024 * 1024 * 1024; // 2GB
  if (size < CHASE_SIZE_BYTES)
  {
    fprintf(stderr,
            "Error: Memory size (%zu bytes) is less than required chase size "
            "(2GB)\n",
            size);
    return {0, 0};
  }
  size_t chase_size = CHASE_SIZE_BYTES;
  size_t load_size = size - chase_size;
  void *chase_data = data;
  void *load_data = (char *)data + chase_size;

  size_t num_nodes = chase_size / NODE_SIZE;
  if (num_nodes == 0)
  {
    fprintf(stderr, "Error: Memory size too small for pointer chase\n");
    return {0, 0};
  }

  // Find auxiliary NUMA node (different from membind_node) for index arrays
  // This avoids bandwidth contention with the main data
  int aux_node = -1;
  if (membind_node >= 0 && numa_available() >= 0)
  {
    int max_node = numa_max_node();
    for (int node = 0; node <= max_node; node++)
    {
      if (node != membind_node)
      {
        aux_node = node;
        break;
      }
    }
    if (aux_node >= 0)
    {
      printf(
          "Using auxiliary NUMA node %d for index arrays (data on node %d)\n",
          aux_node, membind_node);
    }
  }

  // Initialize pointer chase structure (256B nodes with random links)
  printf(
      "Initializing pointer chase structure (%zu nodes, chase: %zu MB, load: "
      "%zu MB)...\n",
      num_nodes, chase_size / (1000 * 1000), load_size / (1000 * 1000));

  // Create array of node indices (allocate on auxiliary node if available)
  size_t *indices = nullptr;
  if (aux_node >= 0)
  {
    indices = (size_t *)numa_alloc_onnode(num_nodes * sizeof(size_t), aux_node);
  }
  if (indices == nullptr)
  {
    indices = (size_t *)malloc(num_nodes * sizeof(size_t));
  }
  for (size_t i = 0; i < num_nodes; i++)
  {
    indices[i] = i;
  }

  // Shuffle using Fisher-Yates
  random_device rd;
  mt19937 gen(rd());
  for (size_t i = num_nodes - 1; i > 0; i--)
  {
    uniform_int_distribution<size_t> dis(0, i);
    size_t j = dis(gen);
    swap(indices[i], indices[j]);
  }

  // Build circular linked list with shuffled order
  for (size_t i = 0; i < num_nodes; i++)
  {
    char *current_node = (char *)chase_data + indices[i] * NODE_SIZE;
    char *next_node =
        (char *)chase_data + indices[(i + 1) % num_nodes] * NODE_SIZE;
    *(void **)current_node = next_node;
  }

  printf("Pointer chase structure initialized.\n");

  // Shared control flags
  atomic<bool> start_flag(false);
  atomic<bool> stop_flag(false);
  atomic<int> ready_count(0);

  // Latency measurement thread
  thread measurement_thread([&]()
                            {
                              // Set CPU affinity if enabled
                              if (!g_cpu_affinity_list.empty())
                              {
                                int cpu_id = g_cpu_affinity_list[0];
                                setThreadCpuAffinity(cpu_id);
                              }

#ifdef ENABLE_LATENCY_MEASURE
                              reserveLatencyBuffer(NUM_SAMPLES);
#endif

                              // Signal ready
                              ready_count.fetch_add(1);

                              // Wait for start signal
                              while (!start_flag.load())
                              {
                                this_thread::yield();
                              }

                              // Perform pointer chase
                              void *volatile current = chase_data; // volatile to prevent optimization
                              size_t total_iterations = WARMUP_SAMPLES + NUM_SAMPLES;
                              for (size_t i = 0; i < total_iterations; i++)
                              {
                                uint64_t start = rdtscp();

                                // Memory barrier + volatile load to prevent compiler optimization
                                asm volatile("" ::: "memory");
                                current = *(void *volatile *)current; // Chase to next node
                                asm volatile("" ::: "memory");

                                uint64_t end = rdtscp();

#ifdef ENABLE_LATENCY_MEASURE
                                // Skip warmup samples
                                if (i >= WARMUP_SAMPLES)
                                {
                                  latency_buffer.push_back(end - start);
                                }
#endif
                              }

                              // Prevent optimization
                              asm volatile("" ::"r"(current) : "memory");

                              // Signal stop to load threads
                              stop_flag.store(true);

#ifdef ENABLE_LATENCY_MEASURE
                              collectLatencyBuffers();
#endif
                            });

  // Load threads (read + write)
  vector<thread> load_threads;
  vector<size_t> load_thread_bytes(num_load_threads, 0);

  int num_read_threads = num_load_threads / 2;

  for (int t = 0; t < num_load_threads; t++)
  {
    bool is_read = (t < num_read_threads);

    load_threads.emplace_back([&, t, is_read, aux_node]()
                              {
      // Set CPU affinity if enabled
      if (!g_cpu_affinity_list.empty()) {
        int cpu_id = g_cpu_affinity_list[(t + 1) % g_cpu_affinity_list.size()];
        setThreadCpuAffinity(cpu_id);
      }

      size_t per_thread_size = load_size / num_load_threads;
      int32_t* thread_data =
          reinterpret_cast<int32_t*>((char*)load_data + t * per_thread_size);
      size_t total_ints = per_thread_size / sizeof(int32_t);

      // Signal ready
      ready_count.fetch_add(1);

      // Wait for start signal
      while (!start_flag.load()) {
        this_thread::yield();
      }

      // Load generation loop
      size_t bytes_accessed = 0;
      size_t offset = 0;

      if (is_read) {
        // Sequential read
        __m256i acc = _mm256_setzero_si256();

        while (!stop_flag.load()) {
          __m256i data_vec = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(&thread_data[offset]));
          acc = _mm256_add_epi32(acc, data_vec);
          bytes_accessed += 32;
          offset += 8;  // 8 int32 = 32 bytes

          if (offset >= total_ints) {
            offset = 0;  // Wrap around
          }

          // Inject delay
          if (inject_delay_cycles > 0) {
            uint64_t delay_start = rdtscp();
            while (rdtscp() - delay_start < inject_delay_cycles) {
              _mm_pause();
            }
          }
        }

        // Prevent optimization
        volatile long long sum = 0;
        int32_t temp[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), acc);
        for (int j = 0; j < 8; j++) sum += temp[j];

      } else {
        // Sequential write
        __m256i value = _mm256_set1_epi32(t);

        while (!stop_flag.load()) {
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&thread_data[offset]),
                              value);
          bytes_accessed += 32;
          offset += 8;

          if (offset >= total_ints) {
            offset = 0;  // Wrap around
          }

          // Inject delay
          if (inject_delay_cycles > 0) {
            uint64_t delay_start = rdtscp();
            while (rdtscp() - delay_start < inject_delay_cycles) {
              _mm_pause();
            }
          }
        }
      }

      load_thread_bytes[t] = bytes_accessed; });
  }

  // Wait for all threads to be ready
  while (ready_count.load() < num_load_threads + 1)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  // Start all threads simultaneously
  start_flag.store(true);

  // Wait for measurement thread to complete
  measurement_thread.join();

  // Wait for load threads to complete
  for (auto &th : load_threads)
  {
    th.join();
  }

  auto end_time = steady_clock::now();
  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

  // Calculate total load bandwidth
  size_t total_bytes = 0;
  for (size_t bytes : load_thread_bytes)
  {
    total_bytes += bytes;
  }

  double load_bandwidth_gbps =
      (total_bytes / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  // Free indices array
  if (aux_node >= 0 && indices != nullptr)
  {
    numa_free(indices, num_nodes * sizeof(size_t));
  }
  else if (indices != nullptr)
  {
    free(indices);
  }

  return {load_bandwidth_gbps, elapsed_ms};
}

// Phase 3 of the prefetch test: cold sequential read of [data, data+size) that
// issues a PXL device prefetch `ahead` blocks in front of the current block.
// ahead == 0 means no prefetch (baseline). Returns the read bandwidth.
// `chunk_size` is the device-prefetch granularity: each pxl::prefetchMemory
// call covers one chunk, and prefetch is issued `ahead` chunks in front of the
// current read position. The read itself is a flat sequential scan and does not
// depend on the chunk size.
static BandwidthResult prefetchReadPass(void *data, size_t size, int num_threads,
                                        bool bypass_cache, int ahead, size_t chunk_size)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  ProgressMonitor progress(size, "PrefetchRead");

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache, ahead, chunk_size]()
                         {
                           if (!g_cpu_affinity_list.empty())
                           {
                             int cpu_id = g_cpu_affinity_list[t % g_cpu_affinity_list.size()];
                             setThreadCpuAffinity(cpu_id);
                           }

                           size_t per_thread_size = size / num_threads;
                           int32_t *thread_data =
                               reinterpret_cast<int32_t *>((char *)data + t * per_thread_size);
                           size_t per_thread_ints = per_thread_size / sizeof(int32_t);
                           const size_t chunk_ints = chunk_size / sizeof(int32_t);
                           const size_t num_full_chunks = per_thread_ints / chunk_ints;

                           // Flush host cache so the read is cold (skip for non-temporal loads)
                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           ready_count.fetch_add(1);
                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           // Warmup: pre-issue device prefetch for the first `ahead` chunks
                           if (ahead > 0)
                           {
                             size_t warmup = std::min((size_t)ahead, num_full_chunks);
                             for (size_t a = 0; a < warmup; a++)
                             {
                               devicePrefetch(reinterpret_cast<char *>(thread_data) + a * chunk_size, chunk_size);
                             }
                           }

                           __m256i acc = _mm256_setzero_si256();
                           size_t local_bytes = 0;
                           for (size_t chunk = 0; chunk < num_full_chunks; chunk++)
                           {
                             // Prefetch the chunk `ahead` chunks in front of the current one
                             if (ahead > 0)
                             {
                               size_t target = chunk + (size_t)ahead;
                               if (target < num_full_chunks)
                               {
                                 devicePrefetch(reinterpret_cast<char *>(thread_data) + target * chunk_size, chunk_size);
                               }
                             }

                             const size_t base_ints = chunk * chunk_ints;
                             for (size_t j = 0; j + 8 <= chunk_ints; j += 8)
                             {
                               size_t i = base_ints + j;
                               __m256i data_vec = bypass_cache
                                                      ? _mm256_stream_load_si256(
                                                            reinterpret_cast<__m256i *>(&thread_data[i]))
                                                      : _mm256_loadu_si256(
                                                            reinterpret_cast<const __m256i *>(&thread_data[i]));
                               acc = _mm256_add_epi32(acc, data_vec);

                               local_bytes += 32;
                               if (local_bytes >= PROGRESS_UPDATE_BYTES)
                               {
                                 progress.add_bytes(local_bytes);
                                 local_bytes = 0;
                               }
                             }
                           }

                           // Drain accumulator so the loads are not optimized away
                           volatile long long sum = 0;
                           int32_t temp[8];
                           _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc);
                           for (int j = 0; j < 8; j++)
                             sum += temp[j];

                           size_t i = num_full_chunks * chunk_ints;
                           for (; i < per_thread_ints; i++)
                           {
                             sum += thread_data[i];
                             local_bytes += 4;
                           }
                           progress.add_bytes(local_bytes);

                           auto thread_end = steady_clock::now();
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
                         });
  }

  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();
  start_flag.store(true);
  progress.wait_until_done();
  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = steady_clock::now();
  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  return {bandwidth_gbps, elapsed_ms};
}

// Explicit-prefetch test (PXL only). See benchmark.h for the region layout.
// Phase 1: write region A. Phase 2: read region B (evicts A). Phase 3: cold
// read of A with device prefetch `explicit_prefetch_ahead` blocks ahead — the
// returned bandwidth is for this phase.
BandwidthResult measurePrefetchTest(void *data, size_t size, int num_threads,
                                    bool bypass_cache, int explicit_prefetch_ahead,
                                    size_t prefetch_chunk_size)
{
  // Phases 1 and 2 are just setup (fill region A, evict it via region B), so
  // they run with a fixed thread count. Only the measured phase 3 uses the
  // caller-specified num_threads.
  const int kSetupThreads = 16;

  void *region_a = data;
  void *region_b = static_cast<char *>(data) + size;

  printf("  [prefetch_test] Phase 1: sequential write to region A (%d threads)...\n",
         kSetupThreads);
  measureSequentialWrite(region_a, size, kSetupThreads, bypass_cache);

  printf("  [prefetch_test] Phase 2: sequential read of region B (evict A, %d threads)...\n",
         kSetupThreads);
  measureSequentialRead(region_b, size, kSetupThreads, bypass_cache);

  printf("  [prefetch_test] Phase 3: cold read of region A (%d threads, ahead=%d chunks of %zu MiB)...\n",
         num_threads, explicit_prefetch_ahead, prefetch_chunk_size / (1024 * 1024));
  return prefetchReadPass(region_a, size, num_threads, bypass_cache,
                          explicit_prefetch_ahead, prefetch_chunk_size);
}
