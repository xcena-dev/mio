#include "benchmark.h"

#include <immintrin.h>
#include <numa.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "numa_affinity.h"
#include "thread_tracer.hpp"

const char *g_thread_trace_dir = nullptr;
int g_thread_trace_interval_ms = 50;

bool g_detail_enabled = false;

double g_steady_lo_frac = 0.40;
double g_steady_hi_frac = 0.60;

#ifdef ENABLE_TRACING
#include "tracing.h"
#endif

#ifdef ENABLE_LATENCY_MEASURE
#include "latency_measure.h"
#endif

using namespace std;
using namespace chrono;

#ifndef ENABLE_LATENCY_MEASURE
static inline uint64_t rdtscp()
{
  uint32_t aux;
  uint64_t rax, rdx;
  asm volatile("rdtscp" : "=a"(rax), "=d"(rdx), "=c"(aux));
  return (rdx << 32) | rax;
}
#endif

size_t BLOCK_SIZE = 64;
size_t STRIDE_SIZE = 64;

std::vector<int> g_cpu_affinity_list;

bool g_progress_enabled = true;

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

BandwidthResult measureSequentialRead(void *data, size_t size,
                                      int num_threads, bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  vector<steady_clock::time_point> thread_end_tp(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  mio::ThreadTracer tracer(num_threads, "seq_read", size);

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
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

#ifdef ENABLE_TRACING
                           reserveTraceBuffer(num_threads);
#endif
#ifdef ENABLE_LATENCY_MEASURE

                           size_t expected_samples =
                               (per_thread_ints / 8 / LATENCY_SAMPLE_INTERVAL) * 2;
                           reserveLatencyBuffer(expected_samples);
#endif

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

                           __m256i acc = _mm256_setzero_si256();
                           auto trace = tracer.getSlot(t);
                           const size_t block_ints = BLOCK_SIZE / sizeof(int32_t);
                           const size_t num_full_blocks = per_thread_ints / block_ints;
#ifdef ENABLE_LATENCY_MEASURE

                           size_t cache_line_count = 0;
                           const size_t AVX_LOADS_PER_CACHELINE = 2;
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

                             trace.add(BLOCK_SIZE);
                           }

                           volatile long long sum = 0;
                           int32_t temp[8];
                           _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc);
                           for (int j = 0; j < 8; j++)
                             sum += temp[j];

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
                           }

                           trace.add((per_thread_ints - num_full_blocks * block_ints) *
                                     sizeof(int32_t));
                           trace.flush();

                           auto thread_end = steady_clock::now();
                           thread_end_tp[t] = thread_end;
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

  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  start_flag.store(true);

  tracer.start(start_time);

  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = *max_element(thread_end_tp.begin(), thread_end_tp.end());

  tracer.finish(end_time);

  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  BandwidthResult result{bandwidth_gbps, elapsed_ms, {}};
  result.regions = tracer.getRegions();
  if (result.regions.valid)
    result.bandwidth_gbps = result.regions.steady_gbps;
  return result;
}

BandwidthResult measureSequentialWrite(void *data, size_t size, int num_threads,
                                       bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  vector<steady_clock::time_point> thread_end_tp(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  mio::ThreadTracer tracer(num_threads, "seq_write", size);

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
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

#ifdef ENABLE_TRACING
                           reserveTraceBuffer(num_threads);
#endif

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

                           auto trace = tracer.getSlot(t);
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

                             trace.add(BLOCK_SIZE);
                           }

                           if (bypass_cache)
                           {
                             _mm_sfence();
                           }

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
                           }

                           trace.add((per_thread_ints - num_full_blocks * block_ints) *
                                     sizeof(int32_t));
                           trace.flush();

                           auto thread_end = steady_clock::now();
                           thread_end_tp[t] = thread_end;
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
                           flushHostCache(thread_data, per_thread_size);
#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
                         });
  }

  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  start_flag.store(true);

  tracer.start(start_time);

  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = *max_element(thread_end_tp.begin(), thread_end_tp.end());

  tracer.finish(end_time);

  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  BandwidthResult result{bandwidth_gbps, elapsed_ms, {}};
  result.regions = tracer.getRegions();
  if (result.regions.valid)
    result.bandwidth_gbps = result.regions.steady_gbps;
  return result;
}

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

static size_t sampleZipfian(const vector<double> &cdf, double u)
{
  return lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin();
}

BandwidthResult measureRandomRead(void *data, size_t size, int num_threads,
                                  bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  vector<steady_clock::time_point> thread_end_tp(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  mio::ThreadTracer tracer(num_threads, "random_read", size);

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
                         {
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

                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           vector<size_t> indices(num_blocks);
                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             indices[i] = i;
                           }

                           random_device rd;
                           mt19937 gen(rd() + t);
                           shuffle(indices.begin(), indices.end(), gen);

                           ready_count.fetch_add(1);

                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           __m256i acc = _mm256_setzero_si256();
                           auto trace = tracer.getSlot(t);
                           const size_t block_ints = BLOCK_SIZE / sizeof(int32_t);
#ifdef ENABLE_LATENCY_MEASURE
                           size_t block_count = 0;
#endif

                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             const size_t offset_ints = (indices[i] * BLOCK_SIZE) / sizeof(int32_t);
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
                             trace.add(BLOCK_SIZE);
                           }
                           trace.flush();

                           volatile long long sum = 0;
                           int32_t temp[8];
                           _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc);
                           for (int j = 0; j < 8; j++)
                             sum += temp[j];

                           auto thread_end = steady_clock::now();
                           thread_end_tp[t] = thread_end;
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

  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  start_flag.store(true);

  tracer.start(start_time);

  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = *max_element(thread_end_tp.begin(), thread_end_tp.end());

  tracer.finish(end_time);

  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  BandwidthResult result{bandwidth_gbps, elapsed_ms, {}};
  result.regions = tracer.getRegions();
  if (result.regions.valid)
    result.bandwidth_gbps = result.regions.steady_gbps;
  return result;
}

BandwidthResult measureRandomWrite(void *data, size_t size, int num_threads,
                                   bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  vector<steady_clock::time_point> thread_end_tp(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  mio::ThreadTracer tracer(num_threads, "random_write", size);

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
                         {
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

                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           vector<size_t> indices(num_blocks);
                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             indices[i] = i;
                           }

                           random_device rd;
                           mt19937 gen(rd() + t);
                           shuffle(indices.begin(), indices.end(), gen);

                           ready_count.fetch_add(1);

                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           auto trace = tracer.getSlot(t);
                           const size_t block_ints = BLOCK_SIZE / sizeof(int32_t);

                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             const size_t offset_ints = (indices[i] * BLOCK_SIZE) / sizeof(int32_t);

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
                             trace.add(BLOCK_SIZE);
                           }
                           trace.flush();

                           if (bypass_cache)
                           {
                             _mm_sfence();
                           }

                           auto thread_end = steady_clock::now();
                           thread_end_tp[t] = thread_end;
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
                           flushHostCache(thread_data, per_thread_size);

#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
                         });
  }

  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  start_flag.store(true);

  tracer.start(start_time);

  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = *max_element(thread_end_tp.begin(), thread_end_tp.end());

  tracer.finish(end_time);

  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  BandwidthResult result{bandwidth_gbps, elapsed_ms, {}};
  result.regions = tracer.getRegions();
  if (result.regions.valid)
    result.bandwidth_gbps = result.regions.steady_gbps;
  return result;
}

BandwidthResult measureZipfianRead(void *data, size_t size, int num_threads,
                                   double zipfian_alpha, bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  vector<steady_clock::time_point> thread_end_tp(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  mio::ThreadTracer tracer(num_threads, "zipfian_read", size);

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache, zipfian_alpha]()
                         {
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

                           if (!bypass_cache)
                           {
                             flushHostCache(thread_data, per_thread_size);
                             _mm_mfence();
                           }

                           vector<size_t> indices(num_blocks);
                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             indices[i] = i;
                           }
                           random_device rd;
                           mt19937 gen(rd() + t);
                           shuffle(indices.begin(), indices.end(), gen);

                           vector<double> cdf;
                           buildZipfianCDF(cdf, num_blocks, zipfian_alpha);

                           vector<size_t> access_sequence(num_blocks);
                           uniform_real_distribution<double> udist(0.0, 1.0);
                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             size_t rank = sampleZipfian(cdf, udist(gen));
                             access_sequence[i] = indices[rank];
                           }

                           ready_count.fetch_add(1);

                           while (!start_flag.load())
                           {
                             this_thread::yield();
                           }

                           auto thread_start = steady_clock::now();

                           __m256i acc = _mm256_setzero_si256();
                           auto trace = tracer.getSlot(t);
#ifdef ENABLE_LATENCY_MEASURE
                           size_t block_count = 0;
#endif
                           for (size_t i = 0; i < num_blocks; i++)
                           {
                             size_t offset_ints = (access_sequence[i] * BLOCK_SIZE) / sizeof(int32_t);
                             size_t block_ints = BLOCK_SIZE / sizeof(int32_t);
                             int32_t *block_ptr = &thread_data[offset_ints];

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
                             trace.add(BLOCK_SIZE);
                           }
                           trace.flush();

                           volatile long long sum = 0;
                           int32_t temp[8];
                           _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc);
                           for (int j = 0; j < 8; j++)
                             sum += temp[j];

                           auto thread_end = steady_clock::now();
                           thread_end_tp[t] = thread_end;
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

  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  start_flag.store(true);

  tracer.start(start_time);

  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = *max_element(thread_end_tp.begin(), thread_end_tp.end());

  tracer.finish(end_time);

  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;
  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  BandwidthResult result{bandwidth_gbps, elapsed_ms, {}};
  result.regions = tracer.getRegions();
  if (result.regions.valid)
    result.bandwidth_gbps = result.regions.steady_gbps;
  return result;
}

BandwidthResult measureStrideRead(void *data, size_t size, int num_threads,
                                  size_t stride, bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  vector<steady_clock::time_point> thread_end_tp(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  mio::ThreadTracer tracer(num_threads, "stride_read", size);

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
                         {
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

                           __m256i acc = _mm256_setzero_si256();
                           auto trace = tracer.getSlot(t);
                           size_t access_count = 0;

                           for (size_t phase = 0; phase < stride + BLOCK_SIZE; phase += BLOCK_SIZE)
                           {
                             for (size_t offset = phase; offset < per_thread_size;
                                  offset += (stride + BLOCK_SIZE))
                             {
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
                                   trace.add(32);
                                   access_count++;
                                 }
                               }
                             }
                           }
                           trace.flush();

                           volatile long long sum = 0;
                           int32_t temp[8];
                           _mm256_storeu_si256(reinterpret_cast<__m256i *>(temp), acc);
                           for (int j = 0; j < 8; j++)
                             sum += temp[j];

                           auto thread_end = steady_clock::now();
                           thread_end_tp[t] = thread_end;
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
                         });
  }

  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  start_flag.store(true);

  tracer.start(start_time);

  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = *max_element(thread_end_tp.begin(), thread_end_tp.end());

  tracer.finish(end_time);

  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  BandwidthResult result{bandwidth_gbps, elapsed_ms, {}};
  result.regions = tracer.getRegions();
  if (result.regions.valid)
    result.bandwidth_gbps = result.regions.steady_gbps;
  return result;
}

BandwidthResult measureStrideWrite(void *data, size_t size, int num_threads,
                                   size_t stride, bool bypass_cache)
{
  vector<thread> threads;
  vector<double> thread_times(num_threads);
  vector<steady_clock::time_point> thread_end_tp(num_threads);
  atomic<bool> start_flag(false);
  atomic<int> ready_count(0);

  mio::ThreadTracer tracer(num_threads, "stride_write", size);

  for (int t = 0; t < num_threads; t++)
  {
    threads.emplace_back([&, t, bypass_cache]()
                         {
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

                           __m256i value = _mm256_set1_epi32(t);
                           auto trace = tracer.getSlot(t);
                           size_t access_count = 0;

                           for (size_t phase = 0; phase < stride + BLOCK_SIZE; phase += BLOCK_SIZE)
                           {
                             for (size_t offset = phase; offset < per_thread_size;
                                  offset += (stride + BLOCK_SIZE))
                             {
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
                                   trace.add(32);
                                   access_count++;
                                 }
                               }
                             }
                           }
                           trace.flush();

                           if (bypass_cache)
                           {
                             _mm_sfence();
                           }

                           auto thread_end = steady_clock::now();
                           thread_end_tp[t] = thread_end;
                           thread_times[t] =
                               duration_cast<microseconds>(thread_end - thread_start).count() /
                               1000.0;
#ifdef ENABLE_TRACING
                           collectTraceBuffer();
#endif
                         });
  }

  while (ready_count.load() < num_threads)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  start_flag.store(true);

  tracer.start(start_time);

  for (auto &th : threads)
  {
    th.join();
  }

  auto end_time = *max_element(thread_end_tp.begin(), thread_end_tp.end());

  tracer.finish(end_time);

  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

  double bandwidth_gbps =
      (size / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

  BandwidthResult result{bandwidth_gbps, elapsed_ms, {}};
  result.regions = tracer.getRegions();
  if (result.regions.valid)
    result.bandwidth_gbps = result.regions.steady_gbps;
  return result;
}

BandwidthResult measurePointerChaseWithLoad(void *data, size_t size,
                                            int num_load_threads,
                                            uint64_t inject_delay_cycles,
                                            int membind_node)
{
  const size_t NODE_SIZE = 64;
  const size_t NUM_SAMPLES = 200000;
  const size_t WARMUP_SAMPLES = 100000;

  const size_t CHASE_SIZE_BYTES = 2ULL * 1024 * 1024 * 1024;
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

  printf(
      "Initializing pointer chase structure (%zu nodes, chase: %zu MB, load: "
      "%zu MB)...\n",
      num_nodes, chase_size / (1000 * 1000), load_size / (1000 * 1000));

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

  random_device rd;
  mt19937 gen(rd());
  for (size_t i = num_nodes - 1; i > 0; i--)
  {
    uniform_int_distribution<size_t> dis(0, i);
    size_t j = dis(gen);
    swap(indices[i], indices[j]);
  }

  for (size_t i = 0; i < num_nodes; i++)
  {
    char *current_node = (char *)chase_data + indices[i] * NODE_SIZE;
    char *next_node =
        (char *)chase_data + indices[(i + 1) % num_nodes] * NODE_SIZE;
    *(void **)current_node = next_node;
  }

  printf("Pointer chase structure initialized.\n");

  atomic<bool> start_flag(false);
  atomic<bool> stop_flag(false);
  atomic<int> ready_count(0);

  thread measurement_thread([&]()
                            {
                              if (!g_cpu_affinity_list.empty())
                              {
                                int cpu_id = g_cpu_affinity_list[0];
                                setThreadCpuAffinity(cpu_id);
                              }

#ifdef ENABLE_LATENCY_MEASURE
                              reserveLatencyBuffer(NUM_SAMPLES);
#endif

                              ready_count.fetch_add(1);

                              while (!start_flag.load())
                              {
                                this_thread::yield();
                              }

                              void *volatile current = chase_data;
                              size_t total_iterations = WARMUP_SAMPLES + NUM_SAMPLES;
                              for (size_t i = 0; i < total_iterations; i++)
                              {
                                uint64_t start = rdtscp();

                                asm volatile("" ::: "memory");
                                current = *(void *volatile *)current;
                                asm volatile("" ::: "memory");

                                uint64_t end = rdtscp();

#ifdef ENABLE_LATENCY_MEASURE

                                if (i >= WARMUP_SAMPLES)
                                {
                                  latency_buffer.push_back(end - start);
                                }
#endif
                              }

                              asm volatile("" ::"r"(current) : "memory");

                              stop_flag.store(true);

#ifdef ENABLE_LATENCY_MEASURE
                              collectLatencyBuffers();
#endif
                            });

  vector<thread> load_threads;
  vector<size_t> load_thread_bytes(num_load_threads, 0);

  int num_read_threads = num_load_threads / 2;

  for (int t = 0; t < num_load_threads; t++)
  {
    bool is_read = (t < num_read_threads);

    load_threads.emplace_back([&, t, is_read, aux_node]()
                              {
      if (!g_cpu_affinity_list.empty()) {
        int cpu_id = g_cpu_affinity_list[(t + 1) % g_cpu_affinity_list.size()];
        setThreadCpuAffinity(cpu_id);
      }

      size_t per_thread_size = load_size / num_load_threads;
      int32_t* thread_data =
          reinterpret_cast<int32_t*>((char*)load_data + t * per_thread_size);
      size_t total_ints = per_thread_size / sizeof(int32_t);

      ready_count.fetch_add(1);

      while (!start_flag.load()) {
        this_thread::yield();
      }

      size_t bytes_accessed = 0;
      size_t offset = 0;

      if (is_read) {
        __m256i acc = _mm256_setzero_si256();

        while (!stop_flag.load()) {
          __m256i data_vec = _mm256_loadu_si256(
              reinterpret_cast<const __m256i*>(&thread_data[offset]));
          acc = _mm256_add_epi32(acc, data_vec);
          bytes_accessed += 32;
          offset += 8;

          if (offset >= total_ints) {
            offset = 0;
          }

          if (inject_delay_cycles > 0) {
            uint64_t delay_start = rdtscp();
            while (rdtscp() - delay_start < inject_delay_cycles) {
              _mm_pause();
            }
          }
        }

        volatile long long sum = 0;
        int32_t temp[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp), acc);
        for (int j = 0; j < 8; j++) sum += temp[j];
      } else {
        __m256i value = _mm256_set1_epi32(t);

        while (!stop_flag.load()) {
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&thread_data[offset]),
                              value);
          bytes_accessed += 32;
          offset += 8;

          if (offset >= total_ints) {
            offset = 0;
          }

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

  while (ready_count.load() < num_load_threads + 1)
  {
    this_thread::yield();
  }

  auto start_time = steady_clock::now();

  start_flag.store(true);

  measurement_thread.join();

  for (auto &th : load_threads)
  {
    th.join();
  }

  auto end_time = steady_clock::now();
  double elapsed_ms =
      duration_cast<microseconds>(end_time - start_time).count() / 1000.0;

  size_t total_bytes = 0;
  for (size_t bytes : load_thread_bytes)
  {
    total_bytes += bytes;
  }

  double load_bandwidth_gbps =
      (total_bytes / (1000.0 * 1000.0 * 1000.0)) / (elapsed_ms / 1000.0);

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
