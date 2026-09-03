#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct RegionBreakdown {
    double head_gbps, steady_gbps, tail_gbps;
    double head_ms, steady_ms, tail_ms;
    double head_bytes, steady_bytes, tail_bytes;
    bool valid;
};

struct BandwidthResult {
    double bandwidth_gbps;
    double elapsed_ms;
    RegionBreakdown regions;
};

extern size_t BLOCK_SIZE;
extern size_t STRIDE_SIZE;

extern std::vector<int> g_cpu_affinity_list;

extern bool g_progress_enabled;

extern const char *g_thread_trace_dir;

extern int g_thread_trace_interval_ms;

extern bool g_detail_enabled;

extern double g_steady_lo_frac;
extern double g_steady_hi_frac;

void flushHostCache(void *hostVirtualPtr, size_t size);

BandwidthResult measureSequentialRead(void* data, size_t size, int num_threads, bool bypass_cache = false);
BandwidthResult measureSequentialWrite(void* data, size_t size, int num_threads, bool bypass_cache = false);

BandwidthResult measureRandomRead(void* data, size_t size, int num_threads, bool bypass_cache = false);
BandwidthResult measureRandomWrite(void* data, size_t size, int num_threads, bool bypass_cache = false);

BandwidthResult measureStrideRead(void* data, size_t size, int num_threads, size_t stride, bool bypass_cache = false);
BandwidthResult measureStrideWrite(void* data, size_t size, int num_threads, size_t stride, bool bypass_cache = false);

BandwidthResult measureZipfianRead(void* data, size_t size, int num_threads, double zipfian_alpha = 0.99, bool bypass_cache = false);

BandwidthResult measurePointerChaseWithLoad(void* data, size_t size, int num_load_threads, uint64_t inject_delay_cycles, int membind_node = -1);

#endif
