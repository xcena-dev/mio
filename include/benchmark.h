#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cstddef>
#include <cstdint>
#include <vector>

// Bandwidth measurement result
struct BandwidthResult {
    double bandwidth_gbps;
    double elapsed_ms;
};

// Global configuration for block and stride sizes
extern size_t BLOCK_SIZE;   // Default: 64 bytes (used in random and stride patterns)
extern size_t STRIDE_SIZE;  // Default: 64 bytes

// Global CPU affinity list (empty means no affinity)
extern std::vector<int> g_cpu_affinity_list;

// Cache flush function
void flushHostCache(void *hostVirtualPtr, size_t size);

// Sequential access measurements
BandwidthResult measureSequentialRead(void* data, size_t size, int num_threads, bool bypass_cache = false);
BandwidthResult measureSequentialWrite(void* data, size_t size, int num_threads, bool bypass_cache = false);

// Random access measurements
BandwidthResult measureRandomRead(void* data, size_t size, int num_threads, bool bypass_cache = false);
BandwidthResult measureRandomWrite(void* data, size_t size, int num_threads, bool bypass_cache = false);

// Stride access measurements
BandwidthResult measureStrideRead(void* data, size_t size, int num_threads, size_t stride, bool bypass_cache = false);
BandwidthResult measureStrideWrite(void* data, size_t size, int num_threads, size_t stride, bool bypass_cache = false);

// Zipfian access measurements
BandwidthResult measureZipfianRead(void* data, size_t size, int num_threads, double zipfian_alpha = 0.99, bool bypass_cache = false);

// Pointer chase with load measurements
// membind_node: NUMA node where data is allocated (-1 for devdax/default)
//               Used to allocate auxiliary arrays on a different node to avoid BW contention
BandwidthResult measurePointerChaseWithLoad(void* data, size_t size, int num_load_threads, uint64_t inject_delay_cycles, int membind_node = -1);

#endif // BENCHMARK_H
