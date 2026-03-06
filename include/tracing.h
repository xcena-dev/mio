#ifndef TRACING_H
#define TRACING_H

#ifdef ENABLE_TRACING

#include <cstdint>
#include <vector>

// Trace entry structure
struct TraceEntry {
    uint64_t timestamp;   // CPU cycle from rdtsc (for sorting only)
    uintptr_t address;    // Memory address accessed
    uint32_t size;        // Access size in bytes
    uint8_t access_type;  // 0=read, 1=write
};

// Thread-local trace buffers
extern thread_local std::vector<TraceEntry> trace_buffer;

// Get CPU timestamp
static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

// Clear all trace buffers
void clearTraceBuffers();

// Reserve trace buffer capacity ((1M / num_threads) * 2)
void reserveTraceBuffer(int num_threads);

// Collect trace buffer from current thread
void collectTraceBuffer();

// Save metadata to file
void saveMetadata(const char* dir, void* base_addr, size_t size, int num_threads);

// Save trace to file (merge, sort by timestamp, write CSV)
void saveTrace(const char* dir, uintptr_t base_address);

#endif // ENABLE_TRACING

#endif // TRACING_H
