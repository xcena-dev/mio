#ifndef TRACING_H
#define TRACING_H

#ifdef ENABLE_TRACING

#include <cstdint>
#include <vector>

struct TraceEntry {
    uint64_t timestamp;
    uintptr_t address;
    uint32_t size;
    uint8_t access_type;
};

extern thread_local std::vector<TraceEntry> trace_buffer;

static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

void clearTraceBuffers();

void reserveTraceBuffer(int num_threads);

void collectTraceBuffer();

void saveMetadata(const char* dir, void* base_addr, size_t size, int num_threads);

void saveTrace(const char* dir, uintptr_t base_address);

#endif

#endif
