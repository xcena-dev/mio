#pragma once

#ifdef ENABLE_LATENCY_MEASURE

#include <vector>
#include <cstdint>
#include <cstddef>

// rdtscp inline function (serialized timestamp)
static inline uint64_t rdtscp() {
    unsigned int aux;
    uint64_t rax, rdx;
    asm volatile("rdtscp" : "=a"(rax), "=d"(rdx), "=c"(aux));
    return (rdx << 32) | rax;
}

// Get CPU frequency in GHz (for cycle-to-nanosecond conversion)
double getCpuFrequencyGHz();

// Convert cycles to nanoseconds
inline double cyclesToNs(uint64_t cycles, double cpu_freq_ghz) {
    return (double)cycles / cpu_freq_ghz;
}

// Thread-local latency buffer
extern thread_local std::vector<uint64_t> latency_buffer;

// Sampling interval (measure 1 out of every LATENCY_SAMPLE_INTERVAL accesses)
constexpr int LATENCY_SAMPLE_INTERVAL = 100;

// Buffer management functions
void reserveLatencyBuffer(size_t expected_samples);
void collectLatencyBuffers();
void saveLatencyLog(const char* filepath, double cpu_freq_ghz);
void clearLatencyBuffers();

#endif // ENABLE_LATENCY_MEASURE
