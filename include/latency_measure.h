#pragma once

#ifdef ENABLE_LATENCY_MEASURE

#include <vector>
#include <cstdint>
#include <cstddef>

static inline uint64_t rdtscp() {
    unsigned int aux;
    uint64_t rax, rdx;
    asm volatile("rdtscp" : "=a"(rax), "=d"(rdx), "=c"(aux));
    return (rdx << 32) | rax;
}

double getCpuFrequencyGHz();

inline double cyclesToNs(uint64_t cycles, double cpu_freq_ghz) {
    return (double)cycles / cpu_freq_ghz;
}

extern thread_local std::vector<uint64_t> latency_buffer;

constexpr int LATENCY_SAMPLE_INTERVAL = 100;

void reserveLatencyBuffer(size_t expected_samples);
void collectLatencyBuffers();
void saveLatencyLog(const char* filepath, double cpu_freq_ghz);
void clearLatencyBuffers();

#endif
