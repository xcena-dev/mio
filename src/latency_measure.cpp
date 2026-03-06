#ifdef ENABLE_LATENCY_MEASURE

#include "latency_measure.h"
#include <fstream>
#include <cstdio>
#include <mutex>
#include <algorithm>
#include <ctime>

// Thread-local latency buffer definition
thread_local std::vector<uint64_t> latency_buffer;

// Global storage for collected latencies from all threads
static std::vector<uint64_t> global_latencies;
static std::mutex global_latencies_mutex;

double getCpuFrequencyGHz() {
    // TSC calibration using clock_gettime (accurate method)
    struct timespec start_ts, end_ts;

    clock_gettime(CLOCK_MONOTONIC, &start_ts);
    uint64_t start_cycles = rdtscp();

    // Wait 100ms for calibration
    struct timespec sleep_ts = {0, 100000000};  // 100ms
    nanosleep(&sleep_ts, nullptr);

    uint64_t end_cycles = rdtscp();
    clock_gettime(CLOCK_MONOTONIC, &end_ts);

    // Calculate actual elapsed time in nanoseconds
    double elapsed_ns = (end_ts.tv_sec - start_ts.tv_sec) * 1e9
                      + (end_ts.tv_nsec - start_ts.tv_nsec);

    uint64_t cycles = end_cycles - start_cycles;
    return cycles / elapsed_ns;  // cycles/ns = GHz
}

void reserveLatencyBuffer(size_t expected_samples) {
    latency_buffer.reserve(expected_samples);
}

void collectLatencyBuffers() {
    std::lock_guard<std::mutex> lock(global_latencies_mutex);
    global_latencies.insert(global_latencies.end(),
                           latency_buffer.begin(),
                           latency_buffer.end());
    latency_buffer.clear();
}

void saveLatencyLog(const char* filepath, double cpu_freq_ghz) {
    std::lock_guard<std::mutex> lock(global_latencies_mutex);

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        fprintf(stderr, "Error: Failed to open latency log file: %s\n", filepath);
        return;
    }

    // Write header
    ofs << "# Latency Log (ns)\n";
    ofs << "# Samples: " << global_latencies.size() << "\n";
    ofs << "# CPU Frequency: " << cpu_freq_ghz << " GHz\n";

    // Write all latency samples in nanoseconds
    for (uint64_t cycles : global_latencies) {
        double ns = cyclesToNs(cycles, cpu_freq_ghz);
        ofs << ns << "\n";
    }

    ofs.close();
    printf("Latency log saved: %s (%zu samples)\n", filepath, global_latencies.size());
}

void clearLatencyBuffers() {
    std::lock_guard<std::mutex> lock(global_latencies_mutex);
    global_latencies.clear();
    latency_buffer.clear();
}

#endif // ENABLE_LATENCY_MEASURE
