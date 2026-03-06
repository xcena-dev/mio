#include "tracing.h"

#ifdef ENABLE_TRACING

#include <cstdio>
#include <mutex>
#include <algorithm>

using namespace std;

// Thread-local trace buffers
thread_local vector<TraceEntry> trace_buffer;

// Global trace buffers for each thread (collected after measurement)
vector<vector<TraceEntry>> global_trace_buffers;
mutex trace_mutex;

// Clear all trace buffers
void clearTraceBuffers() {
    global_trace_buffers.clear();
}

// Reserve trace buffer capacity ((1M / num_threads) * 2)
void reserveTraceBuffer(int num_threads) {
    const size_t buffer_capacity = (1000000 / num_threads) * 2;
    trace_buffer.reserve(buffer_capacity);
}

// Collect trace buffer from current thread
void collectTraceBuffer() {
    lock_guard<mutex> lock(trace_mutex);
    global_trace_buffers.push_back(trace_buffer);
    trace_buffer.clear();
}

// Save metadata to file
void saveMetadata(const char* dir, void* base_addr, size_t size, int num_threads) {
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/metadata.dat", dir);
    FILE* f = fopen(filepath, "w");
    if (!f) {
        fprintf(stderr, "Error: Failed to open %s\n", filepath);
        return;
    }
    fprintf(f, "base_address,size,num_threads\n");
    fprintf(f, "%p,%zu,%d\n", base_addr, size, num_threads);
    fclose(f);
}

// Save trace to file (merge, sort by timestamp, write CSV)
void saveTrace(const char* dir, uintptr_t base_address) {
    // Step 1: Process each thread's buffer - merge consecutive 32-byte accesses to 64-byte
    vector<vector<TraceEntry>> processed_buffers;
    for (auto& buf : global_trace_buffers) {
        vector<TraceEntry> processed;

        for (size_t i = 0; i < buf.size(); i++) {
            // Check if we can merge with next entry
            if (i + 1 < buf.size() &&
                buf[i].size == 32 &&
                buf[i + 1].size == 32 &&
                buf[i].access_type == buf[i + 1].access_type &&
                buf[i + 1].address == buf[i].address + 32) {
                // Merge two consecutive 32-byte accesses into one 64-byte access
                TraceEntry merged_entry = buf[i];
                merged_entry.size = 64;
                processed.push_back(merged_entry);
                i++;  // Skip next entry as it's merged
            } else {
                // Keep as is
                processed.push_back(buf[i]);
            }
        }
        processed_buffers.push_back(processed);
    }

    // Step 2: Merge all processed thread buffers
    vector<TraceEntry> merged;
    for (auto& buf : processed_buffers) {
        merged.insert(merged.end(), buf.begin(), buf.end());
    }

    // Step 3: Sort by timestamp
    sort(merged.begin(), merged.end(), [](const TraceEntry& a, const TraceEntry& b) {
        return a.timestamp < b.timestamp;
    });

    // Step 4: Limit to top 1 million entries
    if (merged.size() > 1000000) {
        merged.resize(1000000);
    }

    // Step 5: Write to file with offset addresses
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/trace.dat", dir);
    FILE* f = fopen(filepath, "w");
    if (!f) {
        fprintf(stderr, "Error: Failed to open %s\n", filepath);
        return;
    }

    fprintf(f, "address,size,access_type\n");
    for (const auto& entry : merged) {
        // Convert absolute address to offset
        uintptr_t offset = entry.address - base_address;
        fprintf(f, "0x%lx,%u,%s\n",
                offset,
                entry.size,
                entry.access_type == 0 ? "read" : "write");
    }
    fclose(f);

    printf("Saved %zu trace entries to %s\n", merged.size(), filepath);
}

#endif // ENABLE_TRACING
