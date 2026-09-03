#include "tracing.h"

#ifdef ENABLE_TRACING

#include <cstdio>
#include <mutex>
#include <algorithm>

using namespace std;

thread_local vector<TraceEntry> trace_buffer;

vector<vector<TraceEntry>> global_trace_buffers;
mutex trace_mutex;

void clearTraceBuffers() {
    global_trace_buffers.clear();
}

void reserveTraceBuffer(int num_threads) {
    const size_t buffer_capacity = (1000000 / num_threads) * 2;
    trace_buffer.reserve(buffer_capacity);
}

void collectTraceBuffer() {
    lock_guard<mutex> lock(trace_mutex);
    global_trace_buffers.push_back(trace_buffer);
    trace_buffer.clear();
}

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

void saveTrace(const char* dir, uintptr_t base_address) {
    vector<vector<TraceEntry>> processed_buffers;
    for (auto& buf : global_trace_buffers) {
        vector<TraceEntry> processed;

        for (size_t i = 0; i < buf.size(); i++) {
            if (i + 1 < buf.size() &&
                buf[i].size == 32 &&
                buf[i + 1].size == 32 &&
                buf[i].access_type == buf[i + 1].access_type &&
                buf[i + 1].address == buf[i].address + 32) {
                TraceEntry merged_entry = buf[i];
                merged_entry.size = 64;
                processed.push_back(merged_entry);
                i++;
            } else {
                processed.push_back(buf[i]);
            }
        }
        processed_buffers.push_back(processed);
    }

    vector<TraceEntry> merged;
    for (auto& buf : processed_buffers) {
        merged.insert(merged.end(), buf.begin(), buf.end());
    }

    sort(merged.begin(), merged.end(), [](const TraceEntry& a, const TraceEntry& b) {
        return a.timestamp < b.timestamp;
    });

    if (merged.size() > 1000000) {
        merged.resize(1000000);
    }

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/trace.dat", dir);
    FILE* f = fopen(filepath, "w");
    if (!f) {
        fprintf(stderr, "Error: Failed to open %s\n", filepath);
        return;
    }

    fprintf(f, "address,size,access_type\n");
    for (const auto& entry : merged) {
        uintptr_t offset = entry.address - base_address;
        fprintf(f, "0x%lx,%u,%s\n",
                offset,
                entry.size,
                entry.access_type == 0 ? "read" : "write");
    }
    fclose(f);

    printf("Saved %zu trace entries to %s\n", merged.size(), filepath);
}

#endif
