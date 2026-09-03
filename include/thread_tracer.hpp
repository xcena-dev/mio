#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "benchmark.h"

namespace mio {
class ThreadTracer
{
    static constexpr uint64_t UpdateBytes = 4ULL * 1024 * 1024;
    static constexpr int64_t PrintIntervalMs = 1000;
    static constexpr size_t CacheLineSize = 64;
    static constexpr size_t SampleReserveCount = 4096;
    static constexpr size_t TracePathSize = 1024;
    static constexpr double BytesPerGb = 1e9;
    static constexpr double BytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    static constexpr double MsPerSecond = 1000.0;

    struct alignas(CacheLineSize) Snapshot_t
    {
        std::atomic<uint64_t> bytesDone{0};
        char padding[CacheLineSize - sizeof(std::atomic<uint64_t>)]{};
    };

public:
    class Slot
    {
    public:
        explicit Slot(Snapshot_t* snapshot) : snapshot_{snapshot} {}

        void add(size_t bytes)
        {
            pending_ += bytes;
            if (pending_ >= UpdateBytes)
                flush();
        }

        void flush()
        {
            total_ += pending_;
            pending_ = 0;
            snapshot_->bytesDone.store(total_, std::memory_order_relaxed);
        }

    private:
        Snapshot_t* snapshot_;
        uint64_t pending_{0};
        uint64_t total_{0};
    };

    ThreadTracer(int32_t numThreads, const char* modeName, size_t totalBytes);
    ~ThreadTracer();

    ThreadTracer(const ThreadTracer&) = delete;
    ThreadTracer& operator=(const ThreadTracer&) = delete;

    [[nodiscard]] Slot getSlot(int32_t threadIndex) { return Slot(&snapshots_[threadIndex]); }

    void start(std::chrono::steady_clock::time_point startTime);
    void finish(std::chrono::steady_clock::time_point endTime = {});

    [[nodiscard]] RegionBreakdown getRegions(double loFrac = g_steady_lo_frac,
                                             double hiFrac = g_steady_hi_frac) const;

private:
    [[nodiscard]] bool isSamplingNeeded() const;
    [[nodiscard]] uint64_t getLastTotal() const;
    [[nodiscard]] double getTimeAtCumulative(const std::vector<double>& cumulative,
                                             double target) const;

    void printProgress() const;
    void writeTrace() const;
    void sampleAt(std::chrono::steady_clock::time_point timePoint);
    void pollLoop();

    int32_t numThreads_;
    const char* mode_;
    size_t totalBytes_;
    std::vector<Snapshot_t> snapshots_;
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> samples_;
    std::chrono::steady_clock::time_point startTime_;
    std::atomic<bool> stopRequested_{false};
    std::mutex mutex_;
    std::condition_variable condition_;
    uint64_t epochMs_{0};
    std::thread logger_;
};
}
