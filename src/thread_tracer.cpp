#include "thread_tracer.hpp"

#include <cstdio>
#include <ctime>
#include <fstream>

namespace mio {
ThreadTracer::ThreadTracer(int32_t numThreads, const char* modeName, size_t totalBytes)
    : numThreads_{numThreads},
      mode_{modeName},
      totalBytes_{totalBytes},
      snapshots_(static_cast<size_t>(numThreads))
{
    samples_.reserve(SampleReserveCount);
}

ThreadTracer::~ThreadTracer()
{
    finish();
}

void ThreadTracer::start(std::chrono::steady_clock::time_point startTime)
{
    if (!isSamplingNeeded())
        return;

    startTime_ = startTime;

    const auto nowSteady = std::chrono::steady_clock::now();
    const auto nowReal = std::chrono::system_clock::now();
    epochMs_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            (nowReal - (nowSteady - startTime)).time_since_epoch())
            .count());

    stopRequested_.store(false, std::memory_order_relaxed);
    logger_ = std::thread([this] { this->pollLoop(); });
}

void ThreadTracer::finish(std::chrono::steady_clock::time_point endTime)
{
    if (!logger_.joinable())
        return;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopRequested_.store(true, std::memory_order_relaxed);
    }
    condition_.notify_one();
    logger_.join();

    sampleAt(endTime == std::chrono::steady_clock::time_point{}
                 ? std::chrono::steady_clock::now()
                 : endTime);
    printProgress();

    writeTrace();
}

RegionBreakdown ThreadTracer::getRegions(double loFrac, double hiFrac) const
{
    RegionBreakdown regions{};
    if (samples_.size() < 3)
        return regions;

    std::vector<double> cumulative(samples_.size());
    for (size_t i = 0; i < samples_.size(); i++)
    {
        double sum = 0.0;
        for (const auto bytes : samples_[i].second)
            sum += static_cast<double>(bytes);
        cumulative[i] = sum;
    }

    const double total = cumulative.back();
    if (total <= 0.0)
        return regions;

    size_t lastIndex = 0;
    while (lastIndex + 1 < cumulative.size() && cumulative[lastIndex] < total)
        lastIndex++;

    const double endMs = static_cast<double>(samples_[lastIndex].first);
    if (endMs <= 0.0)
        return regions;

    const double loMs = getTimeAtCumulative(cumulative, total * loFrac);
    const double hiMs = getTimeAtCumulative(cumulative, total * hiFrac);
    const double loBytes = total * loFrac;
    const double hiBytes = total * hiFrac;
    if (!(loMs > 0.0) || !(hiMs > loMs) || !(endMs > hiMs))
        return regions;

    const auto toGbps = [](double bytes, double elapsedMs)
    { return elapsedMs > 0.0 ? (bytes / BytesPerGb) / (elapsedMs / MsPerSecond) : 0.0; };

    regions.head_bytes = loBytes;
    regions.steady_bytes = hiBytes - loBytes;
    regions.tail_bytes = total - hiBytes;
    regions.head_ms = loMs;
    regions.steady_ms = hiMs - loMs;
    regions.tail_ms = endMs - hiMs;
    regions.head_gbps = toGbps(regions.head_bytes, regions.head_ms);
    regions.steady_gbps = toGbps(regions.steady_bytes, regions.steady_ms);
    regions.tail_gbps = toGbps(regions.tail_bytes, regions.tail_ms);
    regions.valid = true;
    return regions;
}

bool ThreadTracer::isSamplingNeeded() const
{
    return (g_thread_trace_dir && *g_thread_trace_dir) || g_detail_enabled ||
           (g_progress_enabled && totalBytes_ > 0);
}

uint64_t ThreadTracer::getLastTotal() const
{
    if (samples_.empty())
        return 0;

    uint64_t total = 0;
    for (const auto bytes : samples_.back().second)
        total += bytes;
    return total;
}

double ThreadTracer::getTimeAtCumulative(const std::vector<double>& cumulative,
                                         double target) const
{
    if (target <= cumulative.front())
        return static_cast<double>(samples_.front().first);
    if (target >= cumulative.back())
        return static_cast<double>(samples_.back().first);

    size_t lo = 0;
    size_t hi = cumulative.size() - 1;
    while (lo < hi)
    {
        const size_t mid = (lo + hi) / 2;
        if (cumulative[mid] < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == 0)
        return static_cast<double>(samples_.front().first);

    const double span = cumulative[lo] - cumulative[lo - 1];
    const double prevMs = static_cast<double>(samples_[lo - 1].first);
    const double currMs = static_cast<double>(samples_[lo].first);
    if (span <= 0.0)
        return currMs;
    return prevMs + (target - cumulative[lo - 1]) / span * (currMs - prevMs);
}

void ThreadTracer::printProgress() const
{
    if (!g_progress_enabled || totalBytes_ == 0)
        return;

    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    char timeBuf[16];
    std::strftime(timeBuf, sizeof(timeBuf), "%H:%M:%S", std::localtime(&nowTime));

    const uint64_t doneBytes = getLastTotal();
    double percent = static_cast<double>(doneBytes) / static_cast<double>(totalBytes_) * 100.0;
    if (percent > 100.0)
        percent = 100.0;

    std::printf("  [%s] [%s] %.1f%% (%.2f / %.2f GiB)\n", timeBuf, mode_, percent,
                static_cast<double>(doneBytes) / BytesPerGiB,
                static_cast<double>(totalBytes_) / BytesPerGiB);
    std::fflush(stdout);
}

void ThreadTracer::writeTrace() const
{
    if (!g_thread_trace_dir || !*g_thread_trace_dir)
        return;

    char path[TracePathSize];
    std::snprintf(path, sizeof(path), "%s/thread_trace_%s.csv", g_thread_trace_dir, mode_);

    std::ofstream traceFile(path);
    if (!traceFile)
        return;

    traceFile << "# epoch_ms=" << epochMs_ << "\n";
    traceFile << "ms";
    for (int32_t i = 0; i < numThreads_; i++)
        traceFile << ",t" << i;
    traceFile << "\n";

    for (const auto& [timeMs, snapshot] : samples_)
    {
        traceFile << timeMs;
        for (const auto bytes : snapshot)
            traceFile << "," << bytes;
        traceFile << "\n";
    }
    traceFile.close();

    std::printf("Thread trace saved: %s (%zu rows)\n", path, samples_.size());
}

void ThreadTracer::sampleAt(std::chrono::steady_clock::time_point timePoint)
{
    const uint64_t timeMs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(timePoint - startTime_).count());

    std::vector<uint64_t> snapshot(static_cast<size_t>(numThreads_));
    for (int32_t i = 0; i < numThreads_; i++)
        snapshot[i] = snapshots_[i].bytesDone.load(std::memory_order_relaxed);

    while (!samples_.empty() && samples_.back().first >= timeMs)
        samples_.pop_back();

    samples_.emplace_back(timeMs, std::move(snapshot));
}

void ThreadTracer::pollLoop()
{
    if (g_progress_enabled && totalBytes_ > 0)
        std::printf("\n");

    auto lastPrint =
        std::chrono::steady_clock::now() - std::chrono::milliseconds(PrintIntervalMs);
    for (;;)
    {
        const auto now = std::chrono::steady_clock::now();
        sampleAt(now);
        if (now - lastPrint >= std::chrono::milliseconds(PrintIntervalMs))
        {
            printProgress();
            lastPrint = now;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        if (condition_.wait_for(lock, std::chrono::milliseconds(g_thread_trace_interval_ms),
                                [this]
                                { return stopRequested_.load(std::memory_order_relaxed); }))
            break;
    }
}
}
