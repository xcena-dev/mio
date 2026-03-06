#include "output.h"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <thread>
#include <unistd.h>

using namespace std;

void saveEnvironmentJson(const char* result_dir, const string& mode, size_t block_size,
                         size_t stride_size, bool bypass_cache, const char* devdax_path) {
    char filepath[1024];
    snprintf(filepath, sizeof(filepath), "%s/environment.json", result_dir);

    FILE* fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "Warning: Failed to create environment.json\n");
        return;
    }

    // Get system information
    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname));

    char kernel[256] = "unknown";
    FILE* uname_fp = popen("uname -r", "r");
    if (uname_fp) {
        if (fgets(kernel, sizeof(kernel), uname_fp)) {
            // Remove newline
            kernel[strcspn(kernel, "\n")] = 0;
        }
        pclose(uname_fp);
    }

    char cpu_model[512] = "unknown";
    FILE* cpu_fp = popen("lscpu | grep 'Model name' | cut -d':' -f2 | xargs", "r");
    if (cpu_fp) {
        if (fgets(cpu_model, sizeof(cpu_model), cpu_fp)) {
            cpu_model[strcspn(cpu_model, "\n")] = 0;
        }
        pclose(cpu_fp);
    }

    int cpu_logical_cores = thread::hardware_concurrency();

    size_t memory_total_kb = 0;
    FILE* mem_fp = fopen("/proc/meminfo", "r");
    if (mem_fp) {
        char line[256];
        while (fgets(line, sizeof(line), mem_fp)) {
            if (sscanf(line, "MemTotal: %zu", &memory_total_kb) == 1) {
                break;
            }
        }
        fclose(mem_fp);
    }

    // Get current timestamp
    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", localtime(&now));

    // Write JSON
    fprintf(fp, "{\n");
    fprintf(fp, "  \"timestamp\": \"%s\",\n", timestamp);
    fprintf(fp, "  \"mode\": \"%s\",\n", mode.c_str());
    fprintf(fp, "  \"block_size\": %zu,\n", block_size);
    if (stride_size > 0) {
        fprintf(fp, "  \"stride_size\": %zu,\n", stride_size);
    } else {
        fprintf(fp, "  \"stride_size\": null,\n");
    }
    fprintf(fp, "  \"bypass_cache\": %s,\n", bypass_cache ? "true" : "false");
    fprintf(fp, "  \"memory_type\": \"%s\",\n", devdax_path ? "devdax" : "numa");
    if (devdax_path) {
        fprintf(fp, "  \"devdax_device\": \"%s\",\n", devdax_path);
    } else {
        fprintf(fp, "  \"devdax_device\": null,\n");
    }
    fprintf(fp, "  \"system_info\": {\n");
    fprintf(fp, "    \"hostname\": \"%s\",\n", hostname);
    fprintf(fp, "    \"kernel\": \"%s\",\n", kernel);
    fprintf(fp, "    \"cpu_model\": \"%s\",\n", cpu_model);
    fprintf(fp, "    \"cpu_logical_cores\": %d,\n", cpu_logical_cores);
    fprintf(fp, "    \"memory_total_kb\": %zu\n", memory_total_kb);
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    fclose(fp);
    printf("Environment configuration saved to %s\n", filepath);
}
