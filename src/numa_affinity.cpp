#include "numa_affinity.h"
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <set>

std::vector<int> parseNumaNodes(const char* numa_node_str) {
    std::vector<int> numa_nodes;
    std::set<int> unique_nodes; // To avoid duplicates

    if (!numa_node_str) {
        return numa_nodes;
    }

    char* str_copy = strdup(numa_node_str);
    char* token = strtok(str_copy, ",");

    while (token) {
        // Trim whitespace
        while (*token == ' ' || *token == '\t') token++;

        char* end;
        long node = strtol(token, &end, 10);

        // Check for parsing errors
        if (end == token || *end != '\0') {
            fprintf(stderr, "Error: Invalid NUMA node format: '%s'\n", token);
            free(str_copy);
            return std::vector<int>(); // Return empty vector on error
        }

        if (node < 0) {
            fprintf(stderr, "Error: NUMA node cannot be negative: %ld\n", node);
            free(str_copy);
            return std::vector<int>();
        }

        unique_nodes.insert((int)node);
        token = strtok(NULL, ",");
    }

    free(str_copy);

    // Convert set to vector (maintains sorted order)
    numa_nodes.assign(unique_nodes.begin(), unique_nodes.end());
    return numa_nodes;
}

std::vector<int> getCpusFromNumaNodes(const std::vector<int>& numa_nodes) {
    std::vector<int> cpu_list;

    if (numa_available() < 0) {
        fprintf(stderr, "Error: NUMA is not available on this system\n");
        return cpu_list;
    }

    struct bitmask* cpumask = numa_allocate_cpumask();

    for (int node : numa_nodes) {
        // Clear the mask
        numa_bitmask_clearall(cpumask);

        // Get CPUs for this NUMA node
        if (numa_node_to_cpus(node, cpumask) < 0) {
            fprintf(stderr, "Warning: Failed to get CPUs for NUMA node %d\n", node);
            continue;
        }

        // Add all set bits (CPUs) to our list
        for (unsigned int cpu = 0; cpu < cpumask->size; cpu++) {
            if (numa_bitmask_isbitset(cpumask, cpu)) {
                cpu_list.push_back(cpu);
            }
        }
    }

    numa_free_cpumask(cpumask);
    return cpu_list;
}

int setThreadCpuAffinity(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int ret = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

    if (ret != 0) {
        fprintf(stderr, "Error: Failed to set CPU affinity to CPU %d: %s\n",
                cpu_id, strerror(ret));
        return -1;
    }

    return 0;
}

bool validateNumaNodes(const std::vector<int>& numa_nodes) {
    if (numa_nodes.empty()) {
        fprintf(stderr, "Error: No NUMA nodes specified\n");
        return false;
    }

    if (numa_available() < 0) {
        fprintf(stderr, "Error: NUMA is not available on this system\n");
        return false;
    }

    int max_node = numa_max_node();

    for (int node : numa_nodes) {
        // Check if node exists
        if (node > max_node) {
            fprintf(stderr, "Error: NUMA node %d does not exist (max: %d)\n",
                    node, max_node);
            return false;
        }

        // Check if node has CPUs
        struct bitmask* cpumask = numa_allocate_cpumask();
        numa_bitmask_clearall(cpumask);

        if (numa_node_to_cpus(node, cpumask) < 0) {
            fprintf(stderr, "Error: Failed to query CPUs for NUMA node %d\n", node);
            numa_free_cpumask(cpumask);
            return false;
        }

        // Check if any CPU is set
        bool has_cpus = false;
        for (unsigned int cpu = 0; cpu < cpumask->size; cpu++) {
            if (numa_bitmask_isbitset(cpumask, cpu)) {
                has_cpus = true;
                break;
            }
        }

        numa_free_cpumask(cpumask);

        if (!has_cpus) {
            fprintf(stderr, "Error: NUMA node %d has no CPUs\n", node);
            return false;
        }
    }

    return true;
}
