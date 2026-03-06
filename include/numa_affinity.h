#ifndef NUMA_AFFINITY_H
#define NUMA_AFFINITY_H

#include <vector>
#include <string>

// Parse comma-separated NUMA node list (e.g., "0,2,4" -> [0, 2, 4])
std::vector<int> parseNumaNodes(const char* numa_node_str);

// Get list of CPUs from specified NUMA nodes
// Returns a vector of CPU IDs that belong to any of the specified nodes
std::vector<int> getCpusFromNumaNodes(const std::vector<int>& numa_nodes);

// Set CPU affinity for the current thread to a specific CPU
// Returns 0 on success, -1 on failure
int setThreadCpuAffinity(int cpu_id);

// Validate that all specified NUMA nodes exist and have CPUs
// Returns true if valid, false otherwise (prints error message)
bool validateNumaNodes(const std::vector<int>& numa_nodes);

#endif // NUMA_AFFINITY_H
