#ifndef NUMA_AFFINITY_H
#define NUMA_AFFINITY_H

#include <vector>
#include <string>

std::vector<int> parseNumaNodes(const char* numa_node_str);

std::vector<int> getCpusFromNumaNodes(const std::vector<int>& numa_nodes);

int setThreadCpuAffinity(int cpu_id);

bool validateNumaNodes(const std::vector<int>& numa_nodes);

#endif
