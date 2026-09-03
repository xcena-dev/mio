#ifndef OUTPUT_H
#define OUTPUT_H

#include <string>

void saveEnvironmentJson(const char* result_dir, const std::string& mode, size_t block_size,
                         size_t stride_size, bool bypass_cache, const char* devdax_path);

#endif
