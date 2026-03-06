cmake_minimum_required(VERSION 3.11)
project(mio)

# Option to enable tracing
option(ENABLE_TRACING "Enable memory access tracing" OFF)

# Option to enable latency measurement
option(ENABLE_LATENCY_MEASURE "Enable per-access latency measurement" OFF)

set(src
    src/main.cpp
    src/benchmark.cpp
    src/hw_prefetcher.cpp
    src/output.cpp
    src/numa_affinity.cpp
)

# Add tracing source if enabled
if(ENABLE_TRACING)
    list(APPEND src src/tracing.cpp)
endif()

# Add latency measurement source if enabled
if(ENABLE_LATENCY_MEASURE)
    list(APPEND src src/latency_measure.cpp)
endif()

set(include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(static_libs
)

set(shared_libs
    numa
    pthread
)

add_compile_options(-mclflushopt)

# Create single executable from all source files
set(target microbench)
add_executable(${target} ${src})

target_compile_options(${target} PRIVATE -mavx2)

# Add ENABLE_TRACING definition if enabled
if(ENABLE_TRACING)
    target_compile_definitions(${target} PRIVATE ENABLE_TRACING)
    message(STATUS "Memory access tracing is ENABLED for ${target}")
else()
    message(STATUS "Memory access tracing is DISABLED for ${target}")
endif()

# Add ENABLE_LATENCY_MEASURE definition if enabled
if(ENABLE_LATENCY_MEASURE)
    target_compile_definitions(${target} PRIVATE ENABLE_LATENCY_MEASURE)
    message(STATUS "Latency measurement is ENABLED for ${target}")
else()
    message(STATUS "Latency measurement is DISABLED for ${target}")
endif()

target_include_directories(${target} PRIVATE ${include})
target_link_libraries(${target}
    -Wl,--whole-archive ${static_libs} -Wl,--no-whole-archive
    ${shared_libs}
)

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "  Target: ${target}")
message(STATUS "  Source: ${src}")
message(STATUS "  Include: ${include}")
message(STATUS "  Static Libs: ${static_libs}")
message(STATUS "  Shared Libs: ${shared_libs}")