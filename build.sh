#!/bin/bash

# Set ENABLE_TRACING to ON/OFF (default: OFF)
ENABLE_TRACING=${ENABLE_TRACING:-OFF}

# Set ENABLE_LATENCY_MEASURE to ON/OFF (default: OFF)
ENABLE_LATENCY_MEASURE=${ENABLE_LATENCY_MEASURE:-OFF}

# Install pre-requisites
sudo apt-get install cmake g++ gnuplot msr-tools numactl libnuma-dev

# Load MSR module
sudo modprobe msr

rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_TRACING=${ENABLE_TRACING} -DENABLE_LATENCY_MEASURE=${ENABLE_LATENCY_MEASURE}

make -j 4
cd -
