#!/bin/bash

# This script is used to run the baseline transformer model and profilling the performance of the model

# Run the baseline transformer model
g++ -pg transformer.cpp -o transformer
./transformer

# Flat and Call Graph Profiling the performance of the model
gprof -b transformer gmon.out > analysis.txt

# Line by Line Profiling the performance of the model
gcov -b -c transformer.cpp > coverage.txt
cat transformer.cpp.gcov > code_coverage.txt

# Likwid Profiling the performance of the model
likwid-perfctr -C S0:0 -g L3 -m ./transformer
likwid-perfctr -C S0:0-15 -g L3 -m ./transformer
# memory 
likwid-perfctr -C S0:0 -g MEM -m ./transformer
# cache
likwid-perfctr -C S0:0 -g L2 -m ./transformer
# branch
likwid-perfctr -C S0:0 -g BRANCH -m ./transformer
# energy
likwid-perfctr -C S0:0 -g ENERGY -m ./transformer
# flops
likwid-perfctr -C S0:0 -g FLOPS_DP -m ./transformer