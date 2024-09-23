#!/bin/bash

# Ensure the script stops if any command fails
set -e

export CC=clang
export CXX=clang++

clang++ -v -E -x c++ - < /dev/null

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py11
python setup.py build
python setup.py build_ext --inplace

rm -r build/*
rmdir build
