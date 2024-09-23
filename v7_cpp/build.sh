#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py11

python setup.py build_ext --inplace

# Remove build directories and its subdirectories
rm -r build/*
rmdir build