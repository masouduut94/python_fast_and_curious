#!/bin/bash

# Ensure the script stops if any command fails
set -e

# Activate the Conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py11

# Build the Cython file
python setup.py build_ext --inplace

# (Optional) Deactivate the Conda environment after the build
conda deactivate

# Remove build directories and its subdirectories
rm -r build/*
rmdir build
rm evaluator.c