#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]
then
    echo "No CUDA device number provided. Usage: ./run_script.sh <device_number>"
    exit 1
fi

# conda stuff
source /opt/anaconda/anaconda3/etc/profile.d/conda.sh 
conda activate /home/pfirouzabadi/.conda/envs/mmlab

# Get the CUDA device number from the first argument
CUDA_DEVICE_NUMBER=$1

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUMBER python populate_pipeline.py


conda deactivate

# conda activate /home/pfirouzabadi/.conda/envs/mujoco

# # Get the CUDA device number from the first argument
# CUDA_DEVICE_NUMBER=$1

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_NUMBER python populate_mjx.py


# conda deactivate