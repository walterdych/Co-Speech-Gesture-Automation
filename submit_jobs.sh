#!/bin/bash

# Specify your directories here
INPUT_DIR="VIDEOS"
OUTPUT_DIR="Motion Tracking Annotations"

# Update environment variables
export INPUT_DIR=${INPUT_DIR}
export OUTPUT_DIR=${OUTPUT_DIR}

# Activate your Python environment if needed
module load python/3.10

# Submit a job for each .mp4 or .MOV file
for filename in $(ls ${INPUT_DIR}/*.mp4 ${INPUT_DIR}/*.MOV); do
    sbatch --job-name=process_file --output=process_file.out --error=process_file.err --time=24:00:00 --mem=16G --ntasks=1 --cpus-per-task=4 --wrap="python3 Pointing_Task.py ${filename}"
done
