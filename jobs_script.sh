#!/bin/bash
#SBATCH --job-name=video_processing
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Load necessary modules (python, etc)
# module load ...

# Loop over all your video files
for video in  $VIDEOS/*.mp4; do
    srun python process_video.py $video &
done

# Wait for all jobs to finish
wait