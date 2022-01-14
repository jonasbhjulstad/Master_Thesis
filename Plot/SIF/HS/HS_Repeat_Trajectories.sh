#!/bin/bash
SIF_PROBLEMS="../Data/SIF/Problem_Classification/Filtered_SIF_Names.txt"
SIF_DATA="../Data/SIF/HS/"
FILEDIR=$(pwd)
cd ../Release/
for file in `cat SIF_traj_log.txt`; do
    echo $file
    cmake .. -DSIF_PROBLEM="$(basename $file)" -DCMAKE_BUILD_TYPE=Release
    make
    python ../Plot/SIF/Trajectory_Plot.py
done