#!/bin/bash
SIF_PROBLEMS="../Data/SIF/Problem_Classification/Filtered_SIF_Names.txt"
SIF_DATA="../Data/SIF/HS/"
FILEDIR=$(pwd)
cd ../Release/
for file in ${MASTSIF}/HS*.SIF; do
    if ! (grep -q "$(basename $file)" "$FILEDIR/SIF_log_traj_plot.txt"); then
        cmake .. -DSIF_PROBLEM="$(basename $file)" -DCMAKE_BUILD_TYPE=Release
        make
        python ../Plot/SIF/Trajectory_Plot.py
        echo "$(basename $file)" >> "$FILEDIR/SIF_log_traj_plot.txt"
    fi
done