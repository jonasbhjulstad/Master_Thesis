#!/bin/bash
SIF_PROBLEMS="../Data/SIF/Problem_Classification/Filtered_SIF_Names.txt"
SIF_DATA="../Data/SIF/"
FILEDIR=$(pwd)
cd ../Release/
for file in ${MASTSIF}/HS*.SIF; do
    if ! (grep -q "$(basename $file)" "$FILEDIR/SIF_log_plot.txt"); then
        cmake .. -DSIF_PROBLEM="$(basename $file)" -DCMAKE_BUILD_TYPE=Release
        make
        echo "$(basename $file)" >> "$FILEDIR/SIF_log_plot.txt"
        python ../Plot/SIF/Plot_All.py
    fi
done