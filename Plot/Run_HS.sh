#!/bin/bash
SIF_DATA="../Data/SIF"
FILEDIR=$(pwd)
cd ../Release/
for file in ${MASTSIF}/HS*.SIF; do
    if ! (grep -q "$(basename $file)" "$FILEDIR/SIF_log.txt"); then

        cmake .. -DSIF_PROBLEM="$(basename $file)" -DCMAKE_BUILD_TYPE=Release
        make
        mkdir -p "$SIF_DATA/$(basename $file)/" && touch "$SIF_DATA/$(basename $file)/timing_memoized.txt"
        python -m timeit "__import__('os').system('./test/SIF/Run_SIF')" > "$SIF_DATA/$(basename $file)/timing_memoized.txt"
        echo "$(basename $file)" >> "$FILEDIR/SIF_log.txt"
        python ../Plot/SIF/Plot_All.py
        rm -f fort*
    fi
done
