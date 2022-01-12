#!/bin/bash
SIF_DATA="../../../Data/SIF/HS"
FILEDIR=$(pwd)
cd ../../../Release/
make clean
for file in `ls -v1 ${MASTSIF}/HS*.SIF`; do
    if ! (grep -q "$(basename $file)" "$FILEDIR/SIF_log.txt"); then

        cmake .. -DSIF_PROBLEM="$(basename $file)" -DCMAKE_BUILD_TYPE=Release
	cmake --build ./test/SIF/ --target RUN_SIF
        make -j4
        mkdir -p "$SIF_DATA/$(basename $file)/" && touch "$SIF_DATA/$(basename $file)/timing_memoized.txt"
        python -m timeit "__import__('os').system('./test/SIF/Run_SIF')" > "$SIF_DATA/$(basename $file)/timing_memoized.txt"
        echo "$(basename $file)" >> "$FILEDIR/SIF_log.txt"
        python $FILEDIR/Plot_All.py
        rm -f fort*
    fi
done
