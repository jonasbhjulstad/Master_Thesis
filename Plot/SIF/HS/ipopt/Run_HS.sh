#!/bin/bash
SIF_PROBLEMS="../Data/SIF/Problem_Classification/Filtered_SIF_Names.txt"
FILEDIR=$(pwd)
# rm -f SIF_log.txt
# touch SIF_log.txt
# for file in $(cat < "$SIF_PROBLEMS"); do
for file in ${MASTSIF}/HS*.SIF; do
    echo "output_file $(basename $file).txt" > ipopt.opt
    echo file_print_level 8 >> ipopt.opt
    runcutest -D "$(basename $file)" -p ipopt
done
# while read file; do
#     cmake .. -DSIF_PROBLEM="$file"
#     make
#     ./test/SIF/SIF_Constrained
#     # echo $file >> "$FILEDIR/SIF_log.txt"
# done < "../Plot/SIF/failed.txt"