#!/bin/bash

for file in ${MASTSIF}/HS*.SIF; do
    # python ./SIF/Multiplier_Plot.py $(basename $file)
    echo $(basename $file)
    python ./SIF/Objective_Plot.py $(basename $file)
done