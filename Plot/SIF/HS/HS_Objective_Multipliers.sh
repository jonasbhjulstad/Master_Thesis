#!/bin/bash

for file in ${MASTSIF}/HS*.SIF; do
    python ../Multiplier_Plot.py $(basename $file)
    echo $(basename $file)
    python ../Objective_Plot.py $(basename $file)
done