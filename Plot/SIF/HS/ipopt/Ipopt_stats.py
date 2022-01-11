import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import ticker as mticker
import sys
import os

baseFolder = "/home/build/FIPOPT/Data/SIF/ipopt/"
rootFolder = "/home/build/FIPOPT/"
sys.path.append(rootFolder + "build/test/Plot/")

figFolder = "/home/build/MT/figures/"

def ipopt_stats():

    N_eq = 0
    N_ineq = 0
    SIF_files = []
    for dirname in os.listdir(baseFolder):
        if dirname.endswith('.SIF.txt'):
            SIF_files.append(dirname)
    N_converged = 0

    SIF_stats = {"N_problems": len(SIF_files)}
    for res in SIF_files:
        SIF_res = {"name": res, "converged": False}
        with open(baseFolder + res, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("Number of Iterations"):
                    SIF_res["N_iter"] = [int(i) for i in line.split() if i.isdigit()][0]
                if line.startswith("EXIT: Optimal Solution"):
                    SIF_res["converged"] = True
            N_converged += 1
        SIF_stats["N_converged"] = N_converged

        SIF_stats[res[:-4]] = SIF_res
    return SIF_stats





    
