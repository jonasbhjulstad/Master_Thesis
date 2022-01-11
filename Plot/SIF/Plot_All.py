import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import ticker as mticker
import sys
import os
matplotlib.use
from Objective_Plot import Plot_Objective
from Trajectory_Plot import Plot_Trajectory
from Multiplier_Plot import Plot_Multipliers
import multiprocessing as mp

baseFolder = "/home/deb/Documents/FIPOPT/Data/SIF/"

def plot_single_SIF(fPath):
    Plot_Objective(fPath)
    Plot_Trajectory(fPath)
    Plot_Multipliers(fPath)

if __name__ == '__main__':
    plt.close('all')
    N_eq = 0
    N_ineq = 0
    SIF_folders = []


    with open(baseFolder + 'Problem/probname.txt', 'r') as file:
        probname = file.read()[:-1]

    plot_single_SIF(baseFolder + probname + "/")
    
    

    