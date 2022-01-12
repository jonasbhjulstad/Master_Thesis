import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import ticker as mticker
import sys
matplotlib.use
sys.path.append("/home/deb/Documents/gitFIPOPT/Plot/SIF/")
from Objective_Plot import Plot_Objective
from Trajectory_Plot import Plot_Trajectory
from Multiplier_Plot import Plot_Multipliers
baseFolder = "/home/deb/Documents/gitFIPOPT/Data/SIF"
HS_Folder = baseFolder + "/HS"

def plot_single_SIF(fPath):
    Plot_Objective(fPath)
    Plot_Trajectory(fPath)
    Plot_Multipliers(fPath)

if __name__ == '__main__':
    plt.close('all')
    N_eq = 0
    N_ineq = 0
    SIF_folders = []


    with open(baseFolder + '/Problem/probname.txt', 'r') as file:
        probname = file.read()[:-1]

    plot_single_SIF(HS_Folder + "/" + probname + "/")
    
    

    