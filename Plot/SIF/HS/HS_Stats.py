import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import ticker as mticker
import sys
import os
import re
matplotlib.use
sys.path.append('//home/deb/Documents/FIPOPT/Data/SIF/ipopt/')
sys.path.append('./ipopt')

baseFolder = "//home/deb/Documents/FIPOPT/Data/SIF/HS/"
pFolder = baseFolder + "Problem/"
SIF_Folder = "/home/deb/Downloads/cutest/sifdecode/sif/"
ipopt_Folder = "./ipopt/"

def ipopt_stats():
    
    N_eq = 0
    N_ineq = 0
    SIF_files = []
    for dirname in os.listdir(ipopt_Folder):
        if dirname.endswith('.SIF.txt'):
            SIF_files.append(dirname)
    N_converged = 0

    SIF_stats = {"N_problems": len(SIF_files)}
    for res in SIF_files:
        SIF_res = {"name": res, "converged": False}
        with open(ipopt_Folder + res, 'r') as file:
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


def count_subproblem_iter(fPath):
    N_iter = 0
    for subdirname in os.listdir(fPath):
        if subdirname.startswith("mu_"):
            N_iter += sum(1 for line in open(fPath + "/" +
                          subdirname + "/" + "x.csv"))
    return N_iter


rootFolder = "//home/build/FIPOPT/"
sys.path.append(rootFolder + "build/test/Plot/")

figFolder = "//home/deb/Documents/MT/figures/"


def read_timing(fpath):
    with open(fpath, 'r') as file:
        tline = file.readlines()[-1]
        num = float(re.findall(r'\d+', tline)[-1])
        if 'msec' in tline:
            num = num*1e-3
    return num


def read_ipopt_x_sol(dirname):
    with open(baseFolder + 'ipopt/' + dirname + '.txt', 'r') as file:
        x = []
        for line in file.readlines():
            xn = []
            if line.startswith('final x unscaled'):
                xn = float(line.split('=')[-1])
                str = line.split('=')
                x.append(xn)
        return np.array(x)

def read_ipopt_x_sol(dirname):
    with open('ipopt/' + dirname + '.txt', 'r') as file:
        x = []
        for line in file.readlines():
            xn = []
            if line.startswith('final x unscaled'):
                xn = float(line.split('=')[-1])
                str = line.split('=')
                x.append(xn)
        return np.array(x)

def is_almost_converged(dirname):
    with open(baseFolder + dirname + '/mu_0.000000/x.csv', 'r') as file:
        x_sol = np.fromstring(file.readlines()[-1], sep=', ')
        x_ipopt_sol = read_ipopt_x_sol(dirname)
        inf_dist = np.linalg.norm(x_sol - x_ipopt_sol, np.inf)
        if inf_dist < 1e-3:
            return True
        return False




def read_obj(dirname):
    obj = np.genfromtxt(baseFolder + dirname + '/obj.csv', delimiter=', ')
    if len(obj.shape) > 1:
        res = obj[-1, 0]
    elif obj.shape == 1:
        res = obj[-1]
    else:
        res = np.inf
    return res


def read_SIF_SOLTN(dirname):
    with open('ipopt/' + dirname + ".txt", 'r') as file:
        for line in file.readlines():
            if 'SOLTN' in line:
                return float(re.findall(r'\d+', line)[-1])


def is_converged(dirname):
    with open(baseFolder + dirname + "/success.txt", 'r') as file:
        if int(file.readline()) == 1:
            return True


def get_dim(file):
    data = np.genfromtxt(file)
    if data.size == 0:
        return 0
    else:
        return data.shape[-1]

if __name__ == '__main__':

    N_eq = 0
    N_ineq = 0
    SIF_folders = []
    for dirname in os.listdir(baseFolder):
        if dirname.endswith('.SIF') and dirname.startswith('HS'):
            SIF_folders.append(dirname)
    N_converged = 0
    N_almost_converged = 0

    SIF_stats = {"N_converged": 0, "N_problems": len(SIF_folders)}

    not_converged = []
    for dirname in SIF_folders:
        probstats = {"N_ineq": 0, "N_eq": 0,
                     "Converged": False, "almost_converged": False}
        probstats['Nx'] = get_dim(baseFolder + dirname + "/x.csv")
        probstats['N_lbd'] = get_dim(baseFolder + dirname + "/lbd.csv")
        probstats['Nz'] = get_dim(baseFolder + dirname + "/z.csv")
        # probstats['time'] = read_timing(baseFolder + dirname + "/timing.txt")
        probstats['time_memoized'] = read_timing(baseFolder + dirname + "/timing_memoized.txt")
        # probstats['time_memoized'] = read_timing(baseFolder + dirname + "/timing_memoized.txt")
        mu_small = False
        for subdirname in os.listdir(baseFolder + dirname + "/"):
            if subdirname.startswith("Inequality_Restoration"):
                probstats["N_ineq"] += 1
            elif subdirname.startswith("Equality_Restoration"):
                probstats["N_eq"] += 1
            if subdirname.startswith("mu_0.000000"):
                mu_small = True

        if (is_converged(dirname)):
            N_converged += 1
            probstats['Converged'] = True
        probstats['obj'] = read_obj(dirname)
        probstats['SOLTN'] = read_SIF_SOLTN(dirname)


        if mu_small and (not probstats['Converged']) and is_almost_converged(dirname):
            probstats['almost_converged'] = True
            N_almost_converged += 1

        probstats["N_iter"] = count_subproblem_iter(baseFolder + dirname + "/")
        probstats["name"] = dirname
        SIF_stats[dirname] = probstats

    SIF_list = [value for key, value in SIF_stats.items() if (
        'SIF' in key) and (value["Converged"])]
    ipopt_dict = ipopt_stats()
    ipopt_list = [ipopt_dict[x['name']] for x in SIF_list]

    SIF_list.sort(key=lambda x: x["N_iter"])

    fig, ax = plt.subplots()
    ax.plot([x['N_iter'] for x in SIF_list],
            color='k', label='Thesis implementation')
    ax.plot([x['N_iter'] for x in ipopt_list], color='k', linestyle='dashed', label='IPOPT')
    ax.grid()
    print(N_converged)
    ax.set_yscale('log')
    ax.set_ylabel('Subproblem iterations')
    ax.set_xlabel('Converged problems in subproblem iteration-ascending order')
    ax.legend()
    # plt.show()
    # fig.savefig(figFolder +"HS_iters.png", format='png')

    # SIF_list.sort(key=lambda x: x['time'])
    fig2, ax = plt.subplots(2)
    ax[0].plot([x['time']  for x in SIF_list], color='k', label='Thesis implementation')
    # ax[0].plot([x['time_memoized']  for x in SIF_list], color='k',linestyle='dotted', label='Thesis implementation memoized')
    ax[0].plot([x['time'] for x in ipopt_list], color='k', linestyle='dashed', label='IPOPT')
    # ax[0].set_yscale('log')
    ax[0].grid()
    ax[1].plot([x['Nx'] for x in SIF_list], color='k', label=r'$N_x$')
    ax[1].plot([x['N_lbd'] for x in SIF_list], color='k',
               linestyle='dashed', label=r'$N_\lambda$')
    ax[1].plot([x['Nz'] for x in SIF_list], color='k',
               linestyle='dotted', label=r'$N_z$')
    ax[1].grid()
    _ = [x.legend() for x in ax]
    ax[0].set_ylabel('Solve time[s]')
    ax[1].set_ylabel('Problem dimensions')
    ax[1].set_xlabel('Converged problems in solve-time ascending order')
    # plt.show()
    # fig2.savefig(figFolder +"HS_timings.png", format='png')
    # plt.show()
    # ax.set_yscale('log')
    a = 1
    N_ineq_conv, N_ineq_almost_conv, N_ineq_nconv = 0, 0, 0
    N_eq_conv, N_eq_almost_conv, N_eq_nconv = 0, 0, 0

    for stat in list(SIF_stats.values())[2:]:
        if (stat['N_ineq'] > 0):
            if stat['Converged']: 
                N_ineq_conv += 1
            else:
                N_ineq_nconv += 1
            if stat['almost_converged']:
                print(stat['name'])
                N_ineq_almost_conv += 1

        if (stat['N_eq'] > 0):
            if stat['Converged']: 
                N_eq_conv += 1
            else:
                N_eq_nconv += 1
            if stat['almost_converged']:
                N_eq_almost_conv += 1

    print(N_ineq_conv, N_ineq_almost_conv, N_ineq_nconv,
          N_eq_conv, N_eq_almost_conv, N_eq_nconv)
    print("Converged: ", N_converged, ",Almost converged:",
          N_almost_converged, "Not converged: ", len(not_converged))
