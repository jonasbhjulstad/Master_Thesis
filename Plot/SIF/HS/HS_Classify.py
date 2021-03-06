import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import ticker as mticker
import sys
import os
import re
from collections import Counter
matplotlib.use('TkAgg')
sys.path.append('/home/build/FIPOPT/Data/SIF/ipopt/')


# from ipopt_HS_stat import IPOPT_stats
baseFolder = "/home/build/FIPOPT/Data/SIF/HS/"
pFolder = baseFolder + "Problem/"
dimFolder = "/home/build/FIPOPT/include/SIF_Dimensions/Dimensions.csv"
SIF_Folder = "/home/deb/Downloads/cutest/sifdecode/sif/"
Mastsif = "/home/deb/Downloads/cutest/sifdecode/sif/"

def fix_dim_vec(mat):
    if len(mat) == 0:
        return np.ndarray(shape=(0,1), dtype=np.float64)
    else:
        return mat

def fix_dim_mat(mat):
    if len(mat) == 0:
        return np.ndarray(shape=(0,2), dtype=np.float64)
    else:
        return mat
def count_subproblem_iter(fPath):
    N_iter = 0
    for subdirname in os.listdir(fPath):
        if subdirname.startswith("mu_"):
            N_iter += sum(1 for line in open(fPath + "/" + subdirname + "/" + "x.csv"))
    return N_iter

rootFolder = "/home/build/FIPOPT/"
sys.path.append(rootFolder + "build/test/Plot/")

figFolder = "/home/build/FIPOPT/figures/"
outsdif = pFolder + "OUTSDIF.d"

def read_timing(fpath):
    with open(fpath, 'r') as file:      
        tline = file.readlines()[-1]
        num = float(re.findall(r'\d+', tline)[-1])
        if 'msec' in tline:
            num = num*1e-3
    return num

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

    def get_dim(file):
        data = np.genfromtxt(file)
        if data != []:
            return data.shape[-1]
        else:
            return 0
    not_converged = []
    infnorms = []
    objtypes= {'Converged': [], 'almost_converged': [], 'not_converged': []}
    contypes = {'Converged': [], 'almost_converged': [], 'not_converged': []}
    smoothnesses = {'Converged': [], 'almost_converged': [], 'not_converged': []}
    objcon = {'Converged': [], 'almost_converged': [], 'not_converged': []}
    derivative_degrees = {'Converged': [], 'almost_converged': [], 'not_converged': []}
    classification_str = ""
    Lnames = []
    for dirname in SIF_folders:
        probstats = {"N_ineq": 0, "N_eq": 0, "Converged": False, "almost_converged": False}
        probstats['Nx'] = get_dim(baseFolder + dirname + "/x.csv")
        probstats['N_lbd'] = get_dim(baseFolder + dirname + "/lbd.csv")
        probstats['Nz'] = get_dim(baseFolder + dirname + "/z.csv")
        # probstats['time'] = read_timing(baseFolder + dirname + "/timing.txt")
        # probstats['time_memoized'] = read_timing(baseFolder + dirname + "/timing_memoized.txt")
        # probstats['time_memoized'] = read_timing(baseFolder + dirname + "/timing_memoized.txt")
        almost_converged = False
        mu_small = False
        for subdirname in os.listdir(baseFolder + dirname + "/"):
            if subdirname.startswith("Inequality_Restoration"): 
                probstats["N_ineq"] += 1
            elif subdirname.startswith("Equality_Restoration"): 
                probstats["N_eq"] += 1
            if subdirname.startswith("mu_0.000000"):
                mu_small = True
        
        if not almost_converged:
            not_converged.append(dirname)

        with open(baseFolder + dirname + "/success.txt", 'r') as file:
            if int(file.readline()) == 1:
                probstats['Converged'] = True
                N_converged +=1
            else:
                if almost_converged:
                    print(dirname)
        obj = np.genfromtxt(baseFolder + dirname + '/obj.csv', delimiter=', ')
        if len(obj.shape) > 1:
            probstats['obj'] = obj[-1,0]
        elif obj.shape == 1:
            probstats['obj'] = obj[-1]
        else:
            probstats['obj'] = np.inf
        x_ipopt_sol = []
        with open(baseFolder + 'ipopt/' + dirname + '.txt', 'r') as file:
            for line in file.readlines():
                if line.startswith('final x unscaled'):
                    x_ipopt_sol.append(float(line.split('=')[-1]))
        x_sol = []
        if (not probstats['Converged']) and mu_small: 
            with open(baseFolder + dirname + '/mu_0.000000/x.csv', 'r') as file:
                x_sol = np.fromstring(file.readlines()[-1], sep=', ')
                inf_dist = np.linalg.norm(x_sol - np.array(x_ipopt_sol), np.inf)
                print(inf_dist)
                infnorms.append(inf_dist)
                if inf_dist < 1e-3:
                    probstats['almost_converged'] = True
                    N_almost_converged += 1
        # print(inf_dist)
        with open(SIF_Folder + dirname, 'r') as file:
            for line in file.readlines():
                if 'SOLTN' in line:
                    probstats['SOLTN'] = float(re.findall(r'\d+', line)[-1])
                    # print('diff:' , abs(probstats['obj'] - probstats['SOLTN']))
                    # if abs(probstats['obj'] - probstats['SOLTN']) < 10:
                    #     if not (probstats['Converged']):
                            # probstats['almost_converged'] = True
                            # N_almost_converged += 1

        probstats["N_iter"] = count_subproblem_iter(baseFolder + dirname + "/")
        probstats["name"] = dirname
        SIF_stats[dirname] = probstats
        classification_list = [[], [], [], []]
        with open(Mastsif + dirname, 'r') as file:
            for line in file.readlines():
                if 'classification' in line:
                    cgroups = line.split('classification ')[-1].split('-')[0]
                    key = []
                    if probstats['Converged']:
                        objtypes['Converged'].append(cgroups[0])
                        contypes['Converged'].append(cgroups[1])
                        objcon['Converged'].append(cgroups[:2])
                        smoothnesses['Converged'].append(cgroups[2])
                        derivative_degrees['Converged'].append(cgroups[3])
                    elif probstats['almost_converged']:
                        objtypes['almost_converged'].append(cgroups[0])
                        contypes['almost_converged'].append(cgroups[1])
                        objcon['almost_converged'].append(cgroups[:2])
                        smoothnesses['almost_converged'].append(cgroups[2])
                        derivative_degrees['almost_converged'].append(cgroups[3])
                    else:
                        # if cgroups[0] == 'L':
                            # Lnames.append(dirname)
                        objtypes['not_converged'].append(cgroups[0])
                        contypes['not_converged'].append(cgroups[1])
                        objcon['not_converged'].append(cgroups[:2])
                        smoothnesses['not_converged'].append(cgroups[2])
                        derivative_degrees['not_converged'].append(cgroups[3])


    objCount = [Counter(x) for x in objtypes.values()]
    conCount = [Counter(x) for x in contypes.values()]
    objconCount = [Counter(x) for x in objcon.values()]
    smoothCount = [Counter(x) for x in smoothnesses.values()]
    derdegCount = [Counter(x) for x in derivative_degrees.values()]

    plt.close('all')
    fig, ax = plt.subplots(2)

    _ = [x.grid() for x in ax]
    _ = [x.set_axisbelow(True) for x in ax]
    ax[0].set_title('Object classifications')
    ax[1].set_title('Constraint classifications')
    # ax[1][0].set_title('Smoothness')
    # ax[1][1].set_title('Highest derivative degree')
    objvals = []
    ind = np.arange(3)

    objvals.append([list(x.values()) for x in objCount])

    ax[0].bar(objCount[0].keys(), objCount[0].values(), color='gray', fill='false', hatch='..', alpha=0.4, label='Converged')
    ax[0].bar(objCount[1].keys(), objCount[1].values(), bottom=[objCount[0][key] for key in objCount[1].keys()], fill='false', hatch='///', color='gray', alpha=1., label='Almost converged')
    ax[0].bar(objCount[2].keys(), objCount[2].values(), bottom=[objCount[0][key] + objCount[1][key] for key in objCount[2].keys()], fill='false', hatch='xx', color='gray', alpha=.4, label='Not converged')

    ax[1].bar(conCount[0].keys(), conCount[0].values(), color='gray', fill='false', hatch='..', alpha=0.4, label='Converged')
    ax[1].bar(conCount[1].keys(), conCount[1].values(), bottom=[conCount[0][key] for key in conCount[1].keys()], fill='false', hatch='///', color='gray', alpha=1., label='Almost converged')
    ax[1].bar(conCount[2].keys(), conCount[2].values(), bottom=[conCount[0][key] + conCount[1][key] for key in conCount[2].keys()], fill='false', hatch='xx', color='gray', alpha=.4, label='Not converged')

    ax[0].legend()
    print(N_converged, N_almost_converged)
    fig.subplots_adjust(hspace=0.3)
    # plt.show()
    fig.savefig(figFolder + 'HS_classification.pdf')
    plt.close('all')
    # ax[1][0].bar(smoothCount.keys(), smoothCount.values(), color='gray')
    # ax[1][1].bar(derdegCount.keys(), derdegCount.values(), color='gray')
    # plt.show()
    fig, ax = plt.subplots()

    ax.bar(objconCount[0].keys(), objconCount[0].values(), color='gray', fill='false', hatch='..', alpha=0.4, label='Converged')
    ax.bar(objconCount[1].keys(), objconCount[1].values(), bottom=[objconCount[0][key] for key in objconCount[1].keys()], fill='false', hatch='///', color='gray', alpha=1., label='Almost converged')
    ax.bar(objconCount[2].keys(), objconCount[2].values(), bottom=[objconCount[0][key] + objconCount[1][key] for key in objconCount[2].keys()], fill='false', hatch='xx', color='gray', alpha=.4, label='Not converged')
    ax.legend()
    ax.grid()
    fig.savefig(figFolder + 'HS_double_classification.pdf')
    plt.show()

    SIF_list = [value for key, value in SIF_stats.items() if ('SIF' in key) and value["Converged"]]


    SIF_list.sort(key=lambda x: x["N_iter"])


    print(Lnames)
    # ipopt_stats = IPOPT_stats()
    # ipopt_list = [ipopt_stats[s['name']] for s in SIF_list]
    fig, ax = plt.subplots()
    ax.plot([x['N_iter']  for x in SIF_list], color='k', label='Thesis implementation')
    # ax.plot([x['N_iter'] for x in ipopt_list], color='k', linestyle='dashed', label='IPOPT')
    ax.grid()
    print(N_converged)
    # ax.set_yscale('log')
    ax.set_ylabel('Subproblem iterations')
    ax.set_xlabel('Converged problems in subproblem iteration-ascending order')
    ax.legend()
    # plt.show()
    # fig.savefig(figFolder +"HS_iters.png", format='png')


