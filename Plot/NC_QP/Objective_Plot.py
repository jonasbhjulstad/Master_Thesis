
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import ticker as mticker
import sys
import os
matplotlib.use
sys.path.append("/home/deb/Documents/FIPOPT/Release/Plot/SIF/")

# pFolder = "/home/deb/Documents/FIPOPT/Data/QP/Trajectory/"
pFolder = "/home/deb/Documents/FIPOPT/Data/NC_QP/SOC/Trajectory/"
pFolder2 = "/home/deb/Documents/FIPOPT/Data/NC_QP/NONSOC/Trajectory/"

from Binder_SIF import *


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
def load_QP_params():

    Q = np.genfromtxt(pFolder + "Q.csv", delimiter=",")
    c = np.genfromtxt(pFolder + "c.csv", delimiter=",").reshape((-1,1))
    Nx = c.shape[0]
    A = np.genfromtxt(pFolder + "A.csv", delimiter=",").reshape((-1,Nx))
    b = np.genfromtxt(pFolder + "b.csv", delimiter=",").reshape((-1,1))
    D = np.genfromtxt(pFolder + "D.csv", delimiter=",").reshape((-1,Nx))
    e = np.genfromtxt(pFolder + "e.csv", delimiter=",").reshape((-1,1))
    x_lb = np.genfromtxt(pFolder + "x_lb.csv", delimiter=",")
    x_ub = np.genfromtxt(pFolder + "x_ub.csv", delimiter=",")
    x0 = np.genfromtxt(pFolder + "x0.csv", delimiter=",").reshape((-1,1))
    # x_traj = np.genfromtxt(pFolder + "x_iter.csv", delimiter=",")
    # z_traj = np.genfromtxt(pFolder + "z_iter.csv", delimiter=",")
    return Q, fix_dim_vec(c), fix_dim_mat(A), fix_dim_vec(b), fix_dim_mat(D), fix_dim_vec(e), x_lb, x_ub, x0



rootFolder = "/home/deb/Documents/FIPOPT/"
figFolder = "/home/deb/Documents/MT/figures/"
sys.path.append(rootFolder + "build/test/Plot/")


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def read_x_traj(objective_path):
    mus = np.genfromtxt(objective_path + "mu.csv")
    xk_traj = []
    for mu in mus:
        xk_traj.append(np.genfromtxt(
                objective_path + "mu_{m:.6f}/x.csv".format(m=mu), delimiter=", ")[:2].reshape((-1,2)))
    return xk_traj

def read_f_obj(objective_path):
    mus = np.genfromtxt(objective_path + "mu.csv")
    f_obj = []
    for mu in mus:
        f_obj.append(np.genfromtxt(
                objective_path + "mu_{m:.6f}/obj.csv".format(m=mu), delimiter=", ").reshape((-1,2)))
    return f_obj

def read_theta(objective_path):
    mus = np.genfromtxt(objective_path + "mu.csv")
    theta = []
    for mu in mus:
        theta.append(np.genfromtxt(
                objective_path + "mu_{m:.6f}/theta.csv".format(m=mu), delimiter=", ").reshape((-1,1)))
    return theta

if __name__ == '__main__':

    mu = 1
    mu_list = np.genfromtxt(pFolder + "mu.csv")

    objvals = np.genfromtxt(pFolder + "obj.csv", delimiter=", ");
    # f = load_SIF_2_1_0(pFolder + "OUTSDIF.d")
    # phis = [load_barrier_SIF_2_1_0(f, m) for m in mus]
    # barrier_subproblem_solve(f, x0, mu)

    eq_dirs = []
    ineq_dirs = []

    for filename in os.listdir(pFolder):
        if filename.startswith("Inequality_Restoration"): 
            ineq_dirs.append(filename + "/")
        elif filename.startswith("Equality_Restoration"): 
            eq_dirs.append(filename + "/")

    f_vals = read_f_obj(pFolder)
    f_vals2 = read_f_obj(pFolder2)

    fig, ax = plt.subplots()
    k = 0
    for obj in f_vals:
        ax.scatter(k, obj[0,0], color='k', marker='.')
        ax.scatter(k, obj[0,1], color='k', marker='.')
        k += obj.shape[0]

    k = 0
    for th, th2 in zip(read_theta(pFolder), read_theta(pFolder2)):
        
        k += th.shape[0]

    import itertools
    theta = np.concatenate(read_theta(pFolder))
    theta_2 = np.concatenate(read_theta(pFolder2))
    f_vals = np.concatenate(f_vals, axis=0)
    f_vals2 = np.concatenate(f_vals2, axis=0)
    # theta = list(itertools.chain(*read_theta(pFolder)))
    # theta_2 = list(itertools.chain(*read_theta(pFolder2)))
    ax.plot(f_vals[:,1], color='k', label=r"$f(x_k^{(j)})$")
    ax.plot(f_vals[:,0], color='k', label=r"$E_{\mu_j}(x_k^{(j)})$")
    # ax.plot(f_vals2[:,1], color='k', linestyle='dotted', label=r"$f(x_k^{(j)})$")
    # ax.plot(f_vals2[:,0], color='k', linestyle='dashed', label=r"$E_{\mu_j}(x_k^{(j)})$")
    # ax.plot(theta, color='k', linestyle='dashed', label=r"$\theta_{SOC}(x_k^{(j)})$")
    # ax.plot(theta_2, color='k', linestyle='dotted', label=r"$\theta(x_k^{(j)})$")
    # ax.plot(f_vals[:,2], color='k', linestyle='dashed', label=r'$\varphi_{\mu_j}(x_k^{(j)})$')
    ax.grid()
    # ax.set_ylim([-1, 1.2*max(f_vals[1:,1])])
    # ax.set_yscale('log')
    ax.set_xlabel(r'Iterations $k$ for subproblems $j$')

    ax.legend()
    fig.subplots_adjust(hspace=1.)
    fig.subplots_adjust(wspace=1.)
    # plt.show()
    # fig.savefig(figFolder +sys.argv[0] + "Objective.eps", format='eps')
    fig.savefig(figFolder +"NC_QP_feasible.eps", format='eps')
